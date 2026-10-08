import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

from app.tools.schemas import MAX_QUERY_WINDOW_DAYS, ProductId

from .config import (
    DEFAULT_WINDOW_DAYS,
    ROUTING_KEYWORDS,
    ROUTING_PRIORITY,
    ROUTING_PROXIMITY_BONUS,
    ROUTING_PROXIMITY_MAX_CHARS,
)
from .schemas import DateWindows, ParsedRequest, PolicyContext, TaskKind

_PRODUCT_ID_PATTERN = re.compile(r"(?<![A-Z0-9])P[0-9]{3}(?![A-Z0-9])", re.IGNORECASE)
_CLAUSE_SEPARATOR_PATTERN = re.compile(r"[,，、。;；!！?？\n]+")
_REPLACEMENT_MARKER_PATTERN = re.compile(r"(?=(?:改为|转而|而是))")
_REPLACEMENT_PREFIX_PATTERN = re.compile(r"^\s*(?:改为|转而|而是)")
_NEGATION_TOKEN = r"(?:不要|无需|无须|不必|不用|并非|(?<!是)不是|不算|不属于|别)"
_NEGATION_PATTERN = re.compile(_NEGATION_TOKEN)
_ADVERSATIVE_PATTERN = re.compile(r"(?:不过|然而|(?<!不)但是|(?<!不)但)")
_DIRECT_NEGATED_ANALYSIS_PATTERN = re.compile(
    _NEGATION_TOKEN
    + r"\s*(?:再|先|继续)?\s*"
    + r"(?:分析|查看|诊断|检查|查询|统计|对比|比较|复盘|采用|使用)"
    + r"\s*(?:的)?\s*$"
)
_EVIDENCE_DISCLAIMER_PATTERN = re.compile(
    r"(?:不要|不可|不能|别).*(?:当作?|作为|算作?)?.*证据"
)
_CLAIM_PATTERN = re.compile(r"(?:断言|声称|据称|宣称)")
_VERIFY_PATTERN = re.compile(r"(?:核验|验证|核实|确认)")
_CLAIM_REFERENCE_PATTERN = re.compile(
    r"(?:断言|说法|唯一(?:主因|原因)|(?:主因|原因).*(?:断言|说法))"
)
_ANAPHORIC_DISCLAIMER_PATTERN = re.compile(
    r"^\s*(?:该|此)?(?:断言|说法).*(?:不要|不可|不能|别).*(?:证据)"
)
_RELATIVE_WINDOW_PATTERN = re.compile(
    r"(?P<kind>当前|最近|过去|此前|近|前)\s*(?P<days>[0-9]+)\s*天"
)
_DATE_TOKEN = (
    r"(?<![0-9])"
    r"(?P<{prefix}year>[0-9]{{4}})"
    r"(?:-|/|年)"
    r"(?P<{prefix}month>[0-9]{{1,2}})"
    r"(?:-|/|月)"
    r"(?P<{prefix}day>[0-9]{{1,2}})日?"
    r"(?![0-9])"
)
_DATE_PATTERN = re.compile(_DATE_TOKEN.format(prefix=""))
_DATE_RANGE_PATTERN = re.compile(
    _DATE_TOKEN.format(prefix="start_")
    + r"\s*(?:至|到|~|～)\s*"
    + _DATE_TOKEN.format(prefix="end_")
)


class _ClauseStatus(StrEnum):
    ACTIVE = "active"
    NEGATED = "negated"
    REPLACEMENT = "replacement"
    DISCLAIMER = "disclaimer"


@dataclass(frozen=True)
class _Clause:
    text: str
    status: _ClauseStatus


class _UnsupportedDate(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _split_clauses(text: str) -> tuple[str, ...]:
    clauses: list[str] = []
    for segment in _CLAUSE_SEPARATOR_PATTERN.split(text):
        for adversative_segment in _ADVERSATIVE_PATTERN.split(segment):
            clauses.extend(_REPLACEMENT_MARKER_PATTERN.split(adversative_segment))
    return tuple(clause.strip() for clause in clauses if clause.strip())


def _is_disclaimer(clause: str) -> bool:
    if _EVIDENCE_DISCLAIMER_PATTERN.search(clause):
        return True
    if _CLAIM_PATTERN.search(clause) and _PRODUCT_ID_PATTERN.search(clause):
        return True
    return (
        _VERIFY_PATTERN.search(clause) is not None
        and _CLAIM_REFERENCE_PATTERN.search(clause) is not None
    )


def _pair_is_negated(
    text: str,
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    left, right = sorted((first, second))
    prefix = text[: left[0]]
    prefix_negations = tuple(_NEGATION_PATTERN.finditer(prefix))
    if prefix_negations:
        last_negation = prefix_negations[-1]
        if not _ADVERSATIVE_PATTERN.search(prefix[last_negation.end() :]):
            return True
    return _NEGATION_PATTERN.search(text[left[1] : right[0]]) is not None


def _contains_task_signal(text: str) -> bool:
    normalized = text.casefold()
    return any(_task_signal_pairs(normalized, task) for task in ROUTING_PRIORITY)


def _contains_negated_task_signal(text: str) -> bool:
    normalized = text.casefold()
    return any(
        _pair_is_negated(normalized, first, second)
        for task in ROUTING_PRIORITY
        for first, second in _task_signal_pairs(normalized, task)
    )


def _contains_negated_product_id(text: str) -> bool:
    return any(
        _NEGATION_PATTERN.search(text[: match.start()])
        for match in _PRODUCT_ID_PATTERN.finditer(text)
    )


def _contains_directly_negated_date(text: str) -> bool:
    date_starts = (
        *(match.start() for match in _RELATIVE_WINDOW_PATTERN.finditer(text)),
        *(match.start() for match in _DATE_PATTERN.finditer(text)),
    )
    return any(
        _DIRECT_NEGATED_ANALYSIS_PATTERN.search(text[:date_start])
        for date_start in date_starts
    )


def _annotate_clauses(text: str) -> tuple[_Clause, ...]:
    annotated: list[_Clause] = []
    for clause in _split_clauses(text):
        if _is_disclaimer(clause):
            status = _ClauseStatus.DISCLAIMER
        elif _REPLACEMENT_PREFIX_PATTERN.search(clause) and _contains_task_signal(clause):
            status = _ClauseStatus.REPLACEMENT
        elif (
            _contains_negated_task_signal(clause)
            or _contains_negated_product_id(clause)
            or _contains_directly_negated_date(clause)
        ):
            status = _ClauseStatus.NEGATED
        else:
            status = _ClauseStatus.ACTIVE
        annotated.append(_Clause(text=clause, status=status))

    for index, clause in enumerate(annotated):
        if not _ANAPHORIC_DISCLAIMER_PATTERN.search(clause.text) or index == 0:
            continue
        previous = annotated[index - 1]
        if _VERIFY_PATTERN.search(previous.text) and _PRODUCT_ID_PATTERN.search(previous.text):
            annotated[index - 1] = _Clause(
                text=previous.text,
                status=_ClauseStatus.DISCLAIMER,
            )
    return tuple(annotated)


def _effective_clauses(clauses: tuple[_Clause, ...]) -> tuple[_Clause, ...]:
    effective: list[_Clause] = []
    for clause in clauses:
        if clause.status in {_ClauseStatus.NEGATED, _ClauseStatus.DISCLAIMER}:
            continue
        if clause.status is _ClauseStatus.REPLACEMENT:
            effective.clear()
        effective.append(clause)
    return tuple(effective)


def _extract_product_ids(clauses: tuple[_Clause, ...]) -> tuple[ProductId, ...]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for clause in clauses:
        for match in _PRODUCT_ID_PATTERN.finditer(clause.text):
            product_id = match.group().upper()
            if product_id not in seen:
                seen.add(product_id)
                product_ids.append(product_id)
    return tuple(product_ids)


def _task_signal_pairs(
    text: str,
    task: TaskKind,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    groups: list[tuple[tuple[int, int], ...]] = []
    for group in ROUTING_KEYWORDS[task]:
        spans = _keyword_spans(text, group)
        if {"商品", "产品", "货品", "sku"} & set(group):
            spans += tuple(
                (match.start(), match.end()) for match in _PRODUCT_ID_PATTERN.finditer(text)
            )
        groups.append(spans)
    if any(not matches for matches in groups):
        return ()
    return tuple((first, second) for first in groups[0] for second in groups[1])


def _parse_window_days(token: str) -> int:
    canonical = token.lstrip("0") or "0"
    maximum = str(MAX_QUERY_WINDOW_DAYS)
    if len(canonical) > len(maximum) or (
        len(canonical) == len(maximum) and canonical > maximum
    ):
        raise _UnsupportedDate("window_too_large")
    days = int(canonical)
    if days < 1:
        raise _UnsupportedDate("invalid_window_days")
    return days


def _relative_window_days(matches: list[re.Match[str]]) -> int | None:
    if not matches:
        return None
    parsed = [(match.group("kind"), _parse_window_days(match.group("days"))) for match in matches]
    if len(parsed) == 1:
        return parsed[0][1]
    if len(parsed) == 2:
        current = [days for kind, days in parsed if kind not in {"前", "此前"}]
        previous = [days for kind, days in parsed if kind in {"前", "此前"}]
        if len(current) == len(previous) == 1 and current[0] == previous[0]:
            return current[0]
    raise _UnsupportedDate("conflicting_dates")


def _windows_for_period(start_date: date, end_date: date) -> DateWindows:
    days = (end_date - start_date).days + 1
    if days > MAX_QUERY_WINDOW_DAYS:
        raise _UnsupportedDate("window_too_large")
    try:
        comparison_end_date = start_date - timedelta(days=1)
        comparison_start_date = comparison_end_date - timedelta(days=days - 1)
    except OverflowError as exc:
        raise _UnsupportedDate("date_out_of_range") from exc
    return DateWindows(
        start_date=start_date,
        end_date=end_date,
        comparison_start_date=comparison_start_date,
        comparison_end_date=comparison_end_date,
    )


def _default_windows(
    as_of_date: date,
    days: int = DEFAULT_WINDOW_DAYS,
) -> DateWindows:
    if days < 1:
        raise ValueError("days must be positive")
    if days > MAX_QUERY_WINDOW_DAYS:
        raise _UnsupportedDate("window_too_large")
    try:
        start_date = as_of_date - timedelta(days=days - 1)
    except OverflowError as exc:
        raise _UnsupportedDate("date_out_of_range") from exc
    return _windows_for_period(start_date, as_of_date)


def _unsupported_windows(as_of_date: date) -> DateWindows:
    try:
        return _default_windows(as_of_date)
    except _UnsupportedDate:
        return DateWindows(
            start_date=as_of_date,
            end_date=as_of_date,
            comparison_start_date=as_of_date,
            comparison_end_date=as_of_date,
        )


def _date_from_match(match: re.Match[str], prefix: str = "") -> date:
    try:
        return date(
            int(match.group(f"{prefix}year")),
            int(match.group(f"{prefix}month")),
            int(match.group(f"{prefix}day")),
        )
    except ValueError as exc:
        raise _UnsupportedDate("invalid_date") from exc


def _extract_windows(text: str, as_of_date: date) -> DateWindows:
    relative_matches = list(_RELATIVE_WINDOW_PATTERN.finditer(text))
    date_matches = list(_DATE_PATTERN.finditer(text))
    range_matches = list(_DATE_RANGE_PATTERN.finditer(text))

    if relative_matches and date_matches:
        raise _UnsupportedDate("conflicting_dates")
    if len(range_matches) > 1:
        raise _UnsupportedDate("conflicting_dates")

    days = _relative_window_days(relative_matches)
    if days is not None:
        return _default_windows(as_of_date, days)

    if range_matches:
        if len(date_matches) != 2:
            raise _UnsupportedDate("conflicting_dates")
        match = range_matches[0]
        start_date = _date_from_match(match, "start_")
        end_date = _date_from_match(match, "end_")
    elif len(date_matches) == 1:
        start_date = end_date = _date_from_match(date_matches[0])
    elif len(date_matches) > 1:
        raise _UnsupportedDate("conflicting_dates")
    else:
        return _default_windows(as_of_date)

    if start_date > end_date:
        raise _UnsupportedDate("invalid_date_order")
    if end_date > as_of_date:
        raise _UnsupportedDate("date_after_as_of")
    return _windows_for_period(start_date, end_date)


def _keyword_spans(text: str, keywords: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (match.start(), match.end())
        for keyword in keywords
        for match in re.finditer(re.escape(keyword.casefold()), text)
    )


def _minimum_group_gap(
    text: str,
    first_group: tuple[tuple[int, int], ...],
    second_group: tuple[tuple[int, int], ...],
) -> int:
    minimum_gap = len(text)
    for first_start, first_end in first_group:
        for second_start, second_end in second_group:
            if first_end <= second_start:
                between = text[first_end:second_start]
            elif second_end <= first_start:
                between = text[second_end:first_start]
            else:
                return 0
            compact_between = re.sub(r"\s+", "", between)
            minimum_gap = min(minimum_gap, len(compact_between))
    return minimum_gap


def _intent_score(text: str, task: TaskKind) -> int | None:
    signal_pairs = _task_signal_pairs(text, task)
    if not signal_pairs:
        return None
    score = len(ROUTING_KEYWORDS[task])
    gap = min(
        _minimum_group_gap(text, (first,), (second,))
        for first, second in signal_pairs
    )
    if gap <= ROUTING_PROXIMITY_MAX_CHARS:
        score += ROUTING_PROXIMITY_BONUS + ROUTING_PROXIMITY_MAX_CHARS - gap
    return score


def _classify_task(clauses: tuple[_Clause, ...]) -> TaskKind:
    return next(
        (
            task
            for task in ROUTING_PRIORITY
            if any(_intent_score(clause.text.casefold(), task) is not None for clause in clauses)
        ),
        TaskKind.UNSUPPORTED,
    )


def parse_request(text: str, context: PolicyContext) -> ParsedRequest:
    normalized_text = unicodedata.normalize("NFKC", text)
    clauses = _effective_clauses(_annotate_clauses(normalized_text))
    effective_text = "；".join(clause.text for clause in clauses)
    product_ids = _extract_product_ids(clauses)
    try:
        windows = _extract_windows(effective_text, context.as_of_date)
    except _UnsupportedDate as exc:
        return ParsedRequest(
            task=TaskKind.UNSUPPORTED,
            product_ids=product_ids,
            windows=_unsupported_windows(context.as_of_date),
            unsupported_reason=exc.reason,
        )

    task = _classify_task(clauses)
    if task is TaskKind.UNSUPPORTED:
        return ParsedRequest(
            task=task,
            product_ids=product_ids,
            windows=windows,
            unsupported_reason="no_task_signal",
        )
    return ParsedRequest(task=task, product_ids=product_ids, windows=windows)
