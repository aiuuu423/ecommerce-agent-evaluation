import re
import unicodedata
from datetime import date, timedelta

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
_CLAUSE_PATTERN = re.compile(r"[^,，。;；!！?？\n]+")
_INTENT_CHANGE_PATTERN = re.compile(r"改为|转而")
_DIRECT_NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:不要|无需|无须|不必|不用|并非|(?<!是)不是|别)"
    r"\s*(?:再|要|去|把|将)?\s*"
    r"(?:分析|检查|查看|诊断|关注|列出|给出|做|看|排查|复盘|筛选|找出|建立|建议)?"
    r"\s*$"
)
_CLAIM_VERB_PATTERN = re.compile(r"(?:断言|声称|据称|宣称)\s*$")
_VERIFY_VERB_PATTERN = re.compile(r"(?:核验|验证|核实|确认)\s*$")
_UNIQUE_CAUSE_PATTERN = re.compile(r"^\s*(?:是|为|是否为).{0,8}唯一(?:主因|原因)")
_CLAIMED_CAUSE_PATTERN = re.compile(
    r"^\s*(?:是|为|是否为).{0,8}(?:主因|原因).{0,6}(?:断言|说法)"
)
_CLAIM_DISCLAIMER_PATTERN = re.compile(
    r"^[\s,，;；:：]*(?:该|此)?的?(?:断言|说法).{0,8}"
    r"(?:不要|不可|别).{0,4}(?:当|作为).{0,2}证据"
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


class _UnsupportedDate(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _active_request_text(text: str) -> str:
    active_clauses: list[str] = []
    for match in _CLAUSE_PATTERN.finditer(text):
        clause = match.group().strip()
        if not clause:
            continue

        changes = list(_INTENT_CHANGE_PATTERN.finditer(clause))
        replacement = next(
            (
                clause[change.end() :].strip()
                for change in reversed(changes)
                if _contains_active_task_signal(clause[change.end() :])
            ),
            None,
        )
        if replacement is not None:
            active_clauses = [replacement]
            continue

        contrast_index = clause.rfind("而是")
        if contrast_index >= 0 and _contains_negated_task_signal(clause[:contrast_index]):
            contrast = clause[contrast_index + len("而是") :].strip()
            if _contains_active_task_signal(contrast):
                active_clauses = [contrast]
                continue

        active_clauses.append(clause)

    return "；".join(active_clauses)


def _is_claim_lure(clause: str, match: re.Match[str]) -> bool:
    before = clause[max(0, match.start() - 24) : match.start()]
    after = clause[match.end() : match.end() + 24]
    if _CLAIM_VERB_PATTERN.search(before):
        return True
    if _UNIQUE_CAUSE_PATTERN.search(after):
        return True
    if _CLAIMED_CAUSE_PATTERN.search(after):
        return True
    if _CLAIM_DISCLAIMER_PATTERN.search(after):
        return True
    if _VERIFY_VERB_PATTERN.search(before) and re.match(
        r"^\s*(?:的?(?:断言|说法)|是否为.{0,8}(?:主因|原因))",
        after,
    ):
        return True
    return False


def _extract_product_ids(text: str) -> tuple[ProductId, ...]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for match in _PRODUCT_ID_PATTERN.finditer(text):
        if _is_claim_lure(text, match):
            continue
        product_id = match.group().upper()
        if product_id not in seen:
            seen.add(product_id)
            product_ids.append(product_id)
    return tuple(product_ids)


def _is_directly_negated(text: str, signal_start: int) -> bool:
    prefix = text[max(0, signal_start - 16) : signal_start]
    return _DIRECT_NEGATION_PREFIX_PATTERN.search(prefix) is not None


def _task_signal_pairs(
    text: str,
    task: TaskKind,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    groups = tuple(_keyword_spans(text, group) for group in ROUTING_KEYWORDS[task])
    if any(not matches for matches in groups):
        return ()
    return tuple((first, second) for first in groups[0] for second in groups[1])


def _contains_active_task_signal(text: str) -> bool:
    normalized = text.casefold()
    return any(_intent_score(normalized, task) is not None for task in ROUTING_PRIORITY)


def _contains_negated_task_signal(text: str) -> bool:
    normalized = text.casefold()
    return any(
        _is_directly_negated(normalized, first[0])
        or _is_directly_negated(normalized, second[0])
        for task in ROUTING_PRIORITY
        for first, second in _task_signal_pairs(normalized, task)
    )


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
    active_pairs = tuple(
        (first, second)
        for first, second in _task_signal_pairs(text, task)
        if not _is_directly_negated(text, first[0])
        and not _is_directly_negated(text, second[0])
    )
    if not active_pairs:
        return None
    score = len(ROUTING_KEYWORDS[task])
    gap = min(
        _minimum_group_gap(text, (first,), (second,))
        for first, second in active_pairs
    )
    if gap <= ROUTING_PROXIMITY_MAX_CHARS:
        score += ROUTING_PROXIMITY_BONUS + ROUTING_PROXIMITY_MAX_CHARS - gap
    return score


def _classify_task(text: str) -> TaskKind:
    normalized = text.casefold()
    return next(
        (
            task
            for task in ROUTING_PRIORITY
            if _intent_score(normalized, task) is not None
        ),
        TaskKind.UNSUPPORTED,
    )


def parse_request(text: str, context: PolicyContext) -> ParsedRequest:
    normalized_text = unicodedata.normalize("NFKC", text)
    active_text = _active_request_text(normalized_text)
    product_ids = _extract_product_ids(active_text)
    try:
        windows = _extract_windows(active_text, context.as_of_date)
    except _UnsupportedDate as exc:
        return ParsedRequest(
            task=TaskKind.UNSUPPORTED,
            product_ids=product_ids,
            windows=_unsupported_windows(context.as_of_date),
            unsupported_reason=exc.reason,
        )

    task = _classify_task(active_text)
    if task is TaskKind.UNSUPPORTED:
        return ParsedRequest(
            task=task,
            product_ids=product_ids,
            windows=windows,
            unsupported_reason="no_task_signal",
        )
    return ParsedRequest(task=task, product_ids=product_ids, windows=windows)
