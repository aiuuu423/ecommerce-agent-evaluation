import re
import unicodedata
from datetime import date, timedelta

from app.tools.schemas import MAX_QUERY_WINDOW_DAYS, ProductId

from .config import DEFAULT_WINDOW_DAYS, ROUTING_KEYWORDS, ROUTING_PRIORITY
from .schemas import DateWindows, ParsedRequest, PolicyContext, TaskKind

_PRODUCT_ID_PATTERN = re.compile(r"(?<![A-Z0-9])P[0-9]{3}(?![A-Z0-9])", re.IGNORECASE)
_CLAUSE_PATTERN = re.compile(r"[^,，。;；!！?？\n]+")
_RELATIVE_WINDOW_PATTERN = re.compile(
    r"(?P<kind>最近|近|过去|前)\s*(?P<days>[0-9]+)\s*天"
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
_TRAILING_GENERIC_INTENT_PATTERN = re.compile(
    r"(?:原因|异常)(?:分析|诊断|情况)?\s*[。.!！?？]?\s*$"
)


class _UnsupportedDate(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _extract_product_ids(text: str) -> tuple[ProductId, ...]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for clause_match in _CLAUSE_PATTERN.finditer(text):
        clause = clause_match.group()
        if _clause_uses_ids_as_claims(clause):
            continue
        for match in _PRODUCT_ID_PATTERN.finditer(clause):
            product_id = match.group().upper()
            if product_id not in seen:
                seen.add(product_id)
                product_ids.append(product_id)
    return tuple(product_ids)


def _clause_uses_ids_as_claims(clause: str) -> bool:
    rejects_assertion = (
        any(keyword in clause for keyword in ("不要", "不能", "不可", "别"))
        and any(keyword in clause for keyword in ("断言", "说法", "结论"))
        and any(keyword in clause for keyword in ("证据", "事实"))
    )
    verifies_unique_cause = (
        any(keyword in clause for keyword in ("核验", "验证", "查证", "确认"))
        and any(keyword in clause for keyword in ("唯一主因", "唯一原因"))
    )
    return rejects_assertion or verifies_unique_cause


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
        current = [days for kind, days in parsed if kind != "前"]
        previous = [days for kind, days in parsed if kind == "前"]
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


def _classify_task(text: str) -> TaskKind:
    normalized = text.casefold()
    matched_tasks = [
        task
        for task in ROUTING_PRIORITY
        if all(
            any(keyword.casefold() in normalized for keyword in group)
            for group in ROUTING_KEYWORDS[task]
        )
    ]
    if (
        len(matched_tasks) > 1
        and matched_tasks[0] is TaskKind.PRODUCT_ANOMALY
        and _TRAILING_GENERIC_INTENT_PATTERN.search(normalized)
    ):
        return matched_tasks[1]
    return matched_tasks[0] if matched_tasks else TaskKind.UNSUPPORTED


def parse_request(text: str, context: PolicyContext) -> ParsedRequest:
    normalized_text = unicodedata.normalize("NFKC", text)
    product_ids = _extract_product_ids(normalized_text)
    try:
        windows = _extract_windows(normalized_text, context.as_of_date)
    except _UnsupportedDate as exc:
        return ParsedRequest(
            task=TaskKind.UNSUPPORTED,
            product_ids=product_ids,
            windows=_unsupported_windows(context.as_of_date),
            unsupported_reason=exc.reason,
        )

    task = _classify_task(normalized_text)
    if task is TaskKind.UNSUPPORTED:
        return ParsedRequest(
            task=task,
            product_ids=product_ids,
            windows=windows,
            unsupported_reason="no_task_signal",
        )
    return ParsedRequest(task=task, product_ids=product_ids, windows=windows)
