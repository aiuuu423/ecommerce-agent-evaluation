import re
from datetime import date, timedelta

from app.tools.schemas import MAX_QUERY_WINDOW_DAYS, ProductId

from .config import DEFAULT_WINDOW_DAYS, ROUTING_KEYWORDS, ROUTING_PRIORITY
from .schemas import DateWindows, ParsedRequest, PolicyContext, TaskKind

_PRODUCT_ID_PATTERN = re.compile(r"(?<![A-Z0-9])P[0-9]{3}(?![A-Z0-9])", re.IGNORECASE)
_RELATIVE_WINDOW_PATTERN = re.compile(r"(?:最近|近|过去|前)\s*([0-9]+)\s*天")
_DATE_TOKEN = (
    r"(?P<{prefix}year>[0-9]{{4}})"
    r"(?:-|/|年)"
    r"(?P<{prefix}month>[0-9]{{1,2}})"
    r"(?:-|/|月)"
    r"(?P<{prefix}day>[0-9]{{1,2}})日?"
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


def _extract_product_ids(text: str) -> tuple[ProductId, ...]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for match in _PRODUCT_ID_PATTERN.finditer(text):
        product_id = match.group().upper()
        if product_id not in seen:
            seen.add(product_id)
            product_ids.append(product_id)
    return tuple(product_ids)


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
    if len(relative_matches) > 1 or len(range_matches) > 1:
        raise _UnsupportedDate("conflicting_dates")

    if relative_matches:
        days = int(relative_matches[0].group(1))
        if days < 1:
            raise _UnsupportedDate("invalid_window_days")
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
    for task in ROUTING_PRIORITY:
        if all(
            any(keyword.casefold() in normalized for keyword in group)
            for group in ROUTING_KEYWORDS[task]
        ):
            return task
    return TaskKind.UNSUPPORTED


def parse_request(text: str, context: PolicyContext) -> ParsedRequest:
    product_ids = _extract_product_ids(text)
    try:
        windows = _extract_windows(text, context.as_of_date)
    except _UnsupportedDate as exc:
        return ParsedRequest(
            task=TaskKind.UNSUPPORTED,
            product_ids=product_ids,
            windows=_unsupported_windows(context.as_of_date),
            unsupported_reason=exc.reason,
        )

    task = _classify_task(text)
    if task is TaskKind.UNSUPPORTED:
        return ParsedRequest(
            task=task,
            product_ids=product_ids,
            windows=windows,
            unsupported_reason="no_task_signal",
        )
    return ParsedRequest(task=task, product_ids=product_ids, windows=windows)
