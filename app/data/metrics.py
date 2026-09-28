import math
from decimal import Decimal

Number = int | float | Decimal


def safe_divide(
    numerator: Number | None,
    denominator: Number | None,
) -> float | Decimal | None:
    if numerator is None or denominator is None:
        return None
    if not _is_finite(numerator) or not _is_finite(denominator) or denominator == 0:
        return None
    if isinstance(numerator, Decimal) or isinstance(denominator, Decimal):
        return Decimal(str(numerator)) / Decimal(str(denominator))
    return numerator / denominator


def _is_finite(value: Number) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    return math.isfinite(value)
