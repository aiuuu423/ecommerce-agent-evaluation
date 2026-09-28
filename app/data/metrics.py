from decimal import Decimal

Number = int | float | Decimal


def safe_divide(
    numerator: Number | None,
    denominator: Number | None,
) -> float | Decimal | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator
