from datetime import date

import pytest

from app.baselines.v1.parsing import parse_request
from app.baselines.v1.schemas import PolicyContext, TaskKind


@pytest.fixture
def context() -> PolicyContext:
    return PolicyContext(
        dataset_id="e1e81533c25e03e5",
        dataset_version="v1",
        as_of_date=date(2026, 4, 30),
    )


def test_parse_request_normalizes_and_deduplicates_product_ids(
    context: PolicyContext,
) -> None:
    parsed = parse_request(
        "分析最近30天 p003、P001、P003 和 P1234 的转化下降",
        context,
    )

    assert parsed.task is TaskKind.CONVERSION_DECLINE
    assert parsed.product_ids == ("P003", "P001")
    assert isinstance(parsed.product_ids, tuple)
    assert parsed.windows.model_dump(mode="json") == {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
    }


def test_parse_request_uses_default_window_and_all_products(
    context: PolicyContext,
) -> None:
    parsed = parse_request("请诊断 GMV 变化", context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.product_ids == ()
    assert parsed.windows.model_dump(mode="json") == {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "分析最近7天的销售额趋势",
            {
                "start_date": "2026-04-24",
                "end_date": "2026-04-30",
                "comparison_start_date": "2026-04-17",
                "comparison_end_date": "2026-04-23",
            },
        ),
        (
            "分析 2026-04-01 至 2026-04-10 的销售额趋势",
            {
                "start_date": "2026-04-01",
                "end_date": "2026-04-10",
                "comparison_start_date": "2026-03-22",
                "comparison_end_date": "2026-03-31",
            },
        ),
        (
            "诊断2026年4月15日的营收变化",
            {
                "start_date": "2026-04-15",
                "end_date": "2026-04-15",
                "comparison_start_date": "2026-04-14",
                "comparison_end_date": "2026-04-14",
            },
        ),
    ],
)
def test_parse_request_supports_relative_and_explicit_dates(
    context: PolicyContext,
    text: str,
    expected: dict[str, str],
) -> None:
    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.windows.model_dump(mode="json") == expected


def test_parse_request_supports_recent_366_days(context: PolicyContext) -> None:
    parsed = parse_request("分析最近366天的 GMV 变化", context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert (parsed.windows.end_date - parsed.windows.start_date).days + 1 == 366
    assert (
        parsed.windows.comparison_end_date - parsed.windows.comparison_start_date
    ).days + 1 == 366


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("分析 2026-02-30 至 2026-03-02 的 GMV 变化", "invalid_date"),
        ("分析 2026-04-20 至 2026-04-10 的 GMV 变化", "invalid_date_order"),
        ("分析 2026-04-20 至 2026-05-01 的 GMV 变化", "date_after_as_of"),
        ("分析最近367天的 GMV 变化", "window_too_large"),
        ("分析最近999999999999999999999999天的 GMV 变化", "window_too_large"),
        ("分析 2025-04-29 至 2026-04-30 的 GMV 变化", "window_too_large"),
        ("分析最近7天、2026-04-01至2026-04-10的 GMV 变化", "conflicting_dates"),
        (
            "比较 2026-04-01 至 2026-04-10 和 2026-04-11 至 2026-04-20 的 GMV 变化",
            "conflicting_dates",
        ),
    ],
)
def test_parse_request_returns_structured_unsupported_for_bad_dates(
    context: PolicyContext,
    text: str,
    reason: str,
) -> None:
    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.UNSUPPORTED
    assert parsed.unsupported_reason == reason
    assert parsed.product_ids == ()


@pytest.mark.parametrize(
    "text",
    [
        "分析最近1天的 GMV 变化",
        "分析 0001-01-01 的 GMV 变化",
    ],
)
def test_parse_request_returns_structured_unsupported_for_date_underflow(
    text: str,
) -> None:
    context = PolicyContext(
        dataset_id="e1e81533c25e03e5",
        dataset_version="v1",
        as_of_date=date.min,
    )

    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.UNSUPPORTED
    assert parsed.unsupported_reason == "date_out_of_range"
    assert parsed.windows.start_date == date.min
    assert parsed.windows.end_date == date.min


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("检查商品 P001 的异常原因", TaskKind.PRODUCT_ANOMALY),
        ("这个 SKU 为什么有问题", TaskKind.PRODUCT_ANOMALY),
        ("分析商品表现异常", TaskKind.PRODUCT_ANOMALY),
        ("分析转化率下降", TaskKind.CONVERSION_DECLINE),
        ("CVR 为什么下滑", TaskKind.CONVERSION_DECLINE),
        ("查找转化下降原因", TaskKind.CONVERSION_DECLINE),
        ("给出下周优先事项", TaskKind.NEXT_WEEK_PRIORITY),
        ("建议下一步重点", TaskKind.NEXT_WEEK_PRIORITY),
        ("下周应该采取哪些动作", TaskKind.NEXT_WEEK_PRIORITY),
        ("哪些商品需要关注", TaskKind.PRODUCTS_TO_WATCH),
        ("列出有风险的 SKU", TaskKind.PRODUCTS_TO_WATCH),
        ("给出商品预警", TaskKind.PRODUCTS_TO_WATCH),
        ("诊断 GMV 变化", TaskKind.GMV_DIAGNOSIS),
        ("查看销售额趋势", TaskKind.GMV_DIAGNOSIS),
        ("分析营收变化", TaskKind.GMV_DIAGNOSIS),
    ],
)
def test_parse_request_classifies_general_task_signals(
    context: PolicyContext,
    text: str,
    task: TaskKind,
) -> None:
    assert parse_request(text, context).task is task


def test_parse_request_uses_frozen_priority_for_multiple_signals(
    context: PolicyContext,
) -> None:
    text = "商品异常导致转化下降，下周优先关注风险 SKU 并诊断 GMV 变化"

    assert parse_request(text, context).task is TaskKind.PRODUCT_ANOMALY


def test_parse_request_ignores_adversarial_irrelevant_identifiers(
    context: PolicyContext,
) -> None:
    parsed = parse_request(
        "备注编号 G999 与 P12 无关；请查看 P007 的成交趋势",
        context,
    )

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.product_ids == ("P007",)


def test_parse_request_returns_structured_unsupported_without_task_signal(
    context: PolicyContext,
) -> None:
    parsed = parse_request("请帮我总结一下 P003", context)

    assert parsed.task is TaskKind.UNSUPPORTED
    assert parsed.unsupported_reason == "no_task_signal"
    assert parsed.product_ids == ("P003",)
