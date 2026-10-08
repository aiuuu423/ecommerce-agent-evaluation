from datetime import date

import pytest

from app.baselines.v1.parsing import _annotate_clauses, parse_request
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
    "text",
    [
        "对比最近7天和前7天的销售额走向",
        "对比前７天与最近７天的营收增减",
    ],
)
def test_parse_request_accepts_matching_current_and_previous_windows(
    context: PolicyContext,
    text: str,
) -> None:
    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.unsupported_reason is None
    assert parsed.windows.model_dump(mode="json") == {
        "start_date": "2026-04-24",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-04-17",
        "comparison_end_date": "2026-04-23",
    }


@pytest.mark.parametrize(
    "separator",
    [";", "；", ",", "，", "。", "、", "!", "！", "?", "？", "\n"],
)
@pytest.mark.parametrize(
    ("negated_window", "active_window", "expected"),
    [
        (
            "最近7天",
            "最近30天",
            {
                "start_date": "2026-04-01",
                "end_date": "2026-04-30",
                "comparison_start_date": "2026-03-02",
                "comparison_end_date": "2026-03-31",
            },
        ),
        (
            "2026-04-01",
            "2026年4月15日",
            {
                "start_date": "2026-04-15",
                "end_date": "2026-04-15",
                "comparison_start_date": "2026-04-14",
                "comparison_end_date": "2026-04-14",
            },
        ),
        (
            "2026-03-01至2026-03-07",
            "2026/04/01～2026/04/10",
            {
                "start_date": "2026-04-01",
                "end_date": "2026-04-10",
                "comparison_start_date": "2026-03-22",
                "comparison_end_date": "2026-03-31",
            },
        ),
    ],
)
def test_parse_request_ignores_directly_negated_date_windows(
    context: PolicyContext,
    separator: str,
    negated_window: str,
    active_window: str,
    expected: dict[str, str],
) -> None:
    parsed = parse_request(
        f"不要分析{negated_window}{separator}请诊断{active_window} GMV",
        context,
    )

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.unsupported_reason is None
    assert parsed.windows.model_dump(mode="json") == expected


@pytest.mark.parametrize("adversative", ["但", "但是", "不过", "然而"])
@pytest.mark.parametrize(
    ("negated_window", "active_window", "expected_start", "expected_end"),
    [
        ("最近7天", "最近30天", date(2026, 4, 1), date(2026, 4, 30)),
        (
            "2026-04-01至2026-04-07",
            "2026-04-15至2026-04-20",
            date(2026, 4, 15),
            date(2026, 4, 20),
        ),
    ],
)
def test_parse_request_treats_adversatives_as_semantic_clause_boundaries(
    context: PolicyContext,
    adversative: str,
    negated_window: str,
    active_window: str,
    expected_start: date,
    expected_end: date,
) -> None:
    parsed = parse_request(
        f"不要分析{negated_window}{adversative}请诊断{active_window} GMV",
        context,
    )

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.unsupported_reason is None
    assert parsed.windows.start_date == expected_start
    assert parsed.windows.end_date == expected_end


def test_parse_request_does_not_treat_ordinary_negation_as_a_negated_date(
    context: PolicyContext,
) -> None:
    parsed = parse_request("数据不是空但请分析最近7天 GMV 变化", context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.unsupported_reason is None
    assert parsed.windows.start_date == date(2026, 4, 24)
    assert parsed.windows.end_date == date(2026, 4, 30)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("分析 2026-02-30 至 2026-03-02 的 GMV 变化", "invalid_date"),
        ("分析 2026-04-20 至 2026-04-10 的 GMV 变化", "invalid_date_order"),
        ("分析 2026-04-20 至 2026-05-01 的 GMV 变化", "date_after_as_of"),
        ("分析最近367天的 GMV 变化", "window_too_large"),
        ("分析最近999999999999999999999999天的 GMV 变化", "window_too_large"),
        ("对比最近7天和前8天的销售额走向", "conflicting_dates"),
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


def test_parse_request_handles_arbitrarily_long_relative_window(
    context: PolicyContext,
) -> None:
    text = f"分析最近{'9' * 5000}天的销售额走向"

    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.UNSUPPORTED
    assert parsed.unsupported_reason == "window_too_large"


@pytest.mark.parametrize(
    "text",
    [
        "参考编号12026-04-010，分析 GMV 变化",
        "参考编号9992026/04/01999，分析销售额走向",
        "参考编号12026年04月010，分析营收增减",
    ],
)
def test_parse_request_does_not_parse_dates_inside_longer_numbers(
    context: PolicyContext,
    text: str,
) -> None:
    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.windows.start_date == date(2026, 4, 1)
    assert parsed.windows.end_date == date(2026, 4, 30)


def test_parse_request_normalizes_fullwidth_digits(context: PolicyContext) -> None:
    parsed = parse_request(
        "查看 P００７ 在 ２０２６－０４－１５ 的销售额走向",
        context,
    )

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.product_ids == ("P007",)
    assert parsed.windows.start_date == date(2026, 4, 15)
    assert parsed.windows.end_date == date(2026, 4, 15)


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
        ("复盘销售额走向及原因", TaskKind.GMV_DIAGNOSIS),
        ("排查商品购买效率降低原因", TaskKind.CONVERSION_DECLINE),
        ("列出需要重点跟进的商品", TaskKind.PRODUCTS_TO_WATCH),
        ("下周需要关注哪些工作", TaskKind.NEXT_WEEK_PRIORITY),
        ("对照本月和上月的 GMV 表现", TaskKind.GMV_DIAGNOSIS),
        ("筛选发生经营偏离的货品", TaskKind.PRODUCT_ANOMALY),
        ("找出访购漏斗退步的产品", TaskKind.CONVERSION_DECLINE),
        ("建立后续监控对象清单", TaskKind.PRODUCTS_TO_WATCH),
        ("给出下一周期运营处置顺序", TaskKind.NEXT_WEEK_PRIORITY),
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


@pytest.mark.parametrize(
    ("text", "task"),
    [
        (
            "商品存在问题；请重点关注风险商品并详细诊断销售额变化",
            TaskKind.PRODUCT_ANOMALY,
        ),
        (
            "转化下降；下周请优先安排重点动作并诊断销售额变化",
            TaskKind.CONVERSION_DECLINE,
        ),
        (
            "下周安排动作；请重点关注风险商品并详细诊断销售额变化",
            TaskKind.NEXT_WEEK_PRIORITY,
        ),
        (
            "关注风险商品；请详细诊断销售额变化趋势",
            TaskKind.PRODUCTS_TO_WATCH,
        ),
    ],
)
def test_parse_request_never_lets_lower_priority_score_override_frozen_priority(
    context: PolicyContext,
    text: str,
    task: TaskKind,
) -> None:
    assert parse_request(text, context).task is task


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("不要分析商品异常，改为诊断 GMV 变化", TaskKind.GMV_DIAGNOSIS),
        ("无需关注风险商品；请分析转化下降", TaskKind.CONVERSION_DECLINE),
        ("不是查看转化下降，而是给出下周优先动作", TaskKind.NEXT_WEEK_PRIORITY),
        ("先看商品异常，改为列出需要关注的风险商品", TaskKind.PRODUCTS_TO_WATCH),
        ("是不是要分析商品异常", TaskKind.PRODUCT_ANOMALY),
        ("数据不是空；请分析商品异常", TaskKind.PRODUCT_ANOMALY),
        ("不要分析商品异常；请诊断 GMV 变化", TaskKind.GMV_DIAGNOSIS),
        ("商品不是异常；请诊断 GMV 变化", TaskKind.GMV_DIAGNOSIS),
        ("请诊断 GMV 变化；展示口径改为含税", TaskKind.GMV_DIAGNOSIS),
        ("分析商品异常；随后转而使用新版口径", TaskKind.PRODUCT_ANOMALY),
    ],
)
def test_parse_request_ignores_negated_or_replaced_task_signals(
    context: PolicyContext,
    text: str,
    task: TaskKind,
) -> None:
    assert parse_request(text, context).task is task


def test_parse_request_ignores_adversarial_irrelevant_identifiers(
    context: PolicyContext,
) -> None:
    parsed = parse_request(
        "备注编号 G999 与 P12 无关；请查看 P007 的成交趋势",
        context,
    )

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.product_ids == ("P007",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "不要把 P001 是异常主因的断言当证据；请分析 P002 的商品异常",
            ("P002",),
        ),
        (
            "核验 P003 是否为唯一主因；仅检查 P004 的购买效率",
            ("P004",),
        ),
        (
            "业务断言 P005 导致下滑；查看 P006 的转化表现",
            ("P006",),
        ),
        (
            "请验证 P007 的说法，不要当证据；再检查 P008",
            ("P008",),
        ),
        (
            "不要分析 P009，改为检查 P010 的商品异常",
            ("P010",),
        ),
        (
            "核验 P001 是否为唯一主因；分析 P002 异常并给结论",
            ("P002",),
        ),
        (
            "分析 P002 异常并给结论",
            ("P002",),
        ),
        (
            "核验 P001；该断言不要当证据；分析 P002 异常",
            ("P002",),
        ),
    ],
)
def test_parse_request_filters_claim_lures_locally_and_keeps_action_objects(
    context: PolicyContext,
    text: str,
    expected: tuple[str, ...],
) -> None:
    assert parse_request(text, context).product_ids == expected


@pytest.mark.parametrize(
    ("separator", "modifier", "replacement"),
    [
        ("；", "请", ""),
        ("，", "然后请", "改为"),
        ("。", "接下来", ""),
        ("、", "现在请", "改为"),
        ("\n", "再", ""),
    ],
)
def test_parse_request_uses_clause_semantics_for_task_and_product_ids(
    context: PolicyContext,
    separator: str,
    modifier: str,
    replacement: str,
) -> None:
    text = (
        f"不要分析P009{separator}"
        f"{modifier}{replacement}诊断P010 GMV{separator}"
        "不要把P001当证据"
    )

    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.GMV_DIAGNOSIS
    assert parsed.product_ids == ("P010",)


def test_parse_request_annotates_clause_semantics_before_interpretation() -> None:
    clauses = _annotate_clauses(
        "分析P002异常；不要分析这次商品异常；改为诊断P010 GMV；不要把P001当证据"
    )

    assert tuple(clause.status.value for clause in clauses) == (
        "active",
        "negated",
        "replacement",
        "disclaimer",
    )


@pytest.mark.parametrize("separator", ["；", "，", "。", "、", "\n"])
def test_parse_request_keeps_active_specific_anomaly_over_negated_generic_clause(
    context: PolicyContext,
    separator: str,
) -> None:
    text = f"分析P002异常{separator}不要分析这次商品异常{separator}诊断GMV"

    parsed = parse_request(text, context)

    assert parsed.task is TaskKind.PRODUCT_ANOMALY
    assert parsed.product_ids == ("P002",)


def test_parse_request_returns_structured_unsupported_without_task_signal(
    context: PolicyContext,
) -> None:
    parsed = parse_request("请帮我总结一下 P003", context)

    assert parsed.task is TaskKind.UNSUPPORTED
    assert parsed.unsupported_reason == "no_task_signal"
    assert parsed.product_ids == ("P003",)
