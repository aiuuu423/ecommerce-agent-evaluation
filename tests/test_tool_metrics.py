import json
from pathlib import Path

import pytest

from app.data.database import open_dataset
from app.data.generator import build_snapshot
from app.tools import ToolContext, build_default_registry
from app.tools.metrics import METRIC_REQUIREMENTS, calculate_metrics
from app.tools.schemas import (
    CalculateMetricsInput,
    MetricName,
    PriorToolExecution,
    QueryMarketingResult,
    QuerySalesResult,
    QueryTrafficResult,
)

ROOT = Path(__file__).parents[1]
CASES = ROOT / "data/evaluation_cases/v1/cases.jsonl"
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
DATASET_ID = "e1e81533c25e03e5"
SOURCE_LABEL = "Synthetic E-commerce Data"


class SummaryOnlyCatalog:
    def __init__(
        self,
        dataset_id: str = DATASET_ID,
        source_label: str = SOURCE_LABEL,
    ) -> None:
        self.verified_summary = {
            "dataset_id": dataset_id,
            "source_label": source_label,
        }

    def execute(self, *_args, **_kwargs):
        raise AssertionError("calculate_metrics must not read the database")


def window_arguments(**overrides: object) -> dict[str, object]:
    return {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
        **overrides,
    }


def sales_row(
    period: str,
    product_id: str = "P001",
    *,
    category: str = "A",
    region: str = "East",
    channel: str = "organic",
    gmv: float = 0.0,
    orders: int = 0,
    units: int = 0,
    refund_orders: int = 0,
) -> dict[str, object]:
    return {
        "period": period,
        "product_id": product_id,
        "category": category,
        "region": region,
        "channel": channel,
        "gmv": gmv,
        "orders": orders,
        "units": units,
        "refund_orders": refund_orders,
    }


def traffic_row(
    period: str,
    product_id: str = "P001",
    *,
    category: str = "A",
    impressions: int = 0,
    clicks: int = 0,
    visits: int = 0,
    observed_days: int = 30,
    missing_days: int = 0,
) -> dict[str, object]:
    return {
        "period": period,
        "product_id": product_id,
        "category": category,
        "impressions": impressions,
        "clicks": clicks,
        "visits": visits,
        "observed_days": observed_days,
        "missing_days": missing_days,
    }


def marketing_row(
    period: str,
    product_id: str = "P001",
    *,
    category: str = "A",
    campaign_id: str = "C001",
    spend: float = 0.0,
) -> dict[str, object]:
    return {
        "period": period,
        "product_id": product_id,
        "category": category,
        "campaign_id": campaign_id,
        "spend": spend,
    }


def execution(
    tool_name: str,
    rows: list[dict[str, object]],
    *,
    call_id: str | None = None,
    result_id: str = "result_0001",
    arguments: dict[str, object] | None = None,
    dataset_id: str = DATASET_ID,
    source_label: str = SOURCE_LABEL,
) -> PriorToolExecution:
    models = {
        "query_sales": (
            QuerySalesResult,
            [
                "period",
                "product_id",
                "category",
                "region",
                "channel",
                "gmv",
                "orders",
                "units",
                "refund_orders",
            ],
            window_arguments(include_refunds=True),
        ),
        "query_traffic": (
            QueryTrafficResult,
            [
                "period",
                "product_id",
                "category",
                "impressions",
                "clicks",
                "visits",
                "observed_days",
                "missing_days",
            ],
            window_arguments(include_missing=True),
        ),
        "query_marketing": (
            QueryMarketingResult,
            ["period", "product_id", "category", "campaign_id", "spend"],
            window_arguments(),
        ),
    }
    model, columns, default_arguments = models[tool_name]
    result = model.model_validate(
        {
            "result_id": result_id,
            "tool_name": tool_name,
            "dataset_id": dataset_id,
            "source_label": source_label,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    )
    return PriorToolExecution(
        call_id=call_id or f"call_{tool_name}",
        arguments=arguments or default_arguments,
        result=result,
    )


def context_with(
    *executions: PriorToolExecution,
    catalog: object | None = None,
    result_id: str = "result_0099",
) -> ToolContext:
    return ToolContext(
        catalog=catalog or SummaryOnlyCatalog(),
        prior_executions=executions,
        next_result_id=lambda: result_id,
    )


def compatible_sources() -> tuple[PriorToolExecution, ...]:
    return (
        execution(
            "query_sales",
            [
                sales_row(
                    "current",
                    gmv=120.0,
                    orders=3,
                    units=4,
                    refund_orders=1,
                ),
                sales_row("previous", gmv=100.0, orders=2, units=3),
            ],
            result_id="result_0001",
        ),
        execution(
            "query_traffic",
            [
                traffic_row(
                    "current",
                    impressions=1000,
                    clicks=100,
                    visits=80,
                ),
                traffic_row(
                    "previous",
                    impressions=800,
                    clicks=80,
                    visits=50,
                ),
            ],
            result_id="result_0002",
        ),
        execution(
            "query_marketing",
            [
                marketing_row("current", spend=40.0),
                marketing_row("previous", spend=25.0),
            ],
            result_id="result_0003",
        ),
    )


def test_metric_requirements_explicitly_cover_metric_vocabulary() -> None:
    frozen_metrics = {
        metric
        for line in CASES.read_text(encoding="utf-8").splitlines()
        for call in json.loads(line)["expected_tool_calls"]
        if call["name"] == "calculate_metrics"
        for metric in call["parameters"]["metrics"]
    }
    assert len(frozen_metrics) == 19
    assert frozen_metrics <= set(METRIC_REQUIREMENTS)
    assert set(METRIC_REQUIREMENTS) == {metric.value for metric in MetricName}


def test_calculate_metrics_uses_only_prior_executions_and_safe_division() -> None:
    sales, traffic, _marketing = compatible_sources()
    metrics = [
        "current_gmv",
        "gmv_change_rate",
        "current_aov",
        "aov_change_rate",
        "current_cvr",
        "previous_cvr",
        "cvr_change",
        "cvr_change_rate",
        "traffic_change_rate",
        "current_observed_days",
        "previous_observed_days",
        "current_refund_rate",
        "refund_rate",
        "evidence_value",
    ]
    result = calculate_metrics(
        CalculateMetricsInput(metrics=metrics, group_by=["product_id"]),
        context_with(sales, traffic),
    )

    assert result.columns == ["product_id", *metrics]
    assert result.dataset_id == DATASET_ID
    assert result.result_id == "result_0099"
    assert result.rows[0].model_dump(exclude_unset=True) == {
        "product_id": "P001",
        "current_gmv": 120.0,
        "gmv_change_rate": pytest.approx(0.2),
        "current_aov": pytest.approx(40.0),
        "aov_change_rate": pytest.approx(-0.2),
        "current_cvr": pytest.approx(3 / 80),
        "previous_cvr": pytest.approx(2 / 50),
        "cvr_change": pytest.approx(3 / 80 - 2 / 50),
        "cvr_change_rate": pytest.approx(((3 / 80) - (2 / 50)) / (2 / 50)),
        "traffic_change_rate": pytest.approx((80 - 50) / 50),
        "current_observed_days": 30,
        "previous_observed_days": 30,
        "current_refund_rate": pytest.approx(1 / 3),
        "refund_rate": pytest.approx(1 / 3),
        "evidence_value": pytest.approx(1 / 3),
    }


def test_all_supported_metrics_are_aggregated_before_ratios() -> None:
    sales, traffic, marketing = compatible_sources()
    sales = execution(
        "query_sales",
        [
            *[
                row.model_dump(exclude_unset=True)
                for row in sales.result.rows
            ],
            sales_row(
                "current",
                    "P002",
                region="West",
                channel="paid",
                gmv=80.0,
                orders=1,
                refund_orders=0,
            ),
        ],
    )
    traffic = execution(
        "query_traffic",
        [
            *[
                row.model_dump(exclude_unset=True)
                for row in traffic.result.rows
            ],
            traffic_row(
                "current",
                    "P002",
                impressions=100,
                clicks=50,
                visits=20,
            ),
        ],
    )
    marketing = execution(
        "query_marketing",
        [
            *[
                row.model_dump(exclude_unset=True)
                for row in marketing.result.rows
            ],
            marketing_row(
                "current",
                "P002",
                campaign_id="C002",
                spend=10.0,
            ),
        ],
    )
    metrics = [metric.value for metric in MetricName if metric.value != "evidence_value"]

    result = calculate_metrics(
        CalculateMetricsInput(metrics=metrics, group_by=["category"]),
        context_with(sales, traffic, marketing),
    )

    row = result.rows[0]
    assert row.current_gmv == 200.0
    assert row.current_orders == 4
    assert row.current_aov == pytest.approx(50.0)
    assert row.current_impressions == 1100
    assert row.current_clicks == 150
    assert row.current_ctr == pytest.approx(150 / 1100)
    assert row.current_visits == 100
    assert row.current_spend == 50.0
    assert row.current_roas == pytest.approx(4.0)


def test_zero_denominators_return_none() -> None:
    sources = (
        execution(
            "query_sales",
            [
                sales_row("current", gmv=0.0, orders=0),
                sales_row("previous", gmv=0.0, orders=0),
            ],
        ),
        execution(
            "query_traffic",
            [
                traffic_row("current", impressions=0, visits=0),
                traffic_row("previous", impressions=0, visits=0),
            ],
            result_id="result_0002",
        ),
        execution(
            "query_marketing",
            [
                marketing_row("current", spend=0.0),
                marketing_row("previous", spend=0.0),
            ],
            result_id="result_0003",
        ),
    )
    metrics = [
        "gmv_change_rate",
        "current_aov",
        "aov_change_rate",
        "current_ctr",
        "current_cvr",
        "current_roas",
        "refund_rate",
    ]
    result = calculate_metrics(
        CalculateMetricsInput(metrics=metrics, group_by=[]),
        context_with(*sources),
    )
    assert all(getattr(result.rows[0], metric) is None for metric in metrics)


def test_aov_change_rate_is_none_when_previous_aov_is_zero() -> None:
    sales = execution(
        "query_sales",
        [
            sales_row("current", gmv=120.0, orders=3),
            sales_row("previous", gmv=0.0, orders=2),
        ],
    )
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=["current_aov", "previous_aov", "aov_change_rate"],
            group_by=["product_id"],
        ),
        context_with(sales),
    )
    assert result.rows[0].current_aov == pytest.approx(40.0)
    assert result.rows[0].previous_aov == 0.0
    assert result.rows[0].aov_change_rate is None


def test_cvr_requires_each_period_to_have_its_complete_dynamic_window() -> None:
    sales, _traffic, _marketing = compatible_sources()
    traffic = execution(
        "query_traffic",
        [
            traffic_row(
                "current",
                visits=80,
                observed_days=30,
                missing_days=1,
            ),
            traffic_row(
                "previous",
                visits=50,
                observed_days=31,
                missing_days=0,
            ),
        ],
        arguments=window_arguments(
            start_date="2026-04-01",
            end_date="2026-05-01",
            comparison_start_date="2026-03-01",
            comparison_end_date="2026-03-31",
            include_missing=True,
        ),
    )
    sales = sales.model_copy(
        update={"arguments": window_arguments(
            start_date="2026-04-01",
            end_date="2026-05-01",
            comparison_start_date="2026-03-01",
            comparison_end_date="2026-03-31",
            include_refunds=True,
        )},
    )
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=[
                "current_cvr",
                "previous_cvr",
                "cvr_change",
                "cvr_change_rate",
                "traffic_change_rate",
            ],
            group_by=["product_id"],
        ),
        context_with(sales, traffic),
    )
    row = result.rows[0]
    assert row.current_cvr is None
    assert row.previous_cvr == pytest.approx(2 / 50)
    assert row.cvr_change is None
    assert row.cvr_change_rate is None
    assert row.traffic_change_rate == pytest.approx((80 - 50) / 50)


def test_evidence_value_priority_and_grouping_contract() -> None:
    sales, traffic, _marketing = compatible_sources()
    result = calculate_metrics(
        CalculateMetricsInput(metrics=["evidence_value"], group_by=["product_id"]),
        context_with(sales, traffic),
    )
    assert result.rows[0].evidence_value == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="group_by"):
        calculate_metrics(
            CalculateMetricsInput(metrics=["evidence_value"], group_by=[]),
            context_with(sales, traffic),
        )


@pytest.mark.parametrize(
    ("metrics", "sources", "message"),
    [
        (["current_gmv"], (), "requires query_sales"),
        (
            ["current_cvr"],
            (compatible_sources()[0],),
            "requires query_traffic",
        ),
        (
            ["current_gmv"],
            (
                compatible_sources()[0],
                compatible_sources()[0].model_copy(
                    update={"call_id": "call_sales_2"}
                ),
            ),
            "duplicate query_sales",
        ),
    ],
)
def test_missing_and_duplicate_required_sources_fail(
    metrics: list[str],
    sources: tuple[PriorToolExecution, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_metrics(
            CalculateMetricsInput(metrics=metrics, group_by=[]),
            context_with(*sources),
        )


@pytest.mark.parametrize(
    ("source_index", "argument_updates", "message"),
    [
        (1, {"end_date": "2026-04-29"}, "window"),
        (1, {"product_ids": ["P001"]}, "product_ids"),
        (0, {"include_refunds": False}, "include_refunds"),
        (1, {"include_missing": False}, "include_missing"),
    ],
)
def test_incompatible_arguments_fail(
    source_index: int,
    argument_updates: dict[str, object],
    message: str,
) -> None:
    sources = list(compatible_sources())
    source = sources[source_index]
    sources[source_index] = source.model_copy(
        update={"arguments": {**source.arguments, **argument_updates}},
    )
    with pytest.raises(ValueError, match=message):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["current_cvr", "current_roas", "refund_rate"],
                group_by=[],
            ),
            context_with(*sources),
        )


@pytest.mark.parametrize("field", ["dataset_id", "source_label"])
def test_incompatible_result_provenance_fails(field: str) -> None:
    sales, traffic, _marketing = compatible_sources()
    invalid = traffic.result.model_copy(
        update={field: "0123456789abcdef" if field == "dataset_id" else "Other"}
    )
    traffic = PriorToolExecution.model_construct(
        call_id=traffic.call_id,
        arguments=traffic.arguments,
        result=invalid,
    )
    with pytest.raises(ValueError, match=field):
        calculate_metrics(
            CalculateMetricsInput(metrics=["current_cvr"], group_by=[]),
            context_with(sales, traffic),
        )


def test_group_by_must_exist_in_every_source_and_duplicate_keys_fail() -> None:
    sales, traffic, _marketing = compatible_sources()
    with pytest.raises(ValueError, match="region"):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["current_cvr"],
                group_by=["region"],
            ),
            context_with(sales, traffic),
        )

    duplicate_traffic = execution(
        "query_traffic",
        [
            traffic_row("current", visits=1),
            traffic_row("current", visits=2),
            traffic_row("previous", visits=3),
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["current_visits"],
                group_by=["product_id"],
            ),
            context_with(duplicate_traffic),
        )


def test_dynamic_column_order_and_rows_follow_requested_group_order() -> None:
    sales = execution(
        "query_sales",
        [
            sales_row("current", "P002", category="B", gmv=20, orders=2),
            sales_row("previous", "P002", category="B", gmv=10, orders=1),
            sales_row("current", "P001", category="A", gmv=30, orders=3),
            sales_row("previous", "P001", category="A", gmv=20, orders=2),
        ],
    )
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=["previous_orders", "current_gmv"],
            group_by=["category", "product_id"],
        ),
        context_with(sales),
    )
    assert result.columns == [
        "category",
        "product_id",
        "previous_orders",
        "current_gmv",
    ]
    assert [
        (row.category, row.product_id)
        for row in result.rows
    ] == [("A", "P001"), ("B", "P002")]


def test_registry_is_wired_to_calculate_metrics() -> None:
    sales, _traffic, _marketing = compatible_sources()
    result = build_default_registry().invoke(
        "calculate_metrics",
        {"metrics": ["current_gmv"], "group_by": []},
        context_with(sales),
    )
    assert result.rows[0].current_gmv == 120.0


def test_case_003_matches_frozen_gold_without_runtime_gold_access(tmp_path: Path) -> None:
    case = next(
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["case_id"] == "CASE_003"
    )
    dataset_dir = tmp_path / "v1"
    build_snapshot(CONFIG, dataset_dir)
    registry = build_default_registry()
    ids = iter(["result_0001", "result_0002"])

    with open_dataset(dataset_dir) as catalog:
        query_context = ToolContext(
            catalog=catalog,
            prior_executions=(),
            next_result_id=lambda: next(ids),
        )
        sales_call, metrics_call = case["expected_tool_calls"]
        sales = registry.invoke(
            sales_call["name"],
            sales_call["parameters"],
            query_context,
        )
        metrics_context = ToolContext(
            catalog=catalog,
            prior_executions=(
                PriorToolExecution(
                    call_id="call_0001",
                    arguments=sales_call["parameters"],
                    result=sales,
                ),
            ),
            next_result_id=lambda: next(ids),
        )
        result = registry.invoke(
            metrics_call["name"],
            metrics_call["parameters"],
            metrics_context,
        )

    assert result.row_count == 1
    assert result.columns == metrics_call["parameters"]["metrics"]
    actual = result.rows[0].model_dump(exclude_unset=True)
    for metric, expected in case["gold_metrics"].items():
        if metric in {"current_orders", "previous_orders"}:
            assert actual[metric] == expected
        else:
            tolerance = case["numeric_tolerances"][metric]
            assert actual[metric] == pytest.approx(expected, abs=tolerance)
    assert actual["aov_change_rate"] == pytest.approx(
        0.003732444516315593,
        abs=0.001,
    )
