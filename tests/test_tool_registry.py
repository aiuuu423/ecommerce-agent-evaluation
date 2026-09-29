import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tools.schemas import (
    CalculateMetricsInput,
    CalculateMetricsResult,
    MetricName,
    QueryMarketingInput,
    QueryMarketingResult,
    QueryProductInput,
    QueryProductResult,
    QuerySalesInput,
    QuerySalesResult,
    QueryTrafficInput,
    QueryTrafficResult,
    canonical_tool_result_payload,
)

ROOT = Path(__file__).parents[1]
CASES = ROOT / "data/evaluation_cases/v1/cases.jsonl"
FROZEN_CASE_METRICS = {
    "aov_change_rate",
    "current_aov",
    "current_cvr",
    "current_gmv",
    "current_observed_days",
    "current_orders",
    "current_refund_rate",
    "current_visits",
    "cvr_change",
    "evidence_value",
    "gmv_change_rate",
    "previous_aov",
    "previous_cvr",
    "previous_gmv",
    "previous_observed_days",
    "previous_orders",
    "previous_visits",
    "refund_rate",
    "traffic_change_rate",
}


def _window_arguments() -> dict[str, object]:
    return {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
    }


def _result_envelope(tool_name: str, columns: list[str], rows: list[dict]) -> dict:
    return {
        "result_id": "result_0001",
        "tool_name": tool_name,
        "dataset_id": "e1e81533c25e03e5",
        "source_label": "Synthetic E-commerce Data",
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


def test_metric_name_covers_exact_frozen_case_metric_vocabulary() -> None:
    case_metrics = {
        metric
        for line in CASES.read_text(encoding="utf-8").splitlines()
        for call in json.loads(line)["expected_tool_calls"]
        if call["name"] == "calculate_metrics"
        for metric in call["parameters"]["metrics"]
    }
    assert len(FROZEN_CASE_METRICS) == 19
    assert case_metrics == FROZEN_CASE_METRICS
    assert case_metrics <= {metric.value for metric in MetricName}


def test_required_empty_collections_cannot_be_omitted() -> None:
    sales = {**_window_arguments(), "include_refunds": True}
    traffic = {**_window_arguments(), "include_missing": True}
    sales.pop("product_ids")
    traffic.pop("product_ids")

    with pytest.raises(ValidationError, match="product_ids"):
        QuerySalesInput.model_validate(sales)
    with pytest.raises(ValidationError, match="product_ids"):
        QueryTrafficInput.model_validate(traffic)
    with pytest.raises(ValidationError, match="group_by"):
        CalculateMetricsInput.model_validate({"metrics": ["current_gmv"]})


def test_all_300_frozen_case_calls_pass_corresponding_strict_input_model() -> None:
    input_models = {
        "query_product": QueryProductInput,
        "query_sales": QuerySalesInput,
        "query_traffic": QueryTrafficInput,
        "calculate_metrics": CalculateMetricsInput,
    }
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
    ]
    calls = [call for case in cases for call in case["expected_tool_calls"]]
    assert len(cases) == 100
    assert len(calls) == 300
    for call in calls:
        model = input_models[call["name"]]
        parsed = model.model_validate_json(
            json.dumps(call["parameters"], separators=(",", ":")),
            strict=True,
        )
        assert parsed.model_dump(mode="json") == call["parameters"]


@pytest.mark.parametrize(
    "model, flag",
    [
        (QuerySalesInput, {"include_refunds": True}),
        (QueryTrafficInput, {"include_missing": True}),
        (QueryMarketingInput, {}),
    ],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {"start_date": "2026-05-01", "end_date": "2026-04-30"},
        {
            "comparison_start_date": "2026-04-01",
            "comparison_end_date": "2026-03-31",
        },
    ],
)
def test_every_query_window_is_ordered(model, flag, overrides) -> None:
    values = {**_window_arguments(), **flag, **overrides}
    with pytest.raises(ValidationError, match="start_date"):
        model.model_validate(values)


@pytest.mark.parametrize(
    "model, flag",
    [
        (QuerySalesInput, {"include_refunds": True}),
        (QueryTrafficInput, {"include_missing": True}),
        (QueryMarketingInput, {}),
    ],
)
def test_all_window_query_inputs_reject_duplicate_product_ids(model, flag) -> None:
    values = {
        **_window_arguments(),
        "product_ids": ["P001", "P001"],
        **flag,
    }
    with pytest.raises(ValidationError, match="unique"):
        model.model_validate(values)


def test_query_product_rejects_duplicate_or_malformed_product_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        QueryProductInput.model_validate({"product_ids": ["P001", "P001"]})
    with pytest.raises(ValidationError, match="product_ids"):
        QueryProductInput.model_validate({"product_ids": ["product-1"]})


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        (QueryProductInput, {"product_ids": ["P001"], "unexpected": True}),
        (
            QuerySalesInput,
            {**_window_arguments(), "include_refunds": True, "unexpected": True},
        ),
        (
            QueryTrafficInput,
            {**_window_arguments(), "include_missing": True, "unexpected": True},
        ),
        (QueryMarketingInput, {**_window_arguments(), "unexpected": True}),
        (
            CalculateMetricsInput,
            {"metrics": ["current_gmv"], "group_by": [], "rows": []},
        ),
    ],
)
def test_all_five_inputs_forbid_extra_fields(model, arguments) -> None:
    with pytest.raises(ValidationError, match="extra"):
        model.model_validate(arguments)


def test_metric_input_matches_frozen_contract_and_rejects_duplicates() -> None:
    parsed = CalculateMetricsInput.model_validate(
        {"metrics": ["current_gmv"], "group_by": []}
    )
    assert parsed.metrics == ["current_gmv"]
    with pytest.raises(ValidationError, match="unique"):
        CalculateMetricsInput.model_validate(
            {"metrics": ["current_gmv", "current_gmv"], "group_by": []}
        )
    with pytest.raises(ValidationError, match="unique"):
        CalculateMetricsInput.model_validate(
            {"metrics": ["current_gmv"], "group_by": ["product_id", "product_id"]}
        )


def test_dedicated_tool_results_bind_their_tool_names() -> None:
    result_models = {
        QueryProductResult: "query_product",
        QuerySalesResult: "query_sales",
        QueryTrafficResult: "query_traffic",
        QueryMarketingResult: "query_marketing",
        CalculateMetricsResult: "calculate_metrics",
    }
    for model, tool_name in result_models.items():
        with pytest.raises(ValidationError, match="tool_name"):
            model.model_validate(
                _result_envelope("wrong_tool", [], [])
            )
        if model is CalculateMetricsResult:
            parsed = model.model_validate(_result_envelope(tool_name, [], []))
            assert parsed.tool_name == tool_name


def test_dedicated_tool_result_rejects_non_finite_or_wrong_row_shape() -> None:
    columns = [
        "period",
        "product_id",
        "category",
        "region",
        "channel",
        "gmv",
        "orders",
        "units",
        "refund_orders",
    ]
    row = {
        "period": "current",
        "product_id": "P001",
        "category": "A",
        "region": "East",
        "channel": "organic",
        "gmv": float("nan"),
        "orders": 1,
        "units": 1,
        "refund_orders": 0,
    }
    with pytest.raises(ValidationError):
        QuerySalesResult.model_validate(
            _result_envelope("query_sales", columns, [row])
        )

    row["gmv"] = 10.0
    with pytest.raises(ValidationError, match="columns"):
        QuerySalesResult.model_validate(
            _result_envelope("query_sales", columns[:-1], [row])
        )


def test_result_envelope_validates_ids_counts_and_unique_columns() -> None:
    with pytest.raises(ValidationError, match="result_id"):
        CalculateMetricsResult.model_validate(
            _result_envelope("calculate_metrics", [], []) | {"result_id": "result_1"}
        )
    with pytest.raises(ValidationError, match="dataset_id"):
        CalculateMetricsResult.model_validate(
            _result_envelope("calculate_metrics", [], []) | {"dataset_id": "not-an-id"}
        )
    with pytest.raises(ValidationError, match="row_count"):
        CalculateMetricsResult.model_validate(
            _result_envelope("calculate_metrics", [], []) | {"row_count": 1}
        )
    with pytest.raises(ValidationError, match="unique"):
        CalculateMetricsResult.model_validate(
            _result_envelope("calculate_metrics", ["category", "category"], [])
        )


def test_calculate_metrics_columns_define_validation_and_json_order() -> None:
    result = CalculateMetricsResult.model_validate(
        _result_envelope(
            "calculate_metrics",
            ["product_id", "previous_orders", "current_gmv"],
            [{
                "product_id": "P001",
                "current_gmv": 120.0,
                "previous_orders": 2,
            }],
        )
    )
    payload = canonical_tool_result_payload(result)
    assert list(payload) == [
        "result_id",
        "tool_name",
        "dataset_id",
        "source_label",
        "columns",
        "rows",
        "row_count",
        "warnings",
    ]
    assert set(payload["rows"][0]) == set(result.columns)
    assert list(payload["rows"][0]) == result.columns
