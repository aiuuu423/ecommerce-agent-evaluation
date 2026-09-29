import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.tools import (
    ToolContext,
    ToolDefinition,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    UnknownToolError,
    build_default_registry,
)
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
CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"
EXPECTED_TOOL_SCHEMA_SHA256 = (
    "9a501e7839e32df5bfbf965d3bfa3921ce6212456e56170bf09750e8ff368343"
)
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


def _valid_sales_arguments() -> dict[str, object]:
    return {**_window_arguments(), "include_refunds": True}


def _valid_context() -> ToolContext:
    return ToolContext(
        catalog=object(),
        prior_executions=(),
        next_result_id=lambda: "result_0001",
    )


def _malformed_query_sales_definition() -> ToolDefinition:
    def malformed_handler(_arguments, _context):
        return QuerySalesResult.model_construct(
            **_result_envelope("query_sales", [], [])
        )

    return ToolDefinition(
        name="query_sales",
        description="Malformed sales tool for boundary testing.",
        input_model=QuerySalesInput,
        output_model=QuerySalesResult,
        handler=malformed_handler,
    )


def test_registry_rejects_duplicates_and_exports_stable_schema() -> None:
    registry = build_default_registry()
    assert registry.names() == (
        "calculate_metrics",
        "query_marketing",
        "query_product",
        "query_sales",
        "query_traffic",
    )
    exported = registry.openai_tools()
    assert [item["function"]["name"] for item in exported] == list(registry.names())
    assert all(
        item["function"]["parameters"]["additionalProperties"] is False
        for item in exported
    )
    assert {
        name: registry.get(name).output_model
        for name in registry.names()
    } == {
        "calculate_metrics": CalculateMetricsResult,
        "query_marketing": QueryMarketingResult,
        "query_product": QueryProductResult,
        "query_sales": QuerySalesResult,
        "query_traffic": QueryTrafficResult,
    }
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("query_sales"))


def test_registry_schema_required_exactly_matches_frozen_contract() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    exported = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in build_default_registry().openai_tools()
    }
    assert set(contract["tools"]) <= set(exported)
    for name, frozen in contract["tools"].items():
        assert exported[name]["required"] == frozen["required_parameters"]


def test_registry_rejects_unknown_tools_and_invalid_input() -> None:
    registry = build_default_registry()
    with pytest.raises(UnknownToolError, match="unknown tool"):
        registry.get("unknown")
    with pytest.raises(UnknownToolError, match="unknown tool"):
        registry.invoke("unknown", {}, context=_valid_context())
    with pytest.raises(ToolInputValidationError) as exc_info:
        registry.invoke(
            "query_sales",
            {"start_date": "not-a-date"},
            context=_valid_context(),
        )
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_registry_does_not_reclassify_handler_key_error() -> None:
    def failing_handler(_arguments, _context):
        raise KeyError("handler-miss")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="query_sales",
            description="Failing sales handler for exception testing.",
            input_model=QuerySalesInput,
            output_model=QuerySalesResult,
            handler=failing_handler,
        )
    )
    with pytest.raises(KeyError, match="handler-miss"):
        registry.invoke("query_sales", _valid_sales_arguments(), _valid_context())


def test_registry_revalidates_handler_output_with_bound_model() -> None:
    registry = ToolRegistry()
    registry.register(_malformed_query_sales_definition())
    with pytest.raises(ToolOutputValidationError) as exc_info:
        registry.invoke("query_sales", _valid_sales_arguments(), _valid_context())
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_calculate_metrics_uses_request_dependent_output_validator() -> None:
    default_definition = build_default_registry().get("calculate_metrics")

    def wrong_column_order_handler(_arguments, _context):
        return CalculateMetricsResult.model_validate(
            _result_envelope(
                "calculate_metrics",
                ["current_gmv", "product_id"],
                [{"current_gmv": 10.0, "product_id": "P001"}],
            )
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name=default_definition.name,
            description=default_definition.description,
            input_model=default_definition.input_model,
            output_model=default_definition.output_model,
            handler=wrong_column_order_handler,
            output_validator=default_definition.output_validator,
        )
    )
    with pytest.raises(ToolOutputValidationError, match="invalid result") as exc_info:
        registry.invoke(
            "calculate_metrics",
            {"metrics": ["current_gmv"], "group_by": ["product_id"]},
            _valid_context(),
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_openai_tool_schema_digest_is_frozen_and_json_serializable() -> None:
    exported = build_default_registry().openai_tools()
    canonical = json.dumps(
        exported,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert json.loads(canonical) == exported
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_TOOL_SCHEMA_SHA256
