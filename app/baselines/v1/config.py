import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from app.tools.schemas import MetricName

from .schemas import TaskKind

POLICY_NAME = "deterministic-baseline-v1"
POLICY_VERSION = "1.0.0"
DEFAULT_WINDOW_DAYS = 30
DEFAULT_TOP_K = 5
ANSWER_TEMPLATE_VERSION = "1.0"

ROUTING_PRIORITY = (
    TaskKind.PRODUCT_ANOMALY,
    TaskKind.CONVERSION_DECLINE,
    TaskKind.NEXT_WEEK_PRIORITY,
    TaskKind.PRODUCTS_TO_WATCH,
    TaskKind.GMV_DIAGNOSIS,
)

ROUTING_KEYWORDS: Mapping[TaskKind, tuple[tuple[str, ...], ...]] = MappingProxyType(
    {
        TaskKind.PRODUCT_ANOMALY: (
            ("商品", "sku"),
            ("异常", "问题", "原因"),
        ),
        TaskKind.CONVERSION_DECLINE: (
            ("转化", "cvr"),
            ("下降", "下滑", "原因"),
        ),
        TaskKind.NEXT_WEEK_PRIORITY: (
            ("下周", "下一步"),
            ("优先", "重点", "动作"),
        ),
        TaskKind.PRODUCTS_TO_WATCH: (
            ("关注", "预警", "风险"),
            ("商品", "sku"),
        ),
        TaskKind.GMV_DIAGNOSIS: (
            ("gmv", "销售额", "营收", "成交"),
            ("变化", "趋势", "诊断"),
        ),
    }
)

TOOL_PATHS: Mapping[TaskKind, tuple[str, ...]] = MappingProxyType(
    {
        TaskKind.PRODUCT_ANOMALY: (
            "query_product",
            "query_sales",
            "query_traffic",
            "calculate_metrics",
        ),
        TaskKind.CONVERSION_DECLINE: (
            "query_traffic",
            "query_sales",
            "calculate_metrics",
        ),
        TaskKind.NEXT_WEEK_PRIORITY: (
            "query_sales",
            "query_traffic",
            "calculate_metrics",
        ),
        TaskKind.PRODUCTS_TO_WATCH: (
            "query_sales",
            "query_traffic",
            "calculate_metrics",
        ),
        TaskKind.GMV_DIAGNOSIS: ("query_sales", "calculate_metrics"),
    }
)

METRICS: Mapping[TaskKind, tuple[MetricName, ...]] = MappingProxyType(
    {
        TaskKind.PRODUCT_ANOMALY: (
            MetricName.CURRENT_GMV,
            MetricName.PREVIOUS_GMV,
            MetricName.GMV_CHANGE_RATE,
            MetricName.CURRENT_ORDERS,
            MetricName.PREVIOUS_ORDERS,
            MetricName.CURRENT_REFUND_RATE,
            MetricName.CURRENT_VISITS,
            MetricName.PREVIOUS_VISITS,
            MetricName.TRAFFIC_CHANGE_RATE,
            MetricName.CURRENT_CTR,
            MetricName.PREVIOUS_CTR,
            MetricName.CURRENT_CVR,
            MetricName.PREVIOUS_CVR,
            MetricName.CVR_CHANGE,
        ),
        TaskKind.CONVERSION_DECLINE: (
            MetricName.CURRENT_VISITS,
            MetricName.PREVIOUS_VISITS,
            MetricName.CURRENT_ORDERS,
            MetricName.PREVIOUS_ORDERS,
            MetricName.CURRENT_CVR,
            MetricName.PREVIOUS_CVR,
            MetricName.CVR_CHANGE,
            MetricName.CVR_CHANGE_RATE,
            MetricName.CURRENT_OBSERVED_DAYS,
            MetricName.PREVIOUS_OBSERVED_DAYS,
        ),
        TaskKind.NEXT_WEEK_PRIORITY: (
            MetricName.CURRENT_GMV,
            MetricName.GMV_CHANGE_RATE,
            MetricName.CURRENT_VISITS,
            MetricName.TRAFFIC_CHANGE_RATE,
            MetricName.CURRENT_CVR,
            MetricName.CVR_CHANGE,
        ),
        TaskKind.PRODUCTS_TO_WATCH: (
            MetricName.EVIDENCE_VALUE,
            MetricName.GMV_CHANGE_RATE,
            MetricName.TRAFFIC_CHANGE_RATE,
            MetricName.CVR_CHANGE,
            MetricName.CURRENT_REFUND_RATE,
        ),
        TaskKind.GMV_DIAGNOSIS: (
            MetricName.CURRENT_GMV,
            MetricName.PREVIOUS_GMV,
            MetricName.GMV_CHANGE_RATE,
            MetricName.CURRENT_ORDERS,
            MetricName.PREVIOUS_ORDERS,
            MetricName.CURRENT_AOV,
            MetricName.PREVIOUS_AOV,
            MetricName.AOV_CHANGE_RATE,
        ),
    }
)

STABLE_SORT_FIELDS: Mapping[TaskKind, tuple[str, ...]] = MappingProxyType(
    {
        TaskKind.PRODUCT_ANOMALY: ("gmv_change_rate:asc", "product_id:asc"),
        TaskKind.CONVERSION_DECLINE: ("cvr_change:asc", "product_id:asc"),
        TaskKind.NEXT_WEEK_PRIORITY: ("gmv_change_rate:asc", "product_id:asc"),
        TaskKind.PRODUCTS_TO_WATCH: ("evidence_value:desc", "product_id:asc"),
        TaskKind.GMV_DIAGNOSIS: ("product_id:asc",),
    }
)


def policy_snapshot() -> dict[str, Any]:
    return {
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "routing_priority": [task.value for task in ROUTING_PRIORITY],
        "routing_keywords": {
            task.value: [list(group) for group in keyword_groups]
            for task, keyword_groups in ROUTING_KEYWORDS.items()
        },
        "default_window_days": DEFAULT_WINDOW_DAYS,
        "default_top_k": DEFAULT_TOP_K,
        "tool_paths": {
            task.value: list(path) for task, path in TOOL_PATHS.items()
        },
        "metrics": {
            task.value: [metric.value for metric in metrics]
            for task, metrics in METRICS.items()
        },
        "stable_sort_fields": {
            task.value: list(fields) for task, fields in STABLE_SORT_FIELDS.items()
        },
        "answer_template_version": ANSWER_TEMPLATE_VERSION,
    }


def canonical_policy_bytes(snapshot: Mapping[str, Any] | None = None) -> bytes:
    payload = policy_snapshot() if snapshot is None else snapshot
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{serialized}\n".encode()
