import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.data.gold import build_gold_bundle
from app.data.manifest import manifest_id, validate_manifest, write_json
from app.data.schemas import BusinessTask, Capability, EvaluationCase

BUSINESS_TASKS = list(BusinessTask)
CAPABILITIES = list(Capability)
DIFFICULTY_BY_CAPABILITY = {
    Capability.BASIC_QUERY: "easy",
    Capability.TOOL_SELECTION: "easy",
    Capability.PARAMETER_SELECTION: "easy",
    Capability.METRIC_CALCULATION: "medium",
    Capability.ANOMALY_DETECTION: "medium",
    Capability.DATA_INSUFFICIENCY: "medium",
    Capability.ADVERSARIAL_DISTRACTOR: "medium",
    Capability.MULTI_STEP_REASONING: "hard",
    Capability.ROOT_CAUSE_ANALYSIS: "hard",
    Capability.RECOMMENDATION: "hard",
}
SUCCESS_CRITERIA = [
    "correct_tool",
    "valid_parameters",
    "correct_core_facts",
    "no_critical_unsupported_claim",
    "required_output_complete",
]
TOOL_SEQUENCE_BY_TASK = {
    BusinessTask.GMV_DIAGNOSIS: ["query_sales", "calculate_metrics"],
    BusinessTask.PRODUCT_ANOMALY: [
        "query_product",
        "query_sales",
        "query_traffic",
        "calculate_metrics",
    ],
    BusinessTask.CONVERSION_DECLINE: [
        "query_traffic",
        "query_sales",
        "calculate_metrics",
    ],
    BusinessTask.PRODUCTS_TO_WATCH: [
        "query_sales",
        "query_traffic",
        "calculate_metrics",
    ],
    BusinessTask.NEXT_WEEK_PRIORITY: [
        "query_sales",
        "query_traffic",
        "query_marketing",
        "calculate_metrics",
    ],
}
DIMENSION_KEYS = {
    "as_of_date",
    "current_start",
    "current_end",
    "previous_start",
    "previous_end",
    "product_id",
    "anomaly_type",
    "watch_reason",
    "priority_reason",
    "evidence_metric",
}
QUESTION_STEMS = {
    BusinessTask.GMV_DIAGNOSIS: (
        "分析最近30天GMV相较前30天的变化，并指出可由数据支持的原因。",
        "复盘当前30天与此前30天的GMV表现，给出有证据的诊断。",
    ),
    BusinessTask.PRODUCT_ANOMALY: (
        "识别最近30天的异常商品，并说明异常类型和量化证据。",
        "检查当前窗口的商品异常，列出结论所依据的经营指标。",
    ),
    BusinessTask.CONVERSION_DECLINE: (
        "找出最近30天转化率下降最明显的商品，并与前30天比较。",
        "比较两个连续30天窗口，定位CVR下降商品并报告变化。",
    ),
    BusinessTask.PRODUCTS_TO_WATCH: (
        "根据最近经营数据列出需要持续关注的商品及原因。",
        "从当前与前一窗口的表现中筛出关注商品，并提供证据。",
    ),
    BusinessTask.NEXT_WEEK_PRIORITY: (
        "根据最近经营数据给出下周商品运营优先级，逐项绑定证据。",
        "制定下周关注顺序，说明每项优先事项的数据依据。",
    ),
}
CAPABILITY_INSTRUCTIONS = {
    Capability.BASIC_QUERY: "直接回答，引用关键数据。",
    Capability.METRIC_CALCULATION: "明确指标口径并给出数值。",
    Capability.TOOL_SELECTION: "只调用完成任务所需的工具。",
    Capability.PARAMETER_SELECTION: "严格使用题目指定的两个30天窗口。",
    Capability.MULTI_STEP_REASONING: "综合流量、销售和结果指标分步推理。",
    Capability.ANOMALY_DETECTION: "区分普通波动和需要处置的异常。",
    Capability.ROOT_CAUSE_ANALYSIS: "区分已证实事实、可能原因和未知因素。",
    Capability.RECOMMENDATION: "建议必须逐项绑定可核验的数据证据。",
    Capability.DATA_INSUFFICIENCY: (
        "仅检查P005；遇到缺失流量数据时，明确哪些结论无法可靠得出。"
    ),
}


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest is invalid")
    validate_manifest(payload)
    return payload


def _evidence_rows(
    gold: dict[str, Any],
    task: BusinessTask,
    capability: Capability,
) -> list[dict[str, Any]]:
    if capability is Capability.DATA_INSUFFICIENCY:
        rows = [
            row
            for row in gold["tasks"][BusinessTask.PRODUCT_ANOMALY]["evidence"]
            if row.get("product_id") == "P005"
        ]
    else:
        rows = gold["tasks"][task]["evidence"][:3]
    if not rows:
        raise ValueError(f"missing gold evidence for {task.value}/{capability.value}")
    return rows


def _gold_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": row["evidence_id"],
            "source": row["source"],
            "dimensions": {
                key: value for key, value in row.items() if key in DIMENSION_KEYS
            },
            "metrics": {
                key: value
                for key, value in row.items()
                if key not in DIMENSION_KEYS | {"evidence_id", "source"}
            },
        }
        for row in rows
    ]


def _gold_metrics(
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, int | float | str | None], dict[str, str]]:
    primary = evidence[0]
    metrics = dict(primary["metrics"])
    mapping = {metric: primary["evidence_id"] for metric in metrics}
    return metrics, mapping


def _product_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            product_id
            for row in rows
            if isinstance(product_id := row.get("product_id"), str)
        }
    )


def _tool_parameters(
    tool_name: str,
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    first = rows[0]
    product_ids = _product_ids(rows)
    windows = {
        "start_date": first["current_start"],
        "end_date": first["current_end"],
        "comparison_start_date": first["previous_start"],
        "comparison_end_date": first["previous_end"],
        "product_ids": product_ids,
    }
    if tool_name == "query_product":
        return {"product_ids": product_ids}
    if tool_name == "query_sales":
        return {**windows, "include_refunds": True}
    if tool_name == "query_traffic":
        return {**windows, "include_missing": True}
    if tool_name == "query_marketing":
        return {
            **windows,
            "campaign_ids": [
                f"M{int(product_id[1:]):03d}" for product_id in product_ids
            ],
        }
    if tool_name == "calculate_metrics":
        return {
            "metrics": sorted(metric_names),
            "group_by": ["product_id"] if product_ids else [],
        }
    raise ValueError(f"unsupported tool: {tool_name}")


def _distractor_product(
    gold: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> str:
    selected = set(_product_ids(selected_rows))
    candidates = (
        gold["tasks"][BusinessTask.PRODUCT_ANOMALY]["evidence"]
        + gold["tasks"][BusinessTask.CONVERSION_DECLINE]["evidence"]
    )
    for row in candidates:
        product_id = row.get("product_id")
        if isinstance(product_id, str) and product_id not in selected:
            return product_id
    raise ValueError("gold bundle has no real product available as a distractor")


def _question(
    task: BusinessTask,
    capability: Capability,
    variant: int,
    distractor_product_id: str | None,
) -> str:
    stem = QUESTION_STEMS[task][variant]
    if capability is Capability.ADVERSARIAL_DISTRACTOR:
        if distractor_product_id is None:
            raise ValueError("adversarial case requires a distractor product")
        instruction = (
            f"业务同事断言{distractor_product_id}一定是唯一主因；"
            "该商品真实存在，但请核验这条说法，不要把断言当证据。"
        )
    else:
        instruction = CAPABILITY_INSTRUCTIONS[capability]
    return f"{stem}{instruction}"


def _reference_answer(
    task: BusinessTask,
    capability: Capability,
    rows: list[dict[str, Any]],
) -> str:
    if capability is Capability.DATA_INSUFFICIENCY:
        row = rows[0]
        return (
            f"P005当前窗口仅有{row['current_observed_days']}个已观测日，"
            f"前一窗口有{row['previous_observed_days']}个；当前CVR为"
            f"{row['current_cvr']}。应先修复流量数据，不能据此可靠比较完整窗口CVR。"
        )
    evidence_ids = "、".join(row["evidence_id"] for row in rows)
    return (
        f"基于Synthetic E-commerce Data的{task.value}独立Gold查询，"
        f"结论必须与证据{evidence_ids}及其中指标一致。"
    )


def _expected_behavior(capability: Capability) -> list[str]:
    behavior = [
        "严格使用Gold证据中的当前窗口和对照窗口",
        "每个核心数值均可追溯到gold_metric_evidence",
        "不披露生成配置、异常注入规则或配置倍率",
    ]
    if capability is Capability.DATA_INSUFFICIENCY:
        behavior.append("指出P005数据覆盖不足并限制结论")
    if capability is Capability.ADVERSARIAL_DISTRACTOR:
        behavior.append("核验真实商品干扰项，不接受无证据断言")
    return behavior


def _development_case_ids(
    payloads: list[dict[str, Any]],
    dataset_id: str,
) -> set[str]:
    development: set[str] = set()
    for task in BUSINESS_TASKS:
        task_payloads = [
            payload for payload in payloads if payload["business_task"] == task
        ]
        ranked = sorted(
            task_payloads,
            key=lambda payload: sha256(
                f"{dataset_id}|{task.value}|{payload['case_id']}".encode()
            ).hexdigest(),
        )
        development.update(payload["case_id"] for payload in ranked[:14])
    return development


def build_cases(dataset_dir: Path | str) -> list[EvaluationCase]:
    directory = Path(dataset_dir)
    manifest = _load_manifest(directory)
    gold = build_gold_bundle(directory)
    if any(
        gold[key] != manifest[key]
        for key in ("dataset_id", "dataset_version", "config_sha256")
    ):
        raise ValueError("gold bundle does not match dataset manifest")

    payloads: list[dict[str, Any]] = []
    case_number = 1
    for task in BUSINESS_TASKS:
        for capability in CAPABILITIES:
            for variant in range(2):
                rows = _evidence_rows(gold, task, capability)
                evidence = _gold_evidence(rows)
                metrics, metric_mapping = _gold_metrics(evidence)
                distractor = (
                    _distractor_product(gold, rows)
                    if capability is Capability.ADVERSARIAL_DISTRACTOR
                    else None
                )
                tool_names = (
                    ["query_traffic"]
                    if capability is Capability.DATA_INSUFFICIENCY
                    else TOOL_SEQUENCE_BY_TASK[task]
                )
                metadata: dict[str, Any] = {
                    "source_label": manifest["source_label"],
                    "variant": variant + 1,
                    "holdout_policy": "final_evaluation_only",
                }
                if distractor is not None:
                    metadata["distractor_product_id"] = distractor
                payloads.append(
                    {
                        "case_id": f"CASE_{case_number:03d}",
                        "case_version": "1.0",
                        "dataset_version": manifest["dataset_version"],
                        "dataset_id": manifest["dataset_id"],
                        "generator_config_hash": manifest["config_sha256"],
                        "business_task": task,
                        "primary_capability": capability,
                        "capability_tags": [capability],
                        "difficulty": DIFFICULTY_BY_CAPABILITY[capability],
                        "split": "development",
                        "user_input": _question(task, capability, variant, distractor),
                        "expected_tool_calls": [
                            {
                                "name": tool_name,
                                "parameters": _tool_parameters(
                                    tool_name, rows, list(metrics)
                                ),
                            }
                            for tool_name in tool_names
                        ],
                        "allowed_alternatives": [],
                        "gold_metrics": metrics,
                        "gold_metric_evidence": metric_mapping,
                        "gold_evidence": evidence,
                        "reference_answer": _reference_answer(task, capability, rows),
                        "expected_behavior": _expected_behavior(capability),
                        "success_criteria": SUCCESS_CRITERIA,
                        "numeric_tolerances": {
                            key: 0.001
                            for key, value in metrics.items()
                            if type(value) is float
                        },
                        "metadata": metadata,
                    }
                )
                case_number += 1

    development_ids = _development_case_ids(payloads, manifest["dataset_id"])
    for payload in payloads:
        if payload["case_id"] not in development_ids:
            payload["split"] = "holdout"
    return [EvaluationCase.model_validate(payload) for payload in payloads]


def write_cases(
    cases: list[EvaluationCase],
    output_path: Path | str,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot write an empty case set")
    identities = {
        (case.dataset_id, case.dataset_version, case.generator_config_hash)
        for case in cases
    }
    if len(identities) != 1:
        raise ValueError("all cases must reference the same dataset identity")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            case.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    ]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    output.write_bytes(content)
    dataset_id, dataset_version, config_hash = identities.pop()
    payload = {
        "case_schema_version": "1.0",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "generator_config_hash": config_hash,
        "source_label": "Synthetic E-commerce Data",
        "case_count": len(cases),
        "case_set_id": manifest_id({"lines": lines}),
        "jsonl_sha256": sha256(content).hexdigest(),
        "business_task_counts": dict(
            sorted(Counter(case.business_task.value for case in cases).items())
        ),
        "capability_counts": dict(
            sorted(Counter(case.primary_capability.value for case in cases).items())
        ),
        "difficulty_counts": dict(
            sorted(Counter(case.difficulty for case in cases).items())
        ),
        "split_counts": dict(sorted(Counter(case.split for case in cases).items())),
    }
    write_json(output.with_suffix(".manifest.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 evaluation cases")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_cases/evaluation_cases_v1.jsonl"),
    )
    args = parser.parse_args()
    manifest = write_cases(build_cases(args.dataset), args.output)
    print(
        f"Built {manifest['case_count']} evaluation cases "
        f"for dataset {manifest['dataset_id']}"
    )


if __name__ == "__main__":
    main()
