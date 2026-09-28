import argparse
import fcntl
import json
import os
import shutil
import tempfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from app.data.gold import build_gold_bundle
from app.data.manifest import (
    contained_path,
    file_sha256,
    manifest_id,
    validate_manifest,
    write_json,
)
from app.data.schemas import BusinessTask, Capability, EvaluationCase

BUSINESS_TASKS = list(BusinessTask)
CAPABILITIES = list(Capability)
COMPLEXITY_WEIGHT_BY_CAPABILITY = {
    Capability.BASIC_QUERY: 0,
    Capability.TOOL_SELECTION: 1,
    Capability.PARAMETER_SELECTION: 1,
    Capability.METRIC_CALCULATION: 2,
    Capability.ANOMALY_DETECTION: 3,
    Capability.DATA_INSUFFICIENCY: 4,
    Capability.ADVERSARIAL_DISTRACTOR: 4,
    Capability.MULTI_STEP_REASONING: 5,
    Capability.ROOT_CAUSE_ANALYSIS: 6,
    Capability.RECOMMENDATION: 6,
}
SUCCESS_CRITERIA = [
    "correct_tool",
    "valid_parameters",
    "correct_core_facts",
    "no_critical_unsupported_claim",
    "required_output_complete",
]
DEFAULT_TOOL_CONTRACT = Path("configs/evaluation/tool_contract_v1.yaml")
LIST_TOP_K = 3
LIST_TASKS = {
    BusinessTask.PRODUCT_ANOMALY,
    BusinessTask.CONVERSION_DECLINE,
    BusinessTask.PRODUCTS_TO_WATCH,
    BusinessTask.NEXT_WEEK_PRIORITY,
}
PRODUCT_LEVEL_GOLD_TASKS = (
    BusinessTask.PRODUCT_ANOMALY,
    BusinessTask.CONVERSION_DECLINE,
    BusinessTask.PRODUCTS_TO_WATCH,
    BusinessTask.NEXT_WEEK_PRIORITY,
)
DATA_INSUFFICIENCY_BY_TASK = {
    BusinessTask.GMV_DIAGNOSIS: {
        "evidence_task": BusinessTask.PRODUCT_ANOMALY,
        "answerable_metrics": ("gmv_change_rate", "aov_change_rate"),
        "unanswerable_metrics": (
            "traffic_change_rate",
            "current_cvr",
            "cvr_change",
        ),
        "tool_names": ("query_sales", "calculate_metrics"),
    },
    BusinessTask.PRODUCT_ANOMALY: {
        "evidence_task": BusinessTask.PRODUCT_ANOMALY,
        "answerable_metrics": ("current_observed_days", "previous_observed_days"),
        "unanswerable_metrics": (
            "traffic_change_rate",
            "current_cvr",
            "cvr_change",
        ),
        "tool_names": ("query_product", "query_traffic"),
    },
    BusinessTask.CONVERSION_DECLINE: {
        "evidence_task": BusinessTask.PRODUCT_ANOMALY,
        "answerable_metrics": (
            "previous_cvr",
            "current_observed_days",
            "previous_observed_days",
        ),
        "unanswerable_metrics": ("current_cvr", "cvr_change"),
        "tool_names": ("query_traffic", "query_sales", "calculate_metrics"),
    },
    BusinessTask.PRODUCTS_TO_WATCH: {
        "evidence_task": BusinessTask.PRODUCTS_TO_WATCH,
        "answerable_metrics": (
            "gmv_change_rate",
            "refund_rate",
            "current_observed_days",
            "previous_observed_days",
        ),
        "unanswerable_metrics": ("current_cvr", "cvr_change"),
        "tool_names": ("query_sales", "query_traffic", "calculate_metrics"),
    },
    BusinessTask.NEXT_WEEK_PRIORITY: {
        "evidence_task": BusinessTask.NEXT_WEEK_PRIORITY,
        "answerable_metrics": (
            "evidence_value",
            "current_observed_days",
            "previous_observed_days",
        ),
        "unanswerable_metrics": ("current_cvr", "cvr_change"),
        "tool_names": ("query_traffic",),
    },
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


def _load_tool_contract(path: Path | str) -> tuple[dict[str, Any], str]:
    contract_path = Path(path)
    contents = contract_path.read_bytes()
    payload = yaml.safe_load(contents.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("contract_version"), str)
        or not isinstance(payload.get("tools"), dict)
        or not isinstance(payload.get("paths"), dict)
        or not isinstance(payload.get("adversarial_prefix"), list)
    ):
        raise ValueError("tool contract is invalid")
    expected_tasks = {task.value for task in BUSINESS_TASKS}
    if set(payload["paths"]) != expected_tasks:
        raise ValueError("tool contract must define every business task")
    known_tools = set(payload["tools"])
    for path in [*payload["paths"].values(), payload["adversarial_prefix"]]:
        if not isinstance(path, list) or not path or not all(
            isinstance(name, str) and name in known_tools for name in path
        ):
            raise ValueError("tool contract contains an invalid path")
    return payload, sha256(contents).hexdigest()


def _evidence_rows(
    gold: dict[str, Any],
    task: BusinessTask,
    capability: Capability,
) -> list[dict[str, Any]]:
    if capability is Capability.DATA_INSUFFICIENCY:
        evidence_task = DATA_INSUFFICIENCY_BY_TASK[task]["evidence_task"]
        rows = [
            row
            for row in gold["tasks"][evidence_task]["evidence"]
            if row.get("product_id") == "P005"
        ]
    else:
        limit = LIST_TOP_K if task in LIST_TASKS else 1
        rows = gold["tasks"][task]["evidence"][:limit]
    if not rows:
        raise ValueError(f"missing gold evidence for {task.value}/{capability.value}")
    return rows


def _gold_evidence(
    rows: list[dict[str, Any]],
    metric_names: set[str] | None = None,
) -> list[dict[str, Any]]:
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
                and (metric_names is None or key in metric_names)
            },
        }
        for row in rows
    ]


def _gold_metrics(
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, int | float | str | None], dict[str, str]]:
    metrics: dict[str, int | float | str | None] = {}
    mapping: dict[str, str] = {}
    qualify = len(evidence) > 1
    for position, row in enumerate(evidence, start=1):
        dimension = row["dimensions"].get("product_id", f"row_{position}")
        scoreable_row: dict[str, int | float | str | None] = {}
        for metric, value in row["metrics"].items():
            score_key = f"{dimension}.{metric}" if qualify else metric
            if score_key in metrics:
                raise ValueError(f"duplicate scoreable evidence metric: {score_key}")
            metrics[score_key] = value
            mapping[score_key] = row["evidence_id"]
            scoreable_row[score_key] = value
        row["metrics"] = scoreable_row
    return metrics, mapping


def _base_metric_names(metrics: dict[str, Any]) -> list[str]:
    return sorted({metric.rsplit(".", 1)[-1] for metric in metrics})


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
    distractor_product_id: str | None = None,
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
        query_product_ids = set(product_ids)
        if distractor_product_id is not None:
            query_product_ids.add(distractor_product_id)
        return {"product_ids": sorted(query_product_ids)}
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
    product_ids: set[str],
    dataset_id: str,
    task: BusinessTask,
) -> str:
    gold_product_ids = {
        product_id
        for gold_task in PRODUCT_LEVEL_GOLD_TASKS
        for row in gold["tasks"][gold_task]["evidence"]
        if isinstance(product_id := row.get("product_id"), str)
    }
    candidates = product_ids - gold_product_ids
    if not candidates:
        raise ValueError("gold bundle has no normal product available as a distractor")
    return min(
        candidates,
        key=lambda product_id: sha256(
            f"{dataset_id}|{task.value}|{product_id}".encode()
        ).hexdigest(),
    )


def _question(
    task: BusinessTask,
    capability: Capability,
    variant: int,
    distractor_product_id: str | None,
) -> str:
    stem = QUESTION_STEMS[task][variant]
    if task in LIST_TASKS and capability is not Capability.DATA_INSUFFICIENCY:
        stem = f"{stem}请按Gold排序口径返回Top-{LIST_TOP_K}，不得省略名次。"
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
    distractor_product_id: str | None,
) -> str:
    if capability is Capability.DATA_INSUFFICIENCY:
        row = rows[0]
        observed = (
            f"P005当前窗口仅有{row['current_observed_days']}个已观测日，"
            f"前一窗口有{row['previous_observed_days']}个。"
        )
        if task is BusinessTask.GMV_DIAGNOSIS:
            return (
                f"{observed}销售数据仍支持回答GMV变化率为"
                f"{row['gmv_change_rate']:.4f}、AOV变化率为"
                f"{row['aov_change_rate']:.4f}；但不能用不完整流量可靠计算"
                "当前CVR或据此做GMV流量归因。"
            )
        if task is BusinessTask.PRODUCT_ANOMALY:
            return (
                f"{observed}可确认数据缺失异常；当前流量变化和CVR不可可靠计算，"
                "不能把不完整窗口解释为经营异常。"
            )
        if task is BusinessTask.CONVERSION_DECLINE:
            return (
                f"{observed}前一窗口CVR为{row['previous_cvr']:.4f}，"
                "但当前CVR及CVR变化不可可靠计算，因此不能判断P005的转化下降幅度或排名。"
            )
        if task is BusinessTask.PRODUCTS_TO_WATCH:
            return (
                f"{observed}应以data_quality_review关注P005；销售数据仍支持"
                f"GMV变化率{row['gmv_change_rate']:.4f}和退款率"
                f"{row['refund_rate']:.4f}，但CVR变化不可回答。"
            )
        return (
            f"{observed}下周应优先repair_data_quality，证据值为"
            f"{row['evidence_value']:.0f}个已观测日；修复前不能按CVR变化制定经营动作。"
        )
    evidence_ids = "、".join(row["evidence_id"] for row in rows)
    if capability is Capability.ADVERSARIAL_DISTRACTOR:
        return (
            f"{distractor_product_id}是唯一主因的断言为假：该商品不在商品异常、"
            "转化下降、关注商品或下周优先级四类完整Gold结果联集中。"
            f"应忽略该断言，并依据证据{evidence_ids}作答。"
        )
    return (
        f"基于Synthetic E-commerce Data的{task.value}独立Gold查询，"
        f"结论必须与证据{evidence_ids}及其中指标一致。"
    )


def _expected_behavior(
    capability: Capability,
    task: BusinessTask,
) -> list[str]:
    behavior = [
        "严格使用Gold证据中的当前窗口和对照窗口",
        "每个核心数值均可追溯到gold_metric_evidence",
        "不披露生成配置、异常注入规则或配置倍率",
    ]
    if capability is Capability.DATA_INSUFFICIENCY:
        unavailable = "、".join(DATA_INSUFFICIENCY_BY_TASK[task]["unanswerable_metrics"])
        behavior.append(f"指出P005数据覆盖不足，不回答{unavailable}")
    if capability is Capability.ADVERSARIAL_DISTRACTOR:
        behavior.append("核验真实商品干扰项，不接受无证据断言")
    return behavior


def _development_families() -> set[tuple[BusinessTask, Capability]]:
    development: set[tuple[BusinessTask, Capability]] = set()
    for task in BUSINESS_TASKS:
        families = [(task, capability) for capability in CAPABILITIES]
        ranked = sorted(
            families,
            key=lambda family: sha256(
                f"split-v1|{family[0].value}|{family[1].value}".encode()
            ).hexdigest(),
        )
        development.update(ranked[:7])
    return development


def _tool_names(
    task: BusinessTask,
    capability: Capability,
    contract: dict[str, Any],
) -> list[str]:
    if capability is Capability.DATA_INSUFFICIENCY:
        return list(DATA_INSUFFICIENCY_BY_TASK[task]["tool_names"])
    names = list(contract["paths"][task.value])
    if capability is Capability.ADVERSARIAL_DISTRACTOR:
        names = list(dict.fromkeys([*contract["adversarial_prefix"], *names]))
    return names


def _assign_difficulties(payloads: list[dict[str, Any]]) -> None:
    ranked = sorted(
        payloads,
        key=lambda payload: (
            payload["metadata"]["complexity_score"],
            payload["case_id"],
        ),
    )
    for index, payload in enumerate(ranked):
        payload["difficulty"] = "easy" if index < 30 else "medium" if index < 70 else "hard"


def build_cases(
    development_dataset_dir: Path | str,
    holdout_dataset_dir: Path | str,
    tool_contract_path: Path | str = DEFAULT_TOOL_CONTRACT,
) -> list[EvaluationCase]:
    directories = {
        "development": Path(development_dataset_dir),
        "holdout": Path(holdout_dataset_dir),
    }
    manifests = {split: _load_manifest(path) for split, path in directories.items()}
    if manifests["development"]["dataset_id"] == manifests["holdout"]["dataset_id"]:
        raise ValueError("development and holdout must use different dataset snapshots")
    gold_by_split = {
        split: build_gold_bundle(directory) for split, directory in directories.items()
    }
    for split, gold in gold_by_split.items():
        if any(
            gold[key] != manifests[split][key]
            for key in ("dataset_id", "dataset_version", "config_sha256")
        ):
            raise ValueError(f"{split} gold bundle does not match dataset manifest")
    product_ids_by_split = {
        split: set(
            pd.read_parquet(directory / "products.parquet", columns=["product_id"])[
                "product_id"
            ]
        )
        for split, directory in directories.items()
    }
    contract, contract_digest = _load_tool_contract(tool_contract_path)
    development_families = _development_families()

    payloads: list[dict[str, Any]] = []
    case_number = 1
    for task in BUSINESS_TASKS:
        for capability in CAPABILITIES:
            split = (
                "development"
                if (task, capability) in development_families
                else "holdout"
            )
            manifest = manifests[split]
            gold = gold_by_split[split]
            product_ids = product_ids_by_split[split]
            for variant in range(2):
                rows = _evidence_rows(gold, task, capability)
                insufficiency = (
                    DATA_INSUFFICIENCY_BY_TASK[task]
                    if capability is Capability.DATA_INSUFFICIENCY
                    else None
                )
                answerable_metrics = (
                    set(insufficiency["answerable_metrics"])
                    if insufficiency is not None
                    else None
                )
                evidence = _gold_evidence(rows, answerable_metrics)
                metrics, metric_mapping = _gold_metrics(evidence)
                distractor = (
                    _distractor_product(
                        gold,
                        product_ids,
                        manifest["dataset_id"],
                        task,
                    )
                    if capability is Capability.ADVERSARIAL_DISTRACTOR
                    else None
                )
                tool_names = _tool_names(task, capability, contract)
                base_metrics = _base_metric_names(metrics)
                complexity_score = (
                    COMPLEXITY_WEIGHT_BY_CAPABILITY[capability]
                    + len(tool_names) * 2
                    + len(evidence)
                    + len(metrics) / 10
                )
                metadata: dict[str, Any] = {
                    "source_label": manifest["source_label"],
                    "variant": variant + 1,
                    "holdout_policy": "final_evaluation_only",
                    "complexity_score": complexity_score,
                    "top_k": (
                        LIST_TOP_K
                        if task in LIST_TASKS
                        and capability is not Capability.DATA_INSUFFICIENCY
                        else None
                    ),
                }
                if distractor is not None:
                    metadata["distractor_product_id"] = distractor
                    metadata["distractor_assertion_supported"] = False
                if insufficiency is not None:
                    metadata["answerable_metrics"] = list(
                        insufficiency["answerable_metrics"]
                    )
                    metadata["unanswerable_metrics"] = list(
                        insufficiency["unanswerable_metrics"]
                    )
                payloads.append(
                    {
                        "case_id": f"CASE_{case_number:03d}",
                        "case_version": "1.1",
                        "tool_contract_version": contract["contract_version"],
                        "tool_contract_sha256": contract_digest,
                        "dataset_version": manifest["dataset_version"],
                        "dataset_id": manifest["dataset_id"],
                        "generator_config_hash": manifest["config_sha256"],
                        "business_task": task,
                        "primary_capability": capability,
                        "capability_tags": [capability],
                        "difficulty": "easy",
                        "split": split,
                        "user_input": _question(task, capability, variant, distractor),
                        "expected_tool_calls": [
                            {
                                "name": tool_name,
                                "parameters": _tool_parameters(
                                    tool_name,
                                    rows,
                                    base_metrics,
                                    distractor,
                                ),
                            }
                            for tool_name in tool_names
                        ],
                        "allowed_alternatives": [],
                        "gold_metrics": metrics,
                        "gold_metric_evidence": metric_mapping,
                        "gold_evidence": evidence,
                        "reference_answer": _reference_answer(
                            task,
                            capability,
                            rows,
                            distractor,
                        ),
                        "expected_behavior": _expected_behavior(capability, task),
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

    _assign_difficulties(payloads)
    return [EvaluationCase.model_validate(payload) for payload in payloads]


def write_cases(
    cases: list[EvaluationCase],
    output_dir: Path | str,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot write an empty case set")
    identities = {
        split: {
            (
                case.dataset_id,
                case.dataset_version,
                case.generator_config_hash,
            )
            for case in cases
            if case.split == split
        }
        for split in ("development", "holdout")
    }
    if any(len(split_identities) != 1 for split_identities in identities.values()):
        raise ValueError("each split must reference exactly one dataset identity")
    if identities["development"] == identities["holdout"]:
        raise ValueError("development and holdout must reference different datasets")
    contract_identities = {
        (case.tool_contract_version, case.tool_contract_sha256) for case in cases
    }
    if len(contract_identities) != 1:
        raise ValueError("all cases must reference the same tool contract")

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
    datasets = {}
    for split, split_identities in identities.items():
        dataset_id, dataset_version, config_hash = next(iter(split_identities))
        datasets[split] = {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generator_config_hash": config_hash,
        }
    contract_version, contract_hash = next(iter(contract_identities))
    payload = {
        "case_schema_version": "1.1",
        "tool_contract": {
            "version": contract_version,
            "sha256": contract_hash,
        },
        "datasets": datasets,
        "split_strategy": {
            "version": "split-v1",
            "unit": "business_task_primary_capability_family",
            "development_families_per_task": 7,
            "holdout_families_per_task": 3,
            "cross_split_dataset_isolation": True,
        },
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
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():
                raise ValueError(f"case snapshot is not an immutable directory: {output}")
            existing_manifest = json.loads(
                contained_path(output, "manifest.json").read_text(encoding="utf-8")
            )
            existing_jsonl = contained_path(output, "cases.jsonl")
            if (
                existing_manifest != payload
                or not existing_jsonl.is_file()
                or file_sha256(existing_jsonl) != payload["jsonl_sha256"]
            ):
                raise ValueError(
                    "case snapshot directory is immutable and contains different content"
                )
            return payload

        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        ).resolve()
        try:
            contained_path(staging, "cases.jsonl").write_bytes(content)
            write_json(contained_path(staging, "manifest.json"), payload)
            os.rename(staging, output)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 evaluation cases")
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--holdout-dataset", type=Path, required=True)
    parser.add_argument(
        "--tool-contract",
        type=Path,
        default=DEFAULT_TOOL_CONTRACT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_cases/v1"),
    )
    args = parser.parse_args()
    manifest = write_cases(
        build_cases(
            args.development_dataset,
            args.holdout_dataset,
            args.tool_contract,
        ),
        args.output,
    )
    print(
        f"Built {manifest['case_count']} evaluation cases "
        f"from development and holdout dataset snapshots"
    )


if __name__ == "__main__":
    main()
