import argparse
import json
import sys
from pathlib import Path

from app.agents import AgentRunner, RunRequest
from app.data.database import open_dataset
from app.data.generator import build_snapshot
from app.llm import AdapterRequest, AssistantAction, DeterministicAdapter, ToolCall
from app.tools import build_default_registry

SALES_ARGUMENTS = {
    "start_date": "2026-04-01",
    "end_date": "2026-04-30",
    "comparison_start_date": "2026-03-02",
    "comparison_end_date": "2026-03-31",
    "product_ids": [],
    "include_refunds": True,
}


def smoke_policy(request: AdapterRequest) -> AssistantAction:
    names = [
        execution.result.tool_name for execution in request.prior_tool_executions
    ]
    if names == []:
        return AssistantAction(
            tool_calls=[
                ToolCall(
                    call_id="smoke_sales",
                    name="query_sales",
                    arguments=SALES_ARGUMENTS,
                )
            ]
        )
    if names == ["query_sales"]:
        return AssistantAction(
            tool_calls=[
                ToolCall(
                    call_id="smoke_metrics",
                    name="calculate_metrics",
                    arguments={
                        "metrics": [
                            "current_gmv",
                            "previous_gmv",
                            "gmv_change_rate",
                        ],
                        "group_by": [],
                    },
                )
            ]
        )
    if names == ["query_sales", "calculate_metrics"]:
        metrics = request.prior_tool_executions[-1].result.rows[0]
        return AssistantAction(
            final_answer=(
                "offline smoke completed: "
                f"gmv_change_rate={metrics.gmv_change_rate}"
            )
        )
    raise ValueError("unexpected smoke state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 offline smoke")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/synthetic_v1.yaml"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    try:
        snapshot = args.work_dir / "snapshot"
        build_snapshot(args.config, snapshot)
        adapter = DeterministicAdapter(smoke_policy)
        with open_dataset(snapshot) as catalog:
            result = AgentRunner(
                registry=build_default_registry(),
                adapter=adapter,
            ).run(
                RunRequest(user_input="Run the Phase 2 offline smoke."),
                catalog,
            )
            dataset_id = catalog.verified_summary["dataset_id"]
        if result.status != "completed":
            print("error: offline smoke runner did not complete", file=sys.stderr)
            return 1
        payload = {
            "status": result.status,
            "adapter": adapter.adapter_name,
            "tool_names": [
                execution.result.tool_name
                for execution in result.prior_tool_executions
            ],
            "dataset_id": dataset_id,
            "trace_event_count": len(result.decision_trace),
            "final_answer": result.final_answer,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"error: offline smoke failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
