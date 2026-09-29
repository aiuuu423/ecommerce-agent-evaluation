# Phase 2 Agent Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不实现 Baseline V1 或 Optimized V2 的前提下，建立五个受控工具、Tool Registry、Deterministic/OpenAI-compatible adapter、通用 Runner、结构化 trace、离线 smoke 与 Phase 1 冻结资产门禁。

**Architecture:** Phase 2 在已验证的 `Catalog` 之上增加窄化工具层；查询工具只接受 Pydantic 输入并返回带来源元数据的结构化结果，`calculate_metrics` 只消费先前工具结果，不能直接接触数据库。Runner 仅执行 adapter 产生的结构化动作、通过 Registry 调用工具并记录公开决策轨迹；Deterministic 与 OpenAI-compatible adapter 共用协议，前者保证无 Key 离线闭环，后者只负责 OpenAI Chat Completions 兼容请求与响应归一化。Phase 1 的 16 个冻结产物、Case Set ID、Dataset ID 与现有 `tool_contract_v1.yaml` 全部只读，任何漂移都由独立门禁阻断。

**Tech Stack:** Python 3.11+、Pydantic v2、DuckDB、Pandas、标准库 `urllib`、PyYAML、Pytest、Ruff、Make

---

## 实施边界

### 本计划包含

- 五个工具：`query_product`、`query_sales`、`query_traffic`、`query_marketing`、`calculate_metrics`。
- 五工具的严格输入/输出 Schema、稳定 JSON Schema 和统一错误语义。
- `ToolRegistry` 的注册、枚举、Schema 导出和受控调用。
- `calculate_metrics` 对已取得行集的派生计算；不允许其打开数据集或执行 SQL。
- 通用 `LLMAdapter` 协议、Deterministic adapter 和 OpenAI-compatible adapter。
- 与版本无关的 Agent Runner、公开 `decision_trace`、工具调用/result trace。
- 无 API Key、无网络、使用临时生成快照的 offline smoke。
- Phase 1 冻结资产与既有工具契约的哈希门禁。
- Phase 2 的测试、Makefile 入口、README/状态更新。

### 本计划不包含

- `Baseline V1` 的 Prompt、策略、实现或实验。
- `Optimized V2` 的 Prompt、策略、实现或实验。
- 读取 `EvaluationCase.expected_tool_calls`、Gold、异常配置来驱动运行。
- Phase 3 的批量 Evaluation Dataset 执行与 Artifact 协议。
- Phase 5 的评分、Task Success、Hallucination 或人工 Rubric。
- Phase 6–11 的错误归因、统计、Dashboard、Pipeline 和作品集。
- Responses API、多供应商专属 SDK、流式输出、并行工具调用或自动重试。

若实施中出现上述需求，停止并留给对应 Phase；不得在 Phase 2 以“顺手实现”为由扩展范围。

## 已冻结决策与术语

1. “五工具”包含四个数据查询工具和一个纯派生工具：
   `query_product`、`query_sales`、`query_traffic`、`query_marketing`、
   `calculate_metrics`。
2. `configs/evaluation/tool_contract_v1.yaml` 是 Phase 1 Case 生成时绑定的冻结评测契约；
   本阶段不修改它。`query_marketing` 可以进入运行时 Registry，但不能反写现有 Cases 或改变其
   `tool_contract_sha256`。
3. 工具不能向 adapter 暴露 `Catalog`、DuckDB connection、文件路径、Parquet 或任意 SQL。
4. `calculate_metrics` 的公开参数必须与冻结契约一致，仅含 `metrics` 与 `group_by`；
   实现从 `ToolContext.prior_executions` 读取当前 run 的全部既有 `PriorToolExecution`。模型不能传
   `result_id`、内联行集或引用其他 run 的结果。
5. trace 仅记录结构化动作、参数、结果摘要和错误，不保存或请求隐藏思维链。
6. Offline smoke 使用固定脚本动作验证基础设施，不是 V1/V2，也不读取 Case Gold。
7. OpenAI-compatible adapter 无 Key 时构造即失败为显式配置错误，模块导入、测试、lint 和
   offline smoke 不受影响。
8. 冻结契约中的 `required_parameters` 是运行时输入必填性的唯一来源；Registry 导出的每个对应
   JSON Schema，其 `required` 必须与冻结列表逐项完全相等。冻结 100 Cases 的每个
   `expected_tool_calls` 参数还必须由对应 InputModel 通过 strict JSON 验证。
9. Phase 1 干净构建会保留三个 `filelock` 同步文件；冻结门禁把它们从 16 个业务资产集合中排除，
   再单独断言同步文件路径集合恰好等于三个预期 `.lock` 文件，不能忽略任意其他额外文件。
10. Registry 以公开的 `UnknownToolError` 唯一表示名称查找失败；handler 自身抛出的 `KeyError`
    仍是工具执行失败，Runner 不得按异常基类或消息文本把它误判为未知工具。
11. Adapter 的返回对象是不可信边界。Runner 在消费前必须用
    `AdapterResponse.model_validate(response.model_dump(), strict=True)` 复验；即使自定义
    adapter 用 `model_construct()` 绕过模型 validator，也必须拒绝多个 tool calls 或
    tool call 与 final answer 同时出现。

## 精确文件结构

### 创建

```text
app/agents/__init__.py
app/agents/runner.py
app/agents/schemas.py
app/llm/__init__.py
app/llm/base.py
app/llm/deterministic.py
app/llm/openai_compatible.py
app/llm/schemas.py
app/tools/__init__.py
app/tools/base.py
app/tools/metrics.py
app/tools/queries.py
app/tools/registry.py
app/tools/schemas.py
app/experiments/__init__.py
app/experiments/phase2_smoke.py
tests/test_agent_runner.py
tests/test_llm_deterministic.py
tests/test_llm_openai_compatible.py
tests/test_phase1_frozen_assets.py
tests/test_phase2_smoke.py
tests/test_tool_metrics.py
tests/test_tool_queries.py
tests/test_tool_registry.py
```

### 修改

```text
.env.example
Makefile
PROJECT_STATUS.md
README.md
pyproject.toml
```

### 仅在缺少 `.tmp/` 规则时修改

```text
.gitignore
```

### 明确禁止修改

```text
app/data/case_generator.py
app/data/generator.py
app/data/gold.py
configs/data/synthetic_v1.yaml
configs/data/synthetic_public_validation_v1.yaml
configs/evaluation/tool_contract_v1.yaml
data/evaluation_cases/v1/cases.jsonl
data/evaluation_cases/v1/manifest.json
data/synthetic/phase1_sha256_baseline.json
sql/gold/*.sql
```

## 公共接口冻结稿

实施时先按本节命名和类型落地；后续任务不得另造同义类型。

### 工具基础类型与五工具专用输出

`app/tools/schemas.py`：

```python
from datetime import date
from enum import StrEnum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeAliasType

JsonValue = TypeAliasType(
    "JsonValue",
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"],
)
ProductId = Annotated[str, Field(pattern=r"^P[0-9]{3}$")]


class ToolModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
    )


class Period(ToolModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def valid_order(self) -> "Period":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class QueryProductInput(ToolModel):
    product_ids: list[ProductId] = Field(min_length=1)


class QuerySalesInput(ToolModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date
    product_ids: list[ProductId]
    include_refunds: bool


class QueryTrafficInput(ToolModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date
    product_ids: list[ProductId]
    include_missing: bool


class QueryMarketingInput(ToolModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date
    product_ids: list[ProductId] = Field(default_factory=list)


class MetricName(StrEnum):
    CURRENT_GMV = "current_gmv"
    PREVIOUS_GMV = "previous_gmv"
    GMV_CHANGE_RATE = "gmv_change_rate"
    CURRENT_ORDERS = "current_orders"
    PREVIOUS_ORDERS = "previous_orders"
    CURRENT_AOV = "current_aov"
    PREVIOUS_AOV = "previous_aov"
    AOV_CHANGE_RATE = "aov_change_rate"
    CURRENT_IMPRESSIONS = "current_impressions"
    PREVIOUS_IMPRESSIONS = "previous_impressions"
    CURRENT_CLICKS = "current_clicks"
    PREVIOUS_CLICKS = "previous_clicks"
    CURRENT_VISITS = "current_visits"
    PREVIOUS_VISITS = "previous_visits"
    CURRENT_CVR = "current_cvr"
    PREVIOUS_CVR = "previous_cvr"
    CVR_CHANGE = "cvr_change"
    CVR_CHANGE_RATE = "cvr_change_rate"
    CURRENT_CTR = "current_ctr"
    PREVIOUS_CTR = "previous_ctr"
    TRAFFIC_CHANGE_RATE = "traffic_change_rate"
    CURRENT_OBSERVED_DAYS = "current_observed_days"
    PREVIOUS_OBSERVED_DAYS = "previous_observed_days"
    CURRENT_SPEND = "current_spend"
    PREVIOUS_SPEND = "previous_spend"
    CURRENT_ROAS = "current_roas"
    PREVIOUS_ROAS = "previous_roas"
    CURRENT_REFUND_RATE = "current_refund_rate"
    REFUND_RATE = "refund_rate"
    EVIDENCE_VALUE = "evidence_value"


class GroupBy(StrEnum):
    PRODUCT_ID = "product_id"
    CATEGORY = "category"
    REGION = "region"
    CHANNEL = "channel"


class CalculateMetricsInput(ToolModel):
    metrics: list[MetricName] = Field(min_length=1)
    group_by: list[GroupBy]


RowT = TypeVar("RowT", bound=ToolModel)


class ToolResult(ToolModel, Generic[RowT]):
    result_id: str = Field(pattern=r"^result_[0-9]{4}$")
    tool_name: str
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    source_label: Literal["Synthetic E-commerce Data"]
    columns: list[str]
    rows: list[RowT]
    row_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class QueryProductRow(ToolModel):
    product_id: ProductId
    product_name: str
    category: str
    price: float
    cost: float
    launch_date: date


class QuerySalesRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    region: str
    channel: str
    gmv: float
    orders: int = Field(ge=0)
    units: int = Field(ge=0)
    refund_orders: int = Field(ge=0)


class QueryTrafficRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    visits: int = Field(ge=0)
    observed_days: int = Field(ge=0)
    missing_days: int = Field(ge=0)


class QueryMarketingRow(ToolModel):
    period: Literal["current", "previous"]
    product_id: ProductId
    category: str
    campaign_id: str
    spend: float = Field(ge=0)


class CalculateMetricsRow(ToolModel):
    # 仅允许四个维度和 MetricName 中列出的指标；未请求字段保持 unset，
    # 序列化时使用 exclude_unset=True，结果仍为 group_by + metrics 的扁平行。
    product_id: ProductId | None = None
    category: str | None = None
    region: str | None = None
    channel: str | None = None
    current_gmv: float | None = None
    previous_gmv: float | None = None
    gmv_change_rate: float | None = None
    current_orders: int | None = Field(default=None, ge=0)
    previous_orders: int | None = Field(default=None, ge=0)
    current_aov: float | None = None
    previous_aov: float | None = None
    aov_change_rate: float | None = None
    current_impressions: int | None = Field(default=None, ge=0)
    previous_impressions: int | None = Field(default=None, ge=0)
    current_clicks: int | None = Field(default=None, ge=0)
    previous_clicks: int | None = Field(default=None, ge=0)
    current_visits: int | None = Field(default=None, ge=0)
    previous_visits: int | None = Field(default=None, ge=0)
    current_cvr: float | None = None
    previous_cvr: float | None = None
    cvr_change: float | None = None
    cvr_change_rate: float | None = None
    current_ctr: float | None = None
    previous_ctr: float | None = None
    traffic_change_rate: float | None = None
    current_observed_days: int | None = Field(default=None, ge=0)
    previous_observed_days: int | None = Field(default=None, ge=0)
    current_spend: float | None = Field(default=None, ge=0)
    previous_spend: float | None = Field(default=None, ge=0)
    current_roas: float | None = None
    previous_roas: float | None = None
    current_refund_rate: float | None = None
    refund_rate: float | None = None
    evidence_value: float | None = None


class QueryProductResult(ToolResult[QueryProductRow]):
    tool_name: Literal["query_product"]


class QuerySalesResult(ToolResult[QuerySalesRow]):
    tool_name: Literal["query_sales"]


class QueryTrafficResult(ToolResult[QueryTrafficRow]):
    tool_name: Literal["query_traffic"]


class QueryMarketingResult(ToolResult[QueryMarketingRow]):
    tool_name: Literal["query_marketing"]


class CalculateMetricsResult(ToolResult[CalculateMetricsRow]):
    tool_name: Literal["calculate_metrics"]


AnyToolResult = (
    QueryProductResult
    | QuerySalesResult
    | QueryTrafficResult
    | QueryMarketingResult
    | CalculateMetricsResult
)


def canonical_tool_result_payload(result: AnyToolResult) -> dict[str, JsonValue]:
    row_payloads = [
        row.model_dump(mode="json", exclude_unset=True)
        for row in result.rows
    ]
    return {
        "result_id": result.result_id,
        "tool_name": result.tool_name,
        "dataset_id": result.dataset_id,
        "source_label": result.source_label,
        "columns": list(result.columns),
        "rows": [
            {column: row[column] for column in result.columns}
            for row in row_payloads
        ],
        "row_count": result.row_count,
        "warnings": list(result.warnings),
    }


class PriorToolExecution(ToolModel):
    call_id: str = Field(min_length=1)
    arguments: dict[str, JsonValue]
    result: AnyToolResult
```

所有查询 Input 再用共享 validator 强制两个窗口各自有序、`product_ids` 唯一；输出行按稳定键排序。
`QuerySalesInput.product_ids`、`QueryTrafficInput.product_ids` 与
`CalculateMetricsInput.group_by` 必须显式传入，即使调用者要表达“全部商品”或“不分组”也必须
分别传 `[]`；不得用 `default_factory`、默认空列表或其他隐式补值绕过冻结契约的必填语义。
五个 Result 的 validator 除检查 `row_count == len(rows)`、`columns` 唯一外，还必须检查
`set(columns)` 与每一行 `model_dump(exclude_unset=True)` 的键集合完全相等；不得比较键列表顺序，
因为 `model_dump()` 的字段顺序来自 Pydantic 模型声明，而不是本次请求。四个查询 Result 另行检查
各自固定的 `columns` 顺序；
`CalculateMetricsResult` 额外检查列恰为本次 `group_by + metrics`（由 handler 构造时显式传入
同一顺序后验证），并通过 `canonical_tool_result_payload()` 按 `columns` 显式投影每一行，不得
依赖 `CalculateMetricsRow` 的声明顺序，也不得用无约束 `dict[str, JsonValue]` 代替专用行模型。
规范 JSON 的顶层键顺序固定为 `result_id`、`tool_name`、`dataset_id`、`source_label`、`columns`、
`rows`、`row_count`、`warnings`；每个 row 的键顺序严格跟随 `columns`。序列化统一使用该 payload
和 `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`，不使用 `sort_keys=True`
重排业务列。所有浮点字段拒绝 NaN/Infinity。
空结果是成功的 `ToolResult(row_count=0)`，不是异常。Registry 必须把输入
`ValidationError` 包装为 `ToolInputValidationError`，把 handler 返回后的任何输出边界校验失败
包装为 `ToolOutputValidationError`；未知工具抛 `UnknownToolError`，重复注册抛 `ValueError`。
Runner 只捕获 `UnknownToolError` 并归一化为 `unknown_tool`；handler 自身抛出的 `KeyError`
必须落入 `tool_execution_error`。输入、输出和执行错误写入 trace 后终止，不伪造空结果。

### Registry 与 Tool 协议

`app/tools/base.py` 与 `app/tools/registry.py`：

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.tools.schemas import AnyToolResult, PriorToolExecution

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class UnknownToolError(LookupError):
    """Registry could not resolve a requested tool name."""


class ToolInputValidationError(ValueError):
    """Registry input boundary rejected tool arguments."""


class ToolOutputValidationError(RuntimeError):
    """Registry output boundary rejected a handler result."""


@dataclass(frozen=True)
class ToolDefinition(Generic[InputT, OutputT]):
    name: str
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]
    handler: Callable[[InputT, "ToolContext"], OutputT]
    output_validator: Callable[[InputT, OutputT], None] | None = None


@dataclass(frozen=True)
class ToolContext:
    catalog: "Catalog"
    prior_executions: tuple[PriorToolExecution, ...]
    next_result_id: Callable[[], str]


class ToolRegistry:
    def register(self, definition: ToolDefinition[Any, Any]) -> None:
        raise NotImplementedError

    def get(self, name: str) -> ToolDefinition[Any, Any]:
        raise NotImplementedError

    def names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def openai_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> AnyToolResult:
        raise NotImplementedError


def build_default_registry() -> ToolRegistry:
    raise NotImplementedError
```

`openai_tools()` 只导出 `{"type":"function","function":{name,description,parameters}}`，
其中 `parameters` 来自 `input_model.model_json_schema()`；按工具名排序并以
`json.dumps(..., sort_keys=True, separators=(",", ":"))` 生成 canonical bytes，确保 Schema
摘要稳定。每个定义必须绑定自己唯一的 `output_model`。`invoke()` 在输入
`model_validate()` 后调用 handler，并对 handler 返回值的 Python dump 使用
`output_model.model_validate(..., strict=True)` 做第二次边界验证；随后再验证
`tool_name == name` 并调用可选的 `output_validator(parsed, validated)`，不得相信 handler
已构造的模型或只做名称检查。输入 `model_validate()` 的 `ValidationError` 必须以原异常为
`__cause__` 包装成 `ToolInputValidationError`；handler 调用保持在输出校验 `try` 块之外，handler
自身异常不得伪装成输出错误；handler 返回后，从 Python dump、严格 `output_model` 复验、
`tool_name` 检查到 `output_validator` 的失败统一以原异常为 `__cause__` 包装成
`ToolOutputValidationError`。`calculate_metrics` 的定义必须绑定 output validator，逐项验证
`columns == [*group_by, *metrics]`，其他四个查询的固定列由各自 Result model validator 验证。
`get()` 对未知名称抛 `UnknownToolError`，`invoke()` 必须通过 `get()` 取得定义，因此未知名称
同样只抛该公开错误；不得捕获或改写 handler 自身的 `KeyError`。`build_default_registry()`
必须恰好注册上述五工具。

### Adapter 与 Runner 协议

`app/llm/schemas.py`、`app/llm/base.py`、`app/agents/schemas.py`：

```python
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.schemas import JsonValue, PriorToolExecution


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(AdapterModel):
    call_id: str
    name: str
    arguments: dict[str, JsonValue]


class AssistantAction(AdapterModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None


class AdapterRequest(AdapterModel):
    user_input: str
    tools: list[dict[str, JsonValue]]
    prior_tool_executions: list[PriorToolExecution]


class Usage(AdapterModel):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def valid_total(self) -> "Usage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class AdapterResponse(AdapterModel):
    action: AssistantAction
    raw_response: dict[str, JsonValue] | None = None
    usage: Usage | None = None


class LLMAdapter(Protocol):
    @property
    def adapter_name(self) -> str:
        raise NotImplementedError

    def start_run(self) -> None:
        raise NotImplementedError

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedError


class TraceEventType(StrEnum):
    ADAPTER_REQUEST = "adapter_request"
    ADAPTER_RESPONSE = "adapter_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class TraceEvent(BaseModel):
    sequence: int = Field(ge=1)
    event_type: TraceEventType
    payload: dict[str, JsonValue]


class RunRequest(BaseModel):
    user_input: str = Field(min_length=1)
    max_steps: int = Field(default=8, ge=1, le=32)


class RunResult(BaseModel):
    status: str
    final_answer: str | None
    prior_tool_executions: list[PriorToolExecution]
    decision_trace: list[TraceEvent]
    usage: Usage | None


class AgentRunner:
    def __init__(self, registry: ToolRegistry, adapter: LLMAdapter) -> None:
        raise NotImplementedError

    def run(self, request: RunRequest, catalog: Catalog) -> RunResult:
        raise NotImplementedError
```

Runner 不接受 `EvaluationCase`，不接受 Gold，也没有 `agent_version`、V1 或 V2 分支。每轮至多接受
一个 tool call；adapter 同时返回 tool call 与 final answer、返回多个 tool calls、重复
`call_id`、超过 `max_steps` 都产生 `status="failed"` 和末尾 `error` trace。
Runner 不能信任 adapter 已返回合法的 Pydantic 实例：每次 `complete()` 返回后、写入 trace 或
读取 action 前，必须执行
`AdapterResponse.model_validate(response.model_dump(), strict=True)`；复验失败统一归一化为
`adapter_error`，且不得执行其中任何工具调用。
`PriorToolExecution` 是 Runner、adapter transcript 与派生工具之间唯一的既往执行记录，必须同时
保留原始结构化 `arguments` 和经 Registry 二次验证后的 `result`；不得只传裸
`ToolResult`，也不得通过结果顺序猜测 provider `call_id`。

## 原子实施任务

### Task 1：建立 Phase 1 冻结资产门禁

**Files:**
- Create: `tests/test_phase1_frozen_assets.py`
- Read only: `data/synthetic/phase1_sha256_baseline.json`
- Read only: `configs/evaluation/tool_contract_v1.yaml`
- Read only: `data/evaluation_cases/v1/cases.jsonl`
- Read only: `data/evaluation_cases/v1/manifest.json`

- [ ] **Step 1：先写失败测试，锁定 16 个资产、身份和工具契约**

```python
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "data/synthetic/phase1_sha256_baseline.json"
CASES = ROOT / "data/evaluation_cases/v1/cases.jsonl"
MANIFEST = ROOT / "data/evaluation_cases/v1/manifest.json"
CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"

FROZEN_BASELINE_SHA256 = "84140e698e2c006e1f3b71018cfd893119038940a395ae51c69f7613030b309f"
FROZEN_CASES_SHA256 = "f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03"
FROZEN_MANIFEST_SHA256 = "9396ce3a86175874d9846bc5c34490e73cdb9cee612c487900c4e1b6c1eeb489"
FROZEN_CONTRACT_SHA256 = "471e0f09d2a609d3f8799565020cc635f201dd2e95fb33e1a170c256d9e944cb"
FROZEN_CASE_SET_ID = "ecebfe8b691271fd"
FROZEN_DATASET_IDS = {
    "development": "e1e81533c25e03e5",
    "public_validation": "c17d4926cfa7cb26",
}
EXPECTED_LOCK_PATHS = {
    "evaluation_cases/.v1.lock",
    "synthetic/.public-validation-v1.lock",
    "synthetic/.v1.lock",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def clean_phase1_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("phase1-frozen")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.data.phase1",
            "--development-config",
            str(ROOT / "configs/data/synthetic_v1.yaml"),
            "--public-validation-config",
            str(ROOT / "configs/data/synthetic_public_validation_v1.yaml"),
            "--tool-contract",
            str(ROOT / "configs/evaluation/tool_contract_v1.yaml"),
            "--output-root",
            str(output_root),
            "--stage",
            "all",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_root


def test_committed_freeze_files_match_literal_hashes() -> None:
    assert _sha256(BASELINE) == FROZEN_BASELINE_SHA256
    assert _sha256(CASES) == FROZEN_CASES_SHA256
    assert _sha256(MANIFEST) == FROZEN_MANIFEST_SHA256
    assert _sha256(CONTRACT) == FROZEN_CONTRACT_SHA256


def test_phase1_clean_build_matches_frozen_baseline(
    clean_phase1_build: Path,
) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["artifact_count"] == 16
    assert baseline["dataset_ids"] == FROZEN_DATASET_IDS
    assert baseline["case_set_id"] == FROZEN_CASE_SET_ID
    expected_paths = {artifact["path"] for artifact in baseline["artifacts"]}
    actual_file_paths = {
        path.relative_to(clean_phase1_build).as_posix()
        for path in clean_phase1_build.rglob("*")
        if path.is_file()
    }
    actual_lock_paths = {
        path for path in actual_file_paths if Path(path).suffix == ".lock"
    }
    actual_asset_paths = actual_file_paths - actual_lock_paths
    assert len(expected_paths) == 16
    assert actual_lock_paths == EXPECTED_LOCK_PATHS
    assert actual_asset_paths == expected_paths
    for artifact in baseline["artifacts"]:
        path = clean_phase1_build / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]


def test_phase1_contract_cases_and_manifest_remain_bound() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert contract["contract_version"] == "1.0"
    assert manifest["case_set_id"] == FROZEN_CASE_SET_ID
    assert manifest["jsonl_sha256"] == FROZEN_CASES_SHA256
    assert manifest["tool_contract"] == {
        "version": "1.0",
        "sha256": FROZEN_CONTRACT_SHA256,
    }
    assert {
        split: item["dataset_id"]
        for split, item in manifest["datasets"].items()
    } == FROZEN_DATASET_IDS
    for line in CASES.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        assert case["tool_contract_version"] == "1.0"
        assert case["tool_contract_sha256"] == FROZEN_CONTRACT_SHA256
        assert case["dataset_id"] == FROZEN_DATASET_IDS[case["split"]]
```

这里不能直接要求 `data/synthetic/v1` 与 `data/synthetic/public-validation-v1` 存在：两套 Parquet
是可重建且 gitignored 的运行产物。门禁必须在 pytest 临时目录执行干净 Phase 1 构建，先把
实际生成的全部普通文件拆成同步锁与业务资产两个集合：同步锁路径必须严格等于
`evaluation_cases/.v1.lock`、`synthetic/.v1.lock`、
`synthetic/.public-validation-v1.lock` 三项；排除这三项后，业务资产路径集合必须与 baseline
中恰好 16 个路径严格相等（额外或缺失任一路径均失败），再逐项比较 16 项 SHA-256。不得笼统
忽略所有点文件或所有 `.lock`，否则意外同步文件无法被门禁发现；同时先用代码中的固定常量直接
校验 baseline 文件、Cases、Manifest 和工具契约本身，禁止通过同时改文件与其自描述哈希绕过门禁。

- [ ] **Step 2：运行 RED，确认当前工作树缺少独立门禁测试**

Run: `python3.12 -m pytest tests/test_phase1_frozen_assets.py -q`

Expected: FAIL，原因是测试文件刚引入时应先用错误期望
`artifact_count == 15` 验证测试确实执行；失败信息包含 `assert 16 == 15`。

- [ ] **Step 3：改回上方正确断言，形成最小 GREEN**

不得更新 baseline、重新生成资产或“接受新哈希”；只把测试期望改为已提交基线中的真实值。

- [ ] **Step 4：运行 GREEN**

Run: `python3.12 -m pytest tests/test_phase1_frozen_assets.py -q`

Expected: `3 passed`。

- [ ] **Step 5：提交冻结门禁**

```bash
git add tests/test_phase1_frozen_assets.py
git commit -m "test(phase2): guard frozen phase 1 assets"
```

### Task 2：定义五工具的严格 Schema

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/schemas.py`
- Create: `tests/test_tool_registry.py`
- Read only: `data/evaluation_cases/v1/cases.jsonl`

- [ ] **Step 1：写 Schema RED 测试**

```python
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.tools.schemas import (
    CalculateMetricsInput,
    CalculateMetricsResult,
    MetricName,
    QueryMarketingInput,
    QueryProductInput,
    QuerySalesInput,
    QueryTrafficInput,
    QuerySalesResult,
    canonical_tool_result_payload,
)

ROOT = Path(__file__).parents[1]
CASES = ROOT / "data/evaluation_cases/v1/cases.jsonl"
CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"
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


def test_metric_name_covers_exact_frozen_case_metric_vocabulary() -> None:
    case_metrics = {
        metric
        for line in CASES.read_text(encoding="utf-8").splitlines()
        for call in json.loads(line)["expected_tool_calls"]
        if call["name"] == "calculate_metrics"
        for metric in call["parameters"]["metrics"]
    }
    assert case_metrics == FROZEN_CASE_METRICS
    assert case_metrics <= {metric.value for metric in MetricName}


def test_required_empty_collections_cannot_be_omitted() -> None:
    sales = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "include_refunds": True,
    }
    traffic = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "include_missing": True,
    }
    with pytest.raises(ValidationError, match="product_ids"):
        QuerySalesInput.model_validate(sales)
    with pytest.raises(ValidationError, match="product_ids"):
        QueryTrafficInput.model_validate(traffic)
    with pytest.raises(ValidationError, match="group_by"):
        CalculateMetricsInput.model_validate({"metrics": ["current_gmv"]})


def test_all_100_frozen_case_calls_pass_corresponding_strict_input_model() -> None:
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
    assert len(cases) == 100
    for case in cases:
        for call in case["expected_tool_calls"]:
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
    values = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
        **flag,
        **overrides,
    }
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
def test_all_query_inputs_reject_duplicate_product_ids(model, flag) -> None:
    with pytest.raises(ValidationError, match="unique"):
        model.model_validate({
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "comparison_start_date": "2026-03-02",
            "comparison_end_date": "2026-03-31",
            "product_ids": ["P001", "P001"],
            **flag,
        })


def test_query_product_rejects_duplicate_product_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        QueryProductInput.model_validate({"product_ids": ["P001", "P001"]})


def test_metric_input_matches_frozen_contract_and_rejects_inline_rows() -> None:
    parsed = CalculateMetricsInput.model_validate({
        "metrics": ["current_gmv"],
        "group_by": [],
    })
    assert parsed.metrics == ["current_gmv"]
    with pytest.raises(ValidationError, match="extra"):
        CalculateMetricsInput.model_validate({
            "metrics": ["current_gmv"],
            "group_by": [],
            "rows": [{"revenue": 1}],
        })


def test_dedicated_tool_result_rejects_non_finite_or_wrong_row_shape() -> None:
    with pytest.raises(ValidationError):
        QuerySalesResult.model_validate({
            "result_id": "result_0001",
            "tool_name": "query_sales",
            "dataset_id": "e1e81533c25e03e5",
            "source_label": "Synthetic E-commerce Data",
            "columns": [
                "period", "product_id", "category", "region", "channel",
                "gmv", "orders", "units", "refund_orders",
            ],
            "rows": [{
                "period": "current",
                "product_id": "P001",
                "category": "A",
                "region": "East",
                "channel": "organic",
                "gmv": float("nan"),
                "orders": 1,
                "units": 1,
                "refund_orders": 0,
            }],
            "row_count": 1,
        })


def test_calculate_metrics_columns_define_validation_and_json_order() -> None:
    result = CalculateMetricsResult.model_validate({
        "result_id": "result_0001",
        "tool_name": "calculate_metrics",
        "dataset_id": "e1e81533c25e03e5",
        "source_label": "Synthetic E-commerce Data",
        # 故意不同于 CalculateMetricsRow 的字段声明顺序。
        "columns": ["product_id", "previous_orders", "current_gmv"],
        "rows": [{
            "product_id": "P001",
            "current_gmv": 120.0,
            "previous_orders": 2,
        }],
        "row_count": 1,
    })
    payload = canonical_tool_result_payload(result)
    assert set(payload["rows"][0]) == set(result.columns)
    assert list(payload["rows"][0]) == result.columns
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_registry.py -q`

Expected: collection FAIL with `ModuleNotFoundError: No module named 'app.tools'`。

- [ ] **Step 3：实现公共接口冻结稿中的 Schema**

除上文代码外，增加：

```python
@model_validator(mode="after")
def validate_result(self) -> "ToolResult[RowT]":
    if self.row_count != len(self.rows):
        raise ValueError("row_count must equal len(rows)")
    if len(self.columns) != len(set(self.columns)):
        raise ValueError("columns must be unique")
    for row in self.rows:
        row_keys = set(row.model_dump(exclude_unset=True))
        if row_keys != set(self.columns):
            raise ValueError("every row must match columns exactly")
    return self
```

`validate_result()` 只验证列集合；四个查询的固定列序和 `calculate_metrics` 的
`[*group_by, *metrics]` 列序分别由专用 validator 锁定。任何对外 JSON（尤其 provider tool
message）都调用 `canonical_tool_result_payload()` 显式投影 rows，从而把验证正确性与 Pydantic
字段声明顺序解耦。

`MetricName` 必须覆盖上述从冻结 Cases 的 `expected_tool_calls[].parameters.metrics` 提取出的
19 个指标；测试同时用字面量集合锁定当前 Cases 词汇，并验证该集合是 Enum 的子集。特别注意
`cvr_change` 是 CVR 的绝对差（百分点以 0–1 小数表示），不能与相对变化率
`cvr_change_rate` 合并或改名。
冻结 Case 兼容性测试必须读取并遍历恰好 100 个 Cases 的全部 `expected_tool_calls`，按工具名映射
到对应 InputModel，并使用 `model_validate_json(..., strict=True)`；这里先转回 JSON 是为了在
strict 模式下保留 JSON 日期字符串的合法语义，同时仍拒绝 bool/number/string 等错误类型。不得
只抽样 Case、只验证字段存在，或用非 strict `model_validate()` 掩盖类型漂移。

复用 `app.data.schemas._validate_json_value` 前先将其提升为公开函数
`validate_json_value` 会触碰 Phase 1 模块且无必要；本阶段在工具模块内实现同等严格的独立
JSON 值检查，避免修改既有数据契约。

- [ ] **Step 4：运行 GREEN 与 lint**

Run: `python3.12 -m pytest tests/test_tool_registry.py -q`

Expected: 参数化用例全部 PASS。

Run: `python3.12 -m ruff check app/tools tests/test_tool_registry.py`

Expected: `All checks passed!`。

- [ ] **Step 5：提交 Schema**

```bash
git add app/tools/__init__.py app/tools/schemas.py tests/test_tool_registry.py
git commit -m "feat(tools): define strict phase 2 schemas"
```

### Task 3：实现 Registry、稳定 Schema 导出和受控调用

**Files:**
- Create: `app/tools/base.py`
- Create: `app/tools/registry.py`
- Modify: `app/tools/__init__.py`
- Modify: `tests/test_tool_registry.py`

- [ ] **Step 1：追加 Registry RED 测试**

```python
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
    assert all(item["function"]["parameters"]["additionalProperties"] is False for item in exported)
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


def test_registry_rejects_unknown_tools_and_validates_before_calling_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_default_registry()
    with pytest.raises(UnknownToolError, match="unknown tool"):
        registry.get("unknown")
    with pytest.raises(UnknownToolError, match="unknown tool"):
        registry.invoke("unknown", {}, context=object())
    with pytest.raises(ToolInputValidationError) as exc_info:
        registry.invoke(
            "query_sales",
            {"start_date": "not-a-date"},
            context=object(),
        )
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_registry_revalidates_handler_output_with_bound_model() -> None:
    registry = ToolRegistry()
    registry.register(malformed_query_sales_definition())
    with pytest.raises(ToolOutputValidationError) as exc_info:
        registry.invoke("query_sales", valid_sales_arguments(), valid_context())
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_openai_tool_schema_digest_is_frozen() -> None:
    canonical = json.dumps(
        build_default_registry().openai_tools(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_TOOL_SCHEMA_SHA256
```

`malformed_query_sales_definition()` 绑定 `output_model=QuerySalesResult`，但测试 handler 使用
`model_construct()` 返回缺列或错误字段类型的结果，以证明 Registry 的第二次严格验证确实执行。
`EXPECTED_TOOL_SCHEMA_SHA256` 必须是实现完成后对五工具 canonical Schema 做一次人工审阅后写入
测试的单一字面量常量；GREEN 后不得从当前输出动态计算或通过自动更新 snapshot 接受漂移。
此外必须直接读取只读的 `tool_contract_v1.yaml`，对其中四个冻结工具逐一断言
`parameters.required == required_parameters`，包括列表内容与顺序，不得只做集合包含、字段数或
digest 间接比较。`query_marketing` 不在冻结契约中，因此仍由五工具 Schema digest 门禁覆盖，
不得伪造其冻结 `required_parameters`。

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_registry.py -q`

Expected: FAIL with import errors for `ToolRegistry`/`build_default_registry`。

- [ ] **Step 3：实现 Registry 最小核心**

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition[Any, Any]] = {}

    def register(self, definition: ToolDefinition[Any, Any]) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition[Any, Any]:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_model.model_json_schema(),
                },
            }
            for definition in (self._definitions[name] for name in self.names())
        ]

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> AnyToolResult:
        definition = self.get(name)
        try:
            parsed = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputValidationError(f"invalid arguments for {name}") from exc

        result = definition.handler(parsed, context)
        try:
            validated = definition.output_model.model_validate(
                result.model_dump(mode="python", exclude_unset=True),
                strict=True,
            )
            if validated.tool_name != name:
                raise ValueError("handler returned mismatched tool_name")
            if definition.output_validator is not None:
                definition.output_validator(parsed, validated)
            return validated
        except Exception as exc:
            raise ToolOutputValidationError(f"invalid result from {name}") from exc
```

此步先给五工具注册会抛 `NotImplementedError` 的私有 handler，使 Schema/Registry 测试通过；
每个定义此时也必须绑定对应的五个专用 Result model，后续 Task 4–8 逐个替换 handler。
Registry 不暴露“执行任意 SQL”或动态模块导入接口。`ToolInputValidationError` 与
`ToolOutputValidationError`、`UnknownToolError` 定义在 `app/tools/base.py` 并由
`app/tools/__init__.py` 稳定导出。上方同一未知工具测试必须同时覆盖 `get()` 与 `invoke()`，
证明两条入口都只抛 `UnknownToolError`；测试还要用一个直接抛 `KeyError("handler-miss")` 的
handler 证明 handler 异常不会被 Registry 包装，也不会与名称查找失败混淆。

- [ ] **Step 4：运行 GREEN**

Run: `python3.12 -m pytest tests/test_tool_registry.py -q`

Expected: 全部 PASS。

- [ ] **Step 5：提交 Registry**

```bash
git add app/tools/__init__.py app/tools/base.py app/tools/registry.py tests/test_tool_registry.py
git commit -m "feat(tools): add controlled tool registry"
```

### Task 4：实现 `query_product`

**Files:**
- Create: `app/tools/queries.py`
- Create: `tests/test_tool_queries.py`
- Modify: `app/tools/registry.py`

- [ ] **Step 1：写 `query_product` RED 测试**

测试 fixture 使用 `build_snapshot()` 在 `tmp_path` 生成数据，再用 `open_dataset()` 打开；禁止直接依赖
gitignored 的本地快照。

```python
def test_query_product_returns_only_requested_products(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_product",
        {"product_ids": ["P002", "P001"]},
        tool_context,
    )
    assert result.tool_name == "query_product"
    assert result.dataset_id == catalog.verified_summary["dataset_id"]
    assert result.columns == [
        "product_id", "product_name", "category", "price", "cost", "launch_date"
    ]
    assert [row.product_id for row in result.rows] == ["P001", "P002"]
    assert result.row_count == 2
    assert all(
        "anomaly" not in key
        for row in result.rows
        for key in row.model_dump().keys()
    )


def test_query_product_empty_match_is_not_an_error(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_product",
        {"product_ids": ["P999"]},
        tool_context,
    )
    assert result.rows == []
    assert result.warnings == ["no rows matched the request"]
```

`P999` 符合格式但不存在，用于验证空集；重复 ID 必须在 Schema 阶段失败。

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_queries.py -q`

Expected: FAIL，`query_product` handler 仍抛 `NotImplementedError`。

- [ ] **Step 3：实现参数化 SQL 与统一序列化**

```python
def query_product(args: QueryProductInput, context: ToolContext) -> QueryProductResult:
    placeholders = ", ".join("?" for _ in args.product_ids)
    frame = context.catalog.execute(
        f"""
        select product_id, product_name, category, price, cost, launch_date
        from products
        where product_id in ({placeholders})
        order by product_id
        """,
        list(args.product_ids),
    ).fetch_df()
    return _frame_result("query_product", frame, context)
```

`_frame_result()` 必须：

- 从 `catalog.verified_summary` 取得 `dataset_id` 与 `source_label`；
- 把 `date/datetime` 转 ISO 字符串、有限 `Decimal` 转 float、Pandas/NumPy scalar 转 Python
  scalar、NA/NaN 转 `None`；
- 用 `context.next_result_id()` 分配 ID；
- 不输出 SQL、文件路径、manifest 全文、生成配置或异常标识；
- 空表保留固定 columns 并增加 `no rows matched the request` warning。

- [ ] **Step 4：运行 GREEN**

Run: `python3.12 -m pytest tests/test_tool_queries.py -q`

Expected: `2 passed`。

- [ ] **Step 5：提交产品查询工具**

```bash
git add app/tools/queries.py app/tools/registry.py tests/test_tool_queries.py
git commit -m "feat(tools): add controlled product query"
```

### Task 5：实现 `query_sales`

**Files:**
- Modify: `app/tools/queries.py`
- Modify: `app/tools/registry.py`
- Modify: `tests/test_tool_queries.py`

`query_sales` 输出固定粒度为
`period × product_id × category × region × channel`，固定列为：

```text
period, product_id, category, region, channel,
gmv, orders, units, refund_orders
```

其中 `period ∈ {"current", "previous"}`；`gmv=sum(revenue)`、`orders=count(distinct
order_id)` 与 Phase 1 Gold 口径一致。`include_refunds=false` 时在聚合前过滤
`is_refund=false`；`true` 时包含全部订单并同时输出 `refund_orders`。不得把退款金额擅自改成负数。

- [ ] **Step 1：写销售口径 RED 测试**

```python
def test_query_sales_matches_independent_catalog_aggregate(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    arguments = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": [],
        "include_refunds": True,
    }
    result = registry.invoke("query_sales", arguments, tool_context)
    current = sum(row.gmv for row in result.rows if row.period == "current")
    expected = catalog.execute(
        "select sum(revenue) from orders where order_date between ? and ?",
        ["2026-04-01", "2026-04-30"],
    ).fetchone()
    assert expected is not None
    assert current == pytest.approx(float(expected[0]))
    assert result.rows == sorted(
        result.rows,
        key=lambda row: (
            row.period, row.product_id, row.region, row.channel
        ),
    )


def test_query_sales_product_filter_and_refund_switch(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    base = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P004"],
    }
    included = registry.invoke(
        "query_sales", {**base, "include_refunds": True}, tool_context
    )
    excluded = registry.invoke(
        "query_sales", {**base, "include_refunds": False}, tool_context
    )
    assert {row.product_id for row in included.rows} == {"P004"}
    assert sum(row.orders for row in excluded.rows) <= sum(
        row.orders for row in included.rows
    )
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_queries.py -k sales -q`

Expected: FAIL，`query_sales` handler 未实现。

- [ ] **Step 3：实现销售查询**

关键 SQL 结构必须是参数化窗口 CTE，不拼接用户值：

```sql
with periods(period, period_start, period_end) as (
  values
    ('current', cast(? as date), cast(? as date)),
    ('previous', cast(? as date), cast(? as date))
)
select
  periods.period,
  orders.product_id,
  products.category,
  customers.region,
  customers.channel,
  sum(orders.revenue) as gmv,
  count(distinct orders.order_id) as orders,
  sum(orders.quantity) as units,
  count(distinct orders.order_id) filter (where orders.is_refund) as refund_orders
from periods
join orders on orders.order_date between period_start and period_end
join products using (product_id)
join customers using (customer_id)
where (? or not orders.is_refund)
  and (len(?) = 0 or orders.product_id in (...))
group by all
order by period, product_id, region, channel
```

DuckDB 不直接绑定 `len(?)` 或列表展开；Python 根据是否有 `product_ids` 仅选择两段固定 SQL
模板，并为 ID 创建 `?, ...` placeholders。模板选择是结构控制，不是用户 SQL 拼接。

- [ ] **Step 4：运行 GREEN 和 Gold 交叉检查**

Run: `python3.12 -m pytest tests/test_tool_queries.py -k sales -q`

Expected: 销售测试全部 PASS。

Run: `python3.12 -m pytest tests/test_gold.py -q`

Expected: 既有 Gold 测试全部 PASS，证明未修改 Gold。

- [ ] **Step 5：提交销售工具**

```bash
git add app/tools/queries.py app/tools/registry.py tests/test_tool_queries.py
git commit -m "feat(tools): add bounded sales query"
```

### Task 6：实现 `query_traffic`

**Files:**
- Modify: `app/tools/queries.py`
- Modify: `app/tools/registry.py`
- Modify: `tests/test_tool_queries.py`

`query_traffic` 固定输出：

```text
period, product_id, category, impressions, clicks, visits,
observed_days, missing_days
```

每个请求窗口与商品构造完整 grid。`include_missing=true` 时保留有缺失日的商品并把数值聚合为
非缺失日之和，同时准确输出 `observed_days`/`missing_days`；`false` 时剔除
`missing_days > 0` 的商品，避免把不完整窗口误作完整数据。

- [ ] **Step 1：写缺失数据 RED 测试**

```python
def test_query_traffic_preserves_missingness(
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    args = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "comparison_start_date": "2026-03-02",
        "comparison_end_date": "2026-03-31",
        "product_ids": ["P005"],
        "include_missing": True,
    }
    included = registry.invoke("query_traffic", args, tool_context)
    current = next(row for row in included.rows if row.period == "current")
    assert current.observed_days + current.missing_days == 30
    assert current.missing_days > 0

    excluded = registry.invoke(
        "query_traffic", {**args, "include_missing": False}, tool_context
    )
    assert all(row.period != "current" for row in excluded.rows)


def test_query_traffic_counts_implicit_missing_product_day(
    catalog_with_one_traffic_day_omitted: Catalog,
) -> None:
    result = registry_for(catalog_with_one_traffic_day_omitted).invoke(
        "query_traffic",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-03",
            "comparison_start_date": "2026-03-29",
            "comparison_end_date": "2026-03-31",
            "product_ids": ["P001"],
            "include_missing": True,
        },
        context_for(catalog_with_one_traffic_day_omitted),
    )
    current = next(row for row in result.rows if row.period == "current")
    assert current.observed_days == 2
    assert current.missing_days == 1
```

该 fixture 只在 `tmp_path` 创建三天最小快照：两条正常 traffic 行，第三天完全没有底表行；
不得用 `is_missing=true` 替代被省略的行，否则该测试无法覆盖“隐式缺日”。

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_queries.py -k traffic -q`

Expected: FAIL，`query_traffic` handler 未实现。

- [ ] **Step 3：实现 traffic CTE**

使用 `periods CROSS JOIN selected_products CROSS JOIN generate_series(period_start,
period_end, interval '1 day') AS calendar(day) LEFT JOIN traffic`，先构造完整
`period × product × day` grid；计数口径：

```sql
count(*) filter (where traffic.date is not null and not traffic.is_missing) as observed_days,
count(*) filter (where traffic.date is null or traffic.is_missing) as missing_days,
coalesce(sum(traffic.impressions) filter (where not traffic.is_missing), 0)
  as impressions,
coalesce(sum(traffic.clicks) filter (where not traffic.is_missing), 0) as clicks,
coalesce(sum(traffic.visits) filter (where not traffic.is_missing), 0) as visits
```

窗口长度由 `date_diff('day', period_start, period_end) + 1` 验证，且每个输出行必须满足
`observed_days + missing_days == window_days`；若底表没有完整 product-day grid，缺少的 join
行也计入 `missing_days`，不能只数 `is_missing=true`。

- [ ] **Step 4：运行 GREEN**

Run: `python3.12 -m pytest tests/test_tool_queries.py -k traffic -q`

Expected: traffic 测试全部 PASS。

- [ ] **Step 5：提交流量工具**

```bash
git add app/tools/queries.py app/tools/registry.py tests/test_tool_queries.py
git commit -m "feat(tools): add missing-aware traffic query"
```

### Task 7：实现 `query_marketing`

**Files:**
- Modify: `app/tools/queries.py`
- Modify: `app/tools/registry.py`
- Modify: `tests/test_tool_queries.py`

`query_marketing` 固定输出：

```text
period, product_id, category, campaign_id, spend
```

按 `period × product × category × campaign` 聚合；营销工具只返回实际 spend，不自行计算
ROAS。

- [ ] **Step 1：写营销查询 RED 测试**

```python
def test_query_marketing_matches_catalog_spend(
    catalog: Catalog,
    registry: ToolRegistry,
    tool_context: ToolContext,
) -> None:
    result = registry.invoke(
        "query_marketing",
        {
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "comparison_start_date": "2026-03-02",
            "comparison_end_date": "2026-03-31",
            "product_ids": ["P001"],
        },
        tool_context,
    )
    actual = sum(row.spend for row in result.rows if row.period == "current")
    expected = catalog.execute(
        """
        select sum(spend) from marketing
        where product_id = ? and date between ? and ?
        """,
        ["P001", "2026-04-01", "2026-04-30"],
    ).fetchone()
    assert expected is not None
    assert actual == pytest.approx(float(expected[0]))
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_queries.py -k marketing -q`

Expected: FAIL，`query_marketing` handler 未实现。

- [ ] **Step 3：实现参数化营销查询并接入 Registry**

复用 Task 5 的固定 periods CTE 与 product filter 模板；不得为 Registry 增加第六个通用 SQL
工具，也不得修改冻结的 `tool_contract_v1.yaml`。

- [ ] **Step 4：运行 GREEN 与五工具注册检查**

Run: `python3.12 -m pytest tests/test_tool_queries.py tests/test_tool_registry.py -q`

Expected: 全部 PASS，Registry 名称集合恰好为五个。

- [ ] **Step 5：提交营销工具**

```bash
git add app/tools/queries.py app/tools/registry.py tests/test_tool_queries.py
git commit -m "feat(tools): add bounded marketing query"
```

### Task 8：实现纯派生 `calculate_metrics`

**Files:**
- Create: `app/tools/metrics.py`
- Create: `tests/test_tool_metrics.py`
- Modify: `app/tools/registry.py`
- Read only: `data/evaluation_cases/v1/cases.jsonl`

指标依赖矩阵必须在代码中显式冻结：

```python
METRIC_REQUIREMENTS = {
    "current_gmv": {"query_sales"},
    "previous_gmv": {"query_sales"},
    "gmv_change_rate": {"query_sales"},
    "current_orders": {"query_sales"},
    "previous_orders": {"query_sales"},
    "current_aov": {"query_sales"},
    "previous_aov": {"query_sales"},
    "aov_change_rate": {"query_sales"},
    "current_impressions": {"query_traffic"},
    "previous_impressions": {"query_traffic"},
    "current_clicks": {"query_traffic"},
    "previous_clicks": {"query_traffic"},
    "current_visits": {"query_traffic"},
    "previous_visits": {"query_traffic"},
    "current_ctr": {"query_traffic"},
    "previous_ctr": {"query_traffic"},
    "current_cvr": {"query_sales", "query_traffic"},
    "previous_cvr": {"query_sales", "query_traffic"},
    "cvr_change": {"query_sales", "query_traffic"},
    "cvr_change_rate": {"query_sales", "query_traffic"},
    "traffic_change_rate": {"query_traffic"},
    "current_observed_days": {"query_traffic"},
    "previous_observed_days": {"query_traffic"},
    "current_spend": {"query_marketing"},
    "previous_spend": {"query_marketing"},
    "current_roas": {"query_sales", "query_marketing"},
    "previous_roas": {"query_sales", "query_marketing"},
    "current_refund_rate": {"query_sales"},
    "refund_rate": {"query_sales"},
    "evidence_value": {"query_sales", "query_traffic"},
}
```

- [ ] **Step 1：写已知小样本 RED 测试**

```python
def test_calculate_metrics_uses_prior_executions_and_safe_division() -> None:
    sales = execution(
        "call_sales",
        "result_0001",
        "query_sales",
        compatible_sales_arguments(),
        [
            row("current", "P001", gmv=120.0, orders=3, units=4, refund_orders=1),
            row("previous", "P001", gmv=100.0, orders=2, units=3, refund_orders=0),
        ],
    )
    traffic = execution(
        "call_traffic",
        "result_0002",
        "query_traffic",
        compatible_traffic_arguments(),
        [
            traffic_row(
                "current", "P001", impressions=1000, clicks=100, visits=80,
                observed_days=30, missing_days=0,
            ),
            traffic_row(
                "previous", "P001", impressions=800, clicks=80, visits=50,
                observed_days=30, missing_days=0,
            ),
        ],
    )
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=[
                "current_gmv", "gmv_change_rate", "current_aov",
                "aov_change_rate", "current_cvr", "previous_cvr",
                "cvr_change", "cvr_change_rate", "traffic_change_rate",
                "current_observed_days", "previous_observed_days",
                "current_refund_rate", "refund_rate", "evidence_value",
            ],
            group_by=["product_id"],
        ),
        context_with(sales, traffic),
    )
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


def test_calculate_metrics_rejects_missing_or_incompatible_sources() -> None:
    with pytest.raises(ValueError, match="requires query_sales"):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["aov_change_rate"],
                group_by=[],
            ),
            empty_context(),
        )
    with pytest.raises(ValueError, match="requires query_traffic"):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["current_cvr"],
                group_by=[],
            ),
            context_with(sales_result()),
        )


def test_calculate_metrics_returns_none_for_every_zero_denominator() -> None:
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=[
                "gmv_change_rate", "current_aov", "aov_change_rate", "current_ctr",
                "current_cvr", "current_roas", "refund_rate",
            ],
            group_by=[],
        ),
        context_with(
            zero_denominator_sales_execution(),
            zero_denominator_traffic_execution(),
            zero_denominator_marketing_execution(),
        ),
    )
    row = result.rows[0]
    assert all(
        getattr(row, metric) is None
        for metric in (
            "gmv_change_rate", "current_aov", "aov_change_rate", "current_ctr",
            "current_cvr", "current_roas", "refund_rate",
        )
    )


def test_aov_change_rate_is_none_when_previous_aov_is_zero() -> None:
    sales = execution(
        "call_sales",
        "result_0001",
        "query_sales",
        compatible_sales_arguments(),
        [
            row("current", "P001", gmv=120.0, orders=3),
            row("previous", "P001", gmv=0.0, orders=2),
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


def test_cvr_requires_complete_periods_and_matches_gold_missingness() -> None:
    sales = sales_execution_with_period_orders(current=3, previous=2)
    traffic = traffic_execution(
        traffic_row(
            "current",
            "P005",
            visits=80,
            observed_days=29,
            missing_days=1,
        ),
        traffic_row(
            "previous",
            "P005",
            visits=50,
            observed_days=30,
            missing_days=0,
        ),
    )
    result = calculate_metrics(
        CalculateMetricsInput(
            metrics=[
                "current_cvr",
                "previous_cvr",
                "cvr_change",
                "cvr_change_rate",
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


def test_calculate_metrics_rejects_duplicate_required_source() -> None:
    with pytest.raises(ValueError, match="duplicate query_sales"):
        calculate_metrics(
            CalculateMetricsInput(metrics=["current_gmv"], group_by=[]),
            context_with(sales_execution("call_1"), sales_execution("call_2")),
        )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (different_current_window, "window"),
        (different_comparison_window, "window"),
        (different_product_filter, "product_ids"),
        (sales_without_refunds, "include_refunds"),
        (traffic_without_missing, "include_missing"),
        (different_dataset, "dataset_id"),
        (different_source_label, "source_label"),
    ],
)
def test_calculate_metrics_rejects_incompatible_sources(mutator, message) -> None:
    executions = compatible_sales_traffic_marketing_executions()
    mutator(executions)
    with pytest.raises(ValueError, match=message):
        calculate_metrics(
            CalculateMetricsInput(
                metrics=["current_cvr", "current_roas", "refund_rate"],
                group_by=[],
            ),
            context_with(*executions),
        )
```

`different_dataset`/`different_source_label` 仅在该防御性单元测试中用 `model_construct()` 制造
绕过 Registry 的不可信历史对象，以证明 `calculate_metrics` 自身仍拒绝跨来源拼接；正常 Runner
路径中的结果必须先通过 Registry 二次验证。

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_tool_metrics.py -q`

Expected: collection FAIL，`app.tools.metrics` 不存在。

- [ ] **Step 3：实现聚合、join 和安全除法**

关键规则：

- 只从当前 `ToolContext.prior_executions` 取值，不接受参数中的结果 ID 或内联 rows；
- 对每个被请求指标所需的 tool name，必须恰好存在一个既往执行；同一所需工具出现两次即以
  `duplicate source execution` 失败，即使 arguments 或 result 完全相同，也不采用
  “最后一个获胜”、合并或去重；
- 所有被选来源必须有相同 `dataset_id`、`source_label`，并与当前 Catalog 的 verified summary
  一致；
- `query_sales`、`query_traffic`、`query_marketing` 的四个窗口边界必须逐项完全相同，
  `product_ids` 按输入 validator 保证唯一后以排序 tuple 比较，空列表表示全商品且不能与显式
  全量 ID 列表视为相同；
- 销售派生指标要求原 `query_sales.arguments["include_refunds"] is True`；流量派生指标要求
  原 `query_traffic.arguments["include_missing"] is True`。这样退款与缺失口径不会在派生阶段
  被隐式改写；不兼容时显式失败，不静默丢行；
- `group_by` 必须存在于所有相关来源；跨来源按 `period + group_by` 一对一 merge，重复键失败；
- 先求和基础量，再求比率，禁止平均行级 ratio；
- 使用 `app.data.metrics.safe_divide`；分母 0 或缺失返回 `None`；
- `gmv_change_rate=(current_gmv-previous_gmv)/previous_gmv`；
- `current_aov=current_gmv/current_orders`、`previous_aov=previous_gmv/previous_orders`；
  `aov_change_rate=(current_aov-previous_aov)/previous_aov`，三者只需要且必须来自同一个
  `query_sales` 执行；先分别用 `safe_divide` 求两个 AOV，再用 `safe_divide` 求变化率，
  `previous_orders == 0` 导致 `previous_aov is None`，或 `previous_aov == 0` 时
  `aov_change_rate` 均为 `None`；
- `current_observed_days`/`previous_observed_days` 直接来自对应 period 的
  `query_traffic.observed_days`；每个 period 的 `expected_days` 必须由该次已规范化查询参数按
  `end_date - start_date + 1` 与
  `comparison_end_date - comparison_start_date + 1` 分别计算，禁止硬编码 30；
- `ctr=clicks/impressions`；`current_cvr` 仅当
  `current_observed_days == current_expected_days` 时为 `current_orders/current_visits`，
  否则为 `None`；`previous_cvr` 同理仅在 previous period 完整时计算。即使已有部分 visits，
  不完整 period 也不得给出 CVR，与冻结 Gold 的缺失口径一致；
- `cvr_change=current_cvr-previous_cvr` 是绝对差；`cvr_change_rate=(current_cvr-
  previous_cvr)/previous_cvr` 是相对变化率。任一 period 不完整、任一 CVR 为 `None`，或相对
  变化率分母为 0 时，相应变化指标返回 `None`；
- `traffic_change_rate=(current_visits-previous_visits)/previous_visits`，分母为 0 时
  返回 `None`；traffic 不完整不会把该指标置空，因为冻结 Gold 对它使用已观测 visits；
- `roas=gmv/spend`；`current_refund_rate=current_refund_orders/current_orders`，
  `refund_rate` 是冻结 Cases 中同一当前窗口口径的兼容名称，两者必须返回相同值，分母为 0 时
  均为 `None`；
- `evidence_value` 只允许在 `group_by == ["product_id"]` 时请求，并严格复现冻结
  `next_week_priority` Gold 的优先级规则：若任一 period 不完整，取
  `float(min(current_observed_days, previous_observed_days))`；否则若 `refund_rate >= 0.12`
  取 `refund_rate`；否则若 `cvr_change <= -0.01` 取 `cvr_change`；否则若
  `traffic_change_rate <= -0.20` 取 `traffic_change_rate`；否则为 `None`。它依赖同一组兼容的
  `query_sales` 与 `query_traffic`，不得读取 Gold SQL、Cases 的 `gold_metrics` 或
  `gold_evidence`；
- 输出列顺序严格为 `group_by` 后接请求中的 `metrics` 顺序，行按 `group_by` 排序；
- 结果不得包含 source rows、SQL 或 Gold。

最小计算骨架：

```python
def calculate_metrics(
    args: CalculateMetricsInput,
    context: ToolContext,
) -> CalculateMetricsResult:
    sources = _resolve_sources(context.prior_executions, args.metrics)
    _validate_source_compatibility(sources, context.catalog)
    _validate_requirements(args.metrics, sources)
    aggregates = _aggregate_sources(sources, args.group_by)
    rows = [
        _calculate_row(group_key, period_values, args.metrics, args.group_by)
        for group_key, period_values in aggregates.items()
    ]
    return _rows_result("calculate_metrics", rows, context)
```

`_resolve_sources()` 必须从每个 `PriorToolExecution.result.tool_name` 识别来源，同时保留该执行的
`arguments`，用对应查询 Input model 再验证并规范化日期、商品列表和布尔 flag 后执行上述
兼容性检查；不得退化为只收集 `result` 或直接比较未解析 JSON。`query_product` 和当前指标
不需要的其他工具执行可以忽略，但不能参与窗口推断或掩盖所需来源的重复。

- [ ] **Step 4：运行 GREEN，再与一个冻结 Gold Case 数值交叉验证**

追加集成测试：从冻结 Cases 中按 `case_id == "CASE_003"` 精确取样，临时生成其
`dataset_id == "e1e81533c25e03e5"` 对应的 Development 快照；使用 CASE_003
`expected_tool_calls` 中的 `query_sales` 参数执行查询，再以原顺序请求
`["aov_change_rate", "current_aov", "current_gmv", "current_orders",
"gmv_change_rate", "previous_aov", "previous_gmv", "previous_orders"]` 和
`group_by=[]` 调用 `calculate_metrics`。断言结果只有一行、`columns` 与该指标顺序完全一致，
并逐项与 CASE_003 `gold_metrics` 比较：存在 `numeric_tolerances` 的浮点指标使用对应绝对容差，
`current_orders`/`previous_orders` 精确相等；必须明确覆盖
`aov_change_rate == pytest.approx(0.003732444516315593, abs=0.001)`。测试只用 CASE_003
作断言样例，运行时代码不得读取 Case，也不得从 Gold 反向填充结果。

Run: `python3.12 -m pytest tests/test_tool_metrics.py -q`

Expected: 全部 PASS。

Run: `python3.12 -m pytest tests/test_tool_queries.py tests/test_tool_registry.py -q`

Expected: 全部 PASS。

- [ ] **Step 5：提交指标工具**

```bash
git add app/tools/metrics.py app/tools/registry.py tests/test_tool_metrics.py
git commit -m "feat(tools): calculate metrics from prior results"
```

### Task 9：建立 Adapter 协议与 Deterministic adapter

**Files:**
- Create: `app/llm/__init__.py`
- Create: `app/llm/base.py`
- Create: `app/llm/schemas.py`
- Create: `app/llm/deterministic.py`
- Create: `tests/test_llm_deterministic.py`

Deterministic adapter 是基础设施测试用的可注入公开策略，不是 Baseline V1。其构造函数只接受
`Callable[[AdapterRequest], AssistantAction]`；不得读取 Evaluation Case、Gold、异常配置或
按 `business_task` 内置 Agent 决策。

- [ ] **Step 1：写协议与确定性 RED 测试**

```python
def test_deterministic_adapter_calls_policy_and_has_no_usage() -> None:
    seen: list[AdapterRequest] = []

    def policy(request: AdapterRequest) -> AssistantAction:
        seen.append(request)
        return AssistantAction(
            tool_calls=[
                ToolCall(
                    call_id="call_0001",
                    name="query_product",
                    arguments={"product_ids": ["P001"]},
                )
            ]
        )

    adapter = DeterministicAdapter(policy)
    request = AdapterRequest(
        user_input="查看 P001",
        tools=[],
        prior_tool_executions=[],
    )
    first = adapter.complete(request)
    second = adapter.complete(request)

    assert first == second
    assert first.usage is None
    assert first.raw_response is None
    assert seen == [request, request]
    assert adapter.adapter_name == "deterministic"


def test_action_requires_exactly_one_mode() -> None:
    with pytest.raises(ValidationError):
        AssistantAction()
    with pytest.raises(ValidationError):
        AssistantAction(
            tool_calls=[ToolCall(call_id="c1", name="query_product", arguments={})],
            final_answer="不能同时出现",
        )


@pytest.mark.parametrize("bad_value", [True, 1.5, "1", -1])
def test_usage_requires_non_negative_strict_integers(bad_value) -> None:
    with pytest.raises(ValidationError):
        Usage(
            prompt_tokens=bad_value,
            completion_tokens=0,
            total_tokens=0,
        )


def test_usage_rejects_inconsistent_total() -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        Usage(prompt_tokens=10, completion_tokens=4, total_tokens=15)
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_llm_deterministic.py -q`

Expected: collection FAIL，`app.llm` 不存在。

- [ ] **Step 3：实现 Schema、Protocol 与 adapter**

给公共接口冻结稿中的 `AssistantAction` 增加互斥 validator：

```python
@model_validator(mode="after")
def exactly_one_mode(self) -> "AssistantAction":
    if bool(self.tool_calls) == (self.final_answer is not None):
        raise ValueError("provide tool_calls or final_answer, exactly one")
    if len(self.tool_calls) > 1:
        raise ValueError("phase 2 supports one tool call per action")
    return self
```

`DeterministicAdapter.start_run()` 是 no-op；`complete()` 只调用 policy 并返回
`AdapterResponse(action=action, raw_response=None, usage=None)`。模块 import 不读取环境变量。
`Usage` 使用公共接口冻结稿中的 strict integer 与总数一致性 validator；不得让 Pydantic 把
布尔值、浮点或数字字符串宽松转换为 token 数。

- [ ] **Step 4：运行 GREEN**

Run: `python3.12 -m pytest tests/test_llm_deterministic.py -q`

Expected: 全部 PASS。

- [ ] **Step 5：提交 adapter 协议**

```bash
git add app/llm/__init__.py app/llm/base.py app/llm/schemas.py app/llm/deterministic.py tests/test_llm_deterministic.py
git commit -m "feat(llm): add deterministic adapter protocol"
```

### Task 10：实现 OpenAI-compatible adapter

**Files:**
- Create: `app/llm/openai_compatible.py`
- Create: `tests/test_llm_openai_compatible.py`
- Modify: `.env.example`

采用 Chat Completions 兼容子集：

```http
POST {base_url}/chat/completions
Authorization: Bearer <api_key>
Content-Type: application/json
```

请求只发送 `model`、`messages`、`tools`、`tool_choice="auto"`、`temperature`；不支持 stream。
上一轮工具结果以 `role="tool"` 消息送回，并用 `PriorToolExecution.call_id` 与上一轮 assistant
tool call 关联。为了不把 provider 行为写进 Runner，adapter 内维护当前单次 run 的完整
message transcript；公共 `LLMAdapter.start_run()` 已在 Task 9 冻结，每个 Runner run 必须先
调用它清空 transcript。OpenAI adapter 不是线程安全单例；每个 Runner 实例串行使用。

- [ ] **Step 1：写配置、请求和响应 RED 测试**

```python
def test_openai_adapter_requires_non_empty_config() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAICompatibleAdapter(
            base_url="https://example.test/v1",
            api_key="",
            model="test-model",
        )


def test_openai_adapter_normalizes_tool_call_without_network() -> None:
    transport = FakeTransport({
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "query_product",
                        "arguments": "{\"product_ids\":[\"P001\"]}",
                    },
                }],
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    })
    adapter = OpenAICompatibleAdapter(
        base_url="https://example.test/v1/",
        api_key="secret",
        model="test-model",
        transport=transport,
    )
    response = adapter.complete(sample_request())
    assert transport.url == "https://example.test/v1/chat/completions"
    assert response.action.tool_calls[0].arguments == {"product_ids": ["P001"]}
    assert response.usage.total_tokens == 14
    assert "secret" not in json.dumps(response.model_dump())


def test_openai_adapter_keeps_two_round_transcript_and_call_id() -> None:
    transport = SequenceTransport([
        tool_payload(call_id="call_abc", arguments='{"product_ids":["P001"]}'),
        answer_payload(content="P001 产品信息已取得。"),
    ])
    adapter = adapter_with(transport)
    adapter.start_run()
    first_request = sample_request()
    first = adapter.complete(first_request)
    execution = PriorToolExecution(
        call_id=first.action.tool_calls[0].call_id,
        arguments=first.action.tool_calls[0].arguments,
        result=sample_product_result(),
    )
    adapter.complete(first_request.model_copy(update={
        "prior_tool_executions": [execution],
    }))

    assert transport.payloads[0]["messages"] == [
        {"role": "user", "content": first_request.user_input},
    ]
    second_messages = transport.payloads[1]["messages"]
    assert second_messages[0] == transport.payloads[0]["messages"][0]
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[1]["tool_calls"][0]["id"] == "call_abc"
    assert second_messages[2]["role"] == "tool"
    assert second_messages[2]["tool_call_id"] == "call_abc"
    assert json.loads(second_messages[2]["content"])["result_id"] == "result_0001"


def test_openai_start_run_clears_previous_transcript() -> None:
    transport = SequenceTransport([
        answer_payload(content="first"),
        answer_payload(content="second"),
    ])
    adapter = adapter_with(transport)
    adapter.start_run()
    adapter.complete(sample_request(user_input="first run"))
    adapter.start_run()
    adapter.complete(sample_request(user_input="second run"))
    assert transport.payloads[1]["messages"] == [
        {"role": "user", "content": "second run"},
    ]


@pytest.mark.parametrize("call_id", ["unknown_call", "call_abc"])
def test_openai_adapter_rejects_unknown_or_replayed_tool_result(call_id) -> None:
    adapter, request, execution = adapter_waiting_for("call_abc")
    if call_id == "call_abc":
        adapter.complete(request_with(request, execution))
    with pytest.raises(OpenAIProtocolError, match="call_id"):
        adapter.complete(request_with(
            request,
            execution.model_copy(update={"call_id": call_id}),
        ))


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"choices": []}, "exactly one choice"),
        (tool_payload(arguments="{bad json"), "arguments must be valid JSON"),
        (tool_payload(arguments="[]"), "arguments must be a JSON object"),
        (answer_payload(content=""), "non-empty content"),
        (answer_payload(content="ok", usage={"prompt_tokens": True}), "integer"),
        (
            answer_payload(
                content="ok",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            ),
            "total_tokens",
        ),
    ],
)
def test_openai_adapter_rejects_malformed_provider_payload(
    payload: dict,
    message: str,
) -> None:
    adapter = adapter_with(FakeTransport(payload))
    with pytest.raises(OpenAIProtocolError, match=message):
        adapter.complete(sample_request())
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_llm_openai_compatible.py -q`

Expected: collection FAIL，adapter 模块不存在。

- [ ] **Step 3：实现可注入 transport 与严格解析**

```python
class JsonTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        raise NotImplementedError


class UrllibJsonTransport:
    def post(self, url, *, headers, payload, timeout_seconds):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise OpenAITransportError(f"provider returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise OpenAITransportError("provider request failed") from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise OpenAIProtocolError("provider response must be a JSON object")
        return parsed
```

实现约束：

- `base_url` 去末尾 `/`，只允许 `http`/`https` 且必须有 host；
- `api_key`、`model` 非空；`temperature` 为有限数且 `0 <= value <= 2`；
- 默认 timeout `30.0` 秒且必须 `> 0`；
- HTTP/URL/JSON 错误去除响应 body，避免泄露；
- `choices` 必须恰好 1 个；
- assistant message 必须恰好是一个 function tool call 或非空文本；
- function `arguments` 必须解析为 JSON object；
- usage 整体缺失时为 `None`，不能填 0；一旦存在就必须同时包含
  `prompt_tokens`/`completion_tokens`/`total_tokens` 三个 strict 非负整数，且 total 等于前两者
  之和，缺字段、布尔值、浮点或数字字符串均拒绝；
- `raw_response` 可保留经过 JSON 类型校验的 provider payload，但不得加入 request headers 或 key；
- `start_run()` 无条件清空 transcript、已发送 call ID 与待回填 call ID；
- 每个 run 的第一次 `complete()` 只追加一条 user message；provider 返回 tool call 后，把原始
  assistant tool-call message追加到 transcript；
- 下一轮仅接受一个此前尚未发送、且 `call_id` 与 transcript 中待回填 assistant tool call
  完全匹配的 `PriorToolExecution`，追加 `role="tool"`、相同 `tool_call_id` 和
  `canonical_tool_result_payload(result)` 经
  `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` 生成的紧凑 JSON；未知、缺失、
  重复或错配 call ID 抛 `OpenAIProtocolError`，不得按位置关联；
- 同一 run 后续 request 的 `user_input` 与 tools 必须与首轮一致，`prior_tool_executions` 必须是
  已知历史的单调追加，不得重放 tool message；
- 发送给 provider 的 tool result 不含数据库路径。

- [ ] **Step 4：运行 GREEN，并证明无 Key 路径不导入网络**

Run: `env -u LLM_API_KEY -u LLM_BASE_URL -u LLM_MODEL python3.12 -m pytest tests/test_llm_openai_compatible.py -q`

Expected: 全部 PASS，所有请求只经过 `FakeTransport`。

- [ ] **Step 5：更新环境变量说明并提交**

`.env.example`：

```dotenv
# Phase 2 offline smoke does not require an API key.
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
```

```bash
git add app/llm/openai_compatible.py tests/test_llm_openai_compatible.py .env.example
git commit -m "feat(llm): add openai compatible adapter"
```

### Task 11：实现通用 Runner 与结构化 trace

**Files:**
- Create: `app/agents/__init__.py`
- Create: `app/agents/schemas.py`
- Create: `app/agents/runner.py`
- Create: `tests/test_agent_runner.py`

本任务提交范围严格限于以上四个文件；Registry 错误类型及其边界测试已在 Task 3 完成，Task 11
只消费这些公开错误类型，不顺带修改 `app/tools/*` 或 `tests/test_tool_registry.py`。

- [ ] **Step 1：写成功链路 RED 测试**

```python
def test_runner_executes_tools_and_records_public_trace(
    catalog: Catalog,
) -> None:
    def policy(request: AdapterRequest) -> AssistantAction:
        if not request.prior_tool_executions:
            return call("call_0001", "query_product", {"product_ids": ["P001"]})
        return AssistantAction(final_answer="P001 产品信息已取得。")

    result = AgentRunner(
        registry=build_default_registry(),
        adapter=DeterministicAdapter(policy),
    ).run(RunRequest(user_input="查看 P001"), catalog)

    assert result.status == "completed"
    assert result.final_answer == "P001 产品信息已取得。"
    assert [event.sequence for event in result.decision_trace] == list(
        range(1, len(result.decision_trace) + 1)
    )
    assert [event.event_type for event in result.decision_trace] == [
        "adapter_request",
        "adapter_response",
        "tool_call",
        "tool_result",
        "adapter_request",
        "adapter_response",
        "final_answer",
    ]
    serialized = result.model_dump_json()
    assert "thought" not in serialized.lower()
    assert "sql" not in serialized.lower()


def test_runner_openai_adapter_fake_transport_full_chain(
    catalog: Catalog,
) -> None:
    transport = FakeTransport([
        tool_payload(
            call_id="call_sales",
            name="query_sales",
            arguments={
                "start_date": "2026-04-01",
                "end_date": "2026-04-30",
                "comparison_start_date": "2026-03-02",
                "comparison_end_date": "2026-03-31",
                "product_ids": [],
                "include_refunds": True,
            },
            usage=(10, 2, 12),
        ),
        tool_payload(
            call_id="call_metrics",
            name="calculate_metrics",
            arguments={
                "metrics": [
                    "current_gmv",
                    "previous_gmv",
                    "gmv_change_rate",
                ],
                "group_by": [],
            },
            usage=(20, 3, 23),
        ),
        answer_payload(
            content="GMV 对比计算完成。",
            usage=(5, 4, 9),
        ),
    ])
    adapter = OpenAICompatibleAdapter(
        base_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
        transport=transport,
    )
    with mock.patch.object(
        adapter,
        "start_run",
        wraps=adapter.start_run,
    ) as start_run:
        result = AgentRunner(
            registry=build_default_registry(),
            adapter=adapter,
        ).run(RunRequest(user_input="比较本期与上期 GMV"), catalog)

    start_run.assert_called_once_with()
    assert result.status == "completed"
    assert result.final_answer == "GMV 对比计算完成。"
    assert [item.call_id for item in result.prior_tool_executions] == [
        "call_sales",
        "call_metrics",
    ]
    assert [item.result.tool_name for item in result.prior_tool_executions] == [
        "query_sales",
        "calculate_metrics",
    ]
    assert result.usage.model_dump() == {
        "prompt_tokens": 35,
        "completion_tokens": 9,
        "total_tokens": 44,
    }
    assert [
        message["role"]
        for message in transport.payloads[2]["messages"]
    ] == ["user", "assistant", "tool", "assistant", "tool"]
    assert transport.payloads[1]["messages"][1]["tool_calls"][0]["id"] == (
        "call_sales"
    )
    assert transport.payloads[1]["messages"][2]["tool_call_id"] == "call_sales"
    assert transport.payloads[2]["messages"][3]["tool_calls"][0]["id"] == (
        "call_metrics"
    )
    assert transport.payloads[2]["messages"][4]["tool_call_id"] == "call_metrics"
    assert [
        json.loads(transport.payloads[2]["messages"][index]["content"])[
            "result_id"
        ]
        for index in (2, 4)
    ] == ["result_0001", "result_0002"]
    assert [
        event.payload.get("call_id")
        for event in result.decision_trace
        if event.event_type == "tool_call"
    ] == ["call_sales", "call_metrics"]
```

该集成测试在 `tests/test_agent_runner.py` 内定义可记录三次请求的 `FakeTransport`，响应序列严格为
`query_sales → calculate_metrics → final answer`；不得 mock `AgentRunner.run()`、Registry
或工具 handler。除最终状态外，必须同时锁定 Runner 调用一次 `start_run()`、两个 provider
`call_id` 在 assistant/tool transcript 中逐一配对、两个 `PriorToolExecution` 的顺序与
`result_0001`/`result_0002` 工具消息、tool-call trace，以及三轮 Usage 按字段累加为
`35/9/44`。测试只使用 fake transport，不访问网络。

- [ ] **Step 2：写失败边界 RED 测试**

```python
@pytest.mark.parametrize(
    "policy, error_code",
    [
        (lambda _: call("c1", "unknown", {}), "unknown_tool"),
        (lambda _: call("c1", "query_sales", {}), "invalid_arguments"),
    ],
)
def test_runner_normalizes_tool_failures(policy, error_code, catalog) -> None:
    result = runner(policy).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.decision_trace[-1].event_type == "error"
    assert result.decision_trace[-1].payload["code"] == error_code


@pytest.mark.parametrize(
    "registry, error_code",
    [
        (registry_with_malformed_result(), "invalid_tool_result"),
        (registry_with_raising_handler(KeyError("handler-miss")), "tool_execution_error"),
    ],
)
def test_runner_distinguishes_invalid_result_from_execution_error(
    registry, error_code, catalog
) -> None:
    result = AgentRunner(
        registry=registry,
        adapter=DeterministicAdapter(lambda _: call("c1", "query_sales", valid_sales_arguments())),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.decision_trace[-1].payload["code"] == error_code


def test_runner_rejects_duplicate_result_id_without_overwriting_execution(
    catalog: Catalog,
) -> None:
    policy = two_distinct_calls_policy("query_product")
    registry, invocation_count = registry_returning_fixed_result_id("result_0001")
    result = AgentRunner(
        registry=registry,
        adapter=DeterministicAdapter(policy),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.decision_trace[-1].payload["code"] == "duplicate_result_id"
    assert invocation_count() == 2
    assert [item.result.result_id for item in result.prior_tool_executions] == [
        "result_0001"
    ]


def test_runner_stops_at_max_steps(catalog: Catalog) -> None:
    counter = itertools.count(1)
    policy = lambda _: call(
        f"c{next(counter)}", "query_product", {"product_ids": ["P001"]}
    )
    result = runner(policy).run(
        RunRequest(user_input="x", max_steps=2),
        catalog,
    )
    assert result.status == "failed"
    assert result.decision_trace[-1].payload["code"] == "max_steps_exceeded"


def test_runner_structures_start_run_exception(catalog: Catalog) -> None:
    result = AgentRunner(
        registry=build_default_registry(),
        adapter=StartRunFailingAdapter(),
    ).run(RunRequest(user_input="x"), catalog)
    assert result.status == "failed"
    assert result.prior_tool_executions == []
    assert result.decision_trace[-1].event_type == "error"
    assert result.decision_trace[-1].payload == {
        "code": "adapter_start_error",
        "detail": "RuntimeError",
    }


def test_runner_strictly_revalidates_untrusted_custom_adapter_response(
    catalog: Catalog,
) -> None:
    call_1 = ToolCall(call_id="c1", name="query_product", arguments={"product_ids": ["P001"]})
    call_2 = ToolCall(call_id="c2", name="query_product", arguments={"product_ids": ["P002"]})
    invalid_actions = [
        AssistantAction.model_construct(
            tool_calls=[call_1, call_2],
            final_answer=None,
        ),
        AssistantAction.model_construct(
            tool_calls=[call_1],
            final_answer="不能与工具调用同时出现",
        ),
    ]
    for action in invalid_actions:
        registry = mock.Mock(wraps=build_default_registry())
        adapter = CustomAdapter(
            AdapterResponse.model_construct(
                action=action,
                raw_response=None,
                usage=None,
            )
        )
        result = AgentRunner(registry=registry, adapter=adapter).run(
            RunRequest(user_input="x"),
            catalog,
        )
        assert result.status == "failed"
        assert result.prior_tool_executions == []
        assert result.decision_trace[-1].payload["code"] == "adapter_error"
        assert result.decision_trace[-1].payload["detail"] == "ValidationError"
        registry.invoke.assert_not_called()
```

其中 `registry_with_malformed_result()` 必须让 handler 正常返回、仅在 Registry 输出复验时触发
`ToolOutputValidationError`；`registry_with_raising_handler(KeyError("handler-miss"))` 必须让
已成功解析出的工具 handler 本身抛 `KeyError`，锁定它被归为 `tool_execution_error`，而不是
`unknown_tool`。`CustomAdapter` 直接返回传入响应；非法 action 只能用 `model_construct()` 构造，
从而证明 Runner 的独立边界复验，而不是 adapter 或普通 Pydantic 构造器，拒绝多个 tool calls
及 tool call + final answer 两种组合，且 Registry 一次也不调用。
`registry_returning_fixed_result_id(...)` 返回 Registry 和可查询的调用
计数器；它必须连续两次返回同一合法 `result_0001`，测试断言 handler 确实被调用两次，两个
tool call 使用不同 `call_id`，从而只命中 `duplicate_result_id` 分支，而不是先被 result Schema
的 pattern 校验截断。另加测试
覆盖重复 `call_id`、final answer 前结果顺序和两次 `run()` 都调用
`start_run()`；第二次 `start_run()` 抛错时也必须返回新的、仅含本次 `adapter_start_error` 的
结构化失败结果，不得泄露上一次 run 的 execution 或 trace。为保持本计划总计 70 个测试 case，
上方单个非法自定义 adapter 测试替换原计划的通用 `adapter exception` case，并在函数内部依次
验证两种非法组合，不增加 pytest case 数。

- [ ] **Step 3：运行 RED**

Run: `python3.12 -m pytest tests/test_agent_runner.py -q`

Expected: collection FAIL，`app.agents.runner` 不存在。

- [ ] **Step 4：实现 Runner 状态机**

`app/agents/runner.py` 从 `app.tools` 导入公开的 `UnknownToolError`、
`ToolInputValidationError` 与 `ToolOutputValidationError`，并从 `app.llm` 导入
`AdapterResponse`。不得继续直接捕获 Pydantic `ValidationError` 来猜测 Registry 失败发生在
输入还是输出边界；adapter 返回值的 `ValidationError` 则在 adapter 边界统一归入
`adapter_error`。

```python
def run(self, request: RunRequest, catalog: Catalog) -> RunResult:
    executions: list[PriorToolExecution] = []
    trace = TraceBuilder()
    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    usage = UsageAccumulator()

    try:
        self._adapter.start_run()
    except Exception as exc:
        return _failed(
            trace, executions, usage, "adapter_start_error", type(exc).__name__
        )

    for _ in range(request.max_steps):
        adapter_request = AdapterRequest(
            user_input=request.user_input,
            tools=self._registry.openai_tools(),
            prior_tool_executions=list(executions),
        )
        trace.adapter_request(adapter_request)
        try:
            response = self._adapter.complete(adapter_request)
            response = AdapterResponse.model_validate(
                response.model_dump(),
                strict=True,
            )
        except Exception as exc:
            return _failed(trace, executions, usage, "adapter_error", type(exc).__name__)
        trace.adapter_response(response.action, response.usage)
        usage.add(response.usage)

        if response.action.final_answer is not None:
            trace.final_answer(response.action.final_answer)
            return _completed(trace, executions, usage, response.action.final_answer)

        call = response.action.tool_calls[0]
        if call.call_id in seen_call_ids:
            return _failed(trace, executions, usage, "duplicate_call_id", call.call_id)
        seen_call_ids.add(call.call_id)
        trace.tool_call(call)
        context = ToolContext(
            catalog=catalog,
            prior_executions=tuple(executions),
            next_result_id=lambda: f"result_{len(executions) + 1:04d}",
        )
        try:
            result = self._registry.invoke(call.name, call.arguments, context)
        except UnknownToolError:
            return _failed(trace, executions, usage, "unknown_tool", call.name)
        except ToolInputValidationError:
            return _failed(trace, executions, usage, "invalid_arguments", call.name)
        except ToolOutputValidationError:
            return _failed(trace, executions, usage, "invalid_tool_result", call.name)
        except Exception as exc:
            return _failed(
                trace, executions, usage, "tool_execution_error", type(exc).__name__
            )
        if result.result_id in seen_result_ids:
            return _failed(
                trace, executions, usage, "duplicate_result_id", result.result_id
            )
        seen_result_ids.add(result.result_id)
        executions.append(PriorToolExecution(
            call_id=call.call_id,
            arguments=call.arguments,
            result=result,
        ))
        trace.tool_result(result)

    return _failed(
        trace, executions, usage, "max_steps_exceeded", request.max_steps
    )
```

trace payload 规则：

- `adapter_request`：只记 user input、工具名列表、已有 result IDs；
- `adapter_response`：只记 final/tool 模式、call ID、tool name、usage availability；
- `tool_call`：记结构化 arguments；
- `tool_result`：记 result ID、tool name、dataset ID、columns、row count、warnings，不复制 rows；
- `error`：记稳定 `code` 与非敏感 detail，不记录 traceback、API key 或 provider body；
- `sequence` 从 1 连续递增；
- Usage 仅在 provider 返回时相加；所有轮都缺失则 `RunResult.usage=None`。
- `start_run()` 在任何 adapter request 之前调用，但在已初始化的本次 trace/error 边界内；
  其异常归一化为 `adapter_start_error`，detail 只含异常类名。
- `complete()` 返回后、记录 `adapter_response` 或读取 action 前，用
  `AdapterResponse.model_validate(response.model_dump(), strict=True)` 做不可信边界复验；
  多个 tool calls、tool call + final answer 或其他非法返回统一为 `adapter_error`，不得调用
  Registry。
- 只有 `UnknownToolError` 归一化为 `unknown_tool`；不得用 `except KeyError` 代替，否则 handler
  内部的查表错误会被误分类。
- Registry 输入拒绝归一化为 `invalid_arguments`；handler 正常返回但输出边界拒绝归一化为
  `invalid_tool_result`；只有 handler 自身或其他未分类调用异常归一化为
  `tool_execution_error`。
- Registry 返回重复 `result_id` 时以 `duplicate_result_id` 失败；不得覆盖此前 execution。

- [ ] **Step 5：运行 GREEN 和相关回归**

Run: `python3.12 -m pytest tests/test_agent_runner.py tests/test_llm_deterministic.py tests/test_llm_openai_compatible.py tests/test_tool_registry.py -q`

Expected: 全部 PASS。

- [ ] **Step 6：提交 Runner**

```bash
git add app/agents/__init__.py app/agents/schemas.py app/agents/runner.py tests/test_agent_runner.py
git commit -m "feat(agent): run adapter actions with structured trace"
```

只 stage 上述四个 Task 11 文件；不得夹带 `app/tools/*`、Registry 测试或其他任务改动。

### Task 12：增加无网络 offline smoke

**Files:**
- Create: `app/experiments/__init__.py`
- Create: `app/experiments/phase2_smoke.py`
- Create: `tests/test_phase2_smoke.py`
- Modify: `tests/test_project_setup.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify only if needed: `.gitignore`

Smoke 只证明“adapter → Runner → Registry → query_sales → calculate_metrics → final answer”贯通；
不读取 Case，不评分，不生成 Phase 3 run artifact，不命名为 V1/V2。

- [ ] **Step 1：写 CLI 与入口契约 RED 测试**

```python
def test_phase2_offline_smoke_needs_no_key_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    exit_code = main([
        "--config", str(ROOT / "configs/data/synthetic_v1.yaml"),
        "--work-dir", str(tmp_path),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["adapter"] == "deterministic"
    assert payload["tool_names"] == ["query_sales", "calculate_metrics"]
    assert payload["dataset_id"] == "e1e81533c25e03e5"
    assert payload["trace_event_count"] > 0
    assert "gmv_change_rate" in payload["final_answer"]
    assert not (tmp_path / "run_manifest.json").exists()
```

同时把 `tests/test_project_setup.py` 中对 `project.scripts` 的断言精确更新为：

```python
assert project["scripts"] == {
    "build-phase1": "app.data.phase1:main",
    "phase2-smoke": "app.experiments.phase2_smoke:main",
}
```

- [ ] **Step 2：运行 RED**

Run: `python3.12 -m pytest tests/test_phase2_smoke.py tests/test_project_setup.py -q`

Expected: FAIL；smoke 模块不存在，且 `project.scripts` 尚未精确包含上述两个入口。

- [ ] **Step 3：实现固定 smoke policy**

```python
def smoke_policy(request: AdapterRequest) -> AssistantAction:
    names = [
        execution.result.tool_name
        for execution in request.prior_tool_executions
    ]
    if names == []:
        return tool_call("smoke_sales", "query_sales", SALES_ARGUMENTS)
    if names == ["query_sales"]:
        return tool_call(
            "smoke_metrics",
            "calculate_metrics",
            {
                "metrics": [
                    "current_gmv",
                    "previous_gmv",
                    "gmv_change_rate",
                ],
                "group_by": [],
            },
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
```

CLI 在 `work-dir/snapshot` 调用
`build_snapshot(config_path, work_dir / "snapshot")`，完成后用 context manager 关闭
Catalog；只向 stdout 打一行稳定 JSON，错误写 stderr 并返回 1。临时 snapshot 是测试/运行中间
产物，不提交。

- [ ] **Step 4：增加入口**

`pyproject.toml`：

```toml
[project.scripts]
build-phase1 = "app.data.phase1:main"
phase2-smoke = "app.experiments.phase2_smoke:main"
```

`Makefile`：

```make
PHASE2_SMOKE_DIR ?= .tmp/phase2-smoke

.PHONY: phase2-smoke

phase2-smoke: check-python
	$(PYTHON) -m app.experiments.phase2_smoke \
		--config "$(DEVELOPMENT_CONFIG)" \
		--work-dir "$(PHASE2_SMOKE_DIR)"
```

`phase2-smoke` 默认目录已由 `.gitignore` 的 `.tmp/` 规则覆盖；若不存在该规则，只追加
`.tmp/`，不得忽略整个 `data/` 或 `outputs/`。

- [ ] **Step 5：运行 GREEN 与真实 CLI**

Run: `python3.12 -m pytest tests/test_phase2_smoke.py tests/test_project_setup.py -q`

Expected: 两个定向测试文件全部 PASS。

Run: `env -u LLM_API_KEY -u LLM_BASE_URL -u LLM_MODEL make PYTHON=python3.12 PHASE2_SMOKE_DIR=/tmp/phase2-smoke phase2-smoke`

Expected: exit 0；stdout JSON 的 `status` 为 `completed`、`adapter` 为
`deterministic`、`tool_names` 为 `["query_sales","calculate_metrics"]`。

- [ ] **Step 6：提交 smoke**

```bash
git add app/experiments/__init__.py app/experiments/phase2_smoke.py tests/test_phase2_smoke.py tests/test_project_setup.py pyproject.toml Makefile
# 仅当本任务确实向 .gitignore 新增了 .tmp/ 时，再单独执行：
git add .gitignore
git commit -m "feat(phase2): add offline agent foundation smoke"
```

只在 `.gitignore` 实际变化时把它加入 `git add`；不得提交 `/tmp/phase2-smoke` 或 `.tmp/`。

### Task 13：全量门禁、文档和阶段状态

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`
- Test: `tests/test_phase1_frozen_assets.py`
- Test: `tests/test_tool_registry.py`
- Test: `tests/test_tool_queries.py`
- Test: `tests/test_tool_metrics.py`
- Test: `tests/test_llm_deterministic.py`
- Test: `tests/test_llm_openai_compatible.py`
- Test: `tests/test_agent_runner.py`
- Test: `tests/test_phase2_smoke.py`

- [ ] **Step 1：先运行冻结资产门禁**

Run: `python3.12 -m pytest tests/test_phase1_frozen_assets.py -q`

Expected: `3 passed`。若失败，停止；不得更新 baseline 或冻结资产来使测试变绿。

- [ ] **Step 2：运行 Phase 2 定向测试**

Run:

```bash
python3.12 -m pytest \
  tests/test_tool_registry.py \
  tests/test_tool_queries.py \
  tests/test_tool_metrics.py \
  tests/test_llm_deterministic.py \
  tests/test_llm_openai_compatible.py \
  tests/test_agent_runner.py \
  tests/test_phase2_smoke.py -q
```

Expected: 全部 PASS；无网络调用、无 API Key。

- [ ] **Step 3：运行全量测试与静态检查**

Run: `make PYTHON=python3.12 test`

Expected: 全部 PASS；Phase 1 当前基线为 `357 passed`，最终数量应为 357 加本计划新增测试的实际
数量，不在文档中预填虚构总数。

Run: `make PYTHON=python3.12 lint`

Expected: `All checks passed!`。

Run: `git diff --check`

Expected: 无输出，exit 0。

- [ ] **Step 4：再次运行离线 smoke 并记录真实输出**

Run: `env -u LLM_API_KEY -u LLM_BASE_URL -u LLM_MODEL make PYTHON=python3.12 PHASE2_SMOKE_DIR="$(mktemp -d)/phase2-smoke" phase2-smoke`

Expected: exit 0；输出为单行合法 JSON；临时目录位于系统 temp 下，仓库数据目录未被写入。

- [ ] **Step 5：更新 README，只写实际可运行能力**

新增 `Phase 2` 小节，包含：

````markdown
## Phase 2

Phase 2 提供五个受控工具、Tool Registry、Deterministic/OpenAI-compatible adapter、
通用 Runner 与结构化 decision trace；尚未实现 Baseline V1 或 Optimized V2。

无 API Key 离线验证：

```bash
make PYTHON=python3.12 PHASE2_SMOKE_DIR=/tmp/phase2-smoke phase2-smoke
```

真实模型仅在显式配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 后可由
`OpenAICompatibleAdapter` 使用；offline smoke 不读取这些变量，也不访问网络。
````

不要声称已经运行 Baseline、Optimized、A/B Evaluation 或获得任何效果指标。

- [ ] **Step 6：更新 `PROJECT_STATUS.md` 为真实结果**

仅在所有门禁通过后：

- `Current Phase` 改为 `PHASE 2 — Agent 基础能力`；
- 状态改为 `Completed / Awaiting Review`；
- Completed 记录五工具、Registry、两个 adapter、Runner、trace、offline smoke、冻结门禁；
- Pending 改为“用户审阅 Phase 2 后进入 Phase 3”；
- Evaluation Results 中 Baseline/Optimized/A/B 等仍保持 `Pending / Not Run`，
  Token Usage/Cost 仍保持 `Unavailable`；
- Validation 填入实际测试数量、实际 smoke JSON 摘要和冻结门禁结果；
- Known Issues 明确 OpenAI-compatible 只做 mock transport 测试，尚未选 provider/model，
  尚未实现 V1/V2。

- [ ] **Step 7：验证文档没有越界声明**

Run:

```bash
python3.12 -c "from pathlib import Path; text=(Path('README.md').read_text()+Path('PROJECT_STATUS.md').read_text()); assert 'Baseline Experiment | Completed' not in text; assert 'Optimized Experiment | Completed' not in text"
```

Expected: exit 0。

- [ ] **Step 8：提交 Phase 2 文档状态**

```bash
git add README.md PROJECT_STATUS.md
git commit -m "docs(phase2): record agent foundation validation"
```

## 最终验收清单

- [ ] Registry 恰好暴露五工具，每个工具绑定独立强类型 Result model，名称和固定
  JSON Schema digest 稳定，handler 输出经过 Registry 二次严格验证；输入与输出边界失败分别
  抛 `ToolInputValidationError`、`ToolOutputValidationError`；`get()`/`invoke()` 的未知名称只抛
  `UnknownToolError`，handler 异常保持原样。
- [ ] 冻结契约四工具的 `required_parameters` 与 Registry JSON Schema `required` 逐项完全
  相等；`query_sales.product_ids`、`query_traffic.product_ids`、
  `calculate_metrics.group_by` 必填且无默认值，冻结 100 Cases 的全部 expected calls 均通过
  对应 InputModel 的 strict JSON 验证。
- [ ] 所有查询经 Pydantic 验证并使用参数化、受限 SQL。
- [ ] Agent/adapter 无法取得 Catalog、数据库路径或任意 SQL 接口。
- [ ] `calculate_metrics` 的公开参数与冻结契约一致，只消费当前 run 中带 arguments/result 的
  `PriorToolExecution`，拒绝重复或窗口、商品、退款、缺失口径、dataset/source 不兼容的来源，
  并按基础量聚合后计算 ratio。
- [ ] 所有查询窗口、重复商品 ID、隐式缺日、零分母、空查询和退款开关均有明确测试。
- [ ] Deterministic adapter 在无 Key、无网络条件下完成真实工具链。
- [ ] OpenAI-compatible adapter 只实现冻结的 Chat Completions 子集；两轮 transcript 通过
  `call_id` 精确关联，`start_run()` 清理状态，错误与 strict Usage 语义明确。
- [ ] AgentRunner + OpenAI-compatible adapter + FakeTransport 完整贯通
  `query_sales → calculate_metrics → final answer`，并验证 `start_run`、`call_id`、tool
  transcript、executions、trace 与多轮 Usage 累加。
- [ ] Runner 对每个 adapter 返回值执行
  `AdapterResponse.model_validate(response.model_dump(), strict=True)`，非法自定义 adapter
  构造的多个 tool calls 或 tool call + final answer 均在 Registry 调用前以 `adapter_error`
  拒绝。
- [ ] Runner 与 V1/V2 无关，不读取 Evaluation Case、Gold 或评分器。
- [ ] Runner 将 Registry 输入拒绝、输出拒绝、handler 执行异常分别归一化为
  `invalid_arguments`、`invalid_tool_result`、`tool_execution_error`，只将
  `UnknownToolError` 归一化为 `unknown_tool`；handler 抛出的 `KeyError` 必须归入
  `tool_execution_error`。Runner 拒绝重复 `result_id` 而不覆盖已有 execution。
- [ ] `decision_trace` 连续、有序、结构化，不含隐藏思维链、SQL、路径或密钥。
- [ ] offline smoke 真实经过 `query_sales → calculate_metrics → final answer`。
- [ ] Phase 1 干净构建的业务文件集合在排除同步锁后与 16 个冻结资产严格相等，且同步锁集合
  恰好为三个预期 `.lock` 路径；两个 Dataset ID、Case Set ID、Cases、Manifest、baseline 与
  工具契约均由测试内固定哈希常量直接校验且未变化。
- [ ] 定向测试、全量测试、Ruff、`git diff --check` 和 smoke 全部通过。
- [ ] README/PROJECT_STATUS 只记录实际结果；V1/V2 与评测结果仍为未运行。

任务结构与测试预算保持为 13 个 Task / 70 个 pytest case；以上三项阻塞修复均通过改写既有
Task 与既有测试位完成，不新增 Task 或测试 case。

## 实施时的提交序列

```text
test(phase2): guard frozen phase 1 assets
feat(tools): define strict phase 2 schemas
feat(tools): add controlled tool registry
feat(tools): add controlled product query
feat(tools): add bounded sales query
feat(tools): add missing-aware traffic query
feat(tools): add bounded marketing query
feat(tools): calculate metrics from prior results
feat(llm): add deterministic adapter protocol
feat(llm): add openai compatible adapter
feat(agent): run adapter actions with structured trace
feat(phase2): add offline agent foundation smoke
docs(phase2): record agent foundation validation
```

每次提交前只 stage 当前任务列出的文件并运行该任务的定向测试。若同一步发现独立缺陷，先增加
失败测试，再做独立 `fix(scope): describe defect` 提交；不得把多任务压成一个大提交。计划文档的创建不属于实施
提交，本轮按要求只写计划、不提交。
