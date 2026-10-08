# Phase 3 Deterministic Baseline V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并真实运行 Deterministic Baseline V1，对冻结的 100 个 Evaluation Cases 生成无评分、可追溯、可复现且不可覆盖的正式 Run Artifact。

**Architecture:** 在 Phase 2 的 `AgentRunner + DeterministicAdapter + ToolRegistry` 之上增加三个窄层：`app/baselines/v1` 负责通用文本解析、固定路由和证据化回答，`app/experiments/cases.py` 负责将完整 Case 严格投影为最小运行输入，`app/experiments/artifacts.py` 与 `baseline_v1.py` 负责批处理、校验和原子发布。每个 Case 新建 Policy、Adapter 和 Runner，Policy 只能接触 `user_input`、公开数据集上下文与当前 Run 的工具结果；完整标签、Gold、预期调用和评分字段永不进入执行边界。

**Tech Stack:** Python 3.11+、Pydantic v2、DuckDB、标准库 `json/hashlib/pathlib/tempfile/datetime`、Pytest、Ruff、Make

---

## 实施边界

### 本计划包含

- 五类主任务的 Deterministic V1 路由与固定工具链。
- 商品 ID、支持日期表达和默认 30 天窗口解析。
- 基于实际 `PriorToolExecution` 的稳定答案格式化。
- Case 文件和数据集身份校验、最小输入投影与反泄漏测试。
- 100 Cases 顺序执行、单 Case 失败隔离、Run 级停止。
- 四类正式 Artifact 的 Schema、canonical 序列化、哈希和原子发布。
- CLI、Makefile、README、`PROJECT_STATUS.md` 与真实运行证据。

### 本计划不包含

- 修改 Phase 1 冻结数据、Cases、Gold 或工具契约。
- 修改 Phase 2 的工具语义、Runner 协议或 OpenAI-compatible Adapter。
- `query_marketing` 路由、真实 LLM、V2、自检、重试、评分、错误归因和统计。
- 针对 Case ID、Case 顺序、Case 原句或 Gold 数值的特例规则。

若实现需要上述内容，停止当前任务并记录为后续 Phase，不得顺手扩展。

## 文件结构

### 创建

```text
app/baselines/__init__.py
app/baselines/v1/__init__.py
app/baselines/v1/answers.py
app/baselines/v1/config.py
app/baselines/v1/parsing.py
app/baselines/v1/policy.py
app/baselines/v1/schemas.py
app/experiments/artifacts.py
app/experiments/baseline_v1.py
app/experiments/cases.py
tests/test_baseline_v1_answers.py
tests/test_baseline_v1_artifacts.py
tests/test_baseline_v1_batch.py
tests/test_baseline_v1_cases.py
tests/test_baseline_v1_parsing.py
tests/test_baseline_v1_policy.py
tests/test_phase3_baseline_cli.py
```

### 修改

```text
Makefile
README.md
PROJECT_STATUS.md
pyproject.toml
```

### 明确禁止修改

```text
app/data/case_generator.py
app/data/generator.py
app/data/gold.py
app/agents/runner.py
app/tools/metrics.py
app/tools/queries.py
configs/data/synthetic_v1.yaml
configs/data/synthetic_public_validation_v1.yaml
configs/evaluation/tool_contract_v1.yaml
data/evaluation_cases/v1/cases.jsonl
data/evaluation_cases/v1/manifest.json
data/synthetic/phase1_sha256_baseline.json
sql/gold/*.sql
```

`app/llm/deterministic.py` 默认不修改；若实现发现必须修改，先证明现有 Adapter 协议无法满足，
单独提交设计变更，不与 Phase 3 业务逻辑混合。

## 公共类型冻结

`app/baselines/v1/schemas.py` 使用以下核心类型，实施中不得另造同义模型：

```python
class TaskKind(StrEnum):
    PRODUCT_ANOMALY = "product_anomaly"
    CONVERSION_DECLINE = "conversion_decline"
    NEXT_WEEK_PRIORITY = "next_week_priority"
    PRODUCTS_TO_WATCH = "products_to_watch"
    GMV_DIAGNOSIS = "gmv_diagnosis"
    UNSUPPORTED = "unsupported"


class PolicyContext(BaseModel):
    dataset_id: str
    dataset_version: str
    as_of_date: date


class DateWindows(BaseModel):
    start_date: date
    end_date: date
    comparison_start_date: date
    comparison_end_date: date


class ParsedRequest(BaseModel):
    task: TaskKind
    product_ids: tuple[ProductId, ...]
    windows: DateWindows
    unsupported_reason: str | None = None
```

`ParsedRequest.product_ids` 使用 `ProductId` tuple 保持深层不可变；解析层始终返回 tuple，
仅在组装现有工具 Schema 的参数边界时转换为 list。

`app/experiments/cases.py` 的执行边界固定为：

```python
class RunnableCase(BaseModel):
    case_id: str
    user_input: str
    dataset_id: str
    dataset_version: str
    split: Literal["development", "public_validation"]
```

Artifact 模型固定使用 `extra="forbid"`、`allow_inf_nan=False` 和 strict revalidation。
`CaseRunRecord` 不含业务任务、能力、难度、cluster、Gold、expected、评分或耗时字段。

## Task 1：冻结策略配置与类型

**Files:**
- Create: `app/baselines/__init__.py`
- Create: `app/baselines/v1/__init__.py`
- Create: `app/baselines/v1/schemas.py`
- Create: `app/baselines/v1/config.py`
- Test: `tests/test_baseline_v1_policy.py`

- [ ] **Step 1: 写策略快照失败测试**

在 `tests/test_baseline_v1_policy.py` 断言：

```python
snapshot = policy_snapshot()
assert snapshot["policy_name"] == "deterministic-baseline-v1"
assert snapshot["policy_version"] == "1.0.0"
assert snapshot["default_window_days"] == 30
assert snapshot["tool_paths"]["gmv_diagnosis"] == [
    "query_sales",
    "calculate_metrics",
]
serialized = canonical_policy_bytes(snapshot).decode("utf-8").lower()
for forbidden in ("case_001", "gold_", "expected_tool_calls", "success_criteria"):
    assert forbidden not in serialized
```

同时断言五类路径均不包含 `query_marketing`，所有指标名可由 `MetricName` 严格解析。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python3 -m pytest tests/test_baseline_v1_policy.py -q
```

Expected: FAIL，原因是 `app.baselines.v1` 尚不存在。

- [ ] **Step 3: 实现最小类型和常量配置**

在 `schemas.py` 落地 `TaskKind`、`PolicyContext`、`DateWindows`、`ParsedRequest`、
`ToolStep`、`ExecutionPlan`。在 `config.py` 只定义：

- 路由关键词与优先级。
- `DEFAULT_WINDOW_DAYS = 30`。
- `DEFAULT_TOP_K = 5`。
- 五类工具路径和指标集合。
- 稳定排序字段与 `ANSWER_TEMPLATE_VERSION = "1.0"`。
- `policy_snapshot()` 和 `canonical_policy_bytes()`。

配置必须由不可变 tuple/mapping 组合，snapshot 返回深拷贝 JSON 对象。

- [ ] **Step 4: 导出公开类型并运行测试**

Run:

```bash
python3 -m pytest tests/test_baseline_v1_policy.py -q
python3 -m ruff check app/baselines tests/test_baseline_v1_policy.py
```

Expected: PASS，Ruff 输出 `All checks passed!`。

- [ ] **Step 5: 提交原子变更**

```bash
git add app/baselines tests/test_baseline_v1_policy.py
git commit -m "feat(baseline): freeze deterministic v1 policy contract"
```

## Task 2：实现通用输入解析

**Files:**
- Modify: `app/baselines/v1/config.py`
- Create: `app/baselines/v1/parsing.py`
- Test: `tests/test_baseline_v1_parsing.py`

- [ ] **Step 1: 写默认窗口和商品解析失败测试**

覆盖：

```python
context = PolicyContext(
    dataset_id="e1e81533c25e03e5",
    dataset_version="v1",
    as_of_date=date(2026, 4, 30),
)
parsed = parse_request("分析最近30天 P003 的转化下降", context)
assert parsed.task == TaskKind.CONVERSION_DECLINE
assert parsed.product_ids == ("P003",)
assert parsed.windows.model_dump(mode="json") == {
    "start_date": "2026-04-01",
    "end_date": "2026-04-30",
    "comparison_start_date": "2026-03-02",
    "comparison_end_date": "2026-03-31",
}
```

再覆盖重复商品去重、大小写规范化、未指定商品、未指定时间、非法日期、冲突日期和无任务信号。

- [ ] **Step 2: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_parsing.py -q
```

Expected: FAIL，原因是 `parse_request` 尚不存在。

- [ ] **Step 3: 实现纯函数解析器**

实现：

- `_extract_product_ids(text) -> tuple[ProductId, ...]`，只接受 `P[0-9]{3}`，去重后保持首次出现顺序。
- `_default_windows(as_of_date, days=30) -> DateWindows`。
- `_extract_windows(text, as_of_date) -> DateWindows`，只支持设计文档列出的表达。
- `_classify_task(text) -> TaskKind`，严格按冻结优先级。
- `parse_request(text, context) -> ParsedRequest`，其中 `product_ids` 保持 tuple，不在解析层转回 list。

不读取文件、环境变量、Case 元数据或数据库；不导入 `app.data.gold`。

- [ ] **Step 4: 加入表驱动路由测试**

每类至少三个通用改写，另加一个多信号优先级和一个 adversarial distractor。测试只验证通用概念，
不得复制 100 个 Cases 做逐句白名单。

- [ ] **Step 5: 运行定向测试与提交**

```bash
python3 -m pytest tests/test_baseline_v1_parsing.py -q
python3 -m ruff check app/baselines/v1/config.py app/baselines/v1/parsing.py tests/test_baseline_v1_parsing.py
git add app/baselines/v1/config.py app/baselines/v1/parsing.py tests/test_baseline_v1_parsing.py
git commit -m "feat(baseline): parse v1 task and query windows"
```

Expected: tests PASS，Ruff PASS。

## Task 3：实现固定工具计划

**Files:**
- Modify: `app/baselines/v1/policy.py`
- Modify: `app/baselines/v1/__init__.py`
- Modify: `tests/test_baseline_v1_policy.py`

- [ ] **Step 1: 写五类首轮动作失败测试**

构造 `AdapterRequest(user_input=..., tools=..., prior_tool_executions=[])`，断言每类首个动作：

```text
gmv_diagnosis       -> query_sales
product_anomaly     -> query_product（有商品 ID）或 query_sales（无商品 ID）
conversion_decline  -> query_traffic
products_to_watch   -> query_sales
next_week_priority  -> query_sales
```

断言窗口参数完全一致，sales 的 `include_refunds` 为 `True`，traffic 的
`include_missing` 为 `True`。

- [ ] **Step 2: 写完整路径状态机失败测试**

使用最小合法 `PriorToolExecution` 序列逐步调用 Policy，断言下一动作名称、唯一且稳定的
`call_id`、`calculate_metrics` 指标和 `group_by`。额外断言：

- 未识别任务直接返回能力限制回答且无工具调用。
- 意外工具序列抛出稳定 Policy 协议错误。
- Policy 不从 `request.tools` 猜测或扩展工具路径。

- [ ] **Step 3: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_policy.py -q
```

Expected: FAIL，原因是 `BaselinePolicyV1` 尚不存在。

- [ ] **Step 4: 实现无重试状态机**

`BaselinePolicyV1(context)` 在首次调用缓存 `ParsedRequest` 和 `ExecutionPlan`。后续步骤仅比较
实际 `execution.result.tool_name` 序列与计划前缀：

```python
if actual_names != expected_names[: len(actual_names)]:
    raise BaselinePolicyProtocolError("unexpected prior tool sequence")
if len(actual_names) < len(expected_names):
    return action_for(expected_names[len(actual_names)])
return render_final_answer(parsed, request.prior_tool_executions)
```

每个 Case 由 Batch Runner 创建新实例，禁止复用跨 Case 状态。

- [ ] **Step 5: 运行测试和现有 Runner 回归**

```bash
python3 -m pytest tests/test_baseline_v1_policy.py tests/test_agent_runner.py -q
python3 -m ruff check app/baselines tests/test_baseline_v1_policy.py
```

Expected: PASS；现有 Runner 行为无变化。

- [ ] **Step 6: 提交原子变更**

```bash
git add app/baselines/v1/policy.py app/baselines/v1/__init__.py tests/test_baseline_v1_policy.py
git commit -m "feat(baseline): add deterministic v1 tool planner"
```

## Task 4：实现证据化回答

**Files:**
- Create: `app/baselines/v1/answers.py`
- Test: `tests/test_baseline_v1_answers.py`
- Modify: `app/baselines/v1/policy.py`

- [ ] **Step 1: 写答案来源失败测试**

为五类任务构造已验证的工具结果，断言：

- 回答中的每个数字都能在传入的 Tool Result 中找到对应原始值或确定性格式化值。
- 修改 Tool Result 数值会同步修改回答，证明没有硬编码 Gold。
- 商品名称只来自 `query_product`；没有名称时使用 `product_id`。
- `None`、空行和 warnings 生成“数据不足”限制，不生成虚假结论。
- 回答不包含 `gold`、`expected`、生成配置或因果断言。

- [ ] **Step 2: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_answers.py -q
```

Expected: FAIL，原因是答案渲染函数尚不存在。

- [ ] **Step 3: 实现稳定格式化和排序**

实现：

- `format_money`、`format_count`、`format_rate`。
- `tool_results_by_name`，拒绝重复来源。
- 五类模板渲染器。
- 商品任务使用冻结排序键并截取 `DEFAULT_TOP_K`。
- `render_final_answer(parsed, executions)`。

不得读取 Catalog、文件、配置 YAML 或原始 Case。

- [ ] **Step 4: 接入 Policy 并运行测试**

```bash
python3 -m pytest \
  tests/test_baseline_v1_answers.py \
  tests/test_baseline_v1_policy.py \
  tests/test_agent_runner.py -q
python3 -m ruff check app/baselines tests/test_baseline_v1_answers.py
```

Expected: PASS。

- [ ] **Step 5: 提交原子变更**

```bash
git add app/baselines/v1/answers.py app/baselines/v1/policy.py tests/test_baseline_v1_answers.py
git commit -m "feat(baseline): render answers from tool evidence"
```

## Task 5：建立 Case 输入隔离

**Files:**
- Create: `app/experiments/cases.py`
- Test: `tests/test_baseline_v1_cases.py`

- [ ] **Step 1: 写身份与投影失败测试**

使用临时 JSONL/Manifest 覆盖：

- 正确 SHA-256、数量、Split、Case Set ID、唯一 `case_id`。
- SHA、行数、Split、数据集映射或重复 ID 不符时拒绝。
- 输入行可含 Gold 和 expected 字段，但输出对象 `model_dump()` 只含五个允许字段。
- 保持 JSONL 原始顺序。

- [ ] **Step 2: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_cases.py -q
```

Expected: FAIL，原因是 Case Loader 尚不存在。

- [ ] **Step 3: 实现严格加载和最小投影**

实现 `EvaluationCaseManifest`、`RunnableCase`、`CaseBundle` 和：

```python
def load_runnable_cases(case_dir: Path) -> CaseBundle:
    ...
```

完整 JSON 对象只在函数局部用于校验。返回前立即构造 `RunnableCase`，不保存原始字典或未知字段。
Case Set ID 复用 Phase 1 已有 canonical 算法；若没有公开 helper，则在本模块实现并用冻结 Manifest
做回归，不导入 Gold。

- [ ] **Step 4: 加入仓库冻结 Cases 回归**

断言真实文件：

```text
case_set_id = ecebfe8b691271fd
jsonl_sha256 = f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03
count = 100
split = 70 / 30
```

- [ ] **Step 5: 运行测试与提交**

```bash
python3 -m pytest tests/test_baseline_v1_cases.py tests/test_phase1_frozen_assets.py -q
python3 -m ruff check app/experiments/cases.py tests/test_baseline_v1_cases.py
git add app/experiments/cases.py tests/test_baseline_v1_cases.py
git commit -m "feat(experiments): isolate runnable case inputs"
```

Expected: PASS，Phase 1 冻结资产无漂移。

## Task 6：实现 Artifact Schema 与原子发布

**Files:**
- Create: `app/experiments/artifacts.py`
- Test: `tests/test_baseline_v1_artifacts.py`

- [ ] **Step 1: 写 Schema 和序列化失败测试**

定义测试夹具并断言：

- `CaseRunRecord` 拒绝 Gold、expected、评分、耗时和未知字段。
- `Summary` 只允许状态计数、Split 分布、顺序边界和评测状态。
- JSON 无 NaN/Infinity，键顺序与换行稳定。
- `case_runs.jsonl` 两次序列化逐字节一致。

- [ ] **Step 2: 写原子发布失败测试**

覆盖：

- 目标目录存在时不覆盖。
- 写入中异常不产生正式目录并清理临时目录。
- 行数或 Case 顺序错误阻止发布。
- 发布后重新读取四文件并通过 Pydantic 校验。
- Manifest 记录其余三个文件的 SHA-256，不自引用。

- [ ] **Step 3: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_artifacts.py -q
```

Expected: FAIL，原因是 Artifact 模块尚不存在。

- [ ] **Step 4: 实现 canonical writer**

实现：

```python
def canonical_json_bytes(model_or_mapping: object) -> bytes: ...
def canonical_jsonl_bytes(records: Sequence[CaseRunRecord]) -> bytes: ...
class ArtifactWriter:
    def publish(self, bundle: RunArtifactBundle) -> Path: ...
```

临时目录命名为 `.<run_id>.tmp-<random>`，必须位于输出根目录内。校验通过后使用
`Path.replace(final_dir)`；任何异常在 `finally` 中删除临时目录。

- [ ] **Step 5: 运行测试与提交**

```bash
python3 -m pytest tests/test_baseline_v1_artifacts.py -q
python3 -m ruff check app/experiments/artifacts.py tests/test_baseline_v1_artifacts.py
git add app/experiments/artifacts.py tests/test_baseline_v1_artifacts.py
git commit -m "feat(experiments): publish immutable baseline artifacts"
```

Expected: PASS。

## Task 7：实现批量 Runner

**Files:**
- Create: `app/experiments/baseline_v1.py`
- Test: `tests/test_baseline_v1_batch.py`

- [ ] **Step 1: 写双数据集路由失败测试**

用两个临时 Phase 1 快照和四个 `RunnableCase` 断言：

- `dataset_id` 选择正确 Catalog。
- 输出顺序与 Case 输入顺序一致。
- 每个 Case 创建独立 Policy/Adapter/Runner。
- `usage is None`。
- Development 与 Public Validation 计数正确。

- [ ] **Step 2: 写故障边界失败测试**

覆盖：

- 一个 Case 的 Runner 返回 `failed` 后下一 Case 仍执行。
- Dataset ID 不匹配、Catalog 打不开或策略哈希漂移时抛 `RunLevelError`。
- Run 级错误不会调用正式发布。
- `as_of_date` 从已验证 Catalog 的事实表最大日期确定，不读取生成配置。

`as_of_date` 查询固定为只读 SELECT，取 `orders.order_date`、`traffic.day`、
`marketing.day` 三者最大值，并断言三表最大日期相同；不一致时作为 Run 级失败。

- [ ] **Step 3: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_baseline_v1_batch.py -q
```

Expected: FAIL，原因是 Batch Runner 尚不存在。

- [ ] **Step 4: 实现批处理核心**

实现：

```python
class BaselineBatchRunner:
    def run(
        self,
        cases: CaseBundle,
        dataset_dirs: Mapping[str, Path],
        output_root: Path,
    ) -> Path:
        ...
```

运行前计算 policy snapshot 与源码组合哈希；运行后重新计算并比较。单 Case 捕获异常时只写稳定
`error_code`，不得写异常文本、绝对路径或 traceback。

- [ ] **Step 5: 实现 summary 和 manifest**

Manifest 采集：

- UTC 时间、Run ID、Git commit 和 dirty 布尔值。
- Python、Pydantic、DuckDB、Pandas、PyArrow 版本。
- Case、Dataset、工具契约和策略身份。
- `evaluation_status="pending_not_run"`、`usage_status="unavailable"`。

不得把未跟踪目录内容写入 Artifact；当前已有用户无关未跟踪文件不应被删除或纳入提交。

- [ ] **Step 6: 运行批处理测试与提交**

```bash
python3 -m pytest \
  tests/test_baseline_v1_batch.py \
  tests/test_baseline_v1_artifacts.py \
  tests/test_baseline_v1_cases.py -q
python3 -m ruff check app/experiments tests/test_baseline_v1_batch.py
git add app/experiments/baseline_v1.py tests/test_baseline_v1_batch.py
git commit -m "feat(experiments): run baseline cases across datasets"
```

Expected: PASS。

## Task 8：增加 CLI 与项目入口

**Files:**
- Modify: `app/experiments/baseline_v1.py`
- Create: `tests/test_phase3_baseline_cli.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `.gitignore` only if `outputs/experiment_runs/` is not already governed

- [ ] **Step 1: 写 CLI 失败测试**

通过 `main([...])` 覆盖：

- 默认 Cases：`data/evaluation_cases/v1`。
- 默认数据集：`data/synthetic/v1` 与 `data/synthetic/public-validation-v1`。
- 默认输出：`outputs/experiment_runs`。
- 成功时 stdout 只有 JSON 摘要且退出码 `0`。
- Run 级错误时 stderr 只有安全错误码且退出码 `1`。
- 不存在的输入不会创建输出目录。

- [ ] **Step 2: 运行测试并确认失败**

```bash
python3 -m pytest tests/test_phase3_baseline_cli.py -q
```

Expected: FAIL，原因是 CLI 参数或入口尚未实现。

- [ ] **Step 3: 实现 CLI**

参数固定为：

```text
--case-dir
--development-dataset
--public-validation-dataset
--output-root
```

不提供 `--case-id`、`--task`、`--expected-tools`、`--gold` 或跳过校验的参数。

- [ ] **Step 4: 注册入口和 Make 目标**

`pyproject.toml`：

```toml
phase3-baseline = "app.experiments.baseline_v1:main"
```

`Makefile` 新增：

```make
PHASE3_CASE_DIR ?= data/evaluation_cases/v1
PHASE3_DEVELOPMENT_DATASET ?= data/synthetic/v1
PHASE3_PUBLIC_VALIDATION_DATASET ?= data/synthetic/public-validation-v1
PHASE3_OUTPUT_ROOT ?= outputs/experiment_runs

phase3-baseline: check-python
	@$(PYTHON) -m app.experiments.baseline_v1 \
		--case-dir "$(PHASE3_CASE_DIR)" \
		--development-dataset "$(PHASE3_DEVELOPMENT_DATASET)" \
		--public-validation-dataset "$(PHASE3_PUBLIC_VALIDATION_DATASET)" \
		--output-root "$(PHASE3_OUTPUT_ROOT)"
```

- [ ] **Step 5: 运行入口测试和提交**

```bash
python3 -m pytest tests/test_phase3_baseline_cli.py -q
python3 -m ruff check app tests
git add app/experiments/baseline_v1.py tests/test_phase3_baseline_cli.py pyproject.toml Makefile .gitignore
git commit -m "feat(cli): add phase3 baseline command"
```

Expected: PASS。若 `.gitignore` 无修改，禁止为了匹配命令而空改文件，也不要加入 `git add`。

## Task 9：建立端到端确定性与防泄漏门禁

**Files:**
- Modify: `tests/test_baseline_v1_batch.py`
- Modify: `tests/test_phase3_baseline_cli.py`
- Modify: `tests/test_phase1_frozen_assets.py` only to add Phase 3 read-only assertions if necessary

- [ ] **Step 1: 写端到端双运行测试**

在临时目录生成两个数据快照和小型 Case Bundle，运行两次不同 Run ID，断言：

- `case_runs.jsonl` 逐字节一致。
- `summary.json` 除发布目录无关字段外逐字节一致；推荐让 summary 完全不含 Run ID 和时间。
- `policy_snapshot.json` 逐字节一致。
- `run_manifest.json` 仅允许 Run ID、UTC 时间和输出哈希产生预期差异。

- [ ] **Step 2: 写反泄漏扫描**

扫描三个稳定 Artifact：

```python
for forbidden in (
    '"business_task"',
    '"primary_capability"',
    '"expected_tool_calls"',
    '"gold_evidence"',
    '"gold_metrics"',
    '"reference_answer"',
    '"success_criteria"',
    '"score"',
    '"accuracy"',
):
    assert forbidden not in artifact_text.lower()
```

另断言 `case_runs.jsonl` 的每行 Case ID、user input、Dataset locator 与源 Case 一致，但没有其他
源 Case 字段。

- [ ] **Step 3: 写 100 Cases dry-run 集成门禁**

测试使用真实冻结 Cases 和临时重建数据集，断言：

- 恰好 100 条记录，顺序等于源 JSONL。
- Development/Public Validation 为 `70/30`。
- 每条 `usage=null`。
- `summary.evaluation_status == "pending_not_run"`。
- 四文件均通过重新读取和哈希验证。

这是测试输出，只写 `.tmp/` 或 pytest 临时目录，不写正式输出根目录。

- [ ] **Step 4: 运行 Phase 3 定向测试**

```bash
python3 -m pytest \
  tests/test_baseline_v1_parsing.py \
  tests/test_baseline_v1_policy.py \
  tests/test_baseline_v1_answers.py \
  tests/test_baseline_v1_cases.py \
  tests/test_baseline_v1_artifacts.py \
  tests/test_baseline_v1_batch.py \
  tests/test_phase3_baseline_cli.py -q
```

Expected: PASS；记录真实数量和耗时，不预填结果。

- [ ] **Step 5: 运行 Phase 1/2 回归**

```bash
python3 -m pytest \
  tests/test_phase1_frozen_assets.py \
  tests/test_tool_queries.py \
  tests/test_tool_metrics.py \
  tests/test_tool_registry.py \
  tests/test_llm_deterministic.py \
  tests/test_agent_runner.py \
  tests/test_phase2_smoke.py -q
```

Expected: PASS，冻结资产、工具和 Runner 无回归。

- [ ] **Step 6: 提交测试门禁**

```bash
git add tests/test_baseline_v1_batch.py tests/test_phase3_baseline_cli.py tests/test_phase1_frozen_assets.py
git commit -m "test(phase3): enforce deterministic leak-free baseline runs"
```

仅提交实际修改的文件。

## Task 10：执行正式 100 Cases Baseline

**Files:**
- Create at runtime: `outputs/experiment_runs/baseline-v1__<UTC时间>__<run短哈希>/`

- [ ] **Step 1: 确认工作树边界**

```bash
git status --short
git diff --check
```

Expected: 已知用户无关未跟踪文件保持原样；Phase 3 实现文件已提交；没有未提交的实现改动。
不得删除、移动或提交既有 `书籍/`。

- [ ] **Step 2: 重建并验证两套冻结数据**

```bash
make PYTHON=python3 phase1
python3 -m pytest tests/test_phase1_frozen_assets.py -q
```

Expected: Phase 1 构建成功且冻结门禁 PASS。若本机项目 Python 不是 `python3`，使用实际已验证的
Python 3.11+ 可执行文件并把命令如实记录。

- [ ] **Step 3: 计算运行前身份**

记录：

```bash
git rev-parse HEAD
shasum -a 256 data/evaluation_cases/v1/cases.jsonl
shasum -a 256 data/evaluation_cases/v1/manifest.json
shasum -a 256 configs/evaluation/tool_contract_v1.yaml
```

Expected: Cases SHA 为
`f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03`；
其他值由 CLI 写入 Run Manifest，不手工伪造。

- [ ] **Step 4: 执行正式命令一次**

```bash
make PYTHON=python3 phase3-baseline
```

Expected: 退出码 `0`，stdout 返回一个新 Run ID、正式目录和真实 completed/failed 计数。不得因
Case 失败而修改 Policy 后覆盖本 Run；需要改策略时创建新版本和新 Run。

- [ ] **Step 5: 验证正式 Artifact**

运行只读验证命令：

```bash
python3 -m pytest tests/test_phase3_baseline_cli.py -q
python3 -c 'from pathlib import Path; from app.experiments.artifacts import verify_published_run; paths = sorted(Path("outputs/experiment_runs").glob("baseline-v1__*")); assert paths; verify_published_run(paths[-1])'
```

`app/experiments/artifacts.py` 必须公开 `verify_published_run(path)`；不得人工目测代替 Schema、
行数、顺序和哈希校验。最终记录真实 Run ID、100 条 Case 数、70/30 Split、
completed/failed 和四文件 SHA。

- [ ] **Step 6: 检查无评分结论**

对正式 Artifact 执行反泄漏和无评分测试，确认：

```text
evaluation_status = pending_not_run
usage_status = unavailable
case_runs count = 100
score/accuracy/task_success/hallucination/significance 不存在
```

- [ ] **Step 7: 不提交正式运行产物前先核对仓库规则**

如果 `outputs/experiment_runs` 设计为作品集可追溯证据并应纳入 Git，则只提交本次正式 Run 的四个
文件；如果仓库既有约定忽略运行产物，则保留在工作区并在状态文档记录路径与哈希。不得临时改变
治理规则来隐藏失败结果。

## Task 11：更新文档与状态

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `docs/superpowers/specs/2026-10-08-phase-3-deterministic-baseline-v1-design.md` only if implementation revealed an approved contract correction

- [ ] **Step 1: 更新 README**

增加：

- Deterministic V1 的定位、限制和输入隔离。
- `make phase3-baseline` 使用方法。
- Artifact 四文件说明。
- “Baseline 完成不等于评分完成”；Real LLM、V2、评分仍为 Pending。

- [ ] **Step 2: 更新项目状态**

把 Current Phase 更新为 Phase 3 的真实状态。只记录实际观察到的：

- 测试命令与通过数量。
- Ruff 结果。
- 正式 Run ID 和 Artifact 相对路径。
- completed/failed、70/30 分布和文件哈希。
- 任何 Case 失败的稳定错误码分布。

不得写准确率、提升率、延迟、Token 或成本；没有 Usage 时保持 `Unavailable`。

- [ ] **Step 3: 运行文档一致性搜索**

用专用搜索工具或项目允许的等价命令确认 README、状态和 Artifact 中没有把 Public Validation
称为 Holdout，也没有把 Baseline 运行误称为评测完成。

- [ ] **Step 4: 提交文档**

```bash
git add README.md PROJECT_STATUS.md docs/superpowers/specs/2026-10-08-phase-3-deterministic-baseline-v1-design.md
git commit -m "docs(phase3): record deterministic baseline run"
```

若设计文档未改，不加入该路径。

## Task 12：最终验证与交付

**Files:**
- Verify only

- [ ] **Step 1: 运行全量测试**

```bash
python3 -m pytest
```

Expected: 全部 PASS；记录真实数量和 warnings，不能复用旧的 `533 passed`。

- [ ] **Step 2: 运行静态检查**

```bash
python3 -m ruff check app tests notebooks
```

Expected: `All checks passed!`。

- [ ] **Step 3: 验证 Phase 2 smoke**

```bash
make PYTHON=python3 phase2-smoke
```

Expected: `status=completed` 且固定工具链仍为
`query_sales → calculate_metrics`。

- [ ] **Step 4: 验证 Git 与产物边界**

```bash
git diff --check
git status --short
git log -12 --oneline
```

Expected: 无未提交的 Phase 3 实现或文档改动；用户原有无关文件保持未跟踪且未被修改。正式
Artifact 是否跟踪必须与 Task 10 的仓库治理决定一致。

- [ ] **Step 5: 最终报告**

按以下格式汇报：

```text
Position: Phase 3 / 完成或明确阻塞点
Done: 最多 3 条
Blocker: 无或具体错误
Validation: 测试、Ruff、Run ID、100 Cases、completed/failed、Artifact 哈希
Next: Phase 4 V2 设计；评分仍留在 Phase 5
```

不得仅凭实现完成宣称 Baseline 已运行；只有 Task 10 成功发布正式 Artifact 后，才能把
Baseline Experiment 标记为 `Completed / Unscored`。

## 计划自审

- [ ] 设计范围覆盖：策略、输入隔离、双数据集、批处理、Artifact、失败边界和正式运行均有任务。
- [ ] TDD 完整：每个生产模块先写失败测试，再做最小实现，再运行回归。
- [ ] 无占位符：每一步均给出具体接口、命令和预期结果，不使用待补充标记。
- [ ] 类型一致：复用 Phase 2 的 `AdapterRequest`、`AssistantAction`、
  `PriorToolExecution`、`AgentRunner` 和现有 Tool Schema。
- [ ] 冻结资产安全：明确禁止修改 Phase 1 数据、Cases、Gold 与工具契约。
- [ ] 防泄漏：Policy 永远不接收完整 Case；Artifact 不包含标签、Gold、expected 或评分字段。
- [ ] 结果诚实：正式运行前不写指标；运行后只记录实际状态，不产生评分或效果结论。
- [ ] 原子提交：策略、解析、回答、Case Loader、Artifact、Batch、CLI、门禁、文档各自提交。
