# Phase 3 Deterministic Baseline V1 Design

## 目标

Phase 3 建立一个可复现、无网络、无真实 LLM 成本的正式 Baseline：用冻结的
`BaselinePolicyV1` 依次运行 `data/evaluation_cases/v1/cases.jsonl` 中全部 100 个 Cases，
记录逐 Case 的公开决策轨迹和完整工具结果，并以不可覆盖的 Run Artifact 原子发布。

本阶段只回答“Baseline 实际执行了什么”。不读取 Gold，不评分，不生成准确率、成功率、提升率、
显著性、延迟或成本结论。

## 范围

### 包含

- 通用业务概念驱动的浅层文本解析和五类单任务路由。
- 固定、短、可审计的工具链。
- Development 与 Public Validation 两套数据集的按 Case 路由。
- Case 输入投影和防评测泄漏边界。
- 100 Cases 的顺序批量执行、单 Case 故障隔离和运行级失败处理。
- `run_manifest.json`、`case_runs.jsonl`、`summary.json`、
  `policy_snapshot.json` 四类不可变 Artifact。
- 输入、策略、源码、输出的 SHA-256 身份记录。
- 无评分语义的运行汇总和项目状态更新。

### 不包含

- 真实 LLM 调用或 Prompt 实验。
- Optimized V2、自检、反思、候选规划、参数修复或回答后校验。
- 读取 `business_task`、`primary_capability`、`expected_tool_calls`、
  `gold_*`、`reference_answer`、`success_criteria` 或评分规则来驱动 Agent。
- Phase 5 评分、Phase 6 错误归因或 Phase 7 统计推断。
- 修改冻结 Cases、Gold、Synthetic Data 或工具契约。
- 针对具体 Case 文案、Case ID 或顺序编写专用规则。

## 设计原则

1. **输入最小化**：运行时只向 Agent 传递 `user_input`；批处理层只额外保留
   `case_id`、`dataset_id`、`dataset_version` 和公开 Split 以完成调度与归档。
2. **确定性**：相同 Cases、数据集、策略快照和源码必须产生逐字节一致的
   `case_runs.jsonl` 与逻辑一致的汇总；时间戳和 Run ID 仅存在于运行清单。
3. **证据约束**：最终回答只能读取当前 Run 的 `PriorToolExecution`，不能查询数据库、
   读取 Case Gold 或补写未经工具返回支持的事实。
4. **失败显式化**：无法识别任务、非法时间、空数据或工具错误均形成明确状态和原因，
   不猜测、不静默回退。
5. **结果不可变**：正式目录不覆盖；临时目录通过完整性校验后一次性重命名发布。
6. **Baseline 克制**：V1 故意不具备 V2 的修复和反思能力，保留可解释的改进空间。

## 系统边界

```text
cases.jsonl
  │
  ├─ CaseLoader：验证文件身份，逐行严格解析
  │      只投影 case_id / user_input / dataset locator / split
  │
  ├─ BaselineBatchRunner：按 JSONL 顺序选择已验证 Catalog
  │
  ├─ DeterministicAdapter(BaselinePolicyV1)
  │
  ├─ AgentRunner：执行单工具动作并记录 decision_trace
  │
  ├─ ToolRegistry：调用现有受控工具
  │
  └─ ArtifactWriter：临时写入、校验、哈希、原子发布
```

`BaselinePolicyV1` 不接收完整 Evaluation Case。它只实现
`Callable[[AdapterRequest], AssistantAction]`，因此天然无法看到 Case 标签和 Gold。
`BaselineBatchRunner` 持有最小 `RunnableCase`，只负责选择正确数据集、调用 Runner 和封装结果。

## 模块设计

### `app/baselines/v1/config.py`

定义唯一的版本化策略配置和 canonical snapshot。配置包含：

- `policy_name = "deterministic-baseline-v1"`
- `policy_version = "1.0.0"`
- 通用关键词集合和路由优先级。
- 默认窗口长度 `30` 天。
- 默认 `top_k`。
- 五类任务的工具路径、指标集合和回答模板版本。
- 未识别与数据不足的稳定响应语义。

配置不得包含 `CASE_###`、Evaluation Case 原文、标签、Gold 数值或预期工具参数。

### `app/baselines/v1/parsing.py`

只解析用户自然语言中的：

- 合法的 `P[0-9]{3}` 商品 ID，去重并保持首次出现顺序。
- 明确日期或“最近 N 天 / 前 N 天”等支持范围内的窗口表达。
- 通用业务信号，如 GMV/营收、商品异常、转化、关注商品、下周重点。

若未给日期，使用当前数据集 Manifest 的 `as_of_date` 形成最近 30 天和此前 30 天。
`_extract_product_ids` 返回 `tuple[ProductId, ...]`，`ParsedRequest.product_ids` 也保持该
tuple 契约，以确保策略状态深层不可变。若未给商品，使用空 tuple 表示全商品。超出数据边界、
窗口逆序或冲突表达返回结构化 `unsupported`，不自动修复。

### `app/baselines/v1/policy.py`

`BaselinePolicyV1` 在构造时只接收该 Case 对应数据集的公开运行上下文（Dataset ID、版本和
`as_of_date`），每个 Case 新建一个 Policy 与 Deterministic Adapter。首次 `complete()` 根据
`user_input` 解析单一主任务并创建固定工具链；后续调用只依据
`prior_tool_executions` 推进到下一工具或生成最终回答。

为保持现有 `DeterministicAdapter` 和 `AgentRunner` 的职责稳定，V1 可作为专用
Deterministic Adapter 使用，但其公开协议仍满足 `LLMAdapter`。策略状态只保存当前 Run 的
解析计划，不跨 Case 共享。

### `app/experiments/cases.py`

严格验证：

- Case 文件 SHA-256 等于 Manifest 的 `jsonl_sha256`。
- Case 数量、Split 数量、Case Set ID 与 Manifest 一致。
- 每个 Case 的 `dataset_id` 和 `dataset_version` 能映射到 Manifest 声明的数据集。
- `case_id` 唯一，JSONL 顺序保持不变。

解析完整 JSON 只用于边界验证，输出为 `RunnableCase`：

```python
class RunnableCase(BaseModel):
    case_id: str
    user_input: str
    dataset_id: str
    dataset_version: str
    split: Literal["development", "public_validation"]
```

任何执行组件不得持有原始 Case 字典。

### `app/experiments/artifacts.py`

定义严格 Pydantic Schema、canonical JSON 序列化和原子发布：

- JSON 使用 UTF-8、稳定键序、有限数值和固定换行。
- `case_runs.jsonl` 按输入顺序写入。
- 临时目录位于目标输出根目录内，避免跨文件系统重命名。
- 正式目录已存在时立即失败。
- 全部文件写完后重新读取并验证 Schema、行数和 SHA-256，再原子重命名。
- 发布失败清理临时目录，不留下貌似正式的半成品。

### `app/experiments/baseline_v1.py`

CLI 负责：

- 接收 Case 目录、Development 数据集目录、Public Validation 数据集目录和输出根目录。
- 在运行前验证 Case 身份、两个 Dataset ID、策略快照和策略源码哈希。
- 按冻结顺序执行全部 100 Cases。
- 单 Case 的 Runner 失败写入该 Case 记录并继续。
- 输入身份错误、数据集打不开、策略哈希在运行中漂移或 Artifact 校验失败时停止整批，
  不发布正式目录。
- 成功发布后只向 stdout 输出机器可读的 Run ID、目录和 completed/failed 数量。

## 路由规则

路由只使用通用业务概念，按以下优先级选择一个主任务：

| 优先级 | 主任务 | 识别信号示例 | 固定工具链 |
|---|---|---|---|
| 1 | 商品异常 | 商品/SKU + 异常/问题/原因 | `query_product → query_sales → query_traffic → calculate_metrics` |
| 2 | 转化下降 | 转化/CVR + 下降/下滑/原因 | `query_traffic → query_sales → calculate_metrics` |
| 3 | 下周重点 | 下周/下一步 + 优先/重点/动作 | `query_sales → query_traffic → calculate_metrics` |
| 4 | 关注商品 | 关注/预警/风险 + 商品/SKU | `query_sales → query_traffic → calculate_metrics` |
| 5 | GMV 诊断 | GMV/销售额/营收/成交 + 变化/趋势/诊断 | `query_sales → calculate_metrics` |

若多个信号同时出现，以优先级最高者为单一主任务。若没有任何支持的组合，Policy 返回明确的
能力限制回答，不调用工具。

`query_marketing` 不在 V1 工具路径中。所有 Case 均使用现有 Registry，但 V1 不主动选择该工具。

## 工具参数

所有窗口查询共享同一组参数：

```text
start_date
end_date
comparison_start_date
comparison_end_date
product_ids
```

`query_sales` 固定 `include_refunds=true`，`query_traffic` 固定
`include_missing=true`。`query_product` 只在解析出至少一个商品 ID 时调用；商品异常任务未指定
商品时跳过该工具并对全商品执行其余路径。

策略和解析层始终保留 `product_ids: tuple[ProductId, ...]`；仅在构造现有工具 Schema 的
`product_ids` 参数时执行 `list(parsed.product_ids)`，不把可变 list 写回策略状态。

指标集合按主任务冻结：

- GMV 诊断：GMV、订单量、AOV 的当前值、对照值和变化率。
- 商品异常：按 `product_id` 计算 GMV、订单、退款率、流量、CTR、CVR 及变化。
- 转化下降：整体或指定商品的 visits、orders、CVR、CVR 变化和数据完整天数。
- 关注商品：按 `product_id` 计算 `evidence_value` 及支持其判断的核心指标。
- 下周重点：按 `product_id` 计算 GMV、流量、CVR 和变化，用固定排序选择 Top-K。

实际指标名称必须来自现有 `MetricName`，并满足 `calculate_metrics` 的来源要求。

## 回答规则

回答模板由任务类型决定，但数值和商品名称只能来自当前 Run 的工具结果。

- 先说明时间窗口和数据来源标签。
- 给出一条主结论。
- 列出支撑结论的核心指标；比率统一格式化，金额和计数采用稳定格式。
- 商品型任务按冻结排序规则输出最多 Top-K。
- `None`、空结果或 warnings 必须转成“数据不足/无法计算”的限制说明。
- 不出现 Gold、expected、Case 标签、生成配置、异常注入规则或隐藏推理。
- 不把相关性表述为因果关系；V1 只能说“与……同时变化”或“数据支持的可能因素”。

## Artifact 协议

正式目录：

```text
outputs/experiment_runs/
└── baseline-v1__<UTC时间>__<run短哈希>/
    ├── run_manifest.json
    ├── case_runs.jsonl
    ├── summary.json
    └── policy_snapshot.json
```

### `run_manifest.json`

记录：

- Artifact Schema 版本、Run ID、Baseline 名称和版本。
- `started_at_utc`、`completed_at_utc`。
- Git commit；工作树存在用户无关未跟踪文件时只记录状态，不纳入策略哈希。
- Python 和关键依赖版本。
- Case Set ID、Case 文件 SHA-256、Case Manifest SHA-256。
- 两个 Dataset ID、版本和 Manifest SHA-256。
- 工具契约版本与 SHA-256。
- Policy snapshot SHA-256、V1 策略源码文件集合与组合哈希。
- 四个输出文件的 SHA-256；`run_manifest.json` 自身不自引用，可记录其余三个文件。
- 评分状态固定为 `pending_not_run`，Usage 状态固定为 `unavailable`。

### `case_runs.jsonl`

每行包含：

- 顺序号、`case_id`、Split、Dataset ID/版本。
- Agent 可见的 `user_input`。
- `status`：`completed` 或 `failed`。
- `final_answer`。
- 完整 `prior_tool_executions` 和公开 `decision_trace`。
- `usage=null`。
- 失败时的稳定 `error_code`，不写堆栈、绝对路径或异常原文。

不得写入任何标签、Gold、expected 字段、评分字段或派生分数。

### `summary.json`

只汇总：

- 总数、completed、failed、stopped。
- Development/Public Validation 的数量与运行状态分布。
- Case 顺序首尾、已发布状态。
- `evaluation_status = "pending_not_run"`。

`summary.json` 不包含准确率、任务成功率、工具正确率、幻觉率、提升率或统计显著性。

### `policy_snapshot.json`

是 `config.py` 的 canonical 序列化，包含通用关键词、优先级、默认窗口、Top-K、工具路径、
指标集合、排序规则和模板版本。它必须通过反泄漏扫描，不得包含 Case ID、完整 Case 文案、
Gold 数值或评分字段名。

## 失败模型

### Case 级失败

以下失败写入当前 Case 并继续：

- Policy 无法构造合法动作。
- Runner 返回 `failed`。
- 工具输入、输出或执行失败。
- 最大步数耗尽。

### Run 级失败

以下失败停止整批且不发布正式 Artifact：

- Case 文件、Manifest 或 Dataset 身份不匹配。
- Case 数量、Split 计数或唯一性不符。
- 数据集无法安全打开。
- Policy snapshot 与源码哈希在执行前后不一致。
- 输出 Schema、行数、顺序或哈希校验失败。
- 目标正式目录已经存在。

## 测试策略

1. Policy 单元测试覆盖五类路由、优先级、日期、商品 ID、固定工具路径、答案只取工具结果和
   unsupported 分支。
2. 防泄漏测试用带诱饵的 Case 验证完整标签/Gold 不会进入 Policy、trace 或 Artifact。
3. Artifact 单元测试覆盖 canonical JSON、不可覆盖、临时目录清理、Schema 和哈希。
4. Batch 集成测试用少量合成 Cases 和两个临时快照验证顺序、数据集路由、Case 故障隔离和
   Run 级中止。
5. 正式门禁验证 100 行、70/30 Split、Case Set ID、数据集身份、四文件 Schema 和无评分字段。
6. 连续两次运行比较 `case_runs.jsonl`、`summary.json` 的确定性内容；排除 Run 时间和 Run ID。

## 完成标准

- 全量测试和 Ruff 通过，Phase 1 冻结资产无漂移。
- 正式命令真实运行全部 100 Cases，并发布一个新且不可覆盖的 Baseline Run 目录。
- `case_runs.jsonl` 恰好 100 行且顺序与冻结 Cases 一致。
- 每条记录只包含允许字段，所有最终回答均可追溯到同条记录的工具结果。
- `summary.json` 只报告运行状态，不报告评分或效果。
- `PROJECT_STATUS.md` 记录真实命令、Run ID、哈希和 completed/failed 数量；任何未执行项目继续
  标记 `Pending / Not Run` 或 `Unavailable`。
