# Real LLM Paired Evaluation Design

## 目标

本阶段在现有 100 个冻结 Evaluation Cases、两套 Synthetic E-commerce Data 快照、
Tool Registry 和通用 Agent Runner 之上，建立真实模型的 V1/V2 配对评测闭环：

```text
同一火山方舟模型与参数
→ 20 Cases Development Pilot
→ 冻结 V1/V2 Prompt 与评分协议
→ 100 Cases 配对正式运行
→ 确定性评分
→ E1-E10 错误归因与 RCA
→ Cluster-aware 配对统计
```

正式实验的唯一 Agent 自变量是 system prompt。V1 使用基础约束，V2 使用显式任务分类、
工具边界、参数检查、运行时证据绑定、数据不足协议、固定输出结构和提交前检查。两个版本必须
使用相同模型、温度、工具、数据、Cases、Runner、最大步数、超时和评分器。

本阶段不修改 Phase 3 的 Deterministic Baseline V1 或其正式 Artifact。Phase 3 继续作为
可复现离线参照；本设计中的 Real LLM V1 和 Real LLM V2 是一组新的真实模型配对实验。

## 已批准的实验配置

- Provider：火山方舟 OpenAI-compatible API。
- 模型：V1/V2 使用相同的 `LLM_MODEL` 或 Endpoint ID。
- Pilot：20 个 Development Cases，V1/V2 各运行一次。
- 正式实验：冻结的 100 个 Cases，V1/V2 各运行一次。
- 重复次数：每个 Case、每个版本 `n_repeats=1`。
- 温度：`0`。
- 最大 Agent 步数：`8`。
- 并发：`1`，避免限流和并发时序干扰。
- 自动重试：正式实验不自动重试模型请求。
- Cost：没有明确版本化价格配置时固定为 `Unavailable`。

模型仍可能在温度为 0 时产生非确定性。由于每个 Case 只运行一次，最终报告必须明确：
本实验未充分测量模型输出方差，统计结论描述的是本次冻结运行结果。

## 范围

### 包含

- OpenAI-compatible adapter 的 system prompt 注入。
- Real LLM V1/V2 Prompt、版本、SHA-256 和快照。
- 从环境变量读取火山方舟连接配置的严格 preflight。
- 20 Cases Development Pilot 和 Prompt 冻结门禁。
- V1/V2 协调式配对 Batch Runner。
- 保存模型、Prompt、Usage、Latency 和公开 decision trace 的 Artifact v2。
- 规则优先的确定性评分器和冻结 Rubric。
- E1-E10 Primary/Secondary Error、RCA 和人工复核边界。
- 按 `case_id` 配对、按 `statistical_cluster_id` 推断的统计结果。
- 项目状态、README 和真实运行证据更新。

### 不包含

- 更换或比较多个模型。
- 用 LLM Judge 替代确定性主评分。
- 自动修改 Prompt、评分规则、Gold、Cases、容差或统计方法。
- 在正式 100 Cases 结果可见后继续调优 V2。
- 生成最终盲测集。
- Dashboard、Portfolio Case Study 和自动化全链路 Pipeline。
- 并发、自动重试、反思树、候选答案投票或多 Agent 编排。

## 关键决策

### 1. 唯一自变量是 system prompt

不沿用 Deterministic `BaselinePolicyV1` 控制真实模型的工具路径。真实模型需要自行选择工具、
构造参数并决定何时回答，否则评测只是在测试硬编码 Policy，而不是 Agent 策略。

V1/V2 共用：

- 相同 `OpenAICompatibleAdapter`。
- 相同用户输入。
- 相同五个 Tool Schema 和 Tool Registry。
- 相同工具实现与数据 Catalog。
- 相同 `AgentRunner`、`max_steps=8` 和 transcript 协议。
- 相同模型、温度、超时和单次运行设置。

两者仅使用不同的版本化 system prompt。Prompt 快照进入 Artifact，并以 SHA-256 标识。

### 2. 运行时证据使用 `result_id`

V2 要求每条关键结论绑定当前工具执行产生的 `result_id`。Evaluation Case 中的
`gold_evidence.evidence_id` 只供评分器使用，绝不传给 Agent。

这一区分避免把 Gold 泄漏到 Prompt 或模型上下文：

```text
Agent-visible evidence handle: PriorToolExecution.result.result_id
Evaluator-only evidence handle: EvaluationCase.gold_evidence[].evidence_id
```

评分器负责检查回答引用的 `result_id` 是否存在，并判断相应工具结果能否支持该结论。

### 3. Pilot 只使用 Development Cases

Pilot 用于验证模型兼容性、工具调用协议、Prompt 可执行性和 Artifact 完整性，也允许基于
Development 输出修正 V1/V2 Prompt。Pilot 不使用 Public Validation Cases，不生成正式效果
结论，也不进入正式统计。

Pilot 固定使用以下 20 个 Case：

```text
CASE_003 CASE_005 CASE_013 CASE_017
CASE_022 CASE_027 CASE_029 CASE_032
CASE_041 CASE_053 CASE_055 CASE_059
CASE_063 CASE_065 CASE_076 CASE_079
CASE_087 CASE_089 CASE_091 CASE_097
```

该集合全部来自 Development，满足：

- 五个业务任务各 4 个 Case。
- 十个 primary capabilities 各 2 个 Case。
- 20 个不同 statistical clusters，每个 cluster 只取一个 Case。
- 难度分布为 easy 5、medium 8、hard 7。

Pilot Case IDs 作为版本化配置保存并写入 Pilot Manifest，不通过运行时随机抽样决定。

### 4. 正式运行采用协调式交替顺序

单独先跑完 V1 再跑 V2 会把服务时间漂移与版本效应混在一起。正式 Runner 按冻结 JSONL 顺序
遍历 Case，但对每个 Case 交替执行版本顺序：

- 奇数 `sequence`：V1 后 V2。
- 偶数 `sequence`：V2 后 V1。

两个版本各自的 `case_runs.jsonl` 仍按原始 Case 顺序写入。运行保持串行，不并发调用模型。
配对运行共享一个 `comparison_group_id`，并生成独立的 V1、V2 Run Artifact。

## Prompt 设计

### Real LLM V1

V1 是基本可用、约束较少的经营分析 Agent，不故意降质。Prompt 只包含：

- 角色：基于 Synthetic E-commerce Data 回答经营分析问题。
- 可用工具：优先使用工具获取事实，不编造工具结果。
- 基本行为：需要数据时调用工具，完成后直接回答。
- 基本输出：给出结论和关键数据。
- 能力边界：无法由工具支持时说明限制。

V1 不包含：

- 显式业务任务分类步骤。
- 各任务的工具选择边界。
- 参数和指标口径检查清单。
- 强制 `result_id` 证据引用。
- 固定回答 Schema。
- 明确的数据不足判定协议。
- 提交前自检清单。

### Real LLM V2

V2 在相同模型和工具上增加以下可解释干预：

1. 在内部选择五类支持任务之一；不要求输出隐藏推理。
2. 根据任务选择必要工具，避免无关调用。
3. 调用前检查日期窗口、对照窗口、商品范围、退款和缺失值参数。
4. 计算前检查所需来源工具是否齐全。
5. 将关键结论绑定到真实存在的 `result_id`。
6. 数据不足时输出缺失内容、影响和可支持的最窄结论。
7. 使用固定回答结构：`结论`、`关键证据`、`建议或限制`。
8. 提交前检查事实是否来自工具结果、指标方向是否一致、建议是否超出证据。

V2 Prompt 不包含 Case ID、业务标签、能力标签、预期工具、Gold、容差、评分字段、错误类型或
具体 Case 文案。Prompt 干预必须是跨 Case 可复用的产品策略。

### Prompt 冻结

每个 Prompt 定义：

- `prompt_name`
- `prompt_version`
- UTF-8 Prompt 文本
- canonical SHA-256
- 变更说明

Pilot 完成后允许基于 Development 证据进行一次或多次显式修订。进入正式运行前同时冻结：

- V1 Prompt。
- V2 Prompt。
- 评分 Rubric。
- Error Taxonomy 规则。
- 统计配置。

冻结后任何变更都必须产生新版本和新的完整配对运行，不得覆盖或续写旧 Artifact。

## 配置与安全

真实模型配置只从进程环境读取：

```text
LLM_API_KEY
LLM_MODEL
LLM_BASE_URL
LLM_TEMPERATURE
LLM_TIMEOUT_SECONDS
```

规则：

- `LLM_API_KEY` 和 `LLM_MODEL` 必填。
- `LLM_BASE_URL` 默认
  `https://ark.cn-beijing.volces.com/api/v3`。
- `LLM_TEMPERATURE` 必须为 `0`。
- `LLM_TIMEOUT_SECONDS` 默认 `30`，必须为有限正数。
- API Key 不得出现在异常、stdout、日志、Prompt 快照、测试 fixture 或 Artifact。
- 不自动加载或打印 `.env` 内容；调用方负责将 Secret 注入环境。
- `run_manifest.json` 记录 adapter、base URL host、模型标识和非敏感参数，不记录 Header。

Preflight 在创建临时 Run 目录前完成。缺失配置、非法 URL、非零温度、模型为空或数据身份错误
都属于运行级失败，不发起网络请求，不发布 Artifact。

## Adapter 设计

`OpenAICompatibleAdapter` 增加不可变的可选 `system_prompt`。首轮 transcript 必须是：

```text
system: <frozen prompt>
user: <case user_input>
```

后续 assistant tool call、tool result 和 assistant final answer 沿用现有严格 transcript
协议。Adapter 仍只接受一次一个工具调用，拒绝：

- 同时返回 final answer 和 tool calls。
- 一次返回多个 tool calls。
- 重复或未知 `call_id`。
- 与 pending call 不一致的工具结果。
- 非 JSON、超大响应、重定向和非法 Usage。

Adapter 不向调用方暴露 provider 错误正文，避免泄漏敏感信息。正式实验不自动重试；HTTP、
超时或协议失败写入稳定错误码，由 Batch Runner 继续下一个 Case。

## 配对 Runner

### 输入

- 冻结 Case Bundle。
- Development/Public Validation Dataset 目录。
- V1/V2 Prompt Snapshot。
- Provider Settings。
- Pilot 或 Formal Run Mode。
- 输出根目录。

Runner 只能把 `RunnableCase.user_input` 传给 Agent。完整 Evaluation Case 只在后续评分进程中
读取，Real LLM 运行进程不持有 Gold、expected 或评分字段。

### 单 Case 执行

每个 Case、每个版本创建新的：

- `OpenAICompatibleAdapter`
- `AgentRunner`
- 对应 Dataset Catalog 上下文

使用 `time.perf_counter_ns()` 测量从 `AgentRunner.run()` 前到返回后的 elapsed time，并写入
整数 `latency_ms`。不将 Catalog、Adapter transcript 或 Policy 状态跨 Case 复用。

### 失败隔离

以下错误只使当前 Case/Version 标记为 `failed`，配对批次继续：

- Provider HTTP 或网络失败。
- Provider 返回非法协议。
- 未知工具或非法参数。
- 工具执行或结果校验失败。
- 重复调用 ID。
- 超过最大步数。

以下错误停止整批，不发布正式 Run：

- Case、Dataset、Tool Contract 或 Prompt 身份不匹配。
- V1/V2 模型参数不一致。
- Git/运行环境身份无法记录。
- Artifact 写入、回读、Schema 或哈希验证失败。
- 运行期间 Prompt 或输入文件发生漂移。

### Pilot 门禁

Pilot 产生 40 条逻辑执行记录。进入 Prompt/Rubric 冻结前必须满足：

- 两个版本各包含相同的 20 个 Pilot Case，且顺序一致。
- 所有 Run、Prompt、Dataset 和 Tool Contract 身份可回读验证。
- API Key 未出现在任何已发布文件。
- 没有认证、URL、模型不存在或 Artifact 协议类运行级错误。
- 技术性 Case 失败率每版本不超过 10%。

若门禁失败，只允许修复兼容性、Prompt 可执行性或运行基础设施，并生成新的 Pilot Run。旧 Pilot
保留，不覆盖。不得把 Pilot 的任务得分解释为正式效果。

## Artifact v2

### 目录

```text
outputs/experiment_runs/
├── real-v1-pilot__<UTC时间>__<hash>/
├── real-v2-pilot__<UTC时间>__<hash>/
├── real-v1__<UTC时间>__<hash>/
└── real-v2__<UTC时间>__<hash>/
```

每个目录包含：

```text
run_manifest.json
case_runs.jsonl
summary.json
prompt_snapshot.json
```

配对组另存：

```text
outputs/comparisons/<comparison_group_id>/
└── comparison_manifest.json
```

### `run_manifest.json`

记录：

- Artifact Schema Version 和 Run ID。
- `comparison_group_id`、实验模式和 Agent 版本。
- Git commit、dirty 状态和关键依赖版本。
- Case Set、Case Manifest 和本次 Case 子集身份。
- 两个 Dataset、Tool Contract 和 Tool Schema 身份。
- Adapter、模型、base URL host、温度、超时、max steps、concurrency 和 repeats。
- Prompt 名称、版本、文件和 SHA-256。
- 开始/结束 UTC 时间。
- 输出文件 SHA-256。
- `evaluation_status=pending_not_run`。
- Usage 可用状态和 Cost 可用状态。

### `case_runs.jsonl`

每条记录包含：

- `sequence`、`case_id`、Split、Dataset ID/版本。
- `agent_version` 和 `repeat_index=1`。
- `user_input`。
- `status`、`final_answer` 和稳定 `error_code`。
- 完整 `prior_tool_executions` 和公开 `decision_trace`。
- `latency_ms`。
- Provider 返回的 prompt/completion/total tokens，未返回时为 `null`。

不得包含：

- API Key、Authorization Header 或 provider 错误正文。
- Gold、expected、Case 标签、评分、错误归因或隐藏思维链。
- 根据公开价格推算但未版本化验证的 Cost。

### `summary.json`

只汇总运行事实：

- total、completed、failed、stopped。
- Split 和状态分布。
- Latency P50/P95 和样本量。
- Token 总量和分位数；Provider 未返回时标为 `Unavailable`。
- Cost 固定为 `Unavailable`，除非未来单独引入冻结价格表。
- `evaluation_status=pending_not_run`。

### `comparison_manifest.json`

绑定：

- V1/V2 Run ID 和 Manifest SHA-256。
- Case 顺序和配对键。
- 交替执行顺序规则。
- 允许不同的字段白名单：Agent Version、Prompt Name/Version/SHA-256、时间戳、Run ID、
  输出哈希和观测结果。
- 必须相同的模型、参数、Cases、数据、工具、Runner 和运行环境字段。

公平性校验不通过时禁止评分和统计。

### 发布协议

沿用 Phase 3 的临时目录、完整回读、SHA-256 校验和原子重命名。正式目录不可覆盖。配对批次
只有在两个 Run 都验证成功后才发布 `comparison_manifest.json`；任一侧运行级失败时两个临时
Run 都不发布为正式配对结果。

## 确定性评分协议

评分进程只读取已发布 Run Artifact 和冻结 Evaluation Case Bundle，不重新调用 Agent 或模型，
不改写原始 Run。

### 状态

每个评分维度只允许：

```text
pass
fail
not_applicable
unavailable
needs_review
```

缺失、无法可靠解析或不适用不得自动记为 0。

### 评分维度

1. **Execution Status**：Run 是否完成并产生 final answer。
2. **Tool Selection**：工具名称和有意义的调用顺序是否匹配 `expected_tool_calls` 或
   `allowed_alternatives`。
3. **Parameter Accuracy**：日期、商品范围、布尔开关、指标集合和 group-by 是否正确。
4. **Factual Accuracy**：回答中的可识别指标是否在 `numeric_tolerances` 内匹配 Gold。
5. **Required Output Completeness**：业务任务所需核心实体和核心指标是否覆盖。
6. **Evidence Binding**：V2 的关键事实是否引用真实存在且支持该事实的 `result_id`；
   V1 为 `not_applicable`，不因此受罚。
7. **Unsupported Claim**：可验证事实是否能映射到本次工具结果。
8. **Instruction Following**：是否遵守用户明确格式、范围和直接回答要求。
9. **Data Insufficiency Handling**：数据不足 Case 是否说明限制且未给出过度结论。
10. **Recommendation Support**：建议是否由已观察事实支持。

Explanation Quality 和 Recommendation Quality 的开放式人工 Rubric 可作为后续补充，不进入
本阶段的确定性 Task Success。

### 规范化

- 工具调用顺序有语义，保持有序比较。
- `product_ids` 去重后按值比较。
- `metrics` 和 `group_by` 按集合比较。
- 日期规范化为 ISO `YYYY-MM-DD`。
- 数字比较使用 Case 自带绝对容差。
- 百分数和小数比率通过指标别名表归一化。
- 无法可靠映射标签与数值时标为 `needs_review`，不猜测。

### 核心事实

Rubric 按业务任务定义核心事实，并只对 Case Gold 中实际存在的字段取交集：

- GMV 诊断：当前/对照 GMV、GMV 变化率、订单和 AOV 方向。
- 商品异常：目标商品、GMV 变化、流量变化、CVR 和退款率。
- 转化下降：目标商品、当前/对照 CVR、CVR 变化和观察天数。
- 关注商品：目标商品、GMV/CVR/退款信号和证据排序。
- 下周重点：目标商品、`evidence_value` 排序、行动建议和限制。
- 数据不足：观察天数、缺失范围和最窄可支持结论。

具体字段别名、任务映射和门控规则保存在版本化 `rubric_v1` 配置中并进入 SHA-256。

### Task Success

Case 的 `success_criteria` 映射为以下门控：

```text
execution_status == pass
AND correct_tool == pass
AND valid_parameters == pass
AND correct_core_facts == pass
AND no_critical_unsupported_claim == pass
AND required_output_complete == pass
```

任何必需维度为 `needs_review` 时，Task Success 也为 `needs_review`，不强制判定 pass/fail。
Evidence Binding 是 V2 的诊断指标，但不加入 V1/V2 共用的 Task Success 门控，避免对 V1
施加其 Prompt 未承诺的额外要求。

### 评分 Artifact

```text
outputs/evaluations/<evaluation_id>/
├── evaluation_manifest.json
├── case_scores.jsonl
└── summary.json
```

Manifest 绑定 Run、Case Bundle、Rubric 和评分器源码 SHA-256。`case_scores.jsonl` 保存每个
维度的状态、机器可审计理由和引用的 Run/Gold 字段；不保存模型隐藏推理。

## Error Taxonomy 与 RCA

只对 `task_success=fail` 的 Case 自动归因；`needs_review` 保持人工复核状态。

错误类型：

- `E1 Tool Selection Error`
- `E2 Parameter Error`
- `E3 Data Retrieval Error`
- `E4 Calculation Error`
- `E5 Reasoning Error`
- `E6 Hallucination`
- `E7 Instruction Following Error`
- `E8 Incomplete Answer`
- `E9 Unsupported Recommendation`
- `E10 Data Insufficiency Handling Error`

### 因果顺序

Primary Error 是最早导致最终失败的可干预原因，不按 E 编号机械排序：

```text
decision trace
→ first causal failure
→ downstream score failures
→ primary_error
→ secondary_errors
→ root_cause
→ intervention
```

自动规则：

- 错误工具或缺少必要工具通常归 E1。
- 工具正确但参数错误通常归 E2。
- 参数正确但工具返回错误、空结果或数据契约失败归 E3。
- 工具结果正确但派生数值错误归 E4。
- 数值正确但解释方向或归因逻辑错误归 E5。
- 回答加入工具结果中不存在的可验证事实归 E6。
- 用户明确格式或范围未遵守归 E7。
- 必要结论缺失且无更早原因归 E8。
- 建议无法由证据支持归 E9。
- 数据不足时仍下确定结论或未说明限制归 E10。

如果两种 Primary 原因都可解释失败且无法从 trace 判定先后，标记
`attribution_status=needs_review`，不强制自动分类。

RCA 记录：

- `symptom`
- `primary_error`
- `secondary_errors`
- `first_causal_event_sequence`
- `root_cause`
- `intervention`
- `attribution_status`
- `review_note`

Error Analysis Artifact 只读取评分结果和 Run，不修改二者。

## 配对统计

### 配对与公平性门禁

统计前必须验证：

- V1/V2 `case_id` 集合、顺序和 Split 完全一致。
- 每个 Case 的 Dataset ID/版本一致。
- `statistical_cluster_id` 可从冻结 Case Bundle 唯一映射。
- 模型、参数、工具、Runner 和运行环境满足 comparison manifest。
- 每个 Case/Version 只有 `repeat_index=1`。

技术失败保留在分母中并视为 Task Success fail；同时单独报告 technical failure rate。

### 描述性结果

分别报告：

- All 100 Cases / 50 clusters。
- Development 70 Cases / 35 clusters。
- Public Validation 30 Cases / 15 clusters。
- 五类业务任务、十类 primary capability 和难度切片。

Public Validation 是公开验证集，不是盲测。Development 包含 Pilot 调优数据，因此不能作为独立
泛化证据。All 100 结果用于项目内总体描述，不声称未见数据表现。

### Cluster Bootstrap

主效应为配对 Task Success 率差：

```text
delta = success_rate_v2 - success_rate_v1
```

Bootstrap 规则在结果可见前冻结：

- 重采样单位：`statistical_cluster_id`。
- 每次有放回抽取完整 cluster，并保留 cluster 内两个 Case 的 V1/V2 配对结果。
- 重采样次数：`10_000`。
- 随机种子：`20261009`。
- 置信区间：percentile 95% CI。
- 同样计算 Tool Selection、Parameter、Factual、Completeness、technical failure、
  latency 和 token 的配对差；这些是次要或探索性指标。

如果子集少于 10 个唯一 clusters，仍展示描述性差异，但不输出推断性 CI。

### Cluster-level McNemar

二元显著性检验使用预先定义的严格 cluster 汇总：

```text
cluster_success(version) = 该 cluster 的两个 Case 均 task_success=pass
```

在 50 个 cluster-level V1/V2 配对结果上运行 Exact McNemar Test。报告：

- V1-only success clusters。
- V2-only success clusters。
- Exact two-sided p-value。
- 有效 cluster 数量。

如果 discordant cluster 总数为 0，p-value 标记为 `not_applicable`。Development 和 Public
Validation 子集只做探索性检验，并明确多重比较风险；不根据最显著的切片选择主结论。

### Latency 与 Token

- 报告每版本 P50/P95、Cases 数和 clusters 数。
- 配对差使用 cluster bootstrap。
- Provider 未返回 Usage 的 Case 不填 0；Token 指标标记 `unavailable` 并报告可用样本数。
- 不用本地估算 Token 替代 Provider Usage。
- 没有冻结价格配置时 Cost 始终为 `Unavailable`。

### Error Shift

按 `case_id` 形成：

```text
V1 primary_error → V2 primary_error
```

成功 Case 使用 `NO_ERROR`。转移矩阵报告错误减少、持续和新增数量。涉及区间时按 cluster
重采样；样本过小的单个错误类型只报告计数，不进行显著性推断。

## 实施批次与门禁

### 批次 A：真实模型运行基础设施与 Pilot

交付 Adapter system prompt、Prompt Snapshot、Provider Settings、配对 Runner、Artifact v2、
20 Cases Pilot 和测试。

门禁：Pilot 技术门禁通过；V1/V2 Prompt 可基于 Development 证据修订，但不发布正式效果。

### 批次 B：评分与分析协议冻结

交付 Rubric、评分器、Error Taxonomy 规则和统计配置。使用手工构造 fixture 和 Pilot 结果验证
实现，不根据正式结果调整规则。

门禁：Prompt、Rubric、Error Taxonomy 和统计配置全部有版本与 SHA-256；冻结清单提交 Git。

### 批次 C：100 Cases 正式配对运行

交付 Real V1、Real V2 和 Comparison Manifest。正式运行前工作树必须明确记录 dirty 状态；
推荐使用已提交且无关文件不参与身份计算的固定 commit。

门禁：两个 Run 原子发布、公平性校验通过、200 条逻辑记录完整。

### 批次 D：评分和错误分析

交付 Case Scores、Summary、E1-E10、RCA 和 Error Shift 数据。不得重新调用模型。

门禁：Artifact 回读与哈希验证通过；`needs_review` 数量和原因被显式报告。

### 批次 E：统计验证

交付配对指标、Cluster Bootstrap、cluster-level McNemar、Latency/Token 和限制说明。

门禁：只读取冻结 Artifact；报告 Cases 数、clusters 数、CI、p-value 可用状态和单次运行限制。

## 测试策略

### Adapter

- system message 只在首轮出现一次。
- system prompt 不随工具轮次变化。
- V1/V2 除 Prompt 外生成相同请求配置。
- API Key 不进入 payload dump、错误或 Artifact。
- 火山方舟 fake transport 覆盖工具调用、Usage、HTTP、超时和非法响应。

### Prompt 与输入隔离

- Prompt 快照 SHA-256 稳定。
- Prompt 文本不包含 Case ID、Gold、expected、score 或错误类型。
- Real Runner 只接收 `RunnableCase`。
- 测试在 Gold 中注入 sentinel，确认 Run Artifact 不出现 sentinel。

### Paired Runner

- Pilot Case 集合、任务、能力、难度和 cluster 覆盖固定。
- 奇偶 sequence 的版本顺序交替。
- 单 Case 失败不影响后续。
- 运行级身份错误停止整批。
- V1/V2 模型参数差异导致 comparison validation 失败。
- Latency 使用单调时钟，Usage 只来自 Provider。

### Artifact

- 临时目录校验后原子发布。
- 目标目录存在时拒绝覆盖。
- 两侧任一 Run 失败时不发布 comparison manifest。
- Schema、行数、顺序、SHA-256 和 Secret 扫描通过。
- Phase 3 Artifact Schema 和正式 Run 不受影响。

### Scoring

- 已知 pass/fail/not_applicable/unavailable/needs_review fixture。
- 工具顺序、参数规范化和 numeric tolerance 边界。
- 百分数/小数、金额、计数和商品级指标解析。
- 无法可靠解析时进入 needs_review。
- Task Success 门控和缺失状态传播。
- 评分进程不修改 Run Artifact。

### Error Analysis

- 每个 fail Case 最多一个 Primary Error。
- Secondary Error 去重且不包含 Primary。
- earliest causal failure 规则覆盖 E1-E10。
- 歧义归因进入 needs_review。
- Error Shift 保持 V1/V2 case pairing。

### Statistics

- cluster 而非 Case 被重采样。
- 同 cluster 的两个 Case 始终共同出现。
- 固定 seed 产生稳定 CI。
- cluster-level all-pass 汇总和 Exact McNemar 已知样例。
- 无 discordant pair、少于 10 clusters、全 pass、全 fail 和 Usage 缺失退化情形。

## 验收标准

- V1/V2 同一模型、同一参数、同一工具和同一 Case 顺序可由 Manifest 验证。
- 20 Cases Pilot 已真实运行或在缺少 Secret/Model 时保持明确未运行状态。
- Prompt、Rubric、Error Taxonomy 和统计配置在正式运行前冻结。
- 正式配对 Run 各包含 100 个 Case 和一次执行记录。
- 所有指标来自真实 Run 和冻结评分器，不使用占位数字。
- API Key、Gold 和评分字段不进入 Agent 可见输入或 Run Artifact。
- 评分结果保留五种状态，不把不可用或待复核写成零。
- 每个失败 Case 最多一个 Primary Error，歧义可追踪。
- 配对统计按 50 个 cluster 处理，不把 100 个 Case 当独立样本。
- 报告明确 Synthetic Data、Public Validation 非盲测和单次运行方差限制。
- Phase 3 Deterministic Baseline 及冻结资产测试继续通过。

## 风险与处理

### 模型工具调用兼容性

风险：指定方舟模型不稳定支持 OpenAI tool calling。

处理：Pilot preflight 和 20 Cases 技术门禁；如果模型不兼容，停止并更换同一 V1/V2 模型后
重跑 Pilot。不得只为某一版本更换模型。

### Prompt 调优污染

风险：基于 Pilot Case 优化 V2 后，在同一 Development 集上报告提升会高估泛化效果。

处理：Pilot 只用 Development；正式报告分开展示 Development/Public Validation，并披露
Pilot Case 被用于调优。最终泛化结论留给未来冻结后的盲测集。

### 单次运行方差

风险：每 Case 单次运行无法区分 Prompt 效应和模型随机波动。

处理：温度固定为 0、采用同模型和交替版本顺序、明确限制；未来预算允许时增加独立重复实验，
但不把重复结果追加到当前 Run。

### 评分解析歧义

风险：自然语言数值、实体和因果表达无法完全由规则可靠解析。

处理：规则优先；不确定项进入 `needs_review`，报告人工复核率，不用宽松启发式强制判分。

### Provider 与网络故障

风险：限流或短暂网络失败造成版本间技术失败不平衡。

处理：串行、交替版本顺序、不自动重试、显式 technical failure rate。若出现运行级服务事故，
作废整个 comparison group 并用新 Run ID 重跑两侧，不局部补跑。

## 明确限制

- 数据为 Synthetic E-commerce Data，不代表真实企业业务。
- Public Validation 已公开，不是最终盲测。
- Pilot Development Cases 会参与 Prompt 调优。
- 每个 Case 仅运行一次。
- 不启用 LLM Judge。
- 没有版本化价格表时不报告 Cost。
- 规则评分无法覆盖所有开放式解释质量，`needs_review` 是预期状态，不是评分器故障。
