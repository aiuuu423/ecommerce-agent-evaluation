# 多版本 Agent 效果评测与错误归因分析

> 面向电商业务场景的 LLM Agent 评测、指标体系与 Error Analysis

## 1. 项目定位

本项目构建一个可运行的“电商经营分析 Agent”，并围绕 Baseline V1 与 Optimized V2 建立完整的评测、错误归因、统计验证和迭代闭环。

项目重点不是展示复杂后端架构，而是证明以下能力：

- 将真实业务问题转化为 Agent 产品能力和任务分类。
- 构建可追溯、可程序化验证的 Evaluation Dataset。
- 设计确定性优先的多维指标体系。
- 对 Agent 失败进行 Error Taxonomy、RCA 和 Error Shift 分析。
- 使用配对统计方法验证版本差异。
- 将实验过程和结果转化为可运行 Dashboard 与求职作品集。

最终闭环：

```text
Synthetic Data
→ Evaluation Dataset
→ Agent Run
→ Metrics
→ A/B Evaluation
→ Error Analysis
→ RCA
→ Statistical Validation
→ Optimization
→ Re-evaluation
```

## 2. 不可违反的原则

### 2.1 数据与结果真实性

- 业务数据可以程序化生成，但必须明确标记为 `Synthetic E-commerce Data / 模拟电商数据`。
- 任何准确率、成功率、延迟、Token、成本、显著性和效果提升必须来自真实项目运行。
- 未执行的实验统一显示 `Pending / Not Run`。
- API 未返回 Token Usage 时显示 `Unavailable`。
- 缺少真实 Usage 或明确价格配置时，Cost 显示 `Unavailable`，禁止估算。
- 不得修改数据、Case、评分规则或统计方法来制造更好的优化结果。

### 2.2 研发边界

- 严格按 Phase 推进，不提前实现后续阶段功能。
- 每个 Phase 都要检查代码、执行测试、记录实际结果并更新 `PROJECT_STATUS.md`。
- 不做与当前阶段无关的重构。
- 先完成 Mock / Deterministic 闭环，再启用 Real LLM Mode。
- 数据、工具、Agent、评测器和 Dashboard 通过明确接口解耦。
- Notebook 只用于探索和验证，不承载生产逻辑。

### 2.3 安全与配置

- API Key 仅通过环境变量加载。
- `.env` 不提交 Git；仓库只提供 `.env.example`。
- API Key 不得出现在代码、README、测试数据、日志和实验产物中。
- 运行产物只记录 adapter、模型和非敏感配置。

## 3. 用户与使用场景

### 3.1 目标用户

- 电商运营或经营分析人员：希望快速定位经营变化与异常商品。
- AI Product / Evaluation 从业者：希望比较 Agent 版本并定位失败原因。
- 面试官或作品集评审者：希望验证项目是否真实可运行、结果是否可追溯。

### 3.2 核心业务问题

首版固定覆盖五类业务任务：

1. 最近 30 天 GMV 为什么下降？
2. 哪些商品最近出现异常？
3. 哪些商品转化率下降最明显？
4. 哪些商品值得进一步关注？
5. 根据最近经营数据，下周应该重点关注什么？

### 3.3 Agent 产品能力

Agent 必须能够：

1. 识别用户意图与任务类型。
2. 判断是否需要工具。
3. 选择正确工具并生成有效参数。
4. 获取结构化数据。
5. 使用统一指标口径完成计算和分析。
6. 将结论绑定到数据证据。
7. 在数据不足时明确不确定性。
8. 输出结构化、可检查的最终答案。
9. 避免生成无数据依据的事实与建议。

## 4. Task Taxonomy

任务使用两个正交维度描述，避免混淆“业务问题”和“评测能力点”。

### 4.1 业务任务维度

| 编码 | 业务任务 | 典型输出 |
|---|---|---|
| B1 | GMV 归因 | 变化幅度、贡献因素、证据 |
| B2 | 商品异常 | 异常商品、异常类型、严重程度 |
| B3 | 转化下降 | CVR 变化、商品排序、关联流量 |
| B4 | 关注商品 | 关注原因、优先级、风险或机会 |
| B5 | 下周重点 | 基于数据的行动建议与限制 |

### 4.2 评测能力维度

| 编码 | 能力类型 |
|---|---|
| T1 | Basic Query |
| T2 | Metric Calculation |
| T3 | Tool Selection |
| T4 | Parameter Selection |
| T5 | Multi-step Reasoning |
| T6 | Anomaly Detection |
| T7 | Root Cause Analysis |
| T8 | Recommendation |
| T9 | Data Insufficiency |
| T10 | Adversarial / Distractor |

每个 Evaluation Case 至少有一个业务任务标签和一个主要能力标签，可附加多个次要能力标签。

## 5. 技术方案

### 5.1 架构选择

采用“评测优先的模块化单体”：

- Python package 承载数据、Agent、工具、评测和统计逻辑。
- DuckDB 执行分析型 SQL。
- Parquet 保存可版本化的数据快照。
- CLI / Python entrypoint 负责实验运行。
- Streamlit 负责只读展示冻结结果。
- OpenAI-compatible adapter 支持真实模型。
- Deterministic adapter 保证无 API Key 时仍可运行完整闭环。

不采用：

- Notebook 驱动架构：核心逻辑容易分散，难以测试和复现。
- 首版服务化架构：FastAPI、任务队列和独立数据库服务会稀释 Evaluation 主线。

### 5.2 推荐技术栈

- Python 3.11+
- Pandas、NumPy
- Pydantic
- DuckDB、PyArrow / Parquet
- SciPy
- Matplotlib、Plotly
- Streamlit
- Pytest
- OpenAI-compatible HTTP client
- Ruff 作为基础代码检查工具

依赖以 `pyproject.toml` 为唯一声明来源；如提供 `requirements.txt`，应由同一依赖定义导出，避免双重维护。

## 6. 系统架构

```text
User / CLI / Streamlit
          ↓
      Agent Runner
      ↙          ↘
Baseline V1    Optimized V2
      ↘          ↙
       Tool Registry
     ↙      ↓       ↘
 Sales   Product   Traffic/Marketing
          ↓
   DuckDB + Parquet
          ↓
 Structured Tool Results
          ↓
   Evaluation Engine
   ↙       ↓        ↘
Metrics  Error/RCA  Statistics
          ↓
 Experiment Artifacts
          ↓
 Dashboard / Reports / Portfolio
```

### 6.1 模块边界

- `app/data`：数据 Schema、生成、校验、加载和版本。
- `app/tools`：受控查询工具和 Tool Registry。
- `app/llm`：Deterministic 与 OpenAI-compatible adapter。
- `app/agents`：V1/V2 策略、Prompt 与统一 Runner。
- `app/evaluation`：Case 评分、指标、错误分类、RCA 和统计。
- `app/experiments`：实验配置、运行编排和 Artifact 写入。
- `sql`：Dashboard 和分析使用的可审阅 SQL。
- `dashboard`：只读查询与展示。
- `reports`：从真实实验产物生成的分析与作品集文档。

## 7. 数据设计

### 7.1 Synthetic Data

基础数据集包含：

- `products`
- `orders`
- `traffic`
- `customers`
- `marketing`

支持计算：

- GMV
- Orders
- AOV
- CTR
- CVR
- Refund Rate
- ROAS

生成器必须支持固定 Seed，并通过配置显式注入：

- 正常商品
- 销量下降
- 转化率异常
- 流量异常
- 高退款
- 缺失数据
- 极端值
- 多因素共同变化

生成器的目标是创建可测试场景，不是模拟真实公司的业务结果。

### 7.2 数据血缘

```text
generator_config + seed
→ Parquet snapshot
→ dataset manifest + hash
→ DuckDB views/tables
→ gold SQL / metric functions
→ gold evidence
→ evaluation_cases.jsonl
```

任何 Case 必须绑定：

- `dataset_version`
- `dataset_hash`
- `generator_config_hash`
- `case_schema_version`

数据版本变化后，不允许静默复用旧 Ground Truth。

### 7.3 Evaluation Case Schema

每个 Case 至少包含：

- `case_id`
- `case_version`
- `statistical_cluster_id`
- `business_task`
- `capability_tags`
- `difficulty`
- `user_input`
- `expected_tool_calls`
- `expected_parameters`
- `allowed_alternatives`
- `gold_metrics`
- `gold_evidence`
- `reference_answer`
- `expected_behavior`
- `success_criteria`
- `numeric_tolerances`
- `dataset_version`
- `metadata`

首批目标为 100 Cases / 50 statistical clusters（每簇 2 Cases）；完成质量检查和覆盖
分析后扩充到 200 Cases。

Ground Truth 优先由 SQL 和指标函数生成。自然语言参考答案不能作为唯一评分依据。

冻结 Case 使用 Development/Public Validation `70/30` Split：每个业务任务固定
`14/6`，Public Validation 十类主要能力各 `3` 个 Case，难度固定为
easy/medium/hard `9/12/9`。同一业务任务与主要能力的复述变体共享
`statistical_cluster_id`；Public Validation 是公开验证集，不是盲测。

## 8. Agent 版本设计

### 8.1 公平对照

V1 与 V2 必须共用：

- 相同数据快照
- 相同工具实现与 Schema
- 相同 Evaluation Cases 与顺序
- 相同模型与模型参数
- 相同评测器
- 相同实验环境

主要自变量是 Agent 策略与 Prompt。

### 8.2 Baseline V1

Baseline 代表“基本可用但约束较少”的 Agent，不故意制造失败：

- 基础任务识别
- 基础工具说明
- 基础结构化回答
- 无强制证据绑定
- 无强制参数校验
- 无强制数据不足协议
- 无强制提交前检查

### 8.3 Optimized V2

V2 增加有限且可解释的干预：

- 显式任务分类
- 工具选择边界
- 参数与指标口径校验
- 必要的受控多步调用
- 结论到 `evidence_id` 的绑定
- 数据不足与不确定性协议
- 固定输出结构
- 轻量提交前规则检查

V2 不得读取 Gold、预期工具或评分器内部规则。优化项必须能够对应到已观察错误或预先声明的产品风险。

### 8.4 双运行模式

- Mock / Deterministic Mode：使用确定性决策策略实际调用工具并生成回答，不返回预设分数。
- Real LLM Mode：使用 OpenAI-compatible adapter 替换决策与生成模块，其他链路保持一致。

运行记录保存结构化 `decision_trace`，不依赖模型隐藏思维链。

## 9. Evaluation Framework

### 9.1 结果状态

统一状态：

- `pass`
- `fail`
- `not_applicable`
- `unavailable`
- `needs_review`

缺失信息不得自动转为 0。

### 9.2 核心指标

| 指标 | 判定原则 |
|---|---|
| Task Success | Case 必要条件门控 |
| Tool Selection Accuracy | 预期调用、顺序与允许替代路径 |
| Parameter Accuracy | 字段、日期、范围、指标口径和参数值 |
| Factual Accuracy | Gold Evidence 与预设数值容差 |
| Completeness | 必要结论覆盖；规则优先，人工 Rubric 辅助 |
| Hallucination Rate | 无法映射到数据证据的可验证事实占比 |
| Instruction Following | 格式、限制与用户要求 |
| Latency | 实际计时，报告 P50、P95 和样本量 |
| Token Usage | 仅使用 API 返回值 |
| Cost | 仅使用真实 Usage 和明确价格配置 |

Explanation Quality 与 Recommendation Quality 使用人工 Rubric。可选 LLM Judge 必须记录 Judge 模型、Prompt、Rubric、原始输出和版本，且不替代确定性主指标。

### 9.3 Task Success

每个 Case 在正式实验前冻结 `success_criteria`。典型门控规则：

```text
correct_tool
AND valid_parameters
AND correct_core_facts
AND no_critical_unsupported_claim
AND required_output_complete
```

禁止在查看实验结果后修改成功标准。

## 10. Error Taxonomy 与 RCA

错误分类：

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

每个失败 Case 只有一个 `primary_error`，可以有多个 `secondary_errors`。

Primary Error 根据因果链确定，而不是按编号固定优先：

```text
Symptom
→ First causal failure
→ Downstream errors
→ Root cause
→ Intervention
```

无法可靠归因时标为 `needs_review`。

Error Shift 分析同时回答：

- 哪些错误减少了？
- 哪些错误仍然存在？
- 是否出现新的错误类型？
- 总错误减少是否伴随错误结构恶化？

## 11. 统计验证

因为 V1/V2 使用相同 Cases，结果先按 `case_id` 对齐和配对；`case_id` 不是独立统计
抽样单位。统计推断、Cluster Bootstrap 和有效样本量计算均以
`statistical_cluster_id` 为基本单位。当前冻结集包含 100 Cases / 50 statistical
clusters（每簇 2 Cases），分层或子集分析必须报告该分析实际包含的唯一 cluster 数量：

- 二元结果：不得直接把 Case 对视为相互独立后运行 Exact McNemar Test。应先冻结簇级
  二元汇总规则并形成每簇一个 V1/V2 配对结果，再使用 McNemar；或使用明确处理簇内
  相关性的配对二元方法。
- 成功率、比例差和连续分数：按 `statistical_cluster_id` 有放回抽取完整簇的 Cluster
  Bootstrap Confidence Interval，并保留簇内全部 `case_id` 配对结果。
- Wilcoxon Signed-Rank Test 等补充检验：使用预先定义的簇级汇总，或改用适合聚类数据的
  方法。
- Latency：报告 P50、P95、Cases 数量、statistical clusters 数量与按 cluster 处理的
  配对分布比较。
- Error Shift：按 `case_id` 形成配对错误转移矩阵；区间或显著性推断按 cluster 处理。

簇级汇总规则、聚类推断方法、Bootstrap 实现、有效样本量口径和退化情形处理必须在
Phase 7 实施前冻结，不得根据实验结果选择。

报告至少包含：

- V1 值
- V2 值
- 绝对差
- 相对差
- 95% CI
- p 值
- 样本量
- 限制说明

统计不显著时必须如实展示。多个指标的探索性比较要明确多重比较风险。

Mock 模式主要验证系统确定性。Real LLM 模式通过 `n_repeats` 记录重复运行；若预算只允许单次运行，必须说明未充分评估模型输出方差。

Public Validation 可用于 V1/V2 开发与选择，不能作为未见数据上的最终泛化证据。
Optimized V2 的方案、Prompt、实现与评分协议冻结后，才生成此前未见的最终盲测集；
该数据集生成前不得查看其 Case 或 Gold，项目也不得把 Public Validation 结果称为盲测。

## 12. 实验可复现协议

每次实验生成不可变 `run_manifest.json`：

- `run_id`
- `git_commit`
- `git_dirty`
- `dataset_version` 与哈希
- `evaluation_dataset_version` 与哈希
- `agent_version`
- Prompt 哈希
- Tool Schema 版本
- model、adapter、temperature、真实 Seed
- Python 和依赖版本
- 开始、结束时间
- Case 数量和运行状态计数
- Token 与 Cost 可用状态

每个 Case 保存：

- 输入
- 结构化决策轨迹
- 工具调用
- 工具结果引用
- 原始回答
- 最终回答
- 实际 latency
- API Usage
- 评分明细
- Primary / Secondary Error
- RCA 与人工复核状态

统计脚本只读取冻结实验产物，不重新调用 Agent，不改写原始运行记录。

## 13. Dashboard

Streamlit Dashboard 使用 SQL 查询冻结结果：

1. Overview：项目、Case 数、运行状态与核心指标。
2. A/B Comparison：V1/V2 指标与分层差异。
3. Error Analysis：错误分布、Pareto、任务和难度切片、Error Shift。
4. Case Explorer：输入、预期、两版本回答、证据、评分、错误和 RCA。
5. Statistical Validation：效应量、区间、簇级或聚类二元配对方法与 Cluster
   Bootstrap。
6. Methodology：数据来源、指标定义、运行配置和限制。

未运行时显示 `Pending / Not Run`；不可获得的 Token 或 Cost 显示 `Unavailable`。所有图表附带样本量、数据版本和 Run ID。

视觉方向：

- 极简、数据产品感
- 黑白灰为基础
- 清晰信息层级
- 图表优先、少装饰
- 不使用大面积渐变
- 面试官应在 30 秒内理解 Agent、评测、差异、错误、根因和建议

## 14. Portfolio 交付

`reports/portfolio_case_study.md` 包含：

- 背景与问题
- Agent 架构
- Evaluation Dataset
- Metric Framework
- V1/V2 设计
- A/B Experiment
- Error Taxonomy 与 RCA
- Statistical Validation
- Product Recommendation
- Limitations
- Future Work

`reports/resume_bullets.md` 提供 AI Product、Data Analyst、Agent Evaluation 三种版本，所有数字从实验结果读取。

`reports/interview_questions.md` 只基于项目真实设计和实际结果回答面试问题。

`reports/career_mapping.md` 说明方法论与目标岗位、既有经历的关系，并明确“方法可迁移，业务环境不同”。

## 15. 推荐目录

```text
/
├── README.md
├── PROJECT_PLAN.md
├── PROJECT_STATUS.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
├── app/
│   ├── agents/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── llm/
│   └── tools/
├── configs/
│   ├── data/
│   ├── evaluation/
│   └── experiments/
├── data/
│   ├── synthetic/
│   ├── evaluation_cases/
│   └── results/
├── dashboard/
├── docs/
│   └── superpowers/specs/
├── notebooks/
├── outputs/
│   ├── experiment_runs/
│   ├── figures/
│   └── tables/
├── reports/
├── sql/
└── tests/
```

## 16. Roadmap 与验收标准

### PHASE 0：项目架构与技术方案

交付：

- `PROJECT_PLAN.md`
- `PROJECT_STATUS.md`
- Phase 0 设计文档

验收：

- 产品范围、技术架构、数据流、评测框架和实验协议已定义。
- 所有结果字段默认 `Pending / Not Run` 或 `Unavailable`。
- 用户确认设计后才制定 Phase 1 实施计划。

### PHASE 1：模拟数据与 Evaluation Dataset

里程碑 A：Synthetic Data。

里程碑 B：首批 100 Cases / 50 statistical clusters（每簇 2 Cases）；目标扩展到
200 Cases。

验收：

- 五张业务表通过 Schema、完整性和指标一致性测试。
- 异常注入规则可由配置复现。
- 数据 Manifest 和哈希可生成。
- Gold 由 SQL / 指标函数程序化生成。
- Case 覆盖五类业务任务和十类评测能力。
- 数据和案例均明确标记 Synthetic。

### PHASE 2：Agent 基础能力

验收：

- Tool Registry 和五类核心工具具备输入输出 Schema。
- Agent 不能绕过工具直接读取整个数据库。
- Deterministic adapter 可执行完整工具链。
- OpenAI-compatible adapter 接口存在，但无 Key 时不影响离线运行。
- 工具、Schema 和 Runner 测试通过。

### PHASE 3：Baseline Agent

验收：

- V1 基本可用且未故意降质。
- 完成离线 Evaluation Dataset 运行。
- 保存完整、可追溯的 Baseline Artifact。
- 所有展示指标来自实际运行。

### PHASE 4：Optimized Agent

验收：

- V2 的每项干预有明确设计理由。
- 除 Agent 策略外的 A/B 条件保持一致。
- V2 不访问 Gold 或评分规则。
- 完成同集离线运行并保存 Artifact。

### PHASE 5：Evaluation Engine

验收：

- 核心确定性指标可自动计算。
- `not_applicable`、`unavailable`、`needs_review` 不被误记为失败或零。
- Task Success 使用冻结的 Case 门控规则。
- 可选 Judge 不影响无 Key 的离线路径。
- 评分器有单元测试和已知样例测试。

### PHASE 6：Error Taxonomy 与 RCA

里程碑 A：E1–E10 分类。

里程碑 B：Symptom → Error → Root Cause → Intervention。

验收：

- 每个失败 Case 最多一个 Primary Error。
- Secondary Error 可追踪。
- 自动规则与人工复核边界明确。
- 可生成错误分布、Pareto 和 Error Shift 数据。

### PHASE 7：Statistical Validation

验收：

- 配对样本对齐检查通过。
- 在实现统计检验前冻结簇级汇总规则、聚类推断方法、Cluster Bootstrap、有效样本量和
  退化情形处理口径。
- 二元配对不把 Case 对当作独立样本；McNemar 仅用于预先定义的簇级二元配对结果，
  否则使用适合聚类数据的方法。
- Cluster Bootstrap 和效应量使用实际结果，并以 `statistical_cluster_id` 为重采样
  单位。
- 输出包含 Cases 数量、statistical clusters 数量、区间和限制。
- 不显著结果被如实保留。

### PHASE 8：Dashboard

验收：

- 六个页面从冻结 Artifact / SQL 读取。
- 无结果时正确显示 `Pending / Not Run`。
- Case Explorer 可追溯到工具结果和 Gold Evidence。
- 面试官可在 30 秒内理解项目主线。

### PHASE 9：自动化实验 Pipeline

验收：

- 单独命令可运行数据、Agent、评测、错误分析和统计。
- 提供等价的 `make` 命令或文档化 CLI。
- 重复运行不会覆盖旧 Artifact。
- 失败时返回非零状态并保留诊断信息。

### PHASE 10：Portfolio Case Study

验收：

- Case Study 从实际 Artifact 生成数字。
- 明确 Synthetic Data、样本规模、Judge Bias 和生产差距。
- 图表、文本和实验结果一致。

### PHASE 11：Resume / Interview Material

验收：

- 三类岗位简历要点从真实结果读取。
- 面试回答与项目实现一致。
- Career Mapping 不夸大生产经验。
- README 可让陌生人从零运行项目。

## 17. 每阶段统一门禁

每个 Phase 完成前必须：

1. 检查本阶段代码和文档。
2. 执行对应测试。
3. 执行最小运行验证。
4. 记录真实命令和结果。
5. 确认数据、配置和 Artifact 可追溯。
6. 更新 `PROJECT_STATUS.md`。
7. 记录 Known Issues 与 Limitations。
8. 明确下一阶段入口。
9. 获得用户确认后再推进。

## 18. Definition of Done

项目只有在以下项目全部真实完成后才算完成：

- [ ] 可以从零运行
- [ ] 有明确标记的模拟电商数据
- [ ] 有版本化 Evaluation Dataset
- [ ] 有 Baseline Agent
- [ ] 有 Optimized Agent
- [ ] 有真实执行的 A/B Evaluation
- [ ] 有 deterministic metrics
- [ ] 有 Error Taxonomy
- [ ] 有 RCA
- [ ] 有统计检验
- [ ] 有 SQL 分析
- [ ] 有 Dashboard
- [ ] 有实验记录
- [ ] 有可复现 Pipeline
- [ ] 有 Portfolio Case Study
- [ ] 有 Resume Bullets
- [ ] 有 Interview Questions
- [ ] 没有伪造结果
- [ ] README 可让陌生人从零运行
