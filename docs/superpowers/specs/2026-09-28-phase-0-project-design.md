# Phase 0 项目设计

## 文档状态

- 项目：多版本 Agent 效果评测与错误归因分析
- 日期：2026-09-28
- 状态：Self-reviewed / Awaiting User Review
- 当前阶段：PHASE 0
- 下一阶段：PHASE 1，需在本文档获批并形成实施计划后开始

## 1. 背景与目标

项目需要构建一个面向电商经营分析的 Agent，并对 Baseline V1 与 Optimized V2 进行系统评测。最终交付必须同时具备可运行系统、可追溯数据、可复现指标、可解释错误分析、统计验证和作品集表达。

本项目服务于 AI Product、AI Product Operations、Agent Evaluation、LLM Evaluation 和 Data Analysis 岗位。工程实现是评测闭环的载体，而不是项目的唯一目标。

## 2. 成功标准

Phase 0 成功不代表 Agent 已实现，而代表以下内容已经成为明确、可执行的设计：

- 产品用户、业务任务和 Agent 能力边界清晰。
- 数据、Ground Truth 和 Evaluation Case 具备可追溯方案。
- V1/V2 的公平对照条件已经定义。
- 指标、错误分类和统计方法具备可执行定义。
- Mock 与 Real LLM 的边界明确。
- Dashboard 和 Portfolio 不会展示未经运行的结果。
- Phase 0–11 的验收门禁明确。

## 3. 已确认决策

### 3.1 架构

采用评测优先的模块化单体：

- Python package 管理领域逻辑。
- DuckDB 执行分析型 SQL。
- Parquet 保存可版本化数据快照。
- CLI / Python entrypoint 运行实验。
- Streamlit 只读展示冻结结果。
- OpenAI-compatible adapter 提供模型供应商解耦。
- Deterministic adapter 提供无 Key 的完整离线路径。

### 3.2 评分

采用规则优先的混合评分：

- 工具、参数、数值、格式和证据关系使用确定性评分。
- 完整性、解释质量和建议质量可使用人工 Rubric。
- LLM Judge 是可选辅助，不是主评测依据。

### 3.3 数据

采用 DuckDB + Parquet。业务数据明确标记为 Synthetic。Ground Truth 优先由固定 SQL 与指标函数生成。

### 3.4 范围

首版固定五类业务任务：

1. GMV 下降归因。
2. 商品异常识别。
3. 转化率下降分析。
4. 值得关注商品识别。
5. 下周经营重点建议。

Evaluation Dataset 首批 100 Cases，质量验证后扩展至 200。

## 4. 方案比较

### 4.1 评测优先的模块化单体

优点：

- 适合单人完成。
- 易于离线运行和测试。
- 能突出 Evaluation 与数据分析能力。
- 后续可以增量增加真实模型与公开展示。

限制：

- 不包含生产级分布式部署。
- 若后续需要多人服务，需要增加 API 层。

结论：采用。

### 4.2 Notebook 驱动

优点是快速、直观；缺点是逻辑容易散落，难以形成稳定运行入口和可复现 Artifact。

结论：Notebook 仅用于探索，不作为核心架构。

### 4.3 API 服务化

优点是工程扩展性强；缺点是首版复杂度高，会稀释评测方法论。

结论：暂不采用。只有出现明确远程服务需求时再评估。

## 5. 产品定义

### 5.1 用户

- 电商运营与经营分析人员。
- Agent Evaluation 与 AI Product 从业者。
- 作品集评审者或面试官。

### 5.2 输入

自然语言经营问题，以及可选时间范围、商品范围、指标口径和输出要求。

### 5.3 输出

结构化回答至少包括：

- 结论。
- 支撑数据与 `evidence_id`。
- 指标口径。
- 不确定性或数据限制。
- 有数据依据的建议。

### 5.4 非目标

- 不连接真实用户或公司经营数据。
- 不声称达到生产环境性能。
- 不构建通用 BI 平台。
- 不构建复杂多 Agent 协作系统。
- 不在首版引入分布式服务、消息队列或独立模型网关。

## 6. 数据与 Ground Truth

### 6.1 三层结构

业务事实层：

- `products`
- `orders`
- `traffic`
- `customers`
- `marketing`

评测案例层：

- 用户问题。
- 业务任务与能力标签。
- 预期工具路径。
- 成功门控条件。

Ground Truth 层：

- 程序化 Gold Metrics。
- Gold Evidence。
- 允许的工具替代路径。
- 数值容差。
- 参考自然语言答案。

### 6.2 血缘

```text
generator_config + seed
→ versioned Parquet
→ DuckDB query
→ gold metric/evidence
→ versioned evaluation case
```

Case 必须绑定数据版本和配置哈希。数据版本变化时，旧 Gold 不得静默继续使用。

### 6.3 异常注入

异常通过配置生成，不通过手工修改最终表格制造：

- 销量下降
- CVR 下降
- 流量异常
- 高退款
- 缺失数据
- 极端值
- 多因素共同变化

异常注入配置既是数据生成依据，也是后续 Case 设计与验证的辅助证据，但不能直接暴露给 Agent。

## 7. Agent 对照设计

### 7.1 控制变量

V1/V2 共用数据、Cases、工具、评分器、模型和模型参数。正式实验的主要自变量是 Prompt 与 Agent 决策策略。

### 7.2 Baseline V1

Baseline 应基本可用，不故意降质。它具有基础任务识别、工具说明和结构化回答，但不强制证据绑定、参数验证、数据不足协议和提交前检查。

### 7.3 Optimized V2

V2 增加：

- 显式任务分类。
- 工具适用边界。
- 参数与指标口径校验。
- 受控多步调用。
- 证据绑定。
- 数据不足处理。
- 固定输出结构。
- 轻量规则检查。

任何优化必须能够映射到已观察错误或预先声明的风险。V2 不得读取 Gold 或评分器内部规则。

### 7.4 模式

Deterministic Mode 实际执行工具并生成结构化回答，不硬编码分数。Real LLM Mode 只替换决策与生成模块。

系统记录结构化 `decision_trace`，不要求或保存模型隐藏思维链。

## 8. Evaluation Framework

### 8.1 状态

所有评分项使用：

- `pass`
- `fail`
- `not_applicable`
- `unavailable`
- `needs_review`

### 8.2 主指标

- Task Success
- Tool Selection Accuracy
- Parameter Accuracy
- Factual Accuracy
- Completeness
- Hallucination Rate
- Instruction Following
- Latency
- Token Usage
- Cost

### 8.3 成功门控

Task Success 不使用任意权重求和。每个 Case 在实验前声明必要检查项，只有必要项全部通过时才判定成功。

### 8.4 Hallucination

Hallucination 的评测对象是可验证事实声明。声明若无法映射到 Tool Result、Gold Evidence 或明确允许的推导，则标记为 unsupported claim。

对主观建议不直接使用“幻觉”标签；缺乏数据支持的建议归入 `E9 Unsupported Recommendation`。

### 8.5 LLM Judge

如后续启用，必须保存：

- Judge 模型与版本。
- Judge Prompt 与哈希。
- Scoring Rubric。
- 原始输出。
- 人工抽检结果。

Judge 结果与确定性指标分开展示。

## 9. Error Taxonomy 与 RCA

错误类型使用 E1–E10：

- E1 Tool Selection Error
- E2 Parameter Error
- E3 Data Retrieval Error
- E4 Calculation Error
- E5 Reasoning Error
- E6 Hallucination
- E7 Instruction Following Error
- E8 Incomplete Answer
- E9 Unsupported Recommendation
- E10 Data Insufficiency Handling Error

每个失败 Case 只有一个 Primary Error。Primary 是最早导致最终失败的可干预原因，其他下游错误记录为 Secondary。

RCA 结构：

```text
Symptom
→ Error
→ Root Cause
→ Intervention
→ Re-evaluation Evidence
```

无法稳定自动归因时，状态为 `needs_review`。

## 10. 统计设计

主要统计单位为 Case。V1/V2 结果按 `case_id` 配对。

- 二元指标使用 Exact McNemar Test。
- 配对比例差使用 Bootstrap Confidence Interval。
- 连续指标使用 Paired Bootstrap，必要时补充 Wilcoxon。
- Error Shift 使用配对转移矩阵。

结果必须报告效应量、95% CI、p 值、样本量与限制。统计不显著时如实展示。

Real LLM 模式允许配置 `n_repeats`。若只有单次运行，报告必须说明模型内在方差未被充分测量。

## 11. Artifact 设计

实验级 `run_manifest.json` 保存：

- Git 状态。
- 数据和 Case 版本。
- Agent、Prompt 和 Tool Schema 版本。
- 模型和非敏感参数。
- Python 与依赖版本。
- 运行时间和状态计数。
- Token / Cost 可用性。

Case 级记录保存：

- 输入。
- 决策轨迹。
- 工具调用和工具结果引用。
- 原始与最终回答。
- 实际 latency 和 Usage。
- 指标明细。
- 错误标签、RCA 和复核状态。

原始 Artifact 不可由分析脚本改写。

## 12. Dashboard 与作品集

Dashboard 包含 Overview、A/B Comparison、Error Analysis、Case Explorer、Statistical Validation 和 Methodology。

Dashboard 只读取冻结结果与 SQL，不在页面加载时调用 Agent。未运行结果显示 `Pending / Not Run`，不可获得的值显示 `Unavailable`。

Portfolio 采用问题、方法、证据、错误、干预、复评和限制的叙事结构。所有效果数字来自 Artifact。

## 13. 目录边界

推荐新增 `app/llm` 和 `app/experiments`，分别隔离模型 adapter 与实验编排；新增 `configs` 保存可版本化配置；新增 `sql` 展示可审阅分析逻辑。

核心逻辑不进入 Notebook、Dashboard 或一次性脚本。

## 14. Phase 管理

唯一主状态机为 PHASE 0–11。

- Evaluation Dataset 是 Phase 1 的内部里程碑，不单独创建 `PHASE 1.5` 状态。
- RCA 是 Phase 6 的内部里程碑，不单独创建 `PHASE 6.5` 状态。
- 每个 Phase 完成后更新 `PROJECT_STATUS.md`，并经用户确认再进入下一阶段。

详细 Roadmap 与验收标准见项目根目录的 `PROJECT_PLAN.md`。

## 15. 风险与缓解

### 15.1 Synthetic Data 过于规则化

风险：Deterministic Agent 可能因模板规律获得虚高结果。

缓解：保留组合异常、缺失数据、干扰条件和未见参数组合；按难度与任务分层报告。

### 15.2 Gold 与实现共享错误

风险：工具和 Gold 使用相同错误函数，导致错误自证。

缓解：关键指标使用独立 SQL 或独立参考实现交叉验证，并设置手算小样本测试。

### 15.3 V2 过拟合 Evaluation Dataset

风险：针对已知 Case 编写 Prompt 或规则。

缓解：冻结开发集与保留集；优化仅使用开发集错误，最终结果在保留集上报告。

### 15.4 LLM 随机性

风险：单次结果不能代表稳定性能。

缓解：记录模型参数，按预算进行重复运行，并明确单次实验限制。

### 15.5 指标定义漂移

风险：看到结果后调整评分规则。

缓解：正式实验前冻结 Case、Rubric、容差和评分器版本。

### 15.6 Dashboard 先于证据

风险：为视觉完整性填充示例结果。

缓解：默认展示 Pending / Not Run；Demo 数据与实验结果严格分离。

## 16. Phase 0 验收

- [x] 项目目标和非目标已定义。
- [x] 用户、场景和五类业务任务已定义。
- [x] Task Taxonomy 已定义。
- [x] 数据与 Ground Truth 血缘已定义。
- [x] 技术架构和模块边界已定义。
- [x] V1/V2 控制变量已定义。
- [x] Evaluation Framework 已定义。
- [x] Error Taxonomy 和 RCA 已定义。
- [x] 统计方法和实验记录协议已定义。
- [x] Dashboard 与 Portfolio 边界已定义。
- [x] Phase 0–11 Roadmap 与阶段门禁已定义。
- [x] 文档完成占位符、矛盾、歧义与范围自审。
- [ ] 用户审阅书面设计。
- [ ] Phase 1 实施计划获批。

## 17. 下一步

1. 对本文档执行占位符、矛盾、歧义和范围检查。
2. 更新 `PROJECT_STATUS.md` 的验证状态。
3. 将 Phase 0 文档作为独立 Git 提交。
4. 请用户审阅书面设计。
5. 用户批准后，编写 Phase 1 的原子化实施计划。
