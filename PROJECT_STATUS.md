# PROJECT STATUS

## Current Phase

`PHASE 0 — 项目架构与技术方案`

状态：`Awaiting Review`

Phase 0 设计已形成书面文档，尚未开始 Phase 1，也尚未创建业务代码。

## Completed

- [x] 明确项目目标、目标岗位与能力展示重点。
- [x] 确认采用评测优先的模块化单体架构。
- [x] 确认使用 Python 3.11+、DuckDB、Parquet、Pydantic、SciPy、Plotly 和 Streamlit。
- [x] 确认支持 Mock / Deterministic Mode 与 Real LLM Mode。
- [x] 确认先跑通 Mock 闭环，再启用真实模型实验。
- [x] 确认首版覆盖五类电商经营分析任务。
- [x] 确认采用业务任务与评测能力双维度 Task Taxonomy。
- [x] 确认采用规则优先的混合评分。
- [x] 确认使用 E1–E10 Error Taxonomy 和因果式 RCA。
- [x] 确认使用配对比较、Exact McNemar Test 与 Paired Bootstrap。
- [x] 确认 Dashboard 只读取冻结实验结果。
- [x] 创建 `PROJECT_PLAN.md`。
- [x] 创建 Phase 0 设计文档。

## Pending

- [ ] 用户审阅 Phase 0 书面文档。
- [ ] 根据审阅意见修订设计。
- [ ] 创建 Phase 1 的原子化实施计划。
- [ ] 初始化项目代码结构。
- [ ] 开始 Synthetic Data Schema 与生成器开发。
- [ ] 建立首批 Evaluation Cases。

## Evaluation Results

| 项目 | 当前状态 |
|---|---|
| Synthetic Dataset | Pending / Not Run |
| Evaluation Dataset | Pending / Not Run |
| Baseline Experiment | Pending / Not Run |
| Optimized Experiment | Pending / Not Run |
| A/B Metrics | Pending / Not Run |
| Error Analysis | Pending / Not Run |
| Statistical Validation | Pending / Not Run |
| Latency | Pending / Not Run |
| Token Usage | Unavailable |
| Cost | Unavailable |

## Known Issues

- 当前尚无代码、测试、数据或实验产物。
- 当前尚未验证本机 Python 版本和依赖兼容性。
- Real LLM provider、具体模型和预算尚未确定；不影响 Mock 路径设计。
- Evaluation Dataset 的 Case 分布与难度比例将在 Phase 1 实施计划中冻结。
- LLM Judge 仅为可选扩展，当前未选择 Judge 模型。

## Decisions

- 数据引擎：`DuckDB + Parquet`。
- 主评分路线：规则优先的混合评分。
- 首版业务任务范围：固定五类。
- Evaluation Cases：首批 100，质量验证后目标扩展到 200。
- Phase 状态机：只使用 `PHASE 0–11`。
- `PHASE 1.5` 作为 Phase 1 内部里程碑。
- `PHASE 6.5` 作为 Phase 6 内部里程碑。
- 未运行结果显示 `Pending / Not Run`。
- 无真实 Usage 或价格配置时显示 `Unavailable`。

## Validation

本文件当前只记录设计阶段状态。

- 文档存在性检查：通过，三份 Phase 0 文档均存在且非空。
- 占位符、矛盾、歧义与范围检查：通过。
- Git 提交：本次 Phase 0 文档提交包含且仅包含三份设计文档；提交哈希以 Git 历史为准。
- 代码测试：不适用，当前尚无代码。
- 实验验证：不适用，当前尚未运行实验。

## Next Step

完成 Phase 0 文档自审与 Git 提交，然后由用户审阅。

用户明确批准书面设计后，进入实施计划阶段；在实施计划获批前，不开始 Phase 1 编码。
