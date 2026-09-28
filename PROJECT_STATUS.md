# PROJECT STATUS

## Current Phase

`PHASE 1 — 模拟数据与 Evaluation Dataset`

状态：`In Progress / Task 11 Completed / Task 12 Not Started`

Phase 0 设计已经用户批准。Phase 1 已完成独立 Development/Holdout 数据快照与
100 个冻结 Evaluation Cases；Task 11 已完成，Task 12 尚未开始。

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
- [x] 用户批准 Phase 0 书面设计。
- [x] 创建 Phase 1 原子化实施计划。
- [x] 生成并验证 Development 与 Holdout 两套独立 Synthetic Data 快照。
- [x] 冻结 100 个 Evaluation Cases（每任务 14/6 Split、每能力 10）。
- [x] 版本化评测工具契约并修正 GMV、Next-week 与 Adversarial 工具路径。
- [x] 以不可变目录原子发布并提交 Case JSONL 与 Manifest。

## Pending

- [ ] 建立数据探索 Notebook。
- [ ] 完成 Phase 1 文档、全量验证与阶段收尾。

## Evaluation Results

| 项目 | 当前状态 |
|---|---|
| Synthetic Dataset | Completed / Validated |
| Evaluation Dataset | Completed / Validated |
| Baseline Experiment | Pending / Not Run |
| Optimized Experiment | Pending / Not Run |
| A/B Metrics | Pending / Not Run |
| Error Analysis | Pending / Not Run |
| Statistical Validation | Pending / Not Run |
| Latency | Pending / Not Run |
| Token Usage | Unavailable |
| Cost | Unavailable |

## Known Issues

- Synthetic Data 和 Evaluation Cases 为模拟数据，只用于受控评测。
- Holdout 仅用于最终评测；开发过程不得读取其 Gold 内容调优 Agent。
- Real LLM provider、具体模型和预算尚未确定；不影响 Mock 路径设计。
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

- Task 11 定向测试：`71 passed`。
- Development Dataset ID：`e1e81533c25e03e5`。
- Holdout Dataset ID：`c17d4926cfa7cb26`。
- Case Set ID：`5e250d180ca6552c`；JSONL SHA-256：
  `dbeab9fc9466d71fb4a425b01ccc4709674c81fe361839896d4799fcb84babae`。
- 全量测试：`342 passed`。
- Ruff：`All checks passed`。

## Next Step

执行 Task 12，建立只读数据探索 Notebook；不提前实现 Phase 2 工具。
