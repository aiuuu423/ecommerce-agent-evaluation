# PROJECT STATUS

## Current Phase

`PHASE 1 — 模拟数据与 Evaluation Dataset`

状态：`Completed / Awaiting Review`

Phase 0 设计已经用户批准。Phase 1 已完成独立 Development/Holdout 数据快照与
100 个冻结 Evaluation Cases；三个 Makefile Phase 1 目标统一委托单一 CLI，并从配置的
`dataset_version` 动态推导输出目录；项目元数据仅公开 `build-phase1` 入口。快照身份绑定
生成器源码，跨平台文件锁与只读数据探索 Notebook 已完成真实执行验证。

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
- [x] 建立只读数据探索 Notebook，并隔离 Holdout 查询句柄。
- [x] 建立统一 Phase 1 CLI 与可覆盖输出根目录的 Makefile 入口。
- [x] 仅公开 `build-phase1 = app.data.phase1:main` console script，并验证三个 Makefile
  目标继续委托统一 CLI。
- [x] 使用锁定版本的 `filelock` 支持 Linux、macOS 与 Windows 并发发布。
- [x] 使用非 `v1` 配置端到端验证动态输出目录。
- [x] 验证干净重建的 16 个产物文件与冻结版本逐字节一致。
- [x] 验证 Gold SQL 不读取生成配置、异常 ID 或异常倍率。
- [x] 完成 Phase 1 全量测试、静态检查与 Notebook 真实执行。

## Pending

- [ ] 用户审阅 Phase 1 实际交付后进入 Phase 2。

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
- Parquet 字节一致性要求使用锁定依赖中的相同 Pandas/PyArrow 写入器版本。
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

- `make PYTHON=<python3.12> PHASE1_OUTPUT_ROOT=<clean-dir> phase1`：PASS。
- 干净输出与冻结产物比较：`16` 个文件，`0` 个字节差异。
- Development Dataset ID：`e1e81533c25e03e5`。
- Development 行数：products `40`、customers `300`、traffic `4,800`、
  marketing `4,800`、orders `26,941`。
- Holdout Dataset ID：`c17d4926cfa7cb26`。
- Holdout 行数：products `40`、customers `300`、traffic `4,800`、
  marketing `4,800`、orders `26,790`。
- Generator Source SHA-256：
  `cdbc078dca3b30e7d00552e275d22171868658e8dd9c805969164e13f56722f0`。
- Case Set ID：`35d8734343a1492d`；JSONL SHA-256：
  `4204ca993981554869e1f5627610846a3687cb9b9a7aea299644ba1b2f9484ea`。
- Cases：`100`，Development/Holdout Split 为 `70/30`。
- Gold 独立性与端到端复现定向测试：`2 passed`。
- Console script 唯一性与 Makefile 统一 CLI 委托定向测试：`5 passed`。
- 全量测试：`357 passed`。
- Notebook 结构与真实执行测试：`6 passed`（项目根目录与 `notebooks/` 两种工作目录）。
- `python -m ruff check app tests notebooks`：`All checks passed!`。
- `git diff --check`：PASS。

## Next Step

等待用户审阅 Phase 1 实际交付；不提前实现 Phase 2 工具。
