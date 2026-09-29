# PROJECT STATUS

## Current Phase

`PHASE 2 — Agent 基础能力`

状态：`Completed / Awaiting Review`

Phase 0 设计已经用户批准。Phase 1 已完成独立 Development/Public Validation 数据快照与
100 个冻结 Evaluation Cases / 50 个 statistical clusters（每簇 2 Cases）；三个 Makefile
Phase 1 目标统一委托单一 CLI，并从配置的 `dataset_version` 动态推导输出目录；项目元数据
公开 `build-phase1` 与 `phase2-smoke` 入口。快照身份绑定生成器源码，跨平台文件锁与只读
数据探索 Notebook 已完成真实执行验证。Phase 2 已完成五个受控工具、Tool Registry、
Deterministic/OpenAI-compatible 双 adapter、通用 Runner、结构化 decision trace 与无网络
offline smoke；Real LLM 与 V1/V2 实验仍为 `Pending / Not Run`。

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
- [x] 确认 V1/V2 结果按 `case_id` 配对，统计推断、Cluster Bootstrap 与有效样本量按
  `statistical_cluster_id` 处理；二元配对的簇级汇总或聚类方法在 Phase 7 实施前冻结。
- [x] 确认 Dashboard 只读取冻结实验结果。
- [x] 创建 `PROJECT_PLAN.md`。
- [x] 创建 Phase 0 设计文档。
- [x] 用户批准 Phase 0 书面设计。
- [x] 创建 Phase 1 原子化实施计划。
- [x] 生成并验证 Development 与 Public Validation 两套独立 Synthetic Data 快照。
- [x] 冻结 100 个 Evaluation Cases / 50 个 statistical clusters（每簇 2 Cases；每任务
  14/6 Split；Public Validation 十类能力各 3，难度 9/12/9）。
- [x] 增加 `statistical_cluster_id`；统计推断、Cluster Bootstrap 与有效样本量按
  cluster 计算。
- [x] 版本化评测工具契约并修正 GMV、Next-week 与 Adversarial 工具路径。
- [x] 以不可变目录原子发布并提交 Case JSONL 与 Manifest。
- [x] 建立只读数据探索 Notebook，并隔离 Public Validation 查询句柄。
- [x] 建立统一 Phase 1 CLI 与可覆盖输出根目录的 Makefile 入口。
- [x] 仅公开 `build-phase1 = app.data.phase1:main` console script，并验证三个 Makefile
  目标继续委托统一 CLI。
- [x] 使用锁定版本的 `filelock` 支持 Linux、macOS 与 Windows 并发发布。
- [x] 使用非 `v1` 配置端到端验证动态输出目录。
- [x] 验证干净重建的 16 个产物文件与冻结版本逐字节一致。
- [x] 验证 Gold SQL 不读取生成配置、异常 ID 或异常倍率。
- [x] 完成 Phase 1 全量测试、静态检查与 Notebook 真实执行。
- [x] 清理旧数据分区语义；Phase 1 统一使用 Development/Public Validation，
  Notebook 避免暴露预设答案与异常配置，Phase 0 记录现行分阶段治理并明确取代旧决策。
- [x] 用户确认 Phase 1 交付并要求编写 Phase 2 原子化实施计划。
- [x] 创建并自审 Phase 2 Agent 基础能力实施计划。
- [x] 实现 `query_product`、`query_sales`、`query_traffic`、`query_marketing` 与
  `calculate_metrics` 五个受控工具及强类型输入/输出边界。
- [x] 建立稳定 Tool Registry；固定五工具 Schema 导出并对 handler 输出执行二次严格验证。
- [x] 实现 Deterministic 与 OpenAI-compatible 双 adapter；后者当前仅通过 fake/mock
  transport 验证。
- [x] 实现通用 Agent Runner、结构化 decision trace、错误归一化与多轮 Usage 累加。
- [x] 建立无 API Key、无网络的 offline smoke，贯通
  `query_sales → calculate_metrics → final answer`。
- [x] 通过 Phase 1 冻结资产门禁，确认 16 个冻结资产、Dataset IDs、Case Set ID、
  manifests、baseline 与工具契约未漂移。

## Pending

- [ ] 用户审阅 Phase 2 实现与验证证据。
- [ ] 审阅通过后进入 Phase 3；在此之前不宣称已运行 V1/V2 或获得评测结果。
- [ ] 选择真实 LLM provider/model 与预算后，另行运行真实模型验证。

## Evaluation Results

| 项目 | 当前状态 |
|---|---|
| Synthetic Dataset | Completed / Validated |
| Evaluation Dataset | Completed / Validated |
| Real LLM | Pending / Not Run |
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
- Public Validation 是公开开发验证集，不是盲测；V2 冻结后才生成此前未见的
  最终盲测集。
- Parquet 字节一致性要求使用锁定依赖中的相同 Pandas/PyArrow 写入器版本。
- OpenAI-compatible adapter 仅通过 fake/mock transport 测试，尚未选择或调用真实
  provider/model，Real LLM 为 `Pending / Not Run`。
- 尚未实现或运行 Baseline V1、Optimized V2、A/B Evaluation；offline smoke 不读取
  Evaluation Cases、不评分、不生成 Phase 3 run artifact。
- LLM Judge 仅为可选扩展，当前未选择 Judge 模型。
- Phase 7 实施前仍须冻结簇级二元汇总或适合聚类数据的配对方法、Cluster Bootstrap
  细节、有效样本量口径与退化情形处理；不得在 100 个 Case 对上直接运行假设相互独立的
  Exact McNemar Test。

## Decisions

- 数据引擎：`DuckDB + Parquet`。
- 主评分路线：规则优先的混合评分。
- 首版业务任务范围：固定五类。
- Evaluation Cases：首批 100 Cases / 50 statistical clusters（每簇 2 Cases），质量
  验证后目标扩展到 200 Cases。
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
- Public Validation Dataset ID：`c17d4926cfa7cb26`。
- Public Validation 行数：products `40`、customers `300`、traffic `4,800`、
  marketing `4,800`、orders `26,790`。
- Generator Source SHA-256：
  `bb07f7c17d8b339cbd4eb5b393e5eb73e49dfa6d781ba27b957374af33af2262`。
- Case Set ID：`ecebfe8b691271fd`；JSONL SHA-256：
  `f32e7822ca9fa7b30fee1ff2017cb4d87d7503a4c49187dfa7c475ca282cfa03`。
- Cases：`100`；statistical clusters：`50`（每簇 `2` Cases）；Development/Public
  Validation Split 为 `70/30`；每任务 `14/6`，Public Validation 十类能力各 `3`、难度
  `9/12/9`。
- Gold 独立性与端到端复现定向测试：`2 passed`。
- Console script 唯一性与 Makefile 统一 CLI 委托定向测试：`5 passed`。
- Phase 1 验收时全量测试：`357 passed`。
- Notebook 结构与真实执行测试：`6 passed`（项目根目录与 `notebooks/` 两种工作目录）。
- `python -m ruff check app tests notebooks`：`All checks passed!`。
- 旧术语搜索：仅 Phase 0 的“Public Validation 不是最终盲测集”和历史决策说明命中。
- 变更 Markdown 检查：`markdownlint-cli2`（忽略既有代码块制表符、行长与列表空行规则）
  `0 error(s)`。
- `git diff --check`：PASS。
- Phase 1 冻结资产门禁：`3 passed in 5.57s`。
- Phase 2 定向测试：`152 passed in 13.35s`。
- 全量测试：`512 passed, 2 warnings in 153.23s`；两条 warning 均为只读用户目录触发的
  IPython 临时目录提示。
- Ruff：`All checks passed!`。
- Registry：恰好 `5` 个工具，顺序为 `calculate_metrics`、`query_marketing`、
  `query_product`、`query_sales`、`query_traffic`；canonical JSON Schema SHA-256 为
  `9a501e7839e32df5bfbf965d3bfa3921ce6212456e56170bf09750e8ff368343`。
- offline smoke：`status=completed`、`adapter=deterministic`、
  `dataset_id=e1e81533c25e03e5`、工具链为
  `query_sales → calculate_metrics`、`trace_event_count=11`、
  `gmv_change_rate=-0.022536435249076572`；环境中未提供 LLM 配置且未访问网络。
- Phase 2 文档更新前 `git diff --check`：PASS，工作树无未提交变更。

## Next Step

等待用户审阅 Phase 2 实现与验证证据；审阅通过后进入 Phase 3。
