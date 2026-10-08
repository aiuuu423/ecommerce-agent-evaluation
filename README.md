# 多版本 Agent 效果评测与错误归因分析

当前阶段：`PHASE 3 — Deterministic Baseline V1 Completed / Unscored`。正式离线
Baseline 已运行；Real LLM、Optimized V2 与评分仍为 `Pending / Not Run`。

本项目使用明确标记的 Synthetic E-commerce Data 构建可复现的 Agent Evaluation 闭环。

## Phase 1

所有项目命令通过 Makefile 的 `PYTHON` 入口运行，并要求 Python 3.11+。默认入口是
`python3`；如果系统默认版本较旧，请显式覆盖：

```bash
make PYTHON=python3.12 install
make PYTHON=python3.12 phase1
make PYTHON=python3.12 test
make PYTHON=python3.12 lint
```

Phase 1 的文件锁使用锁定版本的 `filelock`，支持 Linux、macOS 和 Windows；项目命令仍
依赖 `make`，Windows 用户需在提供 GNU Make 的环境（例如 WSL、MSYS2 或 Git Bash）中
运行 Makefile，或直接调用下述 Python CLI。

`requirements.lock` 固定完整的开发与运行时依赖树。修改 `pyproject.toml` 后，使用已安装
`pip-tools` 的 Python 3.11+ 环境运行 `make PYTHON=python3.12 lock`，审阅 lock 文件差异后提交。

`make phase1-data`、`make phase1-cases` 和 `make phase1` 全部委托统一 CLI
`python -m app.data.phase1`。CLI 从 Development 与 Public Validation 配置的 `dataset_version`
动态推导数据目录，并以 Development 的 `dataset_version` 推导 Evaluation Cases 目录；
不要求版本名为 `v1`。若要在干净临时目录中复现且不触碰冻结产物，可覆盖输出根目录：

```bash
make PYTHON=python3.12 PHASE1_OUTPUT_ROOT=/tmp/phase1-clean phase1
```

也可直接调用 CLI：

```bash
python3.12 -m app.data.phase1 \
  --development-config configs/data/synthetic_v1.yaml \
  --public-validation-config configs/data/synthetic_public_validation_v1.yaml \
  --tool-contract configs/evaluation/tool_contract_v1.yaml \
  --output-root /tmp/phase1-clean
```

## Phase 2

Phase 2 提供五个受控工具（`query_product`、`query_sales`、`query_traffic`、
`query_marketing`、`calculate_metrics`）、稳定的 Tool Registry、
Deterministic/OpenAI-compatible 双 adapter、通用 Runner 与结构化 decision trace。
Optimized V2、评分与真实模型实验尚未运行。

无 API Key 离线验证：

```bash
make PYTHON=python3.12 PHASE2_SMOKE_DIR=/tmp/phase2-smoke phase2-smoke
```

offline smoke 仅验证
`Deterministic adapter → Runner → Registry → query_sales → calculate_metrics → final answer`
链路；它不读取 Evaluation Cases、不评分、不生成 Phase 3 run artifact，也不代表正式
Baseline 或 V2 结果。调用方必须在构造 `OpenAICompatibleAdapter` 时显式传入 `base_url`、`api_key`
和 `model`；当前没有从 `.env` 或环境变量读取这些参数并发起真实运行的 CLI、Makefile 或
应用入口。该路径目前只通过 fake/mock transport 测试，未选择或调用真实 provider/model；
offline smoke 不读取 `.env` 或相关环境变量，也不访问网络。

## Phase 3

Deterministic Baseline V1 是无网络、无真实 LLM 的固定策略基线。运行时只把 `case_id`、
`user_input`、Dataset ID/Version 与 Split 投影到最小执行输入；业务标签、Gold、
expected tool calls 和评分字段不会进入 Policy。每个 Case 使用独立 Policy、Adapter 与
Runner，按冻结路由调用受控工具，并仅根据工具结果生成回答。

默认对冻结的 100 个 Cases 和两套数据快照执行：

```bash
make PYTHON=python3.12 phase3-baseline
```

也可通过 `PHASE3_CASE_DIR`、`PHASE3_DEVELOPMENT_DATASET`、
`PHASE3_PUBLIC_VALIDATION_DATASET` 与 `PHASE3_OUTPUT_ROOT` 覆盖输入和输出位置。CLI
等价入口为：

```bash
python3.12 -m app.experiments.baseline_v1 \
  --case-dir data/evaluation_cases/v1 \
  --development-dataset data/synthetic/v1 \
  --public-validation-dataset data/synthetic/public-validation-v1 \
  --output-root outputs/experiment_runs
```

每次成功运行都会创建不可覆盖的 `outputs/experiment_runs/<run_id>/`，包含四个文件：

- `run_manifest.json`：Run、Git、运行环境、Cases、数据集、工具契约、Policy 与输出文件身份。
- `case_runs.jsonl`：按冻结顺序保存逐 Case 状态、回答、工具执行与 decision trace；`usage`
  为 `null`。
- `summary.json`：总数、completed/failed/stopped、`70/30` Split 与评测状态。
- `policy_snapshot.json`：Deterministic V1 的冻结路由、工具路径、指标与模板配置。

最终正式 Run `baseline-v1__20261008T133926Z__5f8a4770` 基于 Git commit
`2ff6bca522fc957d6699df6132dffbde4ace19db`，已完成 `100` 个 Cases，
`100 completed / 0 failed`，Development/Public Validation 为 `70/30`；
`evaluation_status=pending_not_run`、`usage_status=unavailable`。旧 Run
`baseline-v1__20261008T131041Z__2be87afa` 继续保留，但最终引用统一指向新 Run；两次 Run 的
`case_runs.jsonl`、`summary.json` 与 `policy_snapshot.json` 逐字节一致。Baseline 完成不等于
评分完成；Real LLM、Optimized V2 与评分仍为 `Pending / Not Run`。

该 Baseline 的能力边界是冻结关键词路由、有限日期表达、固定工具链和确定性模板，不进行
开放式规划、自检、重试或因果推断；遇到不支持或数据不足的输入会显式返回限制。正式
Artifact 保留在上述输出目录且遵循仓库现有忽略规则，身份以 `PROJECT_STATUS.md` 记录的
SHA-256 为准。

## 数据真实性

本项目当前仅使用 `Synthetic E-commerce Data`。数据由两份版本化 YAML 配置和固定 Seed
程序化生成，不代表真实企业经营数据，也不得据此推断真实用户或业务表现。

## Phase 1 产物

| 产物 | Development | Public Validation |
|---|---:|---:|
| Dataset ID | `e1e81533c25e03e5` | `c17d4926cfa7cb26` |
| products | 40 | 40 |
| customers | 300 | 300 |
| traffic | 4,800 | 4,800 |
| marketing | 4,800 | 4,800 |
| orders | 26,941 | 26,790 |

- 两套五表 Parquet 快照均包含 Dataset Manifest、文件哈希与数据质量报告。
- 独立 SHA-256 基线记录 16 个 Phase 1 产物的相对路径、哈希、Dataset IDs 与
  Case Set ID；端到端测试使用临时干净构建与该提交基线比较。
- Case Set 共 100 个 Cases，Development/Public Validation 为 `70/30`；每个业务任务
  固定 `14/6`，Public Validation 中十类能力各 `3`，难度分布为 `9/12/9`。
- `statistical_cluster_id` 将同一业务任务与主要能力下的复述变体归入同一统计簇。
  置信区间、显著性检验与有效样本量必须按 cluster 计算，不得把同簇 Case 当作独立样本。
- Public Validation 是公开、可反复使用的开发验证集，不是盲测，也不得称为盲测。
  只有在 Optimized V2 方案、Prompt、实现与评分协议冻结后，才生成此前未见的最终盲测集，
  用于最终确认。
- 冻结产物的 Case Set ID 与 SHA-256 见
  `data/synthetic/phase1_sha256_baseline.json`。
- Phase 1 不需要 API Key；Phase 3 Deterministic Baseline 已离线运行，但 Real LLM、
  Optimized V2、评分、Error Analysis 与 Statistical Validation 仍为
  `Pending / Not Run`，Usage 为 `Unavailable`。
