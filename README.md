# 多版本 Agent 效果评测与错误归因分析

当前阶段：`PHASE 1 — Completed / Awaiting Review`。Agent 实验结果仍为
`Pending / Not Run`。

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
  只有在 Optimized V2 方案与实现冻结后，才生成此前未见的 Final Holdout，用于最终确认。
- 冻结产物的 Case Set ID 与 SHA-256 见
  `data/synthetic/phase1_sha256_baseline.json`。
- Phase 1 不需要 API Key，尚未接入真实模型或运行 Agent。Baseline、Optimized、
  A/B Metrics、Error Analysis、Statistical Validation、Latency 均为
  `Pending / Not Run`，Token Usage 与 Cost 为 `Unavailable`。
