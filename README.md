# 多版本 Agent 效果评测与错误归因分析

当前阶段：PHASE 1 规划已批准，实验结果均为 `Pending / Not Run`。

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

`requirements.lock` 固定完整的开发与运行时依赖树。修改 `pyproject.toml` 后，使用已安装
`pip-tools` 的 Python 3.11+ 环境运行 `make PYTHON=python3.12 lock`，审阅 lock 文件差异后提交。

`make phase1` 通过统一 CLI `python -m app.data.phase1` 读取 Development 与 Holdout
两份配置，依次构建两套数据快照和 Evaluation Cases。若要在干净临时目录中复现且不触碰
冻结产物，可覆盖输出根目录：

```bash
make PYTHON=python3.12 PHASE1_OUTPUT_ROOT=/tmp/phase1-clean phase1
```

也可直接调用 CLI：

```bash
python3.12 -m app.data.phase1 \
  --development-config configs/data/synthetic_v1.yaml \
  --holdout-config configs/data/synthetic_holdout_v1.yaml \
  --tool-contract configs/evaluation/tool_contract_v1.yaml \
  --output-root /tmp/phase1-clean
```

## 数据真实性

本项目当前仅使用 `Synthetic E-commerce Data`。数据由两份版本化 YAML 配置和固定 Seed
程序化生成，不代表真实企业经营数据，也不得据此推断真实用户或业务表现。

## Phase 1 产物

| 产物 | Development | Holdout |
|---|---:|---:|
| Dataset ID | `e1e81533c25e03e5` | `c17d4926cfa7cb26` |
| products | 40 | 40 |
| customers | 300 | 300 |
| traffic | 4,800 | 4,800 |
| marketing | 4,800 | 4,800 |
| orders | 26,941 | 26,790 |

- 两套五表 Parquet 快照均包含 Dataset Manifest、文件哈希与数据质量报告。
- Case Set ID 为 `35d8734343a1492d`，共 100 个 Cases，Development/Holdout 为
  `70/30`，JSONL SHA-256 为
  `4204ca993981554869e1f5627610846a3687cb9b9a7aea299644ba1b2f9484ea`。
- Holdout 只用于最终评测；开发过程不得读取其 Gold 内容调优 Agent。
- Phase 1 不需要 API Key，尚未接入真实模型或运行 Agent。Baseline、Optimized、
  A/B Metrics、Error Analysis、Statistical Validation、Latency 均为
  `Pending / Not Run`，Token Usage 与 Cost 为 `Unavailable`。
