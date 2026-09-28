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
```

`requirements.lock` 固定完整的开发与运行时依赖树。修改 `pyproject.toml` 后，使用已安装
`pip-tools` 的 Python 3.11+ 环境运行 `make PYTHON=python3.12 lock`，审阅 lock 文件差异后提交。

Phase 1 不需要 API Key。真实模型接入将在后续 Phase 实现。
