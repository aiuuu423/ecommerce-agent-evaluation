# 多版本 Agent 效果评测与错误归因分析

当前阶段：PHASE 1 规划已批准，实验结果均为 `Pending / Not Run`。

本项目使用明确标记的 Synthetic E-commerce Data 构建可复现的 Agent Evaluation 闭环。

## Phase 1

```bash
python3 -m pip install -e ".[dev]"
make phase1
python3 -m pytest
```

Phase 1 不需要 API Key。真实模型接入将在后续 Phase 实现。
