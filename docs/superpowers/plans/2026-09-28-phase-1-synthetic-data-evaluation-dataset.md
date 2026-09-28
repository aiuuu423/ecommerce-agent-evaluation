# Phase 1 Synthetic Data and Evaluation Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可复现、明确标记为 Synthetic 的五表电商数据快照，并从冻结数据与独立 Gold 查询程序化生成首批 100 个 Evaluation Cases。

**Architecture:** Phase 1 分为两个顺序里程碑。里程碑 A 使用版本化 YAML 配置和固定 Seed 生成 Pandas DataFrame，经过 Pydantic Schema、业务不变量与独立指标检查后写入 Parquet 和 Manifest；里程碑 B 使用 DuckDB 只读查询冻结快照，生成带 Gold Evidence、成功门控和开发集/保留集标签的 JSONL Cases。业务生成逻辑、Gold SQL 和 Case 模板相互隔离，降低共享错误导致的“自证正确”风险。

**Tech Stack:** Python 3.11+、Pandas、NumPy、Pydantic v2、DuckDB、PyArrow、PyYAML、Pytest、Ruff、Jupyter/nbformat

---

## 实施边界

### 本计划包含

- Python 项目最小骨架与依赖声明。
- 五张 Synthetic Data 表的 Schema。
- 版本化生成配置与固定 Seed。
- 可解释的异常注入规则。
- Parquet 快照、文件哈希和 Dataset Manifest。
- DuckDB 只读 Catalog。
- 独立 SQL Gold 查询。
- Evaluation Case Schema。
- 100 个可复现 Cases。
- 开发集与保留集划分。
- 数据质量报告与探索 Notebook。
- Phase 1 文档和状态更新。

### 本计划不包含

- Agent Tool 的正式实现。
- Baseline 或 Optimized Agent。
- Agent Run 与模型 API。
- Evaluation Engine 的评分实现。
- Error Taxonomy 自动归因。
- 统计检验和 Dashboard。

这些内容分别属于 Phase 2–8。Phase 1 只定义后续模块可依赖的数据契约。

## 文件结构

### 创建

```text
.env.example
.gitignore
Makefile
README.md
pyproject.toml
app/__init__.py
app/config/__init__.py
app/data/__init__.py
app/data/case_generator.py
app/data/config.py
app/data/database.py
app/data/generator.py
app/data/gold.py
app/data/manifest.py
app/data/metrics.py
app/data/schemas.py
app/data/validation.py
configs/data/synthetic_v1.yaml
data/evaluation_cases/.gitkeep
data/results/.gitkeep
data/synthetic/.gitkeep
notebooks/01_data_exploration.ipynb
sql/gold/gmv_change.sql
sql/gold/product_anomalies.sql
sql/gold/conversion_decline.sql
sql/gold/products_to_watch.sql
sql/gold/next_week_priorities.sql
tests/conftest.py
tests/fixtures/hand_checked_metrics.json
tests/test_case_generator.py
tests/test_data_config.py
tests/test_data_generation.py
tests/test_data_manifest.py
tests/test_data_schemas.py
tests/test_data_validation.py
tests/test_database.py
tests/test_gold.py
tests/test_notebook.py
tests/test_project_setup.py
```

### 修改

```text
PROJECT_STATUS.md
```

### 生成但不提交

```text
data/synthetic/v1/*.parquet
data/synthetic/v1/manifest.json
data/synthetic/v1/data_quality_report.json
data/evaluation_cases/evaluation_cases_v1.jsonl
data/evaluation_cases/evaluation_cases_v1.manifest.json
```

生成产物不作为手写源文件提交；它们必须能由配置和代码重新构建。正式发布时是否附带一份小型快照，在 Phase 9 的发布策略中决定。

## 固定数据契约

### 表结构

| 表 | 主键或粒度 | 必需字段 |
|---|---|---|
| products | `product_id` | `product_name`, `category`, `price`, `cost`, `launch_date` |
| customers | `customer_id` | `is_new_customer`, `region`, `channel` |
| traffic | `date + product_id` | `impressions`, `clicks`, `visits`, `is_missing` |
| marketing | `date + product_id + campaign_id` | `spend` |
| orders | `order_id` | `product_id`, `customer_id`, `order_date`, `quantity`, `unit_price`, `revenue`, `is_refund`, `status` |

### 指标口径

- `GMV = sum(revenue)`，包含后续发生退款的已支付订单。
- `Orders = count(distinct order_id)`，只统计 `status IN ('paid', 'refunded')`。
- `AOV = GMV / Orders`。
- `CTR = clicks / impressions`。
- `CVR = Orders / visits`。
- `Refund Rate = refunded_orders / Orders`。
- `ROAS = GMV / marketing_spend`。
- 分母为 0 时返回 `null`，不得返回 0 或无穷大。

### 时间窗口

- 总数据范围：固定 120 天。
- 观察窗口：最后 30 个完整日期。
- 对照窗口：观察窗口之前紧邻的 30 个完整日期。
- 所有窗口边界均使用闭区间，并由同一配置提供。

### Case 分布

- 总数：100。
- 五类业务任务：每类 20。
- 十类能力标签：每类 10 个 Primary Capability。
- 难度：`easy=30`、`medium=40`、`hard=30`。
- Split：`development=70`、`holdout=30`。
- 每个业务任务在 development 中 14 个、holdout 中 6 个。
- Holdout 只能用于最终评测，不能用于 Prompt 优化。

---

### Task 1: 建立最小 Python 项目骨架

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `Makefile`
- Create: `README.md`
- Create: `app/__init__.py`
- Create: `app/config/__init__.py`
- Create: `app/data/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/synthetic/.gitkeep`
- Create: `data/evaluation_cases/.gitkeep`
- Create: `data/results/.gitkeep`

- [x] **Step 1: 写项目元数据测试**

在 `tests/test_project_setup.py` 写入：

```python
from pathlib import Path

import tomllib


ROOT = Path(__file__).parents[1]


def test_project_metadata_declares_python_311_and_phase_1_dependencies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]

    assert project["requires-python"] == ">=3.11"
    dependencies = " ".join(project["dependencies"]).lower()
    for package in ("pandas", "numpy", "pydantic", "duckdb", "pyarrow", "pyyaml"):
        assert package in dependencies


def test_generated_outputs_are_ignored_but_keep_files_are_trackable() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/synthetic/*" in ignore
    assert "!data/synthetic/.gitkeep" in ignore
    assert "data/evaluation_cases/*.jsonl" in ignore
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_project_setup.py -v
```

Expected: FAIL，因为 `pyproject.toml` 和 `.gitignore` 尚不存在。

- [x] **Step 3: 创建最小项目配置**

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ecommerce-agent-evaluation"
version = "0.1.0"
description = "Synthetic e-commerce Agent evaluation and error analysis"
requires-python = ">=3.11"
dependencies = [
  "duckdb>=1.1,<2",
  "numpy>=2.0,<3",
  "pandas>=2.2,<3",
  "pyarrow>=17,<20",
  "pydantic>=2.9,<3",
  "pydantic-settings>=2.5,<3",
  "pyyaml>=6.0,<7",
]

[project.optional-dependencies]
dev = [
  "nbformat>=5.10,<6",
  "pytest>=8.3,<9",
  "ruff>=0.7,<1",
]

[project.scripts]
build-phase1-data = "app.data.generator:main"
build-evaluation-cases = "app.data.case_generator:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

`.gitignore`：

```gitignore
.DS_Store
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

data/synthetic/*
!data/synthetic/.gitkeep
data/evaluation_cases/*.jsonl
data/evaluation_cases/*.manifest.json
data/results/*
!data/results/.gitkeep
```

`.env.example`：

```dotenv
# Phase 1 does not require an API key.
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

`Makefile`：

```makefile
.PHONY: install test lint phase1-data phase1-cases phase1

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

lint:
	python3 -m ruff check app tests

phase1-data:
	python3 -m app.data.generator --config configs/data/synthetic_v1.yaml

phase1-cases:
	python3 -m app.data.case_generator --dataset data/synthetic/v1

phase1: phase1-data phase1-cases
```

`README.md` 先写入最小真实状态：

````markdown
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
````

三个 `__init__.py` 保持空文件。`tests/conftest.py`：

```python
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parents[1]
```

- [x] **Step 4: 安装依赖并运行测试**

Run:

```bash
python3 -m pip install -e ".[dev]" --break-system-packages
python3 -m pytest tests/test_project_setup.py -v
python3 -m ruff check tests/test_project_setup.py
```

Expected: 全部 PASS；安装命令不得要求 API Key。

- [x] **Step 5: 提交骨架**

```bash
git add pyproject.toml .gitignore .env.example Makefile README.md app tests data
git commit -m "build: initialize phase 1 python project"
```

---

### Task 2: 定义五表数据 Schema

**Files:**
- Create: `app/data/schemas.py`
- Create: `tests/test_data_schemas.py`

- [x] **Step 1: 写 Schema 失败测试**

`tests/test_data_schemas.py`：

```python
from datetime import date

import pytest
from pydantic import ValidationError

from app.data.schemas import OrderRow, ProductRow, TrafficRow


def test_product_rejects_cost_above_price() -> None:
    with pytest.raises(ValidationError):
        ProductRow(
            product_id="P001",
            product_name="商品 001",
            category="home",
            price=50,
            cost=60,
            launch_date=date(2026, 1, 1),
        )


def test_traffic_rejects_clicks_above_impressions() -> None:
    with pytest.raises(ValidationError):
        TrafficRow(
            date=date(2026, 1, 1),
            product_id="P001",
            impressions=100,
            clicks=120,
            visits=120,
            is_missing=False,
        )


def test_order_revenue_must_equal_quantity_times_unit_price() -> None:
    with pytest.raises(ValidationError):
        OrderRow(
            order_id="O000001",
            product_id="P001",
            customer_id="C001",
            order_date=date(2026, 1, 1),
            quantity=2,
            unit_price=50,
            revenue=80,
            is_refund=False,
            status="paid",
        )
```

- [x] **Step 2: 运行测试确认导入失败**

Run:

```bash
python3 -m pytest tests/test_data_schemas.py -v
```

Expected: FAIL，提示 `app.data.schemas` 不存在。

- [x] **Step 3: 实现严格 Schema**

`app/data/schemas.py`：

```python
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductRow(StrictModel):
    product_id: str = Field(pattern=r"^P\d{3}$")
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: Decimal = Field(gt=0)
    cost: Decimal = Field(gt=0)
    launch_date: date

    @model_validator(mode="after")
    def validate_margin(self) -> "ProductRow":
        if self.cost >= self.price:
            raise ValueError("cost must be lower than price")
        return self


class CustomerRow(StrictModel):
    customer_id: str = Field(pattern=r"^C\d{4}$")
    is_new_customer: bool
    region: str = Field(min_length=1)
    channel: str = Field(min_length=1)


class TrafficRow(StrictModel):
    date: date
    product_id: str
    impressions: int | None = Field(default=None, ge=0)
    clicks: int | None = Field(default=None, ge=0)
    visits: int | None = Field(default=None, ge=0)
    is_missing: bool

    @model_validator(mode="after")
    def validate_funnel(self) -> "TrafficRow":
        values = (self.impressions, self.clicks, self.visits)
        if self.is_missing:
            if any(value is not None for value in values):
                raise ValueError("missing traffic rows must use null metrics")
            return self
        if any(value is None for value in values):
            raise ValueError("non-missing traffic rows require all metrics")
        if not self.impressions >= self.clicks:
            raise ValueError("clicks cannot exceed impressions")
        if self.visits < self.clicks:
            raise ValueError("visits cannot be lower than clicks")
        return self


class MarketingRow(StrictModel):
    date: date
    product_id: str
    campaign_id: str = Field(pattern=r"^M\d{3}$")
    spend: Decimal = Field(ge=0)


class OrderRow(StrictModel):
    order_id: str = Field(pattern=r"^O\d{6}$")
    product_id: str
    customer_id: str
    order_date: date
    quantity: int = Field(ge=1, le=20)
    unit_price: Decimal = Field(gt=0)
    revenue: Decimal = Field(gt=0)
    is_refund: bool
    status: Literal["paid", "refunded"]

    @model_validator(mode="after")
    def validate_order(self) -> "OrderRow":
        if self.revenue != self.unit_price * self.quantity:
            raise ValueError("revenue must equal unit_price * quantity")
        if self.is_refund != (self.status == "refunded"):
            raise ValueError("refund flag and status must agree")
        return self
```

- [x] **Step 4: 运行 Schema 测试与静态检查**

Run:

```bash
python3 -m pytest tests/test_data_schemas.py -v
python3 -m ruff check app/data/schemas.py tests/test_data_schemas.py
```

Expected: 全部 PASS。

- [x] **Step 5: 提交 Schema**

```bash
git add app/data/schemas.py tests/test_data_schemas.py
git commit -m "feat(data): define strict ecommerce table schemas"
```

---

### Task 3: 定义版本化生成配置

**Files:**
- Create: `configs/data/synthetic_v1.yaml`
- Create: `app/data/config.py`
- Create: `tests/test_data_config.py`

- [x] **Step 1: 写配置解析失败测试**

`tests/test_data_config.py`：

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data.config import SyntheticDataConfig, load_data_config


CONFIG = Path("configs/data/synthetic_v1.yaml")


def test_loads_versioned_config() -> None:
    config = load_data_config(CONFIG)
    assert config.dataset_version == "v1"
    assert config.seed == 20260928
    assert config.days == 120
    assert config.product_count >= 7
    assert len(config.anomalies) == 7


def test_rejects_anomaly_outside_dataset_window() -> None:
    with pytest.raises(ValidationError):
        SyntheticDataConfig.model_validate(
            {
                "dataset_version": "v1",
                "seed": 1,
                "start_date": "2026-01-01",
                "days": 30,
                "product_count": 10,
                "customer_count": 20,
                "categories": ["home"],
                "regions": ["north"],
                "channels": ["organic"],
                "anomalies": [
                    {
                        "anomaly_id": "A1",
                        "kind": "traffic_drop",
                        "product_id": "P001",
                        "start_day": 40,
                        "end_day": 45,
                        "multiplier": 0.5,
                    }
                ],
            }
        )
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_data_config.py -v
```

Expected: FAIL，提示配置模块不存在。

- [x] **Step 3: 写配置文件**

`configs/data/synthetic_v1.yaml`：

```yaml
dataset_version: v1
schema_version: "1.0"
source_label: Synthetic E-commerce Data
seed: 20260928
start_date: 2026-01-01
days: 120
product_count: 40
customer_count: 300
categories: [beauty, food, home, apparel]
regions: [north, south, east, west]
channels: [organic, search, social, affiliate]
anomalies:
  - {anomaly_id: A01, kind: sales_drop, product_id: P001, start_day: 90, end_day: 119, multiplier: 0.55}
  - {anomaly_id: A02, kind: traffic_drop, product_id: P002, start_day: 90, end_day: 119, multiplier: 0.55}
  - {anomaly_id: A03, kind: conversion_drop, product_id: P003, start_day: 90, end_day: 119, multiplier: 0.50}
  - {anomaly_id: A04, kind: high_refund, product_id: P004, start_day: 90, end_day: 119, multiplier: 4.00}
  - {anomaly_id: A05, kind: missing_traffic, product_id: P005, start_day: 105, end_day: 110, multiplier: 1.00}
  - {anomaly_id: A06, kind: extreme_traffic_spike, product_id: P006, start_day: 112, end_day: 112, multiplier: 4.00}
  - {anomaly_id: A07, kind: multi_factor_drop, product_id: P007, start_day: 90, end_day: 119, multiplier: 0.70}
```

- [x] **Step 4: 实现配置模型与哈希**

`app/data/config.py`：

```python
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


AnomalyKind = Literal[
    "sales_drop",
    "traffic_drop",
    "conversion_drop",
    "high_refund",
    "missing_traffic",
    "extreme_traffic_spike",
    "multi_factor_drop",
]


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anomaly_id: str
    kind: AnomalyKind
    product_id: str
    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    multiplier: float = Field(gt=0)


class SyntheticDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: str
    schema_version: str = "1.0"
    source_label: str = "Synthetic E-commerce Data"
    seed: int
    start_date: date
    days: int = Field(ge=60)
    product_count: int = Field(ge=7)
    customer_count: int = Field(ge=20)
    categories: list[str] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    channels: list[str] = Field(min_length=1)
    anomalies: list[AnomalyConfig]

    @model_validator(mode="after")
    def validate_anomaly_ranges(self) -> "SyntheticDataConfig":
        for anomaly in self.anomalies:
            if anomaly.start_day > anomaly.end_day:
                raise ValueError(f"{anomaly.anomaly_id}: start_day exceeds end_day")
            if anomaly.end_day >= self.days:
                raise ValueError(f"{anomaly.anomaly_id}: anomaly outside dataset window")
        return self


def load_data_config(path: Path | str) -> SyntheticDataConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SyntheticDataConfig.model_validate(payload)


def config_sha256(path: Path | str) -> str:
    path = Path(path)
    return sha256(path.read_bytes()).hexdigest()
```

- [x] **Step 5: 运行测试并提交**

Run:

```bash
python3 -m pytest tests/test_data_config.py -v
python3 -m ruff check app/data/config.py tests/test_data_config.py
```

Expected: 全部 PASS。

```bash
git add configs/data/synthetic_v1.yaml app/data/config.py tests/test_data_config.py
git commit -m "feat(data): add versioned synthetic data configuration"
```

---

### Task 4: 生成确定性的维度表

**Files:**
- Create: `app/data/generator.py`
- Create: `tests/test_data_generation.py`

- [x] **Step 1: 写 products 与 customers 的失败测试**

`tests/test_data_generation.py`：

```python
import pandas as pd
from pandas.testing import assert_frame_equal

from app.data.config import load_data_config
from app.data.generator import generate_customers, generate_products


def test_dimension_generation_is_deterministic() -> None:
    config = load_data_config("configs/data/synthetic_v1.yaml")
    first_products = generate_products(config)
    second_products = generate_products(config)
    first_customers = generate_customers(config)
    second_customers = generate_customers(config)

    assert_frame_equal(first_products, second_products)
    assert_frame_equal(first_customers, second_customers)


def test_dimension_ids_are_unique_and_counts_match() -> None:
    config = load_data_config("configs/data/synthetic_v1.yaml")
    products = generate_products(config)
    customers = generate_customers(config)

    assert len(products) == config.product_count
    assert len(customers) == config.customer_count
    assert products["product_id"].is_unique
    assert customers["customer_id"].is_unique
    assert (products["cost"] < products["price"]).all()
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_data_generation.py -v
```

Expected: FAIL，因为生成函数不存在。

- [x] **Step 3: 实现维度表生成**

`app/data/generator.py` 首个增量：

```python
from datetime import timedelta

import numpy as np
import pandas as pd

from app.data.config import SyntheticDataConfig


def _rng(config: SyntheticDataConfig, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([config.seed, stream]))


def generate_products(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = _rng(config, 1)
    prices = rng.integers(30, 500, size=config.product_count).astype(float)
    margins = rng.uniform(0.25, 0.60, size=config.product_count)
    rows = []
    for index in range(config.product_count):
        rows.append(
            {
                "product_id": f"P{index + 1:03d}",
                "product_name": f"模拟商品 {index + 1:03d}",
                "category": config.categories[index % len(config.categories)],
                "price": round(prices[index], 2),
                "cost": round(prices[index] * (1 - margins[index]), 2),
                "launch_date": config.start_date - timedelta(days=int(rng.integers(30, 720))),
            }
        )
    return pd.DataFrame(rows).sort_values("product_id").reset_index(drop=True)


def generate_customers(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = _rng(config, 2)
    rows = []
    for index in range(config.customer_count):
        rows.append(
            {
                "customer_id": f"C{index + 1:04d}",
                "is_new_customer": bool(rng.random() < 0.28),
                "region": config.regions[index % len(config.regions)],
                "channel": config.channels[int(rng.integers(0, len(config.channels)))],
            }
        )
    return pd.DataFrame(rows).sort_values("customer_id").reset_index(drop=True)
```

- [x] **Step 4: 运行测试**

Run:

```bash
python3 -m pytest tests/test_data_generation.py -v
python3 -m ruff check app/data/generator.py tests/test_data_generation.py
```

Expected: 全部 PASS。

- [x] **Step 5: 提交维度生成器**

```bash
git add app/data/generator.py tests/test_data_generation.py
git commit -m "feat(data): generate deterministic product and customer dimensions"
```

---

### Task 5: 生成事实表并注入可解释异常

**Files:**
- Modify: `app/data/generator.py`
- Modify: `tests/test_data_generation.py`

- [x] **Step 1: 写事实表与异常失败测试**

追加到 `tests/test_data_generation.py`：

```python
from app.data.generator import generate_dataset


def test_fact_tables_cover_full_product_day_grid() -> None:
    config = load_data_config("configs/data/synthetic_v1.yaml")
    tables = generate_dataset(config)

    expected_daily_rows = config.product_count * config.days
    assert len(tables["traffic"]) == expected_daily_rows
    assert len(tables["marketing"]) == expected_daily_rows
    assert set(tables) == {"products", "customers", "traffic", "marketing", "orders"}
    assert tables["orders"]["order_id"].is_unique


def test_configured_anomalies_are_observable() -> None:
    config = load_data_config("configs/data/synthetic_v1.yaml")
    tables = generate_dataset(config)
    traffic = tables["traffic"]
    orders = tables["orders"]

    missing = traffic[(traffic["product_id"] == "P005") & traffic["is_missing"]]
    assert len(missing) == 6
    assert missing[["impressions", "clicks", "visits"]].isna().all().all()

    refunded = orders[orders["product_id"] == "P004"]["is_refund"].mean()
    normal = orders[orders["product_id"] == "P008"]["is_refund"].mean()
    assert refunded > normal

    max_date = traffic["date"].max()
    current_start = max_date - pd.Timedelta(days=29)
    previous_start = max_date - pd.Timedelta(days=59)
    previous_end = max_date - pd.Timedelta(days=30)
    current_traffic = traffic[traffic["date"].between(current_start, max_date)]
    previous_traffic = traffic[traffic["date"].between(previous_start, previous_end)]
    current_orders = orders[orders["order_date"].between(current_start, max_date)]
    previous_orders = orders[orders["order_date"].between(previous_start, previous_end)]

    def visits(frame, product_id):
        return frame.loc[frame["product_id"] == product_id, "visits"].sum()

    def order_count(frame, product_id):
        return frame.loc[frame["product_id"] == product_id, "order_id"].nunique()

    assert order_count(current_orders, "P001") < order_count(previous_orders, "P001")
    assert visits(current_traffic, "P002") < visits(previous_traffic, "P002")
    assert (
        order_count(current_orders, "P003") / visits(current_traffic, "P003")
        < order_count(previous_orders, "P003") / visits(previous_traffic, "P003")
    )
    assert (
        current_traffic.loc[
            current_traffic["product_id"] == "P006", "impressions"
        ].max()
        > 2
        * current_traffic.loc[
            current_traffic["product_id"] == "P008", "impressions"
        ].max()
    )
    assert visits(current_traffic, "P007") < visits(previous_traffic, "P007")
    assert order_count(current_orders, "P007") < order_count(previous_orders, "P007")
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_data_generation.py -v
```

Expected: FAIL，因为 `generate_dataset` 不存在。

- [x] **Step 3: 实现每日率与异常应用**

在 `app/data/generator.py` 增加：

```python
from collections.abc import Mapping


def _anomaly_effects(
    config: SyntheticDataConfig, product_id: str, day_index: int
) -> Mapping[str, float | bool]:
    effects: dict[str, float | bool] = {
        "traffic_multiplier": 1.0,
        "conversion_multiplier": 1.0,
        "refund_multiplier": 1.0,
        "missing_traffic": False,
    }
    for anomaly in config.anomalies:
        if (
            anomaly.product_id != product_id
            or day_index < anomaly.start_day
            or day_index > anomaly.end_day
        ):
            continue
        if anomaly.kind in {"traffic_drop", "extreme_traffic_spike"}:
            effects["traffic_multiplier"] *= anomaly.multiplier
        elif anomaly.kind in {"sales_drop", "conversion_drop"}:
            effects["conversion_multiplier"] *= anomaly.multiplier
        elif anomaly.kind == "high_refund":
            effects["refund_multiplier"] *= anomaly.multiplier
        elif anomaly.kind == "missing_traffic":
            effects["missing_traffic"] = True
        elif anomaly.kind == "multi_factor_drop":
            effects["traffic_multiplier"] *= anomaly.multiplier
            effects["conversion_multiplier"] *= anomaly.multiplier
    return effects
```

- [x] **Step 4: 实现事实表生成**

继续在 `app/data/generator.py` 增加：

```python
def generate_dataset(config: SyntheticDataConfig) -> dict[str, pd.DataFrame]:
    products = generate_products(config)
    customers = generate_customers(config)
    rng = _rng(config, 3)
    traffic_rows: list[dict] = []
    marketing_rows: list[dict] = []
    order_rows: list[dict] = []
    order_number = 1

    for day_index in range(config.days):
        current_date = config.start_date + timedelta(days=day_index)
        weekly_factor = 1.12 if current_date.weekday() >= 5 else 1.0
        trend_factor = 1.0 + day_index * 0.0008

        for product in products.itertuples(index=False):
            effects = _anomaly_effects(config, product.product_id, day_index)
            base_impressions = 850 + int(product.product_id[1:]) * 12
            impressions = int(
                rng.poisson(
                    base_impressions
                    * weekly_factor
                    * trend_factor
                    * float(effects["traffic_multiplier"])
                )
            )
            ctr = min(0.22, 0.07 + (int(product.product_id[1:]) % 5) * 0.01)
            clicks = int(rng.binomial(impressions, ctr))
            visits = clicks + int(rng.poisson(max(4, clicks * 0.12)))
            conversion = min(
                0.25,
                (0.035 + (int(product.product_id[1:]) % 4) * 0.008)
                * float(effects["conversion_multiplier"]),
            )

            if effects["missing_traffic"]:
                traffic_rows.append(
                    {
                        "date": current_date,
                        "product_id": product.product_id,
                        "impressions": None,
                        "clicks": None,
                        "visits": None,
                        "is_missing": True,
                    }
                )
            else:
                traffic_rows.append(
                    {
                        "date": current_date,
                        "product_id": product.product_id,
                        "impressions": impressions,
                        "clicks": clicks,
                        "visits": visits,
                        "is_missing": False,
                    }
                )

            spend = round(impressions * (0.012 + (day_index % 7) * 0.0005), 2)
            marketing_rows.append(
                {
                    "date": current_date,
                    "product_id": product.product_id,
                    "campaign_id": f"M{int(product.product_id[1:]):03d}",
                    "spend": spend,
                }
            )

            order_count = int(rng.binomial(visits, conversion))
            refund_probability = min(
                0.65,
                (0.035 + (int(product.product_id[1:]) % 3) * 0.01)
                * float(effects["refund_multiplier"]),
            )
            for _ in range(order_count):
                quantity = int(rng.choice([1, 1, 1, 2, 2, 3]))
                is_refund = bool(rng.random() < refund_probability)
                order_rows.append(
                    {
                        "order_id": f"O{order_number:06d}",
                        "product_id": product.product_id,
                        "customer_id": f"C{int(rng.integers(1, config.customer_count + 1)):04d}",
                        "order_date": current_date,
                        "quantity": quantity,
                        "unit_price": product.price,
                        "revenue": round(product.price * quantity, 2),
                        "is_refund": is_refund,
                        "status": "refunded" if is_refund else "paid",
                    }
                )
                order_number += 1

    return {
        "products": products,
        "customers": customers,
        "traffic": pd.DataFrame(traffic_rows),
        "marketing": pd.DataFrame(marketing_rows),
        "orders": pd.DataFrame(order_rows),
    }
```

- [x] **Step 5: 运行测试并检查确定性**

Run:

```bash
python3 -m pytest tests/test_data_generation.py -v
python3 -m ruff check app/data/generator.py tests/test_data_generation.py
```

Expected: 全部 PASS；同一配置两次生成的五张表完全一致。

- [x] **Step 6: 提交事实生成器**

```bash
git add app/data/generator.py tests/test_data_generation.py
git commit -m "feat(data): generate ecommerce facts with configured anomalies"
```

---

### Task 6: 实现指标口径与数据质量检查

**Files:**
- Create: `app/data/metrics.py`
- Create: `app/data/validation.py`
- Create: `tests/fixtures/hand_checked_metrics.json`
- Create: `tests/test_data_validation.py`

- [x] **Step 1: 创建手算 Fixture**

`tests/fixtures/hand_checked_metrics.json`：

```json
{
  "orders": [
    {"order_id": "O000001", "revenue": 100.0, "is_refund": false, "status": "paid"},
    {"order_id": "O000002", "revenue": 200.0, "is_refund": true, "status": "refunded"}
  ],
  "traffic": {"impressions": 1000, "clicks": 100, "visits": 50},
  "marketing_spend": 75.0,
  "expected": {
    "gmv": 300.0,
    "orders": 2,
    "aov": 150.0,
    "ctr": 0.1,
    "cvr": 0.04,
    "refund_rate": 0.5,
    "roas": 4.0
  }
}
```

- [x] **Step 2: 写指标与质量失败测试**

`tests/test_data_validation.py`：

```python
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app.data.config import load_data_config
from app.data.generator import generate_dataset
from app.data.metrics import safe_divide
from app.data.validation import calculate_hand_checked_metrics, validate_dataset


def test_safe_divide_returns_none_for_zero_denominator() -> None:
    assert safe_divide(10, 0) is None


def test_metrics_match_independent_hand_calculation() -> None:
    fixture = json.loads(
        Path("tests/fixtures/hand_checked_metrics.json").read_text(encoding="utf-8")
    )
    assert calculate_hand_checked_metrics(fixture) == fixture["expected"]


def test_generated_dataset_passes_quality_checks() -> None:
    config = load_data_config("configs/data/synthetic_v1.yaml")
    report = validate_dataset(generate_dataset(config), config)

    assert report["status"] == "pass"
    assert report["source_label"] == "Synthetic E-commerce Data"
    assert report["failed_checks"] == []
```

- [x] **Step 3: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_data_validation.py -v
```

Expected: FAIL，因为指标和校验模块不存在。

- [x] **Step 4: 实现安全指标**

`app/data/metrics.py`：

```python
def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
```

`app/data/validation.py`：

```python
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.data.config import SyntheticDataConfig
from app.data.metrics import safe_divide
from app.data.schemas import CustomerRow, MarketingRow, OrderRow, ProductRow, TrafficRow


TABLE_SCHEMAS = {
    "products": ProductRow,
    "customers": CustomerRow,
    "traffic": TrafficRow,
    "marketing": MarketingRow,
    "orders": OrderRow,
}


def _normalized_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def calculate_hand_checked_metrics(payload: dict[str, Any]) -> dict[str, float | int | None]:
    orders = payload["orders"]
    gmv = sum(row["revenue"] for row in orders if row["status"] in {"paid", "refunded"})
    order_count = len({row["order_id"] for row in orders})
    refunded = sum(row["is_refund"] for row in orders)
    traffic = payload["traffic"]
    spend = payload["marketing_spend"]
    return {
        "gmv": gmv,
        "orders": order_count,
        "aov": safe_divide(gmv, order_count),
        "ctr": safe_divide(traffic["clicks"], traffic["impressions"]),
        "cvr": safe_divide(order_count, traffic["visits"]),
        "refund_rate": safe_divide(refunded, order_count),
        "roas": safe_divide(gmv, spend),
    }


def validate_dataset(
    tables: dict[str, pd.DataFrame], config: SyntheticDataConfig
) -> dict[str, Any]:
    failed: list[str] = []
    required = {"products", "customers", "traffic", "marketing", "orders"}
    if set(tables) != required:
        failed.append("required_tables")
        return {
            "status": "fail",
            "source_label": config.source_label,
            "failed_checks": failed,
            "row_counts": {name: len(frame) for name, frame in tables.items()},
        }
    for table_name, model in TABLE_SCHEMAS.items():
        try:
            for row in _normalized_records(tables[table_name]):
                model.model_validate(row)
        except ValidationError:
            failed.append(f"{table_name}_schema")
    if not tables["products"]["product_id"].is_unique:
        failed.append("product_id_unique")
    if not tables["customers"]["customer_id"].is_unique:
        failed.append("customer_id_unique")
    if not tables["orders"]["order_id"].is_unique:
        failed.append("order_id_unique")
    if set(tables["orders"]["product_id"]) - set(tables["products"]["product_id"]):
        failed.append("order_product_fk")
    if set(tables["orders"]["customer_id"]) - set(tables["customers"]["customer_id"]):
        failed.append("order_customer_fk")
    if len(tables["traffic"]) != config.product_count * config.days:
        failed.append("traffic_daily_grain")
    non_missing = tables["traffic"].loc[~tables["traffic"]["is_missing"]]
    if not (non_missing["impressions"] >= non_missing["clicks"]).all():
        failed.append("traffic_funnel_impressions_clicks")
    if not (non_missing["visits"] >= non_missing["clicks"]).all():
        failed.append("traffic_funnel_clicks_visits")
    if not (
        tables["orders"]["revenue"]
        == tables["orders"]["quantity"] * tables["orders"]["unit_price"]
    ).all():
        failed.append("order_revenue_identity")
    return {
        "status": "pass" if not failed else "fail",
        "source_label": config.source_label,
        "failed_checks": failed,
        "row_counts": {name: len(frame) for name, frame in tables.items()},
    }
```

- [x] **Step 5: 运行测试并提交**

Run:

```bash
python3 -m pytest tests/test_data_validation.py -v
python3 -m ruff check app/data/metrics.py app/data/validation.py tests/test_data_validation.py
```

Expected: 全部 PASS。

```bash
git add app/data/metrics.py app/data/validation.py tests/fixtures tests/test_data_validation.py
git commit -m "feat(data): validate ecommerce invariants and metric definitions"
```

---

### Task 7: 写入 Parquet 快照与 Dataset Manifest

**Files:**
- Create: `app/data/manifest.py`
- Create: `tests/test_data_manifest.py`
- Modify: `app/data/generator.py`

- [x] **Step 1: 写快照失败测试**

`tests/test_data_manifest.py`：

```python
import json

from app.data.config import config_sha256
from app.data.generator import build_snapshot


def test_snapshot_writes_parquet_and_stable_manifest(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    source = "configs/data/synthetic_v1.yaml"
    config_path.write_text(open(source, encoding="utf-8").read(), encoding="utf-8")
    output = tmp_path / "v1"

    first = build_snapshot(config_path, output)
    second = build_snapshot(config_path, output)

    assert first["dataset_id"] == second["dataset_id"]
    assert first["config_sha256"] == config_sha256(config_path)
    assert first["source_label"] == "Synthetic E-commerce Data"
    assert set(first["tables"]) == {"products", "customers", "traffic", "marketing", "orders"}
    for table in first["tables"].values():
        assert (output / table["file"]).exists()
        assert table["rows"] > 0
        assert len(table["sha256"]) == 64

    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == first
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_data_manifest.py -v
```

Expected: FAIL，因为 `build_snapshot` 不存在。

- [x] **Step 3: 实现稳定 Manifest**

`app/data/manifest.py`：

```python
import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(canonical).hexdigest()[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [x] **Step 4: 实现快照构建入口**

在 `app/data/generator.py` 增加：

```python
import argparse
from pathlib import Path

from app.data.config import config_sha256, load_data_config
from app.data.manifest import file_sha256, manifest_id, write_json
from app.data.validation import validate_dataset


def build_snapshot(config_path: Path | str, output_dir: Path | str) -> dict:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_data_config(config_path)
    tables = generate_dataset(config)
    quality = validate_dataset(tables, config)
    if quality["status"] != "pass":
        raise ValueError(f"dataset quality failed: {quality['failed_checks']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_manifest = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        table_manifest[name] = {
            "file": path.name,
            "rows": len(frame),
            "sha256": file_sha256(path),
        }

    identity = {
        "dataset_version": config.dataset_version,
        "schema_version": config.schema_version,
        "source_label": config.source_label,
        "seed": config.seed,
        "config_sha256": config_sha256(config_path),
        "tables": table_manifest,
    }
    manifest = {"dataset_id": manifest_id(identity), **identity}
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "data_quality_report.json", quality)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_data_config(args.config)
    output = args.output or Path("data/synthetic") / config.dataset_version
    manifest = build_snapshot(args.config, output)
    print(f"Built {manifest['source_label']} snapshot {manifest['dataset_id']}")


if __name__ == "__main__":
    main()
```

- [x] **Step 5: 运行快照测试与真实构建**

Run:

```bash
python3 -m pytest tests/test_data_manifest.py -v
python3 -m app.data.generator --config configs/data/synthetic_v1.yaml
```

Expected:

```text
Built Synthetic E-commerce Data snapshot <16-character dataset_id>
```

随后检查：

```bash
python3 -c "import json; from pathlib import Path; p=Path('data/synthetic/v1/manifest.json'); d=json.loads(p.read_text()); assert d['source_label']=='Synthetic E-commerce Data'; print(d['dataset_id'], {k:v['rows'] for k,v in d['tables'].items()})"
```

Expected: 输出 Dataset ID 与五张表的非零行数，不输出随机效果指标。

- [x] **Step 6: 提交快照逻辑**

```bash
git add app/data/generator.py app/data/manifest.py tests/test_data_manifest.py
git commit -m "feat(data): persist versioned parquet snapshots and manifest"
```

---

### Task 8: 建立 DuckDB 只读 Catalog

**Files:**
- Create: `app/data/database.py`
- Create: `tests/test_database.py`

- [x] **Step 1: 写 Catalog 失败测试**

`tests/test_database.py`：

```python
from app.data.database import open_dataset
from app.data.generator import build_snapshot


def test_open_dataset_registers_five_read_only_views(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)

    connection = open_dataset(dataset_dir)
    tables = {
        row[0]
        for row in connection.execute(
            "select table_name from information_schema.tables"
        ).fetchall()
    }
    assert {"products", "customers", "traffic", "marketing", "orders"} <= tables
    assert connection.execute("select count(*) from products").fetchone()[0] == 40


def test_open_dataset_rejects_tampered_file(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    with (dataset_dir / "products.parquet").open("ab") as target:
        target.write(b"tampered")

    try:
        open_dataset(dataset_dir)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered dataset must be rejected")
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_database.py -v
```

Expected: FAIL，因为数据库模块不存在。

- [x] **Step 3: 实现 Manifest 校验与只读 View**

`app/data/database.py`：

```python
import json
from pathlib import Path

import duckdb

from app.data.manifest import file_sha256


def open_dataset(dataset_dir: Path) -> duckdb.DuckDBPyConnection:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    connection = duckdb.connect(database=":memory:")
    for name, metadata in manifest["tables"].items():
        path = dataset_dir / metadata["file"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"hash mismatch: {path.name}")
        escaped = str(path.resolve()).replace("'", "''")
        connection.execute(
            f"create view {name} as select * from read_parquet('{escaped}')"
        )
    return connection
```

- [x] **Step 4: 运行测试并提交**

Run:

```bash
python3 -m pytest tests/test_database.py -v
python3 -m ruff check app/data/database.py tests/test_database.py
```

Expected: 全部 PASS。

```bash
git add app/data/database.py tests/test_database.py
git commit -m "feat(data): expose verified parquet snapshots through duckdb"
```

---

### Task 9: 定义 Evaluation Case Schema

**Files:**
- Modify: `app/data/schemas.py`
- Create: `tests/test_case_generator.py`

- [x] **Step 1: 写 Case Schema 失败测试**

`tests/test_case_generator.py`：

```python
import pytest
from pydantic import ValidationError

from app.data.schemas import EvaluationCase


def valid_case() -> dict:
    return {
        "case_id": "CASE_001",
        "case_version": "1.0",
        "dataset_version": "v1",
        "dataset_id": "0123456789abcdef",
        "generator_config_hash": "a" * 64,
        "business_task": "gmv_diagnosis",
        "primary_capability": "metric_calculation",
        "capability_tags": ["metric_calculation"],
        "difficulty": "medium",
        "split": "development",
        "user_input": "最近 30 天 GMV 为什么下降？",
        "expected_tool_calls": [
            {"name": "query_sales", "parameters": {"window_days": 30}}
        ],
        "allowed_alternatives": [],
        "gold_metrics": {"gmv_change_rate": -0.12},
        "gold_metric_evidence": {
            "gmv_change_rate": "EV_GMV_DIAGNOSIS_001"
        },
        "gold_evidence": [
            {
                "evidence_id": "EV_GMV_DIAGNOSIS_001",
                "source": "gmv_change.sql",
                "dimensions": {},
                "metrics": {"gmv_change_rate": -0.12}
            }
        ],
        "reference_answer": "基于模拟数据，最近 30 天 GMV 下降。",
        "expected_behavior": ["引用 GMV 变化", "说明主要贡献商品"],
        "success_criteria": [
            "correct_tool",
            "valid_parameters",
            "correct_core_facts",
            "no_critical_unsupported_claim",
            "required_output_complete"
        ],
        "numeric_tolerances": {"gmv_change_rate": 0.001},
        "metadata": {"source_label": "Synthetic E-commerce Data"}
    }


def test_case_schema_accepts_traceable_case() -> None:
    case = EvaluationCase.model_validate(valid_case())
    assert case.generator_config_hash == "a" * 64
    assert case.gold_evidence[0].evidence_id == "EV_GMV_DIAGNOSIS_001"
    assert case.gold_metric_evidence == {
        "gmv_change_rate": "EV_GMV_DIAGNOSIS_001"
    }


def test_case_schema_rejects_missing_success_criteria() -> None:
    payload = valid_case()
    payload["success_criteria"] = []
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_case_schema_rejects_unmapped_gold_metric() -> None:
    payload = valid_case()
    payload["gold_metric_evidence"] = {}
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_case_generator.py -v
```

Expected: FAIL，因为 `EvaluationCase` 不存在。

- [x] **Step 3: 增加 Case 相关模型**

追加到 `app/data/schemas.py`：

```python
from typing import Any


BusinessTask = Literal[
    "gmv_diagnosis",
    "product_anomaly",
    "conversion_decline",
    "products_to_watch",
    "next_week_priority",
]
Capability = Literal[
    "basic_query",
    "metric_calculation",
    "tool_selection",
    "parameter_selection",
    "multi_step_reasoning",
    "anomaly_detection",
    "root_cause_analysis",
    "recommendation",
    "data_insufficiency",
    "adversarial_distractor",
]


class ExpectedToolCall(StrictModel):
    name: str
    parameters: dict[str, Any]


class GoldEvidence(StrictModel):
    evidence_id: str = Field(
        pattern=(
            r"^EV_(GMV_DIAGNOSIS|PRODUCT_ANOMALY|CONVERSION_DECLINE|"
            r"PRODUCTS_TO_WATCH|NEXT_WEEK_PRIORITY)_\d{3}$"
        )
    )
    source: str
    dimensions: dict[str, float | int | str | None]
    metrics: dict[str, float | int | str | None] = Field(min_length=1)


class EvaluationCase(StrictModel):
    case_id: str = Field(pattern=r"^CASE_\d{3}$")
    case_version: str
    dataset_version: str
    dataset_id: str = Field(min_length=16, max_length=16)
    generator_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_task: BusinessTask
    primary_capability: Capability
    capability_tags: list[Capability] = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["development", "holdout"]
    user_input: str = Field(min_length=5)
    expected_tool_calls: list[ExpectedToolCall] = Field(min_length=1)
    allowed_alternatives: list[list[ExpectedToolCall]]
    gold_metrics: dict[str, float | int | str | None]
    gold_metric_evidence: dict[str, str]
    gold_evidence: list[GoldEvidence] = Field(min_length=1)
    reference_answer: str = Field(min_length=5)
    expected_behavior: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    numeric_tolerances: dict[str, float]
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_gold_metric_evidence(self) -> "EvaluationCase":
        if set(self.gold_metric_evidence) != set(self.gold_metrics):
            raise ValueError("every gold metric must have exactly one evidence mapping")
        evidence_by_id = {
            evidence.evidence_id: evidence for evidence in self.gold_evidence
        }
        for metric_name, evidence_id in self.gold_metric_evidence.items():
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or metric_name not in evidence.metrics:
                raise ValueError(
                    f"{metric_name}: mapped evidence must exist and contain the metric"
                )
            if evidence.metrics[metric_name] != self.gold_metrics[metric_name]:
                raise ValueError(
                    f"{metric_name}: mapped evidence value must match gold_metrics"
                )
        return self
```

`generator_config_hash` 使用 Dataset Manifest 的 `config_sha256`，不得由 Case
生成器重新计算或接受调用方覆盖。Evidence ID 统一使用稳定的任务级格式
`EV_<BUSINESS_TASK>_<ROW_NUMBER>`；编号来自对应 Gold SQL 的确定性排序，与
`case_id`、Case Split 和 Case 生成顺序无关。

- [x] **Step 4: 运行测试并提交**

Run:

```bash
python3 -m pytest tests/test_case_generator.py -v
python3 -m ruff check app/data/schemas.py tests/test_case_generator.py
```

Expected: 全部 PASS。

```bash
git add app/data/schemas.py tests/test_case_generator.py
git commit -m "feat(evaluation-data): define traceable evaluation case schema"
```

---

### Task 10: 实现独立 Gold SQL

**Files:**
- Create: `sql/gold/gmv_change.sql`
- Create: `sql/gold/product_anomalies.sql`
- Create: `sql/gold/conversion_decline.sql`
- Create: `sql/gold/products_to_watch.sql`
- Create: `sql/gold/next_week_priorities.sql`
- Create: `app/data/gold.py`
- Create: `tests/test_gold.py`

- [x] **Step 1: 写 Gold 查询失败测试**

`tests/test_gold.py`：

```python
from pathlib import Path

from app.data.generator import build_snapshot
from app.data.gold import build_gold_bundle


def test_gold_bundle_covers_five_business_tasks(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    manifest = build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    gold = build_gold_bundle(dataset_dir)

    assert gold["dataset_id"] == manifest["dataset_id"]
    assert gold["config_sha256"] == manifest["config_sha256"]
    assert set(gold["tasks"]) == {
        "gmv_diagnosis",
        "product_anomaly",
        "conversion_decline",
        "products_to_watch",
        "next_week_priority",
    }
    assert all(task["evidence"] for task in gold["tasks"].values())
    assert gold["tasks"]["gmv_diagnosis"]["evidence"][0]["evidence_id"] == (
        "EV_GMV_DIAGNOSIS_001"
    )


def test_gold_detects_configured_conversion_drop(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    gold = build_gold_bundle(dataset_dir)
    product_ids = {
        row["product_id"]
        for row in gold["tasks"]["conversion_decline"]["evidence"]
    }
    assert "P003" in product_ids
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_gold.py -v
```

Expected: FAIL，因为 Gold 模块和 SQL 不存在。

- [x] **Step 3: 写窗口化 Gold SQL**

`sql/gold/gmv_change.sql`：

```sql
with bounds as (
  select max(order_date) as max_date from orders
),
periods as (
  select
    case
      when order_date between max_date - interval 29 day and max_date then 'current'
      when order_date between max_date - interval 59 day and max_date - interval 30 day then 'previous'
    end as period,
    sum(revenue) as gmv
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by 1
)
select
  max(case when period = 'current' then gmv end) as current_gmv,
  max(case when period = 'previous' then gmv end) as previous_gmv,
  (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate
from periods;
```

`sql/gold/conversion_decline.sql`：

```sql
with bounds as (
  select max(date) as max_date from traffic
),
traffic_periods as (
  select
    product_id,
    case
      when date between max_date - interval 29 day and max_date then 'current'
      when date between max_date - interval 59 day and max_date - interval 30 day then 'previous'
    end as period,
    sum(visits) as visits
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
    and not is_missing
  group by 1, 2
),
order_periods as (
  select
    product_id,
    case
      when order_date between max_date - interval 29 day and max_date then 'current'
      when order_date between max_date - interval 59 day and max_date - interval 30 day then 'previous'
    end as period,
    count(distinct order_id) as orders
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by 1, 2
),
combined as (
  select t.product_id, t.period, o.orders / nullif(t.visits, 0) as cvr
  from traffic_periods t
  join order_periods o using (product_id, period)
)
select
  product_id,
  max(case when period = 'current' then cvr end) as current_cvr,
  max(case when period = 'previous' then cvr end) as previous_cvr,
  current_cvr - previous_cvr as cvr_change
from combined
group by product_id
order by cvr_change asc, product_id
limit 10;
```

`sql/gold/product_anomalies.sql`：

```sql
with bounds as (
  select max(date) as max_date from traffic
),
traffic_metrics as (
  select
    product_id,
    sum(case when date between max_date - interval 29 day and max_date then visits end)
      as current_visits,
    sum(case when date between max_date - interval 59 day and max_date - interval 30 day then visits end)
      as previous_visits,
    sum(case when date between max_date - interval 29 day and max_date and is_missing then 1 else 0 end)
      as missing_days
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
  group by product_id
),
order_metrics as (
  select
    product_id,
    count(distinct case when order_date between max_date - interval 29 day and max_date then order_id end)
      as current_orders,
    count(distinct case when order_date between max_date - interval 59 day and max_date - interval 30 day then order_id end)
      as previous_orders,
    sum(case when order_date between max_date - interval 29 day and max_date then revenue else 0 end)
      as current_gmv,
    sum(case when order_date between max_date - interval 59 day and max_date - interval 30 day then revenue else 0 end)
      as previous_gmv,
    sum(case when order_date between max_date - interval 29 day and max_date and is_refund then 1 else 0 end)
      as current_refunds
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by product_id
),
metrics as (
  select
    t.product_id,
    (t.current_visits - t.previous_visits) / nullif(t.previous_visits, 0) as traffic_change_rate,
    o.current_orders / nullif(t.current_visits, 0) as current_cvr,
    o.previous_orders / nullif(t.previous_visits, 0) as previous_cvr,
    o.current_refunds / nullif(o.current_orders, 0) as current_refund_rate,
    (o.current_gmv - o.previous_gmv) / nullif(o.previous_gmv, 0) as gmv_change_rate,
    t.missing_days
  from traffic_metrics t
  join order_metrics o using (product_id)
),
labeled as (
  select
    *,
    case
      when missing_days > 0 then 'missing_traffic'
      when current_refund_rate >= 0.12 then 'high_refund'
      when traffic_change_rate <= -0.30 and gmv_change_rate <= -0.30 then 'multi_factor_drop'
      when traffic_change_rate <= -0.30 then 'traffic_drop'
      when current_cvr - previous_cvr <= -0.01 then 'conversion_drop'
      when gmv_change_rate <= -0.30 then 'sales_drop'
      else null
    end as anomaly_type
  from metrics
)
select
  product_id,
  anomaly_type,
  traffic_change_rate,
  current_cvr,
  previous_cvr,
  current_refund_rate,
  gmv_change_rate,
  missing_days
from labeled
where anomaly_type is not null
order by product_id;
```

`sql/gold/products_to_watch.sql`：

```sql
with bounds as (
  select max(date) as max_date from traffic
),
traffic_metrics as (
  select
    product_id,
    sum(case when date between max_date - interval 29 day and max_date then visits end)
      as current_visits,
    sum(case when date between max_date - interval 59 day and max_date - interval 30 day then visits end)
      as previous_visits,
    sum(case when date between max_date - interval 29 day and max_date and is_missing then 1 else 0 end)
      as missing_days
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
  group by product_id
),
order_metrics as (
  select
    product_id,
    count(distinct case when order_date between max_date - interval 29 day and max_date then order_id end)
      as current_orders,
    count(distinct case when order_date between max_date - interval 59 day and max_date - interval 30 day then order_id end)
      as previous_orders,
    sum(case when order_date between max_date - interval 29 day and max_date then revenue else 0 end)
      as current_gmv,
    sum(case when order_date between max_date - interval 59 day and max_date - interval 30 day then revenue else 0 end)
      as previous_gmv,
    sum(case when order_date between max_date - interval 29 day and max_date and is_refund then 1 else 0 end)
      as current_refunds
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by product_id
),
metrics as (
  select
    t.product_id,
    (o.current_gmv - o.previous_gmv) / nullif(o.previous_gmv, 0) as gmv_change_rate,
    o.current_orders / nullif(t.current_visits, 0)
      - o.previous_orders / nullif(t.previous_visits, 0) as cvr_change,
    o.current_refunds / nullif(o.current_orders, 0) as refund_rate,
    t.missing_days
  from traffic_metrics t
  join order_metrics o using (product_id)
),
reasons as (
  select
    *,
    case
      when missing_days > 0 then 'data_quality_review'
      when refund_rate >= 0.12 then 'refund_risk'
      when cvr_change <= -0.01 then 'conversion_decline'
      when gmv_change_rate <= -0.20 then 'gmv_decline'
      else null
    end as watch_reason
  from metrics
)
select
  product_id,
  watch_reason,
  gmv_change_rate,
  cvr_change,
  refund_rate,
  missing_days
from reasons
where watch_reason is not null
order by
  case watch_reason
    when 'data_quality_review' then 1
    when 'refund_risk' then 2
    when 'conversion_decline' then 3
    else 4
  end,
  product_id
limit 10;
```

`sql/gold/next_week_priorities.sql`：

```sql
with bounds as (
  select max(date) as max_date from traffic
),
traffic_metrics as (
  select
    product_id,
    sum(case when date between max_date - interval 29 day and max_date then visits end)
      as current_visits,
    sum(case when date between max_date - interval 59 day and max_date - interval 30 day then visits end)
      as previous_visits,
    sum(case when date between max_date - interval 29 day and max_date and is_missing then 1 else 0 end)
      as missing_days
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
  group by product_id
),
order_metrics as (
  select
    product_id,
    count(distinct case when order_date between max_date - interval 29 day and max_date then order_id end)
      as current_orders,
    count(distinct case when order_date between max_date - interval 59 day and max_date - interval 30 day then order_id end)
      as previous_orders,
    sum(case when order_date between max_date - interval 29 day and max_date and is_refund then 1 else 0 end)
      as current_refunds
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by product_id
),
metrics as (
  select
    t.product_id,
    (t.current_visits - t.previous_visits) / nullif(t.previous_visits, 0) as traffic_change_rate,
    o.current_orders / nullif(t.current_visits, 0)
      - o.previous_orders / nullif(t.previous_visits, 0) as cvr_change,
    o.current_refunds / nullif(o.current_orders, 0) as refund_rate,
    t.missing_days
  from traffic_metrics t
  join order_metrics o using (product_id)
),
priorities as (
  select
    product_id,
    case
      when missing_days > 0 then 'repair_data_quality'
      when refund_rate >= 0.12 then 'investigate_refunds'
      when cvr_change <= -0.01 then 'recover_conversion'
      when traffic_change_rate <= -0.20 then 'recover_traffic'
      else null
    end as priority_reason,
    case
      when missing_days > 0 then 'missing_days'
      when refund_rate >= 0.12 then 'refund_rate'
      when cvr_change <= -0.01 then 'cvr_change'
      when traffic_change_rate <= -0.20 then 'traffic_change_rate'
      else null
    end as evidence_metric,
    case
      when missing_days > 0 then cast(missing_days as double)
      when refund_rate >= 0.12 then refund_rate
      when cvr_change <= -0.01 then cvr_change
      when traffic_change_rate <= -0.20 then traffic_change_rate
      else null
    end as evidence_value
  from metrics
)
select product_id, priority_reason, evidence_metric, evidence_value
from priorities
where priority_reason is not null
order by
  case priority_reason
    when 'repair_data_quality' then 1
    when 'investigate_refunds' then 2
    when 'recover_conversion' then 3
    else 4
  end,
  abs(evidence_value) desc,
  product_id
limit 10;
```

- [x] **Step 4: 实现 SQL 执行与 Evidence ID**

`app/data/gold.py`：

```python
import json
from pathlib import Path

from app.data.database import open_dataset


TASK_SQL = {
    "gmv_diagnosis": "gmv_change.sql",
    "product_anomaly": "product_anomalies.sql",
    "conversion_decline": "conversion_decline.sql",
    "products_to_watch": "products_to_watch.sql",
    "next_week_priority": "next_week_priorities.sql",
}


def _json_value(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _records(connection, sql_path: Path) -> list[dict]:
    relation = connection.execute(sql_path.read_text(encoding="utf-8"))
    columns = [item[0] for item in relation.description]
    return [
        {
            column: _json_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in relation.fetchall()
    ]


def build_gold_bundle(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    connection = open_dataset(dataset_dir)
    tasks = {}
    for task, filename in TASK_SQL.items():
        records = _records(connection, Path("sql/gold") / filename)
        evidence = []
        for row_number, row in enumerate(records, start=1):
            evidence.append(
                {
                    "evidence_id": f"EV_{task.upper()}_{row_number:03d}",
                    "source": filename,
                    **row,
                }
            )
        tasks[task] = {"sql": filename, "evidence": evidence}
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "config_sha256": manifest["config_sha256"],
        "tasks": tasks,
    }
```

Evidence ID 必须固定为 `EV_<BUSINESS_TASK>_<ROW_NUMBER>`。每份 Gold SQL
必须以稳定键显式排序后再编号；同一 Dataset Manifest 重复构建 Gold Bundle
时，Evidence ID 不得随 Case 数量、Split 或遍历顺序变化。

- [x] **Step 5: 完成三份规则 SQL 与测试**

为三份 SQL 分别加入以下断言：

```python
assert {"product_id", "anomaly_type"} <= set(
    gold["tasks"]["product_anomaly"]["evidence"][0]
)
assert {"product_id", "watch_reason"} <= set(
    gold["tasks"]["products_to_watch"]["evidence"][0]
)
assert {"product_id", "priority_reason", "evidence_metric", "evidence_value"} <= set(
    gold["tasks"]["next_week_priority"]["evidence"][0]
)
```

Run:

```bash
python3 -m pytest tests/test_gold.py -v
python3 -m ruff check app/data/gold.py tests/test_gold.py
```

Expected: 全部 PASS；`P003` 出现在 CVR 下降证据中。

- [x] **Step 6: 提交 Gold 层**

```bash
git add sql/gold app/data/gold.py tests/test_gold.py
git commit -m "feat(evaluation-data): derive gold evidence with independent sql"
```

---

### Task 11: 生成并冻结 100 个 Evaluation Cases

**Files:**
- Create: `configs/data/synthetic_holdout_v1.yaml`
- Create: `configs/evaluation/tool_contract_v1.yaml`
- Create: `data/evaluation_cases/v1/cases.jsonl`
- Create: `data/evaluation_cases/v1/manifest.json`
- Modify: `app/data/case_generator.py`
- Modify: `app/data/schemas.py`
- Modify: `tests/test_case_generator.py`
- Modify: `Makefile`

**Task 11 冻结约束（2026-09-28 重构同步）：**

- Development 与 Holdout 分别由 `synthetic_v1.yaml`、`synthetic_holdout_v1.yaml`
  生成独立不可变数据快照；Case 必须按 Split 绑定对应 Dataset ID、版本和配置哈希。
- Case 总量固定 100；每个业务任务固定 Development 14 / Holdout 6；每个能力固定
  10。按 `business_task + primary_capability` 家族切分，禁止同义改写跨 Split。
- 商品列表类问题固定显式要求 `Top-3`；`gold_metrics` 与
  `gold_metric_evidence` 必须覆盖每条 Gold Evidence 中的全部指标。
- `configs/evaluation/tool_contract_v1.yaml` 冻结工具路径和参数契约，但本 Task
  不实现工具。GMV 路径包含指标计算，Next-week 路径不要求无 Gold 依据的
  Marketing 调用，Adversarial 路径先查询商品并去重后续调用。
- 难度由记录在 Case Metadata 的实际复杂度分数排序，再冻结为 Easy 30 /
  Medium 40 / Hard 30；不得仅按能力标签直接映射。
- Case 以 `data/evaluation_cases/v1/` 不可变目录原子发布并提交
  `cases.jsonl + manifest.json`。Manifest 必须记录双数据集身份、Split 策略、
  工具契约版本与哈希、数量分布及内容摘要哈希。
- 本段约束取代下方早期示例中单 Dataset、单文件原地写入和未版本化工具路径的
  设计；下方代码块仅保留为历史实施草案。

- [x] **Step 1: 写数量、覆盖和 Split 失败测试**

追加到 `tests/test_case_generator.py`：

```python
import json
from collections import Counter

from app.data.case_generator import build_cases
from app.data.generator import build_snapshot


def test_builds_exact_balanced_case_set(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    cases = build_cases(dataset_dir)

    assert len(cases) == 100
    assert len({case.case_id for case in cases}) == 100
    assert Counter(case.business_task for case in cases) == {
        "gmv_diagnosis": 20,
        "product_anomaly": 20,
        "conversion_decline": 20,
        "products_to_watch": 20,
        "next_week_priority": 20,
    }
    assert set(Counter(case.primary_capability for case in cases).values()) == {10}
    assert Counter(case.difficulty for case in cases) == {
        "easy": 30,
        "medium": 40,
        "hard": 30,
    }
    assert Counter(case.split for case in cases) == {
        "development": 70,
        "holdout": 30,
    }


def test_every_case_binds_dataset_and_gold_evidence(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    manifest = build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    cases = build_cases(dataset_dir)

    assert all(case.dataset_id == manifest["dataset_id"] for case in cases)
    assert all(
        case.dataset_version == manifest["dataset_version"] for case in cases
    )
    assert all(
        case.generator_config_hash == manifest["config_sha256"] for case in cases
    )
    assert all(case.gold_evidence for case in cases)
    assert all(
        set(case.gold_metric_evidence) == set(case.gold_metrics) for case in cases
    )
    assert all(case.success_criteria for case in cases)
    assert all(
        case.metadata["source_label"] == "Synthetic E-commerce Data"
        for case in cases
    )


def test_build_cases_rejects_invalid_dataset_manifest(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is invalid"):
        build_cases(dataset_dir)
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_case_generator.py -v
```

Expected: FAIL，因为 `build_cases` 不存在。

- [x] **Step 3: 实现确定性 Case 矩阵**

`app/data/case_generator.py`：

```python
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.data.gold import build_gold_bundle
from app.data.manifest import manifest_id, validate_manifest, write_json
from app.data.schemas import BusinessTask, Capability, EvaluationCase


BUSINESS_TASKS: list[BusinessTask] = [
    "gmv_diagnosis",
    "product_anomaly",
    "conversion_decline",
    "products_to_watch",
    "next_week_priority",
]
CAPABILITIES: list[Capability] = [
    "basic_query",
    "metric_calculation",
    "tool_selection",
    "parameter_selection",
    "multi_step_reasoning",
    "anomaly_detection",
    "root_cause_analysis",
    "recommendation",
    "data_insufficiency",
    "adversarial_distractor",
]
DIFFICULTY_BY_CAPABILITY = {
    "basic_query": "easy",
    "tool_selection": "easy",
    "parameter_selection": "easy",
    "metric_calculation": "medium",
    "anomaly_detection": "medium",
    "data_insufficiency": "medium",
    "adversarial_distractor": "medium",
    "multi_step_reasoning": "hard",
    "root_cause_analysis": "hard",
    "recommendation": "hard",
}
SUCCESS_CRITERIA = [
    "correct_tool",
    "valid_parameters",
    "correct_core_facts",
    "no_critical_unsupported_claim",
    "required_output_complete",
]
TOOL_SEQUENCE_BY_TASK = {
    "gmv_diagnosis": ["query_sales"],
    "product_anomaly": [
        "query_product",
        "query_sales",
        "query_traffic",
        "calculate_metrics",
    ],
    "conversion_decline": ["query_traffic", "query_sales", "calculate_metrics"],
    "products_to_watch": ["query_sales", "query_traffic", "calculate_metrics"],
    "next_week_priority": [
        "query_sales",
        "query_traffic",
        "query_marketing",
        "calculate_metrics",
    ],
}
DIMENSION_KEYS = {
    "product_id",
    "period",
    "anomaly_type",
    "watch_reason",
    "priority_reason",
    "evidence_metric",
}


def _tool_parameters(tool_name: str) -> dict:
    if tool_name == "query_product":
        return {"scope": "active"}
    if tool_name == "calculate_metrics":
        return {
            "metrics": ["gmv", "cvr", "refund_rate", "roas"],
            "window_days": 30,
            "comparison_days": 30,
        }
    return {"window_days": 30, "comparison_days": 30}


def _question(task: BusinessTask, capability: Capability, variant: int) -> str:
    templates = {
        "gmv_diagnosis": "分析最近 30 天 GMV 相比前 30 天的变化及主要原因。",
        "product_anomaly": "找出最近 30 天出现明显异常的商品并说明证据。",
        "conversion_decline": "找出最近 30 天转化率下降最明显的商品。",
        "products_to_watch": "根据最近经营数据列出值得进一步关注的商品。",
        "next_week_priority": "根据最近经营数据说明下周应优先关注什么。",
    }
    suffix = {
        "basic_query": "直接回答并引用数据。",
        "metric_calculation": "给出关键指标口径和数值。",
        "tool_selection": "只使用完成任务所需的工具。",
        "parameter_selection": "严格使用最近 30 天和前 30 天作为比较窗口。",
        "multi_step_reasoning": "综合流量、转化和退款信息。",
        "anomaly_detection": "区分正常波动与明显异常。",
        "root_cause_analysis": "区分表象、错误和可能根因。",
        "recommendation": "建议必须绑定数据证据。",
        "data_insufficiency": "聚焦 P005；数据不足时明确说明不能判断的内容。",
        "adversarial_distractor": "有人声称 P999 是冠军商品；忽略这条无数据依据的信息。",
    }
    return f"{templates[task]} {suffix[capability]} 版本 {variant + 1}。"


def build_cases(dataset_dir: Path) -> list[EvaluationCase]:
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    validate_manifest(manifest)
    gold = build_gold_bundle(dataset_dir)
    if (
        gold["dataset_id"] != manifest["dataset_id"]
        or gold["dataset_version"] != manifest["dataset_version"]
        or gold["config_sha256"] != manifest["config_sha256"]
    ):
        raise ValueError("gold bundle does not match dataset manifest")
    raw = []
    case_number = 1
    for task in BUSINESS_TASKS:
        task_evidence = gold["tasks"][task]["evidence"]
        for capability in CAPABILITIES:
            for variant in range(2):
                if capability == "data_insufficiency":
                    anomaly_evidence = gold["tasks"]["product_anomaly"]["evidence"]
                    evidence = [
                        row for row in anomaly_evidence if row.get("product_id") == "P005"
                    ]
                else:
                    evidence = task_evidence[: min(3, len(task_evidence))]
                if not evidence:
                    raise ValueError(f"missing gold evidence for {task}/{capability}")
                gold_metrics = {
                    key: value
                    for key, value in evidence[0].items()
                    if key not in {"evidence_id", "source"} | DIMENSION_KEYS
                    and isinstance(value, (int, float, str))
                }
                raw.append(
                    {
                        "case_id": f"CASE_{case_number:03d}",
                        "case_version": "1.0",
                        "dataset_version": manifest["dataset_version"],
                        "dataset_id": manifest["dataset_id"],
                        "generator_config_hash": manifest["config_sha256"],
                        "business_task": task,
                        "primary_capability": capability,
                        "capability_tags": [capability],
                        "difficulty": DIFFICULTY_BY_CAPABILITY[capability],
                        "user_input": _question(task, capability, variant),
                        "expected_tool_calls": [
                            {
                                "name": tool_name,
                                "parameters": _tool_parameters(tool_name),
                            }
                            for tool_name in (
                                ["query_traffic"]
                                if capability == "data_insufficiency"
                                else TOOL_SEQUENCE_BY_TASK[task]
                            )
                        ],
                        "allowed_alternatives": [],
                        "gold_metrics": gold_metrics,
                        "gold_metric_evidence": {
                            key: evidence[0]["evidence_id"]
                            for key in gold_metrics
                        },
                        "gold_evidence": [
                            {
                                "evidence_id": row["evidence_id"],
                                "source": row["source"],
                                "dimensions": {
                                    key: value
                                    for key, value in row.items()
                                    if key in DIMENSION_KEYS
                                },
                                "metrics": {
                                    key: value
                                    for key, value in row.items()
                                    if key not in {"evidence_id", "source"} | DIMENSION_KEYS
                                },
                            }
                            for row in evidence
                        ],
                        "reference_answer": (
                            "基于 Synthetic E-commerce Data，参考证据为："
                            f"{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
                        ),
                        "expected_behavior": [
                            "使用配置的比较窗口",
                            "结论绑定 gold evidence",
                            "不生成数据外事实",
                        ],
                        "success_criteria": SUCCESS_CRITERIA,
                        "numeric_tolerances": {
                            key: 0.001 for key in gold_metrics if isinstance(gold_metrics[key], float)
                        },
                        "metadata": {
                            "source_label": "Synthetic E-commerce Data",
                            "variant": variant + 1,
                        },
                    }
                )
                case_number += 1

    rng = np.random.default_rng(20260928)
    by_task: dict[str, list[dict]] = {task: [] for task in BUSINESS_TASKS}
    for payload in raw:
        by_task[payload["business_task"]].append(payload)
    for payloads in by_task.values():
        order = rng.permutation(len(payloads))
        development_indices = set(order[:14].tolist())
        for index, payload in enumerate(payloads):
            payload["split"] = "development" if index in development_indices else "holdout"

    cases = [EvaluationCase.model_validate(payload) for payload in raw]
    assert Counter(case.primary_capability for case in cases) == {
        capability: 10 for capability in CAPABILITIES
    }
    return cases
```

- [x] **Step 4: 实现 JSONL 与 Manifest 输出**

继续在 `app/data/case_generator.py` 增加：

```python
def write_cases(cases: list[EvaluationCase], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for case in cases
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "case_schema_version": "1.0",
        "dataset_id": cases[0].dataset_id,
        "generator_config_hash": cases[0].generator_config_hash,
        "source_label": "Synthetic E-commerce Data",
        "case_count": len(cases),
        "case_set_id": manifest_id({"lines": lines}),
        "business_task_counts": dict(Counter(case.business_task for case in cases)),
        "capability_counts": dict(Counter(case.primary_capability for case in cases)),
        "difficulty_counts": dict(Counter(case.difficulty for case in cases)),
        "split_counts": dict(Counter(case.split for case in cases)),
    }
    write_json(output_path.with_suffix(".manifest.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_cases/evaluation_cases_v1.jsonl"),
    )
    args = parser.parse_args()
    manifest = write_cases(build_cases(args.dataset), args.output)
    print(
        f\"Built {manifest['case_count']} evaluation cases \"
        f\"for dataset {manifest['dataset_id']}\"
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 5: 增加防泄漏测试**

追加到 `tests/test_case_generator.py`：

```python
def test_cases_do_not_expose_anomaly_config_or_holdout_answers(tmp_path) -> None:
    dataset_dir = tmp_path / "v1"
    build_snapshot("configs/data/synthetic_v1.yaml", dataset_dir)
    cases = build_cases(dataset_dir)

    serialized = "\n".join(case.model_dump_json() for case in cases)
    assert "anomaly_id" not in serialized
    assert "multiplier" not in serialized
    assert all("source_label" in case.metadata for case in cases)
```

- [x] **Step 6: 运行测试与真实构建**

Run:

```bash
python3 -m pytest tests/test_case_generator.py -v
python3 -m app.data.generator \
  --config configs/data/synthetic_holdout_v1.yaml
python3 -m app.data.case_generator \
  --development-dataset data/synthetic/v1 \
  --holdout-dataset data/synthetic/holdout-v1 \
  --tool-contract configs/evaluation/tool_contract_v1.yaml \
  --output data/evaluation_cases/v1
```

Expected:

```text
Built 100 evaluation cases for dataset <dataset_id>
```

再执行：

```bash
python3 -c "import json; from pathlib import Path; p=Path('data/evaluation_cases/v1/manifest.json'); d=json.loads(p.read_text()); assert d['case_count']==100; assert d['split_counts']=={'development':70,'holdout':30}; assert d['datasets']['development']['dataset_id'] != d['datasets']['holdout']['dataset_id']; print(d)"
```

Expected: 输出真实 Manifest；不包含任何 Agent 效果指标。

- [x] **Step 7: 提交 Case 生成器**

```bash
git add .gitignore Makefile PROJECT_STATUS.md app/data/case_generator.py \
  app/data/schemas.py configs/data/synthetic_holdout_v1.yaml \
  configs/evaluation/tool_contract_v1.yaml data/evaluation_cases/v1 \
  docs/superpowers/plans/2026-09-28-phase-1-synthetic-data-evaluation-dataset.md \
  tests/test_case_generator.py
git commit -m "feat(evaluation-data): freeze isolated split case set"
```

---

### Task 12: 建立数据探索 Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`
- Create: `tests/test_notebook.py`

- [x] **Step 1: 写 Notebook 结构失败测试**

`tests/test_notebook.py`：

```python
import nbformat


def test_data_exploration_notebook_has_required_sections() -> None:
    notebook = nbformat.read("notebooks/01_data_exploration.ipynb", as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    for section in (
        "Synthetic Data 声明",
        "数据范围",
        "质量检查",
        "指标分布",
        "异常场景",
        "局限性",
    ):
        assert section in markdown


def test_notebook_contains_no_hard_coded_evaluation_results() -> None:
    notebook = nbformat.read("notebooks/01_data_exploration.ipynb", as_version=4)
    content = "\n".join(cell.source for cell in notebook.cells)
    assert "优化后提升" not in content
    assert "准确率提升" not in content
```

- [x] **Step 2: 运行测试确认失败**

Run:

```bash
python3 -m pytest tests/test_notebook.py -v
```

Expected: FAIL，因为 Notebook 尚不存在。

- [x] **Step 3: 创建只读探索 Notebook**

使用 `nbformat` 创建 Notebook，固定包含以下 Cell：

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Phase 1 数据探索\n\n"
        "## Synthetic Data 声明\n"
        "本 Notebook 只分析程序生成的 Synthetic E-commerce Data，不代表真实业务。"
    ),
    nbf.v4.new_markdown_cell("## 数据范围"),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "DATASET = Path('../data/synthetic/v1')\n"
        "manifest = json.loads((DATASET / 'manifest.json').read_text())\n"
        "manifest"
    ),
    nbf.v4.new_markdown_cell("## 质量检查"),
    nbf.v4.new_code_cell(
        "quality = json.loads((DATASET / 'data_quality_report.json').read_text())\n"
        "quality"
    ),
    nbf.v4.new_markdown_cell("## 指标分布"),
    nbf.v4.new_code_cell(
        "traffic = pd.read_parquet(DATASET / 'traffic.parquet')\n"
        "traffic[['impressions', 'clicks', 'visits']].describe()"
    ),
    nbf.v4.new_markdown_cell("## 异常场景"),
    nbf.v4.new_code_cell(
        "traffic.groupby('product_id', as_index=False)['visits'].sum()"
        ".sort_values('visits').head(10)"
    ),
    nbf.v4.new_markdown_cell(
        "## 局限性\n"
        "- 数据由规则生成。\n"
        "- 异常覆盖有限。\n"
        "- 当前不代表真实电商生产分布。\n"
        "- 本阶段没有 Agent Evaluation 结果。"
    ),
]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nbf.write(nb, "notebooks/01_data_exploration.ipynb")
```

该创建代码只在本步骤执行，不保存为项目脚本。Notebook 不保存执行输出，避免提交机器相关状态。

- [x] **Step 4: 运行 Notebook 结构测试**

Run:

```bash
python3 -m pytest tests/test_notebook.py -v
```

Expected: 全部 PASS。

- [x] **Step 5: 提交 Notebook**

```bash
git add notebooks/01_data_exploration.ipynb tests/test_notebook.py
git commit -m "docs(data): add reproducible synthetic data exploration notebook"
```

---

### Task 13: 完成 Phase 1 端到端质量门禁

**Files:**
- Modify: `tests/test_data_generation.py`
- Modify: `tests/test_gold.py`
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: 增加端到端复现测试**

追加到 `tests/test_data_generation.py`：

```python
from app.data.case_generator import build_cases, write_cases
from app.data.generator import build_snapshot
from app.data.manifest import file_sha256


def test_phase1_outputs_are_reproducible(tmp_path) -> None:
    first_dataset = tmp_path / "first" / "v1"
    second_dataset = tmp_path / "second" / "v1"
    first_manifest = build_snapshot("configs/data/synthetic_v1.yaml", first_dataset)
    second_manifest = build_snapshot("configs/data/synthetic_v1.yaml", second_dataset)
    assert first_manifest["dataset_id"] == second_manifest["dataset_id"]

    first_cases = tmp_path / "first" / "cases.jsonl"
    second_cases = tmp_path / "second" / "cases.jsonl"
    first_case_manifest = write_cases(build_cases(first_dataset), first_cases)
    second_case_manifest = write_cases(build_cases(second_dataset), second_cases)
    assert first_case_manifest["case_set_id"] == second_case_manifest["case_set_id"]
    assert file_sha256(first_cases) == file_sha256(second_cases)
```

- [ ] **Step 2: 增加 Gold 独立性测试**

追加到 `tests/test_gold.py`：

```python
def test_gold_sql_does_not_read_generator_config() -> None:
    for path in Path("sql/gold").glob("*.sql"):
        sql = path.read_text(encoding="utf-8").lower()
        assert "synthetic_v1.yaml" not in sql
        assert "anomaly_id" not in sql
        assert "multiplier" not in sql
```

这保证 Gold SQL 只从事实表推导，不直接抄异常配置答案。

- [ ] **Step 3: 运行完整验证**

Run:

```bash
python3 -c "from pathlib import Path; import shutil; shutil.rmtree(Path('data/synthetic/v1'), ignore_errors=True); [p.unlink(missing_ok=True) for p in [Path('data/evaluation_cases/evaluation_cases_v1.jsonl'), Path('data/evaluation_cases/evaluation_cases_v1.manifest.json')]]"
make phase1
python3 -m pytest
python3 -m ruff check app tests
git diff --check
```

Expected:

- `make phase1` 成功生成五张 Parquet、Dataset Manifest、质量报告、100 Cases 和 Case Manifest。
- 全部测试 PASS。
- Ruff 无错误。
- Git whitespace 检查无错误。
- 不产生任何 Baseline、Optimized、Latency、Token、Cost 或显著性结果。

- [ ] **Step 4: 更新 README**

在 `README.md` 增加：

```markdown
## 数据真实性

本项目当前使用 `Synthetic E-commerce Data`。数据由
`configs/data/synthetic_v1.yaml` 和固定 Seed 程序化生成，不代表真实企业经营数据。

## Phase 1 产物

- 五表 Parquet 数据快照
- Dataset Manifest 与文件哈希
- 数据质量报告
- 100 个 Evaluation Cases
- Case Manifest 与开发集/保留集分布

Phase 1 尚未运行 Agent，因此所有 Agent Evaluation 结果仍为 `Pending / Not Run`。
```

- [ ] **Step 5: 更新 Phase 1 状态**

仅在所有验证通过后修改 `PROJECT_STATUS.md`：

```markdown
## Current Phase

`PHASE 1 — 模拟数据与 Evaluation Dataset`

状态：`Completed / Awaiting Review`
```

Completed 增加实际完成项和真实行数；Validation 记录实际执行命令与 PASS 结果；Evaluation Results 只将 Synthetic Dataset 与 Evaluation Dataset 更新为真实生成状态，Agent 相关项目继续保持 `Pending / Not Run`。

- [ ] **Step 6: 创建 Phase 1 验证提交**

```bash
git add README.md PROJECT_STATUS.md tests
git commit -m "test(data): verify phase 1 reproducibility and quality gates"
```

- [ ] **Step 7: 检查最终 Git 状态**

Run:

```bash
git status --short --branch
git log --oneline --max-count=15
```

Expected:

- 工作区干净。
- Phase 1 每个逻辑任务都有独立提交。
- 不存在 `.env`、API Key、真实用户数据或未说明来源的结果文件。

---

## Phase 1 最终验收清单

- [ ] Python 3.11+ 项目可安装。
- [ ] 无 API Key 时可以运行 Phase 1。
- [ ] 五张业务表均通过 Schema 与不变量测试。
- [ ] 数据明确标记为 `Synthetic E-commerce Data`。
- [ ] 同配置与 Seed 生成相同 Dataset ID。
- [ ] 配置中的七类异常可在数据中观察到。
- [ ] 指标定义通过独立手算 Fixture。
- [ ] Parquet 文件与 Manifest 哈希一致。
- [ ] DuckDB 只读加载前验证文件哈希。
- [ ] Gold SQL 不读取异常注入配置。
- [ ] 五类业务任务均有 Gold Evidence。
- [ ] 精确生成 100 个 Evaluation Cases。
- [ ] 五类业务任务各 20 Cases。
- [ ] 十类 Primary Capability 各 10 Cases。
- [ ] 难度分布为 30/40/30。
- [ ] Development/Holdout 分布为 70/30。
- [ ] Case 不泄漏异常配置。
- [ ] Case 与 Dataset ID 绑定。
- [ ] 数据和 Case 可重复生成。
- [ ] Notebook 明确说明 Synthetic 与局限性。
- [ ] 完整 Pytest 与 Ruff 检查通过。
- [ ] `PROJECT_STATUS.md` 记录实际结果。
- [ ] Agent 相关结果仍显示 `Pending / Not Run`。

## 停止条件

出现以下任一情况时停止，不进入 Phase 2：

- 同一配置无法生成相同 Dataset ID。
- Gold SQL 与异常配置直接耦合。
- 100 Cases 的覆盖或 Split 不满足固定分布。
- 数据质量检查失败。
- 测试或 Ruff 未通过。
- 生成产物包含 API Key、真实用户数据或未经运行的效果结论。
- 用户尚未审阅 Phase 1 实际交付。
