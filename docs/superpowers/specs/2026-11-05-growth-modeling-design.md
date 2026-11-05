# 用户 7 日购买倾向与优惠券召回实验设计

## 1. 目标

在现有电商合成数据与 Agent 评测系统之外，新增一个独立、可复现的用户增长建模节点：

1. 使用预测时点之前 28 天的用户行为，预测未来 7 天是否下单。
2. 将购买倾向转化为固定风险分层。
3. 在风险层内执行随机优惠券实验，评估增量转化、成本和 ROI。
4. 输出可审计的人群召回策略，而不是把低购买概率直接等同于高优惠券响应。

该节点覆盖转化预估、用户分群、A/B Test、异质性分析和策略落地。

## 2. 隔离边界

增长建模是新的 `Batch F`，不得修改现有冻结资产：

- 不修改 `data/synthetic/v1/` 或 `data/synthetic/public-validation-v1/`。
- 不修改冻结 Pilot/Formal Case、V1/V2 Prompt、RunSettings、Rubric 或 E1-E10 Taxonomy。
- 不读写 `outputs/experiment_runs/`、`outputs/real_runs/` 或 Phase 3 历史数据。
- 所有增长数据、模型和报告使用独立版本与输出目录。

新增目录：

```text
configs/growth/
data/growth/v1/
app/growth/
outputs/growth/
```

## 3. 系统架构

增长节点由六个边界清晰的模块组成：

1. `generator`：生成版本化用户、行为和实验数据。
2. `features`：按时间点构造无泄漏特征和未来 7 天标签。
3. `modeling`：训练、选择和校准购买倾向模型。
4. `experiment`：在冻结风险层内执行随机优惠券实验分析。
5. `policy`：根据统计证据和经济性生成召回策略。
6. `artifacts`：原子发布数据 Manifest、模型指标、实验结果和模型卡。

数据流：

```text
Versioned config
  -> deterministic growth data
  -> point-in-time snapshots
  -> temporal train/validation/test split
  -> propensity model and calibration
  -> frozen risk cut points
  -> blocked randomized coupon assignment
  -> ITT effect and economics
  -> coupon policy and model card
```

## 4. 数据设计

### 4.1 默认规模

冻结配置 `configs/growth/growth_v1.yaml` 使用：

- 随机种子：`20261105`
- 用户数：`5000`
- 行为跨度：`180` 天
- 观察窗口：`28` 天
- 标签窗口：`7` 天
- 样本锚点：每 7 天一个，预热期后共 `20` 个锚点
- 优惠券面额：`5.00`
- 最低消费门槛：`30.00`
- 有效期：`7` 天
- 单次触达成本：`0.05`
- Treatment 分配概率：每个风险层内 `0.50`

全部值写入配置和 Manifest，不在代码中隐式漂移。

### 4.2 数据表

`customers.parquet`

- `customer_id`
- `signup_date`
- `region`
- `acquisition_channel`
- `is_new_customer`

`customer_events.parquet`

- `event_id`
- `customer_id`
- `event_time`
- `event_type`
- `session_id`
- `product_id`
- `category`
- `unit_price`
- `channel`

`event_type` 限定为：

- `impression`
- `click`
- `visit`
- `favorite`
- `add_to_cart`
- `purchase`
- `refund`

`customer_snapshots.parquet`

- `customer_id`
- `as_of_date`
- 所有预测特征
- `ordered_next_7d`
- `orders_next_7d`
- `gmv_next_7d`

`campaign_assignments.parquet`

- `experiment_id`
- `customer_id`
- `assignment_time`
- `risk_stratum`
- `treatment_group`
- `assignment_probability`
- `coupon_value`
- `coupon_threshold`
- `coupon_expiry_days`

`campaign_outcomes.parquet`

- `experiment_id`
- `customer_id`
- `ordered_next_7d`
- `orders_next_7d`
- `gmv_next_7d`
- `coupon_redeemed`
- `contribution_margin`
- `campaign_cost`

### 4.3 样本定义

每个建模样本为：

```text
customer_id x as_of_date
```

- 特征窗口：`[as_of_date - 28 days, as_of_date)`
- 标签窗口：`[as_of_date, as_of_date + 7 days)`
- 主标签：标签窗口内是否至少产生一笔有效订单
- 有效订单：`status=completed` 且不是全额退款

相邻锚点间隔固定为 7 天，使标签窗口不重叠。

### 4.4 Campaign 召回人群

最终 Campaign Cohort 满足：

- 过去 28 天至少有一次站内行为。
- 过去 7 天没有有效订单。
- 历史上至少有一次购买或加购行为。
- 不使用标签窗口信息决定资格。

模型在历史无 Treatment 样本上训练。Campaign Cohort 在评分后按冻结风险切点分层，再在每层内进行 1:1 随机分配。

### 4.5 合成处理效应

优惠券处理效应由冻结生成器决定，并允许异质性：

- 中低自然购买倾向、近期加购但未购买的用户具有较高响应。
- 极低活跃用户响应较弱。
- 高自然购买倾向用户可能发生补贴浪费。

Treatment 分配与潜在结果独立。模型和策略模块只能读取观测结果，不读取潜在结果或生成器的隐藏响应参数。

## 5. 特征设计

所有特征只能读取 `as_of_date` 之前的数据。

### 5.1 RFM

- 距最近一次购买天数
- 7/14/28 天订单数
- 7/14/28 天 GMV
- 28 天客单价
- 历史累计购买次数

### 5.2 活跃与漏斗

- 7/14/28 天访问次数
- 活跃天数
- 点击次数与点击率
- 收藏次数
- 加购次数与加购率
- 历史购买转化率

### 5.3 偏好与质量

- 浏览品类数
- 购买品类数
- 品类集中度
- 平均浏览价格带
- 平均购买价格带
- 退款次数与退款率

### 5.4 用户属性

- 地区
- 获客渠道
- 新老客标记
- 账户年龄

### 5.5 历史营销

- 历史触达次数
- 历史优惠券核销率
- 距上次触达天数

当前 Campaign 的 Treatment、核销状态和任何标签窗口字段禁止进入模型。

## 6. 数据切分与泄漏控制

按 `as_of_date` 做时间切分：

- Train：最早 60% 锚点
- Validation：随后 20% 锚点
- Test：最后 20% 锚点

约束：

- 不随机打散时间。
- 特征聚合必须使用半开窗口并显式传入 `as_of_date`。
- 类别编码、缺失填充、标准化和校准只能在 Train/Validation 上拟合。
- Test 只运行一次最终评估。
- 风险层切点只能由 Validation 预测分布确定。
- Campaign 分层不得使用 Campaign 结果。

## 7. 模型设计

### 7.1 候选模型

基线模型：

- `LogisticRegression`
- 数值特征标准化
- 类别特征 One-Hot
- `class_weight` 是否启用只依据 Validation PR-AUC

非线性模型：

- `HistGradientBoostingClassifier`
- 冻结的类别编码与缺失值处理
- 搜索空间在运行前写入配置

依赖新增：

- `scikit-learn`
- `statsmodels`

### 7.2 模型选择

主指标：Validation PR-AUC。

平分时依次比较：

1. Brier Score 更低
2. Calibration Error 更低
3. 模型复杂度更低

选定后在 Validation 上做概率校准，再对 Test 运行一次。

### 7.3 冻结评估指标

- PR-AUC
- ROC-AUC
- Brier Score
- Expected Calibration Error
- Lift@10%
- Lift@20%
- 正样本率
- 各地区、渠道、新老客分组指标
- 时间段稳定性

风险层为 Validation 分数的固定五分位切点。切点和边界规则写入模型 Artifact。

## 8. 优惠券随机实验

### 8.1 分配

在五个风险层内分别进行确定性种子驱动的 1:1 blocked randomization：

- `treatment`：获得优惠券
- `control`：不获得优惠券

每层输出样本量和预处理变量平衡性。任一核心变量的标准化差异绝对值超过 `0.10` 时标记随机化检查失败，不生成自动投放策略。

### 8.2 估计量

主估计量为 Intent-to-Treat：

```text
incremental_conversion =
  conversion_rate_treatment - conversion_rate_control
```

每个风险层输出：

- Treatment / Control 样本量
- 两组转化率
- 增量转化率
- Newcombe 95% 置信区间
- 双侧比例检验 p-value
- Holm 校正后的 p-value

整体结果同时给出分层加权平均处理效应。

### 8.3 经济性

使用实际商品 `price - cost` 计算贡献利润。

每个风险层输出：

```text
incremental_margin =
  incremental_orders x expected_margin_per_order

campaign_cost =
  redeemed_coupons x coupon_value
  + assigned_users x contact_cost

roi =
  (incremental_margin - campaign_cost) / campaign_cost
```

当 `campaign_cost=0` 时，ROI 状态为 `not_applicable`，不得填充为 0。

## 9. 优惠券决策规则

自动推荐一个风险层进入召回名单必须同时满足：

1. Newcombe 95% 置信区间下界大于 0。
2. Holm 校正后的 p-value 小于 `0.05`。
3. 预期增量贡献利润大于 0。
4. ROI 大于 0。
5. Treatment 和 Control 每组样本数均不少于 `100`。
6. 随机化平衡性检查通过。

状态：

- `recommend`：满足全部规则。
- `do_not_target`：增量或经济性明确不成立。
- `insufficient_evidence`：区间跨 0、样本量不足或平衡性失败。

不得把 `insufficient_evidence` 改写为有效策略。

## 10. Artifact 与发布

输出目录：

```text
outputs/growth/<run_id>/
├── dataset_manifest.json
├── split_manifest.json
├── model_metrics.json
├── calibration.json
├── feature_importance.json
├── risk_cutpoints.json
├── experiment_balance.json
├── segment_effects.json
├── coupon_policy.json
└── model_card.md
```

要求：

- JSON 使用 UTF-8 canonical serialization。
- Manifest 绑定配置、输入数据、代码版本、依赖版本和输出 SHA-256。
- 临时目录完整回读后原子发布。
- 正式目录不可覆盖。
- `data/growth/v1/` 和 `outputs/growth/` 不修改现有 Phase 1-3 资产。
- 输出不包含隐藏潜在结果或生成器内部响应参数。

## 11. 错误处理

- 配置、Schema、数据哈希、时间边界、随机化平衡性均 fail-fast。
- 单个模型失败不阻断另一个候选模型，但失败模型不得参与选择。
- 两个候选模型均失败时整批失败，不发布半成品。
- Test 评估或 Campaign 分析失败时不发布策略。
- 错误 Artifact 只记录稳定错误码，不写入本地绝对路径或敏感内容。

## 12. 测试策略

### 12.1 数据生成

- 固定种子字节级可复现。
- 主键唯一、外键完整、时间范围合法。
- Treatment 在风险层内接近 1:1。
- 潜在结果不进入可训练数据。

### 12.2 泄漏

- 特征窗口不含 `as_of_date` 及之后事件。
- 标签只来自未来 7 天。
- 当前 Campaign 字段无法进入模型 Pipeline。
- 修改标签窗口订单会改变标签但不改变特征。

### 12.3 模型

- 时间切分顺序固定。
- Pipeline 只在 Train/Validation 拟合。
- 指标在常量预测、全错预测和不平衡标签下有明确行为。
- 模型选择只读取 Validation 指标。

### 12.4 实验

- Assignment 可复现且层内唯一。
- ITT 估计、Newcombe 区间、Holm 校正与经济公式有已知值测试。
- 证据不足不会生成 `recommend`。

### 12.5 发布

- Manifest 哈希、canonical bytes、不可覆盖、符号链接拒绝和原子发布测试。
- 全量既有测试保持通过。

## 13. CLI

新增入口：

```text
build-growth-data
train-growth-model
evaluate-coupon-experiment
run-growth-pipeline
```

`run-growth-pipeline` 串联完整流程，但每一步仍可独立执行和验证。

## 14. README 与 GitHub

README 新增 `Growth Modeling` 节点，包含：

- 业务问题
- 数据与时间切分
- 模型和指标
- 随机实验设计
- 策略规则
- 可复现命令
- 真实运行结果
- 局限性

功能在 `feat/growth-modeling` 分支实现并推送 GitHub。完整验证后创建独立 Pull Request，不直接推送或合并到 `main`。

## 15. 验收标准

完成标准：

1. 新增长数据不修改任何冻结评测资产。
2. 数据生成、特征、标签和时间切分可复现。
3. 至少两个候选模型完成 Validation 选择和一次 Test 评估。
4. 五个风险层均有随机实验结果或明确的证据不足状态。
5. 策略严格执行冻结统计和经济规则。
6. Artifact 原子发布且可独立验证。
7. README 仅引用正式 Artifact 指标。
8. 全量测试和 Ruff 通过。
9. 功能分支推送 GitHub，并通过独立 Pull Request 交付。
