# Customer ML｜客户未来价值与复购预测

围绕同一个电商场景，完成两个可以独立运行的机器学习项目：

| 项目 | 需要回答的问题 | 运行参数 |
| --- | --- | --- |
| 未来高价值客户预测 | 截止日之后 90 天，下单商品金额是否达到指定阈值？ | `--target high_value` |
| 未来复购预测（新增） | 已有客户在未来 90 天是否再次下单？ | `--target repeat_purchase` |

**用截止日之前的行为预测之后的结果。** 项目提供数据校验、时间窗口、基线与模型比较、客户隔离的训练/验证/测试集、模型保存和加载预测。默认演示完全离线，数据由程序合成，所有输出都标明来源；演示指标不代表 Olist 真实表现。

## 快速运行

需要 Python 3.10 或以上。首次安装依赖需要网络，安装完成后演示不联网。

```bash
git clone https://github.com/ddnevergiveu/machine-learning-project1.git
cd machine-learning-project1
python -m venv .venv
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装并运行：

```bash
python -m pip install -r requirements.txt
python main.py train --demo --output-dir artifacts/high-value
python main.py train --demo --target repeat_purchase --output-dir artifacts/repeat-purchase
python main.py predict --demo --model artifacts/high-value/model.joblib --output artifacts/predictions.csv
```

也支持 `python -m customer_ml ...` 和安装后的 `customer-ml ...`。根入口保留了 `python main.py --demo` 的简写。

## 本轮修复

- **金额计算**：Olist 的 `order_item_id` 是订单内明细序号；商品金额直接求和 `price`，不再相乘。客单价先按订单汇总，再对订单金额求平均。
- **客户识别**：通过 `customer_id` 连接订单，再用 `customer_unique_id` 识别跨订单的同一位客户。
- **目标泄漏**：旧流程用总消费生成标签，同时将同一总消费作为特征；现在历史金额与未来标签分属不同窗口。
- **评估泄漏**：候选模型只比较验证集，选定后重新拟合训练+验证集，测试集仅评估最后选中的模型。
- **环境与入口**：修复不存在的 `src` 导入，补依赖/安装配置；去除未使用的 XGBoost、LightGBM 导入和 macOS 专用字体依赖，绘图无须桌面环境。
- **数据质量**：拒绝重复主键、缺失客户键、未匹配外键、非法日期、负数/非有限价格，避免连接导致金额膨胀。

## 真实数据

自行从 [Olist 官方数据集页面](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) 获取数据，并遵守该数据集的使用条件。仓库不复制数据集，也不自动下载。将以下四张表放入同一个目录：

| 文件 | 必须字段 |
| --- | --- |
| `olist_orders_dataset.csv` | `order_id`、`customer_id`、`order_purchase_timestamp` |
| `olist_customers_dataset.csv` | `customer_id`、`customer_unique_id` |
| `olist_order_items_dataset.csv` | `order_id`、`order_item_id`、`product_id`、`price` |
| `olist_products_dataset.csv` | `product_id`、`product_category_name` |

所有时间必须使用一致的、不带时区的本地时间。缺失商品类别记为 `unknown`；没有商品明细的订单不进入金额/客户样本，因此分析对象是**有商品明细的下单客户**。

真实训练需要明确两个日期，下面的占位值必须替换：

```bash
python main.py train --data-dir data/olist --cutoff YYYY-MM-DD --data-through YYYY-MM-DD --horizon-days 90 --value-threshold 500 --target high_value
```

- `--cutoff`：特征的截止时刻，不包含此时刻。
- `--data-through`：你核实过的完整订单数据覆盖终点，不包含此时刻，必须覆盖完整标签窗口。程序同时拒绝该终点以后的输入订单。
- **最后一条订单时间不能证明数据完整。** `data-through` 是由数据提供方/采集流程确认的覆盖声明，程序不能替你证明没有漏单。
- 至少需要 40 位历史客户，正负两类各至少 10 位；不足时会给出明确错误。默认 500 使用原始数据的货币单位，并非自动换算后的美元。

如需查看 CSV 输入，可以导出演示数据，再走与真实数据相同的文件入口：

```bash
python main.py demo-data --output-dir data/demo
python main.py train --data-dir data/demo --cutoff 2025-04-01 --data-through 2025-06-30 --output-dir artifacts/demo-csv
```

生成目录包含 `DEMO_METADATA.json`；从该目录训练仍会标注为合成数据。

## 时间与金额口径

| 内容 | 定义 |
| --- | --- |
| 历史特征 | 所有 `purchase_timestamp < cutoff` 的有价订单明细 |
| 标签窗口 | `cutoff <= purchase_timestamp < cutoff + horizon_days` |
| 目标人群 | 截止日之前至少有一笔带有效商品明细订单的客户；未来新增客户不在此模型范围 |
| 历史特征列 | 订单数、商品总金额、平均订单金额、品类数、距最近下单天数、首单距截止日天数 |
| 高价值标签 | 未来窗口内商品下单金额合计达到阈值 |
| 复购标签 | 未来窗口内至少有一个订单 |
| 金额含义 | 商品下单毛额；不含运费，不扣退款/取消，不等于已确认收入 |

历史特征不使用最终交付状态、最终取消状态或未来评论，避免拿截止日之后才知道的信息筛选历史订单。原始 Olist 是静态导出，无法完整还原字段修改/数据到达时点；生产环境仍需在时点快照上核验这些字段当时是否可用。详见 [方法与限制](docs/methodology.md)。

## 模型与输出

每位客户一行，按标签分层为 60% 训练、20% 验证、20% 测试；三部分客户不重叠。中位数填补与标准化位于 `Pipeline` 中，候选训练时只在训练集拟合。比较先验概率基线、逻辑回归、随机森林，以**验证集 Average Precision（AP）**选模，固定分类阈值 0.5。

| 输出文件 | 用途 |
| --- | --- |
| `metrics.json` | 数据来源、时间窗口、版本、随机种子、分区样本数、验证集比较、最终测试集指标 |
| `customer_splits.csv` | 每位客户所属分区，可检查重叠与复现实验 |
| `test_predictions.csv` | 留出的测试客户真实标签和模型分数 |
| `model.joblib` | 已选模型及特征/目标/窗口元数据 |
| `feature_weights.csv` / `.png` | 模型权重诊断；不解释因果关系 |

`predict` 导出的 `probability` 是未经校准的模型分数，尤其在类别加权后，不应直接当作真实复购概率或营销收益。此处客户随机留出验证的是同一历史时点下的人群泛化，**没有证明未来跨月份的效果**；正式使用前应增加滚动时间外验证、业务成本阈值和概率校准。

模型文件使用 joblib，**只加载自己生成或完全信任的文件**，并保持训练和推理依赖版本一致。参见 [scikit-learn 模型保存说明](https://scikit-learn.org/stable/model_persistence.html)。

## 代码导航与验证

| 路径 | 作用 |
| --- | --- |
| `customer_ml/data.py` | 读表、校验、按正确基数连接 |
| `customer_ml/features.py` | 历史特征与未来标签 |
| `customer_ml/model.py` | 分区、Pipeline、验证集选模及测试评估 |
| `customer_ml/demo.py` | 固定种子的合成订单生成器 |
| `customer_ml/cli.py` | 离线演示、CSV 入口、训练与预测 |
| `tests/test_customer_ml.py` | 金额、客户身份、时间边界、泄漏与可复现性回归测试 |
| `.github/workflows/ci.yml` | Python 3.10 / 3.12 自动测试与两种端到端演示 |

```bash
python -m unittest discover -s tests -v
```

根目录旧模块保留为轻量导入入口，函数 API 已以 `customer_ml` 包为准，不再执行原先存在泄漏的旧训练流程。已有两张历史图片保留，但本次新结果以运行后生成的文件为准。

## 相关项目

- [data-analysis](https://github.com/ddnevergiveu/data-analysis)：电商分析、客户分群与分析工具。
- [stock-da](https://github.com/ddnevergiveu/stock-da)：行情数据质量、SQLite 分析与策略研究。

作者：姜圣涛 / [ddnevergiveu](https://github.com/ddnevergiveu)
