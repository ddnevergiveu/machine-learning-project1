# 高价值客户预测（High-Value Customer Prediction）

> 基于电商数据的机器学习项目，用于预测哪些客户是高价值客户（总消费 >= 500），帮助企业进行精准营销和客户管理。

---

## 📂 项目结构

.
├── data/ # 原始数据文件夹（orders, customers, products, order_items, reviews）
├── src/ # Python源代码
│ ├── data_processing.py # 数据读取与清洗
│ ├── feature_engineering.py # 特征工程
│ ├── ml_model.py # 模型训练与比较
│ └── visualization.py # 可视化函数
├── main.py # 主程序入口
├── requirements.txt # 项目依赖
└── README.md # 项目说明


<img width="800" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/98b9b933-7d9d-44c0-a003-c0ed7c1a092a" />


## 🛠 技术栈

- Python 3.11  
- 数据处理：`pandas`, `numpy`  
- 可视化：`matplotlib`, `seaborn`  
- 机器学习模型：`scikit-learn` (LogisticRegression, RandomForest, KNeighbors, etc.)  
- LightGBM（可选，但已去掉训练失败模型）  

---

## 🔍 项目流程

1. **数据读取与清洗**  
   - 合并订单、客户、商品及评论数据  
   - 填补缺失值  

2. **特征工程**  
   - 计算每个客户的：
     - 总订单数 (`total_orders`)  
     - 平均订单金额 (`avg_order_amount`)  
     - 总消费金额 (`total_amount`)  
     - 购买的不同商品类别数量 (`unique_categories`)  
   - 构建标签：高价值客户（消费总额 >= 500）

3. **模型训练与对比**  
   - 使用多种分类模型训练  
   - 对比 Accuracy、F1、ROC-AUC  
   - 输出最佳模型及特征重要性

4. **可视化分析**  
   - 特征重要性柱状图  
   - 各类销售趋势图（可选）

---

## ⚙️ 使用方法

1. 克隆仓库：

```bash
git clone https://github.com/ddnevergiveu/machine-learning.git
cd machine-learning
安装依赖：

bash
复制代码
pip install -r requirements.txt
运行项目：

bash
复制代码
python main.py
📊 项目效果
输出每个模型的性能指标：

Accuracy

F1 Score

ROC-AUC

绘制特征重要性图，帮助分析客户消费行为关键因素

🔧 未来优化方向
增加更多特征：如最近购买时间、复购率趋势等

尝试深度学习模型进行预测

增加类别不均衡处理方法（如 SMOTE）

可视化更多数据分析图表（如客户地域分布、月度销售趋势）

📌 联系方式
作者：姜圣涛

GitHub: ddnevergiveu

yaml
复制代码
