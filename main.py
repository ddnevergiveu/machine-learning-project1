
import pandas as pd
import numpy as np
from src.data_cleaning import load_data, clean_orders, merge_data
from src.feature_engineering import add_repeat_purchase_label, create_features
from src.ml_model import train_and_compare_models
from src.visualization import plot_feature_importance

# ---------------- 1. 读取和清洗数据 ----------------
orders, customers, order_items, products, reviews = load_data()
orders = clean_orders(orders)
df = merge_data(orders, customers, order_items, products)

# ---------------- 2. 特征工程 ----------------
# 原始重复购买标签
labels_repeat = add_repeat_purchase_label(df)

# 特征生成
features = create_features(df)

# 生成新标签：高价值客户（总消费 >= 500）
customer_total = df.groupby('customer_id')['total_amount'].sum()
labels_high_value = pd.DataFrame({
    'customer_id': customer_total.index,
    'high_value_customer': (customer_total >= 500).astype(int)
})

# ---------------- 处理索引和合并 ----------------
# 确保 features 中 customer_id 是列，而不是索引
features = features.reset_index() if 'customer_id' in features.index.names else features
labels_high_value = labels_high_value.reset_index(drop=True)

# 合并特征和标签
data = features.merge(labels_high_value, on='customer_id')

# ---------------- 3. 准备训练数据 ----------------
X = data[['total_orders','avg_order_amount','total_amount','unique_categories']].copy()
y = data['high_value_customer'].copy()

# 检查缺失值并填充
print("标签缺失值：", y.isnull().sum())
y = y.fillna(0)

print("特征缺失值：")
print(X.isnull().sum())
X = X.fillna(0)

# ---------------- 4. 训练并对比多个模型 ----------------
models, results, best_model = train_and_compare_models(X, y)

# ---------------- 5. 可视化特征重要性 ----------------
plot_feature_importance(best_model, X, save_path="feature_importance.png", show=False)







