import pandas as pd
import numpy as np

def add_repeat_purchase_label(df):
    # 每个客户是否有重复购买
    purchase_counts = df.groupby('customer_id')['order_id'].nunique()
    repeat_purchase = purchase_counts.apply(lambda x: 1 if x > 1 else 0)
    df_repeat = df[['customer_id']].drop_duplicates().set_index('customer_id')
    df_repeat['repeat_purchase'] = repeat_purchase
    return df_repeat.reset_index()

def create_features(df):
    # 客户总订单数
    features = df.groupby('customer_id').agg({
        'order_id': 'nunique',
        'total_amount': ['mean', 'sum'],
        'product_category_name': 'nunique'
    })
    features.columns = ['total_orders', 'avg_order_amount', 'total_amount', 'unique_categories']
    features = features.reset_index()
    return features
