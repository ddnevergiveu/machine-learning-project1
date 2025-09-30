import pandas as pd

def load_data():
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    order_items = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")
    return orders, customers, order_items, products, reviews

def clean_orders(orders):
    # 填补缺失值，统一时间格式
    orders['order_approved_at'] = orders['order_approved_at'].fillna(orders['order_purchase_timestamp'])
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    return orders

def merge_data(orders, customers, order_items, products):
    df = orders.merge(customers, on="customer_id", how="left") \
               .merge(order_items, on="order_id", how="left") \
               .merge(products, on="product_id", how="left")
    df['total_amount'] = df['price'] * df['order_item_id']
    return df
