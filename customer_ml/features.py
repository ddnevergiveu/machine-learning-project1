"""Separate the historical feature window from the future label window."""

import numpy as np
import pandas as pd

from .data import timestamp


FEATURE_COLUMNS = [
    "total_orders", "total_amount", "avg_order_amount", "unique_categories",
    "recency_days", "tenure_days",
]
TARGETS = ("high_value", "repeat_purchase")


def create_features(transactions, cutoff):
    cutoff = timestamp(cutoff, "cutoff")
    history = transactions[transactions["order_purchase_timestamp"] < cutoff].copy()
    if history.empty:
        raise ValueError("截止时间之前没有带有效商品金额的订单，无法构造客户特征。")
    per_order = history.groupby(["customer_unique_id", "order_id"], sort=True).agg(
        amount=("total_amount", "sum"), purchased_at=("order_purchase_timestamp", "min")
    ).reset_index()
    features = per_order.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"),
        total_amount=("amount", "sum"), avg_order_amount=("amount", "mean"),
        first_purchase=("purchased_at", "min"), last_purchase=("purchased_at", "max"),
    )
    features["unique_categories"] = history.groupby("customer_unique_id")["product_category_name"].nunique()
    features["recency_days"] = (cutoff - features.pop("last_purchase")).dt.total_seconds() / 86400
    features["tenure_days"] = (cutoff - features.pop("first_purchase")).dt.total_seconds() / 86400
    return features[FEATURE_COLUMNS].reset_index()


def build_dataset(transactions, cutoff, horizon_days, data_through, value_threshold=500.0):
    """Build one record per existing customer; all interval ends are exclusive."""
    cutoff = timestamp(cutoff, "cutoff")
    data_through = timestamp(data_through, "data_through")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, (int, np.integer)) or horizon_days < 1:
        raise ValueError("horizon_days 必须为正整数。")
    if not np.isfinite(value_threshold) or value_threshold <= 0:
        raise ValueError("value_threshold 必须为有限正数。")
    label_end = cutoff + pd.Timedelta(days=int(horizon_days))
    if data_through < label_end:
        raise ValueError(f"标签窗口不完整：数据覆盖终点 {data_through} 早于 {label_end}。")
    if (transactions["order_purchase_timestamp"] >= data_through).any():
        raise ValueError("输入订单超出 --data-through 声明的数据覆盖范围，请核对观察终点。")
    features = create_features(transactions, cutoff).set_index("customer_unique_id")
    future = transactions[
        (transactions["order_purchase_timestamp"] >= cutoff)
        & (transactions["order_purchase_timestamp"] < label_end)
    ]
    future_amount = future.groupby("customer_unique_id")["total_amount"].sum()
    future_orders = future.groupby("customer_unique_id")["order_id"].nunique()
    features["future_amount"] = future_amount.reindex(features.index, fill_value=0)
    features["future_orders"] = future_orders.reindex(features.index, fill_value=0)
    features["high_value"] = (features["future_amount"] >= value_threshold).astype(int)
    features["repeat_purchase"] = (features["future_orders"] > 0).astype(int)
    return features.reset_index()
