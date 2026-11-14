"""Load and validate the four Olist tables needed at purchase time."""

from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
}


def load_data(data_dir):
    """Return tables without downloading data or changing the working directory."""
    result = {}
    for name, filename in FILES.items():
        path = Path(data_dir) / filename
        if not path.is_file():
            raise ValueError(f"缺少数据文件：{path}；可先用 --demo 运行合成数据。")
        result[name] = pd.read_csv(path, dtype={
            "order_id": "string", "customer_id": "string",
            "customer_unique_id": "string", "product_id": "string",
        })
    return result


def timestamp(value, name="timestamp"):
    """Keep source local clock times; require callers to use one naive timezone."""
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} 不是有效时间：{value}") from exc
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError(f"{name} 必须是有效且不带时区的时间；所有文件须统一时区。")
    return result


def _required(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少字段：{', '.join(sorted(missing))}")


def _keys(frame, columns, name):
    if frame[columns].isna().any().any():
        raise ValueError(f"{name} 的主键/连接键不可为空：{columns}")
    for col in columns:
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.{col} 不可为空字符串。")


def _unique(frame, columns, name):
    _keys(frame, columns, name)
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} 存在重复键 {columns}；请先核对来源，避免金额重复计算。")


def prepare_transactions(tables):
    """Produce one row per priced order item, preserving one-to-many semantics.

    No final order status, delivery timestamp or review is used: those can be
    unknown at the historical cutoff. Amount means gross purchase value, not
    net revenue after cancellations, refunds or freight.
    """
    orders = tables["orders"].copy()
    customers = tables["customers"].copy()
    items = tables["items"].copy()
    products = tables["products"].copy()
    _required(orders, ["order_id", "customer_id", "order_purchase_timestamp"], "orders")
    _required(customers, ["customer_id", "customer_unique_id"], "customers")
    _required(items, ["order_id", "order_item_id", "product_id", "price"], "items")
    _required(products, ["product_id", "product_category_name"], "products")
    _unique(orders, ["order_id"], "orders")
    _keys(orders, ["customer_id"], "orders")
    _unique(customers, ["customer_id"], "customers")
    _keys(customers, ["customer_unique_id"], "customers")
    _keys(items, ["order_id", "order_item_id"], "items")
    _keys(items, ["product_id"], "items")
    _unique(products, ["product_id"], "products")
    for col in ["price", "order_item_id"]:
        items[col] = pd.to_numeric(items[col], errors="coerce")
        if not np.isfinite(items[col].to_numpy(dtype=float)).all():
            raise ValueError(f"items.{col} 必须是有限数值。")
    if (items["price"] < 0).any():
        raise ValueError("items.price 不能为负；退款需要独立的明细与口径。")
    if ((items["order_item_id"] < 1) | (items["order_item_id"] % 1 != 0)).any():
        raise ValueError("order_item_id 必须是正整数行序号，不能用作数量。")
    _unique(items, ["order_id", "order_item_id"], "items")
    try:
        orders["order_purchase_timestamp"] = pd.to_datetime(
            orders["order_purchase_timestamp"], errors="raise", format="mixed"
        )
        if orders["order_purchase_timestamp"].dt.tz is not None:
            raise ValueError("时间字段必须不带时区；请先统一所有数据的时区。")
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("orders.order_purchase_timestamp 包含无效或不一致时区的时间。") from exc
    if orders["order_purchase_timestamp"].isna().any():
        raise ValueError("order_purchase_timestamp 不可为空。")
    if not orders["customer_id"].isin(customers["customer_id"]).all():
        raise ValueError("orders 包含在 customers 中找不到的 customer_id。")
    if not items["order_id"].isin(orders["order_id"]).all():
        raise ValueError("items 包含在 orders 中找不到的 order_id。")
    if not items["product_id"].isin(products["product_id"]).all():
        raise ValueError("items 包含在 products 中找不到的 product_id。")
    result = (
        items[["order_id", "order_item_id", "product_id", "price"]]
        .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]],
               on="order_id", validate="many_to_one")
        .merge(customers[["customer_id", "customer_unique_id"]],
               on="customer_id", validate="many_to_one")
        .merge(products[["product_id", "product_category_name"]],
               on="product_id", validate="many_to_one")
    )
    result["product_category_name"] = result["product_category_name"].fillna("unknown")
    result["total_amount"] = result["price"]
    return result.sort_values(["order_purchase_timestamp", "order_id", "order_item_id"]).reset_index(drop=True)
