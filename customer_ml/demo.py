"""Synthetic examples only: not Olist records or evidence of real model quality."""

import numpy as np
import pandas as pd


DEMO_CUTOFF = "2025-04-01"
DEMO_DATA_THROUGH = "2025-06-30"


def make_demo(customers=400, seed=42):
    if customers < 40:
        raise ValueError("合成演示至少需要 40 个客户。")
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(DEMO_CUTOFF)
    orders, customer_rows, items = [], [], []
    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(8)],
        "product_category_name": ["electronics", "home", "books", "sport"] * 2,
    })
    for number in range(customers):
        stable_id = f"synthetic_customer_{number:05d}"
        activity = rng.uniform(0.3, 3.0)
        unit_price = rng.uniform(25, 190)
        history_count = 1 + rng.poisson(activity * 1.5)
        future_count = rng.poisson(activity * 0.65)
        for future, count in [(False, history_count), (True, future_count)]:
            for _ in range(count):
                oid = f"synthetic_order_{len(orders):06d}"
                cid = f"buyer_{len(orders):06d}"
                offset = int(rng.integers(0, 90)) if future else -int(rng.integers(1, 181))
                purchased = cutoff + pd.Timedelta(days=offset, hours=int(rng.integers(0, 24)))
                orders.append({"order_id": oid, "customer_id": cid,
                               "order_purchase_timestamp": purchased})
                customer_rows.append({"customer_id": cid, "customer_unique_id": stable_id})
                for item_id in range(1, int(rng.integers(1, 4)) + 1):
                    items.append({"order_id": oid, "order_item_id": item_id,
                                  "product_id": f"p{int(rng.integers(0, 8))}",
                                  "price": round(float(unit_price * rng.uniform(0.65, 1.35)), 2)})
    return {"orders": pd.DataFrame(orders), "customers": pd.DataFrame(customer_rows),
            "items": pd.DataFrame(items), "products": products}
