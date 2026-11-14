"""Regression tests for amounts, customer identity and information boundaries."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from customer_ml.data import prepare_transactions
from customer_ml.demo import DEMO_CUTOFF, DEMO_DATA_THROUGH, make_demo
from customer_ml.features import FEATURE_COLUMNS, build_dataset, create_features
from customer_ml.model import split_customers, train_and_evaluate


def fixture():
    return {
        "orders": pd.DataFrame({
            "order_id": ["a", "b", "c", "d"],
            "customer_id": ["buyer_a", "buyer_b", "buyer_c", "buyer_d"],
            "order_purchase_timestamp": ["2025-01-01", "2025-02-01", "2025-04-01", "2025-05-01"],
            "order_status": ["delivered", "canceled", "delivered", "delivered"],
        }),
        "customers": pd.DataFrame({
            "customer_id": ["buyer_a", "buyer_b", "buyer_c", "buyer_d"],
            "customer_unique_id": ["person1"] * 4,
        }),
        "items": pd.DataFrame({
            "order_id": ["a", "a", "b", "c", "d"], "order_item_id": [1, 2, 1, 1, 1],
            "product_id": ["p1"] * 5, "price": [10.0, 20.0, 30.0, 100.0, 999.0],
        }),
        "products": pd.DataFrame({"product_id": ["p1"], "product_category_name": ["books"]}),
    }


class DataBoundaryTests(unittest.TestCase):
    def test_item_sequence_is_not_quantity_and_person_key_is_stable(self):
        frame = prepare_transactions(fixture())
        features = create_features(frame, "2025-04-01").iloc[0]
        self.assertEqual(features["customer_unique_id"], "person1")
        self.assertEqual(features["total_orders"], 2)
        self.assertEqual(features["total_amount"], 60)
        self.assertEqual(features["avg_order_amount"], 30)

    def test_cutoff_included_in_label_but_label_end_excluded(self):
        data = build_dataset(prepare_transactions(fixture()), "2025-04-01", 30, "2025-05-02", 100)
        self.assertEqual(data.loc[0, "future_amount"], 100)
        self.assertEqual(data.loc[0, "high_value"], 1)
        self.assertEqual(data.loc[0, "repeat_purchase"], 1)

    def test_future_changes_cannot_change_features(self):
        original = prepare_transactions(fixture())
        changed = original.copy()
        future = changed["order_purchase_timestamp"] >= pd.Timestamp("2025-04-01")
        changed.loc[future, "total_amount"] = 99999
        changed.loc[future, "product_category_name"] = "future_only_category"
        assert_frame_equal(create_features(original, "2025-04-01"), create_features(changed, "2025-04-01"))
        self.assertFalse(set(["future_amount", "future_orders", "high_value", "repeat_purchase"]) & set(FEATURE_COLUMNS))

    def test_final_order_status_is_not_a_historical_feature_filter(self):
        tables = fixture()
        expected = prepare_transactions(tables)
        tables["orders"]["order_status"] = "canceled"
        assert_frame_equal(expected, prepare_transactions(tables))

    def test_incomplete_label_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "标签窗口不完整"):
            build_dataset(prepare_transactions(fixture()), "2025-04-01", 90, "2025-05-02")

    def test_false_coverage_declaration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "覆盖范围"):
            build_dataset(prepare_transactions(fixture()), "2025-04-01", 1, "2025-04-02")

    def test_duplicate_join_keys_fail_instead_of_inflating_amounts(self):
        for name in ["orders", "customers", "items", "products"]:
            with self.subTest(table=name):
                tables = fixture()
                tables[name] = pd.concat([tables[name], tables[name].iloc[[0]]])
                with self.assertRaisesRegex(ValueError, "重复键"):
                    prepare_transactions(tables)

    def test_bad_price_and_missing_foreign_key_are_rejected(self):
        for bad in [-1, np.nan, np.inf]:
            tables = fixture()
            tables["items"].loc[0, "price"] = bad
            with self.assertRaises(ValueError):
                prepare_transactions(tables)
        tables = fixture()
        tables["items"].loc[0, "product_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "product_id"):
            prepare_transactions(tables)

    def test_duplicate_item_sequence_after_numeric_conversion_is_rejected(self):
        tables = fixture()
        tables["items"]["order_item_id"] = tables["items"]["order_item_id"].astype(str)
        tables["items"].loc[1, "order_item_id"] = "01"
        with self.assertRaisesRegex(ValueError, "重复键"):
            prepare_transactions(tables)

    def test_missing_stable_customer_id_is_rejected(self):
        tables = fixture()
        tables["customers"] = tables["customers"].drop(columns="customer_unique_id")
        with self.assertRaisesRegex(ValueError, "customer_unique_id"):
            prepare_transactions(tables)

    def test_inputs_are_not_mutated(self):
        tables = fixture()
        originals = {name: frame.copy(deep=True) for name, frame in tables.items()}
        prepare_transactions(tables)
        for name in tables:
            assert_frame_equal(tables[name], originals[name])

    def test_new_future_only_customer_is_not_in_existing_customer_population(self):
        tables = fixture()
        tables["customers"].loc[3, "customer_unique_id"] = "new_future_customer"
        data = build_dataset(prepare_transactions(tables), "2025-04-01", 30, "2025-05-02")
        self.assertEqual(data["customer_unique_id"].tolist(), ["person1"])

    def test_invalid_timestamp_or_horizon_is_rejected(self):
        tables = fixture()
        tables["orders"].loc[0, "order_purchase_timestamp"] = "invalid"
        with self.assertRaisesRegex(ValueError, "无效"):
            prepare_transactions(tables)
        for horizon in [0, -2, 1.5, True]:
            with self.assertRaises(ValueError):
                build_dataset(prepare_transactions(fixture()), "2025-04-01", horizon, "2025-05-02")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset(prepare_transactions(make_demo(200)), DEMO_CUTOFF, 90, DEMO_DATA_THROUGH)

    def test_customers_are_disjoint_and_every_split_has_both_classes(self):
        parts = split_customers(self.dataset, "high_value")
        sets = [set(frame["customer_unique_id"]) for frame in parts.values()]
        self.assertEqual(sum(map(len, sets)), len(self.dataset))
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                self.assertFalse(left & right)
        for frame in parts.values():
            self.assertEqual(set(frame["high_value"]), {0, 1})

    def test_single_class_and_duplicate_customer_fail_clearly(self):
        single = self.dataset.copy()
        single["high_value"] = 0
        with self.assertRaisesRegex(ValueError, "正负两类"):
            split_customers(single, "high_value")
        with self.assertRaisesRegex(ValueError, "只有一条"):
            split_customers(pd.concat([self.dataset, self.dataset.iloc[[0]]]), "high_value")

    def test_test_features_cannot_affect_selection_or_preprocessing(self):
        partitions = split_customers(self.dataset, "high_value")
        original_model, original_report, _, _ = train_and_evaluate(self.dataset)
        changed = self.dataset.copy()
        held_out = changed["customer_unique_id"].isin(partitions["test"]["customer_unique_id"])
        changed.loc[held_out, FEATURE_COLUMNS] = 1000000
        changed_model, changed_report, _, _ = train_and_evaluate(changed)
        self.assertEqual(original_report["selected_model"], changed_report["selected_model"])
        self.assertEqual(original_report["validation"], changed_report["validation"])
        np.testing.assert_allclose(original_model.named_steps["scaler"].mean_, changed_model.named_steps["scaler"].mean_)
        expected = pd.concat([partitions["train"], partitions["validation"]])[FEATURE_COLUMNS].mean()
        np.testing.assert_allclose(original_model.named_steps["scaler"].mean_, expected)

    def test_repeat_purchase_target_and_seed_are_reproducible(self):
        first = train_and_evaluate(self.dataset, target="repeat_purchase")
        second = train_and_evaluate(self.dataset, target="repeat_purchase")
        self.assertEqual(first[1], second[1])
        assert_frame_equal(first[2], second[2])
        assert_frame_equal(first[3], second[3])


if __name__ == "__main__":
    unittest.main()
