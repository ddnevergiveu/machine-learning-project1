"""Compatibility import location for the train/validation/test workflow."""

from customer_ml.model import feature_weights, split_customers, train_and_evaluate

__all__ = ["feature_weights", "split_customers", "train_and_evaluate"]
