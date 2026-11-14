"""Customer-isolated evaluation with validation-only model selection."""

import hashlib
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGETS


def split_customers(dataset, target, seed=42):
    if target not in TARGETS:
        raise ValueError(f"target 必须是 {TARGETS} 之一。")
    required = ["customer_unique_id", target] + FEATURE_COLUMNS
    if set(required) - set(dataset):
        raise ValueError("训练数据缺少客户标识、目标或特征。请先使用 build_dataset。")
    if dataset["customer_unique_id"].isna().any() or dataset["customer_unique_id"].duplicated().any():
        raise ValueError("每位 customer_unique_id 必须只有一条训练记录。")
    counts = dataset[target].value_counts()
    if set(counts.index) != {0, 1} or len(dataset) < 40 or counts.min() < 10:
        raise ValueError("训练至少需要 40 位客户，且正负两类各至少 10 位；请调整截止日、窗口或阈值。")
    numeric = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("训练特征不可包含无穷值。")
    train_val, test = train_test_split(dataset, test_size=0.2, random_state=seed, stratify=dataset[target])
    train, validation = train_test_split(
        train_val, test_size=0.25, random_state=seed, stratify=train_val[target]
    )
    return {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}


def _metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "positive_rate": float(np.mean(labels)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(dataset, target="high_value", seed=42):
    partitions = split_customers(dataset, target, seed)
    estimators = {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=1,
        ),
    }
    models, validation_scores = {}, {}
    train, validation, test = (partitions[k] for k in ("train", "validation", "test"))
    for name, estimator in estimators.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()), ("model", estimator),
        ])
        model.fit(train[FEATURE_COLUMNS], train[target])
        models[name] = model
        validation_scores[name] = _metrics(
            validation[target], model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        )
    selected = max(validation_scores, key=lambda name: validation_scores[name]["average_precision"])
    train_validation = pd.concat([train, validation])
    final_model = clone(models[selected]).fit(train_validation[FEATURE_COLUMNS], train_validation[target])
    probabilities = final_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = pd.DataFrame({
        "customer_unique_id": test["customer_unique_id"].to_numpy(),
        "actual": test[target].to_numpy(), "probability": probabilities,
        "prediction_at_0_5": (probabilities >= 0.5).astype(int),
    })
    assignments = pd.concat([
        rows[["customer_unique_id"]].assign(split=name) for name, rows in partitions.items()
    ], ignore_index=True)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(dataset.sort_values("customer_unique_id")[
            ["customer_unique_id", target] + FEATURE_COLUMNS
        ], index=False).to_numpy().tobytes()
    ).hexdigest()
    report = {
        "target": target, "seed": seed, "selected_model": selected,
        "selection_metric": "validation_average_precision", "classification_threshold": 0.5,
        "protocol": "customer-stratified 60/20/20; choose on validation; refit train+validation; evaluate test once",
        "dataset_sha256": fingerprint,
        "splits": {name: {"customers": len(rows), "positive_customers": int(rows[target].sum())}
                   for name, rows in partitions.items()},
        "validation": validation_scores, "test": _metrics(test[target], probabilities),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    return final_model, report, predictions, assignments


def feature_weights(model):
    """Diagnostic weights; coefficient magnitude is not a causal explanation."""
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values, kind = estimator.feature_importances_, "impurity_importance"
    elif hasattr(estimator, "coef_"):
        values, kind = estimator.coef_[0], "standardized_logistic_coefficient"
    else:
        values, kind = np.zeros(len(FEATURE_COLUMNS)), "baseline_has_no_learned_weights"
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "weight": values, "kind": kind})
