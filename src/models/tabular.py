# src/models/tabular.py
#
# XGBoost tabular classifier with class-weight balancing.
# Wrapped in a sklearn-compatible class for use in pipelines.

import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.labels import LABEL_MAP


NUM_CLASSES = len(LABEL_MAP)


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights.
    Returns dict {class_id: weight} for use with sample_weight.
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {int(c): total / (NUM_CLASSES * cnt) for c, cnt in zip(classes, counts)}
    return weights


def build_xgb(
    n_estimators:  int   = 500,
    max_depth:     int   = 6,
    learning_rate: float = 0.05,
    random_state:  int   = 42,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
                random_state=random_state,
        n_jobs=-1,
    )


class TabularPipeline:
    """
    Wraps StandardScaler + XGBClassifier.
    Scaler is fitted on training data only and reused for val/test.
    """

    def __init__(
        self,
        n_estimators:  int   = 500,
        max_depth:     int   = 6,
        learning_rate: float = 0.05,
    ):
        self.scaler = StandardScaler()
        self.model  = build_xgb(n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_s = self.scaler.fit_transform(X_train)

        sample_weights = self._sample_weights(y_train)

        fit_kwargs = dict(sample_weight=sample_weights)
        if X_val is not None and y_val is not None:
            X_val_s = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_val_s, y_val)]
            fit_kwargs["verbose"]  = 50

        self.model.fit(X_train_s, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def _sample_weights(self, y):
        weights = compute_class_weights(y)
        return np.array([weights[yi] for yi in y])
