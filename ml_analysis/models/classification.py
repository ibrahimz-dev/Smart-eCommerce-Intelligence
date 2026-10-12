import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import joblib
import os


class ProductClassifier:
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build_model()
        self.feature_importance_ = None

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1, class_weight="balanced"
            )
        elif self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=self.random_state, eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_train, y_train)]
        if X_val is not None and self.model_type in ("xgboost", "lightgbm"):
            eval_set.append((X_val, y_val))

        if self.model_type in ("xgboost",):
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

        else:
            self.model.fit(X_train, y_train)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["classification_report"] = classification_report(y_test, y_pred,
                                                                  output_dict=True)
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


def train_all_classifiers(X_train, y_train, X_val, y_val) -> dict:
    results = {}
    for model_type in ["random_forest", "xgboost"]:
        clf = ProductClassifier(model_type=model_type)
        clf.train(X_train, y_train, X_val, y_val)
        results[model_type] = clf
    return results
