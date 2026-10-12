import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)


class SupervisedEvaluator:
    def __init__(self, model, X_test, y_test, model_name: str = "Model"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = None
        self.y_proba = None
        self.metrics = {}

    def evaluate(self) -> dict:
        self.y_pred = self.model.predict(self.X_test)
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]

        self.metrics = {
            "model": self.model_name,
            "accuracy": round(accuracy_score(self.y_test, self.y_pred), 4),
            "precision": round(precision_score(self.y_test, self.y_pred, zero_division=0), 4),
            "recall": round(recall_score(self.y_test, self.y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(self.y_test, self.y_pred, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred).tolist(),
        }

        if self.y_proba is not None:
            self.metrics["roc_auc"] = round(roc_auc_score(self.y_test, self.y_proba), 4)

        return self.metrics

    def plot_confusion_matrix(self, save_path: str = None):
        if not self.metrics or "confusion_matrix" not in self.metrics:
            self.evaluate()

        cm = np.array(self.metrics["confusion_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Top", "Top"],
                    yticklabels=["Non-Top", "Top"])
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(self, save_path: str = None):
        if self.y_proba is None:
            return

        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = self.metrics.get("roc_auc", 0)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.model_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def get_classification_report(self) -> str:
        return classification_report(self.y_test, self.y_pred)


def compare_classifiers(classifiers: dict, X_test, y_test) -> pd.DataFrame:
    results = []
    for name, model in classifiers.items():
        eval = SupervisedEvaluator(model, X_test, y_test, name)
        metrics = eval.evaluate()
        results.append(metrics)

    df = pd.DataFrame(results)
    display_cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_cols = [c for c in display_cols if c in df.columns]
    return df[display_cols].sort_values("f1_score", ascending=False)
