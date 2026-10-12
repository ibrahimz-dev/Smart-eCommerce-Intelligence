import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


class ClusteringEvaluator:
    def __init__(self, clustering_model, X: np.ndarray, labels: np.ndarray,
                 algorithm_name: str = "Clustering"):
        self.model = clustering_model
        self.X = X
        self.labels = labels
        self.algorithm_name = algorithm_name
        self.metrics = {}
        self.X_scaled = StandardScaler().fit_transform(X)

    def evaluate(self) -> dict:
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels_).count(-1) if hasattr(self, "labels_") else 0
        n_noise = list(self.labels).count(-1) if -1 in self.labels else 0

        self.metrics = {
            "algorithm": self.algorithm_name,
            "n_clusters": n_clusters,
            "n_samples": len(self.labels),
            "n_noise": n_noise,
        }

        mask = self.labels != -1
        if n_clusters > 1 and mask.sum() > 1:
            self.metrics["silhouette_score"] = round(
                silhouette_score(self.X_scaled[mask], self.labels[mask]), 4
            )
            sil_samples = silhouette_samples(self.X_scaled[mask], self.labels[mask])
            self.metrics["silhouette_samples_mean"] = round(np.mean(sil_samples), 4)
            self.metrics["silhouette_samples_std"] = round(np.std(sil_samples), 4)

        return self.metrics

    def plot_silhouette(self, save_path: str = None):
        mask = self.labels != -1
        if mask.sum() < 2:
            return

        sil_values = silhouette_samples(self.X_scaled[mask], self.labels[mask])
        y_lower = 10
        n_clusters = len(set(self.labels[mask]))

        plt.figure(figsize=(10, 6))
        for i in sorted(set(self.labels[mask])):
            cluster_sil = sil_values[self.labels[mask] == i]
            cluster_sil.sort()
            size = len(cluster_sil)
            y_upper = y_lower + size
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                              alpha=0.7, label=f"Cluster {i}")
            plt.text(-0.05, y_lower + 0.5 * size, str(i))
            y_lower = y_upper + 10

        avg_score = self.metrics.get("silhouette_score", 0)
        plt.axvline(x=avg_score, color="red", linestyle="--",
                    label=f"Avg = {avg_score:.4f}")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.title(f"Silhouette Plot - {self.algorithm_name} (n={n_clusters})")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def plot_clusters_2d(self, X_pca: np.ndarray, save_path: str = None):
        if X_pca.shape[1] < 2:
            return

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels,
                              cmap="viridis", alpha=0.7, s=50)
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Cluster Visualization (PCA) - {self.algorithm_name}")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()


def compare_clustering_algorithms(X: np.ndarray, algorithms: dict) -> pd.DataFrame:
    results = []
    for name, params in algorithms.items():
        algo = params.get("algorithm", "kmeans")
        kwargs = {k: v for k, v in params.items() if k != "algorithm"}

        from ml_analysis.models.clustering import ProductClustering
        clusterer = ProductClustering(algorithm=algo, **kwargs)
        clusterer.fit(X)

        evaluator = ClusteringEvaluator(clusterer.model, X, clusterer.labels_, name)
        metrics = evaluator.evaluate()
        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_association_rules(rules_df: pd.DataFrame) -> dict:
    if rules_df.empty:
        return {"n_rules": 0}

    return {
        "n_rules": len(rules_df),
        "support_range": (round(rules_df["support"].min(), 4),
                          round(rules_df["support"].max(), 4)),
        "confidence_range": (round(rules_df["confidence"].min(), 4),
                             round(rules_df["confidence"].max(), 4)),
        "lift_range": (round(rules_df["lift"].min(), 4),
                       round(rules_df["lift"].max(), 4)),
        "avg_support": round(rules_df["support"].mean(), 4),
        "avg_confidence": round(rules_df["confidence"].mean(), 4),
        "avg_lift": round(rules_df["lift"].mean(), 4),
        "rules_with_lift_gt_2": int((rules_df["lift"] > 2).sum()),
        "rules_with_lift_gt_5": int((rules_df["lift"] > 5).sum()),
    }
