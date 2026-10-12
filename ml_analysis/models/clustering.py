import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os


class ProductClustering:
    def __init__(self, algorithm: str = "kmeans", **kwargs):
        self.algorithm = algorithm
        self.params = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None

    def _build_model(self, n_samples: int = None):
        if self.algorithm == "kmeans":
            n_clusters = self.params.get("n_clusters", 5)
            return KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init="auto",
                max_iter=300,
            )
        elif self.algorithm == "hierarchical":
            n_clusters = self.params.get("n_clusters", 5)
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.params.get("linkage", "ward"),
            )
        elif self.algorithm == "dbscan":
            eps = self.params.get("eps", 0.5)
            min_samples = self.params.get("min_samples", 5)
            return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(self, X: np.ndarray, sample_labels: list = None):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_model(n_samples=X.shape[0])
        self.labels_ = self.model.fit_predict(X_scaled)
        return self

    def get_cluster_profiles(self, df_original: pd.DataFrame) -> pd.DataFrame:
        if self.labels_ is None:
            return pd.DataFrame()

        profiledf = df_original.copy()
        profiledf["cluster"] = self.labels_

        profiles = profiledf.groupby("cluster").agg({
            "price": ["mean", "std", "min", "max"],
            "rating": "mean",
            "num_reviews": "mean",
            "stock": "mean",
            "discount_pct": "mean",
            "delivery_days": "mean",
        }).round(2)

        profiles["count"] = profiledf["cluster"].value_counts().sort_index()
        profiles["pct"] = (profiles["count"] / len(profiledf) * 100).round(1)
        return profiles

    def evaluate(self, X: np.ndarray) -> dict:
        X_scaled = self.scaler.transform(X)
        unique_labels = len(set(self.labels_))
        n_noise = list(self.labels_).count(-1) if -1 in self.labels_ else 0

        metrics = {
            "algorithm": self.algorithm,
            "n_clusters": unique_labels - (1 if -1 in self.labels_ else 0),
            "n_noise_points": n_noise,
            "noise_pct": round(n_noise / len(self.labels_) * 100, 2),
        }

        mask = self.labels_ != -1
        if unique_labels > 1 and mask.sum() > 1:
            metrics["silhouette_score"] = round(
                silhouette_score(X_scaled[mask], self.labels_[mask]), 4
            )
            metrics["davies_bouldin_score"] = round(
                davies_bouldin_score(X_scaled[mask], self.labels_[mask]), 4
            )
            metrics["calinski_harabasz_score"] = round(
                calinski_harabasz_score(X_scaled[mask], self.labels_[mask]), 4
            )

        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self


def find_optimal_k(X: np.ndarray, max_k: int = 15) -> dict:
    results = {"k": [], "inertia": [], "silhouette": [], "davies_bouldin": []}
    X_scaled = StandardScaler().fit_transform(X)

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)

        if len(set(labels)) > 1:
            results["silhouette"].append(silhouette_score(X_scaled, labels))
            results["davies_bouldin"].append(davies_bouldin_score(X_scaled, labels))
        else:
            results["silhouette"].append(-1)
            results["davies_bouldin"].append(999)

    return results
