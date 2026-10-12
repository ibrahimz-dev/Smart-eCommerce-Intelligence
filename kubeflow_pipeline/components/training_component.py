from kfp.dsl import component, InputPath, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "joblib"],
)
def train_classification_models(
    train_data: InputPath("csv"),
    val_data: InputPath("csv"),
    test_data: InputPath("csv"),
    best_model_path: OutputPath("joblib"),
    metrics_path: OutputPath("json"),
    target_col: str = "is_top_product",
):
    import pandas as pd
    import numpy as np
    import json
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(val_data)
    test_df = pd.read_csv(test_data)

    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        ),
    }

    best_model = None
    best_f1 = 0
    all_metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }

        if hasattr(model, "feature_importances_"):
            metrics["feature_importances"] = model.feature_importances_.tolist()

        all_metrics[name] = metrics
        print(f"{name}: F1={metrics['f1_score']}, Acc={metrics['accuracy']}")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model

    joblib.dump(best_model, best_model_path)

    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nBest model feature importance:\n{feature_importance_df.head(10).to_string()}")

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def train_clustering_models(
    train_data: InputPath("csv"),
    clustering_model_path: OutputPath("joblib"),
    cluster_profiles_path: OutputPath("csv"),
    target_col: str = "is_top_product",
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(train_data)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = 5
    best_sil = -1
    for k in range(2, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k

    print(f"Optimal K: {best_k} (silhouette: {best_sil:.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    joblib.dump({"model": kmeans, "scaler": scaler}, clustering_model_path)

    profiles = []
    for i in range(best_k):
        mask = labels == i
        cluster_df = df[mask]
        profiles.append({
            "cluster": i,
            "count": int(mask.sum()),
            "pct": round(float(mask.sum() / len(df) * 100), 1),
            "avg_price": float(cluster_df.iloc[:, 0].mean()) if len(cluster_df) else 0,
        })

    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(cluster_profiles_path, index=False)
    print(f"Cluster profiles:\n{profiles_df.to_string()}")
