from kfp.dsl import component, InputPath, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def score_products(
    data_path: InputPath("csv"),
    model_path: InputPath("joblib"),
    scored_products_path: OutputPath("csv"),
    target_col: str = "is_top_product",
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(data_path)
    model_data = joblib.load(model_path)
    model = model_data.get("model") if isinstance(model_data, dict) else model_data

    feature_cols = [c for c in df.columns if c not in ["product_id", "name", target_col]]
    X = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_scaled)[:, 1]
    else:
        scores = model.predict(X_scaled)

    df["ml_score"] = np.round(scores, 4)
    df["ml_score_norm"] = np.round(
        (scores - scores.min()) / (scores.max() - scores.min() + 1e-8), 4
    )

    df = df.sort_values("ml_score", ascending=False)
    df.to_csv(scored_products_path, index=False)

    print(f"Scored {len(df)} products")
    print(f"Score range: [{df['ml_score'].min():.4f}, {df['ml_score'].max():.4f}]")
    print(f"\nTop-10 products:")
    print(df[["name", "price", "rating", "ml_score"]].head(10).to_string())


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy"],
)
def calculate_product_score(
    data_path: InputPath("csv"),
    scored_path: OutputPath("csv"),
    top_k: int = 50,
):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(data_path)

    df["score_rating"] = df["rating"] / 5.0
    df["score_reviews"] = np.log1p(df["num_reviews"]) / np.log1p(df["num_reviews"].max())
    df["score_price"] = 1 - (df["price"] - df["price"].min()) / (
        df["price"].max() - df["price"].min() + 1
    )
    df["score_stock"] = df["stock"] / df["stock"].max()

    df["composite_score"] = (
        df["score_rating"] * 0.35
        + df["score_reviews"] * 0.25
        + df["score_price"] * 0.20
        + df["score_stock"] * 0.20
    )

    df = df.sort_values("composite_score", ascending=False)
    df.to_csv(scored_path, index=False)

    print(f"Scored {len(df)} products with composite metric")
    print(f"\nTop-{top_k} products:")
    print(df[["name", "price", "rating", "num_reviews", "composite_score"]]
          .head(top_k).to_string())
