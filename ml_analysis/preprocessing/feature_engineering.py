import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["price_to_review"] = features["price"] / (features["num_reviews"] + 1)
    features["rating_weighted"] = features["rating"] * np.log1p(features["num_reviews"])
    features["stock_ratio"] = features["stock"] / (features["stock"].max() + 1)
    features["delivery_efficiency"] = 1 / (features["delivery_days"] + 1)
    features["discount_depth"] = features["discount_pct"] / 100
    features["premium_score"] = (features["price"] > features["price"].median()).astype(int)
    if "old_price" in features.columns:
        features["price_drop_ratio"] = (features["old_price"] - features["price"]) / (features["old_price"] + 0.01)
    if "tags" in features.columns:
        features["n_tags"] = features["tags"].str.split(",").str.len()
    if "listing_date" in features.columns:
        features["days_since_listed"] = (
            pd.Timestamp.now() - pd.to_datetime(features["listing_date"])
        ).dt.days
    features["popularity_score"] = (
        (features["rating"] >= 4.0).astype(int) * 0.5
        + (features["num_reviews"] > features["num_reviews"].median()).astype(int) * 0.3
        + (features["stock"] > 0).astype(int) * 0.2
    )

    shop_stats = df.groupby("shop_name").agg(
        shop_avg_rating=("rating", "mean"),
        shop_avg_price=("price", "mean"),
        shop_total_products=("product_id", "count"),
    ).reset_index()
    features = features.merge(shop_stats, on="shop_name", how="left")

    country_stats = df.groupby("shop_country").agg(
        country_avg_rating=("rating", "mean"),
        country_avg_price=("price", "mean"),
    ).reset_index()
    features = features.merge(country_stats, on="shop_country", how="left")

    features["shop_market_share"] = features["shop_total_products"] / len(df)

    return features


def apply_pca(X: np.ndarray, n_components: int = 2, feature_names: list = None) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    components_df = None
    if feature_names is not None:
        components_df = pd.DataFrame(
            pca.components_,
            columns=feature_names[:pca.n_components_],
            index=[f"PC{i+1}" for i in range(pca.n_components_)]
        )

    return X_pca, pca, components_df


def get_top_k_products(df: pd.DataFrame, k: int = 50,
                       weight_rating: float = 0.35,
                       weight_reviews: float = 0.25,
                       weight_price: float = 0.20,
                       weight_stock: float = 0.20) -> pd.DataFrame:
    scoring_df = df.copy()

    scoring_df["score_rating"] = scoring_df["rating"] / 5.0
    scoring_df["score_reviews"] = np.log1p(scoring_df["num_reviews"]) / np.log1p(scoring_df["num_reviews"].max())
    scoring_df["score_price"] = 1 - (scoring_df["price"] - scoring_df["price"].min()) / (
        scoring_df["price"].max() - scoring_df["price"].min() + 1
    )
    scoring_df["score_stock"] = scoring_df["stock"] / scoring_df["stock"].max()

    scoring_df["composite_score"] = (
        weight_rating * scoring_df["score_rating"]
        + weight_reviews * scoring_df["score_reviews"]
        + weight_price * scoring_df["score_price"]
        + weight_stock * scoring_df["score_stock"]
    )

    top_k = scoring_df.nlargest(k, "composite_score")
    return top_k.sort_values("composite_score", ascending=False)
