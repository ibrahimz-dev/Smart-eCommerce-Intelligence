#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Ensure the correct path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_data_quality(df):
    """Performs rigorous data quality checks on the scraped products."""
    print("=" * 60)
    print("📋 DATA QUALITY ASSESSMENT REPORT")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Missing Values
    print("\n🔍 Missing Value Counts:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} ({count/len(df)*100:.1f}%) missing")
    if missing.sum() == 0:
        print("  - Excellent! Zero missing values detected across all columns.")
        
    # Duplicate entries
    duplicates = df.duplicated(subset=["url"]).sum()
    print(f"\n🔍 Duplicate URLs: {duplicates} duplicate products")
    
    # Value Distributions & Anomalies
    print("\n🔍 Numerical Column Ranges:")
    for col in ["price", "rating", "num_reviews", "stock", "delivery_days"]:
        if col in df.columns:
            print(f"  - {col:13}: Min={df[col].min():6.2f} | Max={df[col].max():8.2f} | Mean={df[col].mean():6.2f}")
            
    # Subcategory / Category stats
    print("\n🔍 Category Cardinality:")
    print(f"  - Unique Categories: {df['category'].nunique() if 'category' in df.columns else 0}")
    print(f"  - Unique Brands:     {df['brand'].nunique() if 'brand' in df.columns else 0}")
    print(f"  - Unique Shop Names: {df['shop_name'].nunique() if 'shop_name' in df.columns else 0}")
    print("=" * 60)

def clean_data(df):
    """Cleans and standardizes columns for downstream ML pipelines."""
    cleaned = df.copy()
    
    # 1. Clean prices and format numeric columns
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce").fillna(0.0)
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce").fillna(0.0)
    cleaned["num_reviews"] = pd.to_numeric(cleaned["num_reviews"], errors="coerce").fillna(0).astype(int)
    cleaned["stock"] = pd.to_numeric(cleaned["stock"], errors="coerce").fillna(0).astype(int)
    cleaned["delivery_days"] = pd.to_numeric(cleaned["delivery_days"], errors="coerce").fillna(3).astype(int)
    cleaned["discount_pct"] = pd.to_numeric(cleaned["discount_pct"], errors="coerce").fillna(0).astype(int)
    
    # 2. Add old_price for ML & dashboard compatibility
    # promo_price in our scraper holds compare_at_price, which represents the original/old price
    if "promo_price" in cleaned.columns:
        cleaned["old_price"] = cleaned["promo_price"].fillna(0.0)
    else:
        # fallback: if no promo price, old price matches price
        cleaned["old_price"] = cleaned["price"]
        
    # Ensure all string values are stripped and filled
    for col in ["category", "brand", "shop_name", "shop_country", "tags"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()
            
    # Deduplicate by URL
    cleaned = cleaned.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return cleaned

def feature_engineering(df):
    """Applies pipeline's feature engineering formulas."""
    print("\n⚙️ Running Feature Engineering & Scoring...")
    engineered = df.copy()
    
    # Custom ML and dashboard enrichment features
    engineered["price_log"] = np.log1p(engineered["price"])
    engineered["rating_high"] = (engineered["rating"] >= 4.0).astype(int)
    engineered["has_discount"] = (engineered["discount_pct"] > 0).astype(int)
    engineered["low_stock"] = (engineered["stock"] < 10).astype(int)
    engineered["fast_delivery"] = (engineered["delivery_days"] <= 3).astype(int)
    engineered["reviews_log"] = np.log1p(engineered["num_reviews"])
    engineered["stock_log"] = np.log1p(engineered["stock"])
    engineered["engagement"] = engineered["rating"] * np.log1p(engineered["num_reviews"])
    engineered["price_drop"] = engineered["price"] / (engineered["old_price"] + 0.01)
    engineered["n_tags"] = engineered["tags"].apply(lambda t: len(t.split(",")) if t else 0)
    
    engineered["price_to_review"] = engineered["price"] / (engineered["num_reviews"] + 1)
    engineered["rating_weighted"] = engineered["rating"] * np.log1p(engineered["num_reviews"])
    engineered["stock_ratio"] = engineered["stock"] / (engineered["stock"].max() + 1)
    engineered["delivery_efficiency"] = 1 / (engineered["delivery_days"] + 1)
    engineered["discount_depth"] = engineered["discount_pct"] / 100
    engineered["premium_score"] = (engineered["price"] > engineered["price"].median()).astype(int)
    engineered["popularity_score"] = (
        (engineered["rating"] >= 4.0).astype(int) * 0.5
        + (engineered["num_reviews"] > engineered["num_reviews"].median()).astype(int) * 0.3
        + (engineered["stock"] > 0).astype(int) * 0.2
    )
    
    # Shop averages
    shop_stats = df.groupby("shop_name").agg(
        shop_avg_rating=("rating", "mean"),
        shop_avg_price=("price", "mean"),
        shop_total_products=("product_id", "count"),
    ).reset_index()
    engineered = engineered.merge(shop_stats, on="shop_name", how="left")
    engineered["shop_market_share"] = engineered["shop_total_products"] / len(df)
    
    # Country averages
    country_stats = df.groupby("shop_country").agg(
        country_avg_rating=("rating", "mean"),
        country_avg_price=("price", "mean"),
    ).reset_index()
    engineered = engineered.merge(country_stats, on="shop_country", how="left")
    
    # Composite Scoring & is_top labelling
    # Matches the get_top_k_products weighting: Rating 35%, Reviews 25%, Price 20%, Stock 20%
    score_rating = engineered["rating"] / 5.0
    score_reviews = np.log1p(engineered["num_reviews"]) / np.log1p(engineered["num_reviews"].max())
    score_price = 1 - (engineered["price"] - engineered["price"].min()) / (
        engineered["price"].max() - engineered["price"].min() + 1
    )
    score_stock = engineered["stock"] / (engineered["stock"].max() + 1e-5)
    
    engineered["composite_score"] = (
        0.35 * score_rating +
        0.25 * score_reviews +
        0.20 * score_price +
        0.20 * score_stock
    )
    # save score column
    engineered["score"] = np.round(engineered["composite_score"], 4)
    
    # Assign target labels (is_top & is_top_product)
    # Define top 10% of items as 'top products'
    top_threshold = engineered["composite_score"].quantile(0.90)
    engineered["is_top"] = (engineered["composite_score"] >= top_threshold).astype(int)
    engineered["is_top_product"] = engineered["is_top"]
    
    print(f"  - Features engineered: {len(engineered.columns)} columns total.")
    print(f"  - Labels assigned: {engineered['is_top'].sum()} products marked as Top Products (is_top=1).")
    
    return engineered

def apply_kmeans_clustering(df):
    """Clusters products using KMeans clustering and adds cluster ID to dataframe."""
    print("\n🤖 Running K-Means Clustering...")
    feature_cols = [
        "price_log", "rating", "reviews_log", "stock_log", 
        "delivery_days", "discount_pct", "engagement"
    ]
    
    X = df[feature_cols].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We cluster into 3 groups (e.g. Budget, Mid-range, Premium)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_transform(X_scaled).argmin(axis=1) # Consistent index
    
    print("  - K-Means completed successfully!")
    cluster_counts = df["cluster"].value_counts().to_dict()
    print(f"  - Cluster sizes: {cluster_counts}")
    
    return df

def train_eval_classifiers(df):
    """Trains and tests Random Forest and XGBoost classification models to verify training correctness."""
    print("\n🎯 Training Classification Models to Predict Top Products (is_top)...")
    
    # Feature columns for classification
    feature_cols = [
        "price", "rating", "num_reviews", "stock", "delivery_days",
        "discount_pct", "price_log", "reviews_log", "stock_log",
        "engagement", "has_discount", "low_stock", "n_tags"
    ]
    
    X = df[feature_cols].fillna(0).copy()
    y = df["is_top"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"  - [Random Forest] Accuracy: {rf_acc:.4f} | F1-Score: {rf_f1:.4f}")
    print(f"  - [XGBoost]       Accuracy: {xgb_acc:.4f} | F1-Score: {xgb_f1:.4f}")

def main():
    raw_path = "data/scraped/scraped_products.csv"
    analyzed_path = "data/scraped/analyzed_products.csv"
    
    if not os.path.exists(raw_path):
        print(f"❌ Error: Raw file '{raw_path}' does not exist.")
        sys.exit(1)
        
    # Load raw
    df = pd.read_csv(raw_path)
    
    # 1. Check Data Quality
    check_data_quality(df)
    
    # 2. Clean Data
    cleaned_df = clean_data(df)
    
    # 3. Feature Engineering & Label Generation
    engineered_df = feature_engineering(cleaned_df)
    
    # 4. Run K-Means Clustering
    clustered_df = apply_kmeans_clustering(engineered_df)
    
    # 5. Train & Evaluate ML classifiers to ensure high model fit
    train_eval_classifiers(clustered_df)
    
    # Save final enriched dataset
    os.makedirs(os.path.dirname(analyzed_path), exist_ok=True)
    clustered_df.to_csv(analyzed_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Processed dataset saved to: {analyzed_path}")
    print(f"   Final CSV contains {len(clustered_df)} products and {len(clustered_df.columns)} columns!")
    print("=" * 60)

if __name__ == "__main__":
    main()
