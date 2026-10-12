import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.processed_df: pd.DataFrame = None
        self.label_encoders: dict = {}
        self.scaler: StandardScaler = None

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"No data file found at {self.data_path}.\n"
                "Please run scraping first:\n"
                "  python3 scraping/run_scraping.py --mode shopify --output data/scraped"
            )

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        return self.df

    def get_basic_stats(self) -> dict:
        if self.df is None:
            self.load()
        return {
            "n_products": len(self.df),
            "n_categories": self.df["category"].nunique() if "category" in self.df.columns else 0,
            "n_brands": self.df["brand"].nunique() if "brand" in self.df.columns else 0,
            "n_shops": self.df["shop_name"].nunique() if "shop_name" in self.df.columns else 0,
            "n_countries": self.df["shop_country"].nunique() if "shop_country" in self.df.columns else 0,
            "avg_price": round(self.df["price"].mean(), 2) if "price" in self.df.columns else 0,
            "avg_rating": round(self.df["rating"].mean(), 2) if "rating" in self.df.columns else 0,
            "avg_reviews": round(self.df["num_reviews"].mean(), 1) if "num_reviews" in self.df.columns else 0,
            "categories": self.df["category"].value_counts().to_dict() if "category" in self.df.columns else {},
            "countries": self.df["shop_country"].value_counts().to_dict() if "shop_country" in self.df.columns else {},
        }

    def preprocess(self, target_col: str = "is_top_product",
                   encode_categorical: bool = True,
                   scale_features: bool = True) -> tuple:
        if self.df is None:
            self.load()

        df = self.df.copy()

        if "price" in df.columns:
            df["price_log"] = np.log1p(df["price"])
        if "rating" in df.columns:
            df["rating_high"] = (df["rating"] >= 4.0).astype(int)
        if "discount_pct" in df.columns:
            df["has_discount"] = (df["discount_pct"] > 0).astype(int)
        if "stock" in df.columns:
            df["low_stock"] = (df["stock"] < 10).astype(int)
        if "delivery_days" in df.columns:
            df["fast_delivery"] = (df["delivery_days"] <= 3).astype(int)
        if "num_reviews" in df.columns:
            df["reviews_log"] = np.log1p(df["num_reviews"])
        if "stock" in df.columns:
            df["stock_log"] = np.log1p(df["stock"])
        if "rating" in df.columns and "num_reviews" in df.columns:
            df["engagement"] = df["rating"] * np.log1p(df["num_reviews"])
        if "price" in df.columns and "old_price" in df.columns:
            df["price_drop"] = df["price"] / (df["old_price"] + 0.01)
        if "tags" in df.columns:
            df["tags"] = df["tags"].fillna("").astype(str)
            df["n_tags"] = df["tags"].str.split(",").str.len()

        feature_cols = []
        for col in ["price", "rating", "num_reviews", "stock", "delivery_days",
                     "discount_pct", "price_log", "reviews_log", "stock_log",
                     "engagement", "price_drop", "n_tags", "has_discount", "low_stock"]:
            if col in df.columns:
                feature_cols.append(col)

        cat_cols = [c for c in ["category", "brand", "shop_name", "shop_country"] if c in df.columns]
        if encode_categorical:
            for col in cat_cols:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_cols.append(col + "_enc")

        X = df[feature_cols].copy()
        X = X.fillna(X.median())

        if scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_cols)

        y = df[target_col].values if target_col in df.columns else None

        if y is not None:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            X_train = X_val = X_test = X
            y_train = y_val = y_test = None

        self.processed_df = df
        return (X_train, X_val, X_test, y_train, y_val, y_test,
                df, feature_cols)
