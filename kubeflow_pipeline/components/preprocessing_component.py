from kfp.dsl import component, InputPath, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def preprocess_data(
    data_path: InputPath("csv"),
    train_data: OutputPath("csv"),
    val_data: OutputPath("csv"),
    test_data: OutputPath("csv"),
    preprocessor_path: OutputPath("joblib"),
    target_col: str = "is_top_product",
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = pd.read_csv(data_path)

    df["price_log"] = np.log1p(df["price"])
    df["reviews_log"] = np.log1p(df["num_reviews"])
    df["stock_log"] = np.log1p(df["stock"])
    df["engagement"] = df["rating"] * np.log1p(df["num_reviews"])
    df["has_discount"] = (df["discount_pct"] > 0).astype(int)
    df["low_stock"] = (df["stock"] < 10).astype(int)

    feature_cols = [
        "price", "rating", "num_reviews", "stock", "delivery_days",
        "discount_pct", "price_log", "reviews_log", "stock_log",
        "engagement", "has_discount", "low_stock",
    ]

    cat_cols = ["category", "brand", "shop_name", "shop_country"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col + "_enc")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[target_col] = y_train
    train_df.to_csv(train_data, index=False)

    val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_df[target_col] = y_val
    val_df.to_csv(val_data, index=False)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[target_col] = y_test
    test_df.to_csv(test_data, index=False)

    preprocessor = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {feature_cols}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
