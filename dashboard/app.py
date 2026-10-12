import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Smart E-Commerce Intelligence Dashboard", layout="wide")

DATA_PATH = st.sidebar.text_input("Data file path", "data/scraped/analyzed_products_clean.csv")


@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_data(DATA_PATH)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Select Page", [
    " Overview",
    " Product Explorer",
    " ML Analysis",
    " Top-K Products",
    " Association Rules",
    " PCA Visualization",
    " LLM Dashboard",
])

if df.empty:
    st.warning(f"No data found at **{DATA_PATH}**")
    st.info("""
    **Run scraping first:**
    ```bash
    python3 scraping/run_scraping.py --mode shopify --output data/scraped
    ```
    Then refresh this page.
    """)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
categories = ["All"] + sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"]
selected_category = st.sidebar.selectbox("Category", categories)
countries = ["All"] + sorted(df["shop_country"].unique().tolist()) if "shop_country" in df.columns else ["All"]
selected_country = st.sidebar.selectbox("Country", countries)
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 0.0, 0.5)
max_price = st.sidebar.slider("Max Price ($)", 0.0, float(df["price"].max()) if "price" in df.columns else 1000.0, float(df["price"].max()) if "price" in df.columns else 1000.0, 10.0)

filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]
if selected_country != "All":
    filtered_df = filtered_df[filtered_df["shop_country"] == selected_country]
if "rating" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["rating"] >= min_rating]
if "price" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["price"] <= max_price]


if page == " Overview":
    st.header(" Smart E-Commerce Intelligence")
    st.caption("ML & Data Mining Pipeline | A2A Agents | LLM Enrichment")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", f"{len(filtered_df):,}")
    col2.metric("Categories", filtered_df["category"].nunique() if "category" in filtered_df.columns else 0)
    col3.metric("Countries", filtered_df["shop_country"].nunique() if "shop_country" in filtered_df.columns else 0)
    top_pct = (filtered_df["is_top"].sum() / len(filtered_df) * 100) if "is_top" in filtered_df.columns and len(filtered_df) > 0 else 0
    col4.metric("Top Products", f"{top_pct:.1f}%")

    col1, col2 = st.columns(2)
    if "category" in filtered_df.columns:
        with col1:
            st.subheader("Products by Category")
            cat_counts = filtered_df["category"].value_counts()
            import plotly.express as px
            fig = px.pie(values=cat_counts.values, names=cat_counts.index)
            st.plotly_chart(fig, use_container_width=True)
    if "rating" in filtered_df.columns and "category" in filtered_df.columns:
        with col2:
            st.subheader("Avg Rating by Category")
            avg_rating = filtered_df.groupby("category")["rating"].mean().sort_values()
            fig = px.bar(
                x=avg_rating.values.tolist(), 
                y=avg_rating.index.tolist(), 
                orientation="h",
                labels={'x': 'Avg Rating', 'y': 'Category'}
            )
            fig.update_layout(xaxis_title="Avg Rating", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

elif page == " Product Explorer":
    st.header(" Product Explorer")
    search = st.text_input("Search products")
    display_df = filtered_df.copy()
    if search:
        display_df = display_df[display_df["name"].str.contains(search, case=False, na=False)]
    cols = [c for c in ["product_id", "name", "category", "brand", "price", "rating",
                        "num_reviews", "stock", "shop_name", "shop_country"]
            if c in display_df.columns]
    st.dataframe(display_df[cols], use_container_width=True, height=500)

    if not display_df.empty:
        st.subheader("Product Details")
        selected = st.selectbox("Select a product", display_df["name"].tolist())
        prod = display_df[display_df["name"] == selected].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {prod['name']}")
            if "category" in prod: st.write(f"**Category:** {prod['category']}")
            if "brand" in prod: st.write(f"**Brand:** {prod['brand']}")
            if "price" in prod: st.write(f"**Price:** ${prod['price']:.2f}")
            if "old_price" in prod and prod['old_price'] > 0: st.write(f"**Old Price:** ${prod['old_price']:.2f}")
            if "promo_price" in prod and prod['promo_price'] > 0: st.write(f"**Promo:** ${prod['promo_price']:.2f}")
            if "tags" in prod: st.write(f"**Tags:** {prod['tags']}")
        with col2:
            if "rating" in prod: st.write(f"**Rating:** {prod['rating']}/5")
            if "num_reviews" in prod: st.write(f"**Reviews:** {prod['num_reviews']:,}")
            if "stock" in prod: st.write(f"**Stock:** {prod['stock']}")
            if "shop_name" in prod: st.write(f"**Shop:** {prod['shop_name']}")
            if "shop_country" in prod: st.write(f"**Country:** {prod['shop_country']}")
            if "listing_date" in prod: st.write(f"**Listed:** {prod['listing_date']}")

elif page == " ML Analysis":
    st.header(" Machine Learning Analysis")
    tab1, tab2 = st.tabs([" Classification Models", " Clustering Analysis"])
    with tab1:
        st.subheader("Top Product Prediction (Random Forest + XGBoost)")
        st.write("Target: `is_top`")
        if st.button("Run Classification"):
            with st.spinner("Training..."):
                from ml_analysis.preprocessing.data_loader import DataLoader
                from ml_analysis.models.classification import train_all_classifiers
                from ml_analysis.evaluation.supervised_eval import SupervisedEvaluator
                import plotly.express as px

                loader = DataLoader(DATA_PATH)
                X_train, X_val, X_test, y_train, y_val, y_test, _, _ = loader.preprocess(target_col="is_top")

                if y_train is not None:
                    classifiers = train_all_classifiers(X_train, y_train, X_val, y_val)
                    for name, model in classifiers.items():
                        eval = SupervisedEvaluator(model, X_test, y_test, name)
                        m = eval.evaluate()
                        st.write(f"**{name}** — Accuracy: {m['accuracy']:.4f}, F1: {m['f1_score']:.4f}, AUC: {m.get('roc_auc', 0):.4f}")
                        fig = px.imshow(m["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                                        x=["Non-Top", "Top"], y=["Non-Top", "Top"])
                        fig.update_layout(title=f"Confusion Matrix - {name}")
                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader(" ROC Curves")
                    col1, col2 = st.columns(2)
                    if Path("output/roc_random_forest.png").exists():
                        col1.image("output/roc_random_forest.png", caption="Random Forest ROC")
                    if Path("output/roc_xgboost.png").exists():
                        col2.image("output/roc_xgboost.png", caption="XGBoost ROC")

                    st.subheader(" Confusion Matrices (Saved)")
                    col3, col4 = st.columns(2)
                    if Path("output/cm_random_forest.png").exists():
                        col3.image("output/cm_random_forest.png", caption="Random Forest CM")
                    if Path("output/cm_xgboost.png").exists():
                        col4.image("output/cm_xgboost.png", caption="XGBoost CM")
                else:
                    st.error("Target column 'is_top' not found in data.")
    with tab2:
        st.subheader("Clustering (KMeans, Hierarchical, DBSCAN)")
        if st.button("Run Clustering"):
            with st.spinner("Clustering..."):
                from ml_analysis.preprocessing.data_loader import DataLoader
                from ml_analysis.models.clustering import ProductClustering

                loader = DataLoader(DATA_PATH)
                X_train, _, _, _, _, _, _, _ = loader.preprocess()
                from sklearn.preprocessing import StandardScaler
                X = StandardScaler().fit_transform(X_train)

                for algo, params in [("kmeans", {"n_clusters": 3}), ("hierarchical", {"n_clusters": 3}), ("dbscan", {"eps": 1.5, "min_samples": 5})]:
                    c = ProductClustering(algorithm=algo, **params)
                    c.fit(X)
                    from ml_analysis.evaluation.unsupervised_eval import ClusteringEvaluator
                    e = ClusteringEvaluator(c.model, X, c.labels_, algo)
                    m = e.evaluate()
                    st.write(f"**{algo}**: {m['n_clusters']} clusters, silhouette={m.get('silhouette_score', 'N/A')}")

elif page == " Top-K Products":
    st.header(" Top-K Product Selection")
    from ml_analysis.preprocessing.feature_engineering import get_top_k_products
    k = st.number_input("K", 5, 200, 50, 5)
    top_k_df = get_top_k_products(filtered_df, k=k)
    st.dataframe(top_k_df[["name", "category", "price", "rating", "num_reviews", "stock", "composite_score"]],
                 use_container_width=True, height=400)

elif page == " Association Rules":
    st.header(" Association Rules Mining")
    st.caption("Discover patterns: price range + rating level → category insights")

    # Feature engineering for meaningful rules
    df_rules = filtered_df.copy()

    # Create derived features
    df_rules["price_range"] = pd.cut(
        df_rules["price"],
        bins=[0, 50, 150, 500, float("inf")],
        labels=["Budget", "Mid-Range", "Premium", "Luxury"]
    ).astype(str)

    df_rules["rating_level"] = pd.cut(
        df_rules["rating"],
        bins=[0, 4.0, 4.5, 5.0],
        labels=["Good", "Very Good", "Excellent"]
    ).astype(str)

    df_rules["stock_level"] = pd.cut(
        df_rules["stock"],
        bins=[-1, 0, 50, 100, float("inf")],
        labels=["Out_of_Stock", "Low", "Medium", "High"]
    ).astype(str)

    df_rules["review_level"] = pd.cut(
        df_rules["num_reviews"],
        bins=[0, 100, 250, float("inf")],
        labels=["Few_Reviews", "Popular", "Trending"]
    ).astype(str)

    st.write("**Features used:** price_range + rating_level + stock_level + review_level + category")
    st.write(f"Products analyzed: {len(df_rules)}")

    if st.button("Mine Rules"):
        with st.spinner("Mining association rules..."):
            from mlxtend.preprocessing import TransactionEncoder
            import pandas as pd

            # Use derived features instead of trivial shop/brand
            transactions = df_rules[
                ["price_range", "rating_level", "stock_level", "review_level", "category"]
            ].replace("nan", None).dropna().astype(str).values.tolist()

            if len(transactions) < 10:
                st.error("Not enough data after filtering. Try reducing filters.")
            else:
                te = TransactionEncoder()
                te_ary = te.fit_transform(transactions)
                basket = pd.DataFrame(te_ary, columns=te.columns_)

                from mlxtend.frequent_patterns import apriori, association_rules
                itemsets = apriori(basket, min_support=0.03, use_colnames=True, max_len=3)

                if not itemsets.empty:
                    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
                    rules = rules[rules["confidence"] >= 0.15].sort_values("lift", ascending=False)

                    st.write(f"**Rules found: {len(rules)}**")
                    if len(rules) > 0:
                        display = rules.head(15).copy()
                        display["rule"] = display.apply(
                            lambda r: f"{{{' + '.join(list(r['antecedents']))}}} → {{{' + '.join(list(r['consequents']))}}}", axis=1
                        )
                        st.dataframe(
                            display[["rule", "support", "confidence", "lift"]].round(3),
                            use_container_width=True
                        )

                        # Business interpretation
                        if not rules.empty:
                            st.subheader(" Business Insights")
                            top_rule = rules.iloc[0]
                            ant = ', '.join(list(top_rule['antecedents']))
                            cons = ', '.join(list(top_rule['consequents']))
                            st.info(f"**Top insight:** Products with **{ant}** are {top_rule['lift']:.1f}x more likely to be **{cons}**")
                    else:
                        st.write("No significant rules found. Try lowering thresholds.")
                else:
                    st.write("No itemsets found. Try lowering min_support.")

elif page == " PCA Visualization":
    st.header(" Principal Component Analysis")
    if st.button("Compute PCA"):
        with st.spinner("Computing..."):
            from ml_analysis.models.pca_analysis import PCAAnalyzer
            from sklearn.preprocessing import StandardScaler
            num_cols = [c for c in ["price", "rating", "num_reviews", "stock", "delivery_days", "discount_pct"] if c in filtered_df.columns]
            if num_cols:
                X = StandardScaler().fit_transform(filtered_df[num_cols].fillna(0))
                pca = PCAAnalyzer(n_components=2)
                pca.fit(X, feature_names=num_cols)
                st.write(f"Explained variance: {pca.get_summary()['explained_variance_ratio']}")
                fig = pca.get_2d_projection(filtered_df, color_col="category" if "category" in filtered_df.columns else None)
                st.plotly_chart(fig, use_container_width=True)

elif page == " LLM Dashboard":
    st.header(" LLM-Powered Analysis")
    st.info("LLM configured with Groq (Llama 3). Ask questions or generate summaries.")
    from llm_module.llm_agent import LLMAgent
    agent = LLMAgent()
    if st.button("Generate Trend Summary"):
        from ml_analysis.preprocessing.feature_engineering import get_top_k_products
        top = get_top_k_products(filtered_df, k=10)
        summary = agent.generate_trend_summary(top)
        st.write(summary)
    st.subheader("Ask a Question")
    query = st.text_input("Your question about the products")
    if query:
        st.write(agent.answer_question(query, filtered_df))
