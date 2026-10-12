import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ml_analysis.preprocessing.data_loader import DataLoader
from ml_analysis.preprocessing.feature_engineering import (
    create_product_features, apply_pca, get_top_k_products
)
from ml_analysis.models.classification import ProductClassifier, train_all_classifiers
from ml_analysis.models.clustering import ProductClustering, find_optimal_k
from ml_analysis.models.association_rules import AssociationRuleMiner
from ml_analysis.models.pca_analysis import PCAAnalyzer
from ml_analysis.evaluation.supervised_eval import SupervisedEvaluator, compare_classifiers
from ml_analysis.evaluation.unsupervised_eval import (
    ClusteringEvaluator, evaluate_association_rules
)


def run_full_analysis(data_path: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SMART E-COMMERCE INTELLIGENCE - FULL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Data: {data_path}\n")

    # 1. Load data
    print("[1] Loading data...")
    loader = DataLoader(data_path)
    df = loader.load()
    stats = loader.get_basic_stats()
    print(f"    Products: {stats['n_products']}")
    print(f"    Categories: {stats['n_categories']}")
    print(f"    Avg price: ${stats['avg_price']}")

    # 2. Preprocess
    print("\n[2] Preprocessing...")
    (X_train, X_val, X_test, y_train, y_val, y_test, df_proc, feat_cols) = loader.preprocess()
    print(f"    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"    Features: {len(feat_cols)}")

    # 3. Top-K scoring
    print("\n[3] Top-K product scoring...")
    top50 = get_top_k_products(df, k=50)
    print(f"    Top 50 products identified")
    top50_path = os.path.join(output_dir, "top50_products.csv")
    top50.to_csv(top50_path, index=False)

    # 4. Classification (Supervised)
    if y_train is not None:
        print("\n[4] Classification models (Random Forest + XGBoost)...")
        classifiers = train_all_classifiers(X_train, y_train, X_val, y_val)
        comparison = compare_classifiers(classifiers, X_test, y_test)
        print(f"\n    Model Comparison:")
        print(comparison.to_string(index=False))
        comparison.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)

        for name, model in classifiers.items():
            eval = SupervisedEvaluator(model, X_test, y_test, name)
            metrics = eval.evaluate()
            eval.plot_confusion_matrix(os.path.join(output_dir, f"cm_{name}.png"))
            eval.plot_roc_curve(os.path.join(output_dir, f"roc_{name}.png"))
            print(f"    {name}: F1={metrics['f1_score']}, AUC={metrics.get('roc_auc', 'N/A')}")
    else:
        print("\n[4] Classification skipped (no target column)")

    # 5. Clustering (Unsupervised)
    print("\n[5] Clustering analysis...")
    X_for_clustering = X_train.copy()
    if isinstance(X_for_clustering, pd.DataFrame):
        X_for_clustering = X_for_clustering.values

    if X_for_clustering.shape[0] > 10:
        optimal_k = find_optimal_k(X_for_clustering, max_k=min(12, X_for_clustering.shape[0]-1))
        best_k = optimal_k["k"][np.argmax(optimal_k["silhouette"])]
        print(f"    Optimal K (silhouette): {best_k}")

        for algo_name, params in [
            ("kmeans", {"n_clusters": best_k}),
            ("hierarchical", {"n_clusters": best_k}),
            ("dbscan", {"eps": 1.5, "min_samples": 5}),
        ]:
            clusterer = ProductClustering(algorithm=algo_name, **params)
            clusterer.fit(X_for_clustering)
            evaluator = ClusteringEvaluator(clusterer.model, X_for_clustering,
                                             clusterer.labels_, algo_name)
            m = evaluator.evaluate()
            print(f"    {algo_name}: clusters={m['n_clusters']}, "
                  f"silhouette={m.get('silhouette_score', 'N/A')}")
    else:
        print("    Skipped (too few samples)")

    # 6. Association Rules
    print("\n[6] Association rules...")
    try:
        ar = AssociationRuleMiner(min_support=0.05, min_confidence=0.3, min_lift=1.2)
        ar.fit(df, group_col="shop_name", item_col="subcategory")
        summary = ar.get_summary()
        print(f"    Rules found: {summary['n_rules']}")
        if ar.rules is not None and not ar.rules.empty:
            readable = ar.rules_to_readable(n=10)
            print(readable.to_string(index=False))
            ar.rules.to_csv(os.path.join(output_dir, "association_rules.csv"), index=False)
    except Exception as e:
        print(f"    Skipped: {e}")

    # 7. PCA Visualization
    print("\n[7] PCA analysis...")
    pca = PCAAnalyzer(n_components=2)
    pca.fit(X_for_clustering, feature_names=feat_cols)
    summary_pca = pca.get_summary()
    print(f"    Explained variance: {summary_pca['explained_variance_ratio']}")
    print(f"    Cumulative: {summary_pca['cumulative_variance']}")

    df_pca = df_proc.iloc[:X_for_clustering.shape[0]].reset_index(drop=True)
    fig2d = pca.get_2d_projection(df_pca)
    fig2d.write_html(os.path.join(output_dir, "pca_2d.html"))

    # 8. Save results
    results = {
        "data_stats": stats,
        "classification": comparison.to_dict("records") if y_train is not None else [],
        "pca": summary_pca,
        "top_k_path": top50_path,
    }
    with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Results saved to:", output_dir)
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Analysis Pipeline")
    parser.add_argument("--data", required=True, help="Path to scraped product CSV")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run_full_analysis(args.data, args.output)
