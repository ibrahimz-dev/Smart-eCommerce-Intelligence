import sys
import os
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np


def make_test_csv(n=100):
    data = {
        "product_id": [f"P{i:05d}" for i in range(n)],
        "name": [f"Product {i}" for i in range(n)],
        "category": np.random.choice(["Electronics", "Fashion", "Home"], n),
        "subcategory": np.random.choice(["A", "B", "C"], n),
        "brand": np.random.choice(["Brand1", "Brand2"], n),
        "price": np.random.uniform(10, 200, n),
        "rating": np.random.uniform(1, 5, n).round(1),
        "num_reviews": np.random.randint(0, 5000, n),
        "stock": np.random.randint(0, 500, n),
        "delivery_days": np.random.randint(1, 14, n),
        "discount_pct": np.random.choice([0, 10, 20, 30], n),
        "shop_name": np.random.choice(["Shop1", "Shop2"], n),
        "shop_country": np.random.choice(["USA", "UK"], n),
        "is_top_product": np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    path = os.path.join(tempfile.gettempdir(), "test_products.csv")
    df.to_csv(path, index=False)
    return path


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        from ml_analysis.preprocessing.data_loader import DataLoader
        self.csv_path = make_test_csv()
        self.loader = DataLoader(self.csv_path)

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_load(self):
        df = self.loader.load()
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn("product_id", df.columns)

    def test_preprocess(self):
        result = self.loader.preprocess()
        X_train, _, _, y_train, _, _, _, feat_cols = result
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertGreater(len(feat_cols), 0)

    def test_get_basic_stats(self):
        stats = self.loader.get_basic_stats()
        self.assertIn("n_products", stats)
        self.assertIn("avg_price", stats)

    def test_no_file_raises_error(self):
        from ml_analysis.preprocessing.data_loader import DataLoader
        with self.assertRaises(FileNotFoundError):
            DataLoader("nonexistent.csv")


class TestTopKSelection(unittest.TestCase):
    def setUp(self):
        from ml_analysis.preprocessing.data_loader import DataLoader
        self.csv_path = make_test_csv()
        self.loader = DataLoader(self.csv_path)
        self.df = self.loader.load()

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_get_top_k(self):
        from ml_analysis.preprocessing.feature_engineering import get_top_k_products
        top_k = get_top_k_products(self.df, k=10)
        self.assertEqual(len(top_k), 10)
        self.assertIn("composite_score", top_k.columns)


class TestClassification(unittest.TestCase):
    def setUp(self):
        from ml_analysis.preprocessing.data_loader import DataLoader
        self.csv_path = make_test_csv()
        loader = DataLoader(self.csv_path)
        self.X_train, _, self.X_test, self.y_train, _, self.y_test, _, _ = loader.preprocess()
        if isinstance(self.X_train, pd.DataFrame):
            self.X_train = self.X_train.values
            self.X_test = self.X_test.values

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_random_forest(self):
        from ml_analysis.models.classification import ProductClassifier
        clf = ProductClassifier(model_type="random_forest")
        clf.train(self.X_train, self.y_train)
        metrics = clf.evaluate(self.X_test, self.y_test)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_xgboost(self):
        from ml_analysis.models.classification import ProductClassifier
        clf = ProductClassifier(model_type="xgboost")
        clf.train(self.X_train, self.y_train)
        metrics = clf.evaluate(self.X_test, self.y_test)
        self.assertIn("f1_score", metrics)

    def test_all_classifiers(self):
        from ml_analysis.models.classification import train_all_classifiers
        from ml_analysis.evaluation.supervised_eval import compare_classifiers
        classifiers = train_all_classifiers(self.X_train, self.y_train, self.X_test, self.y_test)
        comparison = compare_classifiers(classifiers, self.X_test, self.y_test)
        self.assertEqual(len(comparison), 2)


class TestClustering(unittest.TestCase):
    def setUp(self):
        from ml_analysis.preprocessing.data_loader import DataLoader
        self.csv_path = make_test_csv()
        loader = DataLoader(self.csv_path)
        X_train, _, _, _, _, _, _, _ = loader.preprocess()
        from sklearn.preprocessing import StandardScaler
        self.X = StandardScaler().fit_transform(X_train)[:50]

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_kmeans(self):
        from ml_analysis.models.clustering import ProductClustering
        clusterer = ProductClustering(algorithm="kmeans", n_clusters=3)
        clusterer.fit(self.X)
        self.assertIsNotNone(clusterer.labels_)
        self.assertEqual(len(clusterer.labels_), len(self.X))

    def test_evaluate(self):
        from ml_analysis.models.clustering import ProductClustering
        clusterer = ProductClustering(algorithm="kmeans", n_clusters=3)
        clusterer.fit(self.X)
        metrics = clusterer.evaluate(self.X)
        self.assertIn("silhouette_score", metrics)


class TestAssociationRules(unittest.TestCase):
    def test_apriori(self):
        from ml_analysis.models.association_rules import AssociationRuleMiner
        df = pd.DataFrame({"shop_name": ["S1", "S1", "S1", "S2", "S2", "S3"], "subcategory": ["A", "B", "C", "A", "B", "C"]})
        ar = AssociationRuleMiner(min_support=0.1)
        ar.fit(df, group_col="shop_name", item_col="subcategory")
        self.assertIn("n_rules", ar.get_summary())


class TestPCAAnalysis(unittest.TestCase):
    def test_pca(self):
        from ml_analysis.models.pca_analysis import PCAAnalyzer
        X = np.random.randn(50, 5)
        pca = PCAAnalyzer(n_components=2)
        pca.fit(X)
        summary = pca.get_summary()
        self.assertEqual(len(summary["explained_variance_ratio"]), 2)


class TestSupervisedEval(unittest.TestCase):
    def test_evaluator(self):
        from ml_analysis.evaluation.supervised_eval import SupervisedEvaluator
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X[:80], y[:80])
        eval = SupervisedEvaluator(model, X[80:], y[80:], "TestRF")
        metrics = eval.evaluate()
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)


if __name__ == "__main__":
    unittest.main()
