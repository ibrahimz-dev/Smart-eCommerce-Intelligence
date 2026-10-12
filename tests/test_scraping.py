import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scraping.agents.base_agent import ProductData, BaseScrapingAgent


class TestProductData(unittest.TestCase):
    def test_product_data_creation(self):
        p = ProductData(
            product_id="TEST-001",
            name="Test Product",
            price=49.99,
            rating=4.5,
        )
        self.assertEqual(p.product_id, "TEST-001")
        self.assertEqual(p.name, "Test Product")
        self.assertEqual(p.price, 49.99)
        self.assertEqual(p.rating, 4.5)

    def test_product_data_defaults(self):
        p = ProductData()
        self.assertEqual(p.price, 0.0)
        self.assertEqual(p.rating, 0.0)
        self.assertEqual(p.in_stock, True)
        self.assertEqual(p.currency, "USD")

    def test_to_dict(self):
        p = ProductData(product_id="P1", name="Prod", price=10.0)
        d = p.to_dict()
        self.assertEqual(d["product_id"], "P1")
        self.assertEqual(d["name"], "Prod")
        self.assertEqual(d["price"], 10.0)


class FakeAgent(BaseScrapingAgent):
    def scrape_product(self, url):
        return ProductData(name=f"Product from {url}", price=25.0)

    def scrape_category(self, url, max_products=50):
        return [self.scrape_product(url) for _ in range(min(3, max_products))]

    def scrape_search(self, query, max_products=20):
        return [ProductData(name=f"Search: {query}", price=15.0)]


class TestBaseScrapingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent("TestAgent", "TestPlatform")

    def test_agent_initialization(self):
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.platform, "TestPlatform")

    def test_clean_data(self):
        p = ProductData(name="  Test  ", price=-5, rating=6.0, num_reviews=-10, stock=-1)
        cleaned = self.agent.clean_data(p)
        self.assertEqual(cleaned.name, "Test")
        self.assertEqual(cleaned.price, -5)
        self.assertEqual(cleaned.rating, 5.0)
        self.assertEqual(cleaned.num_reviews, 0)
        self.assertEqual(cleaned.stock, 0)

    def test_scrape_product(self):
        p = self.agent.scrape_product("http://example.com/prod")
        self.assertEqual(p.name, "Product from http://example.com/prod")
        self.assertEqual(p.price, 25.0)

    def test_get_stats(self):
        results = self.agent.scrape_category("http://example.com/cat")
        self.agent.products.extend(results)
        stats = self.agent.get_stats()
        self.assertEqual(stats["products_scraped"], 3)
        self.assertAlmostEqual(stats["avg_price"], 25.0)


if __name__ == "__main__":
    unittest.main()
