import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProductData:
    product_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    subcategory: str = ""
    brand: str = ""
    price: float = 0.0
    promo_price: Optional[float] = None
    discount_pct: float = 0.0
    currency: str = "USD"
    rating: float = 0.0
    num_reviews: int = 0
    stock: int = 0
    delivery_days: int = 0
    in_stock: bool = True
    colors: str = ""
    sizes: str = ""
    image_urls: str = ""
    shop_name: str = ""
    shop_country: str = ""
    shop_product_count: int = 0
    tags: str = ""
    url: str = ""
    scraped_at: str = ""

    def to_dict(self):
        return asdict(self)


class BaseScrapingAgent(ABC):
    def __init__(self, name: str, platform: str):
        self.name = name
        self.platform = platform
        self.products: list[ProductData] = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{name}]")

    @abstractmethod
    def scrape_product(self, url: str) -> Optional[ProductData]:
        pass

    @abstractmethod
    def scrape_category(self, category_url: str, max_products: int = 50) -> list[ProductData]:
        pass

    @abstractmethod
    def scrape_search(self, query: str, max_products: int = 20) -> list[ProductData]:
        pass

    def clean_data(self, product: ProductData) -> ProductData:
        if product.name:
            product.name = product.name.strip()
        if product.description:
            product.description = product.description.strip()[:500]
        if product.price <= 0:
            self.logger.warning(f"Invalid price for {product.name}: {product.price}")
        product.rating = min(5.0, max(0.0, product.rating))
        product.num_reviews = max(0, product.num_reviews)
        product.stock = max(0, product.stock)
        return product

    def save_results(self, filepath: str):
        import pandas as pd
        df = pd.DataFrame([p.to_dict() for p in self.products])
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(self.products)} products to {filepath}")

    def get_stats(self) -> dict:
        return {
            "agent": self.name,
            "platform": self.platform,
            "products_scraped": len(self.products),
            "categories_found": len(set(p.category for p in self.products)),
            "avg_rating": sum(p.rating for p in self.products) / max(len(self.products), 1),
            "avg_price": sum(p.price for p in self.products) / max(len(self.products), 1),
        }
