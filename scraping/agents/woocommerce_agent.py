import time
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .base_agent import BaseScrapingAgent, ProductData

logger = logging.getLogger(__name__)


class WooCommerceAgent(BaseScrapingAgent):
    def __init__(self, name: str = "WooScraper", base_url: str = "",
                 consumer_key: str = "", consumer_secret: str = "",
                 request_delay: float = 0.3, shop_name: str = "", shop_country: str = "US"):
        super().__init__(name, "WooCommerce")
        self.base_url = base_url.rstrip("/")
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.request_delay = request_delay
        self.shop_name = shop_name or self._domain_from_url(base_url)
        self.shop_country = shop_country
        self.api_url = f"{self.base_url}/wp-json/wc/v3"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        })

    def _domain_from_url(self, url: str) -> str:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace("www.", "")

    def _get_via_api(self, endpoint: str, params: dict = None) -> dict:
        auth = (self.consumer_key, self.consumer_secret) if self.consumer_key else None
        try:
            resp = self.session.get(f"{self.api_url}/{endpoint}", params=params, auth=auth, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"API error on {endpoint}: {e}")
            return {}

    def scrape_product(self, url: str) -> Optional[ProductData]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            name = ""
            h1 = soup.find("h1", class_="product_title")
            if h1:
                name = h1.get_text(strip=True)

            price = 0.0
            price_el = soup.select_one('[class*="price"] .amount, .price')
            if price_el:
                text = price_el.get_text(strip=True).replace("$", "").replace(",", "")
                try:
                    price = float(text)
                except ValueError:
                    pass

            desc_div = soup.select_one(".woocommerce-product-details__short-description, #tab-description")
            description = desc_div.get_text(strip=True)[:500] if desc_div else ""

            rating_div = soup.select_one(".star-rating")
            rating = 0.0
            if rating_div:
                style = rating_div.get("style", "")
                try:
                    pct = float(style.replace("width:", "").replace("%", "").replace(";", "").strip())
                    rating = round(pct / 20, 1)
                except ValueError:
                    pass

            sku_el = soup.select_one(".sku")
            product_id = sku_el.get_text(strip=True) if sku_el else ""

            stock_el = soup.select_one('.in-stock, .out-of-stock')
            in_stock = stock_el and "out-of-stock" not in stock_el.get("class", [])

            categories = []
            cat_el = soup.select_one(".posted_in a")
            if cat_el:
                categories = [a.get_text(strip=True) for a in soup.select(".posted_in a")]

            product = ProductData(
                name=name,
                product_id=product_id,
                price=price,
                description=description,
                rating=rating,
                category=categories[0] if categories else "",
                in_stock=in_stock,
                url=url,
                shop_name=self.shop_name,
                shop_country=self.shop_country,
            )
            return self.clean_data(product)

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None

    def scrape_category(self, category_url: str, max_products: int = 50) -> list[ProductData]:
        results = []
        page = 1
        while len(results) < max_products:
            url = f"{category_url}/page/{page}" if page > 1 else category_url
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                links = soup.select('a[class*="product"], li.product a, .woocommerce-loop-product__link')

                product_urls = []
                for a in links:
                    href = a.get("href", "")
                    if href and "/product/" in href:
                        product_urls.append(href)

                if not product_urls:
                    break

                for pu in product_urls[:max_products - len(results)]:
                    prod = self.scrape_product(pu)
                    if prod:
                        results.append(prod)
                    time.sleep(self.request_delay)

                page += 1
                if page > 20:
                    break
            except Exception as e:
                logger.error(f"Category error {url}: {e}")
                break

        self.products.extend(results)
        return results

    def scrape_search(self, query: str, max_products: int = 20) -> list[ProductData]:
        results = []
        if self.consumer_key:
            data = self._get_via_api("products", {"search": query, "per_page": min(max_products, 100)})
            if isinstance(data, list):
                for item in data:
                    product = ProductData(
                        product_id=str(item.get("id", "")),
                        name=item.get("name", ""),
                        description=item.get("description", ""),
                        price=float(item.get("price", 0)),
                        category=item.get("categories", [{}])[0].get("name", "") if item.get("categories") else "",
                        brand=item.get("attributes", [{}])[0].get("name", "") if item.get("attributes") else "",
                        rating=float(item.get("average_rating", 0)),
                        num_reviews=int(item.get("rating_count", 0)),
                        stock=int(item.get("stock_quantity", 0)),
                        in_stock=item.get("stock_status") == "instock",
                        image_urls=item.get("images", [{}])[0].get("src", "") if item.get("images") else "",
                        shop_name=self.shop_name,
                        shop_country=self.shop_country,
                    )
                    results.append(product)
                    if len(results) >= max_products:
                        break
        else:
            url = f"{self.base_url}/?s={query}&post_type=product"
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                links = soup.select('a[href*="/product/"]')
                for a in links[:max_products]:
                    prod = self.scrape_product(a["href"])
                    if prod:
                        results.append(prod)
                    time.sleep(self.request_delay)
            except Exception as e:
                logger.error(f"Search error: {e}")

        self.products.extend(results)
        return results
