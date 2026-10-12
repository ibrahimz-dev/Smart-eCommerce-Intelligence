import time
import json
import re
import logging
from typing import Optional
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

from .base_agent import BaseScrapingAgent, ProductData

logger = logging.getLogger(__name__)


class ShopifyAgent(BaseScrapingAgent):
    def __init__(self, name: str = "ShopifyScraper", base_url: str = "",
                 use_storefront_api: bool = False, access_token: str = "",
                 request_delay: float = 0.3, shop_name: str = "", shop_country: str = "US"):
        super().__init__(name, "Shopify")
        self.base_url = base_url.rstrip("/")
        self.use_storefront_api = use_storefront_api
        self.access_token = access_token
        self.request_delay = request_delay
        self.shop_name = shop_name or self._domain_from_url(base_url)
        self.shop_country = shop_country
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })

    def _domain_from_url(self, url: str) -> str:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace("www.", "")

    def _extract_product_json(self, soup: BeautifulSoup) -> Optional[dict]:
        scripts = soup.find_all("script")
        for script in scripts:
            text = script.string or ""

            # Look for window.__INITIAL_STATE__ (common in modern Shopify)
            m = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.*?});', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    product_data = data.get("product") or data.get("products", {}).get("product")
                    if product_data:
                        return product_data
                except json.JSONDecodeError:
                    pass

            # Look for Shopify.Product or product JSON in any script
            for pattern in [
                r'var\s+meta\s*=\s*({.*?});',
                r'product:\s*({.*?})\s*\n',
                r'"product":\s*(\{.*?\})\s*\}',
            ]:
                m2 = re.search(pattern, text, re.DOTALL)
                if m2:
                    try:
                        data = json.loads(m2.group(1))
                        if "variants" in data or "title" in data:
                            return data
                    except json.JSONDecodeError:
                        pass

        return None

    def _find_json_ld(self, soup: BeautifulSoup) -> Optional[dict]:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "Product":
                    return data
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") == "Product":
                            return item
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    def _extract_price_html(self, soup: BeautifulSoup) -> float:
        selectors = [
            'meta[property="product:price:amount"]',
            'meta[itemprop="price"]',
            '[class*="price"]',
            '[data-price]',
            '[class*="Price"]',
            '.money',
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                text = el.get("content") or el.get("data-price") or el.get_text(strip=True)
                if text:
                    text = text.replace("$", "").replace(",", "").replace("€", "")
                    try:
                        return float(text)
                    except ValueError:
                        pass
        return 0.0

    def discover_collections(self, base_url: str = None) -> list[str]:
        if base_url is None:
            base_url = self.base_url
        try:
            resp = self.session.get(base_url, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            urls = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/collections/" in href and "/collections/all" not in href:
                    full = urljoin(base_url, href.split("?")[0])
                    urls.add(full.rstrip("/"))
            return sorted(urls)[:15]
        except Exception as e:
            self.logger.debug(f"Collection discovery failed: {e}")
            return []

    def _extract_from_listing(self, soup, max_products: int, category_hint: str = "") -> list[ProductData]:
        results = []
        seen_urls = set()

        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/products/" not in href:
                continue
            full_url = urljoin(self.base_url, href.split("?")[0])
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            card = a
            for _ in range(6):
                if card.name in ("div", "li", "article") and any(
                    c in str(card.get("class", [])).lower()
                    for c in ["product", "grid", "card", "item"]
                ):
                    break
                p = card.parent
                if p is None or p.name in ("body", "html"):
                    card = a
                    break
                card = p

            name = a.get("title", "").strip()
            if not name:
                for sel in [
                    '[class*="title"]', '[class*="name"]',
                    '[class*="product-card__title"]', "h2", "h3",
                ]:
                    el = card.select_one(sel)
                    if el:
                        name = el.get_text(strip=True)
                        break
            if not name:
                name = a.get_text(strip=True)[:200]

            price = 0.0
            for sel in [
                '[class*="price"]', '[class*="Price"]',
                ".money", '[data-price]',
                "span.price", '[class*="amount"]',
            ]:
                el = card.select_one(sel)
                if el:
                    raw = el.get("content", "") or el.get_text(strip=True)
                    m = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', raw)
                    if m:
                        try:
                            price = float(m.group(1).replace(",", ""))
                            break
                        except ValueError:
                            pass

            rating = 0.0
            rev_count = 0
            for el in card.select('[class*="rating"], [class*="Rating"], [class*="stars"], [class*="review"]'):
                txt = el.get_text(strip=True)
                m = re.search(r"([\d.]+)\s*/\s*5", txt)
                if m:
                    rating = float(m.group(1))
                m2 = re.search(r"(\d+)\s*review", txt, re.I)
                if m2:
                    rev_count = int(m2.group(1))

            image_urls = ""
            img = card.select_one("img[src]")
            if img:
                image_urls = img.get("src", "")

            brand = ""
            brand_el = card.select_one('[class*="vendor"], [class*="brand"]')
            if brand_el:
                brand = brand_el.get_text(strip=True)

            product = ProductData(
                name=name or "Unknown Product",
                price=price,
                rating=rating,
                num_reviews=rev_count,
                image_urls=image_urls,
                url=full_url,
                category=category_hint,
                brand=brand,
                shop_name=self.shop_name,
                shop_country=self.shop_country,
            )
            results.append(product)
            if len(results) >= max_products:
                break

        return results

    def _enrich_with_page_data(self, product: ProductData) -> Optional[ProductData]:
        """Visit individual product page to get description, stock, promo_price, etc."""
        try:
            resp = self.session.get(product.url, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            prod_json = self._extract_product_json(soup)
            if prod_json:
                enriched = self._json_to_product(prod_json, product.url)
                if enriched:
                    return enriched

            json_ld = self._find_json_ld(soup)
            if json_ld:
                enriched = self._json_to_product(json_ld, product.url)
                if enriched:
                    return enriched

            name = product.name
            h1 = soup.find("h1")
            if h1:
                name = h1.get_text(strip=True)
            meta_og = soup.find("meta", property="og:title")
            if not name and meta_og:
                name = meta_og.get("content", "")

            description = ""
            meta_desc = soup.find("meta", {"name": "description"})
            if meta_desc:
                description = meta_desc.get("content", "")

            price = product.price
            if price == 0:
                price = self._extract_price_html(soup)
            if price == 0:
                pm = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', resp.text)
                if pm:
                    try:
                        price = float(pm.group(1).replace(",", ""))
                    except ValueError:
                        pass

            product.name = name or product.name
            product.description = description
            product.price = price or product.price
            product.shop_name = self.shop_name
            product.shop_country = self.shop_country
            return self.clean_data(product)

        except Exception as e:
            self.logger.debug(f"Enrich failed for {product.url}: {e}")
            return product

    def _json_to_product(self, product_json: dict, url: str = "") -> Optional[ProductData]:
        try:
            variants = product_json.get("variants", [{}])
            first_variant = variants[0] if variants else {}

            price_str = first_variant.get("price", "0")
            if isinstance(price_str, (int, float)):
                price = float(price_str)
            else:
                price = float(price_str.replace(",", ""))

            compare_at = first_variant.get("compare_at_price")
            if compare_at:
                if isinstance(compare_at, str):
                    compare_at = float(compare_at.replace(",", ""))
                else:
                    compare_at = float(compare_at)
                promo_price = compare_at if compare_at > price else None
            else:
                promo_price = None

            discount = round((1 - price / compare_at) * 100) if promo_price and compare_at > 0 else 0

            images = product_json.get("images", [])
            image_urls = ", ".join(
                img.get("src") or img.get("original") or ""
                for img in images[:5]
            )

            return ProductData(
                product_id=str(product_json.get("id", "")),
                name=product_json.get("title", "") or product_json.get("name", ""),
                description=(product_json.get("body_html", "") or product_json.get("description", "")).strip(),
                category=product_json.get("product_type", "") or product_json.get("type", ""),
                brand=product_json.get("vendor", "") or "",
                price=price,
                promo_price=promo_price,
                discount_pct=discount,
                currency="USD",
                rating=float(product_json.get("rating", 0) if product_json.get("rating") else 0),
                num_reviews=int(product_json.get("review_count", 0) if product_json.get("review_count") else 0),
                stock=first_variant.get("inventory_quantity", 0) if first_variant.get("inventory_quantity") is not None else 0,
                in_stock=first_variant.get("available", True),
                image_urls=image_urls,
                tags=product_json.get("tags", ""),
                url=url,
                shop_name=self.shop_name,
                shop_country=self.shop_country,
            )
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None

    def scrape_product(self, url: str) -> Optional[ProductData]:
        try:
            resp = self.session.get(url, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            product = None

            product_json = self._extract_product_json(soup)
            if product_json:
                product = self._json_to_product(product_json, url)

            if not product or product.price == 0:
                json_ld = self._find_json_ld(soup)
                if json_ld:
                    product = self._json_to_product(json_ld, url)

            name = product.name if product and product.name else ""
            description = product.description if product and product.description else ""

            if not name:
                h1 = soup.find("h1")
                if h1:
                    name = h1.get_text(strip=True)
                meta_og = soup.find("meta", property="og:title")
                if not name and meta_og:
                    name = meta_og.get("content", "")

            if not description:
                meta_desc = soup.find("meta", {"name": "description"})
                if meta_desc:
                    description = meta_desc.get("content", "")

            price = product.price if product and product.price > 0 else 0
            if price == 0:
                price = self._extract_price_html(soup)
            if price == 0:
                price_m = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', resp.text)
                if price_m:
                    try:
                        price = float(price_m.group(1).replace(",", ""))
                    except ValueError:
                        pass

            promo = product.promo_price if product else None
            discount = product.discount_pct if product else 0
            stock = product.stock if product else 0

            product = ProductData(
                name=name,
                price=price,
                promo_price=promo,
                discount_pct=discount,
                rating=product.rating if product else 0,
                num_reviews=product.num_reviews if product else 0,
                stock=stock,
                description=description,
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
        consecutive_errors = 0
        category_hint = ""
        cm = re.search(r'/collections/([^/?]+)', category_url)
        if cm:
            category_hint = cm.group(1).replace("-", " ").replace("_", " ").title()
        while len(results) < max_products:
            url = f"{category_url}?page={page}" if page > 1 else category_url
            try:
                resp = self.session.get(url, timeout=8)
                if resp.status_code in (429, 403):
                    self.logger.warning(f"{url}: {resp.status_code} (rate limited/blocked)")
                    break
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                page_products = self._extract_from_listing(soup, max_products - len(results), category_hint)

                if not page_products:
                    # Fallback: find product links and try individual scraping
                    links = soup.select('a[href*="/products/"]')
                    urls = list(set(urljoin(self.base_url, a["href"]) for a in links
                                  if "/products/" in a.get("href", "")))
                    if not urls:
                        self.logger.debug(f"No product links on {url}")
                        break
                    urls_to_fetch = list(urls[:max_products - len(results)])
                    if not urls_to_fetch:
                        break
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        fetched = pool.map(lambda u: self.scrape_product(u), urls_to_fetch)
                    for prod in fetched:
                        if prod:
                            results.append(prod)
                else:
                    results.extend(page_products)

                page += 1
                consecutive_errors = 0
                if page > 20:
                    break
            except Exception as e:
                consecutive_errors += 1
                if "403" in str(e) or "429" in str(e):
                    self.logger.warning(f"Blocked on {url}")
                    break
                self.logger.error(f"Category page error {url}: {e}")
                if consecutive_errors >= 2:
                    break

        self.products.extend(results)
        return results

    def scrape_search(self, query: str, max_products: int = 20) -> list[ProductData]:
        results = []
        url = f"{self.base_url}/search?q={query}&type=product"
        try:
            resp = self.session.get(url, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select('a[href*="/products/"]')
            urls = list(set(urljoin(self.base_url, a["href"]) for a in links[:max_products]))

            with ThreadPoolExecutor(max_workers=3) as pool:
                fetched = pool.map(lambda u: self.scrape_product(u), urls)
            for prod in fetched:
                if prod:
                    results.append(prod)
        except Exception as e:
            logger.error(f"Search error: {e}")

        self.products.extend(results)
        return results
