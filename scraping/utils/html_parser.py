import re
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup


def detect_platform(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if "shopify" in domain or "myshopify.com" in domain:
        return "shopify"
    if "woocommerce" in domain or "wp-json" in url:
        return "woocommerce"
    try:
        resp = __import__("requests").get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        if soup.find("meta", {"name": "generator", "content": re.compile(r"Shopify", re.I)}):
            return "shopify"
        if soup.find("meta", {"name": "generator", "content": re.compile(r"WooCommerce|Woo", re.I)}):
            return "woocommerce"
        if "/wp-content/" in resp.text:
            return "woocommerce"
    except Exception:
        pass
    return "unknown"


def extract_price(text: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9.,]", "", text)
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_rating(soup: BeautifulSoup) -> float:
    patterns = [
        r"rating[_-]?value['\"]?\s*[:=]\s*['\"]?([0-9.]+)",
        r"ratingValue['\"]?\s*:\s*([0-9.]+)",
        r"average[_-]?rating['\"]?\s*[:=]\s*['\"]?([0-9.]+)",
    ]
    for script in soup.find_all("script"):
        text = script.string or ""
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    return min(5.0, float(m.group(1)))
                except ValueError:
                    pass
    return 0.0


def extract_review_count(soup: BeautifulSoup) -> int:
    patterns = [
        r"reviewCount['\"]?\s*:\s*(\d+)",
        r"review[_-]?count['\"]?\s*[:=]\s*['\"]?(\d+)",
    ]
    for script in soup.find_all("script"):
        text = script.string or ""
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
    return 0


def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_product_links(soup: BeautifulSoup, base_domain: str) -> list[str]:
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        for keyword in ["/product/", "/products/", "/item/", "/p/"]:
            if keyword in href:
                if href.startswith("http"):
                    links.add(href)
                else:
                    links.add(f"https://{base_domain}{href}")
    return list(links)
