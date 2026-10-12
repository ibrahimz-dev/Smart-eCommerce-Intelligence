import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_price(price_str: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.,]", "", price_str)
    if "," in cleaned and "." in cleaned:
        if cleaned.rindex(",") > cleaned.rindex("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def clean_product_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\s*\|.*$", "", name)
    name = re.sub(r"\s*[-–—]\s*.*$", "", name, count=1) if len(name) > 60 else name
    return name.strip()


def extract_brand_from_name(name: str, known_brands: list[str]) -> str:
    name_lower = name.lower()
    for brand in known_brands:
        if brand.lower() in name_lower:
            return brand
    return ""


def clean_description(desc: str, max_words: int = 100) -> str:
    desc = normalize_text(desc)
    words = desc.split()
    return " ".join(words[:max_words])


def parse_stock_status(text: str) -> tuple[bool, int]:
    text = text.lower()
    if "out of stock" in text or "rupture" in text or "épuisé" in text:
        return False, 0
    if "in stock" in text or "en stock" in text or "available" in text:
        numbers = re.findall(r"\d+", text)
        qty = int(numbers[0]) if numbers else 1
        return True, max(1, qty)
    return True, 1


def unify_category(category: str, mapping: dict[str, str] = None) -> str:
    if mapping is None:
        mapping = {
            "electronics": "Electronics",
            "clothing": "Fashion",
            "shoes": "Fashion",
            "home": "Home & Kitchen",
            "sports": "Sports & Outdoors",
            "books": "Books & Media",
            "beauty": "Beauty & Health",
            "toys": "Toys & Games",
            "automotive": "Automotive",
        }
    cat_lower = category.lower().strip()
    for key, val in mapping.items():
        if key in cat_lower:
            return val
    return category.title()
