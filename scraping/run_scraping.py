import os
import sys
import logging
import argparse
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scraping.agents.shopify_agent import ShopifyAgent
from scraping.agents.woocommerce_agent import WooCommerceAgent
from scraping.utils.html_parser import detect_platform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def domain_name(url):
    return urlparse(url).netloc.replace("www.", "")

def guess_country(url):
    domain = urlparse(url).netloc
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    country_map = {
        "uk": "UK", "ca": "CA", "de": "DE", "fr": "FR", "jp": "JP",
        "au": "AU", "it": "IT", "es": "ES", "nl": "NL", "br": "BR",
        "in": "IN", "cn": "CN", "se": "SE", "no": "NO", "dk": "DK",
    }
    return country_map.get(tld, "US")

# Import stores from json_api_scraper
try:
    from scraping.json_api_scraper import STORES as JSON_SHOPIFY_STORES
    SHOPIFY_STORES = [f"https://{s['domain']}" for s in JSON_SHOPIFY_STORES]
except ImportError:
    # Fallback to core stores if import fails
    SHOPIFY_STORES = [
        "https://www.brooklinen.com",
        "https://rothys.com",
        "https://www.olaplex.com",
        "https://www.liquiddeath.com",
        "https://www.chubbies.com",
        "https://www.mackweldon.com",
        "https://www.saxxunderwear.com",
        "https://www.pajamagram.com",
        "https://www.burrow.com",
        "https://www.kith.com",
    ]

WOOCOMMERCE_STORES = [
    "https://themes.woocommerce.com/storefront/",
    "https://wpfashiondemo.com",
    "https://www.8theme.com/demo/woocommerce/",
    "https://demo.woostify.com",
    "https://www.wpzoom.com/woocommerce-demo/",
]

COLLECTION_PATHS = ["/collections/all", "/collections"]


def scrape_store_with_fallback(agent, store_url, max_products, agent_name):
    all_prods = []
    seen = set()

    def add_unique(products):
        count = 0
        for p in products:
            if p.url not in seen:
                seen.add(p.url)
                all_prods.append(p)
                count += 1
        return count

    remaining = max_products

    entry_points = [store_url]
    for cp in COLLECTION_PATHS:
        entry_points.append(store_url.rstrip("/") + cp)

    for entry in entry_points:
        if remaining <= 0:
            break
        try:
            prods = agent.scrape_category(entry, max_products=remaining)
            new_c = add_unique(prods)
            if new_c > 0:
                logger.info(f"  [{agent_name}] +{new_c} from {entry.split('/')[-1]} "
                            f"(total: {len(all_prods)})")
            remaining -= new_c
        except Exception as e:
            logger.debug(f"  [{agent_name}] {entry}: {e}")

    if remaining > 0:
        try:
            collections = agent.discover_collections()
            for coll_url in collections:
                if remaining <= 0:
                    break
                try:
                    prods = agent.scrape_category(coll_url, max_products=remaining)
                    new_c = add_unique(prods)
                    if new_c > 0:
                        logger.info(f"  [{agent_name}] +{new_c} from discovered "
                                    f"collection (total: {len(all_prods)})")
                    remaining -= new_c
                except Exception:
                    pass
        except Exception:
            pass

    return all_prods


def run_scraping(mode: str = "shopify", output_dir: str = "data/scraped",
                 max_products_per_store: int = 500, target: int = 5000):
    os.makedirs(output_dir, exist_ok=True)
    all_products = []
    seen_urls = set()

    logger.info(f"Target: {target} products | Max per store: {max_products_per_store}")

    if mode in ("shopify", "auto"):
        stores = SHOPIFY_STORES
        for i, store_url in enumerate(stores, 1):
            if len(all_products) >= target:
                logger.info(f"Reached target of {target} products. Stopping.")
                break

            logger.info(f"\n[{i}/{len(stores)}] Shopify: {store_url}")
            sname = domain_name(store_url)
            scountry = guess_country(store_url)
            agent = ShopifyAgent(
                name=f"Shopify-{sname}",
                base_url=store_url,
                request_delay=1.0,
                shop_name=sname,
                shop_country=scountry,
            )

            try:
                products = scrape_store_with_fallback(
                    agent, store_url, max_products_per_store, agent.name
                )
            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                continue

            new_products = []
            for p in products:
                if p.url not in seen_urls:
                    seen_urls.add(p.url)
                    new_products.append(p)
            all_products.extend(new_products)

            logger.info(f"  >>> Running total: {len(all_products)} / {target}")

            if new_products:
                out = os.path.join(output_dir, f"shopify_{store_url.split('.')[1]}.csv")
                agent.save_results(out)

    if mode in ("woocommerce", "auto"):
        stores = WOOCOMMERCE_STORES
        for i, store_url in enumerate(stores, 1):
            if len(all_products) >= target:
                break

            logger.info(f"\n[{i}/{len(stores)}] WooCommerce: {store_url}")
            sname = domain_name(store_url)
            scountry = guess_country(store_url)
            agent = WooCommerceAgent(
                name=f"Woo-{sname}",
                base_url=store_url,
                request_delay=1.0,
                shop_name=sname,
                shop_country=scountry,
            )

            try:
                products = agent.scrape_category(store_url, max_products=max_products_per_store)
            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                continue

            new_products = []
            for p in products:
                if p.url not in seen_urls:
                    seen_urls.add(p.url)
                    new_products.append(p)
            all_products.extend(new_products)

            logger.info(f"  >>> Running total: {len(all_products)} / {target}")

    if all_products:
        df = pd.DataFrame([p.to_dict() for p in all_products])
        out = os.path.join(output_dir, "scraped_products.csv")
        df.to_csv(out, index=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"FINISHED: {len(df)} unique products saved to {out}")
        logger.info(f"{'='*60}")
        return df
    else:
        logger.warning("No products scraped from any store.")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Scraping Agent Runner - Scrapes real Shopify/WooCommerce stores")
    parser.add_argument("--mode", choices=["shopify", "woocommerce", "auto"],
                        default="shopify", help="Platform to scrape")
    parser.add_argument("--output", default="data/scraped", help="Output directory")
    parser.add_argument("--max", type=int, default=500,
                        help="Max products per store (default: 500)")
    parser.add_argument("--target", type=int, default=5000,
                        help="Target total products (default: 5000)")
    args = parser.parse_args()

    print("=" * 60)
    print("A2A SCRAPING AGENTS - HIGH-VOLUME SCRAPING")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Max per store: {args.max}")
    print(f"Target total: {args.target}")
    print(f"Shopify stores: {len(SHOPIFY_STORES)}")
    print(f"WooCommerce stores: {len(WOOCOMMERCE_STORES)}")
    print()

    start = time.time()
    df = run_scraping(mode=args.mode, output_dir=args.output,
                      max_products_per_store=args.max, target=args.target)
    elapsed = time.time() - start

    if not df.empty:
        print(f"\nScraped {len(df)} products in {elapsed/60:.1f} minutes!")
        print(df[["name", "price", "rating", "category"]].head(10).to_string())
    else:
        print("\nNo products scraped.")
        print("Tip: Some stores may block scrapers. Try adding stores to SHOPIFY_STORES list.")
