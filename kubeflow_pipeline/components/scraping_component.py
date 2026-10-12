from kfp.dsl import component, InputPath, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "requests", "beautifulsoup4", "lxml"],
)
def scrape_products(
    store_urls: list[str],
    output_path: OutputPath("csv"),
):
    import pandas as pd
    import sys
    import os

    sys.path.append("/app")
    from scraping.agents.shopify_agent import ShopifyAgent
    from scraping.agents.woocommerce_agent import WooCommerceAgent

    all_products = []
    for url in store_urls:
        try:
            agent = ShopifyAgent(base_url=url)
            products = agent.scrape_category(url, max_products=50)
            all_products.extend(products)
        except Exception as e:
            print(f"Failed {url}: {e}")

    df = pd.DataFrame([p.to_dict() for p in all_products])
    df.to_csv(output_path, index=False)
    print(f"Scraped {len(df)} products from {len(store_urls)} stores")


@component(
    base_image="python:3.11",
    packages_to_install=["pandas"],
)
def validate_scraped_data(
    data_path: InputPath("csv"),
    validation_report: OutputPath("json"),
):
    import pandas as pd
    import json

    df = pd.read_csv(data_path)

    report = {
        "n_products": int(len(df)),
        "n_columns": int(len(df.columns)),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "is_valid": int(df["price"].isna().sum()) == 0,
    }

    with open(validation_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation: {report['n_products']} products, {report['n_columns']} columns")
