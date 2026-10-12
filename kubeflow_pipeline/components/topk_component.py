from kfp.dsl import component, InputPath, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy"],
)
def select_top_k_products(
    scored_data: InputPath("csv"),
    raw_data: InputPath("csv"),
    top_k_list: OutputPath("csv"),
    top_k_report: OutputPath("json"),
    k: int = 50,
):
    import pandas as pd
    import numpy as np
    import json

    scored_df = pd.read_csv(scored_data)
    raw_df = pd.read_csv(raw_data)

    top_k = scored_df.head(k)

    enriched = top_k.merge(
        raw_df[["product_id", "name", "description", "category", "brand",
                 "shop_name", "shop_country"]],
        on="product_id", how="left", suffixes=("", "_raw")
    )

    enriched.to_csv(top_k_list, index=False)

    report = {
        "k": k,
        "total_products": len(scored_df),
        "selection_pct": round(k / len(scored_df) * 100, 2),
        "avg_score": round(float(enriched["composite_score"].mean()), 4),
        "price_range": [float(enriched["price"].min()), float(enriched["price"].max())],
        "avg_rating": round(float(enriched["rating"].mean()), 2),
        "avg_reviews": int(round(enriched["num_reviews"].mean())),
        "category_distribution": enriched["category"].value_counts().to_dict(),
        "country_distribution": enriched["shop_country"].value_counts().to_dict(),
        "brand_distribution": enriched["brand"].value_counts().head(10).to_dict(),
    }

    with open(top_k_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Top-{k} products selected")
    print(f"Categories: {report['category_distribution']}")
    print(f"Countries: {report['country_distribution']}")
    print(f"\nTop-10 products:")
    print(enriched[["name", "price", "rating", "composite_score"]]
          .head(10).to_string())


@component(
    base_image="python:3.11",
    packages_to_install=["pandas"],
)
def generate_bi_report(
    top_k_data: InputPath("csv"),
    clustering_data: InputPath("csv"),
    bi_report: OutputPath("json"),
):
    import pandas as pd
    import json

    top_k = pd.read_csv(top_k_data)
    clusters = pd.read_csv(clustering_data) if clustering_data else pd.DataFrame()

    report = {
        "top_k_products": {
            "count": len(top_k),
            "avg_price": round(float(top_k["price"].mean()), 2),
            "avg_rating": round(float(top_k["rating"].mean()), 2),
            "avg_score": round(float(top_k["composite_score"].mean()), 4),
        },
        "categories": top_k["category"].value_counts().to_dict(),
        "shops": top_k["shop_name"].value_counts().head(10).to_dict(),
        "countries": top_k["shop_country"].value_counts().to_dict(),
        "clusters": clusters.to_dict("records") if not clusters.empty else [],
    }

    with open(bi_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"BI Report generated")
    print(json.dumps(report, indent=2))
