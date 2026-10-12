import pandas as pd
import logging
from typing import Optional

from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class ProductEnricher:
    def __init__(self):
        self.llm = LLMAgent()

    def summarize_product(self, name: str, description: str,
                          category: str = "", price: float = 0.0,
                          rating: float = 0.0, num_reviews: int = 0) -> str:
        return self.llm.summarize_product(
            name, description, category, price, rating, num_reviews
        )

    def enrich_product_tags(self, name: str, description: str,
                            category: str) -> list[str]:
        prompt = (
            f"Generate 5-10 relevant SEO tags/keywords for this e-commerce product:\n"
            f"Name: {name}\nDescription: {description[:300]}\nCategory: {category}\n\n"
            f"Return as a comma-separated list."
        )
        result = self.llm._call_llm(prompt)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        return tags[:10]

    def generate_marketing_copy(self, name: str, description: str,
                                price: float) -> str:
        prompt = (
            f"Write a compelling 2-3 sentence marketing description for:\n"
            f"Product: {name}\n"
            f"Current Description: {description[:300]}\n"
            f"Price: ${price}\n\n"
            f"Marketing Copy:"
        )
        return self.llm._call_llm(prompt)

    def extract_key_features(self, description: str) -> list[str]:
        prompt = (
            f"Extract the top 5 key product features from this description. "
            f"Return them as a bullet list:\n\n{description[:500]}"
        )
        result = self.llm._call_llm(prompt)
        features = [f.strip().lstrip("- ") for f in result.split("\n") if f.strip()]
        return features[:5]


class BatchEnricher:
    def __init__(self):
        self.enricher = ProductEnricher()

    def enrich_dataframe(self, df: pd.DataFrame,
                         batch_size: int = 10) -> pd.DataFrame:
        enriched = df.copy()
        summaries = []

        for i, row in df.head(batch_size).iterrows():
            summary = self.enricher.summarize_product(
                name=row.get("name", ""),
                description=row.get("description", ""),
                category=row.get("category", ""),
                price=row.get("price", 0),
                rating=row.get("rating", 0),
                num_reviews=row.get("num_reviews", 0),
            )
            summaries.append(summary)

        enriched["llm_summary"] = pd.Series(summaries, index=df.head(batch_size).index)
        enriched["llm_summary"] = enriched["llm_summary"].fillna("")

        logger.info(f"Enriched {len(summaries)} products with LLM summaries")
        return enriched
