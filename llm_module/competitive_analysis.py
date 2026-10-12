import pandas as pd
import logging
from typing import Optional

from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class CompetitiveAnalyzer:
    def __init__(self):
        self.llm = LLMAgent()

    def generate_category_report(self, df: pd.DataFrame,
                                  category: str = None) -> str:
        if category:
            data = df[df["category"] == category]
        else:
            data = df

        if data.empty:
            return "No data available for analysis."

        return self.llm.generate_competitive_report(df, category or "all")

    def compare_products(self, products_df: pd.DataFrame) -> str:
        prompt_parts = ["Compare the following products and provide a competitive analysis:\n"]
        for i, (_, row) in enumerate(products_df.iterrows(), 1):
            prompt_parts.append(
                f"{i}. {row.get('name', 'Unknown')} | "
                f"Price: ${row.get('price', 0):.2f} | "
                f"Rating: {row.get('rating', 0)}/5 | "
                f"Reviews: {row.get('num_reviews', 0)} | "
                f"Stock: {row.get('stock', 0)}"
            )

        prompt_parts.append("\nAnalysis of competitive positioning, value proposition, and recommendations:")
        return self.llm._call_llm("\n".join(prompt_parts))

    def detect_emerging_trends(self, df: pd.DataFrame,
                                top_n: int = 5) -> str:
        top_products = df.nlargest(top_n * 5, "num_reviews").head(top_n)

        prompt = (
            f"Based on the following top-reviewed products, identify 3-5 emerging "
            f"market trends:\n\n"
        )
        for _, row in top_products.iterrows():
            prompt += f"- {row['name']} ({row['category']}, ${row['price']:.2f}, {row['rating']}/5, {row['num_reviews']} reviews)\n"

        prompt += "\nEmerging Trends:"
        return self.llm._call_llm(prompt)

    def pricing_strategy_analysis(self, df: pd.DataFrame) -> str:
        avg_price = df["price"].mean()
        median_price = df["price"].median()
        avg_discount = df[df["discount_pct"] > 0]["discount_pct"].mean()

        prompt = (
            f"Analyze the following pricing data and provide strategic recommendations:\n\n"
            f"Total Products: {len(df)}\n"
            f"Average Price: ${avg_price:.2f}\n"
            f"Median Price: ${median_price:.2f}\n"
            f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}\n"
            f"Products with Discount: {(df['discount_pct'] > 0).sum()}\n"
            f"Average Discount: {avg_discount:.1f}%\n"
            f"Categories: {df['category'].nunique()}\n\n"
            f"Pricing Strategy Analysis and Recommendations:"
        )
        return self.llm._call_llm(prompt)

    def weekly_trend_report(self, df: pd.DataFrame) -> str:
        rising = df.nlargest(20, "num_reviews")
        top_rated = df.nlargest(20, "rating")

        prompt = (
            "Generate a weekly market trend report:\n\n"
            "Most Reviewed Products:\n"
            f"{rising[['name', 'category', 'price', 'rating', 'num_reviews']].head(10).to_string()}\n\n"
            "Highest Rated Products:\n"
            f"{top_rated[['name', 'category', 'price', 'rating', 'num_reviews']].head(10).to_string()}\n\n"
            "Report Format:\n"
            "1. Top Performers This Week\n"
            "2. Category Trends\n"
            "3. Pricing Insights\n"
            "4. Recommendations for Next Week"
        )
        return self.llm._call_llm(prompt)
