import os
import json
import logging
from typing import Optional

from .config import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_MODEL, ANTHROPIC_MODEL, TEMPERATURE, MAX_TOKENS,
    GROQ_API_KEY, GROQ_MODEL,
)
from .prompts import (
    TREND_ANALYSIS_PROMPT, COMPETITIVE_REPORT_PROMPT,
    QUESTION_ANSWERING_PROMPT, PRODUCT_SUMMARIZATION_PROMPT,
    SCRAPING_PROMPT_TEMPLATE, CLIENT_PROFILING_PROMPT,
    RECOMMENDATION_PROMPT,
)
from ml_analysis.preprocessing.feature_engineering import get_top_k_products

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "groq" and GROQ_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                self._model = GROQ_MODEL
                logger.info(f"Groq client initialized (model: {GROQ_MODEL})")
                return
            except Exception as e:
                logger.warning(f"Failed to init Groq: {e}")
                self._client = None
        if self.provider == "openai" and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                self._model = OPENAI_MODEL
                logger.info("OpenAI client initialized")
                return
            except Exception as e:
                logger.warning(f"Failed to init OpenAI: {e}")
                self._client = None
        if self.provider == "anthropic" and ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
                self._model = ANTHROPIC_MODEL
                logger.info("Anthropic client initialized")
                return
            except Exception as e:
                logger.warning(f"Failed to init Anthropic: {e}")
                self._client = None
        logger.info("No LLM API key configured. Using mock responses.")
        self._client = None

    def _call_llm(self, prompt: str) -> str:
        if self._client and self.provider in ("openai", "xai", "groq"):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"API call failed ({self.provider}): {e}")
                return self._mock_response(prompt)

        if self._client and self.provider == "anthropic":
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")
                return self._mock_response(prompt)

        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        if "summary" in prompt.lower() or "summarize" in prompt.lower():
            return (
                "This product offers excellent value in its category with competitive pricing "
                "and strong customer ratings. It is positioned well in the market with "
                "positive reviews indicating high customer satisfaction."
            )
        if "trend" in prompt.lower() or "market" in prompt.lower():
            return (
                "**Market Trend Analysis:**\n\n"
                "1. **Premiumization Trend**: Higher-priced products with strong ratings dominate the top positions.\n"
                "2. **Category Opportunities**: Electronics and Fashion show highest demand with competitive pricing.\n"
                "3. **Geographic Insights**: US and UK markets show strongest product diversity and competition.\n"
                "4. **Recommendations**: Focus on products with ratings above 4.0, competitive pricing "
                "under $150, and maintain adequate stock levels."
            )
        if "compete" in prompt.lower() or "competitive" in prompt.lower():
            return (
                "**Competitive Analysis Report:**\n\n"
                "1. The market shows fragmented competition across multiple brands and categories.\n"
                "2. Price range varies significantly with premium and budget segments clearly defined.\n"
                "3. Top performers combine high ratings with moderate pricing and strong availability.\n"
                "4. Opportunities exist in the mid-range segment with high-quality products."
            )
        if "recommend" in prompt.lower() or "strategy" in prompt.lower():
            return (
                "**Strategic Recommendations:**\n\n"
                "1. Focus on products with strong engagement scores for marketing campaigns.\n"
                "2. Maintain competitive pricing within 10-15% of market average.\n"
                "3. Ensure stock availability for top-rated products to maximize revenue.\n"
                "4. Target categories with high ratings but limited competition."
            )
        return (
            "Based on the data analysis, the e-commerce landscape shows significant opportunities "
            "in high-rated product categories with competitive pricing. Recommended focus areas "
            "include maintaining strong inventory levels for popular items and leveraging "
            "customer reviews for marketing."
        )

    def generate_trend_summary(self, top_products_df, k: int = 10) -> str:
        top10 = top_products_df.head(k)
        table = top10[["name", "category", "price", "rating", "num_reviews",
                        "stock", "composite_score"]].to_string(index=False)
        prompt = TREND_ANALYSIS_PROMPT.format(top_products_table=table)
        return self._call_llm(prompt)

    def generate_competitive_report(self, df, category: str = "Electronics") -> str:
        cat_data = df[df["category"] == category].head(20).to_string(index=False)
        prompt = COMPETITIVE_REPORT_PROMPT.format(
            category=category,
            products_data=cat_data,
        )
        return self._call_llm(prompt)

    def answer_question(self, question: str, df) -> str:
        categories = ", ".join(df["category"].unique()[:10])
        countries = ", ".join(df["shop_country"].unique())
        brands = ", ".join(df["brand"].unique()[:10])

        prompt = QUESTION_ANSWERING_PROMPT.format(
            question=question,
            n_products=len(df),
            categories=categories,
            min_price=round(df["price"].min(), 2),
            max_price=round(df["price"].max(), 2),
            avg_rating=round(df["rating"].mean(), 2),
            countries=countries,
            brands=brands,
        )
        return self._call_llm(prompt)

    def summarize_product(self, name: str, description: str,
                          category: str = "", price: float = 0.0,
                          rating: float = 0.0, num_reviews: int = 0) -> str:
        prompt = PRODUCT_SUMMARIZATION_PROMPT.format(
            name=name,
            description=description[:500],
            category=category,
            price=price,
            rating=rating,
            num_reviews=num_reviews,
        )
        return self._call_llm(prompt)

    def generate_recommendations(self, analysis_data: dict) -> str:
        prompt = RECOMMENDATION_PROMPT.format(
            analysis_data=json.dumps(analysis_data, indent=2)
        )
        return self._call_llm(prompt)

    def generate_scraping_config(self, platform: str, url: str) -> str:
        prompt = SCRAPING_PROMPT_TEMPLATE.format(platform=platform, url=url)
        return self._call_llm(prompt)

    def generate_customer_profile(self, product_names: list[str]) -> str:
        products_str = "\n".join(f"- {p}" for p in product_names[:20])
        prompt = CLIENT_PROFILING_PROMPT.format(products=products_str)
        return self._call_llm(prompt)

    def clean_product_data(self, raw_text: str) -> str:
        prompt = (
            "You are a data cleaning assistant. Clean and standardize the following "
            "e-commerce product data. Fix capitalization, remove duplicates, "
            "standardize measurements, and format consistently.\n\n"
            f"Raw data:\n{raw_text}\n\nCleaned data:"
        )
        return self._call_llm(prompt)

    def standardize_product_name(self, name: str) -> str:
        prompt = (
            "Standardize this e-commerce product name following the format: "
            "'Brand Product Type - Key Feature'. Remove excess keywords.\n\n"
            f"Original: {name}\nStandardized:"
        )
        return self._call_llm(prompt)
