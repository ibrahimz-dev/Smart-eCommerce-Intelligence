PRODUCT_SUMMARIZATION_PROMPT = """
You are a product analyst. Given the following e-commerce product information,
generate a concise and informative summary (2-3 sentences).

Product Name: {name}
Description: {description}
Category: {category}
Price: ${price}
Rating: {rating}/5
Reviews: {num_reviews}

Summary:
"""

TREND_ANALYSIS_PROMPT = """
You are a market intelligence analyst. Based on the following Top-10 products data,
provide a strategic market analysis.

Top Products Data:
{top_products_table}

Your analysis should cover:
1. Key market trends observed
2. Pricing strategy insights
3. Category/segment opportunities
4. Recommendations for merchants

Analysis:
"""

COMPETITIVE_REPORT_PROMPT = """
You are a competitive intelligence analyst. Based on the following product data
for the "{category}" category, generate a competitive landscape report.

Products Data:
{products_data}

Your report should include:
1. Top competitors and their positioning
2. Price range analysis
3. Quality vs price comparison
4. Market gaps and opportunities
5. Strategic recommendations

Competitive Analysis Report:
"""

QUESTION_ANSWERING_PROMPT = """
You are an e-commerce intelligence assistant. Based on the following dataset summary,
answer the user's question.

Dataset Summary:
- Total Products: {n_products}
- Categories: {categories}
- Price Range: ${min_price} - ${max_price}
- Average Rating: {avg_rating}
- Top Countries: {countries}
- Top Brands: {brands}

User Question: {question}

Provide a clear, data-driven answer:
"""

SCRAPING_PROMPT_TEMPLATE = """
You are a web scraping specialist. Generate a scraping configuration for {platform}
store at URL: {url}.

Relevant product selectors to look for:
1. Product title/name
2. Price (current and original)
3. Description
4. Rating and review count
5. Stock/availability
6. Variants (color, size)

Provide the configuration as a structured plan:
"""

CLIENT_PROFILING_PROMPT = """
Based on the following products frequently browsed/bought together,
create a customer profile describing the typical buyer.

Products: {products}

Customer Profile:
"""

RECOMMENDATION_PROMPT = """
You are a marketing strategist. Based on the following analysis data,
generate actionable marketing recommendations.

Data:
{analysis_data}

Recommend at least 3 specific, data-driven marketing strategies:
"""
