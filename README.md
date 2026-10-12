# Smart eCommerce Intelligence

**FST Tanger — LSI 2 — DM & SID — 2025/2026**

---

## Dataset

| Metric | Value |
|--------|-------|
| Products | **4133** (cleaned from 5000) |
| Columns | **53** (original + computed features) |
| Categories | **16** (Footwear, T-Shirts, Sleepwear, Underwear, etc.) |
| Shops | 15 e-commerce sites |
| Data file | `data/scraped/analyzed_products_clean.csv` |

---

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (optional - for LLM features)
# Edit .env file and add your Groq/OpenAI/Anthropic API key
# Get free Groq key at https://groq.com
```

---

## Usage

```bash
# 1. Dashboard BI (Streamlit on port 8501)
streamlit run dashboard/app.py

# 2. ML Analysis
python3 ml_analysis/run_analysis.py \
  --data data/scraped/analyzed_products_clean.csv \
  --output output

# 3. Scraping (requires network access)
python3 scraping/run_scraping.py --mode shopify --output data/scraped

# 4. Tests
python3 tests/test_scraping.py
python3 tests/test_ml_analysis.py
```

---

## Modules

| Module | Description |
|--------|-------------|
| **Scraping A2A** | Agents for Shopify + WooCommerce (`scraping/agents/`) |
| **ML Analysis** | RF, XGBoost, KMeans, DBSCAN, Hierarchical, PCA, Apriori |
| **Kubeflow** | 6-stage ML pipeline (`kubeflow_pipeline/`) |
| **Dashboard BI** | Streamlit — 7 pages (Overview, Explorer, ML, Top-K, Rules, PCA, LLM) |
| **LLM** | Summarization, competitive analysis, Q&A, profiling (via .env config) |
| **MCP** | Responsible AI architecture documentation |

---

## Project Structure

```
sma/
├── scraping/
│   ├── run_scraping.py           # Main scraper
│   └── agents/                   # A2A agents (base, shopify, woocommerce)
│   └── utils/                    # HTML parsing, data cleaning
├── ml_analysis/
│   ├── models/                   # RF, XGBoost, KMeans, DBSCAN, PCA, Association Rules
│   ├── preprocessing/            # Data loader, feature engineering, Top-K scoring
│   └── evaluation/               # Metrics, confusion matrix, ROC, silhouette
├── kubeflow_pipeline/           # Docker + KFP components + pipeline
├── dashboard/                   # Streamlit BI dashboard
├── llm_module/                   # LLM agent, prompts, enrichment, competitive analysis
├── mcp_reflection/              # MCP architecture documentation
├── tests/                        # Unit tests
├── data/scraped/                 # CSV datasets
├── output/                       # ML results (ROC, CM, Top-K)
├── .env                          # API keys configuration
├── requirements.txt             # Dependencies
└── README.md
```

---

## Requirements Covered

- ✅ Scraping Shopify + WooCommerce (A2A agents)
- ✅ Classification: Random Forest, XGBoost (accuracy, precision, recall, F1, ROC)
- ✅ Clustering: KMeans, Hierarchical, DBSCAN (silhouette score)
- ✅ Association rules: Apriori (support, confidence, lift)
- ✅ PCA 2D visualization
- ✅ Top-K weighted scoring (rating 35% + reviews 25% + price 20% + stock 20%)
- ✅ Kubeflow pipeline (6 components + Docker)
- ✅ Dashboard BI (Streamlit + Plotly — 7 pages)
- ✅ LLM enrichment (summaries, trends, competitive analysis, Q&A, profiles)
- ✅ Responsible AI architecture (MCP - Model Context Protocol)
- ✅ CI/CD (GitHub Actions)
- ✅ Unit tests

---

## API Keys

Create `.env` file for LLM features:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
# Or use OpenAI/Anthropic
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Get free API keys at:
- **Groq**: https://groq.com (recommended - free tier)
- **OpenAI**: https://platform.openai.com
- **Anthropic**: https://www.anthropic.com