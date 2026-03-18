# AutoInvestAI – Indian Auto Sector Sentiment Engine
## Master System Design Prompt (Save This File)

---

> **Usage:** Paste this entire prompt into a new chat whenever you need to continue building this project. It contains the full system specification, architecture decisions, and engineering constraints.

---

You are a senior Data Engineer and Quant AI Architect.

Design and implement a production-quality system called **AutoInvestAI – Indian Auto Sector Sentiment Engine**.

The goal is to analyze **sentiment from financial news and earnings transcripts for Indian automobile companies**, combine it with **stock market data**, and generate **investment intelligence signals** that feed into an existing Django + React application called AutoInvestAI, already deployed at `https://autoinvest.duckdns.org`.

The system must follow **modern data engineering best practices and MLOps standards**.

---

# Hard Constraints (Read First)

- **All tools must be free.** No paid APIs, no paid cloud tiers.
- **Databricks Community Edition only.** Single-node cluster. No autoscaling, no Delta Live Tables, no Databricks Workflows GUI. Use notebook-based job orchestration only.
- **All Spark/Delta operations must be compatible with a single-node CE cluster.** Avoid `spark.streams`, DLT pipelines, or any distributed-only features.
- **LLM for report generation:** Use Groq free tier (`llama3-8b-8192`) via the Groq Python SDK, or a local HuggingFace model. Do not assume OpenAI or Anthropic API access.
- **Sentiment model:** `ProsusAI/finbert` from HuggingFace Transformers. Do not suggest alternatives.
- **Vector store:** ChromaDB (in-memory inside Databricks notebook), with embeddings persisted manually to a Delta table so they survive restarts.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (free, lightweight, runs on CE).
- The developer is a **student**, so solutions should minimize cost and complexity while staying production-quality in structure.

---

# Existing Infrastructure (Do Not Redesign)

- Django REST backend already deployed on Azure VM (Ubuntu 24.04, Standard B2als v2)
- React frontend already deployed and served via Nginx
- PM2 for process management, DuckDNS for DNS, Certbot for SSL
- GitHub Actions CI/CD with appleboy SSH deploy action
- Existing models: LSTM and XGBoost for stock/crypto prediction

The new Auto Sector Sentiment Engine should be **additive** — new Django API endpoints and new React dashboard sections, not a replacement.

---

# Target Companies

| Company | Ticker |
|---|---|
| Maruti Suzuki | MARUTI.NS |
| Tata Motors | TATAMOTORS.NS |
| Mahindra & Mahindra | M&M.NS |
| Bajaj Auto | BAJAJ-AUTO.NS |
| Hero MotoCorp | HEROMOTOCO.NS |
| NIFTY AUTO Index (benchmark) | ^NSEAUTO |

---

# Data Sources (Free Only)

| Data Type | Source | Tool |
|---|---|---|
| Stock OHLCV | Yahoo Finance | `yfinance` |
| News articles | NewsAPI free tier (100 req/day) or GNews API | `requests` |
| Earnings transcripts | Screener.in (public HTML/PDF pages) | `BeautifulSoup`, `pdfplumber` |
| Sector benchmark | Yahoo Finance | `yfinance` |

> **Do not use Moneycontrol scraping** — paywalled and fragile. Do not use X/Twitter API — not reliably free.

---

# Required Output Structure

When responding to this prompt, structure the output in this exact order:

1. **Architecture Overview** — system diagram description and data flow
2. **Bronze Layer** — table schemas, ingestion code, data quality checks
3. **Silver Layer** — preprocessing code, feature engineering, source quality scoring
4. **Vector DB Layer** — ChromaDB setup, embedding pipeline, Delta persistence schema
5. **Gold Layer** — analytics tables, sentiment scoring, signal generation
6. **Intelligence Layer** — BUY/NEUTRAL/RISK ALERT logic with thresholds
7. **Report Generation** — LLM prompt template and Groq integration
8. **Django API Layer** — endpoint definitions with JSON response schemas
9. **React Dashboard Layer** — component breakdown and data flow
10. **Orchestration** — how notebooks are scheduled/chained on CE

---

# Medallion Architecture

## BRONZE LAYER — Raw Ingestion

### Tables

**`bronze.raw_stock_prices`**
```
ticker          STRING
date            DATE
open            DOUBLE
high            DOUBLE
low             DOUBLE
close           DOUBLE
volume          LONG
ingested_at     TIMESTAMP
source          STRING  -- 'yfinance'
```

**`bronze.raw_news_articles`**
```
article_id      STRING  -- MD5 hash of URL
url             STRING
title           STRING
body            STRING
source          STRING
published_at    TIMESTAMP
company_tag     STRING
ingested_at     TIMESTAMP
```

**`bronze.raw_transcripts`**
```
transcript_id   STRING
company         STRING
quarter         STRING  -- e.g. 'Q3FY25'
raw_text        STRING
source_url      STRING
ingested_at     TIMESTAMP
```

**`bronze.prompt_versions`**
```
version         STRING  -- e.g. 'v1.2'
task            STRING  -- e.g. 'sentiment_analysis'
model_name      STRING  -- e.g. 'ProsusAI/finbert'
prompt_template STRING
parameters      STRING  -- JSON string
created_at      TIMESTAMP
is_active       BOOLEAN
```

### Data Quality Checks (Bronze)
- Null price detection: flag rows where `close IS NULL`
- Duplicate news: deduplicate on `article_id` (URL hash)
- Empty transcripts: flag rows where `LENGTH(raw_text) < 100`
- Timestamp consistency: flag future timestamps or dates before 2020

---

## SILVER LAYER — Cleaning and Feature Engineering

### Tables

**`silver.processed_stocks`**
```
ticker              STRING
date                DATE
open, high, low, close, volume  (same as bronze)
daily_return        DOUBLE  -- (close - prev_close) / prev_close
ma_5                DOUBLE  -- 5-day moving average
ma_20               DOUBLE  -- 20-day moving average
rsi_14              DOUBLE  -- RSI indicator
volatility_20d      DOUBLE  -- rolling 20-day std of returns
rel_return_vs_index DOUBLE  -- ticker return - NIFTY AUTO return
trading_session_date DATE   -- after-market news maps to next session
processed_at        TIMESTAMP
```

**`silver.cleaned_news_chunks`**
```
chunk_id        STRING
article_id      STRING
company         STRING
source          STRING
source_quality  DOUBLE  -- scoring: Reuters=1.0, ET=0.8, blogs=0.4
chunk_text      STRING  -- 512-token window
chunk_index     INT
published_at    TIMESTAMP
trading_session_date DATE
processed_at    TIMESTAMP
```

**`silver.cleaned_transcript_chunks`**
```
chunk_id        STRING
transcript_id   STRING
company         STRING
quarter         STRING
chunk_text      STRING
chunk_index     INT
chunk_tokens    INT
processed_at    TIMESTAMP
```

### Source Quality Scoring
```python
SOURCE_QUALITY = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "economic times": 0.8,
    "business standard": 0.8,
    "livemint": 0.75,
    "moneycontrol": 0.7,
    "default": 0.4
}
```

---

## VECTOR DB LAYER — Semantic Search

### ChromaDB Setup (Databricks Notebook)
- Collection per company: `maruti_chunks`, `tatamotors_chunks`, etc.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Persist embeddings to Delta after each run

**`vector_store.embedding_metadata`**
```
chunk_id            STRING
company             STRING
collection_name     STRING
embedding_model     STRING
chroma_doc_id       STRING
source_type         STRING  -- 'news' or 'transcript'
created_at          TIMESTAMP
```

### Semantic Query Examples to Support
- "EV strategy"
- "production capacity guidance"
- "margin pressure"
- "demand outlook rural"

---

## GOLD LAYER — Analytics and Intelligence

### Tables

**`gold.company_sentiment_scores`**
```
company             STRING
score_date          DATE
sentiment_score     DOUBLE  -- 0 to 10
weighted_sentiment  DOUBLE  -- raw weighted value
article_count       INT
transcript_count    INT
sentiment_intensity DOUBLE  -- score × log(article_count)
granularity         STRING  -- 'daily' | 'weekly' | 'monthly'
```

**`gold.sentiment_momentum`**
```
company             STRING
score_date          DATE
sentiment_score     DOUBLE
ma_7d_sentiment     DOUBLE
momentum            DOUBLE  -- current_score - ma_7d_sentiment
```

**`gold.event_detection`**
```
company             STRING
event_date          DATE
event_type          STRING  -- 'EV_LAUNCH' | 'PRICE_INCREASE' | 'PRODUCTION_EXPANSION' | 'RECALL' | 'LABOR_STRIKE'
confidence          DOUBLE
source_chunk_id     STRING
```

**`gold.investment_signals`**
```
company             STRING
signal_date         DATE
signal              STRING  -- 'BUY' | 'NEUTRAL' | 'RISK_ALERT'
sentiment_score     DOUBLE
momentum            DOUBLE
price_vs_ma20       DOUBLE  -- (close - ma_20) / ma_20
reasoning           STRING  -- human-readable explanation
```

**`gold.lag_correlation`**
```
company             STRING
lag_days            INT     -- 1, 2, or 3
correlation         DOUBLE
computed_at         TIMESTAMP
```

---

## SENTIMENT SCORING FORMULAS

### Probability-Weighted Sentiment
```python
# For each chunk, FinBERT returns: label + probability
label_weights = {"positive": 1, "neutral": 0, "negative": -1}

weighted_score = sum(label_weights[label] * prob for label, prob in results) / len(results)

# Normalize to 0-10
sentiment_score = (weighted_score + 1) / 2 * 10
```

### Sentiment Momentum
```python
momentum = current_sentiment_score - rolling_7d_avg_sentiment_score
```

### News Volume Impact
```python
sentiment_intensity = sentiment_score * math.log(article_count + 1)
```

---

## INVESTMENT SIGNAL LOGIC

```python
def generate_signal(sentiment_score, momentum, price_vs_ma20):
    if sentiment_score > 7 and momentum > 0 and price_vs_ma20 > 0:
        return "BUY"
    elif sentiment_score < 4 or momentum < -1.5:
        return "RISK_ALERT"
    else:
        return "NEUTRAL"
```

---

## REPORT GENERATION

Use Groq free tier (`llama3-8b-8192`). Generate weekly.

### Prompt Template (store in `bronze.prompt_versions`)
```
version: v1.0
task: weekly_sector_report
model: llama3-8b-8192
template: |
  You are a financial analyst covering Indian automobile stocks.
  Based on the following data for the week ending {report_date}:

  Top sentiment company: {top_company} with score {top_score}/10
  Lowest sentiment: {bottom_company} with score {bottom_score}/10
  Sector average sentiment: {sector_avg}/10

  Key events detected: {event_summary}

  Sentiment vs price correlation: {correlation_summary}

  Write a 3-paragraph professional sector intelligence report covering:
  1. Overall sector sentiment and what's driving it
  2. Standout companies (positive and negative)
  3. Investment outlook and key risks to watch

  Keep the tone objective and data-driven. No disclaimers.
```

---

## DJANGO API LAYER

Add these new endpoints to the existing AutoInvestAI Django project:

### `/api/auto-sector/report/`
**GET** → Returns latest weekly sector report
```json
{
  "report_date": "2026-03-15",
  "sector_avg_sentiment": 6.4,
  "report_text": "...",
  "top_company": "Maruti Suzuki",
  "bottom_company": "Tata Motors",
  "generated_by": "llama3-8b-8192"
}
```

### `/api/auto-sector/sentiment/`
**GET** `?company=MARUTI&granularity=weekly`
```json
{
  "company": "Maruti Suzuki",
  "ticker": "MARUTI.NS",
  "scores": [
    { "date": "2026-03-10", "score": 7.2, "momentum": 0.4, "article_count": 12 }
  ],
  "signal": "BUY"
}
```

### `/api/auto-sector/heatmap/`
**GET** → Returns all 5 companies latest sentiment for heatmap
```json
{
  "date": "2026-03-18",
  "companies": [
    { "company": "Maruti Suzuki", "score": 7.2, "signal": "BUY" },
    { "company": "Tata Motors", "score": 4.1, "signal": "RISK_ALERT" }
  ]
}
```

### `/api/auto-sector/events/`
**GET** `?company=TATAMOTORS`
```json
{
  "company": "Tata Motors",
  "events": [
    { "date": "2026-03-12", "event_type": "EV_LAUNCH", "confidence": 0.87 }
  ]
}
```

**Implementation note:** Use `databricks-sql-connector` Python package to query Gold Delta tables. Use connection pooling — don't open a new connection per request.

---

## REACT DASHBOARD

Add a new route `/auto-sector` to the existing AutoInvestAI React app with these components:

1. **SectorHeatmap** — 5-company grid with color-coded sentiment scores (red < 4, yellow 4–7, green > 7)
2. **SentimentLeaderboard** — ranked table: company, score, momentum, signal badge
3. **SentimentVsPrice** — dual-axis recharts line chart: sentiment score (left) vs stock price (right) over 3 months
4. **EventTimeline** — chronological list of detected corporate events with company tag and confidence
5. **SectorReport** — rendered weekly report text with metadata

---

## NOTEBOOK ORCHESTRATION (Databricks CE)

Since CE has no Workflows GUI, chain notebooks using `%run` or `dbutils.notebook.run()`:

```
master_pipeline.py
├── 01_bronze_ingest.py       (runs daily via CE scheduled job)
├── 02_silver_process.py      (runs after bronze)
├── 03_vector_embed.py        (runs after silver)
├── 04_gold_sentiment.py      (runs after silver)
├── 05_gold_signals.py        (runs after gold sentiment)
└── 06_report_generate.py     (runs weekly)
```

Schedule using Databricks CE's built-in job scheduler (single job, single cluster).

---

# Summary of All Delta Tables

| Layer | Table | Purpose |
|---|---|---|
| Bronze | raw_stock_prices | Raw OHLCV from yfinance |
| Bronze | raw_news_articles | Raw news with URL hash dedup |
| Bronze | raw_transcripts | Raw earnings transcript text |
| Bronze | prompt_versions | LLM prompt version registry |
| Silver | processed_stocks | Cleaned stocks + all indicators |
| Silver | cleaned_news_chunks | Chunked + tagged news |
| Silver | cleaned_transcript_chunks | Chunked transcripts |
| Vector | embedding_metadata | ChromaDB embedding registry |
| Gold | company_sentiment_scores | Daily/weekly/monthly scores |
| Gold | sentiment_momentum | 7-day momentum per company |
| Gold | event_detection | Corporate event classifications |
| Gold | investment_signals | BUY/NEUTRAL/RISK_ALERT |
| Gold | lag_correlation | Sentiment-to-price lag analysis |

---

*Prompt version: v1.1 | Last updated: 2026-03-18 | Project: AutoInvestAI Auto Sector Engine*