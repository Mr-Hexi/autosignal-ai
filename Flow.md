# AutoSignal-AI: End-to-End Pipeline Architecture

This document outlines the technical workflow of the AutoSignal-AI system, from raw data ingestion to the final React dashboard.

---

## Step 1 — Data Collection (Bronze Layer)
The pipeline initiates with five specialized collection scripts:

* **Stock Prices:** `yfinance` pulls daily OHLVC data for the 5 target companies and the **NIFTY AUTO** index (12-month lookback).
* **Financial Ratios:** Pulls PE ratio, profit margins, ROE, revenue growth, EPS, and market cap.
* **NSE Announcements:** Utilizes the `nse` Python package to fetch official corporate filings (investor meets, earnings, litigation, dividends).
* **ET News:** Parses two Economic Times RSS feeds (approx. 70-80 articles per run).
* **BSE Transcripts:** Fetches earnings call PDFs via the `bse` package; `pdfplumber` extracts raw text.

---

## Step 2 — Cleaning and Feature Engineering (Silver Layer)
Raw data is processed to remove OCR noise, HTML tags, and duplicates.

### Technical Indicators
The system computes 7 features per ticker:
1. **Daily Return:** Percentage change.
2. **Moving Averages:** 5-day and 20-day (MA5/MA20).
3. **RSI (14-day):** Calculated via the exponential weighted method.
4. **Volatility:** 20-day rolling standard deviation.
5. **Price vs MA20:** Deviation percentage from the trendline.
6. **Relative Return:** Performance vs. the NIFTY AUTO benchmark.

### Text & Fundamentals
* **Text Chunking:** News and transcripts are stripped of HTML, deduplicated by URL hash, and split into **384-token windows** to fit FinBERT’s constraints.
* **Fundamentals:** Ratios are normalized to percentages and a composite **Profitability Score (0-10)** is generated.

---

## Step 3 — Sentiment Analysis (Gold Layer)
We utilize **FinBERT** (`ProsusAI/finbert`) to process ~276 text chunks per run. 

### Weighting Logic
For every chunk, FinBERT provides probabilities for `positive`, `neutral`, and `negative`. We calculate a weighted sentiment value:

$$weighted = (+1 \times pos) + (0 \times neu) + (-1 \times neg)$$

**Example:**
If a chunk is $0.72$ positive and $0.07$ negative:
$$weighted = (+1 \times 0.72) + (-1 \times 0.07) = 0.65$$

The final **Sentiment Score (0-10)** is normalized:
$$SentimentScore = \frac{WeightedAvg + 1}{2} \times 10$$

---

## Step 4 — Multi-Factor Signal Engine
The system combines technical, fundamental, and sentiment data using a weighted matrix:

| Factor | Weight | Source |
| :--- | :--- | :--- |
| **Sentiment Score** | 40% | FinBERT Output |
| **Sentiment Momentum** | 0% | (Placeholder for historical delta) |
| **Price Momentum** | 30% | 5-day price return (normalized 0-10) |
| **Technical Signal** | 20% | RSI (Oversold = 7.5, Overbought = 2.5) |
| **Fundamental Score** | 10% | Growth + Margin metrics |

### Signal Logic
* **BUY:** Composite $\ge 5.5$
* **NEUTRAL:** Composite $4.8 - 5.5$
* **RISK_ALERT:** Composite $\le 4.8$

---

## Step 5 — Report Generation
1.  **Sector Report:** A structured prompt is sent to **Groq (Llama-3.1-8b-instant)** to generate a 3-paragraph professional outlook.
2.  **Company Reports:** Individual deep-dives are generated for each ticker, incorporating specific event data and financials.

---

## Step 6 — Vector Embeddings (ChromaDB)
Text chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in **ChromaDB**:
* `autosignal_news`: 139 chunks
* `autosignal_transcripts`: 148 chunks

This powers the **Semantic Search** and "Ask AI" features, allowing users to query topics like "EV strategy" across all filings.

---

## Step 7 — Django API Endpoints
The backend serves the Gold Layer data via the following REST endpoints:

| Endpoint | Returns |
| :--- | :--- |
| `/api/autosignal/heatmap/` | Sentiment + Signals for all companies |
| `/api/autosignal/report/` | Latest Groq sector report |
| `/api/autosignal/events/` | Corporate events timeline |
| `/api/autosignal/company/<slug>/` | Full company detail + stock history |
| `/api/autosignal/search/` | Semantic search via ChromaDB |

---

## Step 8 — React Dashboard
* **Sector Overview:** Heatmap cards (Green/Yellow/Red) and a ranked leaderboard.
* **Company Detail:** Interactive **Recharts** price charts with Price, RSI, Return, and Sentiment modes.
* **Timeline:** Full corporate event history with direct PDF links to NSE/BSE filings.

---

## Data Flow Visualization

```mermaid
graph TD
    A[yfinance / NSE API / ET RSS / BSE PDFs] -->|Raw Data| B(Bronze CSVs)
    B -->|Cleaning & Features| C(Silver CSVs)
    C --> D{Processing}
    D -->|Vectorizing| E[ChromaDB - Semantic Search]
    D -->|Inference| F[FinBERT Sentiment 0-10]
    E --> G[Multi-Factor Signal Engine]
    F --> G
    G -->|BUY / NEUTRAL / RISK| H[Groq LLM Reports]
    H --> I[Django REST API]
    I --> J[React Dashboard]