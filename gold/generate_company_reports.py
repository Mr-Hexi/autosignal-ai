import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import json
import yaml

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SENTIMENT_FILE  = "data/gold/company_sentiment_scores.csv"
SIGNALS_FILE    = "data/gold/investment_signals.csv"
FINANCIALS_FILE = "data/silver/processed_financials.csv"
STOCKS_FILE     = "data/silver/processed_stocks.csv"
NEWS_FILE       = "data/silver/processed_news.csv"
TRANSCRIPTS_FILE= "data/silver/processed_transcripts.csv"
NSE_FILE        = "data/bronze/nse_announcements/_all_announcements.csv"
OUTPUT_DIR      = "data/gold/reports/companies"

GROQ_MODEL      = "llama-3.1-8b-instant"

PROMPT_VERSIONS_DIR = "data/bronze/prompt_versions"

COMPANIES = [
    "Maruti Suzuki",
    "Tata Motors",
    "Mahindra & Mahindra",
    "Bajaj Auto",
    "Hero MotoCorp",
]

COMPANY_SLUGS = {
    "Maruti Suzuki":       "maruti-suzuki",
    "Tata Motors":         "tata-motors",
    "Mahindra & Mahindra": "mahindra-mahindra",
    "Bajaj Auto":          "bajaj-auto",
    "Hero MotoCorp":       "hero-motocorp",
}

TICKER_MAP = {
    "Maruti Suzuki":       "MARUTI.NS",
    "Tata Motors":         "TMPV.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto":          "BAJAJ-AUTO.NS",
    "Hero MotoCorp":       "HEROMOTOCO.NS",
}

PROMPT_VERSION = {
    "version":         "v1.0",
    "task":            "company_report",
    "model_name":      GROQ_MODEL,
    "created_at":      "2026-03-18",
    "is_active":       True,
    "parameters": {
        "max_tokens":  600,
        "temperature": 0.3,
    },
    "prompt_template": """You are a senior equity analyst covering Indian automobile stocks.

Today is {report_date}. Analyze the following data for {company}:

SENTIMENT:
- Sentiment Score: {sentiment_score}/10
- Signal: {signal}
- Momentum: {momentum}
- News chunks analyzed: {article_count}
- Transcript chunks analyzed: {transcript_count}

STOCK PERFORMANCE:
- Current Price: Rs. {close}
- vs 20-day MA: {price_vs_ma20_pct}%
- RSI (14-day): {rsi_14}
- 3-month volatility: {volatility}

FINANCIALS:
- Profit Margin: {profit_margin}%
- Revenue Growth: {revenue_growth}%
- Profitability Score: {profitability_score}/10
- Trailing PE: {trailing_pe}
- EPS: {eps}

RECENT EVENTS:
{events_summary}

Write a professional 3-paragraph company intelligence report covering:
1. Current sentiment and what news/transcripts are driving it
2. Financial health and stock performance assessment
3. Investment outlook with specific risks and opportunities

Rules:
- Be specific, cite the numbers
- Objective and data-driven tone
- No disclaimers
- Maximum 300 words"""
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def format_date():
    return datetime.now().strftime("%d %B %Y")

def safe(val, default="N/A", suffix=""):
    if val is None or (isinstance(val, float) and val != val):
        return default
    return f"{val}{suffix}"

def save_prompt_version(version: dict):
    os.makedirs(PROMPT_VERSIONS_DIR, exist_ok=True)
    json_path = os.path.join(
        PROMPT_VERSIONS_DIR,
        f"prompt_{version['task']}_{version['version']}.json"
    )
    yaml_path = os.path.join(
        PROMPT_VERSIONS_DIR,
        f"prompt_{version['task']}_{version['version']}.yaml"
    )
    with open(json_path, "w") as f:
        json.dump(version, f, indent=2)
    with open(yaml_path, "w") as f:
        yaml.dump(version, f, default_flow_style=False, allow_unicode=True)

# ── Load All Data ─────────────────────────────────────────────────────────────

def load_all_data() -> dict:
    return {
        "sentiment":   pd.read_csv(SENTIMENT_FILE),
        "signals":     pd.read_csv(SIGNALS_FILE),
        "financials":  pd.read_csv(FINANCIALS_FILE),
        "stocks":      pd.read_csv(STOCKS_FILE, parse_dates=["date"]),
        "news":        pd.read_csv(NEWS_FILE),
        "transcripts": pd.read_csv(TRANSCRIPTS_FILE),
        "nse":         pd.read_csv(NSE_FILE),
    }

# ── Build Company Data ────────────────────────────────────────────────────────

def get_company_data(company: str, data: dict) -> dict:
    ticker = TICKER_MAP[company]

    # Sentiment
    sent = data["sentiment"][data["sentiment"]["company"] == company]
    sig  = data["signals"][data["signals"]["company"] == company]
    fin  = data["financials"][data["financials"]["company"] == company]

    # Latest stock row
    stock = data["stocks"][data["stocks"]["ticker"] == ticker]
    latest_stock = stock.sort_values("date").iloc[-1] if not stock.empty else {}

    # Stock history for chart
    stock_history = stock.sort_values("date")[
        ["date", "open", "high", "low", "close", "volume",
         "daily_return", "ma_5", "ma_20", "rsi_14", "volatility_20d"]
    ].copy()
    stock_history["date"] = stock_history["date"].dt.strftime("%Y-%m-%d")

    # Recent events
    nse = data["nse"][data["nse"]["company"] == company].head(5)
    events = nse[["subject", "broadcast_date", "attachment_url"]].fillna("").to_dict(orient="records")

    # News chunks for this company
    news_chunks = data["news"][
        data["news"]["company_tags"].str.contains(company.split()[0], case=False, na=False)
    ]["chunk_text"].head(5).tolist()

    # Transcript chunks
    transcript_chunks = data["transcripts"][
        (data["transcripts"]["company"] == company) &
        (data["transcripts"]["has_text"] == True)
    ][["chunk_text", "quarter", "filing_date"]].head(3).fillna("").to_dict(orient="records")

    return {
        "company":           company,
        "slug":              COMPANY_SLUGS[company],
        "ticker":            ticker,
        "report_date":       format_date(),
        "generated_at":      now(),
        # Sentiment
        "sentiment_score":   float(sent["sentiment_score"].iloc[0]) if not sent.empty else 5.0,
        "article_count":     int(sent["article_count"].iloc[0]) if not sent.empty else 0,
        "transcript_count":  int(sent["transcript_count"].iloc[0]) if not sent.empty else 0,
        "sentiment_intensity": float(sent["sentiment_intensity"].iloc[0]) if not sent.empty else 0,
        # Signal
        "signal":            sig["signal"].iloc[0] if not sig.empty else "NEUTRAL",
        "momentum":          float(sig["momentum"].iloc[0]) if not sig.empty else 0.0,
        "rsi_14":            float(sig["rsi_14"].iloc[0]) if not sig.empty else 50.0,
        "price_vs_ma20":     float(sig["price_vs_ma20"].iloc[0]) if not sig.empty else 0.0,
        "profitability_score": float(sig["profitability_score"].iloc[0]) if not sig.empty else 5.0,
        "reasoning":         sig["reasoning"].iloc[0] if not sig.empty else "",
        # Stock
        "close":             float(latest_stock.get("close", 0)),
        "ma_20":             float(latest_stock.get("ma_20", 0)),
        "volatility_20d":    float(latest_stock.get("volatility_20d", 0)),
        "stock_history":     stock_history.to_dict(orient="records"),
        # Financials
        "profit_margin_pct":    float(fin["profit_margin_pct"].iloc[0]) if not fin.empty else None,
        "revenue_growth_pct":   float(fin["revenue_growth_pct"].iloc[0]) if not fin.empty else None,
        "trailing_pe":          float(fin["trailing_pe"].iloc[0]) if not fin.empty else None,
        "eps_trailing":         float(fin["eps_trailing"].iloc[0]) if not fin.empty else None,
        "debt_to_equity":       float(fin["debt_to_equity"].iloc[0]) if not fin.empty else None,
        "market_cap_cr":        float(fin["market_cap_cr"].iloc[0]) if not fin.empty else None,
        "roe_pct":              float(fin["roe_pct"].iloc[0]) if not fin.empty else None,
        # Events
        "recent_events":     events,
        # Text samples
        "news_samples":      news_chunks,
        "transcript_samples": transcript_chunks,
    }

# ── Build Prompt ──────────────────────────────────────────────────────────────

def build_prompt(cd: dict) -> str:
    events_summary = "\n".join([
        f"- {e.get('broadcast_date', '')[:10]}: {e.get('subject', '')}"
        for e in cd["recent_events"]
    ]) or "No recent events"

    return f"""You are a senior equity analyst covering Indian automobile stocks.

Today is {cd['report_date']}. Analyze the following data for {cd['company']}:

SENTIMENT:
- Sentiment Score: {cd['sentiment_score']}/10
- Signal: {cd['signal']}
- Momentum: {cd['momentum']}
- News chunks analyzed: {cd['article_count']}
- Transcript chunks analyzed: {cd['transcript_count']}

STOCK PERFORMANCE:
- Current Price: Rs. {cd['close']:.2f}
- vs 20-day MA: {cd['price_vs_ma20']*100:.1f}%
- RSI (14-day): {cd['rsi_14']:.1f}
- 3-month volatility: {cd['volatility_20d']:.4f}

FINANCIALS:
- Profit Margin: {safe(cd['profit_margin_pct'])}%
- Revenue Growth: {safe(cd['revenue_growth_pct'])}%
- Profitability Score: {cd['profitability_score']}/10
- Trailing PE: {safe(cd['trailing_pe'])}
- EPS: {safe(cd['eps_trailing'])}

RECENT EVENTS:
{events_summary}

Write a professional 3-paragraph company intelligence report covering:
1. Current sentiment and what news/transcripts are driving it
2. Financial health and stock performance assessment
3. Investment outlook with specific risks and opportunities

Rules:
- Be specific, cite the numbers
- Objective and data-driven tone
- No disclaimers
- Maximum 300 words"""

# ── Call Groq ─────────────────────────────────────────────────────────────────

def call_groq(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ── Save ──────────────────────────────────────────────────────────────────────

def save_company_report(cd: dict, report_text: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = cd["slug"]

    output = {**cd, "report_text": report_text, "generated_by": GROQ_MODEL}

    # Remove large stock history from main file — serve separately
    output_slim = {k: v for k, v in output.items() if k != "stock_history"}

    path = os.path.join(OUTPUT_DIR, f"{slug}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_slim, f, indent=2, default=str)

    # Save stock history separately
    history_path = os.path.join(OUTPUT_DIR, f"{slug}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(cd["stock_history"], f, indent=2, default=str)

    print(f"  Saved -> {path}")
    print(f"  Saved -> {history_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Gold Company Report Generation")
    print(f"{'='*50}\n")

    save_prompt_version(PROMPT_VERSION)
    print(f"  Prompt version saved: {PROMPT_VERSION['version']}\n")

    data = load_all_data()

    for company in COMPANIES:
        print(f"  Processing: {company}")
        cd      = get_company_data(company, data)
        prompt  = build_prompt(cd)
        report  = call_groq(prompt)
        save_company_report(cd, report)
        print(f"  Report preview: {report[:100]}...\n")

    print("Company report generation complete.\n")

if __name__ == "__main__":
    run()