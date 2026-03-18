
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = {
    "MARUTI.NS":      "Maruti Suzuki",
    "TMPV.NS":  "Tata Motors",
    "M&M.NS":         "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "HEROMOTOCO.NS":  "Hero MotoCorp",
}

OUTPUT_DIR = "data/bronze/financials"

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_get(d: dict, key: str):
    val = d.get(key, None)
    return None if val == "Infinity" or val != val else val  # handles NaN too

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

# ── Fetch Key Ratios from .info ───────────────────────────────────────────────

def fetch_key_ratios(ticker: str, company: str) -> pd.DataFrame:
    print(f"  Fetching key ratios: {ticker}...")
    t = yf.Ticker(ticker)
    info = t.info

    row = {
        "ticker":               ticker,
        "company":              company,
        "ingested_at":          now(),
        # Valuation
        "trailing_pe":          safe_get(info, "trailingPE"),
        "forward_pe":           safe_get(info, "forwardPE"),
        "price_to_book":        safe_get(info, "priceToBook"),
        "enterprise_to_ebitda": safe_get(info, "enterpriseToEbitda"),
        # Profitability
        "profit_margin":        safe_get(info, "profitMargins"),
        "operating_margin":     safe_get(info, "operatingMargins"),
        "gross_margin":         safe_get(info, "grossMargins"),
        "return_on_equity":     safe_get(info, "returnOnEquity"),
        "return_on_assets":     safe_get(info, "returnOnAssets"),
        # Growth
        "revenue_growth":       safe_get(info, "revenueGrowth"),
        "earnings_growth":      safe_get(info, "earningsGrowth"),
        # Financial health
        "debt_to_equity":       safe_get(info, "debtToEquity"),
        "current_ratio":        safe_get(info, "currentRatio"),
        "free_cashflow":        safe_get(info, "freeCashflow"),
        "total_revenue":        safe_get(info, "totalRevenue"),
        "total_debt":           safe_get(info, "totalDebt"),
        # Per share
        "eps_trailing":         safe_get(info, "trailingEps"),
        "eps_forward":          safe_get(info, "forwardEps"),
        "book_value":           safe_get(info, "bookValue"),
        # Market
        "market_cap":           safe_get(info, "marketCap"),
        "52w_high":             safe_get(info, "fiftyTwoWeekHigh"),
        "52w_low":              safe_get(info, "fiftyTwoWeekLow"),
        "dividend_yield":       safe_get(info, "dividendYield"),
    }

    return pd.DataFrame([row])

# ── Fetch Income Statement (Quarterly) ───────────────────────────────────────

def fetch_quarterly_financials(ticker: str, company: str) -> pd.DataFrame:
    print(f"  Fetching quarterly financials: {ticker}...")
    t = yf.Ticker(ticker)

    try:
        df = t.quarterly_financials.T  # transpose: rows=quarters, cols=metrics
        if df.empty:
            print(f"  WARNING: No quarterly financials for {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"index": "period"})
        df["ticker"]      = ticker
        df["company"]     = company
        df["ingested_at"] = now()
        df["period"]      = df["period"].astype(str)

        # Normalize column names
        df.columns = [
            c.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            for c in df.columns
        ]

        return df

    except Exception as e:
        print(f"  ERROR fetching quarterly financials for {ticker}: {e}")
        return pd.DataFrame()

# ── Fetch Balance Sheet (Quarterly) ──────────────────────────────────────────

def fetch_quarterly_balance_sheet(ticker: str, company: str) -> pd.DataFrame:
    print(f"  Fetching quarterly balance sheet: {ticker}...")
    t = yf.Ticker(ticker)

    try:
        df = t.quarterly_balance_sheet.T
        if df.empty:
            print(f"  WARNING: No balance sheet for {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"index": "period"})
        df["ticker"]      = ticker
        df["company"]     = company
        df["ingested_at"] = now()
        df["period"]      = df["period"].astype(str)

        df.columns = [
            c.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            for c in df.columns
        ]

        return df

    except Exception as e:
        print(f"  ERROR fetching balance sheet for {ticker}: {e}")
        return pd.DataFrame()

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    null_pct = df.isnull().mean().mean() * 100
    print(f"  QC [{label}]: {len(df)} rows, {null_pct:.1f}% null values overall")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, subfolder: str, filename: str):
    if df.empty:
        return
    path = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False)
    print(f"  Saved → {full_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Bronze Financials Ingestion")
    print(f"{'='*50}\n")

    all_ratios = []

    for ticker, company in TICKERS.items():
        print(f"\n── {company} ({ticker}) ──")

        # Key ratios
        ratios = fetch_key_ratios(ticker, company)
        ratios = run_quality_checks(ratios, f"{ticker} ratios")
        save(ratios, "key_ratios", f"{ticker.replace('&','_')}_ratios.csv")
        all_ratios.append(ratios)

        # Quarterly income statement
        qf = fetch_quarterly_financials(ticker, company)
        qf = run_quality_checks(qf, f"{ticker} quarterly_financials")
        save(qf, "quarterly_financials", f"{ticker.replace('&','_')}_quarterly_financials.csv")

        # Quarterly balance sheet
        bs = fetch_quarterly_balance_sheet(ticker, company)
        bs = run_quality_checks(bs, f"{ticker} balance_sheet")
        save(bs, "balance_sheet", f"{ticker.replace('&','_')}_balance_sheet.csv")

    # Combined key ratios for all companies
    if all_ratios:
        combined = pd.concat(all_ratios, ignore_index=True)
        save(combined, "key_ratios", "_all_ratios.csv")
        print(f"\nCombined ratios → {len(combined)} companies")

    print("\nBronze financials ingestion complete.\n")

if __name__ == "__main__":
    run()