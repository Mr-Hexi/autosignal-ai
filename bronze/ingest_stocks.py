import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = [
    "MARUTI.NS",
    "TMPV.NS",
    "M&M.NS",
    "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS",
    "^CNXAUTO",  # NIFTY AUTO benchmark
]

END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=365)  # 1 year

OUTPUT_DIR = "data/bronze/stocks"

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    df = df.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df["ticker"]       = ticker
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["source"]       = "yfinance"

    # Rename to match bronze schema
    df = df.rename(columns={"date": "date"})

    # Keep only schema columns
    df = df[["ticker", "date", "open", "high", "low", "close", "volume", "ingested_at", "source"]]

    return df

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    original_len = len(df)

    # Check 1: null close prices
    null_close = df["close"].isnull().sum()
    if null_close > 0:
        print(f"  QC WARNING [{ticker}]: {null_close} rows with null close price")

    # Check 2: future timestamps
    future_rows = df[pd.to_datetime(df["date"]) > datetime.now(timezone.utc).replace(tzinfo=None)]
    if not future_rows.empty:
        print(f"  QC WARNING [{ticker}]: {len(future_rows)} rows with future dates — dropping")
        df = df[pd.to_datetime(df["date"]) <= datetime.utcnow()]

    # Check 3: zero or negative prices
    invalid_price = df[df["close"] <= 0]
    if not invalid_price.empty:
        print(f"  QC WARNING [{ticker}]: {len(invalid_price)} rows with zero/negative close — dropping")
        df = df[df["close"] > 0]

    print(f"  QC [{ticker}]: {original_len} rows in → {len(df)} rows out")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, ticker: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_ticker = ticker.replace("^", "IDX_").replace("&", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_ticker}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Bronze Stock Ingestion")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"{'='*50}\n")

    all_frames = []

    for ticker in TICKERS:
        df = fetch_stock_data(ticker)
        if df.empty:
            continue
        df = run_quality_checks(df, ticker)
        save_to_csv(df, ticker)
        all_frames.append(df)

    # Save combined file too
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "_all_stocks.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined file → {combined_path}")
        print(f"Total rows: {len(combined)}")

    print("\nBronze stock ingestion complete.\n")

if __name__ == "__main__":
    run()