import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "data/bronze/stocks/_all_stocks.csv"
INDEX_FILE  = "data/bronze/stocks/IDX_CNXAUTO.csv"
OUTPUT_DIR  = "data/silver"
OUTPUT_FILE = "data/silver/processed_stocks.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Daily return
    df["daily_return"] = df["close"].pct_change().round(6)

    # Moving averages
    df["ma_5"]  = df["close"].rolling(window=5,  min_periods=1).mean().round(2)
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean().round(2)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"])

    # Rolling 20-day volatility (std of daily returns)
    df["volatility_20d"] = df["daily_return"].rolling(window=20, min_periods=5).std().round(6)

    # Price vs MA20 (how far price is from 20-day MA)
    df["price_vs_ma20"] = ((df["close"] - df["ma_20"]) / df["ma_20"]).round(6)

    df["processed_at"] = now()
    return df


def add_relative_return(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Add return relative to NIFTY AUTO index."""

    # Compute index daily return
    index_df = index_df.sort_values("date")[["date", "close"]].copy()
    index_df["index_return"] = index_df["close"].pct_change().round(6)
    index_df = index_df[["date", "index_return"]]

    # Merge and compute relative return
    df = df.merge(index_df, on="date", how="left")
    df["rel_return_vs_index"] = (df["daily_return"] - df["index_return"]).round(6)
    df = df.drop(columns=["index_return"])

    return df

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Drop rows where close is null
    null_close = df["close"].isnull().sum()
    if null_close > 0:
        print(f"  QC: Dropping {null_close} rows with null close price")
        df = df.dropna(subset=["close"])

    # Drop duplicates on ticker + date
    dupes = df.duplicated(subset=["ticker", "date"]).sum()
    if dupes > 0:
        print(f"  QC: Dropping {dupes} duplicate ticker+date rows")
        df = df.drop_duplicates(subset=["ticker", "date"])

    print(f"  QC: {original_len} rows in → {len(df)} rows out")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved → {OUTPUT_FILE}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Silver Stock Processing")
    print(f"{'='*50}\n")

    # Load bronze data
    print("  Loading bronze stock data...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    # Load index data
    print("  Loading NIFTY AUTO index data...")
    index_df = pd.read_csv(INDEX_FILE, parse_dates=["date"])

    # Separate index from company stocks
    company_df = df[df["ticker"] != "^CNXAUTO"].copy()
    print(f"  Tickers: {company_df['ticker'].unique().tolist()}")
    print(f"  Date range: {company_df['date'].min().date()} → {company_df['date'].max().date()}")

    # Quality checks
    print("\n  Running quality checks...")
    company_df = run_quality_checks(company_df)

    # Process each ticker separately (features are per-ticker)
    print("\n  Computing features per ticker...")
    processed_frames = []

    for ticker in company_df["ticker"].unique():
        ticker_df = company_df[company_df["ticker"] == ticker].copy()
        ticker_df = add_features(ticker_df)
        processed_frames.append(ticker_df)
        print(f"  ✓ {ticker}: {len(ticker_df)} rows | "
              f"RSI range: {ticker_df['rsi_14'].min():.1f} – {ticker_df['rsi_14'].max():.1f} | "
              f"Volatility avg: {ticker_df['volatility_20d'].mean():.4f}")

    combined = pd.concat(processed_frames, ignore_index=True)

    # Add relative return vs NIFTY AUTO
    print("\n  Adding relative return vs NIFTY AUTO index...")
    combined = add_relative_return(combined, index_df)

    # Final sort
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Summary
    print(f"\n  Features added: daily_return, ma_5, ma_20, rsi_14, volatility_20d, price_vs_ma20, rel_return_vs_index")
    print(f"  Total rows: {len(combined)}")
    print(f"  Columns: {list(combined.columns)}")

    save(combined)
    print("\nSilver stock processing complete.\n")

if __name__ == "__main__":
    run()