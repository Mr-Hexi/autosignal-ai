import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os

# ── Config ────────────────────────────────────────────────────────────────────

SENTIMENT_FILE    = "data/gold/company_sentiment_scores.csv"
MOMENTUM_FILE     = "data/gold/sentiment_momentum.csv"
STOCKS_FILE       = "data/silver/processed_stocks.csv"
FINANCIALS_FILE   = "data/silver/processed_financials.csv"
OUTPUT_DIR        = "data/gold"
OUTPUT_SIGNALS    = "data/gold/investment_signals.csv"
OUTPUT_LAG        = "data/gold/lag_correlation.csv"

TICKER_TO_COMPANY = {
    "MARUTI.NS":     "Maruti Suzuki",
    "TMPV.NS":       "Tata Motors",
    "M&M.NS":        "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HEROMOTOCO.NS": "Hero MotoCorp",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

# ── Signal Logic ──────────────────────────────────────────────────────────────

def generate_signal(
    sentiment_score: float,
    momentum: float,
    price_vs_ma20: float,
    profitability_score: float,
) -> tuple:
    """
    Returns (signal, reasoning)

    BUY:        sentiment > 7 OR (sentiment > 6 AND momentum > 0 AND price above MA20)
    RISK_ALERT: sentiment < 4 OR momentum < -1.5 OR profitability < 4
    NEUTRAL:    everything else
    """
    reasons = []

    # BUY conditions
    strong_sentiment  = sentiment_score > 7
    good_sentiment    = sentiment_score > 6
    positive_momentum = momentum > 0
    above_ma20        = price_vs_ma20 > 0
    healthy_profits   = profitability_score >= 6

    # RISK conditions
    weak_sentiment    = sentiment_score < 4
    negative_momentum = momentum < -1.5
    poor_profits      = profitability_score < 4

    if weak_sentiment or negative_momentum or poor_profits:
        if weak_sentiment:
            reasons.append(f"low sentiment ({sentiment_score:.1f}/10)")
        if negative_momentum:
            reasons.append(f"negative momentum ({momentum:.2f})")
        if poor_profits:
            reasons.append(f"weak profitability ({profitability_score}/10)")
        return "RISK_ALERT", "; ".join(reasons)

    if strong_sentiment:
        reasons.append(f"strong sentiment ({sentiment_score:.1f}/10)")
        if positive_momentum:
            reasons.append("positive momentum")
        if above_ma20:
            reasons.append("price above MA20")
        return "BUY", "; ".join(reasons)

    if good_sentiment and positive_momentum and above_ma20 and healthy_profits:
        reasons.append(f"good sentiment ({sentiment_score:.1f}/10)")
        reasons.append("positive momentum")
        reasons.append("price above MA20")
        reasons.append(f"healthy profitability ({profitability_score}/10)")
        return "BUY", "; ".join(reasons)

    # Neutral
    reasons.append(f"sentiment {sentiment_score:.1f}/10")
    reasons.append(f"momentum {momentum:.2f}")
    reasons.append(f"price vs MA20: {price_vs_ma20:.2%}")
    return "NEUTRAL", "; ".join(reasons)

# ── Lag Correlation ───────────────────────────────────────────────────────────

def compute_lag_correlation(stocks_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each company, compute correlation between sentiment score
    and stock return lagged by 1, 2, 3 days.
    Uses daily sentiment (same score repeated) vs daily returns.
    """
    print("  Computing lag correlations...")
    rows = []

    for ticker, company in TICKER_TO_COMPANY.items():
        stock = stocks_df[stocks_df["ticker"] == ticker][["date", "daily_return"]].copy()
        stock["date"] = pd.to_datetime(stock["date"])

        # Get sentiment score for this company (single value for now)
        sent = sentiment_df[sentiment_df["company"] == company]
        if sent.empty or stock.empty:
            continue

        sentiment_score = sent["sentiment_score"].iloc[0]

        # Assign same sentiment score to all dates (will improve with daily scores later)
        stock["sentiment"] = sentiment_score

        for lag in [1, 2, 3]:
            stock[f"return_lag{lag}"] = stock["daily_return"].shift(-lag)
            corr = stock["sentiment"].corr(stock[f"return_lag{lag}"])
            rows.append({
                "company":      company,
                "ticker":       ticker,
                "lag_days":     lag,
                "correlation":  round(corr, 4) if not np.isnan(corr) else None,
                "computed_at":  now(),
            })

    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Gold Signal Generation")
    print(f"{'='*50}\n")

    # Load data
    sentiment_df    = pd.read_csv(SENTIMENT_FILE)
    momentum_df     = pd.read_csv(MOMENTUM_FILE)
    stocks_df       = pd.read_csv(STOCKS_FILE, parse_dates=["date"])
    financials_df   = pd.read_csv(FINANCIALS_FILE)

    print(f"  Loaded: {len(sentiment_df)} sentiment scores, "
          f"{len(stocks_df)} stock rows, "
          f"{len(financials_df)} financial rows")

    # Get latest stock data per company for price_vs_ma20
    latest_stocks = (
        stocks_df.sort_values("date")
                 .groupby("ticker")
                 .last()
                 .reset_index()
    )
    latest_stocks["company"] = latest_stocks["ticker"].map(TICKER_TO_COMPANY)

    # Merge all signals data
    signals_data = sentiment_df.merge(momentum_df, on="company", how="left")
    signals_data = signals_data.merge(
        latest_stocks[["company", "price_vs_ma20", "rsi_14", "close", "ma_20"]],
        on="company", how="left"
    )
    signals_data = signals_data.merge(
        financials_df[["company", "profitability_score"]],
        on="company", how="left"
    )

    # Generate signals
    print("\n  Generating investment signals...\n")
    signal_rows = []

    for _, row in signals_data.iterrows():
        sentiment_score     = float(row.get("sentiment_score", 5))
        momentum            = float(row.get("momentum", 0) or 0)
        price_vs_ma20       = float(row.get("price_vs_ma20", 0) or 0)
        profitability_score = float(row.get("profitability_score", 5) or 5)

        signal, reasoning = generate_signal(
            sentiment_score,
            momentum,
            price_vs_ma20,
            profitability_score,
        )

        signal_rows.append({
            "company":              row["company"],
            "signal_date":          now(),
            "signal":               signal,
            "sentiment_score":      round(sentiment_score, 2),
            "momentum":             round(momentum, 2),
            "price_vs_ma20":        round(price_vs_ma20, 4),
            "profitability_score":  round(profitability_score, 2),
            "rsi_14":               round(float(row.get("rsi_14", 50) or 50), 2),
            "close":                round(float(row.get("close", 0) or 0), 2),
            "ma_20":                round(float(row.get("ma_20", 0) or 0), 2),
            "reasoning":            reasoning,
        })

    signals_df = pd.DataFrame(signal_rows)

    # Print results
    print(f"  {'Company':<25} {'Signal':<12} {'Sentiment':<12} {'Momentum':<12} {'RSI':<8} Reasoning")
    print(f"  {'-'*90}")
    for _, row in signals_df.iterrows():
        print(f"  {row['company']:<25} {row['signal']:<12} "
              f"{row['sentiment_score']:<12} {row['momentum']:<12} "
              f"{row['rsi_14']:<8} {row['reasoning']}")

    # Save signals
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signals_df.to_csv(OUTPUT_SIGNALS, index=False)
    print(f"\n  Saved → {OUTPUT_SIGNALS}")

    # Lag correlation
    lag_df = compute_lag_correlation(stocks_df, sentiment_df)
    if not lag_df.empty:
        lag_df.to_csv(OUTPUT_LAG, index=False)
        print(f"  Saved → {OUTPUT_LAG}")
        print(f"\n  Lag Correlations:")
        print(f"  {'Company':<25} {'Lag 1':>8} {'Lag 2':>8} {'Lag 3':>8}")
        print(f"  {'-'*55}")
        for company in lag_df["company"].unique():
            co = lag_df[lag_df["company"] == company]
            lags = {row["lag_days"]: row["correlation"] for _, row in co.iterrows()}
            print(f"  {company:<25} "
                  f"{str(lags.get(1,'N/A')):>8} "
                  f"{str(lags.get(2,'N/A')):>8} "
                  f"{str(lags.get(3,'N/A')):>8}")

    print("\nGold signal generation complete.\n")

if __name__ == "__main__":
    run()