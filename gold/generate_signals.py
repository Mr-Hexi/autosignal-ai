import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import math

# ── Config ────────────────────────────────────────────────────────────────────

SENTIMENT_FILE  = "data/gold/company_sentiment_scores.csv"
MOMENTUM_FILE   = "data/gold/sentiment_momentum.csv"
STOCKS_FILE     = "data/silver/processed_stocks.csv"
FINANCIALS_FILE = "data/silver/processed_financials.csv"
OUTPUT_DIR      = "data/gold"
OUTPUT_SIGNALS  = "data/gold/investment_signals.csv"
OUTPUT_LAG      = "data/gold/lag_correlation.csv"

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

def normalize(value, min_val, max_val, target_min=0, target_max=10):
    """Normalize value to target range."""
    if max_val == min_val:
        return (target_min + target_max) / 2
    clipped = max(min_val, min(max_val, value))
    return round(target_min + (clipped - min_val) / (max_val - min_val) * (target_max - target_min), 4)

# ── Multi-Factor Signal Engine ────────────────────────────────────────────────

def compute_composite_score(
    sentiment_score: float,
    sentiment_momentum: float,
    price_vs_ma20: float,
    rsi_14: float,
    revenue_growth: float,
    profit_margin: float,
    price_momentum_5d: float = 0.0,  # ← add this
) -> dict:
    """
    Composite signal score (0-10) from 5 factors:

    Factor weights:
    - Sentiment score      : 35%
    - Sentiment momentum   : 20%
    - Price momentum       : 20%
    - Technical (RSI)      : 15%
    - Fundamental          : 10%
    """

    # ── Factor 1: Sentiment (already 0-10) ──
    f_sentiment = sentiment_score  # 0-10

    # ── Factor 2: Sentiment Momentum (-3 to +3 typically) → normalize to 0-10 ──
    f_momentum = normalize(price_momentum_5d or 0, -0.10, 0.10, 0, 10)

    # ── Factor 3: Price Momentum (price vs MA20) → normalize to 0-10 ──
    # price_vs_ma20 ranges roughly -0.20 to +0.20
    f_price = normalize(price_vs_ma20, -0.20, 0.20, 0, 10)

    # ── Factor 4: Technical Signal (RSI) → normalize to 0-10 ──
    # RSI < 30 = oversold (opportunity) → higher score
    # RSI > 70 = overbought (risk) → lower score
    # We invert RSI slightly: sweet spot is 40-60
    if rsi_14 <= 30:
        f_technical = 7.5  # oversold — potential buy
    elif rsi_14 <= 45:
        f_technical = 6.5  # approaching oversold
    elif rsi_14 <= 55:
        f_technical = 5.0  # neutral
    elif rsi_14 <= 70:
        f_technical = 4.0  # approaching overbought
    else:
        f_technical = 2.5  # overbought — caution

    # ── Factor 5: Fundamental Score ──
    f_fundamental = 5.0  # default neutral
    try:
        if revenue_growth is not None and not math.isnan(revenue_growth):
            if revenue_growth > 20:   f_fundamental += 2.0
            elif revenue_growth > 10: f_fundamental += 1.0
            elif revenue_growth > 0:  f_fundamental += 0.5
            elif revenue_growth < -20: f_fundamental -= 2.0
            elif revenue_growth < 0:  f_fundamental -= 1.0

        if profit_margin is not None and not math.isnan(profit_margin):
            if profit_margin > 15:   f_fundamental += 1.5
            elif profit_margin > 10: f_fundamental += 0.5
            elif profit_margin < 5:  f_fundamental -= 1.0

        f_fundamental = max(0, min(10, f_fundamental))
    except Exception:
        f_fundamental = 5.0

    # ── Composite Score ──
    composite = (
        0.35 * f_sentiment +
        0.20 * f_momentum +
        0.20 * f_price +
        0.15 * f_technical +
        0.10 * f_fundamental
    )
    composite = round(max(0, min(10, composite)), 2)

    return {
        "composite_score":   composite,
        "f_sentiment":       round(f_sentiment, 2),
        "f_momentum":        round(f_momentum, 2),
        "f_price":           round(f_price, 2),
        "f_technical":       round(f_technical, 2),
        "f_fundamental":     round(f_fundamental, 2),
    }

def generate_signal(composite_score: float) -> str:
    if composite_score >= 5.5:
        return "BUY"
    elif composite_score <= 4.8:
        return "RISK_ALERT"
    else:
        return "NEUTRAL"

def build_reasoning(row: dict, factors: dict) -> str:
    parts = []
    parts.append(f"sentiment {row['sentiment_score']:.1f}/10")
    parts.append(f"momentum {row['momentum']:+.2f}")
    parts.append(f"price vs MA20 {row['price_vs_ma20']*100:.1f}%")
    parts.append(f"RSI {row['rsi_14']:.1f}")

    score = factors["composite_score"]
    if score >= 6.5:
        parts.append("composite score BUY territory")
    elif score <= 4.5:
        parts.append("composite score RISK territory")
    else:
        parts.append("composite score NEUTRAL territory")

    return "; ".join(parts)

# ── Lag Correlation ───────────────────────────────────────────────────────────

def compute_lag_correlation(stocks_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing lag correlations...")
    rows = []

    for ticker, company in TICKER_TO_COMPANY.items():
        stock = stocks_df[stocks_df["ticker"] == ticker][["date", "daily_return"]].copy()
        stock["date"] = pd.to_datetime(stock["date"])

        sent = sentiment_df[sentiment_df["company"] == company]
        if sent.empty or stock.empty:
            continue

        sentiment_score = sent["sentiment_score"].iloc[0]
        stock["sentiment"] = sentiment_score

        for lag in [1, 2, 3]:
            stock[f"return_lag{lag}"] = stock["daily_return"].shift(-lag)
            corr = stock["sentiment"].corr(stock[f"return_lag{lag}"])
            rows.append({
                "company":     company,
                "ticker":      ticker,
                "lag_days":    lag,
                "correlation": round(corr, 4) if not np.isnan(corr) else None,
                "computed_at": now(),
            })

    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Gold Signal Generation (Multi-Factor Engine)")
    print(f"{'='*50}\n")

    # Load data
    sentiment_df  = pd.read_csv(SENTIMENT_FILE)
    momentum_df   = pd.read_csv(MOMENTUM_FILE)
    stocks_df     = pd.read_csv(STOCKS_FILE, parse_dates=["date"])
    financials_df = pd.read_csv(FINANCIALS_FILE)

    print(f"  Loaded: {len(sentiment_df)} sentiment scores")

    # Latest stock per company
    latest_stocks = (
        stocks_df.sort_values("date")
                 .groupby("ticker")
                 .last()
                 .reset_index()
    )
    latest_stocks["company"] = latest_stocks["ticker"].map(TICKER_TO_COMPANY)

    # Compute 5-day price momentum per company
    momentum_5d = {}
    for ticker, company in TICKER_TO_COMPANY.items():
        ticker_data = stocks_df[stocks_df["ticker"] == ticker].sort_values("date")
        if len(ticker_data) >= 5:
            recent_return = (
                ticker_data["close"].iloc[-1] / ticker_data["close"].iloc[-5] - 1
            )
            momentum_5d[company] = round(float(recent_return), 6)
        else:
            momentum_5d[company] = 0.0

    print(f"\n  5-day price momentum:")
    for company, mom in momentum_5d.items():
        print(f"    {company}: {mom:+.2%}")

    # Merge all data
    merged = sentiment_df.copy()

    # Fix column names from momentum merge
    mom_cols = momentum_df[["company", "momentum"]].copy()
    merged = merged.merge(mom_cols, on="company", how="left")
    merged = merged.merge(
        latest_stocks[["company", "price_vs_ma20", "rsi_14", "close", "ma_20"]],
        on="company", how="left"
    )
    merged = merged.merge(
        financials_df[["company", "profitability_score", "revenue_growth_pct", "profit_margin_pct"]],
        on="company", how="left"
    )

    # Add 5d momentum
    merged["price_momentum_5d"] = merged["company"].map(momentum_5d)

    print(f"\n  Generating multi-factor signals...\n")
    print(f"  {'Company':<25} {'Sent':>6} {'5dMom':>7} {'Price':>8} {'RSI':>6} {'Comp':>6} {'Signal':<12}")
    print(f"  {'-'*78}")

    signal_rows = []

    for _, row in merged.iterrows():
        sentiment_score   = float(row.get("sentiment_score", 5) or 5)
        momentum          = float(row.get("momentum", 0) or 0)
        price_vs_ma20     = float(row.get("price_vs_ma20", 0) or 0)
        rsi_14            = float(row.get("rsi_14", 50) or 50)
        price_momentum_5d = float(row.get("price_momentum_5d", 0) or 0)
        profitability     = float(row.get("profitability_score", 5) or 5)

        revenue_growth = row.get("revenue_growth_pct")
        profit_margin  = row.get("profit_margin_pct")

        try:
            revenue_growth = float(revenue_growth) if revenue_growth is not None and not math.isnan(float(revenue_growth)) else None
        except Exception:
            revenue_growth = None

        try:
            profit_margin = float(profit_margin) if profit_margin is not None and not math.isnan(float(profit_margin)) else None
        except Exception:
            profit_margin = None

        factors = compute_composite_score(
            sentiment_score=sentiment_score,
            sentiment_momentum=momentum,
            price_vs_ma20=price_vs_ma20,
            rsi_14=rsi_14,
            revenue_growth=revenue_growth,
            profit_margin=profit_margin,
            price_momentum_5d=price_momentum_5d,
        )

        signal    = generate_signal(factors["composite_score"])
        reasoning = build_reasoning(
            {"sentiment_score": sentiment_score, "momentum": momentum,
             "price_vs_ma20": price_vs_ma20, "rsi_14": rsi_14},
            factors
        )

        print(f"  {row['company']:<25} {sentiment_score:>6.2f} {price_momentum_5d:>+7.2%} "
              f"{price_vs_ma20*100:>7.1f}% {rsi_14:>6.1f} "
              f"{factors['composite_score']:>6.2f} {signal:<12}")

        signal_rows.append({
            "company":             row["company"],
            "signal_date":         now(),
            "signal":              signal,
            "composite_score":     factors["composite_score"],
            "sentiment_score":     round(sentiment_score, 2),
            "momentum":            round(momentum, 2),
            "price_momentum_5d":   round(price_momentum_5d, 6),
            "price_vs_ma20":       round(price_vs_ma20, 4),
            "rsi_14":              round(rsi_14, 2),
            "profitability_score": round(profitability, 2),
            "f_sentiment":         factors["f_sentiment"],
            "f_momentum":          factors["f_momentum"],
            "f_price":             factors["f_price"],
            "f_technical":         factors["f_technical"],
            "f_fundamental":       factors["f_fundamental"],
            "close":               round(float(row.get("close", 0) or 0), 2),
            "ma_20":               round(float(row.get("ma_20", 0) or 0), 2),
            "reasoning":           reasoning,
        })

    signals_df = pd.DataFrame(signal_rows)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signals_df.to_csv(OUTPUT_SIGNALS, index=False)
    print(f"\n  Saved -> {OUTPUT_SIGNALS}")

    # Signal distribution
    print(f"\n  Signal Distribution:")
    for signal, count in signals_df["signal"].value_counts().items():
        print(f"    {signal}: {count}")

    # Lag correlation
    lag_df = compute_lag_correlation(stocks_df, sentiment_df)
    if not lag_df.empty:
        lag_df.to_csv(OUTPUT_LAG, index=False)

    print("\nMulti-factor signal generation complete.\n")
    
    
    

if __name__ == "__main__":
    run()