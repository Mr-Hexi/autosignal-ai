import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import glob

# ── Config ────────────────────────────────────────────────────────────────────

RATIOS_FILE   = "data/bronze/financials/key_ratios/_all_ratios.csv"
QF_DIR        = "data/bronze/financials/quarterly_financials"
BS_DIR        = "data/bronze/financials/balance_sheet"
OUTPUT_DIR    = "data/silver"
OUTPUT_RATIOS = "data/silver/processed_financials.csv"
OUTPUT_QF     = "data/silver/processed_quarterly_financials.csv"

COMPANY_MAP = {
    "MARUTI.NS":      "Maruti Suzuki",
    "TMPV.NS":        "Tata Motors",
    "M&M.NS":         "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "HEROMOTOCO.NS":  "Hero MotoCorp",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def safe_pct(val):
    """Convert decimal ratio to percentage, round to 2dp."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val) * 100, 2)
    except Exception:
        return None

def safe_round(val, decimals=2):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)
    except Exception:
        return None

def crore(val):
    """Convert raw INR value to crores."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val) / 1e7, 2)
    except Exception:
        return None

# ── Process Key Ratios ────────────────────────────────────────────────────────

def process_key_ratios(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Processing key ratios for {len(df)} companies...")

    processed = []
    for _, row in df.iterrows():
        ticker  = row.get("ticker", "")
        company = COMPANY_MAP.get(ticker, row.get("company", ticker))

        processed.append({
            "ticker":               ticker,
            "company":              company,
            # Valuation
            "trailing_pe":          safe_round(row.get("trailing_pe")),
            "forward_pe":           safe_round(row.get("forward_pe")),
            "price_to_book":        safe_round(row.get("price_to_book")),
            "ev_to_ebitda":         safe_round(row.get("enterprise_to_ebitda")),
            # Profitability (converted to %)
            "profit_margin_pct":    safe_pct(row.get("profit_margin")),
            "operating_margin_pct": safe_pct(row.get("operating_margin")),
            "gross_margin_pct":     safe_pct(row.get("gross_margin")),
            "roe_pct":              safe_pct(row.get("return_on_equity")),
            "roa_pct":              safe_pct(row.get("return_on_assets")),
            # Growth (converted to %)
            "revenue_growth_pct":   safe_pct(row.get("revenue_growth")),
            "earnings_growth_pct":  safe_pct(row.get("earnings_growth")),
            # Financial health
            "debt_to_equity":       safe_round(row.get("debt_to_equity")),
            "current_ratio":        safe_round(row.get("current_ratio")),
            "free_cashflow_cr":     crore(row.get("free_cashflow")),
            "total_revenue_cr":     crore(row.get("total_revenue")),
            "total_debt_cr":        crore(row.get("total_debt")),
            "market_cap_cr":        crore(row.get("market_cap")),
            # Per share
            "eps_trailing":         safe_round(row.get("eps_trailing")),
            "eps_forward":          safe_round(row.get("eps_forward")),
            "book_value":           safe_round(row.get("book_value")),
            "dividend_yield_pct":   safe_pct(row.get("dividend_yield")),
            # 52-week range
            "week52_high":          safe_round(row.get("52w_high")),
            "week52_low":           safe_round(row.get("52w_low")),
            # Profitability score (0-10) for signal generation
            "profitability_score":  compute_profitability_score(row),
            "processed_at":         now(),
        })

    return pd.DataFrame(processed)


def compute_profitability_score(row) -> float:
    """
    Simple 0-10 profitability score based on key metrics.
    Higher = more profitable/healthy company.
    """
    score = 5.0  # start neutral

    try:
        # Profit margin > 10% is good for auto sector
        pm = float(row.get("profit_margin") or 0)
        if pm > 0.15:   score += 1.5
        elif pm > 0.10: score += 1.0
        elif pm > 0.05: score += 0.5
        elif pm < 0:    score -= 1.5

        # ROE > 15% is healthy
        roe = float(row.get("return_on_equity") or 0)
        if roe > 0.20:  score += 1.0
        elif roe > 0.15: score += 0.5
        elif roe < 0.05: score -= 0.5

        # Revenue growth
        rg = float(row.get("revenue_growth") or 0)
        if rg > 0.10:   score += 1.0
        elif rg > 0.05: score += 0.5
        elif rg < 0:    score -= 1.0

        # Debt to equity — lower is better for auto
        de = float(row.get("debt_to_equity") or 0)
        if de < 0.3:    score += 0.5
        elif de > 1.0:  score -= 0.5
        elif de > 2.0:  score -= 1.0

    except Exception:
        pass

    return round(max(0.0, min(10.0, score)), 2)

# ── Process Quarterly Financials ──────────────────────────────────────────────

def process_quarterly_financials() -> pd.DataFrame:
    print("  Processing quarterly financials...")
    frames = []

    for filepath in glob.glob(os.path.join(QF_DIR, "*.csv")):
        if "_quarterly_financials" not in filepath:
            continue
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                continue

            # Standardize ticker from filename
            filename = os.path.basename(filepath)
            ticker   = filename.replace("_quarterly_financials.csv", "").replace("_", "&", 1) \
                               if "M_M" in filename else \
                       filename.replace("_quarterly_financials.csv", "")

            df["ticker"]       = ticker
            df["company"]      = COMPANY_MAP.get(ticker, ticker)
            df["processed_at"] = now()

            # Convert revenue/income columns to crores if they exist
            for col in df.columns:
                if any(kw in col for kw in ["revenue", "income", "profit", "ebitda", "expense"]):
                    df[col] = pd.to_numeric(df[col], errors="coerce").apply(crore)

            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not process {filepath}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Quarterly financials: {len(combined)} rows across {len(frames)} companies")
    return combined

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame, label: str) -> pd.DataFrame:
    null_pct = df.isnull().mean().mean() * 100
    print(f"  QC [{label}]: {len(df)} rows | {null_pct:.1f}% null overall")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Silver Financials Processing")
    print(f"{'='*50}\n")

    # Key ratios
    print("  Loading key ratios...")
    ratios_df = pd.read_csv(RATIOS_FILE)
    ratios_df = process_key_ratios(ratios_df)
    ratios_df = run_quality_checks(ratios_df, "key_ratios")

    print(f"\n  Profitability scores:")
    for _, row in ratios_df.iterrows():
        print(f"    {row['company']}: {row['profitability_score']}/10 | "
              f"Margin: {row['profit_margin_pct']}% | "
              f"ROE: {row['roe_pct']}% | "
              f"Revenue growth: {row['revenue_growth_pct']}%")

    save(ratios_df, OUTPUT_RATIOS)

    # Quarterly financials
    qf_df = process_quarterly_financials()
    if not qf_df.empty:
        qf_df = run_quality_checks(qf_df, "quarterly_financials")
        save(qf_df, OUTPUT_QF)

    print("\nSilver financials processing complete.\n")

if __name__ == "__main__":
    run()