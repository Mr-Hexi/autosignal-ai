from nse import NSE
import pandas as pd
import hashlib
from datetime import datetime, timezone, timedelta
import os

# ── Config ────────────────────────────────────────────────────────────────────

COMPANIES = {
    "MARUTI":     "Maruti Suzuki",
    "TMPV":       "Tata Motors",      # ← fixed
    "M&M":        "Mahindra & Mahindra",
    "BAJAJ-AUTO": "Bajaj Auto",
    "HEROMOTOCO": "Hero MotoCorp",
}

OUTPUT_DIR  = "data/bronze/nse_announcements"
COOKIE_DIR  = "data/nse_cookies"  # nse package stores session cookies here

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def make_id(symbol: str, subject: str, date: str) -> str:
    return hashlib.md5(f"{symbol}_{subject}_{date}".encode()).hexdigest()

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_announcements(nse: NSE, symbol: str, company: str) -> pd.DataFrame:
    print(f"  Fetching announcements: {symbol}...")
    try:
        from_date = datetime.now() - timedelta(days=90)
        to_date   = datetime.now()

        data = nse.announcements(
            index     = "equities",
            symbol    = symbol,
            from_date = from_date,
            to_date   = to_date,
        )
        
    except Exception as e:
        print(f"  ERROR [{symbol}]: {e}")
        return pd.DataFrame()

    if not data:
        print(f"  WARNING [{symbol}]: No announcements found")
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "article_id":      make_id(symbol, item.get("desc", ""), item.get("exchdisstime", "")),
            "ticker":          symbol,
            "company":         company,
            "subject":         item.get("desc", ""),           # ← was "subject", now "desc"
            "description":     item.get("attchmntText", ""),   # ← full text, more useful
            "category":        item.get("desc", ""),           # announcement type
            "attachment_url":  item.get("attchmntFile", ""),
            "broadcast_date":  item.get("exchdisstime", ""),
            "sort_date":       item.get("sort_date", ""),
            "company_name":    item.get("sm_name", ""),
            "industry":        item.get("smIndustry", ""),
            "exchange":        "NSE",
            "source":          "nse_announcements",
            "ingested_at":     now(),
        })

    df = pd.DataFrame(rows)
    print(f"  Got {len(df)} announcements for {symbol}")
    return df

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    original_len = len(df)

    dupes = df.duplicated(subset=["article_id"]).sum()
    if dupes > 0:
        print(f"  QC WARNING [{symbol}]: {dupes} duplicates — dropping")
        df = df.drop_duplicates(subset=["article_id"])

    empty_subject = df["subject"].isnull() | (df["subject"].str.strip() == "")
    if empty_subject.sum() > 0:
        print(f"  QC WARNING [{symbol}]: {empty_subject.sum()} empty subjects — dropping")
        df = df[~empty_subject]

    print(f"  QC [{symbol}]: {original_len} rows in → {len(df)} rows out")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Bronze NSE Announcements Ingestion")
    print(f"{'='*50}\n")

    os.makedirs(COOKIE_DIR, exist_ok=True)
    all_frames = []

    with NSE(download_folder=COOKIE_DIR) as nse:
        for symbol, company in COMPANIES.items():
            df = fetch_announcements(nse, symbol, company)
            if df.empty:
                continue
            df = run_quality_checks(df, symbol)
            safe_name = symbol.replace("&", "_")
            save(df, f"{safe_name}_announcements.csv")
            all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        save(combined, "_all_announcements.csv")
        print(f"\nTotal announcements: {len(combined)}")

    print("\nNSE announcements ingestion complete.\n")

if __name__ == "__main__":
    run()