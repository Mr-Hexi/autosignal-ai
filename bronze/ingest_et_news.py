import requests
import pandas as pd
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────

COMPANIES = {
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors":   "TATAMOTORS.NS",
    "Mahindra":      "M&M.NS",
    "Bajaj Auto":    "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
}

# ET RSS feeds — markets + auto sector
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/industry/auto/rssfeeds/13358575.cms",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
]

OUTPUT_DIR = "data/bronze/news"
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def tag_company(text: str) -> list:
    text_lower = text.lower()
    tags = []
    for company in COMPANIES:
        if company.lower() in text_lower:
            tags.append(company)
    return tags if tags else ["sector"]

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_rss(url: str) -> list:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = root.findall(".//item")
        print(f"  Fetched {len(items)} items from {url.split('/')[4]}")
        return items
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return []

def parse_items(items: list, feed_url: str) -> pd.DataFrame:
    rows = []
    for item in items:
        title       = item.findtext("title", "").strip()
        url         = item.findtext("link", "").strip()
        description = item.findtext("description", "").strip()
        pub_date    = item.findtext("pubDate", "").strip()

        if not url:
            continue

        companies_tagged = tag_company(f"{title} {description}")

        rows.append({
            "article_id":   make_id(url),
            "title":        title,
            "url":          url,
            "description":  description,
            "published_at": pub_date,
            "company_tags": ", ".join(companies_tagged),
            "source":       "economic_times",
            "source_quality": 0.8,  # ET quality score
            "feed_url":     feed_url,
            "ingested_at":  now(),
        })

    return pd.DataFrame(rows)

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Deduplicate by article_id (URL hash)
    dupes = df.duplicated(subset=["article_id"]).sum()
    if dupes > 0:
        print(f"  QC: {dupes} duplicate URLs — dropping")
        df = df.drop_duplicates(subset=["article_id"])

    # Drop empty titles
    empty_titles = df["title"].isnull() | (df["title"].str.strip() == "")
    if empty_titles.sum() > 0:
        print(f"  QC: {empty_titles.sum()} empty titles — dropping")
        df = df[~empty_titles]

    print(f"  QC: {original_len} rows in → {len(df)} rows out")
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
    print(f"Bronze ET News Ingestion")
    print(f"{'='*50}\n")

    all_frames = []

    for feed_url in RSS_FEEDS:
        print(f"\nFeed: {feed_url}")
        items = fetch_rss(feed_url)
        if not items:
            continue
        df = parse_items(items, feed_url)
        all_frames.append(df)
        time.sleep(1)

    if not all_frames:
        print("No data fetched.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = run_quality_checks(combined)

    save(combined, "et_news_raw.csv")
    print(f"\nTotal articles: {len(combined)}")

    # Also save per company
    for company in COMPANIES:
        company_df = combined[combined["company_tags"].str.contains(company, case=False, na=False)]
        if not company_df.empty:
            safe = company.replace(" ", "_").replace("&", "_")
            save(company_df, f"et_news_{safe}.csv")
            print(f"  {company}: {len(company_df)} articles")

    print("\nET news ingestion complete.\n")

if __name__ == "__main__":
    run()