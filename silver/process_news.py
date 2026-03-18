import pandas as pd
import hashlib
from datetime import datetime, timezone
import os
import re

# ── Config ────────────────────────────────────────────────────────────────────

ET_NEWS_FILE       = "data/bronze/news/et_news_raw.csv"
NSE_ANN_FILE       = "data/bronze/nse_announcements/_all_announcements.csv"
OUTPUT_DIR         = "data/silver"
OUTPUT_FILE        = "data/silver/processed_news.csv"

MAX_CHUNK_TOKENS   = 512   # FinBERT max input
WORDS_PER_TOKEN    = 0.75  # approx: 1 token ≈ 0.75 words
MAX_CHUNK_WORDS    = int(MAX_CHUNK_TOKENS * WORDS_PER_TOKEN)  # ~384 words

SOURCE_QUALITY = {
    "reuters":          1.0,
    "bloomberg":        1.0,
    "economic_times":   0.8,
    "business standard":0.8,
    "livemint":         0.75,
    "moneycontrol":     0.7,
    "nse_announcements":0.9,  # official exchange filings = high quality
    "default":          0.4,
}

COMPANY_KEYWORDS = {
    "Maruti Suzuki":        ["maruti", "suzuki", "msil"],
    "Tata Motors":          ["tata motors", "tatamotors", "jaguar", "jlr", "tata ev"],
    "Mahindra & Mahindra":  ["mahindra", "m&m", "scorpio", "thar", "xuv"],
    "Bajaj Auto":           ["bajaj", "bajaj auto", "pulsar", "chetak"],
    "Hero MotoCorp":        ["hero motocorp", "hero moto", "splendor", "hero honda"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def make_chunk_id(text: str, index: int) -> str:
    return hashlib.md5(f"{text[:50]}_{index}".encode()).hexdigest()

def get_source_quality(source: str) -> float:
    source_lower = source.lower()
    for key, score in SOURCE_QUALITY.items():
        if key in source_lower:
            return score
    return SOURCE_QUALITY["default"]

def tag_companies(text: str) -> str:
    text_lower = text.lower()
    tags = []
    for company, keywords in COMPANY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(company)
    return ", ".join(tags) if tags else "sector"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove special chars except punctuation
    text = re.sub(r"[^\w\s.,!?;:()\-']", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()

def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list:
    """Split text into chunks of max_words words."""
    if not text:
        return []
    words  = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 20:  # skip tiny chunks
            chunks.append(chunk)
    return chunks

# ── Load & Normalize ET News ──────────────────────────────────────────────────

def load_et_news() -> pd.DataFrame:
    print("  Loading ET news...")
    if not os.path.exists(ET_NEWS_FILE):
        print("  WARNING: ET news file not found")
        return pd.DataFrame()

    df = pd.read_csv(ET_NEWS_FILE)
    df["text"]           = (df["title"].fillna("") + ". " + df["description"].fillna(""))
    df["source"]         = "economic_times"
    df["source_quality"] = 0.8
    df["data_type"]      = "news"

    return df[["article_id", "text", "source", "source_quality",
               "published_at", "company_tags", "data_type"]].copy()

# ── Load & Normalize NSE Announcements ───────────────────────────────────────

def load_nse_announcements() -> pd.DataFrame:
    print("  Loading NSE announcements...")
    if not os.path.exists(NSE_ANN_FILE):
        print("  WARNING: NSE announcements file not found")
        return pd.DataFrame()

    df = pd.read_csv(NSE_ANN_FILE)
    df["text"]           = (df["subject"].fillna("") + ". " + df["description"].fillna(""))
    df["source"]         = "nse_announcements"
    df["source_quality"] = 0.9
    df["published_at"]   = df["broadcast_date"]
    df["data_type"]      = "announcement"

    # Re-tag companies from text
    df["company_tags"] = df["text"].apply(tag_companies)

    return df[["article_id", "text", "source", "source_quality",
               "published_at", "company_tags", "data_type"]].copy()

# ── Process & Chunk ───────────────────────────────────────────────────────────

def process_and_chunk(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Chunking {len(df)} articles...")
    rows = []

    for _, row in df.iterrows():
        cleaned = clean_text(row["text"])
        if not cleaned:
            continue

        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id":       make_chunk_id(chunk, i),
                "article_id":     row["article_id"],
                "chunk_text":     chunk,
                "chunk_index":    i,
                "chunk_words":    len(chunk.split()),
                "source":         row["source"],
                "source_quality": row["source_quality"],
                "published_at":   row["published_at"],
                "company_tags":   row["company_tags"],
                "data_type":      row["data_type"],
                "processed_at":   now(),
            })

    return pd.DataFrame(rows)

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Drop duplicate chunks
    dupes = df.duplicated(subset=["chunk_id"]).sum()
    if dupes > 0:
        print(f"  QC: {dupes} duplicate chunks dropped")
        df = df.drop_duplicates(subset=["chunk_id"])

    # Drop very short chunks
    short = df["chunk_words"] < 10
    if short.sum() > 0:
        print(f"  QC: {short.sum()} chunks under 10 words dropped")
        df = df[~short]

    print(f"  QC: {original_len} chunks in → {len(df)} chunks out")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved → {OUTPUT_FILE}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Silver News Processing")
    print(f"{'='*50}\n")

    # Load sources
    et_df  = load_et_news()
    nse_df = load_nse_announcements()

    # Combine
    frames = [f for f in [et_df, nse_df] if not f.empty]
    if not frames:
        print("No news data found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined)} articles total")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")

    # Deduplicate on article_id
    dupes = combined.duplicated(subset=["article_id"]).sum()
    if dupes > 0:
        print(f"  Deduplicating {dupes} duplicate article IDs...")
        combined = combined.drop_duplicates(subset=["article_id"])

    # Process and chunk
    chunked = process_and_chunk(combined)

    # Quality checks
    print("\n  Running quality checks...")
    chunked = run_quality_checks(chunked)

    # Summary
    print(f"\n  Company tag distribution:")
    for tag, count in chunked["company_tags"].value_counts().items():
        print(f"    {tag}: {count} chunks")

    print(f"\n  Source quality distribution:")
    for source, grp in chunked.groupby("source"):
        print(f"    {source}: {len(grp)} chunks | quality: {grp['source_quality'].iloc[0]}")

    print(f"\n  Total chunks: {len(chunked)}")
    save(chunked)
    print("\nSilver news processing complete.\n")

if __name__ == "__main__":
    run()