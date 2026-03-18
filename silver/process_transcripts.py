import pandas as pd
import hashlib
import re
from datetime import datetime, timezone
import os

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "data/bronze/transcripts/_all_transcripts.csv"
OUTPUT_DIR  = "data/silver"
OUTPUT_FILE = "data/silver/processed_transcripts.csv"

MAX_CHUNK_WORDS = 384  # ~512 tokens for FinBERT

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def make_chunk_id(transcript_id: str, index: int) -> str:
    return hashlib.md5(f"{transcript_id}_{index}".encode()).hexdigest()

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    if text.startswith("[ERROR") or text.startswith("[PDF"):
        return ""
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()

def extract_quarter(title: str, filing_date: str) -> str:
    """Try to extract quarter from title or derive from filing date."""
    if isinstance(title, str):
        # Match patterns like Q3FY25, Q3 FY25, Q3 FY 2025
        match = re.search(r"Q[1-4]\s*FY\s*\d{2,4}", title, re.IGNORECASE)
        if match:
            return match.group(0).replace(" ", "").upper()

    # Derive from filing date
    try:
        date = pd.to_datetime(filing_date)
        month = date.month
        year  = date.year
        fy    = year if month >= 4 else year - 1
        quarter_map = {4: "Q1", 5: "Q1", 6: "Q1",
                       7: "Q2", 8: "Q2", 9: "Q2",
                       10: "Q3", 11: "Q3", 12: "Q3",
                       1: "Q4", 2: "Q4", 3: "Q4"}
        q = quarter_map.get(month, "Q?")
        return f"{q}FY{str(fy)[2:]}"
    except Exception:
        return "Unknown"

def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list:
    if not text:
        return []
    words  = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks

# ── Process Transcripts ───────────────────────────────────────────────────────

def process_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Processing {len(df)} transcript filings...")
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        cleaned = clean_text(str(row.get("raw_text", "")))

        if not cleaned:
            skipped += 1
            # Still store metadata even without text — useful for tracking
            rows.append({
                "chunk_id":       make_chunk_id(str(row.get("transcript_id", "")), 0),
                "transcript_id":  row.get("transcript_id", ""),
                "company":        row.get("company", ""),
                "quarter":        extract_quarter(
                                    str(row.get("title", "")),
                                    str(row.get("filing_date", ""))
                                  ),
                "title":          row.get("title", ""),
                "filing_date":    row.get("filing_date", ""),
                "document_url":   row.get("document_url", ""),
                "chunk_text":     "",
                "chunk_index":    0,
                "chunk_words":    0,
                "has_text":       False,
                "source":         row.get("source", "bse_filings"),
                "processed_at":   now(),
            })
            continue

        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id":       make_chunk_id(str(row.get("transcript_id", "")), i),
                "transcript_id":  row.get("transcript_id", ""),
                "company":        row.get("company", ""),
                "quarter":        extract_quarter(
                                    str(row.get("title", "")),
                                    str(row.get("filing_date", ""))
                                  ),
                "title":          row.get("title", ""),
                "filing_date":    row.get("filing_date", ""),
                "document_url":   row.get("document_url", ""),
                "chunk_text":     chunk,
                "chunk_index":    i,
                "chunk_words":    len(chunk.split()),
                "has_text":       True,
                "source":         row.get("source", "bse_filings"),
                "processed_at":   now(),
            })

    print(f"  Skipped {skipped} filings with no extracted text (PDF-only)")
    return pd.DataFrame(rows)

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    dupes = df.duplicated(subset=["chunk_id"]).sum()
    if dupes > 0:
        print(f"  QC: {dupes} duplicate chunks dropped")
        df = df.drop_duplicates(subset=["chunk_id"])

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
    print(f"Silver Transcript Processing")
    print(f"{'='*50}\n")

    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} transcript filings")
    print(f"  Companies: {df['company'].unique().tolist()}")

    chunked = process_transcripts(df)
    chunked = run_quality_checks(chunked)

    # Summary
    with_text    = chunked[chunked["has_text"] == True]
    without_text = chunked[chunked["has_text"] == False]

    print(f"\n  Chunks with text:    {len(with_text)}")
    print(f"  Filings without text (PDF only): {len(without_text)}")

    print(f"\n  Per company:")
    for company, grp in chunked[chunked["has_text"]].groupby("company"):
        quarters = grp["quarter"].unique().tolist()
        print(f"    {company}: {len(grp)} chunks | quarters: {quarters}")

    save(chunked)
    print("\nSilver transcript processing complete.\n")

if __name__ == "__main__":
    run()