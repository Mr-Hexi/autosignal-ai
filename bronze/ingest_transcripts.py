from bse import BSE
import pandas as pd
import pdfplumber
import requests
import hashlib
import io
from datetime import datetime, timezone, timedelta
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────

COMPANIES = {
    "Maruti Suzuki":       "532500",
    "Tata Motors":         "500570",
    "Mahindra & Mahindra": "500520",
    "Bajaj Auto":          "532977",
    "Hero MotoCorp":       "500182",
}

OUTPUT_DIR  = "data/bronze/transcripts"
COOKIE_DIR  = "data/bse_cookies"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}

CONCALL_KEYWORDS = [
    "transcript", "concall", "con call", "earnings call",
    "investor meet", "analyst meet", "conference call", "investor presentation"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def make_id(company: str, text: str) -> str:
    return hashlib.md5(f"{company}_{text}".encode()).hexdigest()

def is_concall(text: str) -> bool:
    return any(kw in text.lower() for kw in CONCALL_KEYWORDS)

# ── Fetch Announcements ───────────────────────────────────────────────────────

def fetch_concalls(bse: BSE, scrip_code: str, company: str) -> pd.DataFrame:
    print(f"  Fetching: {company} ({scrip_code})...")

    from_date = datetime.now() - timedelta(days=365)  # 1 year back for transcripts
    to_date   = datetime.now()

    try:
        # Fetch all announcements — filter by keyword after
        data = bse.announcements(
            scripcode = scrip_code,
            from_date = from_date,
            to_date   = to_date,
        )
    except Exception as e:
        print(f"  ERROR [{company}]: {e}")
        return pd.DataFrame()

    if not data:
        print(f"  WARNING [{company}]: No announcements returned")
        return pd.DataFrame()

    print(f"  Total announcements: {len(data)} — filtering for concalls...")

    rows = []
    for item in data.get("Table", []):
        # Check headline and subject for concall keywords
        headline = str(item.get("NEWSSUB", "") or "")
        subject  = str(item.get("ANNOUNCEMENT_TYPE", "") or "")
        combined = f"{headline} {subject}".lower()
        if not is_concall(combined):
            continue

        attachment = item.get("ATTACHMENTNAME", "") or item.get("XML_NAME", "") or ""
        pdf_url = (
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{attachment}.pdf"
            if attachment else ""
        )

        rows.append({
            "transcript_id": make_id(company, headline + item.get("NEWS_DT", "")),
            "company":       company,
            "scrip_code":    scrip_code,
            "title":         headline,
            "subject":       subject,
            "document_url":  pdf_url,
            "filing_date":   item.get("NEWS_DT", ""),
            "category":      item.get("ANNOUNCEMENT_TYPE", ""),
            "source":        "bse_filings",
            "ingested_at":   now(),
            "raw_text":      "",
        })

    print(f"  Found {len(rows)} concall filings for {company}")
    return pd.DataFrame(rows)

# ── Extract PDF Text ──────────────────────────────────────────────────────────

def extract_pdf_text(pdf_url: str, max_pages: int = 8) -> str:
    if not pdf_url:
        return ""
    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        text_parts = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text()
                if text:
                    text_parts.append(text.strip())
        full_text = "\n".join(text_parts)
        return " ".join(full_text.split())
    except Exception as e:
        return f"[ERROR: {e}]"

# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame, company: str) -> pd.DataFrame:
    if df.empty:
        return df
    original_len = len(df)
    df = df.drop_duplicates(subset=["transcript_id"])
    empty = df["title"].isnull() | (df["title"].str.strip() == "")
    df = df[~empty]
    print(f"  QC [{company}]: {original_len} → {len(df)} rows")
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
    print(f"Bronze Transcripts Ingestion (BSE Filings)")
    print(f"{'='*50}\n")

    os.makedirs(COOKIE_DIR, exist_ok=True)
    all_frames = []

    with BSE(download_folder=COOKIE_DIR) as bse:
        for company, scrip_code in COMPANIES.items():
            df = fetch_concalls(bse, scrip_code, company)
            if df.empty:
                continue

            df = run_quality_checks(df, company)
            if df.empty:
                continue

            # Extract text from first 2 PDFs per company
            pdf_rows = df[df["document_url"] != ""].head(2)
            if not pdf_rows.empty:
                print(f"  Extracting text from {len(pdf_rows)} PDFs...")
                for idx, row in pdf_rows.iterrows():
                    text = extract_pdf_text(row["document_url"])
                    df.at[idx, "raw_text"] = text
                    time.sleep(1)

            safe = company.replace(" ", "_").replace("&", "_")
            save(df, f"{safe}_transcripts.csv")
            all_frames.append(df)
            time.sleep(2)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        save(combined, "_all_transcripts.csv")
        print(f"\nTotal concall filings: {len(combined)}")
    else:
        print("\nNo concall filings found across all companies.")

    print("\nTranscripts ingestion complete.\n")

if __name__ == "__main__":
    run()