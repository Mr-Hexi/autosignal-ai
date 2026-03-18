import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime, timezone
import math
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

NEWS_FILE        = "data/silver/processed_news.csv"
TRANSCRIPTS_FILE = "data/silver/processed_transcripts.csv"
OUTPUT_DIR       = "data/gold"
OUTPUT_CHUNKS    = "data/gold/sentiment_chunks.csv"
OUTPUT_SCORES    = "data/gold/company_sentiment_scores.csv"
OUTPUT_MOMENTUM  = "data/gold/sentiment_momentum.csv"

LABEL_WEIGHTS = {"positive": 1, "neutral": 0, "negative": -1}
BATCH_SIZE    = 16  # process 16 chunks at a time

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def weighted_sentiment(results: list) -> float:
    """Probability-weighted sentiment score from FinBERT output."""
    total = 0.0
    for r in results:
        weight = LABEL_WEIGHTS.get(r["label"], 0)
        total += weight * r["score"]
    return total

def normalize_score(weighted: float) -> float:
    """Map [-1, 1] → [0, 10]"""
    return round((weighted + 1) / 2 * 10, 2)

# ── Load FinBERT ──────────────────────────────────────────────────────────────

def load_finbert():
    print("  Loading FinBERT...")
    return pipeline(
        "text-classification",
        model   = "ProsusAI/finbert",
        top_k   = None,
        device  = -1,  # CPU
    )

# ── Run Sentiment ─────────────────────────────────────────────────────────────

def run_sentiment(df: pd.DataFrame, finbert, text_col: str = "chunk_text") -> pd.DataFrame:
    """Run FinBERT on all chunks, return df with sentiment columns added."""

    # Filter to rows with actual text
    mask    = df[text_col].notna() & (df[text_col].str.strip() != "") & \
              (~df[text_col].astype(str).str.startswith("[ERROR"))
    valid   = df[mask].copy()
    skipped = len(df) - len(valid)

    if skipped > 0:
        print(f"  Skipping {skipped} rows with no/error text")

    print(f"  Running FinBERT on {len(valid)} chunks (batch_size={BATCH_SIZE})...")

    texts = valid[text_col].tolist()

    # Truncate to 512 tokens worth of text (~384 words) to avoid FinBERT overflow
    texts = [" ".join(t.split()[:384]) for t in texts]

    # Batch inference
    all_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        results = finbert(batch, truncation=True, max_length=512)
        all_results.extend(results)
        print(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...", end="\r")

    print()

    # Parse results
    sentiments = []
    for result in all_results:
        labels = {r["label"]: r["score"] for r in result}
        w      = weighted_sentiment(result)
        sentiments.append({
            "prob_positive":    round(labels.get("positive", 0), 4),
            "prob_neutral":     round(labels.get("neutral", 0), 4),
            "prob_negative":    round(labels.get("negative", 0), 4),
            "weighted_sentiment": round(w, 4),
            "sentiment_score":  normalize_score(w),
            "sentiment_label":  max(labels, key=labels.get),
        })

    sentiment_df = pd.DataFrame(sentiments, index=valid.index)
    result_df    = df.copy()
    for col in sentiment_df.columns:
        result_df[col] = sentiment_df[col]

    return result_df

# ── Aggregate Company Scores ──────────────────────────────────────────────────

def aggregate_company_scores(news_df: pd.DataFrame, transcript_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Aggregating company sentiment scores...")

    rows = []
    all_companies = [
        "Maruti Suzuki", "Tata Motors", "Mahindra & Mahindra",
        "Bajaj Auto", "Hero MotoCorp"
    ]

    for company in all_companies:
        # News chunks for this company
        news_mask = news_df["company_tags"].str.contains(company, case=False, na=False)
        company_news = news_df[news_mask & news_df["weighted_sentiment"].notna()]

        # Sector-level news also counts (lower weight)
        sector_news = news_df[
            (news_df["company_tags"] == "sector") &
            news_df["weighted_sentiment"].notna()
        ]

        # Transcript chunks
        trans_mask   = transcript_df["company"] == company
        company_trans = transcript_df[trans_mask & transcript_df["weighted_sentiment"].notna()]

        # Weighted average: company news (1.0) + sector news (0.3) + transcripts (1.2)
        all_scores = []

        for _, row in company_news.iterrows():
            all_scores.append(row["weighted_sentiment"] * row.get("source_quality", 0.8) * 1.0)

        for _, row in sector_news.iterrows():
            all_scores.append(row["weighted_sentiment"] * 0.3)

        for _, row in company_trans.iterrows():
            all_scores.append(row["weighted_sentiment"] * 1.2)

        if not all_scores:
            continue

        avg_weighted    = np.mean(all_scores)
        sentiment_score = normalize_score(avg_weighted)
        article_count   = len(company_news) + len(sector_news)
        transcript_count = len(company_trans)

        # Sentiment intensity = score × log(article_count + 1)
        intensity = round(sentiment_score * math.log(article_count + 1), 4)

        rows.append({
            "company":            company,
            "score_date":         datetime.now().date().isoformat(),
            "sentiment_score":    sentiment_score,
            "weighted_sentiment": round(avg_weighted, 4),
            "article_count":      article_count,
            "transcript_count":   transcript_count,
            "sentiment_intensity": intensity,
            "granularity":        "daily",
            "computed_at":        now(),
        })

    return pd.DataFrame(rows)

# ── Sentiment Momentum ────────────────────────────────────────────────────────

def compute_momentum(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum = current score - 7-day average.
    Since we only have today's data locally, we simulate with
    a slight random walk for demo. On Databricks this reads
    historical gold table rows.
    """
    print("  Computing sentiment momentum...")
    rows = []
    for _, row in scores_df.iterrows():
        # For now: momentum = 0 (no historical data yet locally)
        # On Databricks: query last 7 days from gold table
        rows.append({
            "company":        row["company"],
            "score_date":     row["score_date"],
            "sentiment_score": row["sentiment_score"],
            "ma_7d_sentiment": row["sentiment_score"],  # will fill once we have history
            "momentum":        0.0,
            "computed_at":    now(),
        })
    return pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Gold Sentiment Scoring")
    print(f"{'='*50}\n")

    # Load silver data
    print("  Loading silver data...")
    news_df        = pd.read_csv(NEWS_FILE)
    transcripts_df = pd.read_csv(TRANSCRIPTS_FILE)

    print(f"  News chunks: {len(news_df)}")
    print(f"  Transcript chunks: {len(transcripts_df[transcripts_df['has_text'] == True])}")

    # Load FinBERT
    finbert = load_finbert()

    # Run sentiment on news
    print("\n── News Sentiment ──")
    news_df = run_sentiment(news_df, finbert, text_col="chunk_text")
    save(news_df, OUTPUT_CHUNKS.replace(".csv", "_news.csv"))

    # Run sentiment on transcripts
    print("\n── Transcript Sentiment ──")
    transcripts_df = run_sentiment(transcripts_df, finbert, text_col="chunk_text")
    save(transcripts_df, OUTPUT_CHUNKS.replace(".csv", "_transcripts.csv"))

    # Aggregate company scores
    scores_df = aggregate_company_scores(news_df, transcripts_df)
    print("\n  Company Sentiment Scores:")
    print(f"  {'Company':<25} {'Score':>6} {'Articles':>9} {'Transcripts':>12} {'Intensity':>10}")
    print(f"  {'-'*65}")
    for _, row in scores_df.iterrows():
        print(f"  {row['company']:<25} {row['sentiment_score']:>6} "
              f"{row['article_count']:>9} {row['transcript_count']:>12} "
              f"{row['sentiment_intensity']:>10}")

    save(scores_df, OUTPUT_SCORES)

    # Momentum
    momentum_df = compute_momentum(scores_df)
    save(momentum_df, OUTPUT_MOMENTUM)

    print("\nGold sentiment scoring complete.\n")

if __name__ == "__main__":
    run()