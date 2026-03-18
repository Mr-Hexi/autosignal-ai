import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────

NEWS_FILE        = "data/silver/processed_news.csv"
TRANSCRIPTS_FILE = "data/silver/processed_transcripts.csv"
CHROMA_DIR       = "data/vector/chroma"
METADATA_FILE    = "data/vector/embedding_metadata.json"

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE       = 32

COMPANIES = [
    "Maruti Suzuki",
    "Tata Motors",
    "Mahindra & Mahindra",
    "Bajaj Auto",
    "Hero MotoCorp",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

# ── Build Collections ─────────────────────────────────────────────────────────

def build_news_collection(client: chromadb.Client, model: SentenceTransformer) -> dict:
    print("\n  Building news collection...")
    df = pd.read_csv(NEWS_FILE)
    df = df[df["chunk_text"].notna() & (df["chunk_text"].str.strip() != "")]

    collection = client.get_or_create_collection(
        name="autosignal_news",
        metadata={"description": "ET news + NSE announcements chunks"}
    )

    # Clear existing
    existing = collection.count()
    if existing > 0:
        print(f"  Clearing {existing} existing embeddings...")
        collection.delete(where={"source": {"$in": ["economic_times", "nse_announcements"]}})

    texts    = df["chunk_text"].tolist()
    ids      = df["chunk_id"].tolist()
    metadatas = []

    for _, row in df.iterrows():
        metadatas.append({
            "company":        str(row.get("company_tags", "sector")),
            "source":         str(row.get("source", "")),
            "source_quality": float(row.get("source_quality", 0.5)),
            "published_at":   str(row.get("published_at", "")),
            "data_type":      str(row.get("data_type", "news")),
        })

    # Embed in batches
    print(f"  Embedding {len(texts)} news chunks (batch_size={BATCH_SIZE})...")
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  News collection: {collection.count()} embeddings stored")
    return {"collection": "autosignal_news", "count": collection.count()}


def build_transcript_collection(client: chromadb.Client, model: SentenceTransformer) -> dict:
    print("\n  Building transcript collection...")
    df = pd.read_csv(TRANSCRIPTS_FILE)
    df = df[
        df["has_text"].astype(str).str.lower() == "true"
    ]
    df = df[df["chunk_text"].notna() & (df["chunk_text"].str.strip() != "")]

    collection = client.get_or_create_collection(
        name="autosignal_transcripts",
        metadata={"description": "BSE earnings call transcript chunks"}
    )

    # Clear existing
    existing = collection.count()
    if existing > 0:
        print(f"  Clearing {existing} existing embeddings...")
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    texts     = df["chunk_text"].tolist()
    ids       = df["chunk_id"].tolist()
    metadatas = []

    for _, row in df.iterrows():
        metadatas.append({
            "company":      str(row.get("company", "")),
            "quarter":      str(row.get("quarter", "")),
            "filing_date":  str(row.get("filing_date", "")),
            "chunk_index":  int(row.get("chunk_index", 0)),
            "source":       "bse_transcripts",
        })

    print(f"  Embedding {len(texts)} transcript chunks (batch_size={BATCH_SIZE})...")
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  Transcript collection: {collection.count()} embeddings stored")
    return {"collection": "autosignal_transcripts", "count": collection.count()}

# ── Semantic Search Test ──────────────────────────────────────────────────────

def test_semantic_search(client: chromadb.Client, model: SentenceTransformer):
    print("\n  Testing semantic search...")

    queries = [
        ("EV strategy and electric vehicle plans",     "autosignal_transcripts"),
        ("revenue decline and margin pressure",         "autosignal_transcripts"),
        ("production capacity expansion new plant",     "autosignal_news"),
        ("quarterly earnings results profit",           "autosignal_news"),
    ]

    for query, collection_name in queries:
        collection     = client.get_collection(collection_name)
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\n  Query: '{query}'")
        print(f"  Collection: {collection_name}")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"  Result {i+1} | company: {meta.get('company', 'N/A')} | distance: {dist:.4f}")
            print(f"    Preview: {doc[:120]}...")

# ── Save Metadata ─────────────────────────────────────────────────────────────

def save_metadata(stats: list):
    os.makedirs("data/vector", exist_ok=True)
    metadata = {
        "embedding_model":  EMBEDDING_MODEL,
        "created_at":       now(),
        "chroma_dir":       CHROMA_DIR,
        "collections":      stats,
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved -> {METADATA_FILE}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Vector Embeddings Build")
    print(f"{'='*50}")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Load embedding model
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded.")

    # Initialize ChromaDB with persistent storage
    print(f"\n  Initializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Build collections
    news_stats       = build_news_collection(client, model)
    transcript_stats = build_transcript_collection(client, model)

    # Test semantic search
    test_semantic_search(client, model)

    # Save metadata
    save_metadata([news_stats, transcript_stats])

    print(f"\n{'='*50}")
    print(f"Vector build complete.")
    print(f"  News chunks:       {news_stats['count']}")
    print(f"  Transcript chunks: {transcript_stats['count']}")
    print(f"  Chroma path:       {CHROMA_DIR}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run()