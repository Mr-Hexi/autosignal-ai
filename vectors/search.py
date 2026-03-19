import chromadb
from sentence_transformers import SentenceTransformer
import os

CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "vector", "chroma")
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client = None
_model  = None

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def search(
    query: str,
    collection: str = "autosignal_transcripts",
    company: str = None,
    n_results: int = 5,
) -> list:
    """
    Semantic search across news or transcript chunks.

    Args:
        query:      Natural language query
        collection: 'autosignal_news' or 'autosignal_transcripts'
        company:    Optional company filter
        n_results:  Number of results to return

    Returns:
        List of dicts with text, metadata, distance
    """
    client     = get_client()
    model      = get_model()
    collection = client.get_collection(collection)

    query_embedding = model.encode([query]).tolist()

    where = {"company": {"$eq": company}} if company else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4),
            "score":    round(1 - dist, 4),  # similarity score
        })

    return output


if __name__ == "__main__":
    # Quick test
    results = search("EV launch electric vehicle strategy", company=None, n_results=3)
    for r in results:
        print(f"Score: {r['score']} | Company: {r['metadata'].get('company')}")
        print(f"  {r['text'][:150]}...")
        print()