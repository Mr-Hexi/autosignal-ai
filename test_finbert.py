# save as test_finbert.py in root folder
from transformers import pipeline

print("Loading FinBERT...")
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None  # return all 3 labels with probabilities
)

test_texts = [
    "Maruti Suzuki reports strong quarterly earnings with record revenue growth.",
    "Tata Motors faces headwinds as EV losses widen and margins compress.",
    "Bajaj Auto maintains stable market share amid competitive pressures.",
]

print("Running inference...\n")
for text in test_texts:
    results = finbert(text)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results_sorted[0]
    print(f"Text: {text[:60]}...")
    for r in results_sorted:
        print(f"  {r['label']:10} {r['score']:.4f}")
    print(f"  → Top: {top['label']} ({top['score']:.4f})\n")