import subprocess
import sys
import time
from datetime import datetime
import io
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ── Config ────────────────────────────────────────────────────────────────────

PIPELINE = [
    # Bronze
    ("Bronze", "Stock Ingestion",           "bronze/ingest_stocks.py"),
    ("Bronze", "Financials Ingestion",      "bronze/ingest_financials.py"),
    ("Bronze", "NSE Announcements",         "bronze/ingest_nse_announcements.py"),
    ("Bronze", "ET News Ingestion",         "bronze/ingest_et_news.py"),
    ("Bronze", "Transcripts Ingestion",     "bronze/ingest_transcripts.py"),
    # Silver
    ("Silver", "Stock Processing",          "silver/process_stocks.py"),
    ("Silver", "News Processing",           "silver/process_news.py"),
    ("Silver", "Transcript Processing",     "silver/process_transcripts.py"),
    ("Silver", "Financials Processing",     "silver/process_financials.py"),
    #VECTOR
    ("Vector", "Build Embeddings", "vectors/build_embeddings.py"),
    # Gold
    ("Gold",   "Sentiment Scoring",         "gold/sentiment_scoring.py"),
    ("Gold",   "Signal Generation",         "gold/generate_signals.py"),
    ("Gold", "Company Report Generation", "gold/generate_company_reports.py"),
    ("Gold",   "Report Generation",         "gold/generate_report.py"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header():
    print("\n" + "="*60)
    print("  AUTOSIGNAL-AI — Full Pipeline Runner")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

def print_step(layer, name, script, index, total):
    print(f"[{index}/{total}] {layer} → {name}")
    print(f"       Running: {script}")

def print_result(success, elapsed):
    status = "✓ Done" if success else "✗ FAILED"
    print(f"       {status} ({elapsed:.1f}s)\n")

def run_script(script_path: str) -> tuple:
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        if not success:
            print(f"\n  ERROR OUTPUT:\n{result.stderr[-1000:]}")
        return success, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  EXCEPTION: {e}")
        return False, elapsed

# ── Main ──────────────────────────────────────────────────────────────────────

def run(skip_bronze=False, skip_transcripts=False):
    print_header()

    total     = len(PIPELINE)
    passed    = 0
    failed    = 0
    skipped   = 0
    start_all = time.time()

    for i, (layer, name, script) in enumerate(PIPELINE, 1):

        # Skip flags
        if skip_bronze and layer == "Bronze":
            print(f"[{i}/{total}] {layer} → {name} — SKIPPED")
            skipped += 1
            continue

        if skip_transcripts and "Transcript" in name:
            print(f"[{i}/{total}] {layer} → {name} — SKIPPED (slow)")
            skipped += 1
            continue

        print_step(layer, name, script, i, total)
        success, elapsed = run_script(script)
        print_result(success, elapsed)

        if success:
            passed += 1
        else:
            failed += 1
            print(f"  Pipeline stopped at: {name}")
            print(f"  Fix the error above and re-run.\n")
            break

    total_time = time.time() - start_all

    print("="*60)
    print(f"  Pipeline Complete")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  Total time: {total_time:.1f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Usage:
    #   python main.py                  ← full pipeline
    #   python main.py --skip-bronze    ← skip bronze (use existing data)
    #   python main.py --skip-transcripts ← skip slow PDF download

    skip_bronze      = "--skip-bronze" in sys.argv
    skip_transcripts = "--skip-transcripts" in sys.argv

    run(skip_bronze=skip_bronze, skip_transcripts=skip_transcripts)