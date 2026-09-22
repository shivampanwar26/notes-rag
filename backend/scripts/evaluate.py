"""
Retrieval evaluation.

CONCEPT — what these metrics mean and why they matter.

  Hit@K: for a question with a known expected source document, did that
  document appear ANYWHERE in the top-K retrieved chunks? Hit@1 is strict
  (the single best match must be right); Hit@5 is more forgiving. If
  Hit@5 is high but Hit@1 is low, retrieval is finding the right document
  but not ranking it first — which is exactly what reranking fixes.

  Retrieval latency: wall-clock time for the whole pipeline — query
  expansion, both retrieval legs, fusion, and reranking. Turn stages off
  in .env (ENABLE_MULTI_QUERY, ENABLE_HYBRID, ENABLE_RERANKER) and re-run
  to see what each one costs and what it buys.

This measures RETRIEVAL quality only. Whether the final answer is correct
needs human judgment or an LLM-as-judge setup, not a number invented here.

USAGE
    Upload the documents named in eval_dataset.json through the running
    app first (they must actually be indexed in ChromaDB), then run:

        cd backend
        source .venv/bin/activate
        python scripts/evaluate.py

    Ollama must be running if ENABLE_MULTI_QUERY is on, since expansion
    calls llama3. Results are computed from whatever is indexed right
    now — nothing here is hard-coded.
"""

import json
import sys
import time
from pathlib import Path

# Make `app` importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.config import settings  # noqa: E402

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def run_evaluation():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    latencies = []

    print(f"Running evaluation on {len(dataset)} questions (TOP_K={settings.TOP_K})\n")
    print("=" * 70)

    for item in dataset:
        question = item["question"]
        expected = item["expected_source"]

        start = time.time()
        # Always fetch 5 so we can compute Hit@1/3/5 from one retrieval call.
        results = rag.retrieve(question, top_k=5)
        latency = time.time() - start
        latencies.append(latency)

        sources_ranked = [r["source"] for r in results]

        hit1 = expected in sources_ranked[:1]
        hit3 = expected in sources_ranked[:3]
        hit5 = expected in sources_ranked[:5]
        hits_at_1 += int(hit1)
        hits_at_3 += int(hit3)
        hits_at_5 += int(hit5)

        print(f"\nQuery: {question}")
        print(f"Expected: {expected}")
        if results:
            print("Retrieved:")
            for i, r in enumerate(results, start=1):
                print(f"  {i}. {r['source']} — Page {r['page']} (score={r['score']})")
        else:
            print("Retrieved: (nothing above RETRIEVAL_THRESHOLD)")
        print(f"Hit@1: {hit1} | Hit@3: {hit3} | Hit@5: {hit5}")
        print(f"Retrieval latency: {latency:.3f} sec")

    n = len(dataset)
    avg_latency = sum(latencies) / n if n else 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {n}")
    print(f"Hit@1: {hits_at_1}/{n} ({100 * hits_at_1 / n:.1f}%)")
    print(f"Hit@3: {hits_at_3}/{n} ({100 * hits_at_3 / n:.1f}%)")
    print(f"Hit@5: {hits_at_5}/{n} ({100 * hits_at_5 / n:.1f}%)")
    print(f"Average retrieval latency: {avg_latency:.3f} sec")


if __name__ == "__main__":
    run_evaluation()
