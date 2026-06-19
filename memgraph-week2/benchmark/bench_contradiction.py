"""
Benchmark 3: Contradiction Detection Accuracy

Measures whether the system correctly identifies contradictions AND
avoids false positives. This is the most unique feature of MemGraph —
if it flags everything as a contradiction, it's useless; if it misses
real ones, the knowledge graph becomes silently inconsistent.

Metrics:
  - Precision: % of flagged contradictions that are real
  - Recall: % of real contradictions that were flagged
  - F1 Score: harmonic mean (balances precision and recall)

Run: python bench_contradiction.py
Prereqs: Neo4j + Qdrant running, GROQ_API_KEY + VOYAGE_API_KEY set
"""

import os
import sys
import time

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1")
WEEK2_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, WEEK1_DIR)
sys.path.insert(0, WEEK2_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(WEEK1_DIR, ".env"))

from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("pipeline", os.path.join(WEEK1_DIR, "04_pipeline.py"))
_pipeline = module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
process_message = _pipeline.process_message
setup_qdrant = _pipeline.setup_qdrant

from core.contradict import check_contradiction
from bench_data import SEED_MESSAGES, CONTRADICTION_TESTS

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COLLECTION_NAME = "memgraph_messages"


def setup_graph(neo4j_driver, qdrant, voyage_client):
    """Ingest all 30 seed messages to build the knowledge graph."""
    print("Ingesting seed data for contradiction testing...\n")

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    setup_qdrant(qdrant)

    for i, msg in enumerate(SEED_MESSAGES):
        if i > 0:
            time.sleep(21)
        print(f"  [{i+1}/{len(SEED_MESSAGES)}] {msg[:55]}...")
        process_message(msg, neo4j_driver, qdrant, voyage_client)

    print(f"\nIngested {len(SEED_MESSAGES)} messages.\n")


def run_contradiction_benchmark(neo4j_driver=None, qdrant=None, voyage_client=None,
                                 skip_ingestion=False) -> dict:
    """Run all contradiction test cases and compute precision/recall/F1."""
    close_driver = False
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        close_driver = True
    if qdrant is None:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if voyage_client is None:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    if not skip_ingestion:
        setup_graph(neo4j_driver, qdrant, voyage_client)

    print("=" * 70)
    print("BENCHMARK 3: CONTRADICTION DETECTION")
    print("=" * 70)
    print(f"Testing {len(CONTRADICTION_TESTS)} cases\n")

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    results = []

    for i, test in enumerate(CONTRADICTION_TESTS):
        msg = test["test_message"]
        expected = test["should_contradict"]
        category = test["category"]

        print(f"\n  [{i+1}/{len(CONTRADICTION_TESTS)}] \"{msg[:60]}...\"")
        print(f"    Expected: {'CONTRADICT' if expected else 'NO CONTRADICTION'} ({category})")

        try:
            contradictions = check_contradiction(msg, neo4j_driver=neo4j_driver)
            detected = len(contradictions) > 0
        except Exception as e:
            print(f"    ERROR: {e}")
            detected = False
            contradictions = []

        if detected and expected:
            true_positives += 1
            verdict = "TP (correct)"
        elif detected and not expected:
            false_positives += 1
            verdict = "FP (false alarm)"
        elif not detected and expected:
            false_negatives += 1
            verdict = "FN (missed)"
        else:
            true_negatives += 1
            verdict = "TN (correct)"

        print(f"    Detected: {'YES' if detected else 'NO'} → {verdict}")
        if contradictions:
            for c in contradictions:
                print(f"      Against: \"{c['existing'][:60]}...\"")
                print(f"      Reason: {c['explanation']}")

        results.append({
            "test_message": msg,
            "expected": expected,
            "detected": detected,
            "verdict": verdict,
            "category": category,
            "contradictions": contradictions,
        })

        time.sleep(1)

    # ── Compute metrics ──
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-category breakdown
    categories = sorted(set(t["category"] for t in CONTRADICTION_TESTS))
    category_stats = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        correct = sum(1 for r in cat_results if (r["detected"] == r["expected"]))
        category_stats[cat] = {
            "total": len(cat_results),
            "correct": correct,
            "accuracy": correct / len(cat_results) if cat_results else 0,
        }

    # ── Print results ──
    print(f"\n{'═' * 70}")
    print("CONTRADICTION DETECTION RESULTS")
    print(f"{'═' * 70}")

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted YES   Predicted NO")
    print(f"    Actual YES      TP = {true_positives:<12} FN = {false_negatives}")
    print(f"    Actual NO       FP = {false_positives:<12} TN = {true_negatives}")

    print(f"\n  Metrics:")
    print(f"    Precision: {precision:.1%}  (of flagged contradictions, how many are real)")
    print(f"    Recall:    {recall:.1%}  (of real contradictions, how many were caught)")
    print(f"    F1 Score:  {f1:.1%}  (harmonic mean — balances precision and recall)")

    print(f"\n  Per Category:")
    print(f"  {'Category':<25} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}")
    print(f"  {'─' * 25}-+-{'─' * 7}-+-{'─' * 5}-+-{'─' * 8}")
    for cat, stats in sorted(category_stats.items()):
        print(f"  {cat:<25} | {stats['correct']:>7} | {stats['total']:>5} | {stats['accuracy']:>8.0%}")

    # Pass/fail
    passed = f1 >= 0.60
    print(f"\n  {'─' * 50}")
    if passed:
        print(f"  PASS — F1 score {f1:.1%} meets threshold (>= 60%)")
    else:
        print(f"  FAIL — F1 score {f1:.1%} below threshold (60%)")
        if precision < recall:
            print(f"  Too many false positives — consider raising CONFIDENCE_THRESHOLD")
        else:
            print(f"  Missing real contradictions — check the NLI prompt for edge cases")

    if close_driver:
        neo4j_driver.close()

    return {
        "benchmark": "contradiction",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "category_stats": category_stats,
        "passed": passed,
        "details": results,
    }


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY in memgraph-week1/.env")
        sys.exit(1)
    if not os.getenv("VOYAGE_API_KEY"):
        print("ERROR: Set VOYAGE_API_KEY in memgraph-week1/.env")
        sys.exit(1)
    run_contradiction_benchmark()
