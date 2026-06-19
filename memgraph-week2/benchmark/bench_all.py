"""
Full Benchmark Suite — Runs All 4 Benchmarks and Prints Summary

This is the one script to run for a complete system evaluation.
It orchestrates: extraction, retrieval, contradiction, and generation
benchmarks, then prints a single pass/fail summary table.

Order matters:
  1. Extraction — standalone, no graph/vector needed
  2. Retrieval — ingests seed data (shared with generation)
  3. Contradiction — needs its own fresh graph (contradicting messages pollute)
  4. Generation — re-ingests seed data for clean state

Run: python bench_all.py
Prereqs: Neo4j + Qdrant running, GROQ_API_KEY + VOYAGE_API_KEY set
Time: ~45-60 minutes (dominated by Voyage AI rate limits at 3 RPM)
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

from bench_extraction import run_extraction_benchmark
from bench_retrieval import run_retrieval_benchmark
from bench_contradiction import run_contradiction_benchmark
from bench_generation import run_generation_benchmark

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def main():
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY in memgraph-week1/.env")
        return
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in memgraph-week1/.env")
        return

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print("Connected to all services.\n")

    start_time = time.time()
    all_results = {}

    # ── Benchmark 1: Extraction (no graph/vector needed) ──
    print(f"\n{'█' * 70}")
    print(f"  RUNNING BENCHMARK 1 OF 4: EXTRACTION")
    print(f"{'█' * 70}\n")
    all_results["extraction"] = run_extraction_benchmark()

    # ── Benchmark 2: Retrieval (ingests seed data) ──
    print(f"\n\n{'█' * 70}")
    print(f"  RUNNING BENCHMARK 2 OF 4: RETRIEVAL")
    print(f"{'█' * 70}\n")
    all_results["retrieval"] = run_retrieval_benchmark(
        neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client,
        skip_ingestion=False,
    )

    # ── Benchmark 3: Contradiction (needs fresh graph) ──
    print(f"\n\n{'█' * 70}")
    print(f"  RUNNING BENCHMARK 3 OF 4: CONTRADICTION DETECTION")
    print(f"{'█' * 70}\n")
    all_results["contradiction"] = run_contradiction_benchmark(
        neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client,
        skip_ingestion=False,
    )

    # ── Benchmark 4: Generation (needs clean seed data — re-ingest) ──
    print(f"\n\n{'█' * 70}")
    print(f"  RUNNING BENCHMARK 4 OF 4: GENERATION QUALITY")
    print(f"{'█' * 70}\n")
    # Re-ingest clean seed data since contradiction benchmark added polluting messages
    from bench_retrieval import ingest_seed_data
    ingest_seed_data(neo4j_driver, qdrant, voyage_client)
    all_results["generation"] = run_generation_benchmark(
        neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client,
    )

    elapsed = time.time() - start_time
    neo4j_driver.close()

    # ── Final Summary ──
    print(f"\n\n{'═' * 70}")
    print(f"  BENCHMARK SUITE SUMMARY")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"{'═' * 70}\n")

    ext = all_results["extraction"]
    ret = all_results["retrieval"]
    con = all_results["contradiction"]
    gen = all_results["generation"]

    mg_ragas = gen.get("memgraph_ragas", {})
    vec_ragas = gen.get("vector_ragas", {})

    print(f"  {'Benchmark':<25} | {'Key Metric':<30} | {'Result':>8} | {'Verdict':>7}")
    print(f"  {'─' * 25}-+-{'─' * 30}-+-{'─' * 8}-+-{'─' * 7}")
    print(f"  {'Extraction':<25} | {'Precision':<30} | {ext['avg_precision']:>7.0%} | {'PASS' if ext['passed'] else 'FAIL':>7}")
    print(f"  {'Extraction':<25} | {'Recall':<30} | {ext['avg_recall']:>7.0%} |")
    print(f"  {'Extraction':<25} | {'Edge Type Accuracy':<30} | {ext['avg_edge_type_accuracy']:>7.0%} |")
    print(f"  {'─' * 25}-+-{'─' * 30}-+-{'─' * 8}-+-{'─' * 7}")
    print(f"  {'Retrieval':<25} | {'Hybrid P@3':<30} | {ret['avg_hybrid_p3']:>8.2f} | {'PASS' if ret['passed'] else 'FAIL':>7}")
    print(f"  {'Retrieval':<25} | {'Vector P@3':<30} | {ret['avg_vector_p3']:>8.2f} |")
    print(f"  {'Retrieval':<25} | {'Hybrid MRR':<30} | {ret['avg_hybrid_mrr']:>8.2f} |")
    print(f"  {'Retrieval':<25} | {'Hybrid Win Rate':<30} | {ret['hybrid_wins']:>5}/{len(ret['details'])} |")
    print(f"  {'─' * 25}-+-{'─' * 30}-+-{'─' * 8}-+-{'─' * 7}")
    print(f"  {'Contradiction':<25} | {'F1 Score':<30} | {con['f1']:>7.0%} | {'PASS' if con['passed'] else 'FAIL':>7}")
    print(f"  {'Contradiction':<25} | {'Precision':<30} | {con['precision']:>7.0%} |")
    print(f"  {'Contradiction':<25} | {'Recall':<30} | {con['recall']:>7.0%} |")
    print(f"  {'─' * 25}-+-{'─' * 30}-+-{'─' * 8}-+-{'─' * 7}")
    print(f"  {'Generation (RAGAS)':<25} | {'MemGraph Mean':<30} | {mg_ragas.get('mean', 0):>8.2f} | {'PASS' if gen['passed'] else 'FAIL':>7}")
    print(f"  {'Generation (RAGAS)':<25} | {'Vector Mean':<30} | {vec_ragas.get('mean', 0):>8.2f} |")
    print(f"  {'Generation (RAGAS)':<25} | {'MG Faithfulness':<30} | {mg_ragas.get('faithfulness', 0):>8.2f} |")
    print(f"  {'Generation (RAGAS)':<25} | {'MG Context Recall':<30} | {mg_ragas.get('context_recall', 0):>8.2f} |")
    print(f"  {'Generation (RAGAS)':<25} | {'MemGraph Win Rate':<30} | {gen.get('memgraph_win_rate', 0):>7.0%} |")

    # Overall verdict
    all_passed = all(r["passed"] for r in all_results.values())
    print(f"\n  {'═' * 55}")
    if all_passed:
        print(f"  ALL 4 BENCHMARKS PASSED — System is ready for MCP packaging")
    else:
        failed = [name for name, r in all_results.items() if not r["passed"]]
        print(f"  {len(failed)} BENCHMARK(S) FAILED: {', '.join(failed)}")
        print(f"  Fix these before packaging as an extension")
    print(f"  {'═' * 55}")


if __name__ == "__main__":
    main()
