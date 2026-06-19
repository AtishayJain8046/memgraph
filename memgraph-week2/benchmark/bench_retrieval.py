"""
Benchmark 2: Retrieval Quality — Does Hybrid Beat Individual Sources?

Measures whether RRF-fused hybrid retrieval (graph + vector) produces
better results than graph-only or vector-only across diverse query types.
This is the core thesis of the project.

Metrics:
  - Precision@3: fraction of top-3 results containing the answer
  - MRR (Mean Reciprocal Rank): 1/rank of first correct result
  - Per-query-type breakdown: which system wins on which question types

Run: python bench_retrieval.py
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

from core.retrieve import retrieve, query_graph, query_vector
from bench_data import SEED_MESSAGES, RETRIEVAL_QUERIES

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COLLECTION_NAME = "memgraph_messages"


def precision_at_k(results: list[dict], ground_truth: str, k: int = 3) -> float:
    """Fraction of top-K results containing the ground truth (case-insensitive)."""
    checked = results[:k]
    if not checked:
        return 0.0
    correct = sum(
        1 for r in checked
        if ground_truth.lower() in r.get("content", "").lower()
    )
    return correct / len(checked)


def reciprocal_rank(results: list[dict], ground_truth: str) -> float:
    """
    1 / rank of the first result containing the ground truth.
    Returns 0 if no result contains it. MRR rewards finding the right
    answer early — Precision@K treats all positions equally.
    """
    for i, r in enumerate(results):
        if ground_truth.lower() in r.get("content", "").lower():
            return 1.0 / (i + 1)
    return 0.0


def ingest_seed_data(neo4j_driver, qdrant, voyage_client):
    """Clear stores and ingest all 30 seed messages."""
    print("Ingesting seed data (this takes ~10 minutes due to rate limits)...\n")

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


def run_retrieval_benchmark(neo4j_driver=None, qdrant=None, voyage_client=None,
                             skip_ingestion=False) -> dict:
    """Run all retrieval queries through 3 modes and score each."""
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
        ingest_seed_data(neo4j_driver, qdrant, voyage_client)

    print("=" * 70)
    print("BENCHMARK 2: RETRIEVAL QUALITY")
    print("=" * 70)
    print(f"Testing {len(RETRIEVAL_QUERIES)} queries across 3 retrieval modes\n")

    results = []

    for i, q in enumerate(RETRIEVAL_QUERIES):
        query = q["query"]
        gt = q["ground_truth"]
        qtype = q["query_type"]

        print(f"  [{i+1}/{len(RETRIEVAL_QUERIES)}] \"{query}\"")

        if i > 0:
            time.sleep(21)

        # Graph only
        graph_results = query_graph(neo4j_driver, query)
        g_p3 = precision_at_k(graph_results, gt, k=3)
        g_rr = reciprocal_rank(graph_results, gt)

        # Vector only
        vector_results = query_vector(qdrant, voyage_client, query, top_k=5)
        v_p3 = precision_at_k(vector_results, gt, k=3)
        v_rr = reciprocal_rank(vector_results, gt)

        # Hybrid
        hybrid_results, _, _ = retrieve(
            query, top_k=5,
            neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client
        )
        h_p3 = precision_at_k(hybrid_results, gt, k=3)
        h_rr = reciprocal_rank(hybrid_results, gt)

        results.append({
            "query": query,
            "query_type": qtype,
            "ground_truth": gt,
            "graph_p3": g_p3, "graph_rr": g_rr,
            "vector_p3": v_p3, "vector_rr": v_rr,
            "hybrid_p3": h_p3, "hybrid_rr": h_rr,
        })

        winner = "HYBRID" if h_p3 >= max(g_p3, v_p3) else ("GRAPH" if g_p3 > v_p3 else "VECTOR")
        print(f"    P@3: G={g_p3:.2f} V={v_p3:.2f} H={h_p3:.2f} | Winner: {winner}")

    # ── Aggregate metrics ──
    avg = lambda key: sum(r[key] for r in results) / len(results) if results else 0

    avg_gp = avg("graph_p3")
    avg_vp = avg("vector_p3")
    avg_hp = avg("hybrid_p3")
    avg_gr = avg("graph_rr")
    avg_vr = avg("vector_rr")
    avg_hr = avg("hybrid_rr")

    # Per-query-type breakdown
    query_types = sorted(set(r["query_type"] for r in results))
    type_stats = {}
    for qt in query_types:
        qt_results = [r for r in results if r["query_type"] == qt]
        type_stats[qt] = {
            "count": len(qt_results),
            "graph_p3": sum(r["graph_p3"] for r in qt_results) / len(qt_results),
            "vector_p3": sum(r["vector_p3"] for r in qt_results) / len(qt_results),
            "hybrid_p3": sum(r["hybrid_p3"] for r in qt_results) / len(qt_results),
        }

    # ── Print results ──
    print(f"\n{'═' * 70}")
    print("RETRIEVAL RESULTS")
    print(f"{'═' * 70}")

    print(f"\n  Overall Averages:")
    print(f"  {'Metric':<15} | {'Graph':>8} | {'Vector':>8} | {'Hybrid':>8}")
    print(f"  {'─' * 15}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}")
    print(f"  {'Precision@3':<15} | {avg_gp:>8.2f} | {avg_vp:>8.2f} | {avg_hp:>8.2f}")
    print(f"  {'MRR':<15} | {avg_gr:>8.2f} | {avg_vr:>8.2f} | {avg_hr:>8.2f}")

    print(f"\n  Per Query Type (Precision@3):")
    print(f"  {'Type':<18} | {'N':>3} | {'Graph':>8} | {'Vector':>8} | {'Hybrid':>8} | {'Winner':>8}")
    print(f"  {'─' * 18}-+-{'─' * 3}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 8}")
    for qt, stats in sorted(type_stats.items()):
        scores = {"Graph": stats["graph_p3"], "Vector": stats["vector_p3"], "Hybrid": stats["hybrid_p3"]}
        winner = max(scores, key=scores.get)
        print(f"  {qt:<18} | {stats['count']:>3} | {stats['graph_p3']:>8.2f} | "
              f"{stats['vector_p3']:>8.2f} | {stats['hybrid_p3']:>8.2f} | {winner:>8}")

    # Win counts
    hybrid_wins = sum(1 for r in results if r["hybrid_p3"] >= max(r["graph_p3"], r["vector_p3"]))
    graph_wins = sum(1 for r in results if r["graph_p3"] > r["hybrid_p3"] and r["graph_p3"] > r["vector_p3"])
    vector_wins = sum(1 for r in results if r["vector_p3"] > r["hybrid_p3"] and r["vector_p3"] > r["graph_p3"])

    print(f"\n  Win counts: Hybrid={hybrid_wins} | Graph={graph_wins} | Vector={vector_wins}")

    # Pass/fail
    hybrid_beats_vector = avg_hp > avg_vp
    hybrid_beats_graph = avg_hp > avg_gp
    passed = hybrid_beats_vector

    print(f"\n  {'─' * 50}")
    if passed:
        improvement = (avg_hp - avg_vp) / avg_vp * 100 if avg_vp > 0 else float('inf')
        print(f"  PASS — Hybrid beats vector-only by {improvement:.0f}%")
    else:
        print(f"  FAIL — Hybrid ({avg_hp:.2f}) does not beat vector-only ({avg_vp:.2f})")
        print(f"  The core thesis is not proven. Consider:")
        print(f"    - Tuning RRF_K (currently 20, try 10 or 60)")
        print(f"    - Improving keyword extraction for graph queries")
        print(f"    - Enriching graph edges with more metadata")

    if close_driver:
        neo4j_driver.close()

    return {
        "benchmark": "retrieval",
        "avg_graph_p3": avg_gp, "avg_vector_p3": avg_vp, "avg_hybrid_p3": avg_hp,
        "avg_graph_mrr": avg_gr, "avg_vector_mrr": avg_vr, "avg_hybrid_mrr": avg_hr,
        "hybrid_wins": hybrid_wins, "graph_wins": graph_wins, "vector_wins": vector_wins,
        "type_stats": type_stats,
        "passed": passed,
        "details": results,
    }


if __name__ == "__main__":
    if not os.getenv("VOYAGE_API_KEY"):
        print("ERROR: Set VOYAGE_API_KEY in memgraph-week1/.env")
        sys.exit(1)
    run_retrieval_benchmark()
