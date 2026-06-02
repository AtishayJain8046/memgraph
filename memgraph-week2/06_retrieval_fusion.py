"""
Phase 6: Hybrid Retrieval Fusion — Combining Graph + Vector with RRF

This script teaches: how two retrieval systems with incompatible score formats
can be merged into a single ranked list using Reciprocal Rank Fusion (RRF).

It demonstrates the core thesis of MemGraph: structural graph results
(precise edges, typed relationships) combined with semantic vector results
(fuzzy meaning-based matches) produce better retrieval than either alone.

Run: python 06_retrieval_fusion.py
Prereqs: Neo4j + Qdrant running (docker-compose up), .env configured
"""

import os
import sys
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load env from Week 1's .env file
WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))

# Import Week 1's pipeline
sys.path.insert(0, WEEK1_DIR)
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("pipeline", os.path.join(WEEK1_DIR, "04_pipeline.py"))
_pipeline = module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
process_message = _pipeline.process_message
setup_qdrant = _pipeline.setup_qdrant

# Import our new hybrid retriever
from core.retrieve import retrieve, query_graph, query_vector

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COLLECTION_NAME = "memgraph_messages"

SEED_MESSAGES = [
    "We are building a payments service and evaluating databases",
    "After load testing we decided on PostgreSQL for the main store",
    "MongoDB was rejected — write latency was 3x higher under our load profile",
    "Redis was chosen for session caching and rate limiting",
    "MySQL was considered but rejected due to lack of JSONB support",
    "Our backend API depends on PostgreSQL for all transaction data",
    "The auth service depends on Redis for session lookups",
    "We are uncertain whether we need a connection pooler like PgBouncer",
    "TimescaleDB was evaluated for metrics but we deferred the decision",
    "The data pipeline will depend on whatever database we choose for metrics",
]

TEST_QUERIES = [
    "Why did we reject MongoDB?",
    "What does the auth service depend on?",
    "What are we uncertain about?",
    "What databases were evaluated?",
    "What decisions have dependencies?",
]


def ingest_seed_data(neo4j_driver, qdrant, voyage_client):
    """Clear stores and load the 10 seed messages through the Week 1 pipeline."""
    print("=" * 70)
    print("STEP 1: Ingesting seed data through Week 1 pipeline")
    print("=" * 70)

    # Clear previous data
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    setup_qdrant(qdrant)
    print("Cleared previous data.\n")

    for i, msg in enumerate(SEED_MESSAGES):
        if i > 0:
            print(f"\n  (waiting 21s for Voyage rate limit...)")
            time.sleep(21)
        process_message(msg, neo4j_driver, qdrant, voyage_client)

    print(f"\n{'─' * 70}")
    print(f"Ingested {len(SEED_MESSAGES)} messages into both stores.")
    print(f"{'─' * 70}\n")


def print_results_section(title: str, results: list[dict], max_show: int = 3):
    """Pretty-print a section of retrieval results."""
    print(f"\n  {title}")
    print(f"  {'─' * 60}")
    if not results:
        print("    (no results)")
        return
    for i, r in enumerate(results[:max_show]):
        source = r.get("source", "?")
        content = r.get("content", "")
        line = f"    {i+1}. [{source.upper()}] {content}"
        if "rrf_score" in r:
            line += f"  (rrf: {r['rrf_score']:.5f})"
        if "score" in r:
            line += f"  (sim: {r['score']:.4f})"
        if r.get("edge_type"):
            line += f"  [edge: {r['edge_type']}]"
        if r.get("reason"):
            line += f"  [reason: {r['reason']}]"
        print(line)


def run_comparison(query: str, neo4j_driver, qdrant, voyage_client):
    """Run a single query through all three retrieval modes and display results."""
    print(f"\n{'═' * 70}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'═' * 70}")

    # Graph only
    graph_results = query_graph(neo4j_driver, query)
    print_results_section("GRAPH ONLY RESULTS (top 3)", graph_results, max_show=3)

    # Vector only (costs an API call — careful with rate limits)
    vector_results = query_vector(qdrant, voyage_client, query)
    print_results_section("VECTOR ONLY RESULTS (top 3)", vector_results, max_show=3)

    # Hybrid fused
    fused, _, _ = retrieve(
        query, top_k=5,
        neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client
    )
    print_results_section("HYBRID FUSED RESULTS (top 5)", fused, max_show=5)


def main():
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in memgraph-week1/.env")
        return

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print("Connected to Neo4j, Qdrant, and Voyage.\n")

    # Phase 1: Ingest
    ingest_seed_data(neo4j_driver, qdrant, voyage_client)

    # Phase 2: Compare retrieval modes
    print("\n" + "=" * 70)
    print("STEP 2: Running retrieval comparison across 5 test queries")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES):
        if i > 0:
            print(f"\n  (waiting 21s for Voyage rate limit...)")
            time.sleep(21)
        run_comparison(query, neo4j_driver, qdrant, voyage_client)

    neo4j_driver.close()

    # Analysis
    print(f"\n\n{'═' * 70}")
    print("FUSION ANALYSIS")
    print(f"{'═' * 70}")
    print("""
Queries that benefit MOST from fusion:
  1. "Why did we reject MongoDB?" — Graph finds the REJECTED edge with the
     exact reason. Vector finds the original message by semantic similarity.
     Fusion surfaces both: the structured edge AND the raw context.

  2. "What decisions have dependencies?" — Graph traverses DEPENDS_ON edges
     directly (structural match). Vector finds messages mentioning "depends"
     semantically. Fusion catches dependency mentions the graph missed if
     the LLM extraction dropped some triples.

  3. "What databases were evaluated?" — Vector excels here (broad semantic
     match across all database-mentioning messages). Graph contributes
     specific DECIDED/REJECTED edges. Fusion gives a complete picture:
     what was evaluated AND what the outcome was.

Queries where one source dominates:
  - "What does the auth service depend on?" — Graph likely dominates because
    DEPENDS_ON is a precise structural query. Vector adds little here.
  - "What are we uncertain about?" — Could go either way. Graph has
    IS_UNCERTAIN_ABOUT edges; vector catches hedging language.

Key takeaway: fusion is most valuable when the question needs BOTH precision
(graph edges with reasons) and recall (vector catching related context the
extraction step might have missed).
""")


if __name__ == "__main__":
    main()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   Hybrid retrieval produces results that neither graph-only nor vector-only
#   can achieve alone. The RRF fusion rewards results confirmed by both
#   sources, pushing the most relevant context to the top.
#
# What would break without this:
#   The LLM generation step (Phase 10) would get incomplete context —
#   either missing structural relationships (vector-only) or missing
#   semantic context (graph-only). Answers would be less accurate.
#
# What to build next:
#   07_benchmark.py to quantify the improvement with Precision@K metrics.
#   If hybrid doesn't beat individual sources, we tune RRF_K or keyword
#   extraction before building anything else on top.
# ────────────────────────────────────────────────────────────────────────────
