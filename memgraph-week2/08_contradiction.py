"""
Phase 8: Contradiction Detection Demo

This script teaches: how to detect when new information conflicts with
existing decisions in the knowledge graph, using an LLM as a Natural
Language Inference engine.

It loads the 10 seed messages, then sends 4 test messages designed to
trigger (or not trigger) contradiction alerts:
  1. MongoDB reconsideration — should CONTRADICT the rejection
  2. MySQL over PostgreSQL — should CONTRADICT the PostgreSQL decision
  3. Redis confirmation — should NOT trigger (consistent with existing)
  4. PgBouncer unnecessary — should CONTRADICT the uncertainty

Run: python 08_contradiction.py
Prereqs: Neo4j + Qdrant running, ANTHROPIC_API_KEY set in .env
"""

import os
import sys
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))
sys.path.insert(0, WEEK1_DIR)

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("pipeline", os.path.join(WEEK1_DIR, "04_pipeline.py"))
_pipeline = module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
process_message = _pipeline.process_message
setup_qdrant = _pipeline.setup_qdrant

from core.contradict import check_contradiction, format_contradiction_alert

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

CONTRADICTION_TESTS = [
    {
        "message": "Actually MongoDB's new storage engine looks promising, maybe we should reconsider",
        "expect": "SHOULD contradict MongoDB rejection",
    },
    {
        "message": "I'm now confident we should use MySQL instead of PostgreSQL",
        "expect": "SHOULD contradict PostgreSQL decision AND MySQL rejection",
    },
    {
        "message": "Redis is definitely the right choice for caching",
        "expect": "Should NOT trigger — consistent with existing Redis decision",
    },
    {
        "message": "We've decided PgBouncer is unnecessary — direct connections are fine",
        "expect": "SHOULD contradict PgBouncer uncertainty (resolves it)",
    },
]


def main():
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in memgraph-week1/.env")
        return
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY in memgraph-week1/.env")
        print("  Get one at: https://console.groq.com/")
        return

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print("Connected to all services.\n")

    # Step 1: Clear and reload seed data
    print("=" * 70)
    print("STEP 1: Loading seed data (10 messages)")
    print("=" * 70)

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    setup_qdrant(qdrant)

    for i, msg in enumerate(SEED_MESSAGES):
        if i > 0:
            time.sleep(21)
        process_message(msg, neo4j_driver, qdrant, voyage_client)

    print(f"\nLoaded {len(SEED_MESSAGES)} seed messages.\n")

    # Step 2: Send contradiction test messages
    print("=" * 70)
    print("STEP 2: Testing contradiction detection")
    print("=" * 70)

    for i, test in enumerate(CONTRADICTION_TESTS):
        msg = test["message"]
        expect = test["expect"]

        print(f"\n{'─' * 70}")
        print(f"TEST {i+1}: \"{msg}\"")
        print(f"EXPECTED: {expect}")
        print(f"{'─' * 70}")

        # First ingest the message through the pipeline
        time.sleep(21)
        process_message(msg, neo4j_driver, qdrant, voyage_client)

        # Then check for contradictions
        print(f"\n  Checking for contradictions...")
        contradictions = check_contradiction(msg, neo4j_driver=neo4j_driver)

        if contradictions:
            print(f"\n  ✅ Found {len(contradictions)} contradiction(s):")
            alert = format_contradiction_alert(contradictions)
            for line in alert.split("\n"):
                print(f"    {line}")
        else:
            print(f"\n  ○ No contradictions detected")

        # Verify against expectation
        should_fire = "SHOULD contradict" in expect
        did_fire = len(contradictions) > 0
        if should_fire == did_fire:
            print(f"  ✓ PASS — matched expectation")
        else:
            print(f"  ✗ FAIL — expected {'contradiction' if should_fire else 'no contradiction'}, "
                  f"got {'contradiction' if did_fire else 'none'}")

    # Step 3: Show all CONTRADICTS edges in the graph
    print(f"\n\n{'═' * 70}")
    print("ALL CONTRADICTS EDGES IN GRAPH")
    print(f"{'═' * 70}")

    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (a)-[r:CONTRADICTS]->(b)
            RETURN a.name AS new_node, b.name AS old_node,
                   r.description AS reason, r.confidence AS confidence,
                   r.detected_at AS detected_at,
                   r.new_statement AS new_stmt, r.existing_statement AS old_stmt
        """)
        edges = list(records)

    if not edges:
        print("  (no CONTRADICTS edges found)")
    else:
        for e in edges:
            print(f"\n  ({e['new_node']}) --[CONTRADICTS]--> ({e['old_node']})")
            print(f"    Reason: {e['reason']}")
            print(f"    Confidence: {e['confidence']:.0%}")
            print(f"    Detected: {e['detected_at']}")
            if e['new_stmt']:
                print(f"    New: \"{e['new_stmt'][:80]}\"")
            if e['old_stmt']:
                print(f"    Old: \"{e['old_stmt'][:80]}\"")

    neo4j_driver.close()
    print(f"\n{'═' * 70}")
    print("Contradiction detection demo complete.")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   LLM-based NLI can catch decision reversals that simple text matching
#   would miss. "MongoDB looks promising" contradicts "MongoDB was rejected"
#   only if you understand the intent behind both statements.
#
# What would break without this:
#   The knowledge graph would accumulate contradictory positions silently.
#   The LLM generation step would be fed conflicting context and produce
#   confused or inconsistent answers.
#
# What to build next:
#   09_epistemic_state.py to aggregate the full knowledge state — decisions,
#   questions, uncertainties, AND contradictions — into a snapshot.
# ────────────────────────────────────────────────────────────────────────────
