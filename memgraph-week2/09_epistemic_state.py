"""
Phase 9: Epistemic State Snapshot

This script teaches: how to aggregate a knowledge graph into a structured
snapshot of what someone knows, believes, has decided, and is uncertain
about. This is the "ledger view" vs the "timeline view" of knowledge.

The epistemic state becomes the system prompt prefix for the LLM — it
gives Claude a complete picture of the user's current positions before
it answers any question.

Run: python 09_epistemic_state.py
Prereqs: Run 08_contradiction.py first (loads seed data + contradiction edges)
"""

import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))

from core.state import get_user_state, get_session_diff, format_state_for_prompt

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")


def main():
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("Connected to Neo4j.\n")

    # Step 1: Get full epistemic state
    print("=" * 70)
    print("FULL EPISTEMIC STATE")
    print("=" * 70)

    state = get_user_state(neo4j_driver=neo4j_driver)

    print(f"\n  Decisions ({len(state['decisions'])}):")
    for d in state["decisions"]:
        print(f"    • {d['name']} — {d['context']}")

    print(f"\n  Rejections ({len(state['rejections'])}):")
    for r in state["rejections"]:
        print(f"    ✗ {r['name']} — {r['reason']}")

    print(f"\n  Uncertainties ({len(state['uncertainties'])}):")
    for u in state["uncertainties"]:
        print(f"    ? {u['topic']} — {u['note']}")

    print(f"\n  Dependencies ({len(state['dependencies'])}):")
    for d in state["dependencies"]:
        print(f"    → {d['component']} depends on {d['depends_on']}")

    print(f"\n  Contradictions ({len(state['contradictions'])}):")
    if state["contradictions"]:
        for c in state["contradictions"]:
            print(f"    ⚠ '{c['new']}' vs '{c['old']}' — {c['reason']}")
    else:
        print("    (none)")

    # Step 2: Natural language summary
    print(f"\n{'═' * 70}")
    print("NATURAL LANGUAGE SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n{state['summary']}")

    # Step 3: Formatted for LLM prompt
    print(f"\n{'═' * 70}")
    print("FORMATTED FOR LLM SYSTEM PROMPT")
    print(f"{'═' * 70}")
    formatted = format_state_for_prompt(state)
    print(f"\n{formatted}")

    # Step 4: Session diff
    print(f"\n{'═' * 70}")
    print("SESSION DIFF (most recent changes)")
    print(f"{'═' * 70}")

    diff = get_session_diff(neo4j_driver=neo4j_driver)

    print(f"\n  Recent nodes ({len(diff['new_nodes'])}):")
    for n in diff["new_nodes"][:5]:
        print(f"    + {n['name']}")

    print(f"\n  Recent edges ({len(diff['new_edges'])}):")
    for e in diff["new_edges"][:5]:
        print(f"    + ({e['subject']}) --[{e['rel_type']}]--> ({e['object']})")

    print(f"\n  Contradictions triggered ({len(diff['contradictions_triggered'])}):")
    if diff["contradictions_triggered"]:
        for c in diff["contradictions_triggered"]:
            print(f"    ⚠ {c['new']} vs {c['old']}: {c['reason']}")
    else:
        print("    (none)")

    # Step 5: Raw JSON dump
    print(f"\n{'═' * 70}")
    print("RAW STATE (JSON)")
    print(f"{'═' * 70}")
    print(json.dumps(state, indent=2, default=str))

    neo4j_driver.close()

    print(f"\n{'═' * 70}")
    print("If this summary accurately reflects the decisions you fed in,")
    print("Phase 9 is complete. Next: Phase 10 (LLM generation pipeline).")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   Graph aggregation queries can produce a structured "epistemic ledger"
#   that's far more concise and useful than raw conversation history.
#   The natural language summary bridges structured data and human reading.
#
# What would break without this:
#   The LLM would lack the big picture. It could answer point queries
#   from retrieval but couldn't reason about aggregate state ("what have
#   we decided?", "are there any conflicts?").
#
# What to build next:
#   10_generation.py + core/chat.py — the full pipeline: ingest -> check
#   contradictions -> retrieve context -> get state -> generate response.
# ────────────────────────────────────────────────────────────────────────────
