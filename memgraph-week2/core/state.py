"""
Epistemic State Aggregation

This module teaches: how to aggregate graph data into a structured snapshot
of what someone knows, believes, has decided, and is uncertain about. This
is the bridge between raw graph data and LLM-consumable context.

Architecture role: the state snapshot becomes the "system prompt prefix" for
the LLM generation step. It gives the LLM a complete picture of the user's
current knowledge before it answers any question.
"""

import os
import json
import time
import functools
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


def retry(max_attempts=3, base_delay=2):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = base_delay * (2 ** attempt)
                    print(f"  (retry {attempt + 1}/{max_attempts} after {wait}s: {e})")
                    time.sleep(wait)
        return wrapper
    return decorator


def get_user_state(neo4j_driver=None, user_name: str = "you") -> dict:
    """
    Aggregate the full epistemic state from the knowledge graph.

    Runs 4 Cypher queries covering:
      1. All decisions (DECIDED edges)
      2. Open questions (IS_UNCERTAIN_ABOUT edges, unanswered)
      3. Active contradictions (CONTRADICTS edges)
      4. Dependencies (DEPENDS_ON edges)

    Then makes one Claude call to generate a natural language summary.

    Note: Our Week 1 schema uses Entity nodes with typed edges, not
    separate Decision/OpenQuestion node types. The queries adapt to this.
    """
    close_driver = False
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    state = {
        "decisions": [],
        "rejections": [],
        "open_questions": [],
        "contradictions": [],
        "uncertainties": [],
        "dependencies": [],
        "summary": "",
    }

    with neo4j_driver.session() as session:
        # Query 1: Decisions (DECIDED edges)
        records = session.run("""
            MATCH (s)-[r:DECIDED]->(o)
            RETURN s.name AS who, o.name AS what,
                   r.relation AS context, r.source AS source_text
        """)
        for rec in records:
            state["decisions"].append({
                "name": rec["what"],
                "context": rec["context"] or rec["source_text"] or "",
                "decided_by": rec["who"],
            })

        # Query 2: Rejections (REJECTED edges)
        records = session.run("""
            MATCH (s)-[r:REJECTED]->(o)
            RETURN s.name AS who, o.name AS what,
                   r.relation AS reason, r.source AS source_text
        """)
        for rec in records:
            state["rejections"].append({
                "name": rec["what"],
                "reason": rec["reason"] or "",
                "source": rec["source_text"] or "",
            })

        # Query 3: Uncertainties (IS_UNCERTAIN_ABOUT edges)
        records = session.run("""
            MATCH (s)-[r:IS_UNCERTAIN_ABOUT]->(o)
            RETURN s.name AS who, o.name AS topic,
                   r.relation AS note, r.source AS source_text
        """)
        for rec in records:
            state["uncertainties"].append({
                "topic": rec["topic"],
                "note": rec["note"] or "",
                "source": rec["source_text"] or "",
            })

        # Query 4: Active contradictions (CONTRADICTS edges)
        records = session.run("""
            MATCH (a)-[r:CONTRADICTS]->(b)
            RETURN a.name AS new_statement, b.name AS old_statement,
                   r.description AS reason, r.detected_at AS detected_at,
                   r.confidence AS confidence
            ORDER BY r.detected_at DESC
            LIMIT 5
        """)
        for rec in records:
            state["contradictions"].append({
                "new": rec["new_statement"],
                "old": rec["old_statement"],
                "reason": rec["reason"] or "",
                "detected_at": rec["detected_at"] or "",
            })

        # Query 5: Dependencies (DEPENDS_ON edges)
        records = session.run("""
            MATCH (s)-[r:DEPENDS_ON]->(o)
            RETURN s.name AS component, o.name AS depends_on,
                   r.relation AS detail, r.source AS source_text
        """)
        for rec in records:
            state["dependencies"].append({
                "component": rec["component"],
                "depends_on": rec["depends_on"],
                "detail": rec["detail"] or "",
            })

    # Generate natural language summary via Claude
    state["summary"] = _generate_summary(state)

    if close_driver:
        neo4j_driver.close()

    return state


@retry(max_attempts=3, base_delay=2)
def _generate_summary(state: dict) -> str:
    """Ask Llama 3.3 (via Groq) to produce a 5-bullet summary of the knowledge state."""
    if not GROQ_API_KEY:
        return "(Summary unavailable — GROQ_API_KEY not set)"

    state_for_prompt = {k: v for k, v in state.items() if k != "summary"}

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=500,
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": (
                f"Given this knowledge state:\n{json.dumps(state_for_prompt, indent=2)}\n\n"
                "Write a 5-bullet summary covering:\n"
                "1. What has been decided (and why)\n"
                "2. What was rejected (and why)\n"
                "3. What remains uncertain\n"
                "4. What dependencies exist\n"
                "5. What contradictions exist (if any)\n\n"
                "Be specific — use the actual names and reasons from the data. "
                "Each bullet should be one sentence."
            ),
        }],
    )

    return response.choices[0].message.content.strip()


def get_session_diff(neo4j_driver=None, session_id: str = None) -> dict:
    """
    Return what was added in a specific session (or the most recent one).
    Shows new nodes, new edges, and any contradictions triggered.
    Used in the UI to show "what changed this session."

    Since our Week 1 schema doesn't track session_id on edges, we
    approximate by looking at the most recently created entities
    and the most recent CONTRADICTS edges.
    """
    close_driver = False
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    diff = {
        "new_nodes": [],
        "new_edges": [],
        "contradictions_triggered": [],
    }

    with neo4j_driver.session() as session:
        # Most recent nodes (by creation timestamp)
        records = session.run("""
            MATCH (n:Entity)
            WHERE n.created IS NOT NULL
            RETURN n.name AS name, n.created AS created
            ORDER BY n.created DESC
            LIMIT 10
        """)
        for rec in records:
            diff["new_nodes"].append({
                "name": rec["name"],
                "created": rec["created"],
            })

        # Most recent edges (by source text — proxy for recency)
        records = session.run("""
            MATCH (s)-[r]->(o)
            RETURN s.name AS subject, type(r) AS rel_type, o.name AS object,
                   r.source AS source_text
            ORDER BY id(r) DESC
            LIMIT 10
        """)
        for rec in records:
            diff["new_edges"].append({
                "subject": rec["subject"],
                "rel_type": rec["rel_type"],
                "object": rec["object"],
                "source": rec["source_text"] or "",
            })

        # Recent contradictions
        records = session.run("""
            MATCH (a)-[r:CONTRADICTS]->(b)
            RETURN a.name AS new_node, b.name AS old_node,
                   r.description AS reason, r.detected_at AS detected_at
            ORDER BY r.detected_at DESC
            LIMIT 5
        """)
        for rec in records:
            diff["contradictions_triggered"].append({
                "new": rec["new_node"],
                "old": rec["old_node"],
                "reason": rec["reason"],
                "detected_at": rec["detected_at"],
            })

    if close_driver:
        neo4j_driver.close()

    return diff


def format_state_for_prompt(state: dict) -> str:
    """Format the epistemic state as a string for injection into an LLM prompt."""
    sections = []

    if state["decisions"]:
        lines = [f"  - {d['name']}: {d['context']}" for d in state["decisions"]]
        sections.append("Decisions made:\n" + "\n".join(lines))

    if state["rejections"]:
        lines = [f"  - {r['name']}: {r['reason']}" for r in state["rejections"]]
        sections.append("Rejected options:\n" + "\n".join(lines))

    if state["uncertainties"]:
        lines = [f"  - {u['topic']}: {u['note']}" for u in state["uncertainties"]]
        sections.append("Uncertainties:\n" + "\n".join(lines))

    if state["dependencies"]:
        lines = [f"  - {d['component']} depends on {d['depends_on']}" for d in state["dependencies"]]
        sections.append("Dependencies:\n" + "\n".join(lines))

    if state["contradictions"]:
        lines = [
            f"  - '{c['new']}' contradicts '{c['old']}': {c['reason']}"
            for c in state["contradictions"]
        ]
        sections.append("Active contradictions:\n" + "\n".join(lines))
    else:
        sections.append("Active contradictions:\n  None")

    if state.get("open_questions"):
        lines = [f"  - {q['question']}" for q in state["open_questions"]]
        sections.append("Open questions:\n" + "\n".join(lines))

    return "\n\n".join(sections)


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   A knowledge graph can be aggregated into a structured "epistemic state"
#   that's far more useful than raw chat history. The LLM gets a concise
#   ledger of current positions instead of re-reading every message.
#
# What would break without this:
#   The LLM generation step would lack the big picture. It could answer
#   point queries ("why was MongoDB rejected?") from retrieval, but
#   couldn't answer aggregate questions ("what have we decided so far?")
#   without re-scanning the entire conversation.
#
# What to build next:
#   10_generation.py + core/chat.py — wire retrieval, contradiction
#   detection, and epistemic state into a single LLM generation pipeline.
# ────────────────────────────────────────────────────────────────────────────
