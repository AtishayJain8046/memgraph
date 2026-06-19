"""
Contradiction Detection via LLM-based Natural Language Inference

This module teaches: how to detect when a new statement conflicts with
existing decisions or beliefs in the knowledge graph. Uses Llama 3.3 70B
(via Groq) as an NLI engine — it compares each new message against
committed positions (DECIDED edges) and flags contradictions.

Architecture role: sits between the extraction step and the graph write step.
Every new message is checked against existing knowledge before being stored.
Contradictions are written as CONTRADICTS edges, creating an audit trail.
"""

import os
import json
import time
import asyncio
import functools
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI, AsyncOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

CONTRADICTION_SYSTEM_PROMPT = """You are a contradiction detector for a personal knowledge graph.
Your job is to identify when a new statement conflicts with an existing belief or decision.

CONTRADICTS — flag these:
- DECIDED X, now says they should use Y instead of X (direct reversal)
- DECIDED X, now says X was wrong or a different approach is better (implicit reversal)
- REJECTED X, now says X looks good or should be reconsidered
- Was UNCERTAIN about X, now makes a firm decision about X (resolves the uncertainty — the uncertain state is no longer true)
- Policy reversal: decided on policy A, now says policy B instead (e.g., "remote-first" then "office-first")

CONSISTENT — do NOT flag:
- Reaffirming or praising an existing decision
- Adding details or elaborating on a decision
- Discussing a topic not related to any existing position

UNRELATED — do NOT flag:
- Hypothetical questions or wondering ("I wonder if X would have been better") — these are NOT reversals
- Re-evaluating a deferred/uncertain decision ("maybe we should look at X again") — this is consistent with uncertainty, not contradicting it
- Observations about rejected options ("X has some nice features") without actually proposing to switch
- Mentioning a technology for a completely different purpose than what was decided

Examples:
  Existing: "We decided on remote-first as our work policy"
  New: "We've decided to go back to office-first, remote work isn't productive enough"
  → CONTRADICTS (policy reversal, confidence 0.95)

  Existing: "We decided to require a take-home project instead of whiteboard interviews"
  New: "We've decided whiteboard interviews are actually better than take-home projects"
  → CONTRADICTS (implicit reversal, confidence 0.90)

  Existing: "We are uncertain whether we need PgBouncer"
  New: "We've realized we definitely need PgBouncer, it's critical"
  → CONTRADICTS (resolves uncertainty with firm decision, confidence 0.85)

  Existing: "MongoDB was rejected — write latency was 3x higher"
  New: "I wonder if MongoDB would have been a better choice"
  → CONSISTENT (hypothetical question, not an actual decision to switch)

  Existing: "TimescaleDB was evaluated but we deferred the decision"
  New: "Maybe we should re-evaluate TimescaleDB now"
  → CONSISTENT (revisiting a deferred decision is not contradicting it)

Respond ONLY with valid JSON."""

CONTRADICTION_USER_TEMPLATE = """Existing statement: {existing}
New statement: {new_message}

Does the new statement contradict the existing one?
Respond with JSON only:
{{
  "verdict": "CONTRADICTS" or "CONSISTENT" or "UNRELATED",
  "confidence": float between 0 and 1,
  "explanation": "one sentence explanation if CONTRADICTS, else empty string"
}}"""

CONFIDENCE_THRESHOLD = 0.70


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


def get_existing_positions(driver, user_name: str = "you") -> list[dict]:
    """
    Query Neo4j for all committed positions: decisions, rejections, and beliefs.
    These are the statements worth checking new messages against.
    """
    positions = []

    with driver.session() as session:
        # Get all edges that represent committed positions
        # We check source text because our Week 1 schema stores Entity nodes
        # with the original message in the edge's `source` property
        records = session.run("""
            MATCH (s)-[r]->(o)
            WHERE type(r) IN ['DECIDED', 'REJECTED', 'IS_UNCERTAIN_ABOUT', 'DEPENDS_ON']
            RETURN s.name AS subject, o.name AS object,
                   type(r) AS rel_type, r.relation AS relation,
                   r.source AS source_text, r.confidence AS confidence
        """)
        for rec in records:
            statement = rec["source_text"] or f"{rec['subject']} {rec['relation']} {rec['object']}"
            positions.append({
                "subject": rec["subject"],
                "object": rec["object"],
                "rel_type": rec["rel_type"],
                "statement": statement,
                "relation": rec["relation"],
                "timestamp": None,
            })

    return positions


@retry(max_attempts=3, base_delay=2)
def check_single_contradiction(client: OpenAI, existing: str,
                                new_message: str) -> dict:
    """Check if a new message contradicts one existing statement via Groq."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=256,
        temperature=0.1,
        messages=[
            {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
            {"role": "user", "content": CONTRADICTION_USER_TEMPLATE.format(
                existing=existing, new_message=new_message
            )},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


async def check_single_contradiction_async(client: AsyncOpenAI,
                                            existing: str,
                                            new_message: str) -> dict:
    """Async version for parallel checking when there are many positions."""
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=256,
        temperature=0.1,
        messages=[
            {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
            {"role": "user", "content": CONTRADICTION_USER_TEMPLATE.format(
                existing=existing, new_message=new_message
            )},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


def write_contradiction_edge(driver, new_message: str, existing: dict,
                              explanation: str, confidence: float):
    """Write a CONTRADICTS edge between the new message's entity and the existing one."""
    with driver.session() as session:
        session.run("""
            MATCH (existing:Entity {name_lower: $existing_name_lower})
            MERGE (new:Entity {name_lower: $new_name_lower})
              ON CREATE SET new.name = $new_message, new.created = timestamp()
            CREATE (new)-[:CONTRADICTS {
                description: $explanation,
                confidence: $confidence,
                detected_at: $detected_at,
                new_statement: $new_message,
                existing_statement: $existing_statement
            }]->(existing)
        """,
            existing_name_lower=existing["subject"].lower(),
            new_name_lower=new_message[:50].lower(),
            new_message=new_message,
            explanation=explanation,
            confidence=confidence,
            detected_at=datetime.now().isoformat(),
            existing_statement=existing["statement"],
        )


def check_contradiction(new_message: str, neo4j_driver=None,
                         user_name: str = "you") -> list[dict]:
    """
    Check if a new message contradicts any existing committed positions.

    Steps:
    1. Fetch all DECIDED/REJECTED/IS_UNCERTAIN_ABOUT edges from graph
    2. For each, ask Llama 3.3 (via Groq): does the new message contradict this?
    3. Flag if verdict == CONTRADICTS and confidence >= 0.80
    4. Write CONTRADICTS edges for confirmed contradictions

    Uses async parallel calls when checking more than 5 positions,
    because each check is an independent LLM call and waiting
    sequentially would be O(n * latency) instead of O(latency).
    """
    if not GROQ_API_KEY:
        print("  WARNING: GROQ_API_KEY not set, skipping contradiction check")
        return []

    close_driver = False
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    positions = get_existing_positions(neo4j_driver, user_name)
    if not positions:
        if close_driver:
            neo4j_driver.close()
        return []

    print(f"  Checking against {len(positions)} existing positions...")
    contradictions = []
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    if len(positions) > 5:
        # Batching matters at scale: if a user has 100 decisions in their graph,
        # sequential checking takes 100 * ~2s = 200s. Parallel brings it to ~2-4s
        # (limited by API concurrency). This is the difference between a usable
        # product and one that hangs for minutes after every message.
        #
        # Note: Groq's free tier has rate limits (30 req/min), so for very large
        # graphs you'd need to chunk the parallel calls into batches of ~25.
        async def run_parallel():
            async_client = AsyncOpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            try:
                tasks = [
                    check_single_contradiction_async(
                        async_client, pos["statement"], new_message
                    )
                    for pos in positions
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await async_client.close()

        results = asyncio.run(run_parallel())
    else:
        results = []
        for pos in positions:
            try:
                result = check_single_contradiction(client, pos["statement"], new_message)
                results.append(result)
            except Exception as e:
                print(f"    Error checking against '{pos['statement'][:50]}...': {e}")
                results.append(e)

    for pos, result in zip(positions, results):
        if isinstance(result, Exception):
            continue

        verdict = result.get("verdict", "UNRELATED")
        confidence = result.get("confidence", 0)
        explanation = result.get("explanation", "")

        if verdict == "CONTRADICTS" and confidence >= CONFIDENCE_THRESHOLD:
            write_contradiction_edge(
                neo4j_driver, new_message, pos, explanation, confidence
            )
            contradictions.append({
                "existing": pos["statement"],
                "new": new_message,
                "explanation": explanation,
                "confidence": confidence,
                "existing_timestamp": pos.get("timestamp", "unknown"),
                "rel_type": pos["rel_type"],
            })

    if close_driver:
        neo4j_driver.close()

    return contradictions


def format_contradiction_alert(contradictions: list[dict]) -> str:
    """Format contradiction(s) as a warning string for the UI."""
    if not contradictions:
        return ""

    alerts = []
    for c in contradictions:
        existing_short = c["existing"][:80]
        alerts.append(
            f"⚠️ Contradiction detected: Your new statement conflicts with a prior "
            f"{c['rel_type']} position: '{existing_short}'. "
            f"Reason: {c['explanation']} (confidence: {c['confidence']:.0%})"
        )
    return "\n".join(alerts)


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   An LLM can serve as a domain-aware NLI engine, catching contradictions
#   that a generic classifier would miss (like decision reversals).
#   The CONTRADICTS edge creates an audit trail in the graph.
#
# What would break without this:
#   The knowledge graph would silently accumulate contradictory beliefs.
#   "We chose PostgreSQL" and "We should use MySQL instead" would coexist
#   without any warning, making the epistemic state unreliable.
#
# What to build next:
#   09_epistemic_state.py to aggregate the full knowledge state (decisions,
#   uncertainties, contradictions) into a snapshot the LLM can reason over.
# ────────────────────────────────────────────────────────────────────────────
