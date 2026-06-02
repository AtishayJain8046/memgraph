"""
Memory-Augmented Chat Pipeline

This module teaches: how to wire retrieval, contradiction detection, and
epistemic state into a single LLM generation pipeline. This is "prompt
augmentation" — injecting structured graph context into the system prompt
so the LLM answers with awareness of past decisions and current state.

Architecture role: this is the top-level function that the Streamlit UI
calls. One function call handles: graph update, contradiction check,
hybrid retrieval, state aggregation, and LLM generation.
"""

import os
import sys
import time
import functools
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient
from openai import OpenAI

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))
sys.path.insert(0, WEEK1_DIR)

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("pipeline", os.path.join(WEEK1_DIR, "04_pipeline.py"))
_pipeline = module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
process_message_w1 = _pipeline.process_message
setup_qdrant = _pipeline.setup_qdrant

from core.retrieve import retrieve
from core.contradict import check_contradiction, format_contradiction_alert
from core.state import get_user_state, format_state_for_prompt

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
COLLECTION_NAME = "memgraph_messages"

SYSTEM_PROMPT_TEMPLATE = """You are a memory-augmented AI assistant. You have access to a personal
knowledge graph built from this user's past conversations.

=== CURRENT KNOWLEDGE STATE ===
{knowledge_state}

=== RETRIEVED CONTEXT ===
{retrieved_context}

=== INSTRUCTIONS ===
- Answer using the knowledge state and retrieved context above
- If the question is about a past decision, state when it was made and why
- If retrieved context contains a REJECTED edge reason, include it explicitly
- If a contradiction exists relevant to this question, surface it with a warning
- If you cannot answer from the context, say so — do not hallucinate"""


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


def _format_retrieval_for_prompt(results: list[dict]) -> str:
    """Format retrieval results for injection into the system prompt."""
    if not results:
        return "(no relevant context found)"

    lines = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown").upper()
        content = r.get("content", "")
        line = f"{i}. [{source}] {content}"
        if r.get("edge_type"):
            line += f"  [edge: {r['edge_type']}]"
        if r.get("reason"):
            line += f"  [reason: {r['reason']}]"
        lines.append(line)

    return "\n".join(lines)


class ChatEngine:
    """
    Manages the full chat pipeline with persistent connections.
    Create once and reuse across multiple chat turns.
    """

    def __init__(self, neo4j_driver=None, qdrant=None, voyage_client=None):
        self.neo4j_driver = neo4j_driver or GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.qdrant = qdrant or QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.voyage_client = voyage_client or (
            voyageai.Client(api_key=VOYAGE_API_KEY) if VOYAGE_API_KEY else None
        )
        self.groq_client = OpenAI(
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
        ) if GROQ_API_KEY else None
        setup_qdrant(self.qdrant)

    def close(self):
        self.neo4j_driver.close()

    def chat(self, user_message: str, conversation_history: list[dict] = None,
             skip_ingestion: bool = False) -> dict:
        """
        The full MemGraph chat pipeline:
        1. Ingest the new message into the graph
        2. Check for contradictions with existing knowledge
        3. Retrieve hybrid context (graph + vector, fused with RRF)
        4. Get the epistemic state snapshot
        5. Build an augmented system prompt
        6. Generate a response with Llama 3.3 (via Groq)

        skip_ingestion: set True for pure questions that shouldn't be stored
        as facts (e.g., "what did we decide about X?")
        """
        if conversation_history is None:
            conversation_history = []

        # Step 1: Ingest (update the graph)
        if not skip_ingestion and self.voyage_client:
            try:
                process_message_w1(
                    user_message, self.neo4j_driver, self.qdrant, self.voyage_client
                )
            except Exception as e:
                print(f"  Warning: ingestion failed: {e}")

        # Step 2: Contradiction check
        contradiction_alerts = []
        try:
            contradictions = check_contradiction(
                user_message, neo4j_driver=self.neo4j_driver
            )
            if contradictions:
                alert_text = format_contradiction_alert(contradictions)
                contradiction_alerts = contradictions
                print(f"\n{alert_text}")
        except Exception as e:
            print(f"  Warning: contradiction check failed: {e}")

        # Step 3: Hybrid retrieval
        context_used = []
        try:
            fused_results, _, _ = retrieve(
                user_message, top_k=5,
                neo4j_driver=self.neo4j_driver,
                qdrant=self.qdrant,
                voyage_client=self.voyage_client,
            )
            context_used = fused_results
        except Exception as e:
            print(f"  Warning: retrieval failed: {e}")

        # Step 4: Epistemic state
        epistemic_state = {}
        try:
            epistemic_state = get_user_state(neo4j_driver=self.neo4j_driver)
        except Exception as e:
            print(f"  Warning: state aggregation failed: {e}")

        # Step 5: Build augmented system prompt
        knowledge_state_str = format_state_for_prompt(epistemic_state) if epistemic_state else "(unavailable)"
        retrieved_context_str = _format_retrieval_for_prompt(context_used)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            knowledge_state=knowledge_state_str,
            retrieved_context=retrieved_context_str,
        )

        # Step 6: Generate response
        messages = list(conversation_history) + [{"role": "user", "content": user_message}]
        response_text = self._generate(system_prompt, messages)

        return {
            "response": response_text,
            "context_used": context_used,
            "contradiction_alerts": contradiction_alerts,
            "epistemic_state": epistemic_state,
        }

    @retry(max_attempts=3, base_delay=2)
    def _generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Call Llama 3.3 (via Groq) with the augmented prompt."""
        if not self.groq_client:
            return "(GROQ_API_KEY not set — cannot generate response)"

        # Groq uses the OpenAI-compatible format: system message first, then user/assistant
        groq_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1024,
            temperature=0.3,
            messages=groq_messages,
        )
        return response.choices[0].message.content


def chat(user_message: str, conversation_history: list[dict] = None) -> dict:
    """
    Convenience function that creates a ChatEngine, runs one turn, and closes.
    For multi-turn conversations, use ChatEngine directly.
    """
    engine = ChatEngine()
    try:
        return engine.chat(user_message, conversation_history)
    finally:
        engine.close()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   Prompt augmentation — injecting structured graph context and epistemic
#   state into the system prompt — produces fundamentally different answers
#   than plain LLM calls or basic RAG with text chunks.
#
# What would break without this:
#   The LLM would have no awareness of past decisions, contradictions, or
#   the user's current knowledge state. Every answer would be generic,
#   missing the "memory" that makes MemGraph useful.
#
# What to build next:
#   app.py — the Streamlit UI that wraps ChatEngine with a chat interface
#   and a live knowledge state panel.
# ────────────────────────────────────────────────────────────────────────────
