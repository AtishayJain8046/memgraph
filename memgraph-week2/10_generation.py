"""
Phase 10: LLM Generation Comparison — Plain vs Vector vs Full MemGraph

This script teaches: the difference between a plain LLM call, basic RAG
(vector chunks injected), and full prompt augmentation (hybrid retrieval +
epistemic state + contradiction alerts). Running the same questions three
ways makes the value of the full pipeline visible.

Run: python 10_generation.py
Prereqs: Seed data loaded (run 06_retrieval_fusion.py or 08_contradiction.py),
         GROQ_API_KEY + VOYAGE_API_KEY set in .env
"""

import os
import sys
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient
from openai import OpenAI

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))
sys.path.insert(0, WEEK1_DIR)

from core.retrieve import query_vector
from core.chat import ChatEngine

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

TEST_QUESTIONS = [
    "Why did we reject MongoDB?",
    "What does our auth service depend on?",
    "What should I think about regarding the connection pooler decision?",
    "What have we decided so far about the payments service?",
    "Are there any conflicts in our current decisions?",
]


def ask_plain(client: OpenAI, question: str) -> str:
    """Mode A: Plain Llama — no context, just the question."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=512,
        temperature=0.3,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def ask_vector_only(client: OpenAI, qdrant: QdrantClient,
                    voyage_client: voyageai.Client, question: str) -> str:
    """Mode B: Llama + vector chunks — basic RAG."""
    vector_results = query_vector(qdrant, voyage_client, question, top_k=3)
    chunks = "\n".join(
        f"- {r['content']} (similarity: {r['score']:.3f})"
        for r in vector_results
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=512,
        temperature=0.3,
        messages=[
            {"role": "system", "content": f"Answer using ONLY this context:\n{chunks}\n\nIf the context doesn't contain the answer, say so."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


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
    groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    engine = ChatEngine(neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client)

    print("Connected to all services.\n")
    print("Running 5 questions x 3 modes = 15 LLM calls. This will take a few minutes.\n")

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'═' * 80}")
        print(f"  QUESTION {i+1}: \"{question}\"")
        print(f"{'═' * 80}")

        # Mode A: Plain LLM
        print(f"\n  ┌─ A) PLAIN LLAMA (no context)")
        print(f"  │")
        answer_a = ask_plain(groq_client, question)
        for line in answer_a.split("\n"):
            print(f"  │  {line}")
        print(f"  └─")

        # Mode B: Vector only
        time.sleep(1)
        print(f"\n  ┌─ B) LLAMA + VECTOR ONLY (basic RAG)")
        print(f"  │")
        answer_b = ask_vector_only(groq_client, qdrant, voyage_client, question)
        for line in answer_b.split("\n"):
            print(f"  │  {line}")
        print(f"  └─")

        # Mode C: Full MemGraph
        time.sleep(1)
        print(f"\n  ┌─ C) LLAMA + FULL MEMGRAPH (hybrid retrieval + epistemic state)")
        print(f"  │")
        result = engine.chat(question, skip_ingestion=True)
        for line in result["response"].split("\n"):
            print(f"  │  {line}")

        if result["contradiction_alerts"]:
            print(f"  │")
            print(f"  │  ⚠️ CONTRADICTIONS SURFACED:")
            for c in result["contradiction_alerts"]:
                print(f"  │    - {c['explanation']}")

        print(f"  │")
        print(f"  │  Context sources: {[r['source'] for r in result['context_used']]}")
        print(f"  └─")

        if i < len(TEST_QUESTIONS) - 1:
            print(f"\n  (waiting for rate limits...)")
            time.sleep(21)

    engine.close()

    print(f"\n\n{'═' * 80}")
    print("The difference between A and C is the product.")
    print(f"{'═' * 80}")
    print("""
  A) Plain Llama: Generic answers. Doesn't know your decisions. Makes up
     plausible but wrong context. Cannot surface contradictions.

  B) Vector RAG: Finds relevant messages by semantic similarity. Can quote
     the original text. But lacks structure — can't traverse dependencies
     or identify which decisions are still active vs superseded.

  C) Full MemGraph: Knows your decisions, their reasons, their dependencies.
     Surfaces contradictions. Distinguishes active decisions from rejected
     options. Answers reflect your actual knowledge state, not generic advice.
""")


if __name__ == "__main__":
    main()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   The full MemGraph pipeline (hybrid retrieval + epistemic state +
#   contradiction alerts) produces fundamentally better answers than
#   a plain LLM or basic vector RAG. The structured graph context gives
#   the LLM awareness of decision history, dependencies, and conflicts.
#
# What would break without this:
#   Without this comparison, you can't articulate the value proposition.
#   "Memory-augmented AI" sounds good but means nothing until you see
#   the same question answered three different ways.
#
# What to build next:
#   app.py — the Streamlit UI that wraps this pipeline in an interactive
#   chat interface with a live knowledge state panel.
# ────────────────────────────────────────────────────────────────────────────
