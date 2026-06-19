"""
Benchmark 4: End-to-End Generation Quality — RAGAS Metrics

Evaluates the full MemGraph pipeline using the four core metrics from the
RAGAS framework (Retrieval Augmented Generation Assessment):

  1. Faithfulness (0-1): Is the answer grounded in retrieved context?
     Decomposes the answer into claims, checks each against context.
     Catches hallucination — the model inventing facts not in the context.

  2. Answer Relevancy (0-1): Does the answer actually address the question?
     A factually correct but off-topic answer scores low here.

  3. Context Precision (0-1): Is the retrieved context relevant?
     Checks each context chunk for relevance, weighted by rank position.
     High precision = the retrieval layer is feeding useful context.

  4. Context Recall (0-1): Did the retrieval find all needed information?
     Decomposes the ground truth into statements, checks if context
     covers each one. Low recall = retrieval is missing key facts.

Reference: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

Run: python bench_generation.py
Prereqs: Neo4j + Qdrant running, GROQ_API_KEY + VOYAGE_API_KEY set,
         seed data already ingested (run bench_retrieval.py first)
"""

import os
import sys
import json
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
from openai import OpenAI

from core.retrieve import query_vector, retrieve
from core.chat import ChatEngine
from bench_data import GENERATION_QUESTIONS

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


# ═══════════════════════════════════════════════════════════════════════════
# RAGAS METRIC PROMPTS
#
# Each metric uses a single LLM call that returns structured JSON.
# The RAGAS paper uses multi-step decomposition (extract statements,
# then verify each one). We batch both steps into one call to stay
# within Groq's free-tier rate limits while preserving the methodology.
# ═══════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are evaluating an AI answer for faithfulness to retrieved context.

TASK:
1. Extract every factual claim from the answer (list them)
2. For each claim, check if it is supported by the retrieved context
3. Compute: faithfulness = number_supported / total_claims

Retrieved context:
{context}

Answer to evaluate:
{answer}

Respond with ONLY valid JSON:
{{
  "claims": [
    {{"claim": "...", "supported": true/false}},
    ...
  ],
  "faithfulness": <float 0.0 to 1.0>
}}"""

ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an AI answer is relevant to the question asked.

A relevant answer directly addresses what was asked. An irrelevant answer may be
factually correct but off-topic, or may answer a different question entirely.

Question: {question}

Answer to evaluate:
{answer}

Score the relevancy from 0.0 to 1.0:
- 1.0 = directly and completely answers the question
- 0.5 = partially relevant, addresses some aspects
- 0.0 = completely off-topic or doesn't address the question

Respond with ONLY valid JSON:
{{"relevancy": <float 0.0 to 1.0>, "reason": "one sentence"}}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating retrieved context quality for answering a question.

For each context chunk, determine if it is relevant to answering the question.
Context precision rewards putting relevant chunks at the top of the list.

Question: {question}

Retrieved context chunks (in ranked order):
{context_numbered}

For each chunk, mark it as relevant (true) or irrelevant (false).
Then compute weighted precision: relevant chunks ranked higher count more.

Respond with ONLY valid JSON:
{{
  "chunks": [
    {{"chunk_index": 1, "relevant": true/false}},
    ...
  ],
  "context_precision": <float 0.0 to 1.0>
}}"""

CONTEXT_RECALL_PROMPT = """You are evaluating whether retrieved context covers all the information needed.

TASK:
1. Extract key statements from the ground truth answer
2. For each statement, check if ANY of the retrieved context chunks support it
3. Compute: context_recall = statements_supported / total_statements

Ground truth answer:
{ground_truth}

Retrieved context:
{context}

Respond with ONLY valid JSON:
{{
  "statements": [
    {{"statement": "...", "supported_by_context": true/false}},
    ...
  ],
  "context_recall": <float 0.0 to 1.0>
}}"""


def _llm_call(client: OpenAI, prompt: str) -> dict:
    """Single LLM call with retry and JSON parsing."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=1024,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                return {"error": str(e)}


def format_context(results: list[dict]) -> str:
    """Format retrieval results as plain text for RAGAS prompts."""
    if not results:
        return "(no context retrieved)"
    return "\n".join(
        f"- {r.get('content', '')}" for r in results
    )


def format_context_numbered(results: list[dict]) -> str:
    """Format retrieval results as numbered list for context precision."""
    if not results:
        return "(no context retrieved)"
    return "\n".join(
        f"{i+1}. {r.get('content', '')}" for i, r in enumerate(results)
    )


def compute_faithfulness(client: OpenAI, answer: str, context: list[dict]) -> dict:
    """RAGAS Faithfulness: fraction of answer claims supported by context."""
    result = _llm_call(client, FAITHFULNESS_PROMPT.format(
        context=format_context(context),
        answer=answer,
    ))
    return {
        "score": result.get("faithfulness", 0.0),
        "claims": result.get("claims", []),
    }


def compute_answer_relevancy(client: OpenAI, question: str, answer: str) -> dict:
    """RAGAS Answer Relevancy: how well the answer addresses the question."""
    result = _llm_call(client, ANSWER_RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
    ))
    return {
        "score": result.get("relevancy", 0.0),
        "reason": result.get("reason", ""),
    }


def compute_context_precision(client: OpenAI, question: str, context: list[dict]) -> dict:
    """RAGAS Context Precision: relevance of retrieved chunks, weighted by rank."""
    result = _llm_call(client, CONTEXT_PRECISION_PROMPT.format(
        question=question,
        context_numbered=format_context_numbered(context),
    ))
    return {
        "score": result.get("context_precision", 0.0),
        "chunks": result.get("chunks", []),
    }


def compute_context_recall(client: OpenAI, ground_truth: str, context: list[dict]) -> dict:
    """RAGAS Context Recall: fraction of ground truth statements covered by context."""
    result = _llm_call(client, CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth,
        context=format_context(context),
    ))
    return {
        "score": result.get("context_recall", 0.0),
        "statements": result.get("statements", []),
    }


def ask_plain(client: OpenAI, question: str) -> str:
    """Plain LLM — no context."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=512,
        temperature=0.3,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def ask_vector_only(client: OpenAI, qdrant: QdrantClient,
                    voyage_client: voyageai.Client, question: str) -> tuple[str, list[dict]]:
    """Vector-only RAG. Returns (answer, context_chunks)."""
    vector_results = query_vector(qdrant, voyage_client, question, top_k=5)
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
    return response.choices[0].message.content, vector_results


def run_generation_benchmark(neo4j_driver=None, qdrant=None, voyage_client=None) -> dict:
    """Run all questions through 3 modes and evaluate with RAGAS metrics."""
    close_driver = False
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        close_driver = True
    if qdrant is None:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if voyage_client is None:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    engine = ChatEngine(neo4j_driver=neo4j_driver, qdrant=qdrant, voyage_client=voyage_client)

    print("=" * 70)
    print("BENCHMARK 4: GENERATION QUALITY (RAGAS Metrics)")
    print("=" * 70)
    print(f"Testing {len(GENERATION_QUESTIONS)} questions x 3 modes")
    print(f"Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall\n")
    print("NOTE: Assumes seed data is already ingested.")
    print("      Run bench_retrieval.py first if the stores are empty.\n")

    results = []

    for i, q in enumerate(GENERATION_QUESTIONS):
        question = q["question"]
        reference = q["reference"]
        key_facts = q["key_facts"]

        print(f"\n  [{i+1}/{len(GENERATION_QUESTIONS)}] \"{question}\"")

        # ── Generate answers in 3 modes ──
        time.sleep(2)
        answer_plain = ask_plain(groq_client, question)

        time.sleep(2)
        answer_vector, vector_context = ask_vector_only(
            groq_client, qdrant, voyage_client, question
        )

        time.sleep(2)
        result_mg = engine.chat(question, skip_ingestion=True)
        answer_memgraph = result_mg["response"]
        memgraph_context = result_mg["context_used"]

        # ── RAGAS scoring ──
        # Plain LLM: only answer relevancy (no context to evaluate)
        time.sleep(2)
        plain_relevancy = compute_answer_relevancy(groq_client, question, answer_plain)

        # Vector RAG: all 4 metrics
        time.sleep(2)
        vec_faithfulness = compute_faithfulness(groq_client, answer_vector, vector_context)
        time.sleep(2)
        vec_relevancy = compute_answer_relevancy(groq_client, question, answer_vector)
        time.sleep(2)
        vec_ctx_precision = compute_context_precision(groq_client, question, vector_context)
        time.sleep(2)
        vec_ctx_recall = compute_context_recall(groq_client, reference, vector_context)

        # MemGraph: all 4 metrics
        time.sleep(2)
        mg_faithfulness = compute_faithfulness(groq_client, answer_memgraph, memgraph_context)
        time.sleep(2)
        mg_relevancy = compute_answer_relevancy(groq_client, question, answer_memgraph)
        time.sleep(2)
        mg_ctx_precision = compute_context_precision(groq_client, question, memgraph_context)
        time.sleep(2)
        mg_ctx_recall = compute_context_recall(groq_client, reference, memgraph_context)

        entry = {
            "question": question,
            "plain": {
                "answer_relevancy": plain_relevancy["score"],
            },
            "vector": {
                "faithfulness": vec_faithfulness["score"],
                "answer_relevancy": vec_relevancy["score"],
                "context_precision": vec_ctx_precision["score"],
                "context_recall": vec_ctx_recall["score"],
            },
            "memgraph": {
                "faithfulness": mg_faithfulness["score"],
                "answer_relevancy": mg_relevancy["score"],
                "context_precision": mg_ctx_precision["score"],
                "context_recall": mg_ctx_recall["score"],
            },
        }
        results.append(entry)

        # Per-question summary
        def avg_ragas(d):
            vals = [v for v in d.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else 0

        print(f"    Plain:    Relevancy={plain_relevancy['score']:.2f}")
        print(f"    Vector:   Faith={vec_faithfulness['score']:.2f} "
              f"Rel={vec_relevancy['score']:.2f} "
              f"CtxP={vec_ctx_precision['score']:.2f} "
              f"CtxR={vec_ctx_recall['score']:.2f} "
              f"Avg={avg_ragas(entry['vector']):.2f}")
        print(f"    MemGraph: Faith={mg_faithfulness['score']:.2f} "
              f"Rel={mg_relevancy['score']:.2f} "
              f"CtxP={mg_ctx_precision['score']:.2f} "
              f"CtxR={mg_ctx_recall['score']:.2f} "
              f"Avg={avg_ragas(entry['memgraph']):.2f}")

    engine.close()

    # ── Aggregate metrics ──
    n = len(results)
    if n == 0:
        if close_driver:
            neo4j_driver.close()
        return {"benchmark": "generation", "passed": False, "details": []}

    def avg_metric(mode: str, metric: str) -> float:
        vals = [r[mode].get(metric, 0) for r in results]
        return sum(vals) / len(vals)

    # Vector averages
    vec_avg = {
        "faithfulness": avg_metric("vector", "faithfulness"),
        "answer_relevancy": avg_metric("vector", "answer_relevancy"),
        "context_precision": avg_metric("vector", "context_precision"),
        "context_recall": avg_metric("vector", "context_recall"),
    }
    vec_avg["mean"] = sum(vec_avg.values()) / 4

    # MemGraph averages
    mg_avg = {
        "faithfulness": avg_metric("memgraph", "faithfulness"),
        "answer_relevancy": avg_metric("memgraph", "answer_relevancy"),
        "context_precision": avg_metric("memgraph", "context_precision"),
        "context_recall": avg_metric("memgraph", "context_recall"),
    }
    mg_avg["mean"] = sum(v for k, v in mg_avg.items() if k != "mean") / 4

    plain_relevancy_avg = avg_metric("plain", "answer_relevancy")

    # Win rate (MemGraph mean RAGAS > Vector mean RAGAS per question)
    memgraph_wins = 0
    for r in results:
        v_vals = [v for v in r["vector"].values() if isinstance(v, (int, float))]
        m_vals = [v for v in r["memgraph"].values() if isinstance(v, (int, float))]
        v_mean = sum(v_vals) / len(v_vals) if v_vals else 0
        m_mean = sum(m_vals) / len(m_vals) if m_vals else 0
        if m_mean >= v_mean:
            memgraph_wins += 1
    win_rate = memgraph_wins / n

    # ── Print results ──
    print(f"\n{'═' * 70}")
    print("GENERATION QUALITY RESULTS (RAGAS Metrics)")
    print(f"{'═' * 70}")

    print(f"\n  Average Scores (0.0 - 1.0, higher is better):")
    print(f"  {'Metric':<22} | {'Plain':>7} | {'Vector':>7} | {'MemGraph':>8}")
    print(f"  {'─' * 22}-+-{'─' * 7}-+-{'─' * 7}-+-{'─' * 8}")
    print(f"  {'Faithfulness':<22} | {'  n/a':>7} | {vec_avg['faithfulness']:>7.2f} | {mg_avg['faithfulness']:>8.2f}")
    print(f"  {'Answer Relevancy':<22} | {plain_relevancy_avg:>7.2f} | {vec_avg['answer_relevancy']:>7.2f} | {mg_avg['answer_relevancy']:>8.2f}")
    print(f"  {'Context Precision':<22} | {'  n/a':>7} | {vec_avg['context_precision']:>7.2f} | {mg_avg['context_precision']:>8.2f}")
    print(f"  {'Context Recall':<22} | {'  n/a':>7} | {vec_avg['context_recall']:>7.2f} | {mg_avg['context_recall']:>8.2f}")
    print(f"  {'─' * 22}-+-{'─' * 7}-+-{'─' * 7}-+-{'─' * 8}")
    print(f"  {'RAGAS Mean':<22} | {'  n/a':>7} | {vec_avg['mean']:>7.2f} | {mg_avg['mean']:>8.2f}")

    print(f"\n  MemGraph win rate: {memgraph_wins}/{n} ({win_rate:.0%})")

    print(f"\n  What each metric tells you:")
    if mg_avg["faithfulness"] > vec_avg["faithfulness"]:
        print(f"    Faithfulness: MemGraph hallucinates LESS than Vector RAG")
    else:
        print(f"    Faithfulness: Vector RAG hallucinates less (MemGraph's extra context may confuse)")
    if mg_avg["context_precision"] > vec_avg["context_precision"]:
        print(f"    Context Precision: Hybrid retrieval returns MORE relevant context")
    else:
        print(f"    Context Precision: Vector retrieval is more focused")
    if mg_avg["context_recall"] > vec_avg["context_recall"]:
        print(f"    Context Recall: Hybrid retrieval FINDS more of the needed information")
    else:
        print(f"    Context Recall: Vector retrieval covers ground truth better")

    # Pass/fail
    passed = mg_avg["mean"] > vec_avg["mean"]
    print(f"\n  {'─' * 50}")
    if passed:
        delta = mg_avg["mean"] - vec_avg["mean"]
        print(f"  PASS — MemGraph RAGAS mean ({mg_avg['mean']:.2f}) beats "
              f"Vector ({vec_avg['mean']:.2f}) by {delta:.2f}")
    else:
        print(f"  FAIL — MemGraph RAGAS mean ({mg_avg['mean']:.2f}) does not beat "
              f"Vector ({vec_avg['mean']:.2f})")

    if close_driver:
        neo4j_driver.close()

    return {
        "benchmark": "generation",
        "vector_ragas": vec_avg,
        "memgraph_ragas": mg_avg,
        "plain_relevancy": plain_relevancy_avg,
        "memgraph_win_rate": win_rate,
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
    run_generation_benchmark()
