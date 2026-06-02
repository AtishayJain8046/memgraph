"""
Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

This module teaches: how to combine results from two fundamentally different
retrieval systems (graph traversal + vector similarity) into a single ranked
list using RRF. The key insight is that rank position is comparable across
systems even when raw scores are not.

Architecture role: this is the retrieval layer that feeds into the LLM
generation step. Every question the user asks goes through this function
to gather context from both stores before the LLM answers.
"""

import os
import re
import time
import functools
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = "voyage-3-lite"
COLLECTION_NAME = "memgraph_messages"

# RRF smoothing constant. Lower K = top ranks dominate more.
# k=60 is standard for large result sets. We use k=20 because our
# result sets are small (10-20 items) and we want the top-ranked
# results from each source to have stronger influence.
RRF_K = 20

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she",
    "her", "it", "its", "they", "them", "their",
}


def retry(max_attempts=3, base_delay=2):
    """Retry decorator with exponential backoff for API calls."""
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


QUERY_EDGE_TYPE_MAP = {
    "reject": "REJECTED",
    "rejected": "REJECTED",
    "decide": "DECIDED",
    "decided": "DECIDED",
    "chosen": "DECIDED",
    "chose": "DECIDED",
    "depend": "DEPENDS_ON",
    "depends": "DEPENDS_ON",
    "dependency": "DEPENDS_ON",
    "dependencies": "DEPENDS_ON",
    "uncertain": "IS_UNCERTAIN_ABOUT",
    "unsure": "IS_UNCERTAIN_ABOUT",
    "contradict": "CONTRADICTS",
    "conflict": "CONTRADICTS",
}


def extract_keywords(query: str) -> list[str]:
    """
    Pull meaningful keywords from a natural language query.
    Strips stop words and short tokens so the Cypher CONTAINS clauses
    match on entity names rather than noise like 'the' or 'did'.
    """
    words = re.findall(r'[a-zA-Z]+', query.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return keywords if keywords else words[:3]


def infer_edge_types(query: str) -> list[str]:
    """
    Map question words to graph edge types. 'Why rejected?' -> REJECTED,
    'depends on?' -> DEPENDS_ON, 'uncertain?' -> IS_UNCERTAIN_ABOUT.
    This lets graph queries match on edge structure, not just node names.
    """
    words = re.findall(r'[a-zA-Z]+', query.lower())
    edge_types = []
    for w in words:
        if w in QUERY_EDGE_TYPE_MAP:
            et = QUERY_EDGE_TYPE_MAP[w]
            if et not in edge_types:
                edge_types.append(et)
    return edge_types


def query_graph(driver, query: str) -> list[dict]:
    """
    Run keyword-based Cypher queries: 1-hop exact matches + 2-hop traversal.
    Also queries by edge type when the question implies one (e.g. "uncertain" -> IS_UNCERTAIN_ABOUT).
    Returns a ranked list of graph results with edge metadata.
    """
    keywords = extract_keywords(query)
    edge_types = infer_edge_types(query)
    results = []
    seen_contents = set()

    with driver.session() as session:
        # Edge-type queries: when the question maps to a specific relationship type,
        # fetch ALL edges of that type. This catches "what are we uncertain about?"
        # even when no node name matches the keyword "uncertain".
        for et in edge_types:
            records = session.run("""
                MATCH (n)-[r]->(m)
                WHERE type(r) = $edge_type
                RETURN n.name AS subject, m.name AS object,
                       type(r) AS rel_type, r.relation AS reason,
                       r.source AS source_text, r.confidence AS confidence
                LIMIT 10
            """, edge_type=et)

            for rec in records:
                content = f"({rec['subject']}) --[{rec['rel_type']}]--> ({rec['object']})"
                if rec['reason']:
                    content += f" | reason: {rec['reason']}"
                if rec['source_text']:
                    content += f" | from: \"{rec['source_text']}\""

                if content not in seen_contents:
                    seen_contents.add(content)
                    results.append({
                        "source": "graph",
                        "content": content,
                        "edge_type": rec["rel_type"],
                        "reason": rec["reason"] or "",
                        "confidence": rec["confidence"],
                    })

        for keyword in keywords:
            # 1-hop: direct matches on node names + source text
            records = session.run("""
                MATCH (n)-[r]->(m)
                WHERE toLower(n.name) CONTAINS $keyword
                   OR toLower(m.name) CONTAINS $keyword
                   OR toLower(r.source) CONTAINS $keyword
                   OR toLower(r.relation) CONTAINS $keyword
                RETURN n.name AS subject, m.name AS object,
                       type(r) AS rel_type, r.relation AS reason,
                       r.source AS source_text, r.confidence AS confidence
                LIMIT 10
            """, keyword=keyword.lower())

            for rec in records:
                content = f"({rec['subject']}) --[{rec['rel_type']}]--> ({rec['object']})"
                if rec['reason']:
                    content += f" | reason: {rec['reason']}"
                if rec['source_text']:
                    content += f" | from: \"{rec['source_text']}\""

                if content not in seen_contents:
                    seen_contents.add(content)
                    results.append({
                        "source": "graph",
                        "content": content,
                        "edge_type": rec["rel_type"],
                        "reason": rec["reason"] or "",
                        "confidence": rec["confidence"],
                    })

            # 2-hop: traversal through intermediate nodes
            records_2hop = session.run("""
                MATCH (n)-[r1]-(m)-[r2]-(p)
                WHERE toLower(n.name) CONTAINS $keyword
                RETURN n.name AS start_node, type(r1) AS rel1,
                       m.name AS mid_node, type(r2) AS rel2,
                       p.name AS end_node,
                       r1.source AS source1, r2.source AS source2
                LIMIT 20
            """, keyword=keyword.lower())

            for rec in records_2hop:
                content = (
                    f"({rec['start_node']}) --[{rec['rel1']}]--> ({rec['mid_node']}) "
                    f"--[{rec['rel2']}]--> ({rec['end_node']})"
                )
                if content not in seen_contents:
                    seen_contents.add(content)
                    results.append({
                        "source": "graph",
                        "content": content,
                        "edge_type": f"{rec['rel1']}->{rec['rel2']}",
                        "reason": "",
                        "confidence": None,
                    })

    # Assign ranks (1-indexed)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


@retry(max_attempts=3, base_delay=25)
def query_vector(qdrant: QdrantClient, voyage_client: voyageai.Client,
                 query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic nearest-neighbour search via Qdrant.
    Returns ranked results with cosine similarity scores.
    """
    embedding = voyage_client.embed(
        [query], model=EMBEDDING_MODEL, input_type="query"
    ).embeddings[0]

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for i, point in enumerate(search_results.points):
        results.append({
            "source": "vector",
            "content": point.payload.get("text", ""),
            "rank": i + 1,
            "score": point.score,
        })

    return results


def normalize_for_matching(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy content matching across sources."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def fuse_rrf(graph_results: list[dict], vector_results: list[dict],
             top_k: int = 5) -> list[dict]:
    """
    Reciprocal Rank Fusion: merge two ranked lists into one.

    For each unique result:
      rrf_score = 1/(graph_rank + K) + 1/(vector_rank + K)

    Results in both lists get score contributions from both,
    making cross-source agreement the strongest ranking signal.
    """
    # Build a lookup: normalized content -> result data
    merged = {}

    for r in graph_results:
        key = normalize_for_matching(r["content"])
        merged[key] = {
            "content": r["content"],
            "graph_rank": r["rank"],
            "vector_rank": None,
            "edge_type": r.get("edge_type", ""),
            "reason": r.get("reason", ""),
            "vector_score": None,
        }

    for r in vector_results:
        key = normalize_for_matching(r["content"])
        # Check if this vector result matches any graph result by substring overlap
        matched_key = None
        for gkey in merged:
            # A vector result's text often appears as a substring in the graph's source_text
            if key in gkey or gkey in key:
                matched_key = gkey
                break
            # Also match if significant words overlap
            gwords = set(gkey.split())
            vwords = set(key.split())
            overlap = gwords & vwords - STOP_WORDS
            if len(overlap) >= 3:
                matched_key = gkey
                break

        if matched_key:
            merged[matched_key]["vector_rank"] = r["rank"]
            merged[matched_key]["vector_score"] = r.get("score")
        else:
            merged[key] = {
                "content": r["content"],
                "graph_rank": None,
                "vector_rank": r["rank"],
                "edge_type": "",
                "reason": "",
                "vector_score": r.get("score"),
            }

    # Compute RRF scores
    fused = []
    for key, data in merged.items():
        graph_contribution = 1 / (data["graph_rank"] + RRF_K) if data["graph_rank"] else 0
        vector_contribution = 1 / (data["vector_rank"] + RRF_K) if data["vector_rank"] else 0
        rrf_score = graph_contribution + vector_contribution

        if data["graph_rank"] and data["vector_rank"]:
            source = "both"
        elif data["graph_rank"]:
            source = "graph"
        else:
            source = "vector"

        result = {
            "source": source,
            "content": data["content"],
            "rrf_score": rrf_score,
        }
        if data["edge_type"]:
            result["edge_type"] = data["edge_type"]
        if data["reason"]:
            result["reason"] = data["reason"]
        if data["vector_score"] is not None:
            result["vector_score"] = data["vector_score"]

        fused.append(result)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused[:top_k]


def retrieve(query: str, top_k: int = 5,
             neo4j_driver=None, qdrant=None, voyage_client=None) -> list[dict]:
    """
    The main retrieval function. Queries both stores, fuses with RRF.

    If client objects are not provided, creates them from env vars.
    Returns top_k results sorted by RRF score.
    """
    close_driver = False

    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True
    if qdrant is None:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if voyage_client is None:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    print(f"  Keywords extracted: {extract_keywords(query)}")

    graph_results = query_graph(neo4j_driver, query)
    print(f"  Graph returned {len(graph_results)} results")

    vector_results = query_vector(qdrant, voyage_client, query)
    print(f"  Vector returned {len(vector_results)} results")

    fused = fuse_rrf(graph_results, vector_results, top_k=top_k)

    if close_driver:
        neo4j_driver.close()

    return fused, graph_results, vector_results


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   RRF lets you combine ranked lists from incompatible scoring systems.
#   The k=60 smoothing constant makes cross-source agreement more important
#   than raw rank in either source alone.
#
# What would break without this:
#   You'd have to query graph and vector separately and manually decide which
#   answer to trust. The LLM generation layer would get inconsistent context
#   depending on which store happened to return better results.
#
# What to build next:
#   07_benchmark.py to prove hybrid retrieval actually beats each source alone.
#   If it doesn't, the RRF parameters or keyword extraction need tuning.
# ────────────────────────────────────────────────────────────────────────────
