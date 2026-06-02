"""
Phase 5: Side-by-Side Retrieval Comparison — Graph vs Vector

This script runs the same questions through both Neo4j (graph traversal)
and Qdrant (vector similarity) and prints the answers side by side.

The goal: see WHEN each approach wins.

- Graph wins on precise structural questions ("why was X rejected?",
  "what does Y depend on?") — it follows typed edges to exact answers.
- Vector wins on fuzzy semantic questions ("what are we unsure about?",
  "anything about databases?") — it finds relevant text by meaning.
- A hybrid system uses both: vector search for recall, graph for precision.
"""

import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient

load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")

# Qdrant + Voyage
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = "voyage-3-lite"
COLLECTION_NAME = "memgraph_messages"


def query_graph(driver, question_key: str) -> list[str]:
    """
    Run a targeted Cypher query for each question.

    Unlike vector search (one-size-fits-all similarity), graph queries
    must be tailored to the question's structure. This is both the
    strength (precision) and the weakness (you need to know what to ask).
    """
    queries = {
        "why_rejected_mongodb": {
            "cypher": """
                MATCH (actor)-[r:REJECTED]->(target)
                WHERE target.name_lower CONTAINS 'mongo'
                RETURN actor.name AS who,
                       target.name AS what,
                       r.relation AS reason,
                       r.source AS source_message
            """,
            "format": lambda recs: [
                f"{r['who']} rejected {r['what']} — \"{r['reason']}\" (from: \"{r['source_message']}\")"
                for r in recs
            ],
        },
        "backend_depends_on": {
            "cypher": """
                MATCH (component)-[r:DEPENDS_ON]->(dep)
                RETURN component.name AS component,
                       dep.name AS depends_on,
                       r.source AS source_message
            """,
            "format": lambda recs: [
                f"{r['component']} depends on {r['depends_on']} (from: \"{r['source_message']}\")"
                for r in recs
            ],
        },
        "uncertain_about": {
            "cypher": """
                MATCH (who)-[r:IS_UNCERTAIN_ABOUT]->(what)
                RETURN who.name AS who,
                       what.name AS topic,
                       r.relation AS detail,
                       r.source AS source_message
            """,
            "format": lambda recs: [
                f"{r['who']} uncertain about {r['topic']} — \"{r['detail']}\" (from: \"{r['source_message']}\")"
                for r in recs
            ],
        },
    }

    q = queries[question_key]
    with driver.session() as session:
        result = session.run(q["cypher"])
        records = [dict(r) for r in result]

    if not records:
        return ["(no graph results)"]
    return q["format"](records)


def query_vector(qdrant: QdrantClient, voyage_client: voyageai.Client, question: str, top_k: int = 2) -> list[str]:
    """
    Semantic nearest-neighbour search — same approach for every question.

    This is the vector store's advantage: you don't need to know the
    schema or write a query. Just ask in natural language and get
    the most semantically similar stored text back.
    """
    result = voyage_client.embed([question], model=EMBEDDING_MODEL, input_type="query")
    query_vector = result.embeddings[0]

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    if not results.points:
        return ["(no vector results)"]

    return [
        f"\"{p.payload['text']}\" (score: {p.score:.4f})"
        for p in results.points
    ]


def compare(
    question: str,
    graph_key: str,
    driver,
    qdrant: QdrantClient,
    voyage_client: voyageai.Client,
    analysis: str,
):
    """Print graph vs vector answers side by side for one question."""
    print(f'\n{"═" * 70}')
    print(f'  QUESTION: "{question}"')
    print(f'{"═" * 70}')

    # Graph answer
    graph_answers = query_graph(driver, graph_key)
    print(f"\n  GRAPH ANSWER (Neo4j Cypher traversal):")
    for a in graph_answers:
        print(f"    → {a}")

    # Vector answer
    vector_answers = query_vector(qdrant, voyage_client, question)
    print(f"\n  VECTOR ANSWER (Qdrant semantic search, top 2):")
    for a in vector_answers:
        print(f"    → {a}")

    # Analysis
    print(f"\n  ANALYSIS: {analysis}")


def main():
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in your .env file.")
        return

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    print("Connected to Neo4j and Qdrant. Running comparisons...\n")
    print("This queries the graph and vector stores built by 04_pipeline.py.")
    print("If you haven't run 04_pipeline.py first, do that now.")

    questions = [
        {
            "question": "Why did we reject MongoDB?",
            "graph_key": "why_rejected_mongodb",
            "analysis": (
                "GRAPH WINS. The graph returns the exact reason from the REJECTED "
                "edge ('rejected', with the source message). The vector store finds "
                "the relevant sentence but can't extract the causal 'why' — it just "
                "returns similar text. For 'why' questions, structured edges beat "
                "semantic similarity."
            ),
        },
        {
            "question": "What does our backend depend on?",
            "graph_key": "backend_depends_on",
            "analysis": (
                "GRAPH WINS. The DEPENDS_ON edge gives a precise, structured answer: "
                "API depends on PostgreSQL. The vector store finds related sentences "
                "but can't distinguish dependency from mere mention. Graph traversal "
                "is ideal for relationship queries."
            ),
        },
        {
            "question": "What are we uncertain about?",
            "graph_key": "uncertain_about",
            "analysis": (
                "TIE / VECTOR SLIGHTLY BETTER for discovery. The graph finds the "
                "exact IS_UNCERTAIN_ABOUT edge, which is precise. But the vector "
                "store might also surface sentences with uncertain language that the "
                "LLM didn't extract as triples — making it better for exploratory, "
                "open-ended questions where you don't know the exact structure."
            ),
        },
    ]

    for i, q in enumerate(questions):
        if i > 0:
            # Pace Voyage API calls
            print("\n  (waiting 21s for Voyage rate limit...)")
            time.sleep(21)
        compare(
            q["question"],
            q["graph_key"],
            neo4j_driver,
            qdrant,
            voyage_client,
            q["analysis"],
        )

    print(f'\n{"═" * 70}')
    print("  SUMMARY")
    print(f'{"═" * 70}')
    print("""
  Graph strengths:   Precise answers, causal reasoning ("why?"),
                     relationship traversal ("depends on?"),
                     structured facts with provenance.

  Vector strengths:  Fuzzy matching, discovery, no schema needed,
                     works with any question phrasing,
                     finds relevant info even without exact structure.

  Hybrid approach:   Use vector search to FIND relevant context,
                     then graph traversal to REASON over it.
                     This is the MemGraph architecture.
""")

    neo4j_driver.close()


if __name__ == "__main__":
    main()
