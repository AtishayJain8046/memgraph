"""
Phase 4: Unified Ingestion Pipeline — From Natural Language to Both Stores

This script ties everything together. One function call:
  process_message("We decided on PostgreSQL")
    1. Calls the LLM to extract structured triples
    2. Writes each triple to Neo4j as nodes + typed edges
    3. Embeds the original text and writes it to Qdrant
    4. Prints a summary of what happened

This is the core of the MemGraph system: every message gets dual-indexed.
The graph captures structure (who decided what, why, dependencies).
The vector store captures meaning (for fuzzy semantic retrieval).

Later, at query time, you can choose the right store for the right question.
"""

import os
import time
import uuid
import importlib.util
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Python can't import modules starting with a digit ("03_extraction"),
# so we use importlib to load it by file path. This avoids needing a
# duplicate file with a different name.
_spec = importlib.util.spec_from_file_location(
    "extraction", os.path.join(os.path.dirname(__file__), "03_extraction.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
extract_triples = _module.extract_triples

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
EMBEDDING_DIM = 512
COLLECTION_NAME = "memgraph_messages"


def setup_qdrant(qdrant: QdrantClient):
    """Create the Qdrant collection if it doesn't exist."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"  Created Qdrant collection '{COLLECTION_NAME}'")


def write_triples_to_neo4j(driver, triples: list[dict], source_message: str):
    """
    Write extracted triples to Neo4j as nodes and typed edges.

    Strategy: MERGE (not CREATE) on nodes so re-running with the same
    entities doesn't create duplicates. Each node gets a normalized name
    (lowercased) for matching, plus the original casing for display.
    """
    with driver.session() as session:
        for t in triples:
            subj = t["subject"]
            obj = t["object"]
            edge_type = t["edge_type"]
            relation = t["relation"]
            confidence = t["confidence"]

            # MERGE finds or creates nodes by normalized name,
            # then creates the relationship between them.
            cypher = f"""
                MERGE (s:Entity {{name_lower: $subj_lower}})
                  ON CREATE SET s.name = $subj, s.created = timestamp()
                MERGE (o:Entity {{name_lower: $obj_lower}})
                  ON CREATE SET o.name = $obj, o.created = timestamp()
                CREATE (s)-[r:{edge_type} {{
                    relation: $relation,
                    confidence: $confidence,
                    source: $source
                }}]->(o)
            """
            session.run(
                cypher,
                subj=subj,
                subj_lower=subj.lower(),
                obj=obj,
                obj_lower=obj.lower(),
                relation=relation,
                confidence=confidence,
                source=source_message,
            )


def embed_with_retry(voyage_client: voyageai.Client, text: str) -> list[float]:
    """Embed text with retry for Voyage rate limits."""
    for attempt in range(3):
        try:
            result = voyage_client.embed([text], model=EMBEDDING_MODEL, input_type="document")
            return result.embeddings[0]
        except Exception as e:
            if "RateLimitError" in type(e).__name__ and attempt < 2:
                wait = 25 * (attempt + 1)
                print(f"    (rate limited, waiting {wait}s...)")
                time.sleep(wait)
            else:
                raise


def write_to_qdrant(qdrant: QdrantClient, voyage_client: voyageai.Client, message: str):
    """Embed the original message and store it in Qdrant."""
    embedding = embed_with_retry(voyage_client, message)

    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": message},
            )
        ],
    )
    return point_id


def process_message(
    message: str,
    neo4j_driver,
    qdrant: QdrantClient,
    voyage_client: voyageai.Client,
):
    """
    The core pipeline function. Takes a natural language message and:
      1. Extracts structured triples via LLM
      2. Writes confirmed triples to Neo4j
      3. Embeds and stores the original text in Qdrant

    This dual-write means the same information is indexed two ways:
    - Graph: precise, structured, traversable
    - Vector: fuzzy, semantic, similarity-based
    """
    print(f'\n{"─" * 60}')
    print(f'Processing: "{message}"')
    print(f'{"─" * 60}')

    # Step 1: Extract triples
    confirmed, rejected = extract_triples(message)
    print(f"  Extraction: {len(confirmed)} confirmed, {len(rejected)} below threshold")

    # Step 2: Write to Neo4j
    if confirmed:
        write_triples_to_neo4j(neo4j_driver, confirmed, message)
        for t in confirmed:
            print(f"    → Neo4j: ({t['subject']}) --[{t['edge_type']}]--> ({t['object']})")
    else:
        print("    → Neo4j: nothing written (no confirmed triples)")

    # Step 3: Write to Qdrant
    point_id = write_to_qdrant(qdrant, voyage_client, message)
    print(f"    → Qdrant: stored embedding (id: {point_id[:8]}...)")

    # Summary
    if rejected:
        print(f"  Filtered out:")
        for t in rejected:
            print(f"    ✗ ({t['subject']}) --[{t['edge_type']}]--> ({t['object']}) conf={t['confidence']}")


def main():
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in your .env file.")
        return

    # Connect to everything
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("Connected to Neo4j")

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Connected to Qdrant")

    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print("Voyage client ready")

    # Clear previous data for clean demo
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    setup_qdrant(qdrant)
    print("Cleared previous data.")

    # Process 5 messages in sequence — each builds on the knowledge graph
    messages = [
        "We are choosing a database for the new service",
        "After evaluation we decided on PostgreSQL",
        "MongoDB was rejected — writes were too slow under load",
        "Our backend API now depends on PostgreSQL",
        "We are uncertain about whether to use a connection pooler",
    ]

    for i, msg in enumerate(messages):
        # Pace Voyage API calls (free tier: 3 RPM)
        if i > 0:
            print("\n  (waiting 21s for Voyage rate limit...)")
            time.sleep(21)
        process_message(msg, neo4j_driver, qdrant, voyage_client)

    # Show final graph state
    print(f'\n{"═" * 60}')
    print("FINAL GRAPH STATE")
    print(f'{"═" * 60}')
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (s)-[r]->(o)
            RETURN s.name AS subject, type(r) AS edge_type, o.name AS object,
                   r.relation AS relation
            ORDER BY s.name
        """)
        for rec in result:
            print(f"  ({rec['subject']}) --[{rec['edge_type']}: {rec['relation']}]--> ({rec['object']})")

    neo4j_driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
