"""
Phase 2: Vector Store Basics — Semantic Search with Qdrant + Voyage Embeddings

This script teaches how vector databases find information by MEANING, not by
exact keyword matches. The workflow:

  1. Turn each sentence into a vector (list of 1024 floats) using Voyage AI's
     embedding model. Sentences with similar meaning get similar vectors.
  2. Store those vectors in Qdrant alongside the original text.
  3. To search, embed the QUERY the same way, then find the stored vectors
     closest to it (nearest-neighbour search).

Key takeaway: vector search finds "database selection was finalized" when
you ask "what database did we pick?" — even though they share zero keywords.
This is the superpower, but also the limit: it can't tell you structured
facts like "MongoDB was rejected BECAUSE it was too slow." That's what the
graph is for.
"""

import os
import time
from dotenv import load_dotenv
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_DIM = 512
COLLECTION_NAME = "memgraph_basics"


def _embed_with_retry(client: voyageai.Client, texts: list[str], input_type: str) -> list[list[float]]:
    """Embed with retry on rate limit. Voyage free tier is 3 RPM."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = client.embed(texts, model=EMBEDDING_MODEL, input_type=input_type)
            return result.embeddings
        except Exception as e:
            if "RateLimitError" in type(e).__name__ and attempt < max_retries - 1:
                wait = 25 * (attempt + 1)
                print(f"  (rate limited, waiting {wait}s...)")
                time.sleep(wait)
            else:
                raise


def get_embedding(client: voyageai.Client, text: str) -> list[float]:
    """
    Turn text into a vector using Voyage AI's embedding model.
    voyage-3-lite produces 512-dim vectors — smaller than OpenAI's 1536
    but still good quality, and pairs well with the Anthropic ecosystem.
    """
    return _embed_with_retry(client, [text], "document")[0]


def get_query_embedding(client: voyageai.Client, text: str) -> list[float]:
    """
    Voyage distinguishes between 'document' and 'query' embeddings.
    Queries are embedded slightly differently to optimize retrieval.
    """
    return _embed_with_retry(client, [text], "query")[0]


def setup_qdrant(qdrant: QdrantClient):
    """Create a fresh collection. A collection is like a table — it holds
    vectors of a fixed dimensionality with a chosen distance metric."""
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}-dim, cosine distance)\n")


def store_sentences(qdrant: QdrantClient, voyage_client: voyageai.Client):
    """Embed 5 sentences and store them in Qdrant with the original text as payload."""
    sentences = [
        "we chose postgres for the database",
        "database selection was finalized last week",
        "mongodb was too slow for our use case",
        "we need a fast caching layer",
        "redis was picked for session storage",
    ]

    # Batch-embed all sentences at once (more efficient than one-by-one)
    embeddings = _embed_with_retry(voyage_client, sentences, "document")

    points = []
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={"text": sentence},
            )
        )
        print(f"  Embedded [{i}]: \"{sentence}\"")
        print(f"           Vector preview: [{embedding[0]:.4f}, {embedding[1]:.4f}, ... {embedding[-1]:.4f}]")

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nStored {len(points)} vectors in Qdrant.\n")


def search(qdrant: QdrantClient, voyage_client: voyageai.Client, query: str, top_k: int = 3):
    """
    Semantic search: embed the query, then find the top_k closest stored vectors.

    Cosine similarity ranges from 0 (unrelated) to 1 (identical meaning).
    Scores above ~0.75 usually indicate strong semantic relevance.
    """
    print(f"=== Query: \"{query}\" ===")
    query_vector = get_query_embedding(voyage_client, query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    for i, point in enumerate(results.points):
        text = point.payload["text"]
        score = point.score
        print(f"  #{i+1} (score: {score:.4f}): \"{text}\"")

    # "database selection was finalized last week" scores high even though
    # it shares zero content words with "what database did we pick?" —
    # the embedding model understands that "selection was finalized" and
    # "did we pick" describe the same concept (making a choice). This is
    # semantic similarity: meaning matters, not keywords.
    print()


def main():
    if not VOYAGE_API_KEY:
        print("ERROR: Set VOYAGE_API_KEY in your .env file.")
        print("  Get one at: https://dash.voyageai.com/")
        return

    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}\n")

    setup_qdrant(qdrant)
    store_sentences(qdrant, voyage_client)

    queries = [
        "what database did we pick?",
        "why was something rejected?",
        "what do we need for caching?",
    ]
    for i, q in enumerate(queries):
        if i > 0:
            # Voyage free tier: 3 RPM. Batch embed above uses 1 call,
            # so we pace the 3 query calls ~21s apart to stay under limit.
            print("  (waiting 21s for Voyage rate limit...)\n")
            time.sleep(21)
        search(qdrant, voyage_client, q)


if __name__ == "__main__":
    main()
