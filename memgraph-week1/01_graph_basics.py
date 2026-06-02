"""
Phase 1: Graph Fundamentals — Building a Knowledge Graph in Neo4j

This script teaches you how graph databases represent knowledge differently
from relational databases. Instead of rows in tables, you have:

  NODES  = entities (people, decisions, technologies)
  EDGES  = typed relationships between them (DECIDED, REJECTED, DEPENDS_ON)

We model one real scenario: "The team chose PostgreSQL, rejected MongoDB
(too slow), and the backend depends on PostgreSQL."

Key takeaway: graph queries follow the shape of the question. "Why was
MongoDB rejected?" becomes a pattern match along a REJECTED edge — no
joins, no subqueries.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")


def connect():
    """Connect to Neo4j and verify the connection is alive."""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {URI}")
    return driver


def clear_database(driver):
    """Wipe all nodes and edges so the script is idempotent (safe to re-run)."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing data.\n")


def create_knowledge_graph(driver):
    """
    Build a small knowledge graph representing a team's database decision.

    Graph structure:
        (team:Actor) --DECIDED--> (pg:Technology)
        (team:Actor) --REJECTED {reason: "too slow"}--> (mongo:Technology)
        (backend:Component) --DEPENDS_ON--> (pg:Technology)

    Notice: nodes have labels (Actor, Technology, Component) and properties
    (name, type). Edges have types (DECIDED, REJECTED) and can carry
    properties too (reason).
    """
    cypher = """
    // Create the actors and technologies as nodes
    CREATE (team:Actor {name: "The Team", role: "engineering"})

    CREATE (pg:Technology {name: "PostgreSQL", type: "database"})
    CREATE (mongo:Technology {name: "MongoDB", type: "database"})
    CREATE (backend:Component {name: "Backend API", type: "service"})

    // Create typed relationships between them
    CREATE (team)-[:DECIDED {
        context: "database selection",
        date: "2024-01-15"
    }]->(pg)

    CREATE (team)-[:REJECTED {
        reason: "writes were too slow under load",
        context: "database selection"
    }]->(mongo)

    CREATE (backend)-[:DEPENDS_ON {
        since: "2024-01-15"
    }]->(pg)

    RETURN
        team.name AS actor,
        pg.name AS chosen,
        mongo.name AS rejected
    """
    with driver.session() as session:
        result = session.run(cypher)
        record = result.single()
        print("=== Knowledge Graph Created ===")
        print(f"  Actor:    {record['actor']}")
        print(f"  Chosen:   {record['chosen']}")
        print(f"  Rejected: {record['rejected']}")
        print()


def run_queries(driver):
    """
    Five Cypher queries that demonstrate different graph traversal patterns.

    Each query follows the shape of the question:
      - "all decisions"       → follow DECIDED edges
      - "why rejected?"       → follow REJECTED edge, read its reason property
      - "depends on what?"    → follow DEPENDS_ON edge
      - "everything about X?" → match all edges touching a node
    """
    queries = [
        {
            "title": "1. What decisions were made?",
            "explanation": "Follow all DECIDED edges to find what was chosen.",
            "cypher": """
                MATCH (actor:Actor)-[d:DECIDED]->(tech:Technology)
                RETURN actor.name AS who,
                       tech.name AS chose,
                       d.context AS context
            """,
        },
        {
            "title": "2. What options were rejected?",
            "explanation": "Follow all REJECTED edges. The reason is stored ON the edge, not the node.",
            "cypher": """
                MATCH (actor:Actor)-[r:REJECTED]->(tech:Technology)
                RETURN actor.name AS who,
                       tech.name AS rejected,
                       r.reason AS reason
            """,
        },
        {
            "title": "3. Why was MongoDB rejected?",
            "explanation": "Pattern match: find the REJECTED edge landing on the MongoDB node.",
            "cypher": """
                MATCH (actor)-[r:REJECTED]->(tech:Technology {name: "MongoDB"})
                RETURN r.reason AS reason
            """,
        },
        {
            "title": "4. What does the backend depend on?",
            "explanation": "Traverse DEPENDS_ON from the Backend API component.",
            "cypher": """
                MATCH (c:Component {name: "Backend API"})-[:DEPENDS_ON]->(dep)
                RETURN c.name AS component,
                       dep.name AS depends_on,
                       labels(dep)[0] AS dep_type
            """,
        },
        {
            "title": "5. Everything connected to PostgreSQL (any direction, any edge type)",
            "explanation": "The -- pattern matches edges in BOTH directions. No arrow = bidirectional.",
            "cypher": """
                MATCH (pg:Technology {name: "PostgreSQL"})-[r]-(connected)
                RETURN connected.name AS entity,
                       type(r) AS relationship,
                       labels(connected)[0] AS entity_type
            """,
        },
    ]

    for q in queries:
        print(f"=== {q['title']} ===")
        print(f"  ({q['explanation']})")
        with driver.session() as session:
            result = session.run(q["cypher"])
            records = list(result)
            if not records:
                print("  No results.")
            for rec in records:
                row = {k: rec[k] for k in rec.keys()}
                print(f"  → {row}")
        print()


def main():
    driver = connect()
    try:
        clear_database(driver)
        create_knowledge_graph(driver)
        run_queries(driver)
    finally:
        driver.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
