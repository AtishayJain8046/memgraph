"""
Phase 11: MemGraph AI — Streamlit Chat UI

This script teaches: how to wrap a complex AI pipeline (retrieval, contradiction
detection, epistemic state, generation) in an interactive chat interface using
Streamlit. Two-panel layout: chat on the left, live knowledge state on the right.

Run: streamlit run app.py
Prereqs: Neo4j + Qdrant running, ANTHROPIC_API_KEY + VOYAGE_API_KEY set in .env
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "memgraph-week1")
load_dotenv(os.path.join(WEEK1_DIR, ".env"))
sys.path.insert(0, WEEK1_DIR)

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

st.set_page_config(page_title="MemGraph AI", layout="wide")


def check_connections() -> dict:
    """Check if Neo4j and Qdrant are reachable."""
    status = {"neo4j": False, "qdrant": False}

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        status["neo4j"] = True
    except Exception:
        pass

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        client.get_collections()
        status["qdrant"] = True
    except Exception:
        pass

    return status


@st.cache_resource
def get_chat_engine():
    """Create a single ChatEngine instance shared across reruns."""
    from core.chat import ChatEngine
    return ChatEngine()


def render_knowledge_panel(state: dict):
    """Render the right-side knowledge state panel."""
    st.subheader("Knowledge State")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Decisions", len(state.get("decisions", [])))
    col_m2.metric("Uncertainties", len(state.get("uncertainties", [])))

    # Decisions
    decisions = state.get("decisions", [])
    if decisions:
        st.markdown("**Decisions**")
        for d in decisions:
            context = d.get("context", "")
            st.markdown(f"- **{d['name']}** — {context}")

    # Rejections
    rejections = state.get("rejections", [])
    if rejections:
        st.markdown("**Rejected**")
        for r in rejections:
            st.markdown(f"- ~~{r['name']}~~ — {r.get('reason', '')}")

    # Uncertainties
    uncertainties = state.get("uncertainties", [])
    if uncertainties:
        st.markdown("**Uncertainties**")
        for u in uncertainties:
            st.markdown(f"- :grey_question: {u['topic']} — {u.get('note', '')}")

    # Dependencies
    dependencies = state.get("dependencies", [])
    if dependencies:
        st.markdown("**Dependencies**")
        for d in dependencies:
            st.markdown(f"- {d['component']} :arrow_right: {d['depends_on']}")

    # Contradictions
    contradictions = state.get("contradictions", [])
    if contradictions:
        st.markdown("**:red[Contradictions]**")
        for c in contradictions:
            st.error(f"'{c['new']}' vs '{c['old']}': {c.get('reason', '')}", icon="⚠️")

    if st.button("Refresh state"):
        st.session_state["knowledge_state"] = _fetch_state()
        st.rerun()


def _fetch_state() -> dict:
    """Fetch the current epistemic state from the graph."""
    try:
        from core.state import get_user_state
        return get_user_state()
    except Exception as e:
        return {"decisions": [], "rejections": [], "uncertainties": [],
                "dependencies": [], "contradictions": [], "open_questions": [],
                "summary": f"(Error fetching state: {e})"}


def main():
    # Check connections
    conn = check_connections()

    if not conn["neo4j"]:
        st.error("Neo4j not connected — start Docker (`docker-compose up -d`)")
    if not conn["qdrant"]:
        st.error("Qdrant not connected — start Docker (`docker-compose up -d`)")

    services_ok = conn["neo4j"] and conn["qdrant"]

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "knowledge_state" not in st.session_state:
        st.session_state["knowledge_state"] = _fetch_state() if services_ok else {}

    # Two-column layout
    left_col, right_col = st.columns([2, 1])

    # Right column: knowledge state
    with right_col:
        render_knowledge_panel(st.session_state["knowledge_state"])

    # Left column: chat
    with left_col:
        st.title("MemGraph AI")
        st.caption("hybrid graph + vector memory")

        # Render chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                # Contradiction alerts above assistant messages
                if msg["role"] == "assistant" and msg.get("alerts"):
                    for alert in msg["alerts"]:
                        existing = alert.get("existing", "")[:80]
                        explanation = alert.get("explanation", "")
                        st.warning(
                            f"Your statement conflicts with a prior position: "
                            f"'{existing}'. {explanation}",
                            icon="⚠️",
                        )

                st.markdown(msg["content"])

                # Context expander for assistant messages
                if msg["role"] == "assistant" and msg.get("context_used"):
                    with st.expander("Context used →"):
                        for ctx in msg["context_used"]:
                            source = ctx.get("source", "unknown").upper()
                            content = ctx.get("content", "")

                            badge = f"**[{source}]**"
                            line = f"{badge} {content}"

                            if ctx.get("edge_type"):
                                line += f"  \n*Edge: {ctx['edge_type']}*"
                            if ctx.get("reason"):
                                line += f"  \n*Reason: {ctx['reason']}*"
                            if ctx.get("vector_score") is not None:
                                line += f"  \n*Similarity: {ctx['vector_score']:.3f}*"
                            if ctx.get("rrf_score") is not None:
                                line += f"  \n*RRF: {ctx['rrf_score']:.5f}*"

                            st.markdown(line)
                            st.divider()

        # Chat input
        if prompt := st.chat_input("Ask about your decisions...", disabled=not services_ok):
            # Show user message immediately
            st.session_state["messages"].append({
                "role": "user",
                "content": prompt,
                "context_used": [],
                "alerts": [],
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        engine = get_chat_engine()
                        is_question = prompt.strip().endswith("?") or prompt.lower().startswith(
                            ("what", "why", "how", "when", "where", "who", "which", "is ", "are ", "do ", "does ", "did ", "can ", "could ")
                        )
                        result = engine.chat(
                            prompt,
                            conversation_history=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state["messages"][:-1]
                            ],
                            skip_ingestion=is_question,
                        )

                        # Show alerts
                        alerts = result.get("contradiction_alerts", [])
                        for alert in alerts:
                            existing = alert.get("existing", "")[:80]
                            explanation = alert.get("explanation", "")
                            st.warning(
                                f"Your statement conflicts with a prior position: "
                                f"'{existing}'. {explanation}",
                                icon="⚠️",
                            )

                        # Show response
                        st.markdown(result["response"])

                        # Context expander
                        context_used = result.get("context_used", [])
                        if context_used:
                            with st.expander("Context used →"):
                                for ctx in context_used:
                                    source = ctx.get("source", "unknown").upper()
                                    content = ctx.get("content", "")
                                    line = f"**[{source}]** {content}"
                                    if ctx.get("edge_type"):
                                        line += f"  \n*Edge: {ctx['edge_type']}*"
                                    if ctx.get("reason"):
                                        line += f"  \n*Reason: {ctx['reason']}*"
                                    st.markdown(line)
                                    st.divider()

                        # Save to history
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": result["response"],
                            "context_used": context_used,
                            "alerts": alerts,
                        })

                        # Update knowledge state
                        if result.get("epistemic_state"):
                            st.session_state["knowledge_state"] = result["epistemic_state"]

                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": error_msg,
                            "context_used": [],
                            "alerts": [],
                        })

            st.rerun()


if __name__ == "__main__":
    main()


# ─── LEARNING NOTE ──────────────────────────────────────────────────────────
# What this file proved:
#   A complex AI pipeline (extraction, contradiction detection, hybrid
#   retrieval, epistemic state, generation) can be wrapped in an
#   interactive UI with ~200 lines of Streamlit code. The two-panel
#   layout makes both the conversation and the knowledge state visible.
#
# What would break without this:
#   The pipeline would only be usable via scripts. A chat UI makes the
#   system interactive and lets you explore how each message updates
#   the knowledge graph in real time.
#
# What to build next:
#   Week 3 — multi-user support, temporal reasoning, graph visualization,
#   and production deployment with authentication.
# ────────────────────────────────────────────────────────────────────────────
