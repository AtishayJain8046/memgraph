# MemGraph 🧠

**A hybrid memory layer for LLMs that combines a knowledge graph with vector search — so AI assistants can actually remember, reason over past decisions, and flag contradictions.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.26-008CC1?logo=neo4j&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.13-DC244C)
![Llama 3.3](https://img.shields.io/badge/Llama_3.3_70B-Groq-F55036)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Problem

Most AI assistants are stateless — they forget everything between sessions. The common fix is **vector RAG**: embed past messages, retrieve semantically similar ones. But pure vector search retrieves *similar text* without understanding **structure** — it can't tell you what was *decided* vs. *rejected*, what *depends on* what, or when a new statement *contradicts* an old decision.

**MemGraph's hypothesis:** combining a **knowledge graph** (typed, structural relationships) with a **vector store** (fuzzy semantic recall) produces better LLM context than either approach alone — and a benchmark suite was built to prove it.

---

## How It Works

Every message is **dual-indexed**: an LLM extracts structured triples into Neo4j, while the raw text is embedded into Qdrant. At query time, both stores are searched and merged with Reciprocal Rank Fusion.

```
                          ┌─────────────────────────┐
   "We rejected MongoDB   │   LLM Triple Extraction  │
    — writes too slow"    │     (Llama 3.3 70B)      │
          │               └───────────┬─────────────┘
          │                           │
          ▼                           ▼
   ┌──────────────┐          ┌──────────────────┐
   │   Qdrant     │          │      Neo4j        │
   │ vector store │          │  knowledge graph  │
   │ (semantics)  │          │ (we)─[REJECTED]→  │
   │              │          │      (MongoDB)    │
   └──────┬───────┘          └─────────┬─────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
          ┌──────────────────────┐
          │  Reciprocal Rank      │   + Contradiction Detection
          │  Fusion (RRF)         │   + Epistemic State
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │  Memory-augmented     │
          │  LLM response         │
          └──────────────────────┘
```

---

## 📊 Benchmark Results

A 4-dimension benchmark suite was built to validate the system on **30 messages across 3 domains** (tech decisions, project planning, hiring). Numbers are measured, not estimated.

| Benchmark | Key Metric | Result | Verdict |
|-----------|-----------|--------|---------|
| **Extraction** | Recall / Edge-type accuracy | **100%** / **87%** | ✅ Pass |
| **Retrieval** | Hybrid win rate (P@3) | **17 / 20** queries | ✅ Pass |
| **Contradiction** | Precision / F1 | **100%** / 44% | ⚠️ Improved |
| **Generation (RAGAS)** | Context Recall (MemGraph vs Vector) | **0.83** vs 0.78 | ✅ Mixed |

**Headline finding — the hypothesis holds:** hybrid retrieval beat vector-only by **26% on Precision@3** and won or tied **17 of 20** queries.

**Where each approach wins:**
- 🕸️ **Graph dominates structural queries** — "what depends on X?", cross-domain (0.58 vs 0.25 vector)
- 🔍 **Vector dominates factual recall** — "what did we choose?" (0.50 vs 0.25 graph)
- 🤝 **Fusion captures both** — the best of each, ranked by cross-source agreement

**Honest tradeoff:** MemGraph's richer context improved **recall** but slightly lowered **faithfulness** (0.78 vs 0.96) — more context can introduce noise. This is a context-filtering optimization, documented rather than hidden.

> Full results in [`MemGraph_Benchmark_Report.pptx`](MemGraph_Benchmark_Report.pptx).

---

## Features

- **Dual-write ingestion** — every message indexed in graph + vector simultaneously
- **LLM-as-semantic-parser** — natural language → typed triples (`DECIDED`, `REJECTED`, `DEPENDS_ON`, `IS_UNCERTAIN_ABOUT`, `CONTRADICTS`)
- **Hybrid retrieval with RRF** — merges incompatible ranking systems into one ranked list
- **Contradiction detection** — LLM-based NLI flags when new info conflicts with past decisions, writing an audit trail
- **Epistemic state aggregation** — a live "ledger" of decisions, rejections, uncertainties, and conflicts injected into the system prompt
- **Interactive chat UI** — Streamlit app with a real-time knowledge-state panel
- **RAGAS evaluation** — Faithfulness, Answer Relevancy, Context Precision/Recall

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Knowledge graph | **Neo4j 5.26** (Bolt + Cypher) |
| Vector store | **Qdrant 1.13** (cosine, 512-dim) |
| LLM (extraction, NLI, generation) | **Llama 3.3 70B** via Groq |
| Embeddings | **Voyage AI** (`voyage-3-lite`) |
| Evaluation | **RAGAS** metrics |
| UI | **Streamlit** |
| Infra | **Docker Compose** |

---

## Project Structure

The project is organized as a two-part learning arc — foundations first, then a production system built on top.

```
.
├── memgraph-week1/              # Part 1: Foundations
│   ├── 01_graph_basics.py       #   Neo4j fundamentals
│   ├── 02_vector_basics.py      #   Qdrant + embeddings
│   ├── 03_extraction.py         #   LLM triple extraction
│   ├── 04_pipeline.py           #   Dual-write ingestion
│   ├── 05_retrieval_comparison.py
│   └── docker-compose.yml       #   Neo4j + Qdrant
│
├── memgraph-week2/              # Part 2: Production System
│   ├── core/                    #   Reusable engine
│   │   ├── retrieve.py          #     Hybrid retrieval + RRF
│   │   ├── contradict.py        #     Contradiction detection
│   │   ├── state.py             #     Epistemic state aggregation
│   │   └── chat.py              #     Full pipeline orchestration
│   ├── benchmark/               #   4-dimension benchmark suite
│   │   ├── bench_data.py        #     Test data + ground truth
│   │   ├── bench_extraction.py
│   │   ├── bench_retrieval.py
│   │   ├── bench_contradiction.py
│   │   ├── bench_generation.py  #     RAGAS evaluation
│   │   └── bench_all.py         #     Runs everything
│   ├── 06–10_*.py               #   Phase demos
│   └── app.py                   #   Streamlit chat UI
│
└── MemGraph_Benchmark_Report.pptx
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker
- Free API keys: [Groq](https://console.groq.com/) (LLM) and [Voyage AI](https://dash.voyageai.com/) (embeddings)

### Setup

```bash
# 1. Start Neo4j + Qdrant
docker compose -f memgraph-week1/docker-compose.yml up -d

# 2. Configure API keys
cp memgraph-week1/.env.example memgraph-week1/.env
#    then edit .env and add your GROQ_API_KEY and VOYAGE_API_KEY

# 3. Install dependencies
pip install -r memgraph-week2/requirements_week2.txt
```

### Run the chat app

```bash
cd memgraph-week2
streamlit run app.py
```

### Run the benchmarks

```bash
cd memgraph-week2/benchmark
python bench_extraction.py     # fastest — Groq only, no Docker needed
python bench_all.py            # full suite (~45 min, rate-limited)
```

---

## Roadmap

- [ ] **Package as an MCP server** — expose `query_memory`, `get_knowledge_state`, `ingest_messages` as tools for Claude Desktop / Cursor / Windsurf
- [ ] **Local backends** — swap Groq → Ollama and Voyage → sentence-transformers to remove rate limits
- [ ] **Context filtering** — relevance-score chunks before prompt injection to lift faithfulness
- [ ] **Chat-history adapters** — import ChatGPT / Claude conversation exports

---

## What I Learned

- **Benchmark before you build.** Measuring retrieval quality *first* meant downstream components were built on a validated foundation, not an assumption.
- **The interesting result is *where* a method wins, not just *whether* it wins.** Graph vs. vector isn't either/or — they're complementary, and fusion is what unlocks both.
- **Report tradeoffs honestly.** The faithfulness/recall tension is a real finding, not a flaw to hide.

---

## License

MIT — see [LICENSE](LICENSE).

*Built during an internship at Mirav Labs.*
