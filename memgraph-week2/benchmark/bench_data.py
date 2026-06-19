"""
Benchmark Test Data — The Ground Truth That Makes Measurement Possible

This file contains all test data for the 4-dimension benchmark suite.
Why a separate file: benchmarks are only as good as their ground truth.
Centralizing test data makes it auditable and easy to expand.

Three domains of seed messages (not just databases), plus expected outputs
for extraction, retrieval, contradiction, and generation benchmarks.
"""

# ═══════════════════════════════════════════════════════════════════════════
# SEED MESSAGES — 30 messages across 3 domains
# ═══════════════════════════════════════════════════════════════════════════

SEED_MESSAGES = [
    # ── Domain A: Tech stack decisions (indices 0-9) ──
    "We are building a payments service and evaluating databases",
    "After load testing we decided on PostgreSQL for the main store",
    "MongoDB was rejected — write latency was 3x higher under our load profile",
    "Redis was chosen for session caching and rate limiting",
    "MySQL was considered but rejected due to lack of JSONB support",
    "Our backend API depends on PostgreSQL for all transaction data",
    "The auth service depends on Redis for session lookups",
    "We are uncertain whether we need a connection pooler like PgBouncer",
    "TimescaleDB was evaluated for metrics but we deferred the decision",
    "The data pipeline will depend on whatever database we choose for metrics",

    # ── Domain B: Project planning (indices 10-19) ──
    "We are launching the v2 release targeting March 15th",
    "The design review for the checkout flow is blocking the frontend team",
    "We decided to use a monorepo structure for all microservices",
    "The CI/CD pipeline depends on Docker containers being under 500MB",
    "We rejected the idea of a separate staging environment — too expensive",
    "Sprint velocity has been declining and we are uncertain about meeting the March deadline",
    "The QA team depends on the API documentation being up to date",
    "We chose GitHub Actions over Jenkins for CI — better integration with our workflow",
    "Feature flags will be managed through LaunchDarkly",
    "The mobile app release depends on the API v2 being stable",

    # ── Domain C: Hiring / team decisions (indices 20-29) ──
    "We decided to hire a senior DevOps engineer for the platform team",
    "The contractor proposal from Acme Corp was rejected — too expensive at $250/hr",
    "We are uncertain whether to promote internally or hire externally for the tech lead role",
    "The new data scientist role depends on the ML infrastructure being ready",
    "We chose Lever over Greenhouse for our applicant tracking system",
    "Remote-first was decided as our work policy going forward",
    "The part-time security consultant was rejected — we need someone full-time",
    "The engineering team depends on HR finalizing the new compensation bands",
    "We decided to require a take-home project instead of whiteboard interviews",
    "We are uncertain about whether to open a second office in Austin",
]

DOMAIN_LABELS = {
    "tech_stack": list(range(0, 10)),
    "project_planning": list(range(10, 20)),
    "hiring_team": list(range(20, 30)),
}


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION GROUND TRUTH
#
# For each message: the triples a correct extraction SHOULD produce.
# We list the minimum required triples — the LLM may extract additional
# valid ones, but these MUST be present for the extraction to count.
#
# Matching is fuzzy: subject/object match if they share >50% of content
# words. edge_type must be exact.
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_GROUND_TRUTH = {
    0: [
        {"subject": "we", "object": "payments service", "edge_type": "RELATED_TO"},
    ],
    1: [
        {"subject": "we", "object": "PostgreSQL", "edge_type": "DECIDED"},
    ],
    2: [
        {"subject": "we", "object": "MongoDB", "edge_type": "REJECTED"},
    ],
    3: [
        {"subject": "we", "object": "Redis", "edge_type": "DECIDED"},
    ],
    4: [
        {"subject": "we", "object": "MySQL", "edge_type": "REJECTED"},
    ],
    5: [
        {"subject": "backend API", "object": "PostgreSQL", "edge_type": "DEPENDS_ON"},
    ],
    6: [
        {"subject": "auth service", "object": "Redis", "edge_type": "DEPENDS_ON"},
    ],
    7: [
        {"subject": "we", "object": "PgBouncer", "edge_type": "IS_UNCERTAIN_ABOUT"},
    ],
    8: [
        {"subject": "we", "object": "TimescaleDB", "edge_type": "IS_UNCERTAIN_ABOUT"},
    ],
    9: [
        {"subject": "data pipeline", "object": "metrics database", "edge_type": "DEPENDS_ON"},
    ],
    10: [
        {"subject": "we", "object": "v2 release", "edge_type": "DECIDED"},
    ],
    11: [
        {"subject": "frontend team", "object": "design review", "edge_type": "DEPENDS_ON"},
    ],
    12: [
        {"subject": "we", "object": "monorepo", "edge_type": "DECIDED"},
    ],
    13: [
        {"subject": "CI/CD pipeline", "object": "Docker containers", "edge_type": "DEPENDS_ON"},
    ],
    14: [
        {"subject": "we", "object": "staging environment", "edge_type": "REJECTED"},
    ],
    15: [
        {"subject": "we", "object": "March deadline", "edge_type": "IS_UNCERTAIN_ABOUT"},
    ],
    16: [
        {"subject": "QA team", "object": "API documentation", "edge_type": "DEPENDS_ON"},
    ],
    17: [
        {"subject": "we", "object": "GitHub Actions", "edge_type": "DECIDED"},
    ],
    18: [
        {"subject": "we", "object": "LaunchDarkly", "edge_type": "DECIDED"},
    ],
    19: [
        {"subject": "mobile app release", "object": "API v2", "edge_type": "DEPENDS_ON"},
    ],
    20: [
        {"subject": "we", "object": "senior DevOps engineer", "edge_type": "DECIDED"},
    ],
    21: [
        {"subject": "we", "object": "Acme Corp", "edge_type": "REJECTED"},
    ],
    22: [
        {"subject": "we", "object": "tech lead role", "edge_type": "IS_UNCERTAIN_ABOUT"},
    ],
    23: [
        {"subject": "data scientist role", "object": "ML infrastructure", "edge_type": "DEPENDS_ON"},
    ],
    24: [
        {"subject": "we", "object": "Lever", "edge_type": "DECIDED"},
    ],
    25: [
        {"subject": "we", "object": "remote-first", "edge_type": "DECIDED"},
    ],
    26: [
        {"subject": "we", "object": "part-time security consultant", "edge_type": "REJECTED"},
    ],
    27: [
        {"subject": "engineering team", "object": "compensation bands", "edge_type": "DEPENDS_ON"},
    ],
    28: [
        {"subject": "we", "object": "take-home project", "edge_type": "DECIDED"},
    ],
    29: [
        {"subject": "we", "object": "Austin office", "edge_type": "IS_UNCERTAIN_ABOUT"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL QUERIES
#
# 20 queries with ground truth answers, spanning 5 query types.
# ground_truth is a substring that MUST appear in at least one top-3 result.
# ═══════════════════════════════════════════════════════════════════════════

RETRIEVAL_QUERIES = [
    # ── Factual recall (what was decided?) ──
    {
        "query": "What database did we choose for the main store?",
        "ground_truth": "PostgreSQL",
        "query_type": "factual",
    },
    {
        "query": "What did we pick for session caching?",
        "ground_truth": "Redis",
        "query_type": "factual",
    },
    {
        "query": "What CI system are we using?",
        "ground_truth": "GitHub Actions",
        "query_type": "factual",
    },
    {
        "query": "What ATS did we choose?",
        "ground_truth": "Lever",
        "query_type": "factual",
    },

    # ── Causal reasoning (why was X rejected/decided?) ──
    {
        "query": "Why did we reject MongoDB?",
        "ground_truth": "write latency",
        "query_type": "causal",
    },
    {
        "query": "Why was MySQL rejected?",
        "ground_truth": "JSONB",
        "query_type": "causal",
    },
    {
        "query": "Why did we reject a separate staging environment?",
        "ground_truth": "expensive",
        "query_type": "causal",
    },
    {
        "query": "Why was the Acme Corp contractor rejected?",
        "ground_truth": "expensive",
        "query_type": "causal",
    },

    # ── Dependency traversal (what depends on X?) ──
    {
        "query": "What depends on PostgreSQL?",
        "ground_truth": "backend API",
        "query_type": "dependency",
    },
    {
        "query": "What depends on Redis?",
        "ground_truth": "auth service",
        "query_type": "dependency",
    },
    {
        "query": "What does the QA team need?",
        "ground_truth": "API documentation",
        "query_type": "dependency",
    },
    {
        "query": "What is the mobile app release waiting on?",
        "ground_truth": "API v2",
        "query_type": "dependency",
    },

    # ── Uncertainty (what are we unsure about?) ──
    {
        "query": "What are we uncertain about in the tech stack?",
        "ground_truth": "PgBouncer",
        "query_type": "uncertainty",
    },
    {
        "query": "What hiring decisions are still open?",
        "ground_truth": "tech lead",
        "query_type": "uncertainty",
    },
    {
        "query": "What project risks exist?",
        "ground_truth": "March deadline",
        "query_type": "uncertainty",
    },
    {
        "query": "What office decisions are pending?",
        "ground_truth": "Austin",
        "query_type": "uncertainty",
    },

    # ── Cross-domain (broader questions) ──
    {
        "query": "What have we rejected so far?",
        "ground_truth": "MongoDB",
        "query_type": "cross_domain",
    },
    {
        "query": "What are all the dependencies in our system?",
        "ground_truth": "depends",
        "query_type": "cross_domain",
    },
    {
        "query": "What tools have we chosen?",
        "ground_truth": "Redis",
        "query_type": "cross_domain",
    },
    {
        "query": "What decisions are still deferred?",
        "ground_truth": "TimescaleDB",
        "query_type": "cross_domain",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# CONTRADICTION TEST CASES
#
# 15 test cases: 5 true contradictions, 5 non-contradictions, 5 edge cases.
# Each includes the seed messages that must be loaded first.
# ═══════════════════════════════════════════════════════════════════════════

CONTRADICTION_TESTS = [
    # ── True contradictions (should_contradict=True) ──
    {
        "test_message": "Actually MongoDB's new storage engine looks promising, we should reconsider it",
        "should_contradict": True,
        "category": "direct_reversal",
        "description": "Directly reverses the MongoDB rejection",
    },
    {
        "test_message": "I'm now thinking MySQL would be better than PostgreSQL for our main store",
        "should_contradict": True,
        "category": "direct_reversal",
        "description": "Contradicts the PostgreSQL decision",
    },
    {
        "test_message": "We've realized we definitely need PgBouncer, it's critical for our scale",
        "should_contradict": True,
        "category": "resolves_uncertainty",
        "description": "Resolves the PgBouncer uncertainty with a firm decision",
    },
    {
        "test_message": "We're moving to a polyrepo structure, monorepo is causing too many merge conflicts",
        "should_contradict": True,
        "category": "reversal_with_reason",
        "description": "Reverses the monorepo decision with a new reason",
    },
    {
        "test_message": "We've decided to go back to office-first, remote work isn't productive enough",
        "should_contradict": True,
        "category": "policy_reversal",
        "description": "Reverses the remote-first work policy",
    },

    # ── Non-contradictions (should_contradict=False) ──
    {
        "test_message": "PostgreSQL has been great for our transaction workloads",
        "should_contradict": False,
        "category": "consistent_restatement",
        "description": "Reaffirms the PostgreSQL decision",
    },
    {
        "test_message": "Redis is handling our session load perfectly",
        "should_contradict": False,
        "category": "consistent_restatement",
        "description": "Consistent with Redis for session caching",
    },
    {
        "test_message": "We added a new caching layer using Memcached for static assets",
        "should_contradict": False,
        "category": "different_scope",
        "description": "Different technology for a different purpose — not conflicting with Redis",
    },
    {
        "test_message": "The frontend team is exploring React Native for the mobile app",
        "should_contradict": False,
        "category": "new_topic",
        "description": "Entirely new topic, no existing position to contradict",
    },
    {
        "test_message": "We should document why we chose GitHub Actions for future reference",
        "should_contradict": False,
        "category": "elaboration",
        "description": "Discusses existing decision without contradicting it",
    },

    # ── Edge cases (tricky — tests nuanced understanding) ──
    {
        "test_message": "I wonder if MongoDB would have been a better choice",
        "should_contradict": False,
        "category": "hypothetical",
        "description": "Hypothetical question, not an actual reversal",
    },
    {
        "test_message": "Jenkins has some features GitHub Actions lacks",
        "should_contradict": False,
        "category": "observation",
        "description": "Observation about rejected option, not a reversal",
    },
    {
        "test_message": "Maybe we should re-evaluate TimescaleDB now that the metrics requirements are clearer",
        "should_contradict": False,
        "category": "revisiting_deferred",
        "description": "Revisiting a deferred decision is not contradicting it",
    },
    {
        "test_message": "We've decided whiteboard interviews are actually better than take-home projects",
        "should_contradict": True,
        "category": "implicit_reversal",
        "description": "Implicitly reverses the take-home project decision",
    },
    {
        "test_message": "Austin looks like a great market, we should definitely open there",
        "should_contradict": True,
        "category": "resolves_uncertainty",
        "description": "Resolves the Austin office uncertainty",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION QUESTIONS
#
# 10 questions with reference answers and key facts that MUST be mentioned.
# key_facts are used by the LLM-as-judge to score factual accuracy.
# ═══════════════════════════════════════════════════════════════════════════

GENERATION_QUESTIONS = [
    {
        "question": "Why did we reject MongoDB?",
        "key_facts": ["write latency", "3x higher", "load profile"],
        "reference": "MongoDB was rejected because write latency was 3x higher under our load profile.",
    },
    {
        "question": "What does our auth service depend on?",
        "key_facts": ["Redis", "session lookups"],
        "reference": "The auth service depends on Redis for session lookups.",
    },
    {
        "question": "What decisions are we still uncertain about?",
        "key_facts": ["PgBouncer", "March deadline", "tech lead", "Austin"],
        "reference": "We are uncertain about: needing PgBouncer, meeting the March deadline, the tech lead hiring approach, and opening an Austin office.",
    },
    {
        "question": "What is our work policy?",
        "key_facts": ["remote-first"],
        "reference": "We decided on a remote-first work policy.",
    },
    {
        "question": "What CI/CD system are we using and why?",
        "key_facts": ["GitHub Actions", "Jenkins", "better integration"],
        "reference": "We chose GitHub Actions over Jenkins for CI because of better integration with our workflow.",
    },
    {
        "question": "What are all the things that depend on something else?",
        "key_facts": ["backend API", "PostgreSQL", "auth service", "Redis", "QA team", "API documentation"],
        "reference": "Dependencies include: backend API on PostgreSQL, auth service on Redis, QA team on API docs, mobile app on API v2, data pipeline on metrics DB, CI/CD on Docker containers, data scientist role on ML infrastructure, engineering team on compensation bands.",
    },
    {
        "question": "What did we decide about the interview process?",
        "key_facts": ["take-home project", "whiteboard"],
        "reference": "We decided to require a take-home project instead of whiteboard interviews.",
    },
    {
        "question": "Why was the Acme Corp proposal rejected?",
        "key_facts": ["expensive", "$250"],
        "reference": "The contractor proposal from Acme Corp was rejected because it was too expensive at $250/hr.",
    },
    {
        "question": "What tools and platforms have we committed to?",
        "key_facts": ["PostgreSQL", "Redis", "GitHub Actions", "LaunchDarkly", "Lever"],
        "reference": "We have committed to PostgreSQL, Redis, GitHub Actions, LaunchDarkly, Lever, monorepo structure, and remote-first policy.",
    },
    {
        "question": "What are the blockers for the mobile app release?",
        "key_facts": ["API v2", "stable"],
        "reference": "The mobile app release depends on the API v2 being stable.",
    },
]
