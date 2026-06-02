"""
Phase 3: LLM-Powered Triple Extraction — Turning Natural Language into Graph Structure

This script teaches the core technique behind knowledge graph construction:
using an LLM to extract structured (subject, relation, object) triples from
free-form text.

Why this matters: humans write "we decided to use React for the frontend."
A graph database needs: (Team)-[:DECIDED]->(React) with context "frontend."
The LLM bridges that gap.

The system prompt is the most important part — it constrains the LLM to:
  1. Only emit triples from a fixed set of edge types
  2. Assign a confidence score (and we filter below 0.75)
  3. Handle ambiguity ("I think maybe...") by choosing IS_UNCERTAIN_ABOUT

Uses Groq (free tier) running Llama 3.3 70B via an OpenAI-compatible API.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a knowledge graph extraction engine. Given a natural language message,
extract structured triples that capture the key facts, decisions, and relationships.

Each triple must have:
- subject: the entity performing an action or being described (string)
- relation: a short natural-language description of the relationship (string)
- object: the entity being acted upon or described (string)
- confidence: how confident you are this triple is correct (float, 0.0 to 1.0)
- edge_type: one of these exact values:
    DECIDED       — a firm decision was made to use/adopt something
    REJECTED      — something was explicitly ruled out or dismissed
    DEPENDS_ON    — one thing requires or relies on another
    RELATED_TO    — a general association between two things
    IS_UNCERTAIN_ABOUT — someone is unsure or tentative about something
    CONTRADICTS   — new information conflicts with a previous decision

Rules:
- Extract ALL meaningful triples from the message.
- Use DECIDED only for firm, committed choices — not suggestions or considerations.
- Use IS_UNCERTAIN_ABOUT when the language is tentative ("maybe", "I think", "could", "not sure").
- Use CONTRADICTS when a message reverses or questions a previous decision.
- Set confidence below 0.75 if the message is vague or ambiguous.
- Keep subject and object concise (1-4 words each).
- The relation field should be a brief verb phrase.

Respond with ONLY a JSON array of triples. No other text, no markdown fences."""


def extract_triples(message: str) -> tuple[list[dict], list[dict]]:
    """
    Send a message to Llama 3.3 (via Groq) and get back structured triples.

    The LLM acts as a "semantic parser" — it understands the meaning of
    natural language and maps it onto our fixed schema of edge types.
    This is more flexible than regex or rule-based extraction because it
    handles paraphrasing, implicit meaning, and ambiguity.

    Returns (confirmed, rejected) where confirmed triples have confidence >= 0.75.
    """
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Extract triples from this message:\n\n"{message}"'},
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    triples = json.loads(raw)

    confirmed = [t for t in triples if t.get("confidence", 0) >= 0.75]
    rejected = [t for t in triples if t.get("confidence", 0) < 0.75]

    return confirmed, rejected


def main():
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY in your .env file.")
        print("  Get one at: https://console.groq.com/")
        return

    test_messages = [
        "We decided to use React for the frontend",
        "I think maybe GraphQL could work but I'm not sure",
        "TypeScript was ruled out because the team doesn't know it",
        "The API layer depends on our auth service",
        "Actually we're reconsidering React, Vue might be better",
    ]

    for msg in test_messages:
        print(f'{"━" * 60}')
        print(f'Input: "{msg}"')
        print(f'{"━" * 60}')
        try:
            confirmed, rejected = extract_triples(msg)

            if confirmed:
                print(f"  Confirmed ({len(confirmed)}):")
                for t in confirmed:
                    print(f"    ({t['subject']}) --[{t['edge_type']}: {t['relation']}]--> ({t['object']})")
                    print(f"      confidence: {t['confidence']}")
            else:
                print("  No confirmed triples (all below 0.75 threshold)")

            if rejected:
                print(f"  Below threshold ({len(rejected)}):")
                for t in rejected:
                    print(f"    ({t['subject']}) --[{t['edge_type']}]--> ({t['object']})  conf={t['confidence']}")

        except json.JSONDecodeError as e:
            print(f"  ERROR: LLM returned invalid JSON: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
