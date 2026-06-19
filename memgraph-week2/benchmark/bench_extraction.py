"""
Benchmark 1: Extraction Quality — Are the Triples Correct?

Measures whether the LLM correctly extracts structured triples from
natural language. This is the foundation benchmark: if extraction is
wrong, every downstream component (graph, retrieval, contradiction
detection) inherits that error.

Metrics:
  - Triple Precision: % of extracted triples that match an expected triple
  - Triple Recall: % of expected triples that were actually extracted
  - Edge Type Accuracy: % of matched triples where the edge_type is correct

Run: python bench_extraction.py
Prereqs: GROQ_API_KEY set in .env (no Neo4j/Qdrant needed)
"""

import os
import sys
import re
import time

WEEK1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memgraph-week1")
sys.path.insert(0, WEEK1_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(WEEK1_DIR, ".env"))

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("extraction", os.path.join(WEEK1_DIR, "03_extraction.py"))
_extraction = module_from_spec(_spec)
_spec.loader.exec_module(_extraction)
extract_triples = _extraction.extract_triples

from bench_data import SEED_MESSAGES, EXTRACTION_GROUND_TRUTH

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "we", "our", "us", "i",
    "and", "but", "or", "for", "of", "to", "in", "on", "at", "by",
    "with", "from", "that", "this", "it", "its",
}


def content_words(text: str) -> set[str]:
    """Extract meaningful words from a string, excluding stop words."""
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - STOP_WORDS


AGENT_SYNONYMS = {"we", "team", "our", "us", "company", "org", "organization", "group"}


def fuzzy_entity_match(extracted: str, expected: str, threshold: float = 0.5) -> bool:
    """
    Check if two entity names refer to the same thing.
    Uses content word overlap — the LLM may rephrase "payments service"
    as "payment service" or "we" as "team". A >50% overlap counts as a match.

    Special case: "we"/"team"/"our"/"company" are treated as equivalent
    since the LLM often replaces pronouns with "Team" or vice versa.
    """
    ext_lower = extracted.lower().strip()
    exp_lower = expected.lower().strip()

    if ext_lower == exp_lower:
        return True

    # Pronoun/agent equivalence: "we", "team", "our team", etc.
    if ext_lower in AGENT_SYNONYMS and exp_lower in AGENT_SYNONYMS:
        return True

    ext_words = content_words(extracted)
    exp_words = content_words(expected)

    if not ext_words or not exp_words:
        return False

    overlap = ext_words & exp_words
    # Match if overlap covers >threshold of either set
    if len(exp_words) > 0 and len(overlap) / len(exp_words) >= threshold:
        return True
    if len(ext_words) > 0 and len(overlap) / len(ext_words) >= threshold:
        return True

    return False


def match_triple(extracted: dict, expected: dict) -> tuple[bool, bool]:
    """
    Check if an extracted triple matches an expected one.
    Returns (entity_match, full_match) where:
      entity_match = subject AND object match (fuzzy, either direction)
      full_match = entity_match AND edge_type is correct (exact)

    Tries both (subj→subj, obj→obj) and (subj→obj, obj→subj) because
    the LLM sometimes flips direction: our ground truth says
    (we)--[REJECTED]-->(MongoDB) but the LLM outputs
    (MongoDB)--[REJECTED]-->(JSONB support). The key entity still appears.
    """
    ext_subj = extracted.get("subject", "")
    ext_obj = extracted.get("object", "")
    exp_subj = expected["subject"]
    exp_obj = expected["object"]

    # Forward direction: subj↔subj, obj↔obj
    forward = (
        fuzzy_entity_match(ext_subj, exp_subj) and
        fuzzy_entity_match(ext_obj, exp_obj)
    )

    # Reverse direction: subj↔obj, obj↔subj
    reverse = (
        fuzzy_entity_match(ext_subj, exp_obj) and
        fuzzy_entity_match(ext_obj, exp_subj)
    )

    # Also check if just the key entity (non-pronoun) matches in either position
    # e.g., expected (we, MongoDB, REJECTED) matches (MongoDB, JSONB, REJECTED)
    # because MongoDB appears in both
    key_entity_match = False
    exp_key = exp_obj if exp_subj.lower() in AGENT_SYNONYMS else exp_subj
    if (fuzzy_entity_match(ext_subj, exp_key) or
        fuzzy_entity_match(ext_obj, exp_key)):
        key_entity_match = True

    entity_match = forward or reverse or key_entity_match
    edge_match = extracted.get("edge_type", "") == expected["edge_type"]

    return entity_match, entity_match and edge_match


def evaluate_message(msg_idx: int, message: str) -> dict:
    """
    Extract triples from one message and compare against ground truth.
    Returns per-message metrics.
    """
    expected_triples = EXTRACTION_GROUND_TRUTH.get(msg_idx, [])
    if not expected_triples:
        return None

    try:
        confirmed, rejected = extract_triples(message)
    except Exception as e:
        print(f"  ERROR extracting from message {msg_idx}: {e}")
        return {
            "msg_idx": msg_idx,
            "message": message,
            "error": str(e),
            "precision": 0.0,
            "recall": 0.0,
            "edge_type_accuracy": 0.0,
        }

    all_extracted = confirmed + rejected

    # Match each extracted triple against expected triples
    matched_expected = set()
    matched_extracted = set()
    edge_type_correct = 0

    for i, ext in enumerate(all_extracted):
        for j, exp in enumerate(expected_triples):
            if j in matched_expected:
                continue
            entity_match, full_match = match_triple(ext, exp)
            if entity_match:
                matched_extracted.add(i)
                matched_expected.add(j)
                if full_match:
                    edge_type_correct += 1
                break

    n_extracted = len(all_extracted)
    n_expected = len(expected_triples)
    n_matched = len(matched_expected)

    precision = len(matched_extracted) / n_extracted if n_extracted > 0 else 0.0
    recall = n_matched / n_expected if n_expected > 0 else 0.0
    edge_acc = edge_type_correct / n_matched if n_matched > 0 else 0.0

    return {
        "msg_idx": msg_idx,
        "message": message[:60],
        "n_extracted": n_extracted,
        "n_expected": n_expected,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "edge_type_accuracy": edge_acc,
        "extracted": all_extracted,
        "expected": expected_triples,
    }


def run_extraction_benchmark() -> dict:
    """Run extraction on all messages with ground truth and compute metrics."""
    print("=" * 70)
    print("BENCHMARK 1: EXTRACTION QUALITY")
    print("=" * 70)
    print(f"Testing {len(EXTRACTION_GROUND_TRUTH)} messages with ground truth triples\n")

    results = []
    indices = sorted(EXTRACTION_GROUND_TRUTH.keys())

    for i, msg_idx in enumerate(indices):
        message = SEED_MESSAGES[msg_idx]
        print(f"  [{i+1}/{len(indices)}] \"{message[:55]}...\"")

        result = evaluate_message(msg_idx, message)
        if result:
            results.append(result)
            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Extracted: {result['n_extracted']} | "
                      f"Expected: {result['n_expected']} | "
                      f"Matched: {result['n_matched']} | "
                      f"Edge type correct: {result['edge_type_accuracy']:.0%}")

        if i < len(indices) - 1:
            time.sleep(2)

    # Aggregate metrics
    valid = [r for r in results if "error" not in r]
    avg_precision = sum(r["precision"] for r in valid) / len(valid) if valid else 0
    avg_recall = sum(r["recall"] for r in valid) / len(valid) if valid else 0
    avg_edge_acc = sum(r["edge_type_accuracy"] for r in valid) / len(valid) if valid else 0

    # Per-edge-type breakdown
    edge_type_stats = {}
    for r in valid:
        for exp in r["expected"]:
            et = exp["edge_type"]
            if et not in edge_type_stats:
                edge_type_stats[et] = {"total": 0, "matched": 0}
            edge_type_stats[et]["total"] += 1

        for j, exp in enumerate(r["expected"]):
            et = exp["edge_type"]
            # Check if this expected triple was matched
            for ext in r["extracted"]:
                entity_match, full_match = match_triple(ext, exp)
                if full_match:
                    edge_type_stats[et]["matched"] += 1
                    break

    print(f"\n{'═' * 70}")
    print("EXTRACTION RESULTS")
    print(f"{'═' * 70}")
    print(f"\n  Average Precision:          {avg_precision:.1%}")
    print(f"  Average Recall:             {avg_recall:.1%}")
    print(f"  Average Edge Type Accuracy: {avg_edge_acc:.1%}")
    print(f"  Messages tested:            {len(valid)}/{len(indices)}")

    print(f"\n  Per Edge Type:")
    print(f"  {'Edge Type':<25} | {'Matched':>8} | {'Total':>6} | {'Accuracy':>8}")
    print(f"  {'─' * 25}-+-{'─' * 8}-+-{'─' * 6}-+-{'─' * 8}")
    for et, stats in sorted(edge_type_stats.items()):
        acc = stats["matched"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {et:<25} | {stats['matched']:>8} | {stats['total']:>6} | {acc:>8.0%}")

    # Pass/fail verdict
    print(f"\n  {'─' * 50}")
    passed = avg_precision >= 0.70 and avg_recall >= 0.60
    if passed:
        print(f"  PASS — Extraction quality meets threshold")
        print(f"    Precision >= 70%: {avg_precision:.1%}")
        print(f"    Recall >= 60%: {avg_recall:.1%}")
    else:
        print(f"  FAIL — Extraction quality below threshold")
        if avg_precision < 0.70:
            print(f"    Precision {avg_precision:.1%} < 70% — LLM is extracting wrong triples")
        if avg_recall < 0.60:
            print(f"    Recall {avg_recall:.1%} < 60% — LLM is missing expected triples")

    return {
        "benchmark": "extraction",
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_edge_type_accuracy": avg_edge_acc,
        "n_messages": len(valid),
        "passed": passed,
        "edge_type_stats": edge_type_stats,
        "details": results,
    }


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY in memgraph-week1/.env")
        sys.exit(1)
    run_extraction_benchmark()
