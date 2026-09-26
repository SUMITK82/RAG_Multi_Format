"""
Retrieval evaluation harness.

You provide a small labeled test set (eval_questions.json) of the form:
[
  {"question": "What is the refund policy?", "expected_source": "policy.pdf"},
  ...
]

For each question we retrieve top-k chunks and check whether ANY of them
came from the expected source file. This gives hit-rate@k — a simple but
legitimate retrieval-quality metric you can quote on a resume.

Usage:
    python eval.py --index faiss_index --questions eval_questions.json --k 3
"""

import argparse
import json

from vectorstore import load_index
from rag_chain import retrieve


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(index_path: str, questions_path: str, k: int = 3):
    index = load_index(index_path)
    questions = load_questions(questions_path)

    hits = 0
    results = []

    for item in questions:
        q = item["question"]
        expected = item["expected_source"]
        retrieved = retrieve(index, q, k=k)
        sources = [r.metadata.get("source_file") for r in retrieved]
        hit = expected in sources
        hits += int(hit)
        results.append({"question": q, "expected": expected, "retrieved": sources, "hit": hit})

    hit_rate = hits / len(questions) if questions else 0.0

    print(f"\nEvaluated {len(questions)} questions, k={k}")
    print(f"Hit-rate@{k}: {hit_rate:.2%}\n")
    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} Q: {r['question']}")
        print(f"   expected: {r['expected']} | retrieved: {r['retrieved']}")

    return hit_rate, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="faiss_index")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    evaluate(args.index, args.questions, args.k)
