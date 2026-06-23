#!/usr/bin/env python3
"""
search_system.py
Command-line entry point for Task 4: Unified IR System
Usage:
    python search_system.py <queries_json> <documents_jsonl> <run_output_json> [option]
Options:
    cutoff      -> keep only first 500 docs
    stopwords   -> remove stopwords from queries
    sanity      -> apply small sanity modifications
"""

import sys
import os
import json

# Ensure parent folder is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from index.builders import create_all_indexes
from index.access import _load_package
from ranking.rankers import rank_documents
from utils.text_preprocessing import normalize_tokens

# Inline STOPWORDS
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "with",
    "has", "been", "is", "are", "do", "does"
}

# Preprocess text: lowercase, normalize tokens
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = normalize_tokens(tokens)
    return tokens

# Query expansion with simple synonyms
def expand_query(tokens):
    synonyms = {
        "research": ["study", "investigation"],
        "flow": ["stream", "current"],
        "transition": ["change", "shift"]
    }
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in synonyms:
            expanded.extend(synonyms[t])
    return expanded

# Load documents from JSONL
def load_documents(documents_path):
    docs = {}
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docs[int(doc["id"])] = preprocess_text(doc["text"])
    return docs

# Load queries from JSON
def load_queries(queries_path):
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load qrels for debugging
def load_qrels_for_debug(qrels_path=None):
    if qrels_path is None:
        qrels_path = os.path.join(parent_dir, "data", "dev", "relevance_judge.json")
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_list = json.load(f)
    qrels_dict = {}
    for entry in qrels_list:
        qid = entry["qid"]
        if "relevance_scores" in entry:
            relevant_docs = [int(doc_id) for doc_id, score in entry["relevance_scores"].items() if score > 0]
            qrels_dict[qid] = relevant_docs
        elif "relevant_docs" in entry:
            qrels_dict[qid] = entry["relevant_docs"]
        else:
            qrels_dict[qid] = []
    return qrels_dict

def main(queries_path, documents_path, run_output_path, option=None):
    print("Loading documents...")
    docs = load_documents(documents_path)
    if not docs:
        raise ValueError("No documents loaded! Check your documents JSONL file.")
    doc_ids = list(docs.keys())

    print("Loading queries...")
    queries = load_queries(queries_path)

    # Apply optional modifications
    if option == "cutoff":
        print("Applying candidate cutoff of 500 documents...")
        docs = dict(list(docs.items())[:500])
        doc_ids = list(docs.keys())
    elif option == "stopwords":
        print("Removing stopwords from queries...")
        for q in queries:
            q["query"] = " ".join([t for t in preprocess_text(q["query"]) if t not in STOPWORDS])
    elif option == "sanity":
        print("Applying small sanity modifications (if any)...")
        # Implement sanity modifications if required

    # Load or create index
    index_path = os.path.join("runs", "index_pkg.pkl.gz")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    try:
        index = _load_package(index_path)
        print("Index loaded from disk.")
    except FileNotFoundError:
        print("Index not found. Creating a new one...")
        # Pass docs as dict to support builders.py
        create_all_indexes(docs, index_path)
        index = _load_package(index_path)
        print("Index created and loaded.")

    # Load qrels for debugging
    qrels = load_qrels_for_debug()

    # Rank queries
    results = []
    for q in queries:
        qid = q["qid"]
        query_tokens = expand_query(preprocess_text(q["query"]))
        candidate_docs = [docs[doc_id] for doc_id in doc_ids]

        ranked_doc_ids, ranked_scores = rank_documents(
            query_tokens, candidate_docs, doc_ids, method="bm25"
        )

        top_docs = list(ranked_doc_ids)[:10]
        top_scores = list(ranked_scores)[:10] if ranked_scores else []

        relevant_docs = qrels.get(qid, [])
        matched = [d for d in top_docs if d in relevant_docs]

        print(f"\nQID: {qid}")
        print(f"Query: {q['query']}")
        print(f"Top-10 retrieved doc IDs: {top_docs}")
        print(f"Relevant doc IDs: {relevant_docs}")
        print(f"Matched relevant docs in top-10: {matched} (count: {len(matched)})")

        results.append({
            "qid": qid,
            "doc_ids": top_docs,
            "scores": top_scores
        })

    os.makedirs(os.path.dirname(run_output_path), exist_ok=True)
    with open(run_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRun file saved to {run_output_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python search_system.py <queries_json> <documents_jsonl> <run_output_json> [option]")
        sys.exit(1)
    option = sys.argv[4] if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], option)

