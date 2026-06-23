#!/usr/bin/env python3
"""
eval_map.py
Evaluate MAP@10 for IR system runs.

Usage:
    python eval_map.py [runs_dir] [qrels_json]  # Optional args
"""

import os
import sys
import json

# Load relevance judgments (qrels)
def load_qrels(qrels_path=None):
    """
    Loads the qrels JSON file and returns a dictionary mapping qid -> list of relevant doc IDs.

    Supports different JSON formats:
      - {"qid": "...", "relevant_docs": [...]}
      - {"qid": "...", "ground_truth_order": [...], "relevance_scores": {...}}

    Args:
        qrels_path (str): Path to qrels JSON file. Defaults to 'data/dev/relevance_judge.json'.

    Returns:
        dict: {qid: [list of relevant doc IDs]}
    """
    # Default path if not provided
    if qrels_path is None:
        qrels_path = os.path.join("data", "dev", "relevance_judge.json")
        if not os.path.exists(qrels_path):
            raise FileNotFoundError(f"Default qrels not found: {qrels_path}")

    # Load JSON file
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_list = json.load(f)

    # Convert to dictionary format: qid -> relevant doc IDs
    qrels_dict = {}
    for entry in qrels_list:
        qid = entry["qid"]
        if "relevant_docs" in entry:
            qrels_dict[qid] = entry["relevant_docs"]
        elif "ground_truth_order" in entry and "relevance_scores" in entry:
            # Keep only docs with relevance score > 0
            relevant_docs = [int(doc_id) for doc_id, score in entry["relevance_scores"].items() if score > 0]
            qrels_dict[qid] = relevant_docs
        else:
            qrels_dict[qid] = []  # No relevant docs
    return qrels_dict



# Compute Average Precision @ k
def apk(relevant, retrieved, k=10):
    """
    Computes Average Precision at k for a single query.

    Args:
        relevant (list): List of relevant doc IDs for the query.
        retrieved (list): List of retrieved doc IDs in ranked order.
        k (int): Maximum number of retrieved docs to consider.

    Returns:
        float: Average Precision score.
    """
    if not relevant:
        return 0.0  # No relevant docs, AP=0

    retrieved = retrieved[:k]  # Only top-k retrieved docs
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved):
        # Count as a hit if doc is relevant and not counted before
        if doc_id in relevant and doc_id not in retrieved[:i]:
            num_hits += 1
            score += num_hits / (i + 1)  # Precision at this rank
    # Normalize by min(len(relevant), k)
    return score / min(len(relevant), k)

# Compute Mean Average Precision @ k
def mapk(qrels, run_file, k=10):
    """
    Computes MAP@k for a given run file.

    Args:
        qrels (dict): Ground truth qrels {qid: [relevant docs]}.
        run_file (str): Path to JSON run file.
        k (int): Maximum number of retrieved docs to consider.

    Returns:
        float: MAP@k score.
    """
    # Load run JSON file (list of queries with retrieved doc_ids)
    with open(run_file, "r", encoding="utf-8") as f:
        run = json.load(f)

    scores = []
    for r in run:
        # Ensure entry has qid and doc_ids
        if isinstance(r, dict) and "qid" in r and "doc_ids" in r:
            qid = r["qid"]
            retrieved = r["doc_ids"]
            relevant = qrels.get(qid, [])
            if not relevant:
                print(f"Warning: No relevant docs found for QID {qid}")
            scores.append(apk(relevant, retrieved, k))  # Compute AP for this query
    # Return mean of APs
    return sum(scores) / len(scores) if scores else 0.0

# Command-line Interface (CLI) entry point
def main(runs_dir=None, qrels_path=None):
    """
    Main function to compute MAP@10 for all runs in a directory.
    """
    # Set default directories if not provided
    if runs_dir is None:
        runs_dir = os.path.join("runs")
    if qrels_path is None:
        qrels_path = os.path.join("data", "dev", "relevance_judge.json")

    # Load qrels
    qrels = load_qrels(qrels_path)

    # Evaluate each run JSON file in the runs directory
    for run_file in os.listdir(runs_dir):
        if run_file.endswith(".json"):
            run_path = os.path.join(runs_dir, run_file)
            score = mapk(qrels, run_path)  # Compute MAP@10
            print(f"{run_file} MAP@10: {score:.4f}")


# Execute if called as CLI
if __name__ == "__main__":
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    qrels_path = sys.argv[2] if len(sys.argv) > 2 else "data/dev/relevance_judge.json"
    main(runs_dir, qrels_path)

