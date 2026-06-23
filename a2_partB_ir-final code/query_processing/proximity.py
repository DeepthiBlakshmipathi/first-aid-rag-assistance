from typing import Set, Tuple
from index.access import get_term_positions, get_posting_list
import re

def process_proximity_query(query: str, index_path: str) -> Set[int]:
    """
    Process a NEAR/k proximity query.
    Supports single terms or quoted phrases on both sides.

    Args:
        query (str): Query in the format 'term1 NEAR/k term2' or '"phrase1" NEAR/k "phrase2"'.
        index_path (str): Path to the index package.

    Returns:
        Set[int]: Set of document IDs where the terms/phrases occur within k words of each other.
    """

    # Parse the query using regex
    match = re.fullmatch(
        r'\s*(\"[^\"]+\"|\S+)\s+NEAR/(\d+)\s+(\"[^\"]+\"|\S+)\s*', 
        query
    )
    if not match:
        raise ValueError(f"Malformed proximity query: {query}")

    left, k_str, right = match.groups()
    k = int(k_str)  # Maximum allowed distance

    # Convert operands to tuples of tokens
    def parse_operand(op: str) -> Tuple[str, ...]:
        """
        Convert a term or quoted phrase into a tuple of tokens.
        """
        op = op.strip()
        if op.startswith('"') and op.endswith('"'):
            tokens = tuple(op[1:-1].split())  # Split phrase into tokens
            if not tokens:
                raise ValueError("Empty phrase in proximity query")
            return tokens
        else:
            return (op,)  # Single term as a 1-element tuple

    left_term = parse_operand(left)
    right_term = parse_operand(right)

    # Retrieve candidate documents
    # Only consider docs containing both terms/phrases
    left_docs = set(get_posting_list(left_term, index_path))
    right_docs = set(get_posting_list(right_term, index_path))
    candidate_docs = left_docs & right_docs  # Intersection of postings

    result_docs = set()

    # Check proximity within each candidate document
    for doc_id in candidate_docs:
        # Get positions of left and right terms/phrases
        left_positions = get_term_positions(left_term, doc_id, index_path)
        right_positions = get_term_positions(right_term, doc_id, index_path)

        found = False
        for l_start in left_positions:
            l_end = l_start + len(left_term) - 1  # End position of left term
            for r_start in right_positions:
                r_end = r_start + len(right_term) - 1  # End position of right term

                # Distance from left term to right term
                distance = r_start - l_end - 1
                if 0 <= distance <= k:
                    found = True
                    break

                # Also check reverse order (right before left)
                distance_rev = l_start - r_end - 1
                if 0 <= distance_rev <= k:
                    found = True
                    break

            if found:
                break

        if found:
            result_docs.add(doc_id)  # Add doc if any proximity match found

    # Return all documents satisfying proximity
    return result_docs
