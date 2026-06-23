from typing import Set
from index.access import get_posting_list
import re

def process_wildcard_query(pattern: str, index_path: str) -> Set[int]:
    """
    Process wildcard queries including:
        - Leading wildcard: *tion
        - Trailing wildcard: climat*
        - Internal wildcard: learn*ing

    Args:
        pattern (str): Wildcard pattern containing '*'.
        index_path (str): Path to the index package.

    Returns:
        Set[int]: Set of document IDs that match the wildcard pattern.
    """

    # Validate the wildcard pattern
    if '*' not in pattern or pattern.strip('*') == '':
        # '*' must exist and there must be some non-wildcard characters
        raise ValueError(f"Malformed wildcard: {pattern}")

    # Convert wildcard to regular expression
    # Escape special regex characters, then replace '*' with '.*'
    regex = re.compile('^' + re.escape(pattern).replace('\\*', '.*') + '$')

    # Load the index package
    result_docs = set()
    from index.access import _load_package
    package = _load_package(index_path)
    unified_index = package.get('unified', {})  # Get unified inverted index

    # Iterate over all terms in the index
    for term in unified_index.keys():
        term_str = " ".join(term)  # Convert term tuple to string for regex matching
        if regex.match(term_str):
            # If term matches the wildcard pattern, add its postings to result
            result_docs.update(get_posting_list(term, index_path))

    # Return all matching documents
    return result_docs



