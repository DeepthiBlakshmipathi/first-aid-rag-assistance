from typing import Set
from query_processing.detection import detect_query_type
from query_processing.boolean import process_boolean_query
from query_processing.wildcard import process_wildcard_query
from query_processing.proximity import process_proximity_query


def convert_natural_language(nl_query: str) -> str:
    """
    Convert already-cleaned natural language query to OR-joined Boolean form.
    Example: "effects climate change" -> "effects OR climate OR change"
    """
    tokens = nl_query.strip().split()
    if not tokens:
        return ""
    return " OR ".join(tokens)

def process_query(query: str, index_path: str) -> Set[int]:
    """
    Main query processing with automatic type detection.
    Integrates detection + processors.
    """
    qtype = detect_query_type(query)

    if qtype == "boolean":
        return process_boolean_query(query, index_path)
    elif qtype == "wildcard":
        return process_wildcard_query(query, index_path)
    elif qtype == "proximity":
        return process_proximity_query(query, index_path)
    else:  # natural_language
        boolean_query = convert_natural_language(query)
        return process_boolean_query(boolean_query, index_path) if boolean_query else set()


