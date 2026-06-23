import re

def detect_query_type(query: str) -> str:
    """
    Detect query type: boolean, wildcard, proximity, or natural_language.
    Raise ValueError for malformed cases.
    """
    query = query.strip()

    # ===== Malformed cases =====
    # Empty phrase
    if query in ('""', "''"):
        raise ValueError("Malformed query: empty phrase")

    # Wildcard mixed with Boolean
    if '*' in query and any(op in query for op in ["AND", "OR", "NOT"]):
        raise ValueError("Malformed query: wildcard mixed with Boolean operator")

    # NEAR at start (missing left operand)
    if re.match(r'^\s*NEAR/\d+', query):
        raise ValueError("Malformed query: NEAR missing left operand")

    # ===== Detection rules =====
    # Proximity: detect NEAR/k
    if re.search(r'NEAR/\d+', query):
        return "proximity"

    # Wildcard: any token containing '*'
    if '*' in query:
        return "wildcard"

    # Boolean: contains AND / OR / NOT, or any quoted phrase
    if any(op in query for op in ["AND", "OR", "NOT"]) or '"' in query:
        return "boolean"

    # Otherwise → natural language
    return "natural_language"
