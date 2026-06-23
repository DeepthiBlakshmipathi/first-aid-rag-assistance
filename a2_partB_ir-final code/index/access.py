"""
Unified Index Package Access Functions - Task 1
Provides O(1) access to all three sub-indexes from a single package.
"""
import pickle
import gzip
from typing import List, Union, Tuple, Dict, Any
from .io import load  # Function to load a gzipped pickled index package

# Global cache to store loaded packages for O(1) access
_package_cache: Dict[str, Dict[str, Any]] = {}

def _load_package(index_path: str) -> Dict[str, Any]:
    """
    Load and cache the unified index package.
    
    Args:
        index_path: Path to the gzipped index package
    
    Returns:
        The loaded index package as a dictionary
    """
    if index_path not in _package_cache:
        # Load from disk only if not already cached
        _package_cache[index_path] = load(index_path)
    return _package_cache[index_path]

def get_posting_list(term: Union[str, Tuple[str, ...]], index_path: str) -> List[int]:
    """
    Returns the posting list for a unigram or n-gram from the unified sub-index.
    
    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        index_path: Path to the unified index package
        
    Returns:
        List of document IDs (sorted, deduplicated) containing the term
    """
    package = _load_package(index_path)  # Load the index package
    unified_index = package.get("unified", {})  # Get the unified index
    return sorted(unified_index.get(term, []))  # Return posting list (sorted)

def find_wildcard_matches(ngram: str, index_path: str) -> List[str]:
    """
    Returns the list of terms that match a character-level wildcard n-gram.
    
    Args:
        ngram: A character n-gram (e.g., "$cl", "on$")
        index_path: Path to the unified index package
        
    Returns:
        List of matching terms (sorted lexicographically, deduplicated)
    """
    package = _load_package(index_path)  # Load the index package
    wildcard_index = package.get("wildcard", {})  # Get the wildcard index
    return sorted(set(wildcard_index.get(ngram, [])))  # Return matches

def get_term_positions(term: Union[str, Tuple[str, ...]], doc_id: int, index_path: str) -> List[int]:
    """
    Returns the position list for a unigram or n-gram in a specific document.
    
    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        doc_id: Document ID
        index_path: Path to the unified index package
        
    Returns:
        List of positions (0-based, sorted, deduplicated) where the term appears in the document
    """
    package = _load_package(index_path)  # Load the index package
    proximity_index = package.get("proximity", {})  # Get the proximity index
    return sorted(proximity_index.get(term, {}).get(doc_id, []))  # Return positions


# Function to load a serialized index from disk
def load_index(index_path="cache/index_pkg.pkl.gz"):
    """
    Load the inverted index object from a gzip-compressed pickle file.
    
    Args:
        index_path (str): Path to the compressed index file (default: "cache/index_pkg.pkl.gz")
    
    Returns:
        dict or None: The loaded index object if successful, None if file not found.
    """
    try:
        # Open the gzip-compressed file in binary read mode
        with gzip.open(index_path, 'rb') as f:
            # Deserialize the index object from the file
            index = pickle.load(f)
        return index
    except FileNotFoundError:
        # If the file does not exist, print an error message and return None
        print("Index file not found.")
        return None


# Function to query the index
def query_index(query_text, index_obj):
    """
    Simple candidate retrieval from an inverted index.
    
    Args:
        query_text (str): The query string containing one or more terms.
        index_obj (dict): The inverted index (term -> set of document IDs).
    
    Returns:
        set: A set of document IDs that contain any of the query terms.
    """
    # Split the query text into individual terms (simple whitespace tokenizer)
    query_terms = query_text.strip().split()

    # Initialize an empty set to store candidate document IDs
    candidate_docs = set()

    # Iterate over each term in the query
    for term in query_terms:
        # If the term exists in the index, add its associated documents to candidate_docs
        if term in index_obj:
            candidate_docs.update(index_obj[term])

    # Return the set of candidate documents matching at least one query term
    return candidate_docs
