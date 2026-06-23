"""
Unified Index Package Builder - Task 1
Creates a single on-disk package containing all three sub-indexes.
"""
from typing import List, Tuple, Union, Dict, Optional
from collections import defaultdict
from utils.ngram import make_ngrams_tokens, make_ngrams_chars
from index.io import dump
from utils.text_preprocessing import normalize_tokens
import pickle
import gzip

# Default n-gram lengths (replaces utils.config)
NGRAM_MAX = 2        # max token n-gram length
CHAR_NGRAM_MAX = 4   # max character n-gram length

# Function to build a simple inverted index
def build_index(docs_tokens, save_path="cache/index_pkg.pkl.gz"):
    """
    Build a simple inverted index: term -> set of doc_ids
    
    Args:
        docs_tokens (dict): Mapping of doc_id -> list of tokens
        save_path (str): Path to save the gzip-compressed pickle index

    Returns:
        dict: The constructed inverted index
    """
    # Initialize empty dictionary for the index
    index = {}

    # Iterate over each document and its tokens
    for doc_id, tokens in docs_tokens.items():
        for token in tokens:
            # If token not in index, initialize a set for it
            if token not in index:
                index[token] = set()
            # Add the document ID to the token's set
            index[token].add(doc_id)

    # Serialize the index and save it to a gzip-compressed file
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(index, f)

    print(f"Index saved to {save_path}")
    return index

# Function to create unified index package
def create_all_indexes(
    tokenized_docs: Union[List[List[str]], Dict[int, List[str]]],
    index_path: str,
    doc_ids: Optional[List[int]] = None
) -> None:
    """
    Build a unified index package from tokenized documents.

    Args:
        tokenized_docs: List or dict of tokenized documents
                        If list, each document is a list of tokens
                        If dict, keys are doc IDs, values are list of tokens
        index_path: Path where the unified index package will be saved
        doc_ids: Optional list of document IDs. If None, uses sequential IDs (0, 1, 2, ...)
                 Must match the length of tokenized_docs if provided.
    """
    # If no document IDs provided, assign sequential IDs starting from 0
    if doc_ids is None:
        if isinstance(tokenized_docs, dict):
            doc_ids = list(tokenized_docs.keys())
        else:
            doc_ids = list(range(len(tokenized_docs)))

    # Initialize sub-indexes
    unified_index: Dict[Union[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    wildcard_index: Dict[str, List[str]] = defaultdict(list)
    proximity_index: Dict[Union[str, Tuple[str, ...]], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    doc_lengths: Dict[int, int] = {}

    # Loop over each document in the collection
    for doc_idx, doc_id in enumerate(doc_ids):
        # Handle both dict and list for tokenized_docs
        if isinstance(tokenized_docs, dict):
            tokens = tokenized_docs[doc_id]
        else:
            tokens = tokenized_docs[doc_idx]

        tokens = normalize_tokens(tokens)           # normalize tokens
        doc_lengths[doc_id] = len(tokens)           # store document length

        # Build token n-grams for unified index & proximity
        for n in range(1, NGRAM_MAX + 1):
            for ngram, positions in make_ngrams_tokens(tokens, n):
                if doc_id not in unified_index[ngram]:
                    unified_index[ngram].append(doc_id)
                proximity_index[ngram][doc_id].extend(positions)

        # Build character n-grams for wildcard index
        for token in tokens:
            for cng in make_ngrams_chars(token, CHAR_NGRAM_MAX):
                if token not in wildcard_index[cng]:
                    wildcard_index[cng].append(token)

    # Sort and deduplicate posting lists in the unified index
    for key in unified_index:
        unified_index[key] = sorted(set(unified_index[key]))

    # Sort and deduplicate positions in the proximity index
    for key, doc_pos in proximity_index.items():
        for doc_id, positions in doc_pos.items():
            proximity_index[key][doc_id] = sorted(set(positions))

    # Sort and deduplicate token lists in the wildcard index
    for cng in wildcard_index:
        wildcard_index[cng] = sorted(set(wildcard_index[cng]))

    # Compute meta information about the corpus
    N = len(doc_ids)  # Total number of documents
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 0

    meta = {
        "N": N,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "version": "1.0",
        "ngrams_max": NGRAM_MAX,
        "char_ngrams_max": CHAR_NGRAM_MAX
    }

    # Create the unified index package containing all sub-indexes and meta info
    index_package = {
        "__META__": meta,
        "unified": dict(unified_index),
        "wildcard": dict(wildcard_index),
        "proximity": dict(proximity_index)
    }

    # Save the index package to disk using gzip compression
    dump(index_package, index_path)
    print(f"Unified index package saved to {index_path}")
