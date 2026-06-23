import pickle, pathlib, gzip
from typing import Any

def dump(obj: Any, path: str):
    """
    Save a Python object to a gzipped pickle file.
    
    Args:
        obj: The Python object to save
        path: File path to save the gzipped pickle
    """
    path = pathlib.Path(path)  # Convert path to Path object for consistency
    with gzip.open(path, "wb") as f:  # Open file in write-binary mode with gzip
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)  
        # Use highest pickle protocol for efficiency and compatibility

def load(path: str) -> Any:
    """
    Load a Python object from a gzipped pickle file.
    
    Args:
        path: File path to the gzipped pickle
        
    Returns:
        The Python object stored in the file
    """
    with gzip.open(path, "rb") as f:  # Open file in read-binary mode with gzip
        return pickle.load(f)  # Load and return the object

