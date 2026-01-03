import os
import json
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

def save_db_to_disk(db: Any, path: str):
    """
    Saves the VectorDB to a directory safely.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Save Vectors
    vector_path = os.path.join(path, "vectors.npy")
    np.save(vector_path, db.index.vectors)
    
    # Save Metadata
    # Convert IDs to strings for JSON compatibility
    meta_path = os.path.join(path, "metadata.json")
    meta_json = {str(k): v for k, v in db.metadata_store.items()}
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)
        
    # Save Config
    config_path = os.path.join(path, "config.json")
    config = {"dimension": db.dimension}
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    logger.debug(f"Saved DB artifacts to {path}")

def load_db_from_disk(db_class: Any, path: str) -> Any:
    """
    Loads a VectorDB from a directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database path not found: {path}")
        
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found in {path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    dimension = config.get("dimension")
    if not dimension:
        raise ValueError("Invalid config: missing 'dimension'")
        
    new_db = db_class(dimension=dimension)
    
    # Load Vectors
    vector_path = os.path.join(path, "vectors.npy")
    if os.path.exists(vector_path):
        new_db.index.vectors = np.load(vector_path)
    
    # Load Metadata
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_json = json.load(f)
            # Convert keys back to int
            new_db.metadata_store = {int(k): v for k, v in meta_json.items()}
            
    return new_db
