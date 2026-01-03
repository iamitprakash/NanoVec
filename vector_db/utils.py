import os
import json
import shutil
import numpy as np
from typing import Dict, Any

def save_db(db, path: str):
    """
    Saves the VectorDB to a directory.
    - vectors.npy: Numpy array of vectors
    - metadata.json: Metadata dictionary
    - config.json: Configuration (dimensions)
    """
    if os.path.exists(path):
        # Backup? allow overwrite? For now, we overwrite safely (remove old first or just overwrite files).
        # To be safe, let's just create if not exists, or warn. 
        # For this implementation, we will assume it's okay to write into it.
        pass
    else:
        os.makedirs(path)
        
    # Save Vectors
    np.save(os.path.join(path, "vectors.npy"), db.index.vectors)
    
    # Save Metadata
    # Convert int keys to string for JSON
    meta_json = {str(k): v for k, v in db.metadata_store.items()}
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta_json, f, indent=2)
        
    # Save Config
    config = {"dimension": db.dimension}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
        
def load_db(db_class, path: str):
    """
    Loads a VectorDB from a directory.
    Returns a new instance of VectorDB.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database path not found: {path}")
        
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)
        
    dimension = config["dimension"]
    new_db = db_class(dimension=dimension)
    
    # Load Vectors
    vectors = np.load(os.path.join(path, "vectors.npy"))
    new_db.index.vectors = vectors
    
    # Load Metadata
    with open(os.path.join(path, "metadata.json"), "r") as f:
        meta_json = json.load(f)
        # Convert keys back to int
        new_db.metadata_store = {int(k): v for k, v in meta_json.items()}
        
    return new_db
