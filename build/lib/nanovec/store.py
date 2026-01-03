import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .index import VectorIndex
from .utils import save_db_to_disk, load_db_from_disk
from .exceptions import NanoVecError

logger = logging.getLogger(__name__)

class VectorDB:
    """
    Main interface for the NanoVec database.
    Manages the underlying VectorIndex and the Metadata store.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = VectorIndex(dimension)
        # Metadata map: ID -> Data
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        
    def add(self, vector: List[float], metadata: Dict[str, Any] = {}) -> int:
        """
        Add a single vector to the database.
        """
        vec_np = np.array(vector, dtype=np.float32)
        idx = self.index.add(vec_np)
        self.metadata_store[idx] = metadata
        logger.debug(f"Added vector {idx} to database.")
        return idx
        
    def add_many(self, vectors: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple vectors to the database in a batch.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata items must match.")
            
        vecs_np = np.array(vectors, dtype=np.float32)
        indices = self.index.add_many(vecs_np)
        
        for i, idx in enumerate(indices):
            self.metadata_store[idx] = metadata_list[i]
            
        logger.info(f"Batch added {len(indices)} vectors.")
        return indices
        
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific vector ID.
        """
        return self.metadata_store.get(idx)

    def search(self, query_vector: List[float], k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors. Optionally filter by exactly matching metadata fields.
        """
        vec_np = np.array(query_vector, dtype=np.float32)
        raw_results = self.index.search(vec_np, k=k if not filter_meta else 1000) # Fetch more if filtering
        
        formatted_results = []
        count = 0
        
        for idx, score in raw_results:
            meta = self.metadata_store.get(idx, {})
            
            # Simple exact match filtering
            if filter_meta:
                match = True
                for key, val in filter_meta.items():
                    if meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            
            result = {
                "id": idx,
                "score": score,
                "metadata": meta
            }
            formatted_results.append(result)
            count += 1
            if count >= k:
                break
                
        return formatted_results

    def save(self, path: str):
        """Save database to disk."""
        save_db_to_disk(self, path)
        logger.info(f"Database saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'VectorDB':
        """Load database from disk."""
        db = load_db_from_disk(cls, path)
        logger.info(f"Database loaded from {path}")
        return db
