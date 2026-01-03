from typing import List, Dict, Any, Optional
import numpy as np
from .index import VectorIndex
from .utils import save_db, load_db

class VectorDB:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = VectorIndex(dimension)
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    def save(self, path: str):
        save_db(self, path)
        
    @classmethod
    def load(cls, path: str):
        return load_db(cls, path)

        
    def add(self, vector: List[float], metadata: Dict[str, Any]) -> int:
        """
        Add a vector and its associated metadata to the database.
        """
        vec_np = np.array(vector, dtype=np.float32)
        idx = self.index.add(vec_np)
        self.metadata_store[idx] = metadata
        return idx
        
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        Returns a list of results with score and metadata.
        """
        vec_np = np.array(query_vector, dtype=np.float32)
        search_results = self.index.search(vec_np, k)
        
        formatted_results = []
        for idx, score in search_results:
            result = {
                "id": idx,
                "score": score,
                "metadata": self.metadata_store.get(idx, {})
            }
            formatted_results.append(result)
            
        return formatted_results
