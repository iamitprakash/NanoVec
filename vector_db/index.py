import numpy as np
from typing import List, Tuple, Optional

class VectorIndex:
    """
    Manages the dense vector storage and similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        # We start with empty storage. 
        # For efficiency in a real DB, we'd pre-allocate or use chunks, 
        # but for this MVP, vstack is fine or just a list of arrays we stack later.
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        
    def add(self, vector: np.ndarray) -> int:
        """
        Adds a vector to the index.
        Returns the index (ID) of the added vector.
        """
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vector.shape}")
            
        new_idx = self.vectors.shape[0]
        # Append to the array. Note: Repeated vstack is slow for large data, 
        # but acceptable for a simple MVP.
        self.vectors = np.vstack([self.vectors, vector.astype(np.float32)])
        return new_idx

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Performs cosine similarity search.
        Returns a list of (index, score) tuples, sorted by score descending.
        """
        if self.vectors.shape[0] == 0:
            return []
            
        if query_vector.shape != (self.dimension,):
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dimension}, got {query_vector.shape}")

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        
        # 1. Compute dot product of query with all vectors
        # vectors shape: (N, D), query shape: (D,) -> result (N,)
        dot_products = np.dot(self.vectors, query_vector)
        
        # 2. Compute norms
        query_norm = np.linalg.norm(query_vector)
        # axis=1 computes norm for each row vector
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        
        # Avoid division by zero
        if query_norm == 0:
            return []
            
        # 3. Compute cosine similarity
        # We need to handle cases where a stored vector is zero vector to avoid nan
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = dot_products / (vector_norms * query_norm)
            
        # Replace NaNs (from zero vectors) with -1 (lowest similarity)
        scores = np.nan_to_num(scores, nan=-1.0)
        
        # 4. Get top k
        # argsort gives indices of sorted elements (ascending)
        # We want descending (highest score first)
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((int(idx), float(scores[idx])))
            
        return results
