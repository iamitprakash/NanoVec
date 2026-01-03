import numpy as np
import logging
from typing import List, Tuple, Optional
from .exceptions import DimensionMismatchError

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    Manages the dense vector storage and similarity search using NumPy.
    
    Attributes:
        dimension (int): The dimensionality of the vectors.
        vectors (np.ndarray): The 2D array storing all vectors.
    """
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
            
        self.dimension = dimension
        # Pre-allocate? For now, keep dynamic but use float32 explicitly.
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        logger.debug(f"Initialized VectorIndex with dimension={dimension}")
        
    def add(self, vector: np.ndarray) -> int:
        """
        Adds a vector to the index.
        
        Args:
            vector (np.ndarray): The vector to add.
            
        Returns:
            int: The index (ID) of the added vector.
            
        Raises:
            DimensionMismatchError: If the vector dimension is incorrect.
        """
        if vector.shape != (self.dimension,):
            raise DimensionMismatchError(
                f"Expected dimension {self.dimension}, got {vector.shape}"
            )
            
        new_idx = self.vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vector.astype(np.float32)])
        return new_idx

    def add_many(self, vectors: np.ndarray) -> List[int]:
        """
        Adds multiple vectors efficiently.
        
        Args:
            vectors (np.ndarray): A 2D array of shape (N, dimension).
            
        Returns:
            List[int]: A list of assigned indices.
        """
        if vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(
                f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
            )
            
        start_idx = self.vectors.shape[0]
        count = vectors.shape[0]
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        
        return list(range(start_idx, start_idx + count))

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Performs cosine similarity search.
        
        Args:
            query_vector (np.ndarray): The query vector.
            k (int): Number of nearest neighbors to return.
            
        Returns:
            List[Tuple[int, float]]: List of (index, score) sorted by score descending.
        """
        if self.vectors.shape[0] == 0:
            return []
            
        if query_vector.shape != (self.dimension,):
            raise DimensionMismatchError(
                f"Query dimension {query_vector.shape} != index dimension {self.dimension}"
            )

        # 1. Dot Product
        dot_products = np.dot(self.vectors, query_vector)
        
        # 2. Norms
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        
        if query_norm == 0:
            logger.warning("Query vector has zero norm. Returning empty results.")
            return []
            
        # 3. Cosine Similarity
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = dot_products / (vector_norms * query_norm)
            
        # Handle zero vectors in DB (NaNs become -1)
        scores = np.nan_to_num(scores, nan=-1.0)
        
        # 4. Top K
        # Check if we have fewer vectors than k
        k = min(k, len(scores))
        
        # Argsort is ascending, so we reverse it
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))
            
        return results
