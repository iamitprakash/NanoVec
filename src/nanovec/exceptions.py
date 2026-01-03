class NanoVecError(Exception):
    """Base exception for all NanoVec errors."""
    pass

class DimensionMismatchError(NanoVecError):
    """Raised when vector dimensions do not match the index dimension."""
    pass

class EmptyIndexError(NanoVecError):
    """Raised when performing operations on an empty index that require data."""
    pass
