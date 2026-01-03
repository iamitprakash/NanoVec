import numpy as np
import pytest
from nanovec import VectorDB, NanoVecError, DimensionMismatchError

def test_initialization():
    db = VectorDB(dimension=3)
    assert db.dimension == 3
    assert db.index.vectors.shape == (0, 3)

def test_add_and_search():
    db = VectorDB(dimension=2)
    
    # Add vectors
    idx1 = db.add([1.0, 0.0], {"name": "A"})
    idx2 = db.add([0.0, 1.0], {"name": "B"})
    
    assert idx1 == 0
    assert idx2 == 1
    
    # Search for A
    results = db.search([1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0]['id'] == 0
    assert results[0]['score'] >= 0.99
    
def test_dimension_mismatch():
    db = VectorDB(dimension=3)
    with pytest.raises(DimensionMismatchError):
        db.add([1.0, 0.0], {}) # Dimension 2 instead of 3

def test_batch_add():
    db = VectorDB(dimension=2)
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    meta = [{"i": 0}, {"i": 1}, {"i": 2}]
    
    indices = db.add_many(vectors, meta)
    assert len(indices) == 3
    assert db.index.vectors.shape == (3, 2)
    
def test_filtering():
    db = VectorDB(dimension=2)
    db.add([1.0, 0.0], {"type": "fruit", "name": "apple"})
    db.add([0.9, 0.1], {"type": "fruit", "name": "pear"})
    db.add([0.0, 1.0], {"type": "car", "name": "f150"})
    
    # Search for fruit-like but filter for car
    # Should get the car even if score is lower, or nothing if score is too low?
    # Logic: It fetches Top K first, THEN filters? Or filters during search?
    # Our implementation: Fetches K (or expanded K), then filters.
    
    results = db.search([1.0, 0.0], k=5, filter_meta={"type": "car"})
    assert len(results) == 1
    assert results[0]['metadata']['name'] == "f150"

def test_persistence(tmp_path):
    db = VectorDB(dimension=2)
    db.add([1.0, 0.0], {"data": "test"})
    
    save_dir = tmp_path / "test_db"
    db.save(str(save_dir))
    
    new_db = VectorDB.load(str(save_dir))
    assert new_db.dimension == 2
    assert len(new_db.metadata_store) == 1
    
    results = new_db.search([1.0, 0.0], k=1)
    assert results[0]['metadata']['data'] == "test"
