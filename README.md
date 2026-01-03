# NanoVec - Lightweight Vector Database

[![Tests](https://github.com/iamitprakash/NanoVec/actions/workflows/python-test.yml/badge.svg)](https://github.com/iamitprakash/NanoVec/actions/workflows/python-test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NanoVec** is a production-ready, strictly-typed Vector Database built from scratch in Python using `numpy`. It provides a simple, dependency-free interface for local RAG (Retrieval Augmented Generation) applications.

## Features

-   **🚀 Pure Python**: Lightweight and fast.
-   **📦 Installable**: standard python package.
-   **💾 Persistence**: Save and Load your index to disk.
-   **🔍 Filtering**: Search with metadata filters.
-   **⚡ Batch Operations**: Efficiently add thousands of vectors at once.

## Installation

You can install NanoVec directly from source:

```bash
pip install .
```

(Coming to PyPI soon!)

## Usage

### 1. Basic Setup
```python
from nanovec import VectorDB

# Initialize
db = VectorDB(dimension=384) # e.g. for SBERT
```

### 2. Batch Ingestion (Recommended)
```python
vectors = [
    [0.1, 0.2, ...], 
    [0.3, 0.4, ...]
]
metadata = [
    {"id": "doc1", "text": "Hello"},
    {"id": "doc2", "text": "World"}
]

db.add_many(vectors, metadata)
```

### 3. Searching with Filters
```python
# Find nearest neighbor that is also a 'fruit'
results = db.search(
    query_vector=[...], 
    k=5, 
    filter_meta={"category": "fruit"}
)
```

## CLI / Examples

Check the `examples/` directory for a complete running demo:

```bash
python examples/demo.py
```

## License

MIT
