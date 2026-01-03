# NanoVec - Lightweight Vector Database

**NanoVec** is a minimal, strictly-typed Vector Database built from scratch in Python using `numpy`. It is designed for **RAG (Retrieval Augmented Generation)** applications where you need a simple, local, and explainable way to store and retrieve embedding vectors.

## Features

-   **🚀 Pure Python & Numpy**: No complex C++ dependencies or heavy servers to run.
-   **📐 Cosine Similarity**: Uses the standard metric for semantic search.
-   **💾 Persistence**: Built-in support to Save and Load your index to disk.
-   **📝 Metadata Support**: Store text chunks, file paths, or JSON data alongside your vectors.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd vector-database
    ```

2.  **Install dependencies**:
    NanoVec only requires `numpy`.
    ```bash
    pip install numpy
    ```
    *Or use the requirements file:*
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

Here is how you can use NanoVec in your python scripts.

### 1. Initialize the Database
```python
from vector_db import VectorDB

# Define the dimension of your embeddings (e.g., 384 for SBERT, 1536 for OpenAI)
# For this example, we use dimension 3.
db = VectorDB(dimension=3)
```

### 2. Add Vectors
Add vectors along with any metadata you want to retrieve later.
```python
# Add "Apple" (Fruit)
db.add([1.0, 0.0, 0.0], {"text": "Apple", "category": "Fruit"})

# Add "Car" (Vehicle) 
db.add([0.0, 1.0, 0.0], {"text": "Car", "category": "Vehicle"})
```

### 3. Semantic Search
Find the most similar vectors to your query.
```python
# Query: Something "Fruit-like"
query_vector = [1.0, 0.0, 0.0]

results = db.search(query_vector, k=1)

for res in results:
    print(f"Found: {res['metadata']['text']} (Score: {res['score']:.4f})")
# Output: Found: Apple (Score: 1.0000)
```

### 4. Save and Load
Persist your database to disk so you don't lose your data.
```python
# Save to a folder named 'my_index'
db.save("my_index")

# Load it back later (no need to re-add vectors)
new_db = VectorDB.load("my_index")
```

## How it Works
NanoVec uses **Brute Force Exact Search**.
1.  **Storage**: Vectors are stacked into a single `numpy.ndarray` matrix.
2.  **Search**: When you query, it computes the **Dot Product** of the query vector against the entire matrix at once.
3.  **Ranking**: It normalizes the results to get **Cosine Similarity** scores (-1 to 1) and sorts them to find the top `k` matches.

This approach is extremely fast for datasets up to ~100k vectors, making it perfect for most local RAG prototypes.
