import numpy as np
from nanovec import VectorDB


def main():
    print("Initializing Vector Database (Dimension=3)...")
    db = VectorDB(dimension=3)

    # data: (vector, text)
    # Visualizing in 3D:
    # Axis 0: Fruits
    # Axis 1: Vehicles
    # Axis 2: Tech
    
    data = [
        ([1.0, 0.0, 0.0], "Apple"),
        ([0.9, 0.1, 0.0], "Pear"),      # Very close to Apple
        ([0.0, 1.0, 0.0], "Car"),
        ([0.0, 0.9, 0.1], "Truck"),     # Close to Car
        ([0.0, 0.0, 1.0], "Laptop"),
        ([0.1, 0.0, 0.9], "Tablet"),    # Close to Laptop
    ]

    print("\nAdding documents...")
    for vec, text in data:
        db.add(vec, {"text": text})
        print(f"Added: {text} -> {vec}")

    # Query 1: Search for fruit-like object
    query_vec = [1.0, 0.0, 0.0] # "Apple-like"
    print(f"\nQuerying for vector: {query_vec} (Fruit-like)")
    results = db.search(query_vec, k=3)
    
    print("Results:")
    for res in results:
        print(f"  {res['metadata']['text']} (Score: {res['score']:.4f})")

    # Query 2: Search for vehicle-like object
    query_vec = [0.0, 1.0, 0.0] # "Car-like"
    print(f"\nQuerying for vector: {query_vec} (Vehicle-like)")
    results = db.search(query_vec, k=3)
    
    print("Results:")
    for res in results:
        print(f"  {res['metadata']['text']} (Score: {res['score']:.4f})")

    # Persistence Demo
    print("\n--- Persistence Demo ---")
    save_path = "my_vector_db"
    print(f"Saving database to '{save_path}'...")
    db.save(save_path)
    
    print("Loading database from disk...")
    new_db = VectorDB.load(save_path)
    
    print("Querying loaded database (Vehicle-like)...")
    results = new_db.search([0.0, 1.0, 0.0], k=1)
    print(f"Top Result: {results[0]['metadata']['text']} (Score: {results[0]['score']:.4f})")
    
    import shutil
    shutil.rmtree(save_path)
    print("Cleaned up.")

if __name__ == "__main__":
    main()
