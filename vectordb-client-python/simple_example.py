"""
Simple Vector Database Example
=============================

A minimal example demonstrating basic vector database operations
using ChromaDB without external dependencies for embeddings.
"""

import chromadb
import numpy as np
from typing import List, Dict, Optional
import json


class SimpleVectorDB:
    """A simplified vector database example using random embeddings."""
    
    def __init__(self, collection_name: str = "simple_docs"):
        """Initialize the simple vector database."""
        self.client = chromadb.Client()
        self.collection_name = collection_name
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
    
    def _simple_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """
        Generate a simple embedding based on text characteristics.
        In production, use proper embedding models like sentence-transformers.
        """
        # Simple hash-based embedding for demonstration
        text_hash = hash(text.lower())
        np.random.seed(abs(text_hash) % (2**31))
        embedding = np.random.normal(0, 1, dimension)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def add(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the vector database."""
        embedding = self._simple_embedding(text)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return doc_id
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents."""
        query_embedding = self._simple_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'distances': results['distances'][0]
        }
    
    def get(self, doc_id: str) -> Optional[Dict]:
        """Get a document by ID."""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except:
            pass
        return None
    
    def update(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        """Update a document."""
        try:
            embedding = self._simple_embedding(text)
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            return True
        except:
            return False
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except:
            return False
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()


def demo():
    """Simple demonstration of vector database operations."""
    print("Simple Vector Database Demo")
    print("=" * 30)
    
    # Initialize database
    db = SimpleVectorDB()
    
    # Add documents
    print("\nAdding documents...")
    db.add("The cat sat on the mat", "doc1", {"type": "sentence"})
    db.add("Dogs are loyal animals", "doc2", {"type": "sentence"})
    db.add("Machine learning is fascinating", "doc3", {"type": "tech"})
    db.add("Python programming is fun", "doc4", {"type": "tech"})
    
    print(f"Total documents: {db.count()}")
    
    # Search
    print("\nSearching for 'animal'...")
    results = db.search("animal", n_results=2)
    for i, (doc_id, text, distance) in enumerate(zip(
        results['ids'], results['documents'], results['distances']
    )):
        print(f"{i+1}. {doc_id}: {text} (distance: {distance:.3f})")
    
    # Get specific document
    print("\nGetting document 'doc3'...")
    doc = db.get("doc3")
    if doc:
        print(f"Found: {doc['document']}")
    
    # Update document
    print("\nUpdating document 'doc1'...")
    db.update("doc1", "The big cat sat on the small mat", {"type": "sentence", "updated": True})
    
    # Search again
    print("\nSearching for 'cat' after update...")
    results = db.search("cat", n_results=1)
    if results['documents']:
        print(f"Found: {results['documents'][0]}")
    
    # Delete document
    print("\nDeleting document 'doc2'...")
    db.delete("doc2")
    print(f"Documents after deletion: {db.count()}")


if __name__ == "__main__":
    demo()