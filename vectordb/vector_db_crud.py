"""
Vector Database CRUD Operations Example
=====================================

This module demonstrates basic CRUD (Create, Read, Update, Delete) operations
using ChromaDB as the vector database backend.

Features:
- Initialize vector database
- Add documents with embeddings
- Search similar documents
- Update existing documents
- Delete documents
- Batch operations
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
import uuid
import os


class VectorDB:
    """
    A simple vector database wrapper for CRUD operations using ChromaDB.
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection to work with
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        return self.encoder.encode(text).tolist()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for documents."""
        return str(uuid.uuid4())
    
    # CREATE operations
    def add_document(self, 
                    text: str, 
                    metadata: Optional[Dict] = None, 
                    document_id: Optional[str] = None) -> str:
        """
        Add a single document to the vector database.
        
        Args:
            text: The text content of the document
            metadata: Optional metadata dictionary
            document_id: Optional custom ID, if not provided, UUID will be generated
            
        Returns:
            The document ID
        """
        if document_id is None:
            document_id = self._generate_id()
        
        if metadata is None:
            metadata = {}
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[document_id]
        )
        
        print(f"Added document with ID: {document_id}")
        return document_id
    
    def add_documents_batch(self, 
                           texts: List[str], 
                           metadatas: Optional[List[Dict]] = None,
                           ids: Optional[List[str]] = None) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            texts: List of text contents
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of custom IDs
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [self._generate_id() for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings for all texts
        embeddings = [self._generate_embedding(text) for text in texts]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(texts)} documents in batch")
        return ids
    
    # READ operations
    def search_similar(self, 
                      query_text: str, 
                      n_results: int = 5,
                      where: Optional[Dict] = None) -> Dict:
        """
        Search for similar documents based on query text.
        
        Args:
            query_text: The query text to search for
            n_results: Number of similar documents to return
            where: Optional metadata filter
            
        Returns:
            Dictionary containing search results
        """
        query_embedding = self._generate_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(ids=[document_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
        except Exception as e:
            print(f"Error retrieving document {document_id}: {e}")
            return None
    
    def get_all_documents(self) -> Dict:
        """
        Get all documents in the collection.
        
        Returns:
            Dictionary containing all documents
        """
        results = self.collection.get()
        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'metadatas': results['metadatas']
        }
    
    def count_documents(self) -> int:
        """Get the total number of documents in the collection."""
        return self.collection.count()
    
    # UPDATE operations
    def update_document(self, 
                       document_id: str, 
                       text: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            document_id: The ID of the document to update
            text: New text content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {}
            
            if text is not None:
                embedding = self._generate_embedding(text)
                update_data['embeddings'] = [embedding]
                update_data['documents'] = [text]
            
            if metadata is not None:
                update_data['metadatas'] = [metadata]
            
            if update_data:
                update_data['ids'] = [document_id]
                self.collection.update(**update_data)
                print(f"Updated document with ID: {document_id}")
                return True
            else:
                print("No update data provided")
                return False
                
        except Exception as e:
            print(f"Error updating document {document_id}: {e}")
            return False
    
    # DELETE operations
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[document_id])
            print(f"Deleted document with ID: {document_id}")
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def delete_documents_batch(self, document_ids: List[str]) -> bool:
        """
        Delete multiple documents by IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=document_ids)
            print(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def delete_by_metadata(self, where: Dict) -> bool:
        """
        Delete documents based on metadata filter.
        
        Args:
            where: Metadata filter dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(where=where)
            print(f"Deleted documents matching filter: {where}")
            return True
        except Exception as e:
            print(f"Error deleting documents by metadata: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                print("Cleared all documents from collection")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False


def main():
    """
    Example usage of the VectorDB class demonstrating all CRUD operations.
    """
    print("Vector Database CRUD Example")
    print("=" * 40)
    
    # Initialize the vector database
    db = VectorDB(collection_name="example_docs")
    
    print(f"\nInitial document count: {db.count_documents()}")
    
    # CREATE - Add documents
    print("\n1. CREATE Operations:")
    print("-" * 20)
    
    # Add single document
    doc_id1 = db.add_document(
        text="Python is a high-level programming language known for its simplicity.",
        metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
    )
    
    # Add multiple documents in batch
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases are optimized for similarity search.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    sample_metadata = [
        {"category": "AI", "topic": "machine_learning"},
        {"category": "database", "topic": "vector_search"},
        {"category": "AI", "topic": "NLP"},
        {"category": "AI", "topic": "deep_learning"}
    ]
    
    batch_ids = db.add_documents_batch(sample_texts, sample_metadata)
    
    print(f"Document count after additions: {db.count_documents()}")
    
    # READ - Search and retrieve
    print("\n2. READ Operations:")
    print("-" * 20)
    
    # Search for similar documents
    search_results = db.search_similar("artificial intelligence and machine learning", n_results=3)
    print(f"Search results for 'artificial intelligence and machine learning':")
    for i, (doc_id, document, distance) in enumerate(zip(
        search_results['ids'], 
        search_results['documents'], 
        search_results['distances']
    )):
        print(f"  {i+1}. ID: {doc_id[:8]}..., Distance: {distance:.4f}")
        print(f"     Text: {document[:50]}...")
    
    # Get specific document
    specific_doc = db.get_document(doc_id1)
    if specific_doc:
        print(f"\nSpecific document ({doc_id1[:8]}...):")
        print(f"  Text: {specific_doc['document'][:50]}...")
        print(f"  Metadata: {specific_doc['metadata']}")
    
    # Get all documents
    all_docs = db.get_all_documents()
    print(f"\nTotal documents in collection: {len(all_docs['ids'])}")
    
    # UPDATE - Modify documents
    print("\n3. UPDATE Operations:")
    print("-" * 20)
    
    # Update a document
    success = db.update_document(
        document_id=doc_id1,
        text="Python is a versatile, high-level programming language widely used in data science.",
        metadata={"category": "programming", "language": "python", "difficulty": "intermediate", "updated": True}
    )
    
    if success:
        updated_doc = db.get_document(doc_id1)
        print(f"Updated document:")
        print(f"  Text: {updated_doc['document'][:60]}...")
        print(f"  Metadata: {updated_doc['metadata']}")
    
    # DELETE - Remove documents
    print("\n4. DELETE Operations:")
    print("-" * 20)
    
    print(f"Document count before deletions: {db.count_documents()}")
    
    # Delete a specific document
    if len(batch_ids) > 0:
        deleted = db.delete_document(batch_ids[0])
        if deleted:
            print(f"Deleted document: {batch_ids[0][:8]}...")
    
    # Delete by metadata
    deleted_by_metadata = db.delete_by_metadata({"category": "database"})
    
    print(f"Document count after deletions: {db.count_documents()}")
    
    # Search with metadata filter
    print("\n5. Filtered Search:")
    print("-" * 20)
    
    filtered_results = db.search_similar(
        "programming language", 
        n_results=2,
        where={"category": "programming"}
    )
    
    print(f"Filtered search results for 'programming language' (category: programming):")
    for i, (doc_id, document) in enumerate(zip(filtered_results['ids'], filtered_results['documents'])):
        print(f"  {i+1}. ID: {doc_id[:8]}..., Text: {document[:50]}...")
    
    print(f"\nFinal document count: {db.count_documents()}")
    
    # Uncomment the line below to clear all documents
    # db.clear_collection()


if __name__ == "__main__":
    main()