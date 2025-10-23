"""
Advanced Vector Database Operations
==================================

This module demonstrates more advanced vector database operations including
batch processing, complex queries, and performance optimizations.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import time
import json
from pathlib import Path


class AdvancedVectorDB:
    """Advanced vector database with additional features and optimizations."""
    
    def __init__(self, 
                 collection_name: str = "advanced_docs",
                 persist_directory: str = "./advanced_chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize advanced vector database.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence
            embedding_model: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
    
    def add_documents_from_csv(self, csv_path: str, text_column: str, 
                              metadata_columns: Optional[List[str]] = None) -> List[str]:
        """
        Add documents from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Column name containing text
            metadata_columns: Columns to include as metadata
            
        Returns:
            List of document IDs
        """
        df = pd.read_csv(csv_path)
        
        texts = df[text_column].astype(str).tolist()
        ids = [f"csv_doc_{i}" for i in range(len(texts))]
        
        if metadata_columns:
            metadatas = df[metadata_columns].to_dict('records')
        else:
            metadatas = [{"source": "csv", "row": i} for i in range(len(texts))]
        
        return self.add_documents_batch(texts, metadatas, ids)
    
    def add_documents_batch(self, texts: List[str], metadatas: List[Dict], 
                           ids: List[str], batch_size: int = 100) -> List[str]:
        """
        Add documents in batches for better performance.
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            batch_size: Size of each batch
            
        Returns:
            List of document IDs
        """
        total_docs = len(texts)
        added_ids = []
        
        for i in range(0, total_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Generate embeddings for batch
            embeddings = self._generate_embeddings_batch(batch_texts)
            
            # Add batch to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            added_ids.extend(batch_ids)
            print(f"Added batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
        
        return added_ids
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts efficiently."""
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def hybrid_search(self, query: str, n_results: int = 10,
                     keyword_weight: float = 0.3) -> Dict:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            n_results: Number of results to return
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            Search results with hybrid scores
        """
        # Semantic search
        semantic_results = self.search_similar(query, n_results * 2)
        
        # Simple keyword matching
        query_keywords = set(query.lower().split())
        
        # Combine scores
        hybrid_results = []
        for i, (doc_id, document, semantic_score) in enumerate(zip(
            semantic_results['ids'],
            semantic_results['documents'], 
            semantic_results['distances']
        )):
            # Calculate keyword score
            doc_keywords = set(document.lower().split())
            keyword_matches = len(query_keywords.intersection(doc_keywords))
            keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # Combine scores (lower distance = higher similarity)
            combined_score = (1 - keyword_weight) * (1 - semantic_score) + keyword_weight * keyword_score
            
            hybrid_results.append({
                'id': doc_id,
                'document': document,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'results': hybrid_results[:n_results],
            'total_found': len(hybrid_results)
        }
    
    def search_similar(self, query_text: str, n_results: int = 5,
                      where: Optional[Dict] = None) -> Dict:
        """Enhanced similarity search with timing."""
        start_time = time.time()
        
        query_embedding = self.encoder.encode([query_text])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        search_time = time.time() - start_time
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'search_time': search_time
        }
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        all_docs = self.collection.get()
        
        # Basic stats
        stats = {
            'total_documents': len(all_docs['ids']),
            'collection_name': self.collection_name,
            'embedding_model': self.encoder.get_sentence_embedding_dimension()
        }
        
        # Metadata analysis
        if all_docs['metadatas']:
            metadata_keys = set()
            for metadata in all_docs['metadatas']:
                metadata_keys.update(metadata.keys())
            
            stats['metadata_fields'] = list(metadata_keys)
            
            # Count unique values for each metadata field
            metadata_counts = {}
            for key in metadata_keys:
                values = [meta.get(key) for meta in all_docs['metadatas'] if key in meta]
                metadata_counts[key] = len(set(str(v) for v in values if v is not None))
            
            stats['metadata_unique_counts'] = metadata_counts
        
        return stats
    
    def export_to_json(self, output_path: str) -> bool:
        """Export all documents to JSON file."""
        try:
            all_docs = self.collection.get()
            
            export_data = {
                'collection_name': self.collection_name,
                'total_documents': len(all_docs['ids']),
                'export_timestamp': time.time(),
                'documents': []
            }
            
            for i in range(len(all_docs['ids'])):
                doc_data = {
                    'id': all_docs['ids'][i],
                    'document': all_docs['documents'][i],
                    'metadata': all_docs['metadatas'][i]
                }
                export_data['documents'].append(doc_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(all_docs['ids'])} documents to {output_path}")
            return True
        
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def import_from_json(self, json_path: str) -> bool:
        """Import documents from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data['documents']
            texts = [doc['document'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            ids = [doc['id'] for doc in documents]
            
            self.add_documents_batch(texts, metadatas, ids)
            print(f"Imported {len(documents)} documents from {json_path}")
            return True
        
        except Exception as e:
            print(f"Import failed: {e}")
            return False
    
    def benchmark_search(self, queries: List[str], n_results: int = 5) -> Dict:
        """Benchmark search performance with multiple queries."""
        results = {
            'total_queries': len(queries),
            'search_times': [],
            'average_time': 0,
            'min_time': float('inf'),
            'max_time': 0
        }
        
        for query in queries:
            start_time = time.time()
            self.search_similar(query, n_results)
            search_time = time.time() - start_time
            
            results['search_times'].append(search_time)
            results['min_time'] = min(results['min_time'], search_time)
            results['max_time'] = max(results['max_time'], search_time)
        
        results['average_time'] = sum(results['search_times']) / len(results['search_times'])
        
        return results


def demo_advanced_features():
    """Demonstrate advanced vector database features."""
    print("Advanced Vector Database Demo")
    print("=" * 40)
    
    # Initialize advanced database
    db = AdvancedVectorDB(collection_name="advanced_demo")
    
    # Sample data
    sample_docs = [
        "Artificial intelligence is transforming modern technology",
        "Machine learning algorithms can identify patterns in data",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret visual information",
        "Deep learning uses neural networks with multiple layers",
        "Data science combines statistics, programming, and domain knowledge",
        "Cloud computing provides scalable infrastructure for applications",
        "Cybersecurity protects digital systems from threats and attacks"
    ]
    
    sample_metadata = [
        {"category": "AI", "difficulty": "intermediate", "topic": "general"},
        {"category": "AI", "difficulty": "advanced", "topic": "ML"},
        {"category": "AI", "difficulty": "intermediate", "topic": "NLP"},
        {"category": "AI", "difficulty": "advanced", "topic": "vision"},
        {"category": "AI", "difficulty": "expert", "topic": "deep_learning"},
        {"category": "data", "difficulty": "intermediate", "topic": "science"},
        {"category": "infrastructure", "difficulty": "beginner", "topic": "cloud"},
        {"category": "security", "difficulty": "intermediate", "topic": "cyber"}
    ]
    
    doc_ids = [f"adv_doc_{i}" for i in range(len(sample_docs))]
    
    # Add documents
    print("\n1. Adding documents in batch...")
    db.add_documents_batch(sample_docs, sample_metadata, doc_ids, batch_size=3)
    
    # Get statistics
    print("\n2. Collection Statistics:")
    stats = db.get_statistics()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Metadata fields: {stats['metadata_fields']}")
    print(f"   Metadata counts: {stats['metadata_unique_counts']}")
    
    # Regular similarity search
    print("\n3. Regular Similarity Search:")
    results = db.search_similar("machine learning and AI", n_results=3)
    print(f"   Search time: {results['search_time']:.4f} seconds")
    for i, (doc_id, document, distance) in enumerate(zip(
        results['ids'], results['documents'], results['distances']
    )):
        print(f"   {i+1}. Distance: {distance:.4f} - {document[:50]}...")
    
    # Hybrid search
    print("\n4. Hybrid Search (Semantic + Keywords):")
    hybrid_results = db.hybrid_search("machine learning patterns", n_results=3)
    for i, result in enumerate(hybrid_results['results']):
        print(f"   {i+1}. Combined Score: {result['combined_score']:.4f}")
        print(f"       Semantic: {result['semantic_score']:.4f}, Keywords: {result['keyword_score']:.4f}")
        print(f"       Text: {result['document'][:50]}...")
    
    # Benchmark search performance
    print("\n5. Search Performance Benchmark:")
    test_queries = [
        "artificial intelligence",
        "data analysis",
        "computer security",
        "neural networks",
        "cloud technology"
    ]
    
    benchmark = db.benchmark_search(test_queries)
    print(f"   Average search time: {benchmark['average_time']:.4f} seconds")
    print(f"   Min/Max search time: {benchmark['min_time']:.4f}/{benchmark['max_time']:.4f} seconds")
    
    # Export data
    print("\n6. Exporting data to JSON...")
    export_success = db.export_to_json("vector_db_export.json")
    if export_success:
        print("   Export completed successfully")
    
    print(f"\nDemo completed. Final document count: {db.count_documents()}")


if __name__ == "__main__":
    demo_advanced_features()