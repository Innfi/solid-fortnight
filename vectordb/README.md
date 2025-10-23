# Vector Database CRUD Examples

This directory contains Python examples demonstrating basic CRUD (Create, Read, Update, Delete) operations with vector databases.

## Files

### 1. `vector_db_crud.py`
A comprehensive example using ChromaDB with sentence-transformers for real embeddings.

**Features:**
- Full CRUD operations (Create, Read, Update, Delete)
- Batch operations for multiple documents
- Metadata filtering
- Similarity search with configurable results
- Document management (count, clear, reset)
- Persistent storage

**Dependencies:**
- chromadb
- sentence-transformers
- numpy
- pandas

### 2. `simple_example.py`
A minimal example using ChromaDB with simple hash-based embeddings (no external ML dependencies).

**Features:**
- Basic CRUD operations
- Simple embedding generation
- Lightweight implementation
- Good for learning the concepts

### 3. `requirements.txt`
Contains all necessary Python package dependencies.

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Run the comprehensive example:
```bash
python vector_db_crud.py
```

### Run the simple example:
```bash
python simple_example.py
```

## CRUD Operations Overview

### Create (Add Documents)
```python
# Single document
doc_id = db.add_document(text="Your document text", metadata={"category": "example"})

# Batch documents
ids = db.add_documents_batch(texts=["doc1", "doc2"], metadatas=[{}, {}])
```

### Read (Search & Retrieve)
```python
# Similarity search
results = db.search_similar("query text", n_results=5)

# Get specific document
doc = db.get_document(doc_id)

# Get all documents
all_docs = db.get_all_documents()
```

### Update (Modify Documents)
```python
# Update document content and metadata
success = db.update_document(doc_id, text="new text", metadata={"updated": True})
```

### Delete (Remove Documents)
```python
# Delete single document
success = db.delete_document(doc_id)

# Delete multiple documents
success = db.delete_documents_batch([id1, id2, id3])

# Delete by metadata filter
success = db.delete_by_metadata({"category": "temp"})
```

## Key Concepts

### Vector Embeddings
- Convert text into numerical vectors
- Enable similarity search based on semantic meaning
- Generated using pre-trained language models

### Similarity Search
- Find documents similar to a query
- Based on cosine similarity or other distance metrics
- Returns ranked results with similarity scores

### Metadata Filtering
- Filter search results based on document metadata
- Useful for categorization and organization
- Supports complex query conditions

## Advanced Features

The comprehensive example includes:
- Persistent storage (data survives restarts)
- Batch operations for efficiency
- Metadata-based filtering and deletion
- Collection management (clear, reset)
- Error handling and logging

## Production Considerations

For production use, consider:
- Using more sophisticated embedding models
- Implementing proper error handling
- Adding authentication and authorization
- Setting up proper indexing strategies
- Monitoring and performance optimization
- Data backup and recovery procedures