# RAG with JSON Data in Mastra

This tutorial demonstrates how to implement Retrieval-Augmented Generation (RAG) with JSON data using Mastra. You'll learn how to embed JSON documents, store them in a vector database, and create intelligent search capabilities.

## What You'll Learn

- How to process and chunk JSON documents for RAG
- Setting up vector storage with pgVector (PostgreSQL)
- Creating embeddings from JSON data using OpenAI
- Building a search agent that can query your JSON knowledge base
- Implementing metadata filtering for precise results

## Prerequisites

- Node.js 20.9.0 or higher
- PostgreSQL with pgvector extension (or you can use other vector stores)
- OpenAI API key for embeddings and LLM

## Overview

RAG (Retrieval-Augmented Generation) enhances AI responses by providing relevant context from your own data. This tutorial focuses specifically on working with JSON data, which is common for:

- Product catalogs
- API documentation
- Configuration files
- Structured business data
- User profiles and preferences

## Project Structure

We'll add the following components to your Mastra project:

```
src/
├── mastra/
│   ├── agents/
│   │   ├── weather-agent.ts           # (existing)
│   │   └── knowledge-agent.ts         # NEW: RAG search agent
│   ├── tools/
│   │   ├── weather-tool.ts           # (existing)
│   │   ├── json-processor-tool.ts    # NEW: JSON processing tool
│   │   └── vector-search-tool.ts     # NEW: Vector search tool
│   ├── workflows/
│   │   ├── weather-workflow.ts       # (existing)
│   │   └── knowledge-workflow.ts     # NEW: RAG workflow
│   └── index.ts
├── data/
│   ├── products.json                 # Sample JSON data
│   ├── documentation.json            # Sample JSON data
│   └── users.json                    # Sample JSON data
└── examples/
    ├── basic-search.ts               # Basic RAG usage
    ├── filtered-search.ts            # Metadata filtering
    └── bulk-embedding.ts             # Batch processing
```

## Step 1: Install Dependencies

First, let's add the required packages for RAG functionality:

```bash
npm install @mastra/rag @mastra/pg @ai-sdk/openai ai
```

## Step 2: Environment Setup

Update your `.env` file with the necessary API keys and database connection:

```env
# OpenAI for embeddings and LLM
OPENAI_API_KEY=your-openai-api-key

# PostgreSQL with pgvector for vector storage
POSTGRES_CONNECTION_STRING=postgresql://username:password@localhost:5432/mastra_rag

# Anthropic (existing)
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Step 3: Understanding JSON Chunking

Mastra provides specialized JSON chunking that understands the structure of your JSON documents. Here's how it works:

### Basic JSON Chunking

```typescript
import { MDocument } from "@mastra/rag";

const jsonData = {
  "products": [
    {
      "id": "p1",
      "name": "Wireless Headphones",
      "description": "High-quality noise-canceling headphones",
      "price": 299.99,
      "category": "Electronics",
      "features": ["Bluetooth 5.0", "30hr battery", "Active noise canceling"]
    }
  ]
};

// Create document from JSON
const doc = MDocument.fromJSON(JSON.stringify(jsonData));

// Chunk with JSON-aware strategy
const chunks = await doc.chunk({
  strategy: "json",
  maxSize: 512,
  minSize: 50,
  ensureAscii: false,
  convertLists: true
});
```

### Advanced JSON Processing with Metadata

```typescript
// Process with custom metadata extraction
const chunks = await doc.chunk({
  strategy: "json",
  maxSize: 512,
  extract: {
    title: true,        // Extract titles from JSON objects
    summary: true,      // Generate summaries
    keywords: true      // Extract keywords
  }
});
```

## Step 4: Vector Storage Setup

We'll use PostgreSQL with pgvector for storing embeddings:

```typescript
import { PgVector } from "@mastra/pg";

const vectorStore = new PgVector({
  connectionString: process.env.POSTGRES_CONNECTION_STRING!,
});

// Create index for embeddings
await vectorStore.createIndex({
  indexName: "json_knowledge_base",
  dimension: 1536, // OpenAI text-embedding-3-small dimension
});
```

## Step 5: Embedding Generation

Generate embeddings using OpenAI's embedding model:

```typescript
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";

// Generate embeddings for chunks
const { embeddings } = await embedMany({
  model: openai.embedding("text-embedding-3-small"),
  values: chunks.map((chunk) => chunk.text),
});

// Store embeddings with metadata
await vectorStore.upsert({
  indexName: "json_knowledge_base",
  vectors: embeddings,
  metadata: chunks.map((chunk, index) => ({
    text: chunk.text,
    id: `chunk_${index}`,
    source: chunk.metadata?.source || "unknown",
    type: chunk.metadata?.type || "json",
    ...chunk.metadata
  })),
});
```

## Step 6: Building the RAG Agent

Create a search agent that can query your JSON knowledge base:

```typescript
import { Agent } from '@mastra/core/agent';
import { createVectorQueryTool } from '@mastra/rag';

const vectorQueryTool = createVectorQueryTool({
  vectorStoreName: "pgVector",
  indexName: "json_knowledge_base",
  model: openai.embedding("text-embedding-3-small"),
});

export const knowledgeAgent = new Agent({
  name: 'Knowledge Search Agent',
  instructions: `
    You are a helpful assistant that searches through JSON-based knowledge.
    
    When users ask questions:
    1. Use the vector search tool to find relevant information
    2. Provide accurate answers based on the retrieved data
    3. If no relevant information is found, say so clearly
    4. Include source references when possible
    
    The knowledge base contains:
    - Product information (names, descriptions, prices, features)
    - Documentation and guides
    - User preferences and settings
    
    Always be precise and cite the specific data sources.
  `,
  model: 'gpt-4o-mini',
  tools: { vectorQueryTool },
});
```

## Step 7: Advanced Features

### Metadata Filtering

Filter search results based on JSON properties:

```typescript
const results = await vectorStore.query({
  indexName: "json_knowledge_base",
  queryVector: queryEmbedding,
  topK: 5,
  filter: {
    category: "Electronics",           // Exact match
    price: { $lt: 500 },              // Less than $500
    features: { $in: ["Bluetooth"] }   // Contains Bluetooth
  }
});
```

### Hybrid Search with Re-ranking

Improve search quality with re-ranking:

```typescript
import { rerankWithScorer, MastraAgentRelevanceScorer } from "@mastra/rag";

// Initial vector search
const initialResults = await vectorStore.query({
  indexName: "json_knowledge_base",
  queryVector: queryEmbedding,
  topK: 10,
});

// Re-rank for better relevance
const relevanceScorer = new MastraAgentRelevanceScorer(
  'relevance-scorer', 
  openai("gpt-4o-mini")
);

const rerankedResults = await rerankWithScorer({
  results: initialResults,
  query: "wireless headphones with good battery life",
  scorer: relevanceScorer,
  options: { topK: 5 }
});
```

## Common Use Cases

### 1. Product Search
Search through product catalogs with complex filtering:

```typescript
// Find products matching criteria
const productResults = await vectorStore.query({
  indexName: "json_knowledge_base",
  queryVector: await embed("bluetooth headphones under $200"),
  filter: {
    type: "product",
    category: "Electronics",
    price: { $lt: 200 }
  }
});
```

### 2. Documentation Search
Find relevant documentation sections:

```typescript
// Search API documentation
const docResults = await vectorStore.query({
  indexName: "json_knowledge_base",
  queryVector: await embed("how to authenticate API requests"),
  filter: {
    type: "documentation",
    section: { $in: ["authentication", "api"] }
  }
});
```

### 3. User Preference Matching
Match user queries to stored preferences:

```typescript
// Find users with similar preferences
const userResults = await vectorStore.query({
  indexName: "json_knowledge_base",
  queryVector: await embed("outdoor activities hiking camping"),
  filter: {
    type: "user_profile",
    active: true
  }
});
```

## Best Practices

### 1. JSON Structure Optimization
- Keep related data together in objects
- Use consistent field naming
- Include relevant metadata fields
- Normalize data types across documents

### 2. Chunking Strategy
- Adjust `maxSize` based on your JSON complexity
- Use `convertLists: true` for array processing
- Consider object boundaries when chunking
- Test different chunk sizes for your use case

### 3. Metadata Design
- Store searchable fields as top-level metadata
- Include categorical data for filtering
- Add timestamps for temporal queries
- Keep metadata consistent across documents

### 4. Embedding Optimization
- Batch process large datasets
- Use appropriate embedding models for your domain
- Consider custom embedding dimensions
- Monitor embedding costs

### 5. Search Quality
- Implement re-ranking for better results
- Use metadata filtering to narrow searches
- Combine semantic and keyword search
- Test with real user queries

## Performance Considerations

- **Batch Processing**: Process large JSON files in chunks
- **Index Optimization**: Create appropriate database indexes
- **Caching**: Cache frequent queries and embeddings
- **Monitoring**: Track search performance and relevance

## Troubleshooting

### Common Issues

1. **Embedding Dimension Mismatch**
   - Ensure vector store dimension matches embedding model
   - Check for model changes or updates

2. **JSON Chunking Problems**
   - Verify JSON is valid before processing
   - Adjust chunk size for complex nested objects
   - Handle special characters and encoding

3. **Search Quality Issues**
   - Review chunking strategy
   - Add metadata filtering
   - Implement re-ranking
   - Test with diverse queries

## Next Steps

1. Run the basic examples in the `examples/` directory
2. Experiment with different JSON datasets
3. Try various vector stores (Pinecone, Qdrant, etc.)
4. Implement custom chunking strategies
5. Add evaluation metrics for search quality

## Resources

- [Mastra RAG Documentation](https://docs.mastra.ai/rag)
- [Vector Database Guide](https://docs.mastra.ai/rag/vector-databases)
- [Chunking Strategies](https://docs.mastra.ai/rag/chunking-and-embedding)
- [Metadata Filtering](https://docs.mastra.ai/reference/rag/metadata-filters)

---

Ready to start building? Follow the implementation guide in the next sections!