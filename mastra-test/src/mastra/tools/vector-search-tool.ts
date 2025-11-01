import { createTool } from '@mastra/core/tools';
import { embed } from 'ai';
import { openai } from '@ai-sdk/openai';
import { PgVector } from '@mastra/pg';
import { rerankWithScorer, MastraAgentRelevanceScorer } from '@mastra/rag';
import { z } from 'zod';

export const vectorSearchTool = createTool({
  id: 'search-knowledge-base',
  description: 'Search through the JSON knowledge base using semantic search with optional filtering and reranking',
  inputSchema: z.object({
    query: z.string().describe('Search query text'),
    indexName: z.string().describe('Name of the vector index to search'),
    topK: z.number().optional().default(5).describe('Number of results to return'),
    filter: z.record(z.any()).optional().describe('Metadata filters for search'),
    useReranking: z.boolean().optional().default(false).describe('Whether to use reranking for better results'),
    contentType: z.enum(['product', 'documentation', 'user', 'all']).optional().default('all').describe('Type of content to search'),
  }),
  outputSchema: z.object({
    results: z.array(z.object({
      text: z.string(),
      score: z.number(),
      metadata: z.record(z.any()),
    })),
    query: z.string(),
    totalResults: z.number(),
    reranked: z.boolean(),
  }),
  execute: async ({ context }) => {
    try {
      const { query, indexName, topK, filter, useReranking, contentType } = context;
      
      // Initialize vector store
      const vectorStore = new PgVector({
        connectionString: process.env.POSTGRES_CONNECTION_STRING!,
      });

      // Generate query embedding
      const { embedding } = await embed({
        model: openai.embedding('text-embedding-3-small'),
        value: query,
      });

      // Build filter based on content type
      let searchFilter = filter || {};
      if (contentType !== 'all') {
        searchFilter = {
          ...searchFilter,
          contentType,
        };
      }

      // Perform vector search
      const searchResults = await vectorStore.query({
        indexName,
        queryVector: embedding,
        topK: useReranking ? topK * 2 : topK, // Get more results if reranking
        filter: Object.keys(searchFilter).length > 0 ? searchFilter : undefined,
      });

      let finalResults = searchResults;

      // Apply reranking if requested
      if (useReranking && searchResults.length > 0) {
        const relevanceScorer = new MastraAgentRelevanceScorer(
          'relevance-scorer',
          openai('gpt-4o-mini')
        );

        const rerankedResults = await rerankWithScorer({
          results: searchResults,
          query,
          scorer: relevanceScorer,
          options: { topK },
        });

        finalResults = rerankedResults.map(item => item.result);
      }

      // Format results
      const formattedResults = finalResults.map(result => ({
        text: result.metadata?.text || '',
        score: result.score || 0,
        metadata: result.metadata || {},
      }));

      return {
        results: formattedResults,
        query,
        totalResults: formattedResults.length,
        reranked: useReranking,
      };

    } catch (error) {
      console.error('Error searching knowledge base:', error);
      throw new Error(`Failed to search knowledge base: ${error}`);
    }
  },
});

// Specialized search tools for different content types
// export const productSearchTool = createTool({
//   id: 'search-products',
//   description: 'Search for products with price, category, and feature filtering',
//   inputSchema: z.object({
//     query: z.string().describe('Product search query'),
//     indexName: z.string().describe('Vector index name'),
//     maxPrice: z.number().optional().describe('Maximum price filter'),
//     category: z.string().optional().describe('Product category filter'),
//     brand: z.string().optional().describe('Brand filter'),
//     inStock: z.boolean().optional().describe('Filter for in-stock products only'),
//   }),
//   outputSchema: z.object({
//     products: z.array(z.object({
//       name: z.string(),
//       price: z.number(),
//       description: z.string(),
//       category: z.string(),
//       brand: z.string(),
//       score: z.number(),
//     })),
//     totalFound: z.number(),
//   }),
//   execute: async ({ context }) => {
//     const { query, indexName, maxPrice, category, brand, inStock } = context;
//     
//     // Build product-specific filter
//     const filter: Record<string, any> = {
//       contentType: 'product',
//     };
//     
//     if (maxPrice) {
//       filter['metadata.price'] = { $lte: maxPrice };
//     }
//     
//     if (category) {
//       filter['metadata.category'] = category;
//     }
//     
//     if (brand) {
//       filter['metadata.brand'] = brand;
//     }
//     
//     if (inStock) {
//       filter['metadata.availability.in_stock'] = true;
//     }
// 
//     // Use the main search tool
//     const searchResult = await vectorSearchTool.execute({
//       context: {
//         query,
//         indexName,
//         topK: 10,
//         filter,
//         useReranking: true,
//         contentType: 'product',
//       },
//     }, { skipValidation: true });
// 
//     // Extract product information from results
//     const products = searchResult.results.map(result => {
//       const metadata = result.metadata;
//       return {
//         name: metadata.name || 'Unknown Product',
//         price: metadata.price || 0,
//         description: metadata.description || result.text,
//         category: metadata.category || 'Unknown',
//         brand: metadata.brand || 'Unknown',
//         score: result.score,
//       };
//     });
// 
//     return {
//       products,
//       totalFound: products.length,
//     };
//   },
// });