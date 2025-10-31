import { createTool } from '@mastra/core/tools';
import { MDocument } from '@mastra/rag';
import { embedMany } from 'ai';
import { openai } from '@ai-sdk/openai';
import { PgVector } from '@mastra/pg';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { z } from 'zod';

interface ProcessedChunk {
  text: string;
  metadata: Record<string, any>;
  embedding?: number[];
}

export const jsonProcessorTool = createTool({
  id: 'process-json-data',
  description: 'Process JSON files and embed them into the vector database',
  inputSchema: z.object({
    filePath: z.string().describe('Path to the JSON file to process'),
    indexName: z.string().describe('Name of the vector index to store embeddings'),
    chunkSize: z.number().optional().default(512).describe('Maximum size of each chunk'),
    overwrite: z.boolean().optional().default(false).describe('Whether to overwrite existing data'),
  }),
  outputSchema: z.object({
    chunksProcessed: z.number(),
    embeddingsGenerated: z.number(),
    indexName: z.string(),
    status: z.string(),
  }),
  execute: async ({ context }) => {
    try {
      const { filePath, indexName, chunkSize, overwrite } = context;
      
      // Initialize vector store
      const vectorStore = new PgVector({
        connectionString: process.env.POSTGRES_CONNECTION_STRING!,
      });

      // Read and parse JSON file
      const absolutePath = join(process.cwd(), 'src', 'data', filePath);
      const jsonContent = await readFile(absolutePath, 'utf-8');
      const jsonData = JSON.parse(jsonContent);

      // Create document from JSON
      const doc = MDocument.fromJSON(jsonContent, {
        source: filePath,
        type: 'json',
        processedAt: new Date().toISOString(),
      });

      // Chunk the document with JSON-aware strategy
      const chunks = await doc.chunk({
        strategy: 'json',
        maxSize: chunkSize,
        minSize: 50,
        ensureAscii: false,
        convertLists: true,
        extract: {
          title: true,
          summary: true,
          keywords: true,
        },
      });

      // Generate embeddings
      const { embeddings } = await embedMany({
        model: openai.embedding('text-embedding-3-small'),
        values: chunks.map((chunk) => chunk.text),
      });

      // Create index if it doesn't exist
      try {
        await vectorStore.createIndex({
          indexName,
          dimension: 1536, // OpenAI text-embedding-3-small dimension
        });
      } catch (error) {
        // Index might already exist, continue
        console.log(`Index ${indexName} may already exist:`, error);
      }

      // Prepare metadata for storage
      const metadata = chunks.map((chunk, index) => ({
        text: chunk.text,
        id: `${filePath}_chunk_${index}`,
        source: filePath,
        type: 'json',
        chunkIndex: index,
        processedAt: new Date().toISOString(),
        ...chunk.metadata,
        // Extract specific fields from the original JSON for filtering
        ...(jsonData.products && { contentType: 'product' }),
        ...(jsonData.documentation && { contentType: 'documentation' }),
        ...(jsonData.users && { contentType: 'user' }),
      }));

      // Store embeddings with metadata
      await vectorStore.upsert({
        indexName,
        vectors: embeddings,
        metadata,
      });

      return {
        chunksProcessed: chunks.length,
        embeddingsGenerated: embeddings.length,
        indexName,
        status: 'success',
      };

    } catch (error) {
      console.error('Error processing JSON data:', error);
      throw new Error(`Failed to process JSON data: ${error}`);
    }
  },
});

// Utility function to batch process multiple JSON files
export const processBatchJsonFiles = async (
  filePaths: string[],
  indexName: string,
  chunkSize: number = 512
) => {
  const results = [];
  
  for (const filePath of filePaths) {
    try {
      const result = await jsonProcessorTool.execute({
        context: {
          filePath,
          indexName,
          chunkSize,
          overwrite: false,
        },
        mastra: null as any, // Not used in this context
      });
      results.push({ filePath, ...result });
    } catch (error) {
      results.push({ 
        filePath, 
        status: 'error', 
        error: (error as Error).message 
      });
    }
  }
  
  return results;
};