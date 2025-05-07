## Setup for Basic RAG

Start with a simple RAG pipeline -

1. Indexing - create fixed sized chunks (1000 tokens), and a fixed size sliding window (200 tokens).
2. Embedding - use nomic-embed-text (137M) model for embedding each chunk and storing it in weaviate (Vector DB).
3. Retrieval - Start with near_text (weaviate built-in), i.e. consine similarity for finding vectors that match the user input query. Input query could come from a web-interface (preferabily) or command line.
4. Generation - Pass the top 5 retrieved chunks along with the user query to an LLM (Llama 3.2 - 1.3B param model) for generation.

This setups a pretty basic RAG application. 

The task is to enhance each step in the pipeline.

## Enhancements

### 1.0 Better Chunking

### 2.0 Better Embeddings
One time cost, we might be able to use a better embedding model. 

### 3.0 Look into Hybrid Retrieval

### 3.1 Reranking

### 4.0 Generation
Look at better prompt strategies.
