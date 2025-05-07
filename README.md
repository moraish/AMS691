Start with a simple RAG pipeline -

1. Indexing - create fixed sized chunks (1000 tokens), and a fixed size sliding window (200 tokens).
2. Embedding - use nomic-embed-text (137M) model for embedding each chunk and storing it in weaviate (Vector DB).
3. Retrieval - Start with near_text (weaviate built-in), i.e. consine similarity for finding vectors that match the user input query. Input query could come from a web-interface (preferabily) or command line.
4. Generation - Pass the top 5 retrieved chunks along with the user query to an LLM (Llama 3.2 - 1.3B param model) for generation.

This setups a pretty basic RAG application. 

The task is to enhance each step in the pipeline.

