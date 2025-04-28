import weaviate
import weaviate.classes as wvc # New import convention >= 4.7
from weaviate.collections import Collection

from langchain_core.documents import Document
from typing import List, Dict, Optional
import logging
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Weaviate Connection ---

def connect_to_weaviate(
    url: str = "http://localhost:8080",
    ollama_url: str = "http://localhost:11434" # Add Ollama URL for readiness check
) -> Optional[weaviate.WeaviateClient]:
    """
    Establishes a connection to a Weaviate instance.

    Args:
        url: The URL of the Weaviate instance.
        ollama_url: The URL of the Ollama instance for checking module readiness.

    Returns:
        A connected weaviate.WeaviateClient object, or None if connection fails.
    """
    logging.info(f"Attempting to connect to Weaviate at {url}")
    try:
        # For local instances, connect_to_local is often easier
        # client = weaviate.connect_to_local()

        # For more control or remote instances:
        client = weaviate.connect_to_custom(
            http_host=url.split(':')[1].replace('//',''), # Extract host
            http_port=int(url.split(':')[2]),            # Extract port
            http_secure=url.startswith("https"),
            grpc_host=url.split(':')[1].replace('//',''), # Often same as http for default setups
            grpc_port=50051, # Default Weaviate gRPC port
            grpc_secure=False, # Adjust if using TLS for gRPC
            # Add headers=... for API keys if needed
        )

        client.connect() # Explicitly connect

        if client.is_ready():
            logging.info("Weaviate client is ready.")
            # Check if Ollama module is available and ready
            meta = client.get_meta()
            if 'text2vec-ollama' in meta['modules']:
                 logging.info("text2vec-ollama module found.")
                 # Note: Direct readiness check for external modules like Ollama isn't
                 # typically part of client.is_ready(). We assume if Weaviate is up
                 # and configured, it will try to reach Ollama when needed.
                 # A basic check could involve trying a small vectorization or query.
            else:
                 logging.warning("text2vec-ollama module not listed in Weaviate meta. Ensure it's enabled in your Weaviate configuration.")

            return client
        else:
            logging.error("Weaviate client failed to connect or become ready.")
            return None
    except Exception as e:
        logging.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
        return None

# --- Collection Management ---

def create_papers_collection(
    client: weaviate.WeaviateClient,
    collection_name: str = "ResearchPapers",
    ollama_api_endpoint: str = "http://host.docker.internal:11434",
    embedding_model: str = "nomic-embed-text",
    generative_model: str = "llama3"
) -> Optional[Collection]:
    # ... docstring ...
    logging.info(f"Attempting to create or get collection: {collection_name}")
    try:
        if client.collections.exists(collection_name):
            logging.info(f"Collection '{collection_name}' already exists. Retrieving it.")
            return client.collections.get(collection_name)

        logging.info(f"Creating collection '{collection_name}'...")
        papers = client.collections.create(
            name=collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_ollama(
                model=embedding_model,
                api_endpoint=ollama_api_endpoint,
            ),
            generative_config=wvc.config.Configure.Generative.ollama(
                model=generative_model,
                api_endpoint=ollama_api_endpoint,
            ),
            properties=[
                # ... properties definitions ...
                wvc.config.Property(name="chunk_text", data_type=wvc.config.DataType.TEXT, description="The text content of the paper chunk", vectorize_property_name=False, tokenization=wvc.config.Tokenization.WORD),
                wvc.config.Property(name="paper_title", data_type=wvc.config.DataType.TEXT, description="Title of the research paper", skip_vectorization=True, tokenization=wvc.config.Tokenization.FIELD),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT, description="Source filename of the PDF", skip_vectorization=True, tokenization=wvc.config.Tokenization.FIELD),
                wvc.config.Property(name="page", data_type=wvc.config.DataType.INT, description="Page number of the chunk", skip_vectorization=True),
                wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT, description="Unique ID for the chunk within the document", skip_vectorization=True, tokenization=wvc.config.Tokenization.FIELD),
                wvc.config.Property(name="start_index", data_type=wvc.config.DataType.INT, description="Start index of the chunk in the original document", skip_vectorization=True),
            ],
            # Corrected vector index configuration:
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                 distance_metric=wvc.config.VectorDistances.COSINE # Specify distance metric within the index type config (e.g., hnsw)
            )
        )
        logging.info(f"Successfully created collection: {collection_name}")
        return papers
    except Exception as e:
        logging.error(f"Failed to create or get collection '{collection_name}': {e}", exc_info=True)
        return None


# --- Data Preparation ---

def prepare_data_for_weaviate(chunked_docs: List[Document]) -> List[Dict]:
    """
    Transforms LangChain Documents into dictionaries for Weaviate batch ingestion.

    Args:
        chunked_docs: A list of LangChain Document objects (from indexing.py).

    Returns:
        A list of dictionaries, each representing an object for Weaviate.
    """
    logging.info(f"Preparing {len(chunked_docs)} documents for Weaviate ingestion.")
    weaviate_objects: List[Dict] = []
    for doc in chunked_docs:
        if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
             logging.warning(f"Skipping invalid document object: {doc}")
             continue

        metadata = doc.metadata
        properties = {
            "chunk_text": doc.page_content,
            "paper_title": metadata.get("title", "Unknown Title"), # Use .get for safety
            "source": metadata.get("source", "Unknown Source"),
            "page": metadata.get("page", -1), # Default to -1 if page not found
            "chunk_id": metadata.get("chunk_id", f"chunk_{len(weaviate_objects)}"), # Provide default
            "start_index": metadata.get("start_index", -1)
        }
        weaviate_objects.append(properties)

    logging.info(f"Successfully prepared {len(weaviate_objects)} objects for Weaviate.")
    return weaviate_objects

# --- Data Ingestion ---

def batch_ingest_data(
    client: weaviate.WeaviateClient,
    collection_name: str,
    weaviate_data: List[Dict],
    batch_size: int = 100,
    requests_per_minute: int = 200 # Adjust based on Weaviate/Ollama capacity
) -> None:
    """
    Ingests data into the specified Weaviate collection using batching.

    Args:
        client: The connected Weaviate client.
        collection_name: The name of the target collection.
        weaviate_data: The list of dictionaries to ingest.
        batch_size: The number of items per batch.
        requests_per_minute: Rate limiting for batch import requests.
    """
    logging.info(f"Starting batch ingestion of {len(weaviate_data)} objects into '{collection_name}'...")
    start_time = time.time()

    try:
        # Access collections through the client instance
        collection = client.collections.get(collection_name)
        processed_count = 0
        # Configure batching
        # Access batch through the collection instance
        with collection.batch.dynamic() as batch: # Dynamic batching is often simplest
        # Or manually configure:
        # with collection.batch.fixed_size(batch_size=batch_size) as batch:
        # with collection.batch.rate_limit(requests_per_minute=requests_per_minute) as batch:
            for i, data_object in enumerate(weaviate_data):
                try:
                    # Add object with its properties
                    batch.add_object(
                        properties=data_object
                        # uuid=... # Optionally provide your own UUIDs
                    )
                    processed_count += 1
                    if (i + 1) % batch_size == 0:
                         logging.info(f"Added {i + 1}/{len(weaviate_data)} objects to current batch...")

                except Exception as e:
                    logging.error(f"Error adding object {i} to batch: {data_object} - {e}", exc_info=True)


        # Check for batch import errors after the 'with' block finishes
        # Access batch errors through the batch manager instance
        if batch.number_errors > 0:
             logging.error(f"Batch import finished with {batch.number_errors} errors.")
             # Log details of failed objects if needed (can be numerous)
             # limit = 10
             # logging.error(f"First {min(limit, len(batch.failed_objects))} failed objects:")
             # for failed in batch.failed_objects[:limit]:
             #      logging.error(f" - {failed.message} (Original object index approx: {failed.original_uuid})") # UUID might not map directly to original list index easily
        else:
             logging.info("Batch import completed successfully with no errors reported by the batch manager.")


    except Exception as e:
        logging.error(f"An unexpected error occurred during batch ingestion: {e}", exc_info=True)

    end_time = time.time()
    logging.info(f"Batch ingestion finished. Processed {processed_count} objects in {end_time - start_time:.2f} seconds.")


# --- Example Usage Placeholder ---
# if __name__ == "__main__":
#     # This is where you would orchestrate the calls,
#     # likely importing functions from indexing.py first.
#
#     # 1. Load and chunk documents (using functions from indexing.py)
#     # from indexing import load_and_parse_pdfs, chunk_documents_recursive
#     # PDF_FOLDER = "../data/papers/" # Adjust path as needed
#     # loaded_docs = load_and_parse_pdfs(PDF_FOLDER)
#     # chunked_docs = chunk_documents_recursive(loaded_docs)
#
#     # 2. Connect to Weaviate
#     # weaviate_client = connect_to_weaviate() # Use default local URL
#
#     # if weaviate_client and chunked_docs:
#     #     # 3. Create or get collection
#     #     papers_collection = create_papers_collection(weaviate_client) # Use defaults
#
#     #     if papers_collection:
#     #         # 4. Prepare data
#     #         weaviate_objects = prepare_data_for_weaviate(chunked_docs)
#
#     #         # 5. Ingest data
#     #         if weaviate_objects:
#     #             batch_ingest_data(
#     #                 client=weaviate_client,
#     #                 collection_name=papers_collection.name,
#     #                 weaviate_data=weaviate_objects
#     #             )
#     #         else:
#     #              logging.warning("No objects prepared for Weaviate ingestion.")
#
#     #     # 6. Close connection (important!)
#     #     weaviate_client.close()
#     #     logging.info("Weaviate client connection closed.")
#     # else:
#     #     logging.error("Could not connect to Weaviate or failed to load/chunk documents. Ingestion aborted.")
#     pass # Keep the placeholder if __name__ == "__main__": block

