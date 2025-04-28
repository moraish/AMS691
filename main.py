import logging
import os

# Import functions from your other modules
from indexing import load_and_parse_pdfs, chunk_documents_recursive
from embeddings import (
    connect_to_weaviate,
    create_papers_collection,
    prepare_data_for_weaviate,
    batch_ingest_data
)

# --- Configuration ---
# Adjust this path relative to where you run main.py
# Assuming main.py is in project_llm, and data is a sibling folder
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "data")
WEAVIATE_URL = "http://localhost:8080" # Default Weaviate URL
OLLAMA_API_ENDPOINT = "http://host.docker.internal:11434" # Default for Docker
COLLECTION_NAME = "ResearchPapers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Execution Logic ---
def main():
    """Orchestrates the PDF processing and Weaviate ingestion pipeline."""
    logging.info("--- Starting PDF Ingestion Pipeline ---")

    # 1. Load and Chunk Documents
    logging.info(f"Loading PDFs from: {PDF_FOLDER}")
    loaded_docs = load_and_parse_pdfs(PDF_FOLDER)
    if not loaded_docs:
        logging.error("No documents were loaded. Exiting.")
        return

    logging.info(f"Chunking {len(loaded_docs)} loaded pages...")
    chunked_docs = chunk_documents_recursive(
        loaded_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    if not chunked_docs:
        logging.error("No chunks were created from the documents. Exiting.")
        return
    logging.info(f"Created {len(chunked_docs)} chunks.")

    # 2. Connect to Weaviate
    weaviate_client = None # Initialize client to None for finally block
    try:
        logging.info(f"Connecting to Weaviate at {WEAVIATE_URL}...")
        weaviate_client = connect_to_weaviate(url=WEAVIATE_URL)

        if not weaviate_client or not weaviate_client.is_connected():
             logging.error("Failed to connect to Weaviate. Exiting.")
             return # Exit if connection failed

        logging.info("Successfully connected to Weaviate.")

        # 3. Create or Get Collection
        logging.info(f"Creating or getting collection: {COLLECTION_NAME}")
        papers_collection = create_papers_collection(
            client=weaviate_client,
            collection_name=COLLECTION_NAME,
            ollama_api_endpoint=OLLAMA_API_ENDPOINT # Pass the endpoint
            # embedding_model and generative_model will use defaults in the function
        )

        if not papers_collection:
            logging.error(f"Failed to create or get collection '{COLLECTION_NAME}'. Exiting.")
            return

        logging.info(f"Using collection: {papers_collection.name}")

        # 4. Prepare Data for Weaviate
        logging.info("Preparing chunked documents for Weaviate ingestion...")
        weaviate_objects = prepare_data_for_weaviate(chunked_docs)

        if not weaviate_objects:
            logging.warning("No objects were prepared for Weaviate ingestion. Skipping ingestion.")
            return

        logging.info(f"Prepared {len(weaviate_objects)} objects for ingestion.")

        # 5. Ingest Data
        logging.info("Starting data ingestion into Weaviate...")
        batch_ingest_data(
            client=weaviate_client,
            collection_name=papers_collection.name,
            weaviate_data=weaviate_objects,
            batch_size=BATCH_SIZE
        )
        logging.info("Data ingestion process finished.")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}", exc_info=True)

    finally:
        # 6. Close Weaviate Connection
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
            logging.info("Weaviate client connection closed.")
        logging.info("--- PDF Ingestion Pipeline Finished ---")


if __name__ == "__main__":
    # Ensure the PDF folder exists before starting
    if not os.path.isdir(PDF_FOLDER):
        logging.error(f"PDF data folder not found at expected location: {PDF_FOLDER}")
        logging.error("Please ensure the 'data' folder exists relative to the project root.")
    else:
        main()