import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from typing import List, Dict

# Setup basic logging (optional, but helpful)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions for Indexing Preparation ---

def extract_title_from_filename(filename: str) -> str:
    """
    Extracts the paper title from a filename expected to be in the format
    'N_{Title_with_underscores}.pdf'. Handles potential variations.

    Args:
        filename: The base name of the PDF file (e.g., "1_Attention_Is_All_You_Need.pdf").

    Returns:
        The extracted and cleaned paper title (e.g., "Attention Is All You Need").
    """
    # Remove '.pdf' extension (case-insensitive)
    name_without_ext, _ = os.path.splitext(filename)

    # Split at the first underscore
    parts = name_without_ext.split('_', 1)

    if len(parts) > 1:
        # Take the part after the first underscore
        title_part = parts[1]
        # Replace remaining underscores with spaces and strip leading/trailing whitespace
        title = title_part.replace('_', ' ').strip()
        # Optional: Capitalize words for better formatting (can be adjusted)
        # title = ' '.join(word.capitalize() for word in title.split())
        return title
    else:
        # Fallback if format is unexpected (e.g., no underscore)
        logging.warning(f"Could not parse standard title format from filename: {filename}. Using name as fallback.")
        # Use the whole name without extension, replacing underscores
        return name_without_ext.replace('_', ' ').strip()

def load_and_parse_pdfs(pdf_folder_path: str) -> List[Document]:
    """
    Loads PDF documents from a specified folder, extracts text using PyPDFLoader,
    and adds 'source' (filename) and 'paper_title' metadata to each page (Document).

    Args:
        pdf_folder_path: The path to the folder containing PDF files.

    Returns:
        A list of Langchain Document objects, where each Document represents a page
        from a PDF, enriched with metadata. Returns an empty list if the folder
        doesn't exist or no PDFs are found.
    """
    all_docs: List[Document] = []
    logging.info(f"Starting PDF loading from: {pdf_folder_path}")

    if not os.path.isdir(pdf_folder_path):
        logging.error(f"PDF folder not found: {pdf_folder_path}")
        return []

    loaded_count = 0
    error_count = 0
    for filename in os.listdir(pdf_folder_path):
        # Ensure processing only PDF files (case-insensitive check)
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            logging.debug(f"Processing file: {filename}") # More detailed logging
            try:
                # Initialize loader for the current PDF
                loader = PyPDFLoader(file_path)
                # Load documents (pages) from the PDF
                pages: List[Document] = loader.load()

                # Extract title from the filename
                paper_title = extract_title_from_filename(filename)

                # Add metadata to each page Document loaded from this PDF
                for page_index, page_doc in enumerate(pages):
                    # Ensure metadata dictionary exists
                    if page_doc.metadata is None:
                         page_doc.metadata = {}
                    page_doc.metadata["source"] = filename # Store original filename
                    page_doc.metadata["paper_title"] = paper_title
                    # PyPDFLoader usually adds 'page', but check just in case
                    if "page" not in page_doc.metadata:
                         page_doc.metadata["page"] = page_index # Add page index if missing

                all_docs.extend(pages)
                loaded_count += 1
                logging.debug(f"Successfully loaded {len(pages)} pages from {filename}.")

            except Exception as e:
                logging.error(f"Failed to load or parse {filename}: {e}", exc_info=True) # Log stack trace
                error_count += 1

    logging.info(f"Finished PDF loading. Processed {loaded_count} files. Encountered errors in {error_count} files.")
    logging.info(f"Total pages loaded into Document objects: {len(all_docs)}")
    return all_docs

def chunk_documents_recursive(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits a list of Langchain Documents into smaller chunks using
    RecursiveCharacterTextSplitter. Preserves existing metadata and adds
    a unique 'chunk_id' and 'start_index' to each new chunk's metadata.

    Args:
        documents: A list of Langchain Document objects (typically pages from PDFs).
        chunk_size: The target size for each chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of new Langchain Document objects representing the chunks.
        Returns an empty list if the input documents list is empty.
    """
    if not documents:
        logging.warning("No documents provided for chunking.")
        return []

    logging.info(f"Starting recursive chunking of {len(documents)} documents...")
    logging.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Adds 'start_index' to metadata
        separators=["\n\n", "\n", " ", ""], # Common separators for academic text
        is_separator_regex=False,
    )

    chunks: List[Document] = text_splitter.split_documents(documents)

    # Add unique chunk IDs based on original source and chunk index
    for i, chunk in enumerate(chunks):
        # Ensure metadata exists
        if chunk.metadata is None:
            chunk.metadata = {}

        # Create a robust chunk ID
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", "unknown_page")
        chunk.metadata["chunk_id"] = f"{source}_page_{page}_chunk_{i}"

    logging.info(f"Successfully split documents into {len(chunks)} chunks.")
    return chunks

# --- Example Usage (can be placed in a __main__ block later) ---
# if __name__ == "__main__":
#     PDF_FOLDER = "data/papers/" # Make sure this folder exists and has PDFs
#
#     # 1. Load documents and add metadata
#     loaded_docs = load_and_parse_pdfs(PDF_FOLDER)
#
#     if loaded_docs:
#         # 2. Chunk the loaded documents
#         chunked_docs = chunk_documents_recursive(loaded_docs, chunk_size=1000, chunk_overlap=150)
#
#         # Now 'chunked_docs' holds the processed chunks ready for embedding/vector store indexing.
#         # You can inspect them:
#         if chunked_docs:
#             print(f"\nExample Chunk 0 Metadata: {chunked_docs[0].metadata}")
#             print(f"Example Chunk 0 Content Start: {chunked_docs[0].page_content[:200]}...")
#             print(f"\nExample Chunk 1 Metadata: {chunked_docs[1].metadata}")
#             print(f"Example Chunk 1 Content Start: {chunked_docs[1].page_content[:200]}...")
#         else:
#              print("Chunking resulted in an empty list.")
#     else:
#         print("Loading PDFs resulted in an empty list. Check PDF folder and logs.")