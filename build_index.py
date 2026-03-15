"""
Build FAISS Vector Index for Medical RAG Chatbot

This script:
1. Loads medical text documents from data/webmd_texts/
2. Chunks them into smaller segments
3. Creates embeddings using Google's embedding model
4. Builds and saves a FAISS index for efficient retrieval

Run this script once to create the vector store before running the chatbot.
"""

import os
import glob
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY not found in .env file")
    print("Please add your Google API key to the .env file")
    print("Get your API key from: https://makersuite.google.com/app/apikey")


def load_text_files(directory: str) -> List[Dict[str, str]]:
    """
    Load all text files from the specified directory.
    
    Args:
        directory: Path to directory containing text files
        
    Returns:
        List of dictionaries with 'text' and 'source' keys
    """
    documents = []
    text_files = glob.glob(os.path.join(directory, "*.txt"))
    
    print(f"Found {len(text_files)} text files in {directory}")
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                source = os.path.basename(file_path)
                documents.append({
                    'text': text,
                    'source': source
                })
                print(f"  Loaded: {source} ({len(text)} characters)")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in tokens (approximated by words)
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
        # Avoid infinite loop for last chunk
        if end == len(words):
            break
    
    return chunks


def create_chunks_with_metadata(documents: List[Dict[str, str]], 
                                 chunk_size: int = 500, 
                                 overlap: int = 100) -> Tuple[List[str], List[Dict]]:
    """
    Create chunks from documents with metadata.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        Tuple of (texts, metadatas) for FAISS
    """
    texts = []
    metadatas = []
    
    for doc in documents:
        chunks = chunk_text(doc['text'], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                'source': doc['source'],
                'chunk_id': i,
                'total_chunks': len(chunks)
            })
    
    return texts, metadatas

def resolve_embedding_model(save_path: str) -> str:
    """Resolve an embedding model that supports embedContent."""
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        return env_model

    model_path = os.path.join(save_path, "embedding_model.txt")
    if os.path.exists(model_path):
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                stored_model = f.read().strip()
                if stored_model:
                    return stored_model
        except Exception as e:
            print(f"Warning: Unable to read saved embedding model: {e}")

    try:
        from google import genai

        client = genai.Client()
        for model in client.models.list():
            name = getattr(model, "name", "")
            if not name:
                continue

            methods = None
            for attr in ("supported_actions", "supported_methods", "methods"):
                value = getattr(model, attr, None)
                if value:
                    methods = value
                    break

            if methods is None:
                continue

            if isinstance(methods, str):
                methods_list = [methods]
            else:
                try:
                    methods_list = list(methods)
                except TypeError:
                    methods_list = []

            methods_lower = [str(m).lower() for m in methods_list]
            if any("embed" in m for m in methods_lower) and "embedding" in name:
                os.makedirs(save_path, exist_ok=True)
                with open(model_path, "w", encoding="utf-8") as f:
                    f.write(name)
                return name
    except Exception as e:
        print(f"Warning: Unable to list embedding models: {e}")

    return "models/text-embedding-004"


def build_faiss_index(texts: List[str], 
                      metadatas: List[Dict], 
                      save_path: str) -> None:
    """
    Build and save FAISS index using LangChain and Google embeddings.
    
    Args:
        texts: List of text chunks
        metadatas: List of metadata dictionaries
        save_path: Path to save the FAISS index
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS
        
        model_name = resolve_embedding_model(save_path)
        print(f"\nInitializing Google Generative AI Embeddings ({model_name})...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="retrieval_document"
        )
        
        print(f"Creating FAISS index with {len(texts)} chunks...")
        db = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Save the index
        os.makedirs(save_path, exist_ok=True)
        db.save_local(save_path)
        print(f"FAISS index saved to: {save_path}")
        
        # Test retrieval
        print("\nTesting retrieval...")
        results = db.similarity_search("chest pain", k=2)
        print(f"Sample query 'chest pain' returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result.page_content[:100]}...")
        
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("Please install: pip install langchain-google-genai langchain-community faiss-cpu")
        raise
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        raise


def main():
    """Main function to build the FAISS index."""
    # Configuration
    DATA_DIR = "data/webmd_texts"
    SAVE_PATH = "vector_store/faiss_index"
    CHUNK_SIZE = 500  # words per chunk
    OVERLAP = 100     # words overlap
    
    print("=" * 60)
    print("Building FAISS Vector Index for Medical RAG Chatbot")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\nStep 1: Loading documents...")
    documents = load_text_files(DATA_DIR)
    
    if not documents:
        print("No documents found! Please add text files to data/webmd_texts/")
        return
    
    print(f"Total documents loaded: {len(documents)}")
    
    # Step 2: Create chunks
    print("\nStep 2: Creating text chunks...")
    texts, metadatas = create_chunks_with_metadata(documents, CHUNK_SIZE, OVERLAP)
    print(f"Total chunks created: {len(texts)}")
    
    # Step 3: Build FAISS index
    print("\nStep 3: Building FAISS index...")
    build_faiss_index(texts, metadatas, SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("Index building completed successfully!")
    print("=" * 60)
    print(f"\nYou can now run the chatbot with: streamlit run app.py")


if __name__ == "__main__":
    main()
