from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory containing the BeagleBoard docs
DOCS_DIR = Path("docs.beagleboard.io")

# List to store document content and their file paths
documents = []
doc_paths = []

def load_documents(directory):
    """Recursively load text content from .rst and other text files."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".rst", ".md", ".txt")):  # Add more extensions if needed
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:  # Only add non-empty files
                            documents.append(content)
                            doc_paths.append(str(file_path.relative_to(DOCS_DIR)))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Load all documents from the directory
load_documents(DOCS_DIR)
print(f"Loaded {len(documents)} documents from {DOCS_DIR}")

# Generate embeddings for the documents
doc_embeddings = model.encode(documents, show_progress_bar=True)

# Create a FAISS index for similarity search
index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance metric
index.add(np.array(doc_embeddings))  # Add document embeddings to the index

def retrieve(query, k=1):
    """
    Retrieve the top k most relevant documents for a given query.
    
    Args:
        query (str): The user's query.
        k (int): Number of documents to return (default: 1).
    
    Returns:
        list: List of tuples (document text, file path) for retrieved documents.
    """
    query_embedding = model.encode([query])  # Encode the query
    distances, indices = index.search(np.array(query_embedding), k)  # Search the index
    return [(documents[i], doc_paths[i]) for i in indices[0]]

# Optional: Expose embeddings and documents for debugging
def get_embeddings():
    return doc_embeddings

def get_documents():
    return documents

def get_doc_paths():
    return doc_paths

if __name__ == "__main__":
    # Test the retrieval
    query = "What is the BeagleBone AI?"
    retrieved_docs = retrieve(query, k=2)
    for doc, path in retrieved_docs:
        print(f"\nRetrieved from {path}:")
        print(doc[:200] + "..." if len(doc) > 200 else doc)  # Print first 200 chars
