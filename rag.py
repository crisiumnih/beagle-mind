from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

DOCS_DIR = Path("docs.beagleboard.io/")

def load_files(directory):
    rst_files = list(directory.rglob("*.rst"))
    documents = []
    for rst_file in rst_files:
        with open(rst_file, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append((rst_file, content))
    return documents

def preprocess_text(text):

    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_documents(documents):
    all_chunks = []
    for doc_path, content in documents:
        preprocessed_text = preprocess_text(content)
        chunks = chunk_text(preprocessed_text)
        for chunk in chunks:
            all_chunks.append((doc_path, chunk))
    return all_chunks

def embed_chunks(chunks):
    embeddings = []
    for doc_path, chunk in chunks:
        embedding = model.encode(chunk)
        embeddings.append((doc_path, chunk, embedding))
    return embeddings

documents = load_files(DOCS_DIR)

chunks = process_documents(documents)

embeddings = embed_chunks(chunks)


for doc_path, chunk, embedding in embeddings:
    print(f"Document: {doc_path}")
    print(f"Chunk: {chunk}")
    print(f"Embedding: {embedding[:5]}...")  # Print first 5 elements of the embedding
    print("-" * 40)


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, embeddings, top_n=3):
    query_embedding = model.encode(query)
    similarities = []
    for doc_path, chunk, embedding_chunk in embeddings:
        similarity = cosine_similarity(query_embedding, embedding_chunk)
        similarities.append((doc_path, chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

query = "How to set up BeagleBoard?"

top_chunks = retrieve(query, embeddings, top_n=3)

print(f"Query: {query}\n")

for i, (doc_path, chunk, similarity) in enumerate(top_chunks, 1):
    print(f"Result {i}:")
    print(f"Document: {doc_path}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Chunk: {chunk}\n")
    print("-" * 80)
