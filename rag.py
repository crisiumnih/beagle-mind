from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import re

print("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

qwen_path = "./models"
DOCS_DIR = Path("docs.beagleboard.io/")

tokenizer = AutoTokenizer.from_pretrained(qwen_path)
qwen = AutoModelForCausalLM.from_pretrained(qwen_path)

def load_files(directory):
    rst_files = list(directory.rglob("*.rst"))
    print(f"Found {len(rst_files)} .rst files.")
    documents = []
    for rst_file in rst_files:
        with open(rst_file, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append((rst_file, content))
    return documents

def preprocess_text(text):
    print("Preprocessing text...")
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()
    return text

def chunk_text(text, chunk_size=256, overlap=25):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def process_documents(documents):
    print("Processing documents into chunks...")
    all_chunks = []
    for doc_path, content in documents:
        preprocessed_text = preprocess_text(content)
        chunks = chunk_text(preprocessed_text)
        for chunk in chunks:
            all_chunks.append((doc_path, chunk))
    print(f"Total chunks generated: {len(all_chunks)}")
    return all_chunks

def embed_chunks(chunks, batch_size=8):
    print("Embedding chunks...")
    embeddings = []
    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            texts = [chunk for _, chunk in batch]
            batch_embeddings = model.encode(
                texts,
                convert_to_numpy=True 
            )

            embeddings.extend([
                (doc_path, chunk, embedding)
                for (doc_path, chunk), embedding in zip(batch, batch_embeddings)
            ])

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            continue
    return embeddings

print("Loading documents...")
documents = load_files(DOCS_DIR)
print(f"Loaded {len(documents)} documents.")

print("Processing documents into chunks...")
chunks = process_documents(documents)
print(f"Generated {len(chunks)} chunks.")

print("Embedding chunks...")
embeddings = embed_chunks(chunks)
print("Embeddings generated.")

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

def generate_response(query, context, max_length=1024, min_context_length=100):

    prompt = f"""Based on the following context, provide a clear and concise answer to the query.
If the context doesn't contain relevant information, say so.

Query: {query}

Context: {context}

Answer:"""
            

    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)

    current_length = inputs.input_ids.shape[1]
    max_new_tokens = max_length - current_length

    if max_new_tokens <= 0:
        max_new_tokens = 512


    outputs = qwen.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=max_new_tokens,
        attention_mask=inputs.attention_mask,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2

    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Response:")[-1].strip()
    return response

def retrieve_qwen(query, embeddings, top_n=3):
    top_chunks = retrieve(query, embeddings, top_n=top_n)
    context = ""
    for doc_path, chunk, similarity in top_chunks:
        context += f"\nFrom {doc_path}:\n{chunk}\n"

    print(f"Context length: {len(context)} characters")

    response = generate_response(query, context)

    
    return response


query = "What is BeaglePlay?"
response = retrieve_qwen(query, embeddings, top_n=3)


print(f"Query: {query}\n")
print(f"Response: {response}\n")

