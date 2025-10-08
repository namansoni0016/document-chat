import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

## Turning data into embeddings
class HFEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()
    
    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()
    
    def name(self):
        return f"sentence-transformers/{self.model_name}"
    
    def to_dict(self):
        return {"model_name": self.model_name}

# Hugging Face embedding function
hf_ef = HFEmbeddingFunction()

# Initialize ChromaDB client with persistent storage
chromadb_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chromadb_client.get_or_create_collection(
    name="document_qa_collection", embedding_function=hf_ef
)

client = Groq(api_key=groq_key)

# Loading all the documents
def load_documents_from_directory(directory_path):
    print("--- Loading documents from directory ---")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Loading documents from directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents.")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("--- Splitting document into chunks ---")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({ "id": f"{doc['id']}_chunk_{i}", "text": chunk })

print(f"Split documents into {len(chunked_documents)} chunks")

# Generate embeddings function
def get_hf_embeddings(text):
    print("--- Generating embeddings ---")
    embedding = hf_ef([text])[0]
    return embedding

# Embedding for chunks
for doc in chunked_documents:
    print("--- Generating embeddings --- ")
    doc["embedding"] = get_hf_embeddings(doc["text"])

print(doc["embedding"])

# Inserting in db
for doc in chunked_documents:
    print("--- Inserting into chrom₹aDB ---")
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]])

# Query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    # Extract the relavant chunks
    relevant_chunks = [doc for sublist in results['documents'] for doc in sublist]
    print("--- Returning relevant chunks ---")
    return relevant_chunks

# Generate answer using Groq
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    answer = response.choices[0].message
    return answer

# Example usage
question = "Tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)
