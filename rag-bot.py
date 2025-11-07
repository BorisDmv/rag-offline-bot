import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

load_dotenv() 

# ----------- Configurable Paths -------------
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH")
CSV_PATH = os.getenv("CSV_PATH")
# --------------------------------------------
documents = pd.read_csv(CSV_PATH)["content"].tolist()

# Load CSV
df = pd.read_csv(CSV_PATH)
documents = df["content"].tolist()

# Load embedding model (MiniLM)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(embedder)

# Embed documents
print("Generating embeddings...")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Load GGUF model with llama-cpp-python
print("Loading Mistral model from GGUF...")
llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_threads=6)

# RAG Response
def rag_respond(user_input, top_k=3):
    query_embedding = embedder.encode([user_input])[0].reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)

    # Compose context from top-k similar documents
    context = "\n".join([f"- {documents[i]}" for i in indices[0]])

    prompt = f"""You are a helpful assistant. Use the following information to answer the user's question as clearly and helpfully as possible.

Knowledge Base:
{context}

User: {user_input}
Answer:"""

    print("Calling LLM...") 
    response = llm(prompt, max_tokens=300, stop=["User:", "\n\n"])
    print("Response received!") 
    return response["choices"][0]["text"].strip()

# Chat loop
if __name__ == "__main__":
    print("🧠 Offline RAG Chatbot is running. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = rag_respond(user_input)
        print("Bot:", response)