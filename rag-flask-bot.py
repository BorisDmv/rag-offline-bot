import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

# ----------- Configurable Paths -------------
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH")
CSV_PATH = os.getenv("CSV_PATH")
# --------------------------------------------

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    documents = df["content"].tolist()
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit()
except KeyError:
    print(f"Error: 'content' column not found in CSV file.")
    exit()

# Load embedding model (MiniLM)
print("Loading embedding model...")
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(embedder)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# Embed documents
print("Generating embeddings...")
try:
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

    # Build FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
except Exception as e:
    print(f"Error generating embeddings or building FAISS index: {e}")
    exit()

# Load GGUF model with llama-cpp-python
print("Loading Mistral model from GGUF...")
try:
    llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_threads=6)
except FileNotFoundError:
    print(f"Error: GGUF model not found at {GGUF_MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading Mistral model: {e}")
    exit()

# RAG Response
def rag_respond(user_input, top_k=3):
    try:
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
    except Exception as e:
        return f"An error occurred during the RAG process: {e}"

# Flask application
app = Flask(__name__)

#CORS(app)
CORS(app, resources={r"/chat": {"origins": "http://localhost:8080"}})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({"error": "Missing 'user_input' in request body"}), 400

    user_input = data['user_input']
    response = rag_respond(user_input)
    return jsonify({"response": response})

@app.route('/', methods=['GET'])
def index():
    return "Offline RAG Chatbot API is running. Send POST requests to /chat with {'user_input': 'your question'}."

# Run server
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=9090)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090, ssl_context=('cert.pem', 'key.pem'))
