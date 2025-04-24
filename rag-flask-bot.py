import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flask import Flask, request, jsonify, g
from flask_cors import CORS

load_dotenv()

# ----------- Configurable Paths -------------
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH")
CSV_PATH = os.getenv("CSV_PATH")
# --------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}}) # Changed to '*' for easier testing

def get_embedder():
    if 'embedder' not in g:
        print("Loading embedding model...")
        g.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print(g.embedder)
    return g.embedder

def get_faiss_index():
    if 'index' not in g:
        print("Loading data and building FAISS index...")
        try:
            df = pd.read_csv(CSV_PATH)
            documents = df["content"].tolist()
            doc_embeddings = get_embedder().encode(documents, convert_to_numpy=True)
            g.index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            g.index.add(doc_embeddings)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {CSV_PATH}")
            return None
        except KeyError:
            print(f"Error: 'content' column not found in CSV file.")
            return None
        except Exception as e:
            print(f"Error loading data or building index: {e}")
            return None
    return g.index

def get_llm():
    if 'llm' not in g:
        print("Loading Mistral model from GGUF...")
        try:
            g.llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_threads=6)
        except FileNotFoundError:
            print(f"Error: GGUF model not found at {GGUF_MODEL_PATH}")
            return None
        except Exception as e:
            print(f"Error loading Mistral model: {e}")
            return None
    return g.llm

# RAG Response
def rag_respond(user_input, top_k=3):
    index = get_faiss_index()
    embedder = get_embedder()
    llm = get_llm()

    if index is None or embedder is None or llm is None:
        return "Error: Could not initialize models or data."

    try:
        query_embedding = embedder.encode([user_input])[0].reshape(1, -1)
        _, indices = index.search(query_embedding, top_k)

        # Compose context from top-k similar documents
        documents = pd.read_csv(CSV_PATH)["content"].tolist() # Reload documents to ensure consistency
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090, ssl_context=('cert.pem', 'key.pem'))