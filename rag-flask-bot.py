import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis

load_dotenv()

# ----------- Configurable Paths -------------
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH")
CSV_PATH = os.getenv("CSV_PATH")
# --------------------------------------------

app = Flask(__name__)
# Initialize Redis connection
redis_connection = Redis(host='localhost', port=6379, db=0)

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",  # Make sure Redis is running
    app=app
)

CORS(app, resources={r"/chat": {"origins": "*"}}) # Changed to '*' for easier testing

def get_embedder():
    if 'embedder' not in g:
        print("Loading embedding model...")
        g.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if g.embedder.device.type != "cuda":
            print("Moving embedder to CUDA...")
            g.embedder = g.embedder.to("cuda")
        print(f"Embedder loaded on device: {g.embedder.device}")
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
            g.llm = Llama(model_path=GGUF_MODEL_PATH, n_gpu_layers=-1, n_batch=512)
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

        prompt = f"""You are a helpful assistant. **Answer the user's question ONLY using the information provided in the following knowledge base.** If the answer cannot be found within the knowledge base, respond with a polite decline, such as "I'm sorry, but I cannot answer that question based on the information I have." or "According to the provided information, I cannot answer that question."

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
@limiter.limit("1 per minute")
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


## For local testing if you want to run the app with SSL, uncomment the following lines and provide your cert and key files:
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=9090, ssl_context=('cert.pem', 'key.pem'))

## For local testing without SSL, use the following line:
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=9090)
