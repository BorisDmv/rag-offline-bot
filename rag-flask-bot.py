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
# Restrict answers to only use knowledge from data folder if enabled
RESTRICT_TO_DATA_KB = os.getenv("RESTRICT_TO_DATA_KB", "false").lower() == "true"
# --------------------------------------------

app = Flask(__name__)
# Initialize Redis connection
redis_connection = Redis(host='redis', port=6379, db=0)

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://redis:6379",  # Use Docker Compose service name
    app=app
)

CORS(
    app,
    resources={r"/*": {"origins": "https://localhost:8080"}},
    allow_headers="*",
    methods=["GET", "POST", "OPTIONS"]
)

def get_embedder():
    if 'embedder' not in g:
        print("Loading embedding model...")
        g.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Only move to CUDA if available
        import torch
        if torch.cuda.is_available():
            print("Moving embedder to CUDA...")
            g.embedder = g.embedder.to("cuda")
        print(f"Embedder loaded on device: {g.embedder.device}")
    return g.embedder

def get_faiss_index():
    if 'index' not in g:
        print("Loading data and building FAISS index...")
        try:
            print(f"CSV_PATH: {CSV_PATH}")
            df = pd.read_csv(CSV_PATH)
            print(f"CSV columns: {df.columns.tolist()}")
            documents = df["content"].tolist()
            print(f"Loaded {len(documents)} documents from CSV.")
            doc_embeddings = get_embedder().encode(documents, convert_to_numpy=True)
            print(f"Embeddings shape: {doc_embeddings.shape}")
            g.index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            g.index.add(doc_embeddings)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {CSV_PATH}")
            return None
        except KeyError:
            print(f"Error: 'content' column not found in CSV file. Columns are: {df.columns.tolist()}")
            return None
        except Exception as e:
            print(f"Error loading data or building index: {e}")
            import traceback
            traceback.print_exc()
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
def rag_respond(user_input, user_id="default", top_k=3):
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

        # Retrieve previous chat history (last 10 messages)
        chat_key = f"chat_history:{user_id}"
        history = redis_connection.lrange(chat_key, 0, -1)
        # Decode bytes to strings
        history = [msg.decode('utf-8') for msg in history]
        # Limit history to a safe character count (e.g., 1000 chars)
        max_history_chars = 1000
        history_text = ""
        if history:
            # Add messages from the end (most recent) until limit is reached
            selected = []
            total = 0
            cutoff = 0
            for i, msg in enumerate(reversed(history)):
                if total + len(msg) + 1 > max_history_chars:
                    cutoff = len(history) - i
                    break
                selected.append(msg)
                total += len(msg) + 1
            # Reverse to restore order
            selected.reverse()
            if cutoff > 0:
                # Summarize older messages in a human-readable way
                summary = f"Earlier in this conversation, you and the assistant exchanged {cutoff} messages."
                selected = [summary] + selected
            history_text = "\n".join(selected)

        # Use strict data-folder-only prompt if RESTRICT_TO_DATA_KB is true
        if RESTRICT_TO_DATA_KB:
            kb_instruction = (
                "You are a helpful assistant. You must answer ONLY using the information provided in the following documents from the data folder (see 'Knowledge Base' below). "
                "If the answer cannot be found within these documents, respond with a polite decline, such as 'I'm sorry, but I cannot answer that question based on the information I have.' "
                "Do NOT use any outside knowledge, general world knowledge, or make assumptions beyond what is in these documents."
            )
            if history_text:
                prompt = f"""{kb_instruction}\n\nPrevious conversation:\n{history_text}\n\nKnowledge Base (from data folder):\n{context}\n\nUser: {user_input}\nAnswer:"""
            else:
                prompt = f"""{kb_instruction}\n\nKnowledge Base (from data folder):\n{context}\n\nUser: {user_input}\nAnswer:"""
        else:
            # Use the original prompt from line 139
            if history_text:
                prompt = f"""You are a helpful assistant. If your answer contains code, wrap the code in triple backticks (```). If your answer does not contain code, respond normally without any backticks.\n\nPrevious conversation:\n{history_text}\n\nUser: {user_input}\nAnswer:"""
            else:
                prompt = f"""You are a helpful assistant. If your answer contains code, wrap the code in triple backticks (```). If your answer does not contain code, respond normally without any backticks.\n\nKnowledge Base:\n{context}\n\nUser: {user_input}\nAnswer:"""

        print("Calling LLM...")
        response = llm(prompt, max_tokens=1024, stop=["User:"])
        print("Response received!")
        bot_response = response["choices"][0]["text"].strip()

        # Store conversation in Redis (as a list per user)
        chat_key = f"chat_history:{user_id}"
        redis_connection.rpush(chat_key, f"user: {user_input}")
        redis_connection.rpush(chat_key, f"bot: {bot_response}")
        # Trim the list to keep only the last 10 messages
        redis_connection.ltrim(chat_key, -10, -1)

        return bot_response
    except Exception as e:
        return f"An error occurred during the RAG process: {e}"

@app.route('/chat', methods=['POST'])
# Use rate limit from environment variable (default to '1 per minute' if not set)
@limiter.limit(os.getenv("RATE_LIMIT", "1 per minute"))
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({"error": "Missing 'user_input' in request body"}), 400
    user_input = data['user_input']
    user_id = data.get('user_id', 'default')
    response = rag_respond(user_input, user_id=user_id)
    return jsonify({"response": response})


@app.route('/reset_user_session', methods=['POST'])
def reset_user_session():
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({"error": "Missing 'user_id' in request body"}), 400
    user_id = data['user_id']
    chat_key = f"chat_history:{user_id}"
    redis_connection.delete(chat_key)
    return jsonify({"status": "success", "message": f"Session for user '{user_id}' has been reset."})
# Endpoint to get chat history for a user
@app.route('/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id', 'default')
    chat_key = f"chat_history:{user_id}"
    history = redis_connection.lrange(chat_key, 0, -1)
    # Decode bytes to strings
    history = [msg.decode('utf-8') for msg in history]
    return jsonify({"history": history})

@app.route('/', methods=['GET'])
def index():
    return "Offline RAG Chatbot API is running. Send POST requests to /chat with {'user_input': 'your question'}."


## For local testing if you want to run the app with SSL, uncomment the following lines and provide your cert and key files:
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=9090, ssl_context=('cert.pem', 'key.pem'))

## For local testing without SSL, use the following line:
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=9090)
