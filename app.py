import os
import faiss
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from groq import Groq  # Import Groq client

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load precomputed embeddings and FAISS index
with open("embeddings.pkl", "rb") as f:
    embeddings_data = pickle.load(f)

index = faiss.read_index("faiss_index.idx")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_documents(query, top_k=1):
    """
    Retrieves the top-k most relevant document chunks based on the query.
    """
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [embeddings_data["documents"][i] for i in indices[0] if i < len(embeddings_data["documents"])]


def format_response(query, context):
    """
    Formats a response using Groq's LLM API.
    """
    summarized_context = " ".join(context)
    prompt = f"""You are an AI assistant. Answer the question using the given context in a **clear, direct, and user-friendly** manner.  
                  
    **Context:** {summarized_context}  
    **Question:** {query}  
    
    - Keep the response **short and to the point**.  
    - Do **not** include unnecessary thoughts or analysis.  
    - Provide a **direct answer** without repeating the question or context."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: API key is missing."

    client = Groq(api_key=api_key)
    
    chat_completion = client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}]
    )

    return chat_completion.choices[0].message.content

@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint to handle user queries and return formatted responses.
    """
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing!"}), 400

    # Retrieve relevant context
    context = retrieve_documents(query)

    # Generate AI-powered response using Groq
    response = format_response(query, context)

    return jsonify({"response": response})


if __name__ == "__main__":
    print("🔥 Starting Flask server...")
    app.run(debug=True)
