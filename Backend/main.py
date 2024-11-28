from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import glob
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# Global variables
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
combined_chunks = []

# Function to load and chunk data from multiple text files
def load_and_chunk_data(file_paths, chunk_size=500):
    chunks = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])
    return chunks

# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_paths):
    combined_text = ""
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text

# Combine both text files and PDFs into one knowledge base
def load_combined_knowledge_base(text_file_paths, pdf_file_paths, chunk_size=500):
    text_chunks = load_and_chunk_data(text_file_paths, chunk_size)
    pdf_text = extract_text_from_pdfs(pdf_file_paths)
    pdf_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
    return text_chunks + pdf_chunks

# Function to create embeddings for the text chunks
def create_embeddings(chunks):
    return model.encode(chunks)

# Function to build a FAISS index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Function to search the FAISS index
def search_faiss(query, index, chunks, top_k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Function to query the LLaMA model with context
def query_llama_with_context(user_query, top_chunks):
    context = " ".join(top_chunks)
    client = InferenceClient(api_key="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")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an associate lawyer in a Commercial Court, tasked with assisting a senior lawyer "
                "in preparing for an upcoming case. Your role is to identify key points, formulate potential defenses, "
                "provide supporting arguments, and recommend relevant case precedents. The case involves a dispute "
                "between two parties over a breach of contract in a commercial transaction."
            )
        },
        {
            "role": "user",
            "content": f"User Query: {user_query}\n\nContext from Knowledge Base: {context}"
        }
    ]
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=messages,
        max_tokens=500
    )
    
    return completion.choices[0].message["content"]

# Function to format the API response
def format_response(raw_response):
    # Parse the raw response into structured data
    response_lines = raw_response.split("\n\n")
    # formatted_response = {}
    return response_lines

# Custom initialization function
def initialize_resources():
    global index, combined_chunks
    text_file_paths = glob.glob('data_summary/*.txt')
    pdf_file_paths = glob.glob('data_summary/*.pdf')
    combined_chunks = load_combined_knowledge_base(text_file_paths, pdf_file_paths)
    embeddings = create_embeddings(combined_chunks)
    index = build_faiss_index(embeddings)

# Test endpoint
@app.route('/test', methods=['GET'])
def test_query():
    return jsonify({'message': 'Server is running!'})

# Main query endpoint
@app.route('/query', methods=['POST'])
def handle_query():
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Expected 'application/json'"}), 415
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        top_chunks = search_faiss(user_query, index, combined_chunks, top_k=4)
        raw_response = query_llama_with_context(user_query, top_chunks)
        formatted_response = format_response(raw_response)
        
        return formatted_response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_resources()
    app.run(debug=True)
