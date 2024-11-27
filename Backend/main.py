from flask import Flask, request, jsonify
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import glob
import requests

app = Flask(__name__)

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
    # Load text files
    text_chunks = load_and_chunk_data(text_file_paths, chunk_size)
    
    # Load PDFs
    pdf_text = extract_text_from_pdfs(pdf_file_paths)
    pdf_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
    
    # Combine both
    return text_chunks + pdf_chunks

# Function to create embeddings for the text chunks
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Function to build a FAISS index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(np.array(embeddings))
    return index

# Function to search the FAISS index
def search_faiss(query, index, chunks, top_k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Function to query the LLaMA model with context
def query_llama_with_context(user_query, top_chunks):
    context = " ".join(top_chunks)  # Combine the top chunks into a single string
    prompt = f"""
    You are an associate lawyer in a Commercial Court, tasked with assisting a senior lawyer in preparing for an upcoming case. Your role is to identify key points, formulate potential defenses, provide supporting arguments, and recommend relevant case precedents. The case involves a dispute between two parties over a breach of contract in a commercial transaction.
    
    Case Details: {user_query}
    Context from Knowledge Base: {context}
    
    Instructions:
    1. **Key Points**: Provide key points from your knowledge and the given context.
    2. **Key Defense Strategies**: Suggest key defense strategies from your knowledge and the given context.
    3. **Supporting Arguments**: Outline supporting arguments from your knowledge and the given context.
    4. **Relevant Case Precedents**: Recommend relevant case precedents based on the context and your knowledge.
    5. **Recommendations for Senior Counsel**: Provide recommendations for senior counsel based on the given context and your expertise.
    """
    
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    return response.json()

# Custom initialization function
def initialize_resources():
    global index, combined_chunks
    text_file_paths = glob.glob('data_summary/*.txt')
    pdf_file_paths = glob.glob('data_summary/*.pdf')
    combined_chunks = load_combined_knowledge_base(text_file_paths, pdf_file_paths)
    embeddings = create_embeddings(combined_chunks)
    index = build_faiss_index(embeddings)

# Define the query endpoint
@app.route('/query', methods=['POST'])
def handle_query():
    # Ensure the request is in JSON format
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Expected 'application/json'"}), 415
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Retrieve relevant chunks based on the query
        top_chunks = search_faiss(user_query, index, combined_chunks, top_k=4)
        # Query the LLaMA model with the relevant context
        analysis_result = query_llama_with_context(user_query, top_chunks)
        # analysis_result = analysis_result.json()
        return jsonify({"query": user_query, "result": analysis_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_resources()  # Initialize resources before starting the server
    app.run(debug=True)
