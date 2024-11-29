from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import glob
from huggingface_hub import InferenceClient
import os
import re
import math
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import PyPDF2

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

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Set the folder containing the documents
DATA_FOLDER = 'data_summary'

# Initialize stop words and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def extract_text_from_file(file_path):
    """Extract text from a given file."""
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stop words, and stemming."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = word_tokenize(text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

def load_and_preprocess_documents(folder_path):
    """Load documents from a folder, extract text, and preprocess it."""
    documents = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".txt", ".pdf")):
            text = extract_text_from_file(file_path)
            preprocessed_text = preprocess_text(text)
            documents[filename] = preprocessed_text
    return documents

# Load and preprocess documents
documents = load_and_preprocess_documents(DATA_FOLDER)

def build_inverted_index(documents):
    """Build an inverted index from preprocessed documents."""
    inverted_index = defaultdict(dict)
    for doc_name, terms in documents.items():
        for term in terms:
            if doc_name in inverted_index[term]:
                inverted_index[term][doc_name] += 1
            else:
                inverted_index[term][doc_name] = 1
    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index(documents)

def compute_tf_idf(inverted_index, total_documents):
    """Compute TF-IDF scores for terms in the inverted index."""
    tf_idf = defaultdict(dict)
    for term, postings in inverted_index.items():
        document_frequency = len(postings)
        idf = math.log(total_documents / document_frequency)
        for doc_name, tf in postings.items():
            tf_idf[term][doc_name] = tf * idf
    return tf_idf

# Calculate total number of documents
total_documents = len(documents)

# Compute TF-IDF scores
tf_idf_scores = compute_tf_idf(inverted_index, total_documents)

def preprocess_query(query):
    """Preprocess the query just like the documents."""
    query = re.sub(r"[^\w\s]", "", query.lower())
    tokens = word_tokenize(query)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

def compute_query_tf_idf(query_terms, tf_idf_scores, total_documents):
    """Compute the TF-IDF vector for the query."""
    query_tf = Counter(query_terms)
    query_vector = {}
    for term in query_terms:
        document_frequency = len(tf_idf_scores.get(term, {}))
        idf = math.log(total_documents / document_frequency) if document_frequency > 0 else 0
        query_vector[term] = query_tf[term] * idf
    return query_vector

def rank_documents(query_vector, tf_idf_scores):
    """Rank documents based on cosine similarity."""
    document_scores = defaultdict(float)
    document_magnitudes = defaultdict(float)
    
    for term, query_weight in query_vector.items():
        if term in tf_idf_scores:
            for doc, doc_weight in tf_idf_scores[term].items():
                document_scores[doc] += query_weight * doc_weight
    
    for term, postings in tf_idf_scores.items():
        for doc, doc_weight in postings.items():
            document_magnitudes[doc] += doc_weight**2
    
    for doc in document_scores:
        document_scores[doc] /= math.sqrt(document_magnitudes[doc]) if document_magnitudes[doc] > 0 else 1
    
    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def search(query, tf_idf_scores, total_documents, top_k=5):
    """Search for the top-k documents matching the query."""
    query_terms = preprocess_query(query)
    query_vector = compute_query_tf_idf(query_terms, tf_idf_scores, total_documents)
    ranked_docs = rank_documents(query_vector, tf_idf_scores)
    return ranked_docs[:top_k]

@app.route('/search', methods=['POST'])
def search_cases():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    top_k_results = search(query, tf_idf_scores, total_documents)

    # Check if the results are valid
    if not top_k_results:
        return jsonify({"error": "No results found"}), 404

    # Ensure scores are numbers and provide default values if not valid
    results_with_scores = []
    for doc, score in top_k_results:
        if score is None:
            score = 0.0  # Assign default score if None
        results_with_scores.append({"filename": doc, "score": round(score, 4)})

    return jsonify({
        "response": f"Top {len(top_k_results)} documents for query: '{query}'",
        "context": results_with_scores
    })


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(DATA_FOLDER, filename, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404


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
