import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import streamlit as st
import concurrent.futures
import glob
import requests

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
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use an appropriate model
    embeddings = model.encode(chunks)
    return embeddings

# Function to build a FAISS index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(np.array(embeddings))
    return index

# Function to search the FAISS index
def search_faiss(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Function to query the external model with retrieved context
def query_model_with_prompt(case_details):
    # Associate lawyer's professional prompt to assist in case analysis
    prompt = f"""
    You are an associate lawyer in a Commercial Court, tasked with assisting a senior lawyer in preparing for an upcoming case. Your role is to identify key points, formulate potential defenses, provide supporting arguments, and recommend relevant case precedents. The case involves a dispute between two parties over a breach of contract in a commercial transaction.
    
    Case Details: {case_details}
    
    Instructions:
    1. **Key Points**: give some key points from your knowledge
    
    2. **Key Defense Strategies**: give some Key Defense Strategies from your knowledge
    
    3. **Supporting Arguments**: give some Supporting Arguments from your knowledge
    
    4. **Relevant Case Precedents**: give some Relevant Case Precedents from your knowledge
    5. **Recommendations for Senior Counsel**: give some Recommendations for Senior Counsel from your knowledge
    """

    # Send the prompt to an external API (such as HuggingFace) to query the model
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    return response.json()

# Dynamically load files from a folder
def load_files_from_directory(directory, file_type='txt'):
    file_paths = glob.glob(os.path.join(directory, f"*.{file_type}"))
    return file_paths

# Streamlit app definition
def run_app():
    st.title("Lex-AI-Commercial")

    # Dynamically load text and PDF files from the folder (adjust the paths as needed)
    text_file_paths = load_files_from_directory('data_summary', 'txt')
    pdf_file_paths = load_files_from_directory('data_summary', 'pdf')

    # Display a loading spinner while the app is processing files
    with st.spinner("Processing files..."):
        # Load combined knowledge base
        combined_chunks = load_combined_knowledge_base(text_file_paths, pdf_file_paths)
        
        # Create embeddings for the combined knowledge base
        embeddings = create_embeddings(combined_chunks)
        
        # Build the FAISS index
        index = build_faiss_index(embeddings)

    st.success("Files loaded and index built successfully!")

    # Query interface
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Searching for relevant information..."):
            top_k = 5  # Number of top results to return
            relevant_chunks = search_faiss(query, SentenceTransformer('all-MiniLM-L6-v2'), index, combined_chunks, top_k)
            
            # Display the top-k relevant chunks
            for chunk in relevant_chunks:
                st.write(chunk)

    # # Case Details Input for Associate Lawyer's Analysis
    # case_details = st.text_area("Enter the case details for analysis:")
    # if case_details:
    #     with st.spinner("Analyzing case..."):
    #         analysis_result = query_model_with_prompt(case_details)
    #         st.write("**Case Analysis by Associate Lawyer**")
    #         st.write(analysis_result)

# Run the app
if __name__ == "__main__":
    run_app()
