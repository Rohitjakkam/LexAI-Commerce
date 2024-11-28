from openai import OpenAI
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import streamlit as st
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
def search_faiss(query, model, index, chunks, top_k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Function to query the LLaMA model with context
from openai import OpenAI

from huggingface_hub import InferenceClient

def query_llama_with_context(user_query, top_chunks):
    # Combine the top chunks into a single string to serve as the context
    context = " ".join(top_chunks)
    
    # Instantiate the InferenceClient with the provided API key
    client = InferenceClient(api_key="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")
    
    # Define the messages for the chat completion, incorporating the role and context
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
    
    # Generate the completion
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=messages,
        max_tokens=500
    )
    
    # Return the generated assistant message
    return completion.choices[0].message["content"]



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
            top_chunks = search_faiss(query, SentenceTransformer('all-MiniLM-L6-v2'), index, combined_chunks, top_k=4)
        
            # Query the external LLaMA model
            with st.spinner("Generating case analysis..."):
                analysis_result = query_llama_with_context(query, top_chunks)
        
            # Display the analysis result
            st.write("**Case Analysis by Associate Lawyer**")
            st.write(analysis_result)


# Run the app
if __name__ == "__main__":
    run_app()



