import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Function to load and chunk text data
def load_and_chunk_data(file_path, chunk_size=500):
    """
    Loads a large text file and splits it into chunks of specified size.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Number of characters per chunk.

    Returns:
        list: List of text chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks

# Function to create embeddings
def create_embeddings(chunks):
    """
    Creates embeddings for a list of text chunks.

    Args:
        chunks (list): List of text chunks.

    Returns:
        numpy.ndarray: Array of embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

# Function to build a FAISS index
def build_faiss_index(embeddings):
    """
    Builds a FAISS index for fast similarity search.

    Args:
        embeddings (numpy.ndarray): Array of embeddings.

    Returns:
        faiss.Index: FAISS index.
    """
    dim = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings)
    return index

# Function to search the FAISS index
def search_faiss(query, model, index, chunks, top_k=5):
    """
    Searches the FAISS index for the most relevant chunks.

    Args:
        query (str): User's query.
        model (SentenceTransformer): Pretrained sentence transformer model.
        index (faiss.Index): FAISS index.
        chunks (list): Original text chunks.
        top_k (int): Number of top results to return.

    Returns:
        list: Top K relevant chunks.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function to query a Hugging Face model API
def query_model(prompt, api_url, api_key):
    """
    Sends a query to a Hugging Face model API.

    Args:
        prompt (str): Input prompt for the model.
        api_url (str): URL of the Hugging Face API.
        api_key (str): Hugging Face API key.

    Returns:
        dict: Response from the API.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    return response.json()

# Main script
if __name__ == "__main__":
    try:
        # File path and chunking
        file_path = "data.txt"
        chunk_size = 500
        data_chunks = load_and_chunk_data(file_path, chunk_size)
        print(f"Total Chunks: {len(data_chunks)}")
        
        # Embedding creation
        embeddings = create_embeddings(data_chunks)
        
        # Build FAISS index
        index = build_faiss_index(embeddings)
        
        # Query and search
        query = "Explain what is commercial courts acts?"
        model = SentenceTransformer('all-MiniLM-L6-v2')
        relevant_chunks = search_faiss(query, model, index, data_chunks)
        print("Relevant Chunks:", relevant_chunks)
        
        # Generate answer using Hugging Face API
        retrieved_context = " ".join(relevant_chunks)
        user_query = "Explain what is commercial courts acts"
        input_prompt = f"Context: {retrieved_context}\n\nQuestion: {user_query}\nAnswer:"
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
        API_KEY = "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"  # Replace with your API key
        output = query_model(input_prompt, API_URL, API_KEY)
        print("Generated Answer:", output)
    except Exception as e:
        print(f"Error: {e}")
