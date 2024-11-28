import os
import re
import math
import zipfile
from io import BytesIO
from collections import defaultdict, Counter
import numpy as np
import PyPDF2
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure required resources for NLTK are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize global variables
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
documents = {}
inverted_index = defaultdict(dict)
tf_idf_scores = defaultdict(dict)
total_documents = 0
data_folder = "data_summary"  # Folder with .txt and .pdf files

# Utility Functions
def extract_text_from_file(file_path):
    """Extract text from text or PDF file."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return ""

def preprocess_text(text):
    """Preprocess the text (remove punctuation, tokenize, and stem)."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = word_tokenize(text)
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def build_inverted_index(documents):
    """Build an inverted index from the documents."""
    index = defaultdict(dict)
    for doc_name, terms in documents.items():
        for term in terms:
            if doc_name in index[term]:
                index[term][doc_name] += 1
            else:
                index[term][doc_name] = 1
    return index

def compute_tf_idf(inverted_index, total_docs):
    """Compute the TF-IDF scores for the documents."""
    tf_idf = defaultdict(dict)
    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log(total_docs / df)
        for doc, tf in postings.items():
            tf_idf[term][doc] = tf * idf
    return tf_idf

def preprocess_query(query):
    """Preprocess the query text."""
    query = re.sub(r"[^\w\s]", "", query.lower())
    tokens = word_tokenize(query)
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def compute_query_tf_idf(query_terms, tf_idf_scores, total_docs):
    """Compute the TF-IDF vector for the query."""
    query_tf = Counter(query_terms)
    query_vector = {}
    for term in query_terms:
        df = len(tf_idf_scores.get(term, {}))
        idf = math.log(total_docs / df) if df > 0 else 0
        query_vector[term] = query_tf[term] * idf
    return query_vector

def rank_documents(query_vector, tf_idf_scores):
    """Rank documents based on cosine similarity with the query."""
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
        document_scores[doc] /= np.sqrt(document_magnitudes[doc]) if document_magnitudes[doc] > 0 else 1
    
    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def save_single_file(doc, folder="downloads"):
    """Save a single file (txt/pdf) for download."""
    file_path = os.path.join(data_folder, doc)
    with open(file_path, "rb") as f:
        return f.read()

# Streamlit App
st.title("Document Search Engine")

# Step 1: Index documents from the data_summary folder
if not documents:
    # Read documents from the data_summary folder
    for root, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_content = extract_text_from_file(file_path)
            documents[file] = preprocess_text(file_content)

    # Build the inverted index and compute TF-IDF scores
    total_documents = len(documents)
    inverted_index = build_inverted_index(documents)
    tf_idf_scores = compute_tf_idf(inverted_index, total_documents)

    st.success(f"Indexed {total_documents} documents from the '{data_folder}' folder.")

# Step 2: Query Search
query = st.text_input("Enter your search query:")
if query and documents:
    query_terms = preprocess_query(query)
    query_vector = compute_query_tf_idf(query_terms, tf_idf_scores, total_documents)
    ranked_docs = rank_documents(query_vector, tf_idf_scores)
    
    if ranked_docs:
        st.write(f"Top-{min(5, len(ranked_docs))} Results:")
        for rank, (doc, score) in enumerate(ranked_docs[:5], start=1):
            st.write(f"{rank}. {doc} (Score: {score:.4f})")

            # Step 3: Download Each Top File
            file_data = save_single_file(doc)
            st.download_button(
                label=f"Download {doc}",
                data=file_data,
                file_name=doc,
                mime="application/octet-stream"
            )
    else:
        st.warning("No matching documents found.")
