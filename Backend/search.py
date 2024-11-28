import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import PyPDF2

# Ensure required resources for NLTK are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Set the folder containing the documents
folder_path = "data_summary"

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
    # Lowercase and remove non-alphanumeric characters
    text = re.sub(r"[^\w\s]", "", text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words and apply stemming
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
documents = load_and_preprocess_documents(folder_path)

# Print the preprocessed documents for verification
for doc_name, content in documents.items():
    print(f"Document: {doc_name}")
    print(f"Content: {content[:10]}...")  # Print first 10 tokens for brevity


from collections import defaultdict

def build_inverted_index(documents):
    """Build an inverted index from preprocessed documents."""
    inverted_index = defaultdict(dict)
    
    for doc_name, terms in documents.items():
        for term in terms:
            # Increment the term's frequency in the document
            if doc_name in inverted_index[term]:
                inverted_index[term][doc_name] += 1
            else:
                inverted_index[term][doc_name] = 1
    
    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index(documents)

# Print a sample of the inverted index for verification
print("Sample Inverted Index:")
for term, postings in list(inverted_index.items())[:5]:
    print(f"Term: {term}")
    print(f"Postings: {postings}")

import math

def compute_tf_idf(inverted_index, total_documents):
    """Compute TF-IDF scores for terms in the inverted index."""
    tf_idf = defaultdict(dict)
    
    for term, postings in inverted_index.items():
        # Calculate IDF for the term
        document_frequency = len(postings)
        idf = math.log(total_documents / document_frequency)
        
        for doc_name, tf in postings.items():
            # Compute TF-IDF score
            tf_idf[term][doc_name] = tf * idf
    
    return tf_idf

# Calculate total number of documents
total_documents = len(documents)

# Compute TF-IDF scores
tf_idf_scores = compute_tf_idf(inverted_index, total_documents)

# Print a sample of the TF-IDF scores for verification
print("Sample TF-IDF Scores:")
for term, scores in list(tf_idf_scores.items())[:5]:
    print(f"Term: {term}")
    print(f"Scores: {scores}")


from collections import Counter
import numpy as np

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
        # Calculate IDF for the query term
        document_frequency = len(tf_idf_scores.get(term, {}))
        idf = math.log(total_documents / document_frequency) if document_frequency > 0 else 0
        # Calculate TF-IDF for the query term
        query_vector[term] = query_tf[term] * idf
    
    return query_vector

def rank_documents(query_vector, tf_idf_scores):
    """Rank documents based on cosine similarity."""
    document_scores = defaultdict(float)  # Store scores for documents
    document_magnitudes = defaultdict(float)  # Store document magnitudes
    
    # Compute document scores based on query vector
    for term, query_weight in query_vector.items():
        if term in tf_idf_scores:
            for doc, doc_weight in tf_idf_scores[term].items():
                # Update score based on term weight in query and document
                document_scores[doc] += query_weight * doc_weight
    
    # Compute document magnitudes
    for term, postings in tf_idf_scores.items():
        for doc, doc_weight in postings.items():
            document_magnitudes[doc] += doc_weight**2
    
    # Normalize scores by magnitudes
    for doc in document_scores:
        document_scores[doc] /= np.sqrt(document_magnitudes[doc]) if document_magnitudes[doc] > 0 else 1
    
    # Sort documents by score in descending order
    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


def search(query, tf_idf_scores, total_documents, top_k=5):
    """Search for the top-k documents matching the query."""
    # Preprocess the query
    query_terms = preprocess_query(query)
    # Compute the query's TF-IDF vector
    query_vector = compute_query_tf_idf(query_terms, tf_idf_scores, total_documents)
    # Rank the documents
    ranked_docs = rank_documents(query_vector, tf_idf_scores)
    return ranked_docs[:top_k]

# Example query
query = "Code of Civil Procedure, 1908 (CPC)"
top_k_results = search(query, tf_idf_scores, total_documents)

# Print the results
print(f"Top-{len(top_k_results)} Results for Query: '{query}'")
for rank, (doc, score) in enumerate(top_k_results, start=1):
    print(f"{rank}. Document: {doc}, Score: {score:.4f}")
