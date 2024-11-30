from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
from huggingface_hub import InferenceClient
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load environment variables
load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")
indian_kanoon_api_key = os.getenv("INDIAN_KANOON_API_KEY")

client = InferenceClient(api_key=hf_api_key)

system_template = """As a highly qualified Legal Advisor specializing in Indian law, your role is to provide expert, accurate, and comprehensive responses to legal inquiries. Utilize your extensive knowledge of Indian jurisprudence, including statutes, case law, and legal principles to formulate your answers. When responding:
1. Conduct a thorough analysis of the query to identify key legal issues and relevant areas of law.
2. Provide clear, concise explanations of applicable laws, acts, and legal concepts, citing specific sections where appropriate.
3. Reference relevant case precedents and judicial pronouncements, including citations and brief summaries of their significance.
4. Offer insights into potential legal strategies or courses of action, considering both short-term and long-term implications.
5. Explain the practical applications of the law in the context of the query, including any potential challenges or considerations.
6. Highlight any ambiguities, areas of legal debate, or recent developments in the law that may impact the situation.
7. Where applicable, mention any relevant statutes of limitations or procedural requirements.
8. Conclude with a succinct summary of key points, critical information, and recommended next steps if appropriate.
"""

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file, filename):
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        elif file_extension in ['doc', 'docx']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file.read())
                temp_file.close()
                doc = docx.Document(temp_file.name)
                text = "\n".join([para.text for para in doc.paragraphs])
            os.unlink(temp_file.name)
        elif file_extension == 'txt':
            text = file.read().decode('utf-8')
        else:
            raise ValueError("Unsupported file format")
        return text
    except Exception as e:
        raise ValueError(f"Error while extracting text: {e}")

def fetch_indian_kanoon_info(query):
    try:
        url = "https://api.indiankanoon.org/search/"
        params = {"formInput": query, "filter": "on", "pagenum": 1}
        headers = {"Authorization": f"Token {indian_kanoon_api_key}"}
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            relevant_info = [
                f"Title: {doc.get('title', '')}\nSnippet: {doc.get('snippet', '')}\n"
                for doc in data.get('docs', [])[:3]
            ]
            return "\n".join(relevant_info)
        else:
            return "Unable to fetch information from Indian Kanoon API."
    except Exception as e:
        return f"Error fetching Indian Kanoon info: {e}"

@app.route("/chat", methods=["POST"])
def chatbot_response():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        kanoon_info = fetch_indian_kanoon_info(query)
        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Indian Kanoon Information: {kanoon_info}"}
        ]
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=500
        )
        response_content = completion.choices[0].message["content"]
        return jsonify({"query": query, "response": response_content})
    except Exception as e:
        return jsonify({"error": f"Error processing query: {e}"}), 500

@app.route("/analyze", methods=["POST"])
def analyze_document():
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file"}), 400
    filename = secure_filename(file.filename)
    try:
        document_text = extract_text_from_file(file, filename)
        kanoon_info = fetch_indian_kanoon_info(document_text[:500])
        analysis_prompt = f"""Analyze the following legal document and provide a comprehensive summary, highlighting relevant legal sections:

        Document Content:
        {document_text}

        Relevant Indian Kanoon Information:
        {kanoon_info}

        Please provide:
        1. A concise summary of the document's content and purpose
        2. Key legal points or sections, with references to specific laws or regulations
        3. Relevant laws, regulations, or case law mentioned or applicable
        4. Potential legal implications or actions to consider
        5. Any areas of ambiguity or potential legal challenges
        6. Recommendations for further legal review or action, if necessary."""

        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": analysis_prompt}
        ]
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=1000
        )
        analysis_content = completion.choices[0].message["content"]
        return jsonify({"analysis": analysis_content})
    except Exception as e:
        return jsonify({"error": f"Error analyzing document: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
