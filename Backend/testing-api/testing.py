import requests

BASE_URL = "http://127.0.0.1:5000"  # Update if hosted elsewhere

def test_chat():
    """
    Test the /chat endpoint with a sample query.
    """
    print("Testing /chat endpoint...")
    query = "What are the legal implications of breach of contract under Indian law?"
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"query": query})
        if response.status_code == 200:
            print("Response from /chat:", response.json())
        else:
            print(f"Error ({response.status_code}):", response.json())
    except requests.exceptions.RequestException as e:
        print("Error occurred while testing /chat endpoint:", e)

def test_analyze():
    """
    Test the /analyze endpoint with a sample file.
    """
    print("Testing /analyze endpoint...")
    file_path = "case_summarycase400.txt"  # Replace with an actual file path
    try:
        with open(file_path, "rb") as file:
            response = requests.post(f"{BASE_URL}/analyze", files={"file": file})
        if response.status_code == 200:
            print("Response from /analyze:", response.json())
        else:
            print(f"Error ({response.status_code}):", response.json())
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please provide a valid file path.")
    except requests.exceptions.RequestException as e:
        print("Error occurred while testing /analyze endpoint:", e)

if __name__ == "__main__":
    print("Starting tests...")
    test_chat()
    test_analyze()
