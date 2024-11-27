import requests

url = "http://127.0.0.1:5000/query"
headers = {"Content-Type": "application/json"}
data = {"query": "What is commercial court act 2015??"}

response = requests.post(url, json=data, headers=headers)
# breakpoint()
print(response.json())
