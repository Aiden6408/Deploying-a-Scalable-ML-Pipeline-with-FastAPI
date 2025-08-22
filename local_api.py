import requests

BASE_URL = "http://127.0.0.1:8000"

# --- GET ---
r = requests.get(f"{BASE_URL}/", timeout=10)
print("Status Code:", r.status_code)
try:
    print("Result:", r.json().get("message"))
except Exception:
    print("Result (raw):", r.text)

# --- POST ---
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post(f"{BASE_URL}/data/", json=data, timeout=10)
print("Status Code:", r.status_code)
try:
    payload = r.json()
    # expected: {"result": "<=50K"} or {">50K"}
    print("Result:", payload["result"] if isinstance(payload, dict) and "result" in payload else payload)
except Exception:
    print("Result (raw):", r.text)