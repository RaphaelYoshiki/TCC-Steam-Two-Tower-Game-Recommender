import requests

BACKEND_URL = "http://localhost:8000/recommend"

def fetch_recommendations(profile):
    response = requests.post(BACKEND_URL, json=profile, timeout=30)
    response.raise_for_status()
    return response.json()["recommendations"]
