import requests

API_URL = "http://127.0.0.1:8000/predict"

def get_rating(review_text: str) -> int:
    payload = {"reviewText": review_text}
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()               
    data = resp.json()
    return data["rating"]

if __name__ == "__main__":
    sample = "This book was amazing—absolutely loved it!"
    rating = get_rating(sample)
    print(f"Predicted rating: {rating}")