import os
import requests

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = (
    "https://api-inference.huggingface.co/"
    "pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def get_embedding(texts):

    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": texts}
    )

    return response.json()