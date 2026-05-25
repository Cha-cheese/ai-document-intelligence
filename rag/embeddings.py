import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/"
    "pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def get_embedding(texts):

    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": texts,
            "options": {
                "wait_for_model": True
            }
        },
        timeout=120
    )

    response.raise_for_status()

    embeddings = response.json()

    return embeddings