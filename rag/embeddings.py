import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(texts):

    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]