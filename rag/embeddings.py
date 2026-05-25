import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(texts):

    if isinstance(texts, str):
        texts = [texts]

    res = openai.Embedding.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [r["embedding"] for r in res["data"]]