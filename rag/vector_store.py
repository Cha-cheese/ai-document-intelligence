import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = None
        self.texts = []

    def add(self, embeddings, texts):
        self.vectors = np.array(embeddings).astype("float32")
        self.texts = texts

    def search(self, query_embedding, top_k=5):
        if self.vectors is None:
            return []

        scores = np.dot(self.vectors, query_embedding)

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "content": self.texts[i],
                "score": float(scores[i]),
                "source_id": i
            }
            for i in top_idx
        ]