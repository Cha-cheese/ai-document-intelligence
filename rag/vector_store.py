import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:

    def __init__(self):

        self.texts = []
        self.embeddings = []

    def add(self, embeddings, texts):

        self.embeddings.extend(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=5):

        if len(self.embeddings) == 0:
            return []

        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({
                "content": self.texts[idx],
                "score": float(similarities[idx])
            })

        return results