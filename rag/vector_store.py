import numpy as np


class VectorStore:

    def __init__(self):

        self.texts = []
        self.embeddings = []

    def add(self, embeddings, texts):

        self.embeddings.extend(embeddings)
        self.texts.extend(texts)

    def cosine_similarity(self, a, b):

        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    def search(self, query_embedding, top_k=5):

        if not self.embeddings:
            return []

        similarities = []

        for emb in self.embeddings:

            sim = self.cosine_similarity(
                query_embedding,
                emb
            )

            similarities.append(sim)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({
                "content": self.texts[idx],
                "score": float(similarities[idx])
            })

        return results