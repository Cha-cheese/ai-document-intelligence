import numpy as np

class VectorStore:

    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, embeddings, documents):
        self.embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
        self.documents = documents

    def search(self, query_embedding, top_k=3):

        if not self.embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)

        scores = []

        for i, emb in enumerate(self.embeddings):
            score = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb))
            scores.append((i, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "content": self.documents[i],
                "score": s,
                "source_id": i
            }
            for i, s in scores[:top_k]
        ]