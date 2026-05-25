import numpy as np

class VectorStore:

    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, embeddings, documents):

        for emb in embeddings:
            self.embeddings.append(np.array(emb, dtype=np.float32))

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=5):

        if len(self.embeddings) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)

        scores = []

        for idx, emb in enumerate(self.embeddings):

            score = np.dot(query, emb) / (
                np.linalg.norm(query) * np.linalg.norm(emb)
            )

            scores.append((idx, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []

        for idx, score in scores[:top_k]:

            results.append({
                "content": self.documents[idx],
                "score": round(score, 4),
                "source_id": idx + 1
            })

        return results