import numpy as np

class VectorStore:
    def __init__(self):
        self.texts = []
        self.vectors = []

    def add(self, text, vector):
        self.texts.append(text)
        self.vectors.append(vector)

    def search(self, query_vec, top_k=5):
        if not self.vectors:
            return []

        q = np.array(query_vec)

        scores = []
        for i, v in enumerate(self.vectors):
            v = np.array(v)
            score = np.dot(q, v) / (np.linalg.norm(v) + 1e-9)
            scores.append((score, i))

        scores.sort(reverse=True, key=lambda x: x[0])

        return [self.texts[i] for _, i in scores[:top_k]]