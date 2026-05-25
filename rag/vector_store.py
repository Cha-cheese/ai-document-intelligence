import numpy as np

class VectorStore:

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)

    def search(self, query_vector, top_k=5):

        if not self.vectors:
            return []

        scores = []

        for i, v in enumerate(self.vectors):
            v = np.array(v)
            score = np.dot(query_vector, v)
            scores.append((score, i))

        scores.sort(reverse=True)

        return [self.texts[i] for _, i in scores[:top_k]]