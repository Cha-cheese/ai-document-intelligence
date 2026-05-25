import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=128):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, vectors, texts):
        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector]).astype("float32")

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i in indices[0]:
            if i < len(self.texts):
                results.append(self.texts[i])

        return results