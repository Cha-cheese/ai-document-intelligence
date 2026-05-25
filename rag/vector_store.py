import faiss
import numpy as np

class VectorStore:

    def __init__(self):
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts = []

    def add(self, vectors, texts):

        vecs = np.array(vectors).astype("float32")

        self.index.add(vecs)
        self.texts.extend(texts)

    def search(self, query_vector, top_k=5):

        if len(self.texts) == 0:
            return []

        q = np.array([query_vector]).astype("float32")

        D, I = self.index.search(q, top_k)

        return [self.texts[i] for i in I[0] if i < len(self.texts)]