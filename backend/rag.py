from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

class SimpleRAG:
    def __init__(self):
        self.index = None
        self.texts = []

    def build(self, chunks):
        self.texts = chunks
        vectors = model.encode(chunks)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors))

    def search(self, query, k=3):
        q = model.encode([query])
        D, I = self.index.search(np.array(q), k)

        return [self.texts[i] for i in I[0] if i < len(self.texts)]