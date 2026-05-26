import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self):
        self.index = None
        self.texts = []
        self.model = None   # ❗ ไม่โหลดทันที

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build(self, chunks):
        self.load_model()

        self.texts = chunks
        vectors = self.model.encode(chunks)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query, k=3):
        self.load_model()

        q = self.model.encode([query])
        D, I = self.index.search(np.array(q).astype("float32"), k)

        return [self.texts[i] for i in I[0] if i < len(self.texts)]