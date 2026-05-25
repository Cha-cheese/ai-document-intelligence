import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def add(self, embeddings, chunks):
        self.chunks = chunks

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            return []

        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "score": 1.0
                })

        return results