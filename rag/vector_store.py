import faiss
import numpy as np


class VectorStore:

    def __init__(self):
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_chunks = []

    def add(self, embeddings, chunks):

        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):

        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        seen = set()

        for distance, idx in zip(distances[0], indices[0]):

            idx = int(idx)
            distance = float(distance)

            if idx >= len(self.text_chunks):
                continue

            content = self.text_chunks[idx]

            if content in seen:
                continue

            seen.add(content)

            similarity = round(float(distance) * 100, 2)

            results.append({
                "source_id": idx,
                "score": f"{similarity}%",
                "content": content
            })

        return results