import numpy as np
def add(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        if len(embeddings.shape) == 3:
            embeddings = embeddings[:, 0, :]

        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        for chunk in chunks:
            self.documents.append(chunk)

def search(self, query_embedding, top_k=5):

        if self.index is None:
            return []

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        if len(query_embedding.shape) == 3:
            query_embedding = query_embedding[:, 0, :]

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx >= len(self.documents):
                continue

            results.append({
                "content": self.documents[idx],
                "score": float(score),
                "source_id": idx
            })

        return results