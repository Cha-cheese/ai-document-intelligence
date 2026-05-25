import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:

    def __init__(self):

        self.embeddings = []
        self.documents = []

    def add(self, embeddings, documents):

        for embedding, document in zip(
            embeddings,
            documents
        ):

            self.embeddings.append(
                np.array(embedding, dtype=np.float32)
            )

            self.documents.append(document)

    def search(self, query_embedding, top_k=3):

        if len(self.embeddings) == 0:
            return []

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        embeddings_matrix = np.array(
            self.embeddings,
            dtype=np.float32
        )

        similarities = cosine_similarity(
            query_embedding,
            embeddings_matrix
        )[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:

            results.append({
                "content": self.documents[idx],
                "score": float(similarities[idx]),
                "source_id": idx + 1
            })

        return results