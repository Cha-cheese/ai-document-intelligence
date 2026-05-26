import faiss
import numpy as np

DIMENSION = 384

index = faiss.IndexFlatL2(DIMENSION)

documents = []


def add_documents(chunks, embeddings):

    global documents

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    index.add(embeddings)

    documents.extend(chunks)


def search(query_embedding, k=5):

    if len(documents) == 0:
        return []

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        if idx < len(documents):
            results.append(documents[idx])

    return results