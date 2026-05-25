import numpy as np

def get_embedding(text: str):

    vec = np.zeros(384)

    for i, c in enumerate(text[:300]):
        vec[i % 384] += ord(c)

    norm = np.linalg.norm(vec) + 1e-8
    return (vec / norm).astype("float32")