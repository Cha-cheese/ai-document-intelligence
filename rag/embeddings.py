import numpy as np


def get_embedding(text):

    vector = np.zeros(384)

    for i, char in enumerate(text[:5000]):
        vector[i % 384] += ord(char)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector.astype("float32")