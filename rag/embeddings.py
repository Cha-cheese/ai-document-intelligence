from sentence_transformers import SentenceTransformer
import torch

# ❌ DO NOT load at import time in production heavy server
_model = None


def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # small model
    return _model


def get_embedding(texts):
    model = load_model()

    if isinstance(texts, str):
        texts = [texts]

    with torch.no_grad():
        return model.encode(
            texts,
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True
        )