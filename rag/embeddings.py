from sentence_transformers import SentenceTransformer
import torch

_model = None


def load():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(texts):
    model = load()

    if isinstance(texts, str):
        texts = [texts]

    with torch.no_grad():
        return model.encode(
            texts,
            batch_size=2,   # 🔥 very important for Render
            convert_to_numpy=True,
            show_progress_bar=False
        )