from sentence_transformers import SentenceTransformer
import torch

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(texts):
    model = get_model()

    if isinstance(texts, str):
        texts = [texts]

    with torch.no_grad():
        return model.encode(
            texts,
            batch_size=1,   # 🔥 สำคัญมาก
            convert_to_numpy=True,
            show_progress_bar=False
        )