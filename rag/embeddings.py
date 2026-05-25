from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# โหลดครั้งเดียว
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embedding(texts):

    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()