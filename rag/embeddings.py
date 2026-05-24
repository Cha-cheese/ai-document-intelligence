from fastembed import TextEmbedding

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def get_embedding(texts):

    embeddings = list(
        embedding_model.embed(texts)
    )

    return [
        embedding.tolist()
        for embedding in embeddings
    ]