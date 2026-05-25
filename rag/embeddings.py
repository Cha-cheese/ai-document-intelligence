from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=384)

def get_embedding(texts):
    # texts = list[str] or str
    if isinstance(texts, str):
        texts = [texts]

    return vectorizer.fit_transform(texts).toarray().tolist()