def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    i = 0

    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap

    return chunks