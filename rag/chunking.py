import re


def _clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        window = text[start:end]

        if end < len(text):
            sentence_end = max(
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("\n")
            )

            if sentence_end > chunk_size * 0.6:
                end = start + sentence_end + 1
                window = text[start:end]

        chunks.append(window.strip())
        start = end

    return [chunk for chunk in chunks if chunk]


def _tail_context(text, overlap):
    tail = text[-overlap:].strip()
    boundary = re.search(r"\s", tail)

    if boundary and boundary.end() < len(tail):
        return tail[boundary.end():].strip()

    return tail


def chunk_text(text, chunk_size=400, overlap=50):
    text = _clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""

            chunks.extend(_split_long_text(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks = []

    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped_chunks.append(chunk)
            continue

        previous_tail = _tail_context(chunks[index - 1], overlap)
        overlapped_chunks.append(f"{previous_tail}\n\n{chunk}".strip())

    return overlapped_chunks
