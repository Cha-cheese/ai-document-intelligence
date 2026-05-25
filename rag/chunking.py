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


def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks