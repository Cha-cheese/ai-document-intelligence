import fitz


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    text = []

    max_pages = min(len(doc), 15)

    for page_num in range(max_pages):

        page = doc.load_page(page_num)

        page_text = page.get_text("text")

        if page_text:
            text.append(page_text)

    doc.close()

    return "\n".join(text)