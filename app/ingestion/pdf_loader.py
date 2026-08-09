import fitz  # PyMuPDF
import re
import hashlib

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(pdf_path: str, source_name: str = None, file_hash: str = None) -> list[dict]:
    if source_name is None:
        source_name = pdf_path.split("/")[-1].split("\\")[-1]

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)
        if cleaned:
            pages.append({
                "page_number": i + 1,
                "text": cleaned,
                "source_file": source_name,
                "file_hash": file_hash,
            })
    doc.close()
    return pages



def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse excessive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse repeated spaces/tabs
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # fix hyphenated line breaks
    text = text.strip()
    return text