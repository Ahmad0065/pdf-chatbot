from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    chunk_id = 0
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page["page_number"],
                "source_file": page["source_file"],   # naya field carry forward
            })
            chunk_id += 1
    return chunks