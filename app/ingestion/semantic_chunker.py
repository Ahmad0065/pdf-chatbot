import re
import numpy as np
from app.embeddings.embedder import Embedder

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunk_pages(pages: list[dict], embedder: Embedder,
                          similarity_threshold: float = 0.5,
                          max_chunk_chars: int = 1500) -> list[dict]:
    """
    Groups sentences into chunks based on semantic similarity.
    A new chunk starts when similarity between consecutive sentences drops
    below the threshold, or when max_chunk_chars is hit.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        sentences = split_into_sentences(page["text"])
        if not sentences:
            continue
        if len(sentences) == 1:
            chunks.append({"chunk_id": chunk_id, "text": sentences[0], "page_number": page["page_number"]})
            chunk_id += 1
            continue

        embeddings = embedder.model.encode(sentences, normalize_embeddings=True)

        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))  # cosine sim (already normalized)
            sentence_len = len(sentences[i])

            if sim < similarity_threshold or current_len + sentence_len > max_chunk_chars:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(current_chunk),
                    "page_number": page["page_number"],
                })
                chunk_id += 1
                current_chunk = [sentences[i]]
                current_len = sentence_len
            else:
                current_chunk.append(sentences[i])
                current_len += sentence_len

        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk),
                "page_number": page["page_number"],
            })
            chunk_id += 1

    return chunks