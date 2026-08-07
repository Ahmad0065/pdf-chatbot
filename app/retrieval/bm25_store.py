from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

class BM25Store:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results