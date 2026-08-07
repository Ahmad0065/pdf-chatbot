def reciprocal_rank_fusion(faiss_results: list[dict], bm25_results: list[dict], k: int = 60, top_n: int = 5) -> list[dict]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score = sum of 1/(k + rank) across both lists — doesn't need score normalization,
    which is the tricky part of merging FAISS (cosine) and BM25 (unbounded) scores directly.
    """
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(faiss_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_lookup[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for cid, score in ranked:
        chunk = chunk_lookup[cid].copy()
        chunk["fused_score"] = score
        results.append(chunk)
    return results