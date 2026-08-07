from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """Takes chunk dicts, returns a numpy array of embeddings (one row per chunk)."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # important for cosine similarity search later
            show_progress_bar=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """For embedding a user's question at retrieval time."""
        return self.model.encode([query], normalize_embeddings=True)[0]