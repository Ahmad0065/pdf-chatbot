import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "pdf_chatbot", dimension: int = 384):
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        self.collection_name = collection_name
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def add(self, embeddings, chunks: list[dict]):
        points = []
        for embedding, chunk in zip(embeddings, chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=chunk,   # stores text, page_number, source_file, chunk_id — sab kuch
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k: int = 5) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        ).points

        output = []
        for r in results:
            chunk = r.payload.copy()
            chunk["score"] = float(r.score)
            output.append(chunk)
        return output