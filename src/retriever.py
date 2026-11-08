"""Step 4 - Retriever: embed the query, run top-k cosine similarity search."""
from typing import Dict, List

import chromadb

from . import config
from .embeddings import get_embedder


class Retriever:
    def __init__(self, top_k: int = config.TOP_K):
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        self.col = client.get_collection(config.COLLECTION)
        self.embedder = get_embedder()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        k = top_k or self.top_k
        res = self.col.query(query_embeddings=self.embedder.embed([query]),
                             n_results=k)
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                    res["distances"][0]):
            hits.append({"text": text, "source": meta["source"],
                         "title": meta["title"],
                         "score": round(1 - dist, 4)})   # cosine similarity
        return hits
