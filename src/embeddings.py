"""Step 3a - Embedding generation.

sbert : real semantic embeddings (SentenceTransformers, runs locally).
hash  : deterministic bag-of-words hashing. NOT semantic; only for offline
        smoke tests when the model cannot be downloaded.
"""
import hashlib
import math
import re
from typing import List

from . import config


class HashEmbedder:
    dim = 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class SbertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


_cache = None


def get_embedder():
    global _cache
    if _cache is None:
        if config.EMBEDDING_BACKEND == "hash":
            _cache = HashEmbedder()
        else:
            _cache = SbertEmbedder(config.EMBEDDING_MODEL)
    return _cache
