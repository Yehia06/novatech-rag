"""Steps 2-3 - Ingestion, chunking, embedding, indexing.

Run:  python -m src.ingest
"""
from pathlib import Path
from typing import Dict, List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .embeddings import get_embedder


def load_documents(data_dir: Path = config.DATA_DIR) -> List[Dict]:
    """Recursively load every .txt file with metadata."""
    docs = []
    for path in sorted(data_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                      if l.startswith("Document:")), path.stem)
        docs.append({"text": text, "source": path.name,
                     "category": path.parent.name, "title": title})
    return docs


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d["text"])):
            chunks.append({
                "id": f'{d["category"]}/{d["source"]}::{i}',
                "text": piece,
                "meta": {"source": d["source"], "category": d["category"],
                         "title": d["title"], "chunk_index": i},
            })
    return chunks


def build_index() -> int:
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION)   # rebuild from scratch
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION,
                                   metadata={"hnsw:space": "cosine"})

    embedder = get_embedder()
    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        col.add(ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] for c in batch],
                embeddings=embedder.embed([c["text"] for c in batch]))
    print(f"Indexed {col.count()} chunks in {config.DB_DIR}")
    return col.count()


if __name__ == "__main__":
    build_index()
