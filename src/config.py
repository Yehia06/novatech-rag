"""Central configuration, read from environment / .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "novatech_kb"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 60))
TOP_K = int(os.getenv("TOP_K", 4))

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")      # gemini | ollama | groq | anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
