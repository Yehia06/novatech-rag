# NovaTech Smart Home Helpdesk Copilot (RAG)

A Retrieval-Augmented Generation system that answers customer-support questions
about a fictional smart-home company (NovaTech), grounded in a 40-document
knowledge base, with source citations and refusal on out-of-scope questions.

Course: Computers and Society - RAG assignment (Dr. Sherif M. Tawfik)

## Architecture

```
data/*.txt -> load -> RecursiveCharacterTextSplitter (500 / 60 overlap)
           -> SentenceTransformers all-MiniLM-L6-v2 embeddings
           -> ChromaDB (cosine similarity, persistent)
query -> embed -> top-k=4 retrieval -> grounded prompt -> LLM -> cited answer
```

| Step (assignment) | File |
|---|---|
| 1 Setup / config | `requirements.txt`, `.env.example`, `src/config.py` |
| 2 Ingestion + chunking | `src/ingest.py` |
| 3 Embedding + indexing | `src/embeddings.py`, `src/ingest.py` |
| 4 Retriever | `src/retriever.py` |
| 5 Prompt + generation | `src/generator.py` |
| 6 Interactive pipeline | `src/app.py` |
| 5.1 Evaluation | `eval/run_eval.py`, `eval/test_questions.json` |

## Dataset
40 plain-text documents in `data/`, 8 per team member:
1. Returns, refunds and warranty
2. Smart hub and Wi-Fi setup
3. Camera and video storage
4. Lock and thermostat error codes
5. Account, cloud plans and privacy

## Run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
python -m src.ingest        # build the vector index
python -m src.app --show-context
python -m eval.run_eval     # evaluation
```

Without an API key, retrieval still works and generation is skipped.
`EMBEDDING_BACKEND=hash` is a non-semantic keyword baseline for offline tests only.

## Example queries
- What does thermostat Error E04 mean?
- Error W105 appears when I set up my camera. What should I do?
- I forgot my 2FA phone, how do I get back in?
- Who won the 2022 World Cup?  (expected: refusal)

## Evaluation metrics
Retrieval Hit@k and MRR (relevance), keyword answer correctness, refusal rate on
out-of-scope questions (faithfulness proxy), and latency. Report your own numbers
from `python -m eval.run_eval`.

## Known limitations
Small corpus, keyword-based correctness check, no re-ranking, no conversation memory.
Natural extensions: hybrid BM25 + vector search, cross-encoder re-ranking, and
agent tools (Agentic RAG) for the follow-up assignment.
