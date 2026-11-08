"""Step 6 - End-to-end interactive CLI.

Run:  python -m src.app            (interactive)
      python -m src.app --show-context
"""
import argparse

from .generator import generate
from .retriever import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-context", action="store_true",
                    help="print retrieved chunks with similarity scores")
    args = ap.parse_args()

    retriever = Retriever()
    print("NovaTech Helpdesk Copilot (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        hits = retriever.retrieve(q)
        if args.show_context:
            for h in hits:
                print(f'  - {h["source"]} (sim={h["score"]})')
        print("\nBot:", generate(q, hits), "\n")


if __name__ == "__main__":
    main()
