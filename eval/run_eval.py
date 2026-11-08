"""Section 5.1 - Evaluation.

Metrics
  Retrieval Hit@k : expected source file is among the retrieved chunks (relevance)
  MRR             : reciprocal rank of the expected source
  Answer Correct  : expected keyword appears in the generated answer (needs LLM key)
  Refusal Rate    : out-of-scope questions correctly refused (faithfulness proxy)
  Latency         : mean retrieval and generation time

Run:  python -m eval.run_eval
"""
import json
import time
from pathlib import Path

from src.generator import generate, llm_available
from src.retriever import Retriever

QS = json.loads((Path(__file__).parent / "test_questions.json").read_text())


def main():
    r = Retriever()
    use_llm = llm_available()
    hit = mrr = correct = refuse_ok = n_in = n_out = 0
    t_ret = t_gen = 0.0
    rows = []
    for item in QS:
        t0 = time.time(); hits = r.retrieve(item["q"]); t_ret += time.time() - t0
        ans = ""
        if use_llm:
            t0 = time.time(); ans = generate(item["q"], hits); t_gen += time.time() - t0
        low = ans.lower().replace("\u2019", "'")
        row = {"q": item["q"], "retrieved": [h["source"] for h in hits], "answer": ans}
        if item["source"]:
            n_in += 1
            srcs = row["retrieved"]
            if item["source"] in srcs:
                hit += 1; mrr += 1 / (srcs.index(item["source"]) + 1)
            row["hit"] = item["source"] in srcs
            if use_llm:
                ok = any(k in low for k in item["keywords"]); correct += ok
                row["correct"] = ok
        else:
            n_out += 1
            if use_llm:
                ok = "don't have that information" in low; refuse_ok += ok
                row["refused"] = ok
        rows.append(row)

    print(f"Questions: {len(QS)} ({n_in} in-scope, {n_out} out-of-scope)")
    print(f"Retrieval Hit@k : {hit}/{n_in} = {hit/n_in:.2%}")
    print(f"MRR             : {mrr/n_in:.3f}")
    print(f"Mean retrieval  : {t_ret/len(QS)*1000:.1f} ms")
    if use_llm:
        print(f"Answer correct  : {correct}/{n_in} = {correct/n_in:.2%}")
        print(f"Correct refusals: {refuse_ok}/{n_out}")
        print(f"Mean generation : {t_gen/len(QS):.2f} s")
    else:
        print("(No LLM key set: generation metrics skipped.)")
    (Path(__file__).parent / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
