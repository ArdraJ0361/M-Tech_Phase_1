"""
scripts/test_markdown_docs.py
------------------------------
Try the fine-tuned model on Markdown documents (headings, tables, lists), short and long.
Documents are invented test data, not real filings. Loads the model directly, no server needed.

Run from the repo root:
    python scripts/test_markdown_docs.py --config configs/pilot_fin10k_qa.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prompts import build_qa_prompt
from src.serving.server import answer_question
from src.training.model_loader import load_peft_checkpoint, load_tokenizer
from src.utils.config_loader import load_config

SHORT_DOC = """# Northwind Retail - Fiscal 2024 Summary

## Results

| Metric | Fiscal 2024 | Fiscal 2023 |
| --- | --- | --- |
| Net sales | $8.6 billion | $8.0 billion |
| Operating income | $910 million | $1.02 billion |
| Stores | 1,285 | 1,240 |

Operating income fell because of higher freight costs.

## Risks

- Fuel price changes
- Reliance on one supplier for 40% of parts
- Competition from online retailers
"""

FILLER = """## Regional review {n}

Region {n} continued to serve customers through stores and online channels. Management reviewed staffing, inventory and logistics during the year and expects similar conditions next year.

| Item | Status |
| --- | --- |
| Staffing | Stable |
| Inventory | Stable |
"""

FACT = "## Debt\n\nThe company repaid $275 million of long-term debt in fiscal 2024.\n"


def long_doc(position: str, sections: int = 30) -> str:
    """A long document with the debt fact at its start, middle or end."""
    parts = [FILLER.format(n=n) for n in range(1, sections + 1)]
    index = {"start": 0, "middle": sections // 2, "end": sections}[position]
    parts.insert(index, FACT)
    return "# Northwind Retail - Annual Report\n\n" + "\n".join(parts)


DEBT_Q = "How much long-term debt was repaid?"
# (name, document, question, accepted answers: any one found in the reply counts as right)
CASES = [
    ("table: net sales", SHORT_DOC, "What were net sales in fiscal 2024?", ["8.6 billion"]),
    ("table: prior year", SHORT_DOC, "What was operating income in fiscal 2023?", ["1.02 billion"]),
    ("table: store count", SHORT_DOC, "How many stores were there in fiscal 2024?", ["1,285"]),
    ("text: reason", SHORT_DOC, "Why did operating income fall?", ["freight"]),
    ("list: supplier", SHORT_DOC, "What share of parts comes from one supplier?", ["40%"]),
    ("list: all risks", SHORT_DOC, "What risks are listed?", ["fuel"]),
    ("not in doc: CEO", SHORT_DOC, "Who is the chief executive officer?", ["unanswerable"]),
    ("not in doc: dividend", SHORT_DOC, "What was the dividend per share?", ["unanswerable"]),
    ("long doc: fact at start", long_doc("start"), DEBT_Q, ["275 million"]),
    ("long doc: fact in middle", long_doc("middle"), DEBT_Q, ["275 million"]),
    ("long doc: fact at end", long_doc("end"), DEBT_Q, ["275 million"]),
    # The caller's full prompt pasted into `context`: the server then wraps it a second time.
    ("full prompt sent as context", build_qa_prompt(SHORT_DOC, "What were net sales in fiscal 2024?"),
     "What were net sales in fiscal 2024?", ["8.6 billion"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_fin10k_qa.yaml")
    parser.add_argument("--checkpoint", help="Adapter folder (default: <output_dir>/adapter)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg["training"]["output_dir"]) / "adapter")
    tokenizer = load_tokenizer(cfg)
    model = load_peft_checkpoint(cfg, checkpoint)

    right = 0
    for name, document, question, accepted in CASES:
        answer, truncated = answer_question(model, tokenizer, cfg, document, question)
        ok = any(a.lower() in answer.lower() for a in accepted)
        right += ok
        tokens = len(tokenizer(document, add_special_tokens=False)["input_ids"])
        print(f"[{'RIGHT' if ok else 'WRONG'}] {name} ({tokens} tokens, truncated={truncated})")
        print(f"    Q: {question}\n    A: {answer}")
    print(f"\n{right}/{len(CASES)} right")


if __name__ == "__main__":
    main()
