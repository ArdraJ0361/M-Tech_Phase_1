"""Sanity-check prepared data: eyeball samples + run the real loader/masking path."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer  # noqa: E402

from src.data.dataset_loader import load_dataset_for_task  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

for cfg_path, d in [("configs/pilot_squad_qa.yaml", "squad_v2"),
                    ("configs/pilot_xlsum_summarization.yaml", "xlsum_en")]:
    print("=" * 70, "\n", d)
    rows = [json.loads(l) for l in Path(f"data/processed/{d}/test.jsonl").open(encoding="utf-8")]
    r = next(x for x in rows if x.get("is_answerable", True))
    print(r["text"][-350:].replace(tok.eos_token, "<EOS>"))
    print("reference:", r["reference"])
    if d == "squad_v2":
        u = next(x for x in rows if not x["is_answerable"])
        print("unanswerable ref:", u["reference"], "| prompt ends:", repr(u["prompt"][-14:]))

    cfg = load_config(cfg_path)
    cfg["dataset"].update(train_samples=50, val_samples=25, test_samples=25)
    cfg["training"]["output_dir"] = "experiments/prepare_data_small"
    ds = load_dataset_for_task(cfg, tok)
    ex = ds["train"][0]
    n_train_tok = sum(1 for t in ex["labels"] if t != -100)
    last = ex["input_ids"][-1]
    print("splits:", {k: len(v) for k, v in ds.items()})
    print("trained tokens:", n_train_tok, "| last token is EOS:", last == tok.eos_token_id,
          "| EOS count:", ex["input_ids"].count(tok.eos_token_id))
    bad = [i for i, e in enumerate(ds["train"]) if e["input_ids"][-1] != tok.eos_token_id]
    print("train rows missing trailing EOS (truncated):", len(bad))
