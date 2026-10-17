"""
Run this script in Colab as:
    !python /content/fix_source_files.py
It rewrites every src/ file with correct zero-indentation Python.
"""
import pathlib

ROOT = pathlib.Path("/content/Mtech")

def w(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    # Strip exactly 4 leading spaces from every line (the notebook embedding indent)
    lines = []
    for line in text.split("\n"):
        if line.startswith("    "):
            lines.append(line[4:])
        else:
            lines.append(line)
    p.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"  wrote {rel}")

# ── __init__ files ─────────────────────────────────────────────────────────────
for f in ["src/__init__.py","src/data/__init__.py","src/training/__init__.py",
          "src/evaluation/__init__.py","src/utils/__init__.py","src/extraction/__init__.py"]:
    (ROOT / f).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / f).write_text("", encoding="utf-8")

# ── src/utils/logging_setup.py ─────────────────────────────────────────────────
w("src/utils/logging_setup.py", """\
import logging, sys

def setup_logging(level="INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers = [handler]
    for lib in ["urllib3","filelock","git","PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
""")

# ── src/utils/config_loader.py ─────────────────────────────────────────────────
w("src/utils/config_loader.py", """\
import yaml
from pathlib import Path

def load_config(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["task", "model_name_or_path", "dataset", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    return cfg
""")

# ── src/utils/experiment_tracker.py ───────────────────────────────────────────
w("src/utils/experiment_tracker.py", """\
import csv, logging
from pathlib import Path

logger = logging.getLogger(__name__)
_COLS = ["experiment","model","dataset","task","method","train_samples",
         "val_samples","test_samples","exact_match","f1","rouge1","rouge2",
         "rougeL","accuracy","macro_f1","train_time_s","peak_gpu_mb",
         "avg_latency_ms","trainable_pct","checkpoint"]

class ExperimentTracker:
    def __init__(self, path="experiments/results.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row):
        existing = self._cols()
        new_keys = [k for k in row if k not in existing]
        all_cols = self._sort(existing + new_keys)
        exists = self.path.exists()
        if not exists or new_keys:
            self._rewrite_header(all_cols)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            if not exists:
                wr.writeheader()
            wr.writerow(row)

    def load(self):
        import pandas as pd
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)

    def print_summary(self):
        df = self.load()
        if df.empty:
            print("No experiments logged yet.")
            return
        cols = [c for c in _COLS if c in df.columns]
        print(df[cols].to_string(index=False))

    def _cols(self):
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            try:
                return next(r)
            except StopIteration:
                return []

    def _sort(self, cols):
        pref = [c for c in _COLS if c in cols]
        ext = [c for c in cols if c not in _COLS]
        return pref + ext

    def _rewrite_header(self, cols):
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        lines[0] = ",".join(cols)
        self.path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
""")

# ── src/data/prompts.py ────────────────────────────────────────────────────────
w("src/data/prompts.py", """\
import re

UNANSWERABLE = "unanswerable"
QA_SEPARATOR = "### Answer:\\n"
SUMMARY_SEPARATOR = "### Summary:\\n"
_WS = re.compile(r"[ \\t\\r\\f\\v]+")
_BL = re.compile(r"\\n{3,}")

def clean_text(text):
    text = (text or "").replace("\\u00a0", " ").replace("\\u200b", "")
    text = _WS.sub(" ", text)
    text = _BL.sub("\\n\\n", text)
    return "\\n".join(l.strip() for l in text.split("\\n")).strip()

def build_qa_prompt(context, question):
    return (f"### Context:\\n{clean_text(context)}\\n\\n"
            f"### Question:\\n{clean_text(question)}\\n\\n"
            f"{QA_SEPARATOR}")

def qa_completion(answers):
    texts = answers.get("text") or []
    return clean_text(texts[0]) if texts else UNANSWERABLE

def qa_references(answers):
    texts = [clean_text(t) for t in (answers.get("text") or [])]
    return texts or [UNANSWERABLE]

def build_summarization_prompt(article):
    return f"### Article:\\n{clean_text(article)}\\n\\n{SUMMARY_SEPARATOR}"

def truncate_to_tokens(text, tokenizer, max_tokens):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).rstrip()

def build_training_text(prompt, completion, eos_token):
    return f"{prompt}{completion}{eos_token or ''}"
""")

# ── src/data/preprocessor.py ───────────────────────────────────────────────────
w("src/data/preprocessor.py", """\
import logging

logger = logging.getLogger(__name__)
_IGNORE_INDEX = -100
_COMPLETION_SEPARATORS = [
    "### Answer:\\n",
    "### Summary:\\n",
    "### Category:\\n",
    "### Extracted fields (key: value):\\n",
]

def tokenize_qa(examples, tokenizer, cfg):
    return _tokenize_prompt_completion(examples, tokenizer,
        cfg["dataset"]["max_source_length"] + cfg["dataset"]["max_target_length"])

def tokenize_seq2seq(examples, tokenizer, cfg):
    return _tokenize_prompt_completion(examples, tokenizer,
        cfg["dataset"]["max_source_length"] + cfg["dataset"]["max_target_length"])

def tokenize_classification(examples, tokenizer, cfg):
    return _tokenize_prompt_completion(examples, tokenizer,
        cfg["dataset"]["max_source_length"] + cfg["dataset"].get("max_target_length", 16))

def _tokenize_prompt_completion(examples, tokenizer, max_length):
    all_ids, all_masks, all_labels, skipped = [], [], [], 0
    prompts = examples["prompt"] if isinstance(examples["prompt"], list) else [examples["prompt"]]
    for prompt in prompts:
        full = tokenizer(prompt, max_length=max_length, truncation=True,
                         padding=False, return_tensors=None)
        input_ids = full["input_ids"]
        attn = full["attention_mask"]
        pp = _extract_prompt_part(prompt)
        prompt_len = len(tokenizer(pp, max_length=max_length, truncation=True,
                                   padding=False, return_tensors=None)["input_ids"])
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = _IGNORE_INDEX
        comp = [t for t in labels if t != _IGNORE_INDEX]
        if not comp:
            logger.warning("SKIPPING example — no completion tokens after masking.")
            skipped += 1
            continue
        all_ids.append(input_ids)
        all_masks.append(attn)
        all_labels.append(labels)
    if skipped:
        logger.warning("%d/%d examples skipped.", skipped, len(prompts))
    return {"input_ids": all_ids, "attention_mask": all_masks, "labels": all_labels}

def _extract_prompt_part(full_prompt):
    for sep in _COMPLETION_SEPARATORS:
        if sep in full_prompt:
            return full_prompt[:full_prompt.rfind(sep) + len(sep)]
    return full_prompt

def get_data_collator(tokenizer):
    from transformers import DataCollatorForSeq2Seq
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
        label_pad_token_id=_IGNORE_INDEX, pad_to_multiple_of=8)
""")

# ── src/data/dataset_loader.py ─────────────────────────────────────────────────
w("src/data/dataset_loader.py", """\
from __future__ import annotations
import json, logging
from pathlib import Path
from .prompts import (UNANSWERABLE, build_qa_prompt, build_summarization_prompt,
                      build_training_text, clean_text, qa_completion, truncate_to_tokens)

logger = logging.getLogger(__name__)
_DATASET_ALIASES = {"squad_v2": "rajpurkar/squad_v2"}
_XLSUM_REPO = "csebuetnlp/xlsum"
_XLSUM_PARQUET = "hf://datasets/csebuetnlp/xlsum@refs%2Fconvert%2Fparquet"
_PROMPT_OVERHEAD_TOKENS = 24
_SQUAD_SPLIT_SOURCE = {"train": "train", "validation": "validation", "test": "validation"}
_HF_SPLIT_SOURCE = {
    "squad_v2": _SQUAD_SPLIT_SOURCE,
    "rajpurkar/squad_v2": _SQUAD_SPLIT_SOURCE,
    "_default": {"train": "train", "validation": "validation", "test": "test"},
}

def load_dataset_for_task(cfg, tokenizer=None):
    task = cfg["task"]
    ds_cfg = cfg["dataset"]
    if task == "qa":
        return _load_qa(ds_cfg, tokenizer, cfg)
    if task == "summarization":
        return _load_summarization(ds_cfg, tokenizer, cfg)
    raise ValueError(f"Unknown task: {task}")

def _load_raw(name, subset=None):
    from datasets import load_dataset
    name = _DATASET_ALIASES.get(name, name)
    if name == _XLSUM_REPO:
        lang = subset or "english"
        base = f"{_XLSUM_PARQUET}/{lang}"
        return load_dataset("parquet",
            data_files={s: f"{base}/{s}/*.parquet" for s in ("train","validation","test")})
    return load_dataset(name, subset) if subset else load_dataset(name)

def _load_qa(ds_cfg, tokenizer, cfg):
    from .preprocessor import tokenize_qa
    name = ds_cfg.get("name", "rajpurkar/squad_v2")
    raw = _load_raw(name, ds_cfg.get("subset"))
    raw = _subsample(raw, ds_cfg, cfg)
    if tokenizer is None:
        return raw
    max_src = ds_cfg["max_source_length"]
    def _fmt(ex):
        q_len = len(tokenizer(ex["question"], add_special_tokens=False)["input_ids"])
        budget = max(max_src - q_len - _PROMPT_OVERHEAD_TOKENS, 32)
        ctx = truncate_to_tokens(clean_text(ex["context"]), tokenizer, budget)
        ans = qa_completion(ex["answers"])
        keep = ans == UNANSWERABLE or ans.lower() in ctx.lower()
        return {"prompt": build_training_text(build_qa_prompt(ctx, ex["question"]),
                                              ans, tokenizer.eos_token),
                "answer": ans, "keep": keep}
    fmt = raw.map(_fmt, remove_columns=raw["train"].column_names)
    n = len(fmt["train"])
    fmt["train"] = fmt["train"].filter(lambda e: e["keep"])
    if len(fmt["train"]) < n:
        logger.info("Dropped %d train examples (answer truncated).", n - len(fmt["train"]))
    fmt = fmt.remove_columns("keep")
    return fmt.map(lambda ex: tokenize_qa(ex, tokenizer, cfg),
                   batched=True, remove_columns=["prompt", "answer"])

def _load_summarization(ds_cfg, tokenizer, cfg):
    from .preprocessor import tokenize_seq2seq
    name = ds_cfg.get("name", "csebuetnlp/xlsum")
    raw = _load_raw(name, ds_cfg.get("subset", "english"))
    raw = _subsample(raw, ds_cfg, cfg)
    if tokenizer is None:
        return raw
    max_src = ds_cfg["max_source_length"]
    max_tgt = ds_cfg["max_target_length"]
    def _fmt(ex):
        art = truncate_to_tokens(clean_text(ex.get("text", "")), tokenizer,
                                 max_src - _PROMPT_OVERHEAD_TOKENS)
        summ = truncate_to_tokens(clean_text(ex.get("summary", "")), tokenizer, max_tgt - 1)
        return {"prompt": build_training_text(build_summarization_prompt(art),
                                              summ, tokenizer.eos_token),
                "summary": summ}
    fmt = raw.map(_fmt, remove_columns=raw["train"].column_names)
    return fmt.map(lambda ex: tokenize_seq2seq(ex, tokenizer, cfg),
                   batched=True, remove_columns=["prompt", "summary"])

def _subsample(raw, ds_cfg, cfg):
    import numpy as np
    from datasets import DatasetDict
    seed = cfg.get("training", {}).get("seed", 42)
    rng = np.random.default_rng(seed)
    dataset_name = ds_cfg.get("name", "")
    srcs = _HF_SPLIT_SOURCE.get(dataset_name, _HF_SPLIT_SOURCE["_default"])
    n_tr = ds_cfg.get("train_samples")
    n_va = ds_cfg.get("val_samples")
    n_te = ds_cfg.get("test_samples")
    result = {}
    val_src = srcs["validation"]
    test_src = srcs["test"]
    if val_src == test_src and val_src in raw:
        shared = raw[val_src]
        total = (n_va or 0) + (n_te or 0)
        if total > 0 and total < len(shared):
            idx = rng.choice(len(shared), size=total, replace=False).tolist()
            idx.sort()
            vi = idx[:n_va]
            ti = idx[n_va:]
        else:
            all_i = list(range(len(shared)))
            vi = all_i[:n_va] if n_va else all_i
            ti = all_i[n_va:n_va + (n_te or 0)] if n_te else []
        if vi:
            result["validation"] = shared.select(vi)
        if ti:
            result["test"] = shared.select(ti)
            _save_test_indices(ti, cfg, seed, test_src)
    else:
        for role, hf in [("validation", val_src), ("test", test_src)]:
            if hf not in raw:
                continue
            ds = raw[hf]
            n = n_va if role == "validation" else n_te
            if n and n < len(ds):
                idx = rng.choice(len(ds), size=n, replace=False).tolist()
                idx.sort()
                ds = ds.select(idx)
                if role == "test":
                    _save_test_indices(idx, cfg, seed, hf)
            result[role] = ds
    tr_src = srcs.get("train", "train")
    if tr_src in raw:
        ds = raw[tr_src]
        if n_tr and n_tr < len(ds):
            idx = rng.choice(len(ds), size=n_tr, replace=False).tolist()
            idx.sort()
            ds = ds.select(idx)
        result["train"] = ds
    return DatasetDict(result)

def _save_test_indices(indices, cfg, seed, hf_split="unknown"):
    out = Path(cfg.get("training", {}).get("output_dir", "experiments/default"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_indices.json").write_text(
        json.dumps({"seed": seed, "dataset": cfg["dataset"].get("name"),
                    "hf_source_split": hf_split, "num_test_examples": len(indices),
                    "indices": indices}, indent=2), encoding="utf-8")
""")

# ── src/evaluation/metrics.py ──────────────────────────────────────────────────
w("src/evaluation/metrics.py", """\
from __future__ import annotations
import re, string, logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

def normalize_answer(s):
    def rm_art(t): return re.sub(r"\\b(a|an|the)\\b", " ", t)
    def ws(t): return " ".join(t.split())
    def rm_punc(t): return "".join(c for c in t if c not in set(string.punctuation))
    return ws(rm_art(rm_punc(s.lower())))

def get_tokens(s): return normalize_answer(s).split()

def compute_exact_match(p, g):
    return float(normalize_answer(p) == normalize_answer(g))

def compute_f1(p, g):
    pt = get_tokens(p); gt = get_tokens(g)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0: return 0.0
    pr = n / len(pt); rc = n / len(gt)
    return 2 * pr * rc / (pr + rc)

def compute_qa_metrics(predictions, references, report_unanswerable_breakdown=True):
    ems = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    f1s = [compute_f1(p, r) for p, r in zip(predictions, references)]
    res = {"exact_match": round(np.mean(ems) * 100, 2),
           "f1": round(np.mean(f1s) * 100, 2),
           "num_examples": len(predictions)}
    if report_unanswerable_breakdown:
        ans = [(p,r,e,f) for p,r,e,f in zip(predictions,references,ems,f1s)
               if normalize_answer(r) != "unanswerable"]
        una = [(p,r,e,f) for p,r,e,f in zip(predictions,references,ems,f1s)
               if normalize_answer(r) == "unanswerable"]
        if ans:
            res["answerable_em"] = round(np.mean([x[2] for x in ans]) * 100, 2)
            res["answerable_f1"] = round(np.mean([x[3] for x in ans]) * 100, 2)
            res["answerable_n"] = len(ans)
        if una:
            res["unanswerable_em"] = round(np.mean([x[2] for x in una]) * 100, 2)
            res["unanswerable_f1"] = round(np.mean([x[3] for x in una]) * 100, 2)
            res["unanswerable_n"] = len(una)
    return res

def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    sc = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(predictions, references):
        res = scorer.score(r, p)
        for k in sc: sc[k].append(res[k].fmeasure)
    return {"rouge1": round(np.mean(sc["rouge1"]) * 100, 2),
            "rouge2": round(np.mean(sc["rouge2"]) * 100, 2),
            "rougeL": round(np.mean(sc["rougeL"]) * 100, 2),
            "num_examples": len(predictions)}

def make_compute_metrics_fn(task, tokenizer):
    def _decode(ids):
        ids = np.where(ids == -100, tokenizer.pad_token_id, ids)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        if preds.ndim == 3: preds = np.argmax(preds, axis=-1)
        dp = [p.strip() for p in _decode(preds)]
        dl = [l.strip() for l in _decode(labels)]
        if task == "qa": return compute_qa_metrics(dp, dl)
        if task == "summarization": return compute_rouge(dp, dl)
        return {}
    return compute_metrics
""")

# ── src/evaluation/evaluator.py ────────────────────────────────────────────────
w("src/evaluation/evaluator.py", """\
from __future__ import annotations
import logging, time
import numpy as np, torch
from .metrics import compute_qa_metrics, compute_rouge

logger = logging.getLogger(__name__)
_TASK_FN = {"qa": compute_qa_metrics, "summarization": compute_rouge}

def _split(ex):
    labels = ex["labels"]
    start = next(i for i, t in enumerate(labels) if t != -100)
    return list(ex["input_ids"][:start]), [t for t in labels[start:] if t != -100]

def _postprocess(pred, task):
    pred = pred.split("\\n###")[0].strip()
    if task == "qa":
        pred = pred.split("\\n")[0].strip()
    return pred

def evaluate_model(model, tokenizer, dataset, cfg, batch_size=8):
    task = cfg["task"]
    max_new = cfg["dataset"].get("max_target_length", 64)
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    model.eval()
    pairs = [_split(dataset[i]) for i in range(len(dataset))]
    all_preds = []
    all_refs = [tokenizer.decode(r, skip_special_tokens=True).strip() for _, r in pairs]
    gen_s = 0.0
    with torch.no_grad():
        for b in range(0, len(pairs), batch_size):
            chunk = pairs[b:b + batch_size]
            prompts = [p for p, _ in chunk]
            w = max(len(p) for p in prompts)
            ids = torch.tensor([[pad] * (w - len(p)) + p for p in prompts], device=device)
            msk = torch.tensor([[0] * (w - len(p)) + [1] * len(p) for p in prompts], device=device)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=msk,
                                 max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
            if device.type == "cuda": torch.cuda.synchronize()
            gen_s += time.perf_counter() - t0
            dec = tokenizer.batch_decode(out[:, w:], skip_special_tokens=True)
            all_preds.extend(_postprocess(p, task) for p in dec)
    fn = _TASK_FN.get(task)
    if fn is None: return {}
    metrics = fn(all_preds, all_refs)
    metrics["avg_latency_ms"] = round(gen_s * 1000 / len(all_preds), 1)
    metrics["throughput_samples_per_s"] = round(len(all_preds) / gen_s, 2)
    metrics["_predictions"] = all_preds
    metrics["_references"] = all_refs
    return metrics
""")

# ── src/training/model_loader.py ───────────────────────────────────────────────
w("src/training/model_loader.py", """\
from __future__ import annotations
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

def load_tokenizer(cfg):
    path = cfg.get("tokenizer_name_or_path") or cfg["model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_base_model(cfg):
    path = cfg["model_name_or_path"]
    bits = cfg.get("load_in_bits")
    bnb = _bnb(bits) if bits in (4, 8) else None
    kw = {"trust_remote_code": True, "device_map": "auto"}
    if bnb:
        kw["quantization_config"] = bnb
    elif torch.cuda.is_available():
        kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        kw["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    model.config.use_cache = False
    return model

def _bnb(bits):
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return BitsAndBytesConfig(load_in_8bit=True)

def attach_lora(model, cfg):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    pc = cfg.get("peft", {})
    bits = cfg.get("load_in_bits")
    if bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lc = LoraConfig(r=pc.get("r", 8), lora_alpha=pc.get("lora_alpha", 16),
                    lora_dropout=pc.get("lora_dropout", 0.05),
                    target_modules=pc.get("target_modules"),
                    bias=pc.get("bias", "none"),
                    task_type=pc.get("task_type", "CAUSAL_LM"))
    model = get_peft_model(model, lc)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    pct = 100 * tr / tot if tot else 0
    print(f"\\n✅ LoRA attached: {tr:,} trainable / {tot:,} total ({pct:.3f}%)")
    cfg["_trainable_params"] = tr
    cfg["_total_params"] = tot
    cfg["_trainable_pct"] = round(pct, 4)
    return model

def load_peft_checkpoint(base_cfg, checkpoint_path):
    from peft import PeftModel
    base = load_base_model(base_cfg)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model
""")

# ── src/training/trainer.py ────────────────────────────────────────────────────
w("src/training/trainer.py", """\
from __future__ import annotations
import logging, time
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
from .model_loader import load_tokenizer, load_base_model, attach_lora
from ..data.dataset_loader import load_dataset_for_task
from ..data.preprocessor import get_data_collator
from ..evaluation.metrics import make_compute_metrics_fn
from ..utils.experiment_tracker import ExperimentTracker
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)

def _peak():
    return torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

def run_training(config_path):
    cfg = load_config(config_path)
    out = Path(cfg["training"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg)
    ds = load_dataset_for_task(cfg, tok)
    model = attach_lora(load_base_model(cfg), cfg)
    tracker = ExperimentTracker(cfg["results_csv"])
    mem0 = _peak()
    tc = cfg["training"]
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=tc.get("num_train_epochs", 3),
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 4),
        learning_rate=float(tc.get("learning_rate", 2e-4)),
        weight_decay=float(tc.get("weight_decay", 0.01)),
        warmup_ratio=float(tc.get("warmup_ratio", 0.03)),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        fp16=tc.get("fp16", False),
        bf16=tc.get("bf16", False),
        logging_dir=str(out / "logs"),
        logging_steps=tc.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=tc.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=tc.get("save_steps", 200),
        save_total_limit=tc.get("save_total_limit", 2),
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "eval_loss"),
        seed=tc.get("seed", 42),
        report_to="tensorboard",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation"),
        tokenizer=tok,
        data_collator=get_data_collator(tok),
        compute_metrics=make_compute_metrics_fn(cfg["task"], tok),
    )
    t0 = time.time()
    trainer.train()
    train_t = time.time() - t0
    mem = _peak() - mem0
    test_m = {}
    if "test" in ds:
        test_m = trainer.evaluate(eval_dataset=ds["test"])
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    row = {"experiment": cfg.get("experiment_name", ""),
           "model": cfg["model_name_or_path"],
           "model_variant": cfg.get("model_variant", "base"),
           "dataset": cfg["dataset"].get("name", ""),
           "task": cfg["task"],
           "method": "LoRA" if not cfg.get("load_in_bits") else "QLoRA",
           "seed": tc.get("seed", 42),
           "train_samples": len(ds.get("train", [])),
           "val_samples": len(ds.get("validation", [])),
           "test_samples": len(ds.get("test", [])),
           "trainable_pct": cfg.get("_trainable_pct"),
           "train_time_s": round(train_t, 1),
           "peak_gpu_mb": round(mem, 1),
           "checkpoint": str(out / "adapter"),
           **{k: round(v, 4) if isinstance(v, float) else v for k, v in test_m.items()}}
    tracker.log(row)
    return test_m
""")

# ── configs ────────────────────────────────────────────────────────────────────
squad_yaml = """\
experiment_name: "squad_qa_lora_pilot"
task: "qa"
model_name_or_path: "Qwen/Qwen1.5-0.5B"
tokenizer_name_or_path: null
model_variant: "base"
use_flash_attention: false
load_in_bits: null
peft:
  method: "lora"
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
  bias: "none"
  task_type: "CAUSAL_LM"
dataset:
  name: "rajpurkar/squad_v2"
  version: "default"
  subset: null
  text_column: "context"
  question_column: "question"
  answer_column: "answers"
  max_source_length: 512
  max_target_length: 64
  train_samples: 5000
  val_samples: 500
  test_samples: 500
training:
  output_dir: "experiments/squad_qa"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: true
  logging_steps: 50
  eval_steps: 200
  save_steps: 200
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "eval_f1"
  seed: 42
evaluation:
  metrics:
    - "exact_match"
    - "f1"
  record_efficiency: true
results_csv: "experiments/results.csv"
"""

xlsum_yaml = """\
experiment_name: "xlsum_summarization_lora_pilot"
task: "summarization"
model_name_or_path: "Qwen/Qwen1.5-0.5B"
tokenizer_name_or_path: null
model_variant: "base"
use_flash_attention: false
load_in_bits: null
peft:
  method: "lora"
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
  bias: "none"
  task_type: "CAUSAL_LM"
dataset:
  name: "csebuetnlp/xlsum"
  version: "default"
  subset: "english"
  text_column: "text"
  summary_column: "summary"
  max_source_length: 512
  max_target_length: 84
  train_samples: 5000
  val_samples: 500
  test_samples: 500
training:
  output_dir: "experiments/xlsum_summarization"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: true
  logging_steps: 50
  eval_steps: 200
  save_steps: 200
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "eval_rougeL"
  seed: 42
evaluation:
  metrics:
    - "rouge1"
    - "rouge2"
    - "rougeL"
  record_efficiency: true
results_csv: "experiments/results.csv"
"""

llm_yaml = """\
experiment_name: "squad_qa_llm_baseline"
task: "qa"
model_name_or_path: "Qwen/Qwen1.5-7B"
tokenizer_name_or_path: null
model_variant: "base"
use_flash_attention: false
load_in_bits: 4
peft:
  method: "lora"
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
  bias: "none"
  task_type: "CAUSAL_LM"
dataset:
  name: "rajpurkar/squad_v2"
  version: "default"
  subset: null
  text_column: "context"
  question_column: "question"
  answer_column: "answers"
  max_source_length: 512
  max_target_length: 64
  train_samples: 5000
  val_samples: 500
  test_samples: 500
training:
  output_dir: "experiments/llm_baseline"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: false
  logging_steps: 50
  eval_steps: 200
  save_steps: 200
  save_total_limit: 2
  load_best_model_at_end: false
  metric_for_best_model: "eval_f1"
  seed: 42
evaluation:
  metrics:
    - "exact_match"
    - "f1"
  record_efficiency: true
results_csv: "experiments/results.csv"
"""

(ROOT / "configs").mkdir(parents=True, exist_ok=True)
(ROOT / "configs/pilot_squad_qa.yaml").write_text(squad_yaml, encoding="utf-8")
(ROOT / "configs/pilot_xlsum_summarization.yaml").write_text(xlsum_yaml, encoding="utf-8")
(ROOT / "configs/llm_baseline_qwen7b.yaml").write_text(llm_yaml, encoding="utf-8")
print("  wrote configs/pilot_squad_qa.yaml")
print("  wrote configs/pilot_xlsum_summarization.yaml")
print("  wrote configs/llm_baseline_qwen7b.yaml")

# ── verify all files parse correctly ──────────────────────────────────────────
import ast, sys
errors = []
for pyf in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in str(pyf):
        continue
    try:
        ast.parse(pyf.read_text(encoding="utf-8"))
    except SyntaxError as e:
        errors.append(f"  SYNTAX ERROR in {pyf.relative_to(ROOT)}: {e}")

print()
if errors:
    print("❌ Syntax errors found:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("✅ All source files written and verified (no syntax errors)")
    print("   Now run the sys.path cell (Step 3) and continue.")
