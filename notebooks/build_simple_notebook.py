"""
Run with:  python notebooks/build_simple_notebook.py
Generates: notebooks/simple_pipeline.ipynb

A short, self-contained version of the SQuAD v2 3-way experiment (see docs/architecture.md).
Every cell is plain code you can read top to bottom -- no embedded copy of src/.
"""
import json, pathlib

CELLS = []
def md(text):   CELLS.append(("markdown", text.strip()))
def code(text): CELLS.append(("code", text.strip()))


md("""
# PEFT-SLM — Simple Pipeline (SQuAD v2 QA)

**Research question:** can a small model fine-tuned with LoRA get close to a large model, at a fraction of the cost?

| Setup | Model | Training | Question it answers |
|---|---|---|---|
| 1 | Qwen1.5-0.5B (base) | none, zero-shot | how weak is a small model untuned? |
| 2 | Qwen1.5-0.5B + LoRA | fine-tuned | **our contribution** — does LoRA help? |
| 3 | Qwen1.5-7B (base) | none, zero-shot | how close can the small tuned model get? |

**Pipeline** (one cell each): config → data split → prompt + masking → metrics → evaluate → Setup 1 → Setup 2 (train) → Setup 3 → results.

**How to run on Colab:** Runtime → Change runtime type → **T4 GPU** → Run all.
Run once with `DRY_RUN = True` (~5 min, proves everything works), then set it to `False` for the real run.
""")

code("""
# Step 0 — Setup: install what Colab lacks, pick the device
import os, sys, subprocess
if 'google.colab' in sys.modules:
    # Colab ships torchao 0.10; recent peft refuses to load with torchao < 0.16
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'peft', 'bitsandbytes', 'torchao>=0.16.0'])

os.environ['HF_HUB_DISABLE_XET'] = '1'   # Xet downloads stall on Colab; plain HTTP is fast
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (use DRY_RUN = True)')
""")

code("""
# Step 1 — Config: everything you might change lives here
import os
DRY_RUN = True     # True = tiny 5-min check. False = real run (~1.5-2 h on a T4)
RUN_LLM = True     # Setup 3: 7B zero-shot baseline, 4-bit, GPU only (~25 min)

SEED = 42                      # same seed -> all 3 models see the same test questions
SLM  = 'Qwen/Qwen1.5-0.5B'     # base variant, not instruct: any gain comes from LoRA alone
LLM  = 'Qwen/Qwen1.5-7B'       # same family, only the size differs

N_TRAIN, N_VAL, N_TEST = (64, 16, 16) if DRY_RUN else (5000, 500, 500)
EPOCHS   = 1 if DRY_RUN else 3   # set 1 for a quicker (~35 min) real run
MAX_LEN  = 384                   # prompt + answer tokens; longer examples are skipped
BATCH, GRAD_ACC = 4, 4           # effective batch 16
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'])

OUT = 'experiments/simple_dryrun' if DRY_RUN else 'experiments/simple'
os.makedirs(OUT, exist_ok=True)
print(f'DRY_RUN={DRY_RUN}  train/val/test={N_TRAIN}/{N_VAL}/{N_TEST}  epochs={EPOCHS}  -> {OUT}')
""")

md("""
## Data split (architecture Part 6)
SQuAD v2 has no public test split, so validation **and** test are drawn from HF `validation`
in **one** random draw — they cannot overlap. Test indices are saved for audit.
""")

code("""
# Step 2 — Load SQuAD v2 and build disjoint train / val / test splits
import json
import numpy as np
from datasets import load_dataset

squad = load_dataset('rajpurkar/squad_v2')

idx = np.random.default_rng(SEED).choice(len(squad['validation']), N_VAL + N_TEST, replace=False)
val_idx, test_idx = sorted(idx[:N_VAL].tolist()), sorted(idx[N_VAL:].tolist())
assert not set(val_idx) & set(test_idx), 'val/test overlap!'
val, test = squad['validation'].select(val_idx), squad['validation'].select(test_idx)

train_idx = np.random.default_rng(SEED).choice(len(squad['train']), N_TRAIN, replace=False)
train = squad['train'].select(sorted(train_idx.tolist()))

json.dump({'seed': SEED, 'test_indices': test_idx}, open(f'{OUT}/test_indices.json', 'w'))
n_unans = sum(not ex['answers']['text'] for ex in test)
print(f'train={len(train)} val={len(val)} test={len(test)} '
      f'(test: {len(test) - n_unans} answerable, {n_unans} unanswerable)')
""")

md("""
## Prompt + masking (architecture Part 5)
The model is trained only on the **answer** tokens: prompt tokens get label `-100`, so no loss flows from them.
Unanswerable questions have the target `unanswerable`.
""")

code("""
# Step 3 — Prompt format, tokenization and prompt masking
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(SLM)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
UNANS = 'unanswerable'

def prompt(ex):
    return f"### Context:\\n{ex['context'].strip()}\\n\\n### Question:\\n{ex['question'].strip()}\\n\\n### Answer:\\n"

def answer(ex):
    return ex['answers']['text'][0].strip() if ex['answers']['text'] else UNANS

def encode(ex):
    p = tok(prompt(ex))['input_ids']
    a = tok(answer(ex) + tok.eos_token, add_special_tokens=False)['input_ids']
    return {'input_ids': p + a, 'attention_mask': [1] * (len(p) + len(a)),
            'labels': [-100] * len(p) + a}          # mask the prompt, learn the answer

def to_tokens(ds):  # skip over-long examples instead of truncating away the answer
    return ds.map(encode, remove_columns=ds.column_names).filter(lambda e: len(e['input_ids']) <= MAX_LEN)

train_tok, val_tok = to_tokens(train), to_tokens(val)
print(f'kept {len(train_tok)}/{len(train)} train, {len(val_tok)}/{len(val)} val (rest too long)')

ex = train_tok[0]
print('\\nprompt ends with :', repr(tok.decode(ex['input_ids'][-12:])))
print('trained tokens   :', repr(tok.decode([t for t in ex['labels'] if t != -100])))
""")

md("""
## Metrics (architecture Part 7)
Exact Match and token F1 (official SQuAD normalisation), reported **overall, answerable and unanswerable**
so a model that always says "unanswerable" cannot look good.
""")

code("""
# Step 4 — SQuAD metrics with the answerable / unanswerable breakdown
import re, string, collections

def norm(s):
    s = ''.join(ch for ch in s.lower() if ch not in string.punctuation)
    s = re.sub(r'\\b(a|an|the)\\b', ' ', s)
    return ' '.join(s.split())

def em(pred, gold):
    return float(norm(pred) == norm(gold))

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)

def qa_scores(preds, examples):
    rows = []
    for pred, ex in zip(preds, examples):
        golds = ex['answers']['text'] or [UNANS]          # best match over all gold answers
        rows.append((max(em(pred, g) for g in golds), max(f1(pred, g) for g in golds),
                     bool(ex['answers']['text'])))
    pct = lambda xs: round(100 * sum(xs) / len(xs), 2) if xs else None
    return {'em': pct([r[0] for r in rows]), 'f1': pct([r[1] for r in rows]),
            'answerable_em': pct([r[0] for r in rows if r[2]]),
            'unanswerable_em': pct([r[0] for r in rows if not r[2]])}

# self-check
_s = qa_scores(['The Paris!', 'unanswerable', 'Rome'],
               [{'answers': {'text': ['Paris']}}, {'answers': {'text': []}}, {'answers': {'text': ['Paris']}}])
assert (_s['answerable_em'], _s['unanswerable_em']) == (50.0, 100.0), _s
print('metrics self-check OK')
""")

code("""
# Step 5 — Evaluation: real generation on the test set + latency, and a results log
import time
import pandas as pd

RESULTS, PREDS = [], {}

def evaluate(model, tokenizer, examples, batch_size=8, max_new_tokens=32):
    model.eval()
    tokenizer.padding_side = 'left'          # left-pad so generation starts right after "### Answer:"
    dev = next(model.parameters()).device
    preds, secs = [], 0.0
    for i in range(0, len(examples), batch_size):
        batch = examples.select(range(i, min(i + batch_size, len(examples))))
        enc = tokenizer([prompt(ex) for ex in batch], return_tensors='pt', padding=True).to(dev)
        if dev.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        if dev.type == 'cuda': torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        for text in tokenizer.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True):
            preds.append(text.strip().split('\\n')[0].strip())   # the answer is the first line
    scores = qa_scores(preds, examples)
    scores['latency_ms'] = round(1000 * secs / len(examples), 1)
    return scores, preds

def log(method, model_name, scores, preds, **extra):
    RESULTS.append({'method': method, 'model': model_name, 'test_n': len(preds), **scores, **extra})
    PREDS[method] = preds
    pd.DataFrame(RESULTS).to_csv(f'{OUT}/results.csv', index=False)
    print(method, scores, extra)
""")

code("""
# Step 6 — Setup 1: small model, zero-shot (the starting point)
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(SLM, dtype=torch.float32).to(DEVICE)  # fp32 weights; fp16 autocast in training
scores, preds = evaluate(base, tok, test)
log('slm_base_zero_shot', SLM, scores, preds)
""")

md("""
## LoRA fine-tuning (architecture Part 3)
The 500M base weights stay **frozen**; only small A·B adapter matrices on `q_proj`/`v_proj` are trained (~0.17%).
Validation is scored by loss only (cheap on memory); the real EM/F1 comes from generation in the next cell.
""")

code("""
# Step 7 — Setup 2: attach LoRA and fine-tune (our contribution)
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', **LORA))
trainable, total = model.get_nb_trainable_parameters()
print(f'trainable {trainable:,} / {total:,} = {100 * trainable / total:.3f}%')

tok.padding_side = 'right'
args = TrainingArguments(
    output_dir=f'{OUT}/checkpoints', seed=SEED,
    num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC, learning_rate=2e-4, warmup_steps=10, lr_scheduler_type='cosine',
    fp16=(DEVICE == 'cuda'), gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True,
    prediction_loss_only=True,                 # never keep full-vocab logits during eval (T4 OOM)
    logging_steps=1 if DRY_RUN else 25, report_to='none',
)
trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))

if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
train_s = round(time.time() - t0, 1)
peak_mb = round(torch.cuda.max_memory_allocated() / 2**20) if DEVICE == 'cuda' else None

model.save_pretrained(f'{OUT}/adapter')                     # only the adapter, a few MB
json.dump(trainer.state.log_history, open(f'{OUT}/train_log.json', 'w'), indent=1)
losses = [h['loss'] for h in trainer.state.log_history if 'loss' in h]
print(f'train {train_s}s, peak GPU {peak_mb} MB, loss {losses[0]:.3f} -> {losses[-1]:.3f}')
""")

code("""
# Step 8 — Score the LoRA model on the SAME test set, the SAME way as Setup 1
scores, preds = evaluate(model, tok, test)
log('slm_lora', SLM, scores, preds, train_time_s=train_s, peak_train_gpu_mb=peak_mb,
    trainable_pct=round(100 * trainable / total, 3))
""")

code("""
# Step 9 — Setup 3: large model, zero-shot, 4-bit so it fits a T4
if RUN_LLM and DEVICE == 'cuda':
    from transformers import BitsAndBytesConfig
    for name in ('model', 'base', 'trainer'):           # free the small model's GPU memory
        globals().pop(name, None)
    torch.cuda.empty_cache()
    llm_tok = AutoTokenizer.from_pretrained(LLM)
    llm = AutoModelForCausalLM.from_pretrained(LLM, device_map='auto', quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16))
    scores, preds = evaluate(llm, llm_tok, test, batch_size=4)
    log('llm_zero_shot_4bit', LLM, scores, preds)
else:
    print('Skipped Setup 3 (needs RUN_LLM = True and a GPU)')
""")

code("""
# Step 10 — Results: 3-way table, sample predictions, loss curve, download
import matplotlib.pyplot as plt
from IPython.display import display

df = pd.DataFrame(RESULTS)
display(df)

side = pd.DataFrame({'question': [ex['question'] for ex in test], 'gold': [answer(ex) for ex in test], **PREDS})
display(side.head(8))
side.to_csv(f'{OUT}/predictions.csv', index=False)

if os.path.exists(f'{OUT}/train_log.json'):
    hist = [h for h in json.load(open(f'{OUT}/train_log.json')) if 'loss' in h]
    plt.plot([h['step'] for h in hist], [h['loss'] for h in hist]); plt.xlabel('step'); plt.ylabel('train loss'); plt.title('LoRA training loss')
    plt.savefig(f'{OUT}/loss_curve.png', dpi=120, bbox_inches='tight'); plt.show()

print('Saved in', OUT, ':', sorted(os.listdir(OUT)))
if 'google.colab' in sys.modules:
    from google.colab import files
    for f in ['results.csv', 'predictions.csv', 'train_log.json']:
        files.download(f'{OUT}/{f}')
""")


def cell(i, kind, text):
    c = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": text}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"},
                 "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"},
    "cells": [cell(i, k, t) for i, (k, t) in enumerate(CELLS)],
}
out = pathlib.Path(__file__).parent / "simple_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out}  ({len(CELLS)} cells)")
