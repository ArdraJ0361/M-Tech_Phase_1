"""
Run with:  python notebooks/build_notebook.py
Generates: notebooks/05_colab_ready.ipynb
"""
import json, pathlib

def code_cell(source):
    return {"cell_type":"code","id":f"c{abs(hash(source[:30]))%99999}",
            "metadata":{},"outputs":[],"execution_count":None,"source":source}

def md_cell(source):
    return {"cell_type":"markdown","id":f"m{abs(hash(source[:30]))%99999}",
            "metadata":{},"source":source}

# ── all source file contents (plain, no embedding tricks) ────────────────────

SRC = {}
SRC["src/__init__.py"] = ""
SRC["src/data/__init__.py"] = ""
SRC["src/training/__init__.py"] = ""
SRC["src/evaluation/__init__.py"] = ""
SRC["src/utils/__init__.py"] = ""
SRC["src/extraction/__init__.py"] = ""

SRC["src/utils/logging_setup.py"] = (
    "import logging, sys\n\n\n"
    "def setup_logging(level='INFO'):\n"
    "    numeric = getattr(logging, level.upper(), logging.INFO)\n"
    "    handler = logging.StreamHandler(sys.stdout)\n"
    "    handler.setLevel(numeric)\n"
    "    handler.setFormatter(logging.Formatter(\n"
    "        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',\n"
    "        datefmt='%Y-%m-%d %H:%M:%S'))\n"
    "    root = logging.getLogger()\n"
    "    root.setLevel(numeric)\n"
    "    root.handlers = [handler]\n"
    "    for lib in ['urllib3', 'filelock', 'git', 'PIL']:\n"
    "        logging.getLogger(lib).setLevel(logging.WARNING)\n"
)

SRC["src/utils/config_loader.py"] = (
    "import yaml\n"
    "from pathlib import Path\n\n\n"
    "def load_config(path):\n"
    "    p = Path(path)\n"
    "    if not p.exists():\n"
    "        raise FileNotFoundError(f'Config not found: {p}')\n"
    "    with p.open('r', encoding='utf-8') as f:\n"
    "        cfg = yaml.safe_load(f)\n"
    "    required = ['task', 'model_name_or_path', 'dataset', 'training']\n"
    "    missing = [k for k in required if k not in cfg]\n"
    "    if missing:\n"
    "        raise ValueError(f'Config missing keys: {missing}')\n"
    "    return cfg\n"
)

SRC["src/utils/experiment_tracker.py"] = (
    "import csv, logging\n"
    "from pathlib import Path\n\n"
    "logger = logging.getLogger(__name__)\n"
    "_COLS = [\n"
    "    'experiment', 'model', 'dataset', 'task', 'method',\n"
    "    'train_samples', 'val_samples', 'test_samples',\n"
    "    'exact_match', 'f1', 'rouge1', 'rouge2', 'rougeL',\n"
    "    'accuracy', 'macro_f1', 'train_time_s', 'peak_gpu_mb',\n"
    "    'avg_latency_ms', 'trainable_pct', 'checkpoint',\n"
    "]\n\n\n"
    "class ExperimentTracker:\n"
    "    def __init__(self, path='experiments/results.csv'):\n"
    "        self.path = Path(path)\n"
    "        self.path.parent.mkdir(parents=True, exist_ok=True)\n\n"
    "    def log(self, row):\n"
    "        existing = self._cols()\n"
    "        new_keys = [k for k in row if k not in existing]\n"
    "        all_cols = self._sort(existing + new_keys)\n"
    "        exists = self.path.exists()\n"
    "        if not exists or new_keys:\n"
    "            self._rewrite_header(all_cols)\n"
    "        with self.path.open('a', newline='', encoding='utf-8') as f:\n"
    "            wr = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')\n"
    "            if not exists:\n"
    "                wr.writeheader()\n"
    "            wr.writerow(row)\n\n"
    "    def load(self):\n"
    "        import pandas as pd\n"
    "        if not self.path.exists():\n"
    "            return pd.DataFrame()\n"
    "        return pd.read_csv(self.path)\n\n"
    "    def print_summary(self):\n"
    "        df = self.load()\n"
    "        if df.empty:\n"
    "            print('No experiments logged yet.')\n"
    "            return\n"
    "        cols = [c for c in _COLS if c in df.columns]\n"
    "        print(df[cols].to_string(index=False))\n\n"
    "    def _cols(self):\n"
    "        if not self.path.exists():\n"
    "            return []\n"
    "        with self.path.open('r', encoding='utf-8') as f:\n"
    "            r = csv.reader(f)\n"
    "            try:\n"
    "                return next(r)\n"
    "            except StopIteration:\n"
    "                return []\n\n"
    "    def _sort(self, cols):\n"
    "        pref = [c for c in _COLS if c in cols]\n"
    "        ext = [c for c in cols if c not in _COLS]\n"
    "        return pref + ext\n\n"
    "    def _rewrite_header(self, cols):\n"
    "        if not self.path.exists():\n"
    "            return\n"
    "        lines = self.path.read_text(encoding='utf-8').splitlines()\n"
    "        if not lines:\n"
    "            return\n"
    "        lines[0] = ','.join(cols)\n"
    "        self.path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')\n"
)

# prompts.py — use raw string to preserve backslashes correctly
SRC["src/data/prompts.py"] = (
    "import re\n\n"
    "UNANSWERABLE = 'unanswerable'\n"
    "QA_SEPARATOR = '### Answer:\\n'\n"
    "SUMMARY_SEPARATOR = '### Summary:\\n'\n"
    "_WS = re.compile(r'[ \\t\\r\\f\\v]+')\n"
    "_BL = re.compile(r'\\n{3,}')\n\n\n"
    "def clean_text(text):\n"
    "    text = (text or '').replace('\\u00a0', ' ').replace('\\u200b', '')\n"
    "    text = _WS.sub(' ', text)\n"
    "    text = _BL.sub('\\n\\n', text)\n"
    "    return '\\n'.join(l.strip() for l in text.split('\\n')).strip()\n\n\n"
    "def build_qa_prompt(context, question):\n"
    "    return (\n"
    "        f'### Context:\\n{clean_text(context)}\\n\\n'\n"
    "        f'### Question:\\n{clean_text(question)}\\n\\n'\n"
    "        f'{QA_SEPARATOR}'\n"
    "    )\n\n\n"
    "def qa_completion(answers):\n"
    "    texts = answers.get('text') or []\n"
    "    return clean_text(texts[0]) if texts else UNANSWERABLE\n\n\n"
    "def qa_references(answers):\n"
    "    texts = [clean_text(t) for t in (answers.get('text') or [])]\n"
    "    return texts or [UNANSWERABLE]\n\n\n"
    "def build_summarization_prompt(article):\n"
    "    return f'### Article:\\n{clean_text(article)}\\n\\n{SUMMARY_SEPARATOR}'\n\n\n"
    "def truncate_to_tokens(text, tokenizer, max_tokens):\n"
    "    ids = tokenizer(text, add_special_tokens=False)['input_ids']\n"
    "    if len(ids) <= max_tokens:\n"
    "        return text\n"
    "    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).rstrip()\n\n\n"
    "def build_training_text(prompt, completion, eos_token):\n"
    "    eos = eos_token or ''\n"
    "    return f'{prompt}{completion}{eos}'\n"
)

SRC["src/data/preprocessor.py"] = (
    "import logging\n\n"
    "logger = logging.getLogger(__name__)\n"
    "_IGNORE_INDEX = -100\n"
    "_COMPLETION_SEPARATORS = [\n"
    "    '### Answer:\\n',\n"
    "    '### Summary:\\n',\n"
    "    '### Category:\\n',\n"
    "    '### Extracted fields (key: value):\\n',\n"
    "]\n\n\n"
    "def tokenize_qa(examples, tokenizer, cfg):\n"
    "    return _tok(examples, tokenizer,\n"
    "                cfg['dataset']['max_source_length'] + cfg['dataset']['max_target_length'])\n\n\n"
    "def tokenize_seq2seq(examples, tokenizer, cfg):\n"
    "    return _tok(examples, tokenizer,\n"
    "                cfg['dataset']['max_source_length'] + cfg['dataset']['max_target_length'])\n\n\n"
    "def tokenize_classification(examples, tokenizer, cfg):\n"
    "    return _tok(examples, tokenizer,\n"
    "                cfg['dataset']['max_source_length'] + cfg['dataset'].get('max_target_length', 16))\n\n\n"
    "def _tok(examples, tokenizer, max_length):\n"
    "    all_ids, all_masks, all_labels, skipped = [], [], [], 0\n"
    "    prompts = examples['prompt'] if isinstance(examples['prompt'], list) else [examples['prompt']]\n"
    "    for prompt in prompts:\n"
    "        full = tokenizer(prompt, max_length=max_length, truncation=True,\n"
    "                         padding=False, return_tensors=None)\n"
    "        input_ids = full['input_ids']\n"
    "        attn = full['attention_mask']\n"
    "        pp = _prompt_part(prompt)\n"
    "        prompt_len = len(tokenizer(pp, max_length=max_length, truncation=True,\n"
    "                                   padding=False, return_tensors=None)['input_ids'])\n"
    "        labels = list(input_ids)\n"
    "        for i in range(min(prompt_len, len(labels))):\n"
    "            labels[i] = _IGNORE_INDEX\n"
    "        comp = [t for t in labels if t != _IGNORE_INDEX]\n"
    "        if not comp:\n"
    "            skipped += 1\n"
    "            continue\n"
    "        all_ids.append(input_ids)\n"
    "        all_masks.append(attn)\n"
    "        all_labels.append(labels)\n"
    "    if skipped:\n"
    "        logger.warning('%d/%d examples skipped.', skipped, len(prompts))\n"
    "    return {'input_ids': all_ids, 'attention_mask': all_masks, 'labels': all_labels}\n\n\n"
    "_tokenize_prompt_completion = _tok\n\n\n"
    "def _prompt_part(full_prompt):\n"
    "    for sep in _COMPLETION_SEPARATORS:\n"
    "        if sep in full_prompt:\n"
    "            return full_prompt[:full_prompt.rfind(sep) + len(sep)]\n"
    "    return full_prompt\n\n\n"
    "def get_data_collator(tokenizer):\n"
    "    from transformers import DataCollatorForSeq2Seq\n"
    "    return DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,\n"
    "                                  label_pad_token_id=_IGNORE_INDEX, pad_to_multiple_of=8)\n"
)

SRC["src/data/dataset_loader.py"] = (
    "from __future__ import annotations\n"
    "import json, logging\n"
    "from pathlib import Path\n"
    "from .prompts import (UNANSWERABLE, build_qa_prompt, build_summarization_prompt,\n"
    "                      build_training_text, clean_text, qa_completion, truncate_to_tokens)\n\n"
    "logger = logging.getLogger(__name__)\n"
    "_DATASET_ALIASES = {'squad_v2': 'rajpurkar/squad_v2'}\n"
    "_XLSUM_REPO = 'csebuetnlp/xlsum'\n"
    "_XLSUM_PARQUET = 'hf://datasets/csebuetnlp/xlsum@refs%2Fconvert%2Fparquet'\n"
    "_PROMPT_OVERHEAD_TOKENS = 24\n"
    "_SQUAD_SRC = {'train': 'train', 'validation': 'validation', 'test': 'validation'}\n"
    "_HF_SPLIT_SOURCE = {\n"
    "    'squad_v2': _SQUAD_SRC,\n"
    "    'rajpurkar/squad_v2': _SQUAD_SRC,\n"
    "    '_default': {'train': 'train', 'validation': 'validation', 'test': 'test'},\n"
    "}\n\n\n"
    "def load_dataset_for_task(cfg, tokenizer=None):\n"
    "    task = cfg['task']\n"
    "    if task == 'qa':\n"
    "        return _load_qa(cfg['dataset'], tokenizer, cfg)\n"
    "    if task == 'summarization':\n"
    "        return _load_summ(cfg['dataset'], tokenizer, cfg)\n"
    "    raise ValueError(f'Unknown task: {task}')\n\n\n"
    "def _load_raw(name, subset=None):\n"
    "    from datasets import load_dataset\n"
    "    name = _DATASET_ALIASES.get(name, name)\n"
    "    if name == _XLSUM_REPO:\n"
    "        lang = subset or 'english'\n"
    "        base = f'{_XLSUM_PARQUET}/{lang}'\n"
    "        return load_dataset('parquet',\n"
    "                            data_files={s: f'{base}/{s}/*.parquet'\n"
    "                                        for s in ('train', 'validation', 'test')})\n"
    "    return load_dataset(name, subset) if subset else load_dataset(name)\n\n\n"
    "def _load_qa(ds_cfg, tokenizer, cfg):\n"
    "    from .preprocessor import tokenize_qa\n"
    "    raw = _subsample(_load_raw(ds_cfg.get('name', 'rajpurkar/squad_v2'),\n"
    "                               ds_cfg.get('subset')), ds_cfg, cfg)\n"
    "    if tokenizer is None:\n"
    "        return raw\n"
    "    max_src = ds_cfg['max_source_length']\n\n"
    "    def _fmt(ex):\n"
    "        q_len = len(tokenizer(ex['question'], add_special_tokens=False)['input_ids'])\n"
    "        budget = max(max_src - q_len - _PROMPT_OVERHEAD_TOKENS, 32)\n"
    "        ctx = truncate_to_tokens(clean_text(ex['context']), tokenizer, budget)\n"
    "        ans = qa_completion(ex['answers'])\n"
    "        keep = ans == UNANSWERABLE or ans.lower() in ctx.lower()\n"
    "        return {'prompt': build_training_text(build_qa_prompt(ctx, ex['question']),\n"
    "                                              ans, tokenizer.eos_token),\n"
    "                'answer': ans, 'keep': keep}\n\n"
    "    fmt = raw.map(_fmt, remove_columns=raw['train'].column_names)\n"
    "    n = len(fmt['train'])\n"
    "    fmt['train'] = fmt['train'].filter(lambda e: e['keep'])\n"
    "    if len(fmt['train']) < n:\n"
    "        logger.info('Dropped %d train examples.', n - len(fmt['train']))\n"
    "    fmt = fmt.remove_columns('keep')\n"
    "    return fmt.map(lambda ex: tokenize_qa(ex, tokenizer, cfg),\n"
    "                   batched=True, remove_columns=['prompt', 'answer'])\n\n\n"
    "def _load_summ(ds_cfg, tokenizer, cfg):\n"
    "    from .preprocessor import tokenize_seq2seq\n"
    "    raw = _subsample(_load_raw(ds_cfg.get('name', 'csebuetnlp/xlsum'),\n"
    "                               ds_cfg.get('subset', 'english')), ds_cfg, cfg)\n"
    "    if tokenizer is None:\n"
    "        return raw\n"
    "    max_src = ds_cfg['max_source_length']\n"
    "    max_tgt = ds_cfg['max_target_length']\n\n"
    "    def _fmt(ex):\n"
    "        art = truncate_to_tokens(clean_text(ex.get('text', '')), tokenizer,\n"
    "                                 max_src - _PROMPT_OVERHEAD_TOKENS)\n"
    "        summ = truncate_to_tokens(clean_text(ex.get('summary', '')), tokenizer, max_tgt - 1)\n"
    "        return {'prompt': build_training_text(build_summarization_prompt(art),\n"
    "                                              summ, tokenizer.eos_token),\n"
    "                'summary': summ}\n\n"
    "    fmt = raw.map(_fmt, remove_columns=raw['train'].column_names)\n"
    "    return fmt.map(lambda ex: tokenize_seq2seq(ex, tokenizer, cfg),\n"
    "                   batched=True, remove_columns=['prompt', 'summary'])\n\n\n"
    "def _subsample(raw, ds_cfg, cfg):\n"
    "    import numpy as np\n"
    "    from datasets import DatasetDict\n"
    "    seed = cfg.get('training', {}).get('seed', 42)\n"
    "    rng = np.random.default_rng(seed)\n"
    "    srcs = _HF_SPLIT_SOURCE.get(ds_cfg.get('name', ''), _HF_SPLIT_SOURCE['_default'])\n"
    "    n_tr = ds_cfg.get('train_samples'); n_va = ds_cfg.get('val_samples'); n_te = ds_cfg.get('test_samples')\n"
    "    result = {}\n"
    "    val_src = srcs['validation']; test_src = srcs['test']\n"
    "    if val_src == test_src and val_src in raw:\n"
    "        shared = raw[val_src]\n"
    "        total = (n_va or 0) + (n_te or 0)\n"
    "        if total > 0 and total < len(shared):\n"
    "            idx = rng.choice(len(shared), size=total, replace=False).tolist(); idx.sort()\n"
    "            vi = idx[:n_va]; ti = idx[n_va:]\n"
    "        else:\n"
    "            all_i = list(range(len(shared)))\n"
    "            vi = all_i[:n_va] if n_va else all_i\n"
    "            ti = all_i[n_va:n_va + (n_te or 0)] if n_te else []\n"
    "        if vi: result['validation'] = shared.select(vi)\n"
    "        if ti:\n"
    "            result['test'] = shared.select(ti)\n"
    "            _save_idx(ti, cfg, seed, test_src)\n"
    "    else:\n"
    "        for role, hf in [('validation', val_src), ('test', test_src)]:\n"
    "            if hf not in raw: continue\n"
    "            ds = raw[hf]; n = n_va if role == 'validation' else n_te\n"
    "            if n and n < len(ds):\n"
    "                idx = rng.choice(len(ds), size=n, replace=False).tolist(); idx.sort()\n"
    "                ds = ds.select(idx)\n"
    "                if role == 'test': _save_idx(idx, cfg, seed, hf)\n"
    "            result[role] = ds\n"
    "    tr_src = srcs.get('train', 'train')\n"
    "    if tr_src in raw:\n"
    "        ds = raw[tr_src]\n"
    "        if n_tr and n_tr < len(ds):\n"
    "            idx = rng.choice(len(ds), size=n_tr, replace=False).tolist(); idx.sort()\n"
    "            ds = ds.select(idx)\n"
    "        result['train'] = ds\n"
    "    return DatasetDict(result)\n\n\n"
    "def _save_idx(indices, cfg, seed, hf_split):\n"
    "    out = Path(cfg.get('training', {}).get('output_dir', 'experiments/default'))\n"
    "    out.mkdir(parents=True, exist_ok=True)\n"
    "    (out / 'test_indices.json').write_text(\n"
    "        json.dumps({'seed': seed, 'dataset': cfg['dataset'].get('name'),\n"
    "                    'hf_source_split': hf_split, 'num_test_examples': len(indices),\n"
    "                    'indices': indices}, indent=2), encoding='utf-8')\n"
)

SRC["src/evaluation/metrics.py"] = (
    "from __future__ import annotations\n"
    "import re, string, logging\n"
    "from collections import Counter\n"
    "import numpy as np\n\n"
    "logger = logging.getLogger(__name__)\n\n\n"
    "def normalize_answer(s):\n"
    "    def rm_art(t): return re.sub(r'\\b(a|an|the)\\b', ' ', t)\n"
    "    def ws(t): return ' '.join(t.split())\n"
    "    def rm_punc(t): return ''.join(c for c in t if c not in set(string.punctuation))\n"
    "    return ws(rm_art(rm_punc(s.lower())))\n\n\n"
    "def get_tokens(s): return normalize_answer(s).split()\n\n\n"
    "def compute_exact_match(p, g):\n"
    "    return float(normalize_answer(p) == normalize_answer(g))\n\n\n"
    "def compute_f1(p, g):\n"
    "    pt = get_tokens(p); gt = get_tokens(g)\n"
    "    common = Counter(pt) & Counter(gt); n = sum(common.values())\n"
    "    if n == 0: return 0.0\n"
    "    pr = n / len(pt); rc = n / len(gt)\n"
    "    return 2 * pr * rc / (pr + rc)\n\n\n"
    "def compute_qa_metrics(predictions, references, report_unanswerable_breakdown=True):\n"
    "    ems = [compute_exact_match(p, r) for p, r in zip(predictions, references)]\n"
    "    f1s = [compute_f1(p, r) for p, r in zip(predictions, references)]\n"
    "    res = {'exact_match': round(np.mean(ems) * 100, 2),\n"
    "           'f1': round(np.mean(f1s) * 100, 2),\n"
    "           'num_examples': len(predictions)}\n"
    "    if report_unanswerable_breakdown:\n"
    "        ans = [(p,r,e,f) for p,r,e,f in zip(predictions,references,ems,f1s)\n"
    "               if normalize_answer(r) != 'unanswerable']\n"
    "        una = [(p,r,e,f) for p,r,e,f in zip(predictions,references,ems,f1s)\n"
    "               if normalize_answer(r) == 'unanswerable']\n"
    "        if ans:\n"
    "            res['answerable_em'] = round(np.mean([x[2] for x in ans]) * 100, 2)\n"
    "            res['answerable_f1'] = round(np.mean([x[3] for x in ans]) * 100, 2)\n"
    "            res['answerable_n'] = len(ans)\n"
    "        if una:\n"
    "            res['unanswerable_em'] = round(np.mean([x[2] for x in una]) * 100, 2)\n"
    "            res['unanswerable_f1'] = round(np.mean([x[3] for x in una]) * 100, 2)\n"
    "            res['unanswerable_n'] = len(una)\n"
    "    return res\n\n\n"
    "def compute_rouge(predictions, references):\n"
    "    from rouge_score import rouge_scorer\n"
    "    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)\n"
    "    sc = {'rouge1': [], 'rouge2': [], 'rougeL': []}\n"
    "    for p, r in zip(predictions, references):\n"
    "        out = scorer.score(r, p)\n"
    "        for k in sc: sc[k].append(out[k].fmeasure)\n"
    "    return {'rouge1': round(np.mean(sc['rouge1']) * 100, 2),\n"
    "            'rouge2': round(np.mean(sc['rouge2']) * 100, 2),\n"
    "            'rougeL': round(np.mean(sc['rougeL']) * 100, 2),\n"
    "            'num_examples': len(predictions)}\n\n\n"
    "def make_compute_metrics_fn(task, tokenizer):\n"
    "    def _decode(ids):\n"
    "        ids = np.where(ids == -100, tokenizer.pad_token_id, ids)\n"
    "        return tokenizer.batch_decode(ids, skip_special_tokens=True)\n"
    "    def compute_metrics(eval_pred):\n"
    "        preds, labels = eval_pred\n"
    "        if isinstance(preds, tuple): preds = preds[0]\n"
    "        if preds.ndim == 3: preds = np.argmax(preds, axis=-1)\n"
    "        dp = [p.strip() for p in _decode(preds)]\n"
    "        dl = [l.strip() for l in _decode(labels)]\n"
    "        if task == 'qa': return compute_qa_metrics(dp, dl)\n"
    "        if task == 'summarization': return compute_rouge(dp, dl)\n"
    "        return {}\n"
    "    return compute_metrics\n"
)

SRC["src/evaluation/evaluator.py"] = (
    "from __future__ import annotations\n"
    "import logging, time\n"
    "import numpy as np\n"
    "import torch\n"
    "from .metrics import compute_qa_metrics, compute_rouge\n\n"
    "logger = logging.getLogger(__name__)\n"
    "_TASK_FN = {'qa': compute_qa_metrics, 'summarization': compute_rouge}\n\n\n"
    "def _split(ex):\n"
    "    labels = ex['labels']\n"
    "    start = next(i for i, t in enumerate(labels) if t != -100)\n"
    "    return list(ex['input_ids'][:start]), [t for t in labels[start:] if t != -100]\n\n\n"
    "def _postprocess(pred, task):\n"
    "    pred = pred.split('\\n###')[0].strip()\n"
    "    if task == 'qa': pred = pred.split('\\n')[0].strip()\n"
    "    return pred\n\n\n"
    "def evaluate_model(model, tokenizer, dataset, cfg, batch_size=8):\n"
    "    task = cfg['task']\n"
    "    max_new = cfg['dataset'].get('max_target_length', 64)\n"
    "    device = next(model.parameters()).device\n"
    "    pad = tokenizer.pad_token_id\n"
    "    model.eval()\n"
    "    pairs = [_split(dataset[i]) for i in range(len(dataset))]\n"
    "    all_preds = []\n"
    "    all_refs = [tokenizer.decode(r, skip_special_tokens=True).strip() for _, r in pairs]\n"
    "    gen_s = 0.0\n"
    "    with torch.no_grad():\n"
    "        for b in range(0, len(pairs), batch_size):\n"
    "            chunk = pairs[b:b + batch_size]\n"
    "            prompts = [p for p, _ in chunk]\n"
    "            wid = max(len(p) for p in prompts)\n"
    "            ids = torch.tensor([[pad] * (wid - len(p)) + p for p in prompts], device=device)\n"
    "            msk = torch.tensor([[0] * (wid - len(p)) + [1] * len(p) for p in prompts], device=device)\n"
    "            if device.type == 'cuda': torch.cuda.synchronize()\n"
    "            t0 = time.perf_counter()\n"
    "            out = model.generate(input_ids=ids, attention_mask=msk,\n"
    "                                 max_new_tokens=max_new, do_sample=False, use_cache=True,\n"
    "                                 pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)\n"
    "            if device.type == 'cuda': torch.cuda.synchronize()\n"
    "            gen_s += time.perf_counter() - t0\n"
    "            dec = tokenizer.batch_decode(out[:, wid:], skip_special_tokens=True)\n"
    "            all_preds.extend(_postprocess(p, task) for p in dec)\n"
    "    fn = _TASK_FN.get(task)\n"
    "    if fn is None: return {}\n"
    "    metrics = fn(all_preds, all_refs)\n"
    "    metrics['avg_latency_ms'] = round(gen_s * 1000 / len(all_preds), 1)\n"
    "    metrics['throughput_samples_per_s'] = round(len(all_preds) / gen_s, 2)\n"
    "    metrics['_predictions'] = all_preds\n"
    "    metrics['_references'] = all_refs\n"
    "    return metrics\n"
)

SRC["src/training/model_loader.py"] = (
    "from __future__ import annotations\n"
    "import logging\n"
    "import torch\n"
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n\n"
    "logger = logging.getLogger(__name__)\n\n\n"
    "def load_tokenizer(cfg):\n"
    "    path = cfg.get('tokenizer_name_or_path') or cfg['model_name_or_path']\n"
    "    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side='right')\n"
    "    if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
    "    return tok\n\n\n"
    "def load_base_model(cfg):\n"
    "    path = cfg['model_name_or_path']; bits = cfg.get('load_in_bits')\n"
    "    bnb = _bnb(bits) if bits in (4, 8) else None\n"
    "    kw = {'trust_remote_code': True, 'device_map': 'auto'}\n"
    "    if bnb: kw['quantization_config'] = bnb\n"
    "    elif torch.cuda.is_available():\n"
    "        kw['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16\n"
    "    else: kw['torch_dtype'] = torch.float32\n"
    "    model = AutoModelForCausalLM.from_pretrained(path, **kw)\n"
    "    model.config.use_cache = False\n"
    "    return model\n\n\n"
    "def _bnb(bits):\n"
    "    if bits == 4:\n"
    "        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,\n"
    "                                  bnb_4bit_quant_type='nf4',\n"
    "                                  bnb_4bit_compute_dtype=torch.bfloat16)\n"
    "    return BitsAndBytesConfig(load_in_8bit=True)\n\n\n"
    "def attach_lora(model, cfg):\n"
    "    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
    "    pc = cfg.get('peft', {}); bits = cfg.get('load_in_bits')\n"
    "    if bits in (4, 8): model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)\n"
    "    lc = LoraConfig(r=pc.get('r', 8), lora_alpha=pc.get('lora_alpha', 16),\n"
    "                    lora_dropout=pc.get('lora_dropout', 0.05),\n"
    "                    target_modules=pc.get('target_modules'),\n"
    "                    bias=pc.get('bias', 'none'), task_type=pc.get('task_type', 'CAUSAL_LM'))\n"
    "    model = get_peft_model(model, lc)\n"
    "    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
    "    tot = sum(p.numel() for p in model.parameters())\n"
    "    pct = 100 * tr / tot if tot else 0\n"
    "    print(f'\\n✅ LoRA: {tr:,} trainable / {tot:,} total ({pct:.3f}%)')\n"
    "    cfg['_trainable_params'] = tr; cfg['_total_params'] = tot; cfg['_trainable_pct'] = round(pct, 4)\n"
    "    return model\n\n\n"
    "def load_peft_checkpoint(base_cfg, checkpoint_path):\n"
    "    from peft import PeftModel\n"
    "    model = PeftModel.from_pretrained(load_base_model(base_cfg), checkpoint_path)\n"
    "    model.eval(); return model\n"
)

SRC["src/training/trainer.py"] = (
    "from __future__ import annotations\n"
    "import inspect, logging, time\n"
    "from pathlib import Path\n"
    "import torch\n"
    "from transformers import TrainingArguments, Trainer\n"
    "from .model_loader import load_tokenizer, load_base_model, attach_lora\n"
    "from ..data.dataset_loader import load_dataset_for_task\n"
    "from ..data.preprocessor import get_data_collator\n"
    "from ..evaluation.metrics import make_compute_metrics_fn\n"
    "from ..utils.experiment_tracker import ExperimentTracker\n"
    "from ..utils.config_loader import load_config\n\n"
    "logger = logging.getLogger(__name__)\n"
    "# Detect which TrainingArguments param names this transformers version supports.\n"
    "_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)\n\n\n"
    "def _ta_kwarg(name, alt, value):\n"
    "    \"\"\"Return {name: value} using 'alt' if 'name' not supported.\"\"\"\n"
    "    return {name if name in _TA_PARAMS else alt: value}\n\n\n"
    "def _peak():\n"
    "    return torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0\n\n\n"
    "def run_training(config_path):\n"
    "    cfg = load_config(config_path)\n"
    "    out = Path(cfg['training']['output_dir']); out.mkdir(parents=True, exist_ok=True)\n"
    "    tok = load_tokenizer(cfg)\n"
    "    ds = load_dataset_for_task(cfg, tok)\n"
    "    model = attach_lora(load_base_model(cfg), cfg)\n"
    "    tracker = ExperimentTracker(cfg['results_csv']); mem0 = _peak()\n"
    "    tc = cfg['training']\n"
    "    # Build kwargs compatible with any transformers version (4.x through 4.50+)\n"
    "    ta_kw = {\n"
    "        'output_dir': str(out),\n"
    "        'num_train_epochs': tc.get('num_train_epochs', 3),\n"
    "        'per_device_train_batch_size': tc.get('per_device_train_batch_size', 4),\n"
    "        'per_device_eval_batch_size': tc.get('per_device_eval_batch_size', 8),\n"
    "        'gradient_accumulation_steps': tc.get('gradient_accumulation_steps', 4),\n"
    "        'learning_rate': float(tc.get('learning_rate', 2e-4)),\n"
    "        'weight_decay': float(tc.get('weight_decay', 0.01)),\n"
    "        'lr_scheduler_type': tc.get('lr_scheduler_type', 'cosine'),\n"
    "        'fp16': tc.get('fp16', False),\n"
    "        'bf16': tc.get('bf16', False),\n"
    "        'logging_steps': tc.get('logging_steps', 50),\n"
    "        'save_steps': tc.get('save_steps', 200),\n"
    "        'save_strategy': 'steps',\n"
    "        'save_total_limit': tc.get('save_total_limit', 2),\n"
    "        'load_best_model_at_end': tc.get('load_best_model_at_end', True),\n"
    "        'metric_for_best_model': tc.get('metric_for_best_model', 'eval_loss'),\n"
    "        'seed': tc.get('seed', 42),\n"
    "        'report_to': 'none',\n"
    "        'dataloader_pin_memory': False,\n"
    "        'gradient_checkpointing': True,\n"
    "    }\n"
    "    # warmup_ratio (>=4.6) vs warmup_steps (all versions)\n"
    "    ta_kw.update(_ta_kwarg('warmup_ratio', 'warmup_steps',\n"
    "                           float(tc.get('warmup_ratio', 0.03))\n"
    "                           if 'warmup_ratio' in _TA_PARAMS else 0))\n"
    "    # eval_steps + strategy: newer versions use eval_strategy, older use evaluation_strategy\n"
    "    eval_strat_key = 'eval_strategy' if 'eval_strategy' in _TA_PARAMS else 'evaluation_strategy'\n"
    "    ta_kw[eval_strat_key] = 'steps'\n"
    "    ta_kw['eval_steps'] = tc.get('eval_steps', 200)\n"
    "    args = TrainingArguments(**ta_kw)\n"
    "    # 'tokenizer' was renamed to 'processing_class' in transformers>=4.47\n"
    "    _TR_PARAMS = set(inspect.signature(Trainer.__init__).parameters)\n"
    "    tok_kw = 'processing_class' if 'processing_class' in _TR_PARAMS else 'tokenizer'\n"
    "    trainer = Trainer(\n"
    "        model=model, args=args, train_dataset=ds.get('train'),\n"
    "        eval_dataset=ds.get('validation'), **{tok_kw: tok},\n"
    "        data_collator=get_data_collator(tok),\n"
    "        compute_metrics=make_compute_metrics_fn(cfg['task'], tok),\n"
    "    )\n"
    "    t0 = time.time(); trainer.train()\n"
    "    train_t = time.time() - t0; mem = _peak() - mem0\n"
    "    test_m = {}\n"
    "    if 'test' in ds: test_m = trainer.evaluate(eval_dataset=ds['test'])\n"
    "    model.save_pretrained(str(out / 'adapter')); tok.save_pretrained(str(out / 'adapter'))\n"
    "    tracker.log({\n"
    "        'experiment': cfg.get('experiment_name', ''), 'model': cfg['model_name_or_path'],\n"
    "        'model_variant': cfg.get('model_variant', 'base'),\n"
    "        'dataset': cfg['dataset'].get('name', ''), 'task': cfg['task'],\n"
    "        'method': 'LoRA' if not cfg.get('load_in_bits') else 'QLoRA',\n"
    "        'seed': tc.get('seed', 42),\n"
    "        'train_samples': len(ds.get('train', [])), 'val_samples': len(ds.get('validation', [])),\n"
    "        'test_samples': len(ds.get('test', [])), 'trainable_pct': cfg.get('_trainable_pct'),\n"
    "        'train_time_s': round(train_t, 1), 'peak_gpu_mb': round(mem, 1),\n"
    "        'checkpoint': str(out / 'adapter'),\n"
    "        **{k: round(v, 4) if isinstance(v, float) else v for k, v in test_m.items()},\n"
    "    })\n"
    "    return test_m\n"
)

CONFIGS = {}
CONFIGS["configs/pilot_squad_qa.yaml"] = (
    'experiment_name: "squad_qa_lora_pilot"\n'
    'task: "qa"\n'
    'model_name_or_path: "Qwen/Qwen1.5-0.5B"\n'
    'tokenizer_name_or_path: null\n'
    'model_variant: "base"\n'
    'use_flash_attention: false\n'
    'load_in_bits: null\n'
    'peft:\n'
    '  method: "lora"\n'
    '  r: 8\n'
    '  lora_alpha: 16\n'
    '  lora_dropout: 0.05\n'
    '  target_modules: ["q_proj", "v_proj"]\n'
    '  bias: "none"\n'
    '  task_type: "CAUSAL_LM"\n'
    'dataset:\n'
    '  name: "rajpurkar/squad_v2"\n'
    '  version: "default"\n'
    '  subset: null\n'
    '  text_column: "context"\n'
    '  question_column: "question"\n'
    '  answer_column: "answers"\n'
    '  max_source_length: 256\n'
    '  max_target_length: 32\n'
    '  train_samples: 5000\n'
    '  val_samples: 500\n'
    '  test_samples: 500\n'
    'training:\n'
    '  output_dir: "experiments/squad_qa"\n'
    '  num_train_epochs: 3\n'
    '  per_device_train_batch_size: 2\n'
    '  per_device_eval_batch_size: 2\n'
    '  gradient_accumulation_steps: 8\n'
    '  learning_rate: 2.0e-4\n'
    '  weight_decay: 0.01\n'
    '  warmup_ratio: 0.03\n'
    '  lr_scheduler_type: "cosine"\n'
    '  fp16: true\n'
    '  bf16: false\n'
    '  logging_steps: 50\n'
    '  eval_steps: 500\n'
    '  save_steps: 500\n'
    '  save_total_limit: 2\n'
    '  load_best_model_at_end: true\n'
    '  metric_for_best_model: "eval_f1"\n'
    '  seed: 42\n'
    'evaluation:\n'
    '  metrics: ["exact_match", "f1"]\n'
    '  record_efficiency: true\n'
    'results_csv: "experiments/results.csv"\n'
)
CONFIGS["configs/pilot_xlsum_summarization.yaml"] = (
    'experiment_name: "xlsum_summarization_lora_pilot"\n'
    'task: "summarization"\n'
    'model_name_or_path: "Qwen/Qwen1.5-0.5B"\n'
    'tokenizer_name_or_path: null\n'
    'model_variant: "base"\n'
    'use_flash_attention: false\n'
    'load_in_bits: null\n'
    'peft:\n'
    '  method: "lora"\n'
    '  r: 8\n'
    '  lora_alpha: 16\n'
    '  lora_dropout: 0.05\n'
    '  target_modules: ["q_proj", "v_proj"]\n'
    '  bias: "none"\n'
    '  task_type: "CAUSAL_LM"\n'
    'dataset:\n'
    '  name: "csebuetnlp/xlsum"\n'
    '  version: "default"\n'
    '  subset: "english"\n'
    '  text_column: "text"\n'
    '  summary_column: "summary"\n'
    '  max_source_length: 512\n'
    '  max_target_length: 84\n'
    '  train_samples: 5000\n'
    '  val_samples: 500\n'
    '  test_samples: 500\n'
    'training:\n'
    '  output_dir: "experiments/xlsum_summarization"\n'
    '  num_train_epochs: 3\n'
    '  per_device_train_batch_size: 4\n'
    '  per_device_eval_batch_size: 8\n'
    '  gradient_accumulation_steps: 4\n'
    '  learning_rate: 2.0e-4\n'
    '  weight_decay: 0.01\n'
    '  warmup_ratio: 0.03\n'
    '  lr_scheduler_type: "cosine"\n'
    '  fp16: false\n'
    '  bf16: true\n'
    '  logging_steps: 50\n'
    '  eval_steps: 200\n'
    '  save_steps: 200\n'
    '  save_total_limit: 2\n'
    '  load_best_model_at_end: true\n'
    '  metric_for_best_model: "eval_rougeL"\n'
    '  seed: 42\n'
    'evaluation:\n'
    '  metrics: ["rouge1", "rouge2", "rougeL"]\n'
    '  record_efficiency: true\n'
    'results_csv: "experiments/results.csv"\n'
)
CONFIGS["configs/llm_baseline_qwen7b.yaml"] = (
    'experiment_name: "squad_qa_llm_baseline"\n'
    'task: "qa"\n'
    'model_name_or_path: "Qwen/Qwen1.5-7B"\n'
    'tokenizer_name_or_path: null\n'
    'model_variant: "base"\n'
    'use_flash_attention: false\n'
    'load_in_bits: 4\n'
    'peft:\n'
    '  method: "lora"\n'
    '  r: 8\n'
    '  lora_alpha: 16\n'
    '  lora_dropout: 0.05\n'
    '  target_modules: ["q_proj", "v_proj"]\n'
    '  bias: "none"\n'
    '  task_type: "CAUSAL_LM"\n'
    'dataset:\n'
    '  name: "rajpurkar/squad_v2"\n'
    '  version: "default"\n'
    '  subset: null\n'
    '  text_column: "context"\n'
    '  question_column: "question"\n'
    '  answer_column: "answers"\n'
    '  max_source_length: 512\n'
    '  max_target_length: 64\n'
    '  train_samples: 5000\n'
    '  val_samples: 500\n'
    '  test_samples: 500\n'
    'training:\n'
    '  output_dir: "experiments/llm_baseline"\n'
    '  num_train_epochs: 1\n'
    '  per_device_train_batch_size: 2\n'
    '  per_device_eval_batch_size: 4\n'
    '  gradient_accumulation_steps: 8\n'
    '  learning_rate: 2.0e-4\n'
    '  weight_decay: 0.01\n'
    '  warmup_ratio: 0.03\n'
    '  lr_scheduler_type: "cosine"\n'
    '  fp16: false\n'
    '  bf16: false\n'
    '  logging_steps: 50\n'
    '  eval_steps: 200\n'
    '  save_steps: 200\n'
    '  save_total_limit: 2\n'
    '  load_best_model_at_end: false\n'
    '  metric_for_best_model: "eval_f1"\n'
    '  seed: 42\n'
    'evaluation:\n'
    '  metrics: ["exact_match", "f1"]\n'
    '  record_efficiency: true\n'
    'results_csv: "experiments/results.csv"\n'
)

# ── Build Cell 2 source: write files using repr() so JSON serialises cleanly ──

lines = [
    "# ── Cell 2: Write all source files ───────────────────────────────────────",
    "# Each file content is stored as a Python repr-escaped string.",
    "# No textwrap, no inspect, no indentation tricks — guaranteed correct.",
    "import pathlib, ast, sys",
    "",
    'ROOT = pathlib.Path("/content/Mtech")',
    "",
    "def _w(rel, text):",
    "    p = ROOT / rel",
    "    p.parent.mkdir(parents=True, exist_ok=True)",
    '    p.write_text(text, encoding="utf-8")',
    "",
]

for rel, content in {**SRC, **CONFIGS}.items():
    # Use repr() so the string is 100% safely embedded in the cell source
    lines.append(f"_w({rel!r}, {content!r})")
    lines.append("")

lines += [
    "# ── verify all .py files parse ───────────────────────────────────────────",
    "errors = []",
    "for pyf in sorted(ROOT.rglob('*.py')):",
    "    if '__pycache__' in str(pyf): continue",
    "    try:",
    "        ast.parse(pyf.read_text(encoding='utf-8'))",
    "    except SyntaxError as e:",
    "        errors.append(f'  {pyf.relative_to(ROOT)}: {e}')",
    "if errors:",
    "    print('SYNTAX ERRORS:'); [print(e) for e in errors]; sys.exit(1)",
    "else:",
    "    print('✅ All source files written — no syntax errors')",
    "    print('   Continue to Cell 3.')",
]

write_files_src = "\n".join(lines)

# ── Notebook cells ────────────────────────────────────────────────────────────

cells = []

cells.append(md_cell(
    "# 🚀 PEFT-SLM — Final Self-Contained Colab Notebook\n"
    "### M.Tech: Parameter-Efficient Fine-Tuning of Small Language Models\n"
    "---\n"
    "**No uploads needed. Run every cell top to bottom.**\n\n"
    "| Cell | What it does | Time |\n"
    "|---|---|---|\n"
    "| 0 — GPU check | Verify T4 | <1 min |\n"
    "| 1 — Install | pip install | ~4 min |\n"
    "| 2 — Write files | Creates all src/ + configs/ | <1 min |\n"
    "| 3 — Imports | Verify project imports | <1 min |\n"
    "| 4 — Smoke test | 9 pipeline checks | ~5 min |\n"
    "| 5 — Quick demo | 3-way QA comparison (5 examples) | ~10 min |\n"
    "| 6A — Base SLM | SQuAD v2 zero-shot baseline | ~20 min |\n"
    "| 6B — LoRA train | Fine-tune PEFT adapter ← your result | ~90 min |\n"
    "| 6C — LLM baseline | Qwen1.5-7B zero-shot | ~25 min |\n"
    "| 6D — Results | 3-way table + chart | <1 min |\n"
    "| 7A — XL-Sum base | Summarization zero-shot | ~20 min |\n"
    "| 7B — XL-Sum LoRA | Summarization fine-tuning | ~2 hr |\n"
    "| 8 — Download | Save results.csv to your PC | <1 min |\n\n"
    "> ⚠️ **Before running:** Runtime → Change runtime type → T4 GPU"
))

cells.append(code_cell(
    "# ── Cell 0: GPU Check ─────────────────────────────────────────────────────\n"
    "import torch\n"
    "if not torch.cuda.is_available():\n"
    "    raise RuntimeError(\n"
    "        'No GPU detected!\\n'\n"
    "        'Fix: Runtime -> Change runtime type -> T4 GPU\\n'\n"
    "        'Then: Runtime -> Disconnect and delete runtime -> reconnect.'\n"
    "    )\n"
    "name = torch.cuda.get_device_name(0)\n"
    "vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
    "print(f'✅ GPU  : {name}')\n"
    "print(f'✅ VRAM : {vram:.1f} GB')\n"
    "print(f'   PyTorch {torch.__version__} | CUDA {torch.version.cuda}')\n"
    "if vram < 12:\n"
    "    print('ℹ️  < 12 GB VRAM — 7B baseline uses 4-bit (already set in config).')"
))

cells.append(code_cell(
    "# ── Cell 1: Install packages ──────────────────────────────────────────────\n"
    "# pymupdf/pytesseract excluded: not needed for Phase 1.\n"
    "# Pillow pinned <11 to avoid PIL._Ink ImportError on Python 3.13 runtimes.\n"
    "# torchao pinned >=0.16.0: peft requires this minimum version.\n"
    "import subprocess, sys\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "    'torch', 'transformers', 'datasets', 'peft', 'accelerate', 'bitsandbytes',\n"
    "    'torchao>=0.16.0',\n"
    "    'evaluate', 'rouge-score', 'nltk',\n"
    "    'Pillow>=10.0,<11.0',\n"
    "    'pandas', 'numpy', 'pyyaml', 'tqdm', 'scikit-learn',\n"
    "    'matplotlib', 'seaborn', 'tensorboard', 'ipywidgets',\n"
    "])\n"
    "import importlib\n"
    "ok = True\n"
    "for pkg in ['torch','transformers','peft','datasets','evaluate','accelerate','bitsandbytes']:\n"
    "    try:\n"
    "        m = importlib.import_module(pkg)\n"
    "        print(f'  [OK] {pkg:<20} {getattr(m,\"__version__\",\"?\")}')\n"
    "    except ImportError as e:\n"
    "        print(f'  [FAIL] {pkg}: {e}'); ok = False\n"
    "print('\\n✅ All packages installed — continue to Cell 2' if ok else '\\n❌ Re-run this cell')"
))

cells.append(code_cell(write_files_src))

cells.append(code_cell(
    "# ── Cell 3: Add project to Python path ───────────────────────────────────\n"
    "import sys, os\n"
    "PROJECT_ROOT = '/content/Mtech'\n"
    "if PROJECT_ROOT not in sys.path:\n"
    "    sys.path.insert(0, PROJECT_ROOT)\n"
    "os.chdir(PROJECT_ROOT)\n"
    "print(f'Working dir: {os.getcwd()}')\n"
    "from src.utils.config_loader import load_config\n"
    "from src.utils.experiment_tracker import ExperimentTracker\n"
    "from src.data.prompts import build_qa_prompt, build_summarization_prompt\n"
    "from src.evaluation.metrics import compute_qa_metrics, compute_rouge, compute_exact_match\n"
    "print('✅ All project imports OK — continue to Cell 4')"
))

cells.append(code_cell(
    "# ── Cell 4: Smoke Test (9 checks) ────────────────────────────────────────\n"
    "import traceback\n"
    "PASS = '[OK] '; FAIL = '[FAIL]'; failures = []\n"
    "def check(name, fn):\n"
    "    try:\n"
    "        r = fn(); print(f'{PASS} {name}'); return r\n"
    "    except Exception as e:\n"
    "        print(f'{FAIL} {name}: {e}'); failures.append(name)\n"
    "        traceback.print_exc(); return None\n"
    "\n"
    "# [1] Config\n"
    "def _c1():\n"
    "    cfg = load_config('configs/pilot_squad_qa.yaml')\n"
    "    assert cfg['task'] == 'qa'\n"
    "    assert cfg['model_name_or_path'] == 'Qwen/Qwen1.5-0.5B'\n"
    "    assert cfg['training']['seed'] == 42\n"
    "    return cfg\n"
    "cfg = check('[1] Config loads correctly', _c1)\n"
    "if cfg is None: raise RuntimeError('Fix config before continuing.')\n"
    "\n"
    "# [2] Dataset splits\n"
    "def _c2():\n"
    "    from datasets import load_dataset\n"
    "    from src.data.dataset_loader import _subsample\n"
    "    c = {**cfg, 'dataset': {**cfg['dataset'], 'train_samples':50,'val_samples':25,'test_samples':25},\n"
    "         'training': {**cfg['training'], 'output_dir':'experiments/smoke_test'}}\n"
    "    raw = load_dataset('rajpurkar/squad_v2')\n"
    "    r = _subsample(raw, c['dataset'], c)\n"
    "    assert len(r['train'])==50 and len(r['validation'])==25 and len(r['test'])==25\n"
    "    un = [e for e in list(r['test']) if not e['answers']['text']]\n"
    "    print(f'         test: {25-len(un)} answerable, {len(un)} unanswerable')\n"
    "    assert len(un) > 0\n"
    "check('[2] Dataset splits: correct sizes, unanswerable present', _c2)\n"
    "\n"
    "# [3] Prompt format\n"
    "def _c3():\n"
    "    p = build_qa_prompt('Paris is the capital of France.', 'Capital?')\n"
    "    assert '### Context:' in p and '### Question:' in p and '### Answer:' in p\n"
    "    print(f'         {p[:60].strip()}...')\n"
    "check('[3] Prompt format correct', _c3)\n"
    "\n"
    "# [4,5,6] Tokenizer + masking\n"
    "def _c4():\n"
    "    from transformers import AutoTokenizer\n"
    "    from src.data.preprocessor import _tokenize_prompt_completion, _IGNORE_INDEX\n"
    "    tok = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B', trust_remote_code=True)\n"
    "    if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
    "    prompts = [\n"
    "        '### Context:\\nParis is capital.\\n\\n### Question:\\nCapital?\\n\\n### Answer:\\nParis',\n"
    "        '### Context:\\nLoRA is PEFT.\\n\\n### Question:\\nWhat?\\n\\n### Answer:\\nunanswerable',\n"
    "    ]\n"
    "    r = _tokenize_prompt_completion({'prompt': prompts}, tok, 256)\n"
    "    assert len(r['input_ids']) == 2\n"
    "    for i, (ids, labels) in enumerate(zip(r['input_ids'], r['labels'])):\n"
    "        assert len(ids) == len(labels)\n"
    "        comp = [t for t in labels if t != _IGNORE_INDEX]\n"
    "        assert len(comp) > 0\n"
    "        print(f'         Ex {i}: completion={len(comp)}, masked={labels.count(_IGNORE_INDEX)}')\n"
    "check('[4,5,6] Tokenisation + prompt masking correct', _c4)\n"
    "\n"
    "# [7] Base model inference\n"
    "def _c7():\n"
    "    import torch\n"
    "    from transformers import AutoModelForCausalLM, AutoTokenizer\n"
    "    print('         Loading Qwen1.5-0.5B (~30s)...')\n"
    "    tok = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B', trust_remote_code=True)\n"
    "    if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
    "    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B', trust_remote_code=True,\n"
    "        torch_dtype=torch.bfloat16, device_map='auto')\n"
    "    model.eval()\n"
    "    prompt = '### Context:\\nParis is the capital of France.\\n\\n### Question:\\nCapital?\\n\\n### Answer:\\n'\n"
    "    inp = tok(prompt, return_tensors='pt').to('cuda')\n"
    "    with torch.no_grad():\n"
    "        out = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)\n"
    "    pred = tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()\n"
    "    print(f\"         Prediction: '{pred[:60]}'\")\n"
    "    assert isinstance(pred, str)\n"
    "    del model; torch.cuda.empty_cache()\n"
    "check('[7] Base model inference runs', _c7)\n"
    "\n"
    "# [8] Metrics\n"
    "def _c8():\n"
    "    assert compute_exact_match('Paris','Paris') == 1.0\n"
    "    assert compute_exact_match('paris','Paris') == 1.0\n"
    "    assert compute_exact_match('London','Paris') == 0.0\n"
    "    r = compute_qa_metrics(['Paris','unanswerable','London','unanswerable'],\n"
    "                            ['Paris','unanswerable','Paris','unanswerable'])\n"
    "    assert r['answerable_em'] == 50.0 and r['unanswerable_em'] == 100.0\n"
    "    rouge = compute_rouge(['A fish was found.'], ['Scientists found a new fish.'])\n"
    "    assert rouge['rouge1'] > 0\n"
    "    print(f\"         EM={r['exact_match']} AnsEM={r['answerable_em']} UnansEM={r['unanswerable_em']}\")\n"
    "check('[8] Metrics: EM, F1, ROUGE, unanswerable breakdown', _c8)\n"
    "\n"
    "# [9] Experiment tracker\n"
    "def _c9():\n"
    "    t = ExperimentTracker('experiments/smoke_test/results.csv')\n"
    "    t.log({'experiment':'smoke_test','model':'Qwen/Qwen1.5-0.5B','task':'qa',\n"
    "           'method':'base_zero_shot','exact_match':20.0,'f1':33.5,\n"
    "           'answerable_em':25.0,'unanswerable_em':15.0})\n"
    "    df = t.load()\n"
    "    assert len(df) >= 1 and 'exact_match' in df.columns\n"
    "check('[9] ExperimentTracker writes and reads CSV', _c9)\n"
    "\n"
    "print()\n"
    "print('=' * 60)\n"
    "if failures:\n"
    "    print(f'❌ FAILED: {failures}')\n"
    "else:\n"
    "    print('✅ ALL SMOKE TEST CHECKS PASSED — continue to Cell 5')"
))

cells.append(code_cell(
    "# ── Cell 5: Quick 3-Way Demo (~10 min) ──────────────────────────────────\n"
    "import torch, time, logging\n"
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
    "from peft import LoraConfig, get_peft_model\n"
    "import torch.optim as optim\n"
    "logging.basicConfig(level=logging.WARNING)\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "\n"
    "EXAMPLES = [\n"
    "    {'context':'The Eiffel Tower is in Paris, France.','question':'Where is the Eiffel Tower?','answer':'Paris, France'},\n"
    "    {'context':'Python was created by Guido van Rossum in 1991.','question':'Who created Python?','answer':'Guido van Rossum'},\n"
    "    {'context':'LoRA adds trainable low-rank matrices to frozen transformer layers.','question':'What does LoRA add?','answer':'trainable low-rank matrices'},\n"
    "    {'context':'The Amazon River flows through Brazil, Peru, and Colombia.','question':'Which countries?','answer':'Brazil, Peru, and Colombia'},\n"
    "    {'context':'BERT stands for Bidirectional Encoder Representations from Transformers.','question':'What does BERT stand for?','answer':'Bidirectional Encoder Representations from Transformers'},\n"
    "]\n"
    "\n"
    "def mkprompt(ctx, q): return f'### Context:\\n{ctx}\\n\\n### Question:\\n{q}\\n\\n### Answer:\\n'\n"
    "def infer(model, tok, prompts, max_new=40):\n"
    "    model.eval(); preds = []; t0 = time.perf_counter()\n"
    "    for p in prompts:\n"
    "        inp = tok(p, return_tensors='pt').to(device)\n"
    "        with torch.no_grad():\n"
    "            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,\n"
    "                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)\n"
    "        pred = tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)\n"
    "        preds.append(pred.split('\\n')[0].strip())\n"
    "    return preds, (time.perf_counter()-t0)*1000\n"
    "\n"
    "prompts = [mkprompt(e['context'], e['question']) for e in EXAMPLES]\n"
    "answers = [e['answer'] for e in EXAMPLES]\n"
    "\n"
    "print('Loading Qwen1.5-0.5B...')\n"
    "tok = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B', trust_remote_code=True)\n"
    "if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
    "base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B', trust_remote_code=True,\n"
    "    torch_dtype=torch.bfloat16, device_map=device)\n"
    "base.config.use_cache = False\n"
    "\n"
    "print('\\n1/3 Base SLM zero-shot...')\n"
    "base_preds, base_lat = infer(base, tok, prompts)\n"
    "base_m = compute_qa_metrics(base_preds, answers)\n"
    "\n"
    "print('2/3 Attach LoRA + fine-tune 10 steps...')\n"
    "lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,\n"
    "    target_modules=['q_proj','v_proj'], bias='none', task_type='CAUSAL_LM')\n"
    "pm = get_peft_model(base, lora_cfg)\n"
    "tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)\n"
    "tot = sum(p.numel() for p in pm.parameters())\n"
    "print(f'   LoRA: {tr:,} trainable / {tot:,} total ({100*tr/tot:.3f}%)')\n"
    "opt = optim.AdamW(pm.parameters(), lr=2e-4)\n"
    "pm.train()\n"
    "full_texts = [p+a for p,a in zip(prompts,answers)]\n"
    "for step in range(10):\n"
    "    ls=0.0\n"
    "    for text in full_texts:\n"
    "        inp=tok(text,return_tensors='pt',truncation=True,max_length=256).to(device)\n"
    "        out=pm(**inp,labels=inp['input_ids'].clone()); out.loss.backward(); ls+=out.loss.item()\n"
    "    opt.step(); opt.zero_grad()\n"
    "    if (step+1)%5==0: print(f'   step {step+1}/10 | loss: {ls/len(full_texts):.4f}')\n"
    "\n"
    "print('3/3 PEFT-SLM inference...')\n"
    "peft_preds, peft_lat = infer(pm, tok, prompts)\n"
    "peft_m = compute_qa_metrics(peft_preds, answers)\n"
    "\n"
    "print('\\n'+'='*65)\n"
    "print('  QUICK DEMO  |  5 examples  |  10 LoRA steps')\n"
    "print('='*65)\n"
    "print(f\"  {'Model/Method':<36} {'EM':>5} {'F1':>5} {'ms/ex':>7}\")\n"
    "print('-'*65)\n"
    "print(f\"  {'SLM base (zero-shot)':<36} {base_m['exact_match']:>5.1f} {base_m['f1']:>5.1f} {base_lat/5:>7.0f}\")\n"
    "print(f\"  {'SLM + LoRA (PEFT) <- OURS':<36} {peft_m['exact_match']:>5.1f} {peft_m['f1']:>5.1f} {peft_lat/5:>7.0f}\")\n"
    "print('='*65)\n"
    "for i,ex in enumerate(EXAMPLES):\n"
    "    print(f\"  [{i+1}] Ref : {ex['answer']}\")\n"
    "    print(f\"       Base: {base_preds[i][:70]}\")\n"
    "    print(f\"       PEFT: {peft_preds[i][:70]}\")\n"
    "del base, pm; torch.cuda.empty_cache()\n"
    "print('\\n✅ Demo complete')"
))

def experiment_cell(label, cfg_path, base_only=True, is_lora=False, task_label=""):
    if is_lora:
        return code_cell(
            f"# ── {label} ─────────────────────────────────────────────────────────────\n"
            "from src.training.trainer import run_training\n"
            f"print('{'='*60}'); print('{label}'); print('{'='*60}')\n"
            f"metrics = run_training('{cfg_path}')\n"
            "print('\\n📊 Results:')\n"
            "for k,v in metrics.items():\n"
            "    if not k.startswith('_'): print(f'  {k}: {v}')\n"
            f"print('\\n✅ {label} complete')"
        )
    exp_name_suffix = "_base"
    return code_cell(
        f"# ── {label} ─────────────────────────────────────────────────────────────\n"
        "from src.utils.config_loader import load_config\n"
        "from src.training.model_loader import load_tokenizer, load_base_model\n"
        "from src.data.dataset_loader import load_dataset_for_task\n"
        "from src.evaluation.evaluator import evaluate_model\n"
        "from src.utils.experiment_tracker import ExperimentTracker\n"
        "import torch\n"
        f"print('{'='*60}'); print('{label}'); print('{'='*60}')\n"
        f"cfg = load_config('{cfg_path}')\n"
        "tok = load_tokenizer(cfg); model = load_base_model(cfg); model.eval()\n"
        "ds = load_dataset_for_task(cfg, tok)\n"
        "print(f'Test split: {len(ds[\"test\"])} examples')\n"
        "metrics = evaluate_model(model, tok, ds['test'], cfg)\n"
        "print('\\n📊 Results:')\n"
        "for k,v in metrics.items():\n"
        "    if not k.startswith('_'): print(f'  {k}: {v}')\n"
        "ExperimentTracker(cfg['results_csv']).log({\n"
        f"    'experiment': cfg['experiment_name']+'{exp_name_suffix}',\n"
        "    'model': cfg['model_name_or_path'], 'model_variant': 'base',\n"
        "    'dataset': cfg['dataset']['name'], 'task': cfg['task'],\n"
        "    'method': 'base_zero_shot', 'seed': 42,\n"
        "    'test_samples': len(ds['test']),\n"
        "    **{k:v for k,v in metrics.items() if not k.startswith('_')},\n"
        "})\n"
        "del model; torch.cuda.empty_cache()\n"
        f"print('\\n✅ {label} complete')"
    )

cells.append(experiment_cell("Cell 6A: Base SLM zero-shot — SQuAD v2 (~15-20 min)",
                              "configs/pilot_squad_qa.yaml"))
cells.append(code_cell(
    "# ── Cell 6B: LoRA fine-tuning — SQuAD v2 (~90 min) ← YOUR RESULT ─────────\n"
    "# OOM? Run Cell 6B-OOM below then re-run this cell.\n"
    "import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
    "from src.training.trainer import run_training\n"
    "print('='*60)\n"
    "print('Cell 6B: LoRA fine-tuning | SQuAD v2 | ~90 min')\n"
    "print('Watch: loss decreasing. Adapter saved to experiments/squad_qa/adapter/')\n"
    "print('='*60)\n"
    "metrics = run_training('configs/pilot_squad_qa.yaml')\n"
    "print('\\n📊 LoRA Results:')\n"
    "for k,v in metrics.items():\n"
    "    if not k.startswith('_'): print(f'  {k}: {v}')\n"
    "print('\\n✅ Cell 6B complete')"
))
cells.append(code_cell(
    "# ── Cell 6B-OOM: Switch to 4-bit QLoRA — ONLY if Cell 6B gave OOM ─────────\n"
    "import yaml, pathlib\n"
    "p = pathlib.Path('/content/Mtech/configs/pilot_squad_qa.yaml')\n"
    "d = yaml.safe_load(p.read_text()); d['load_in_bits'] = 4\n"
    "p.write_text(yaml.dump(d, default_flow_style=False))\n"
    "print('✅ Switched to QLoRA 4-bit — re-run Cell 6B')"
))
cells.append(experiment_cell("Cell 6C: LLM baseline — Qwen1.5-7B zero-shot (~25 min)",
                              "configs/llm_baseline_qwen7b.yaml"))
cells.append(code_cell(
    "# ── Cell 6D: 3-Way QA Results Table + Chart ──────────────────────────────\n"
    "import pandas as pd, matplotlib.pyplot as plt\n"
    "print('='*70)\n"
    "print('3-WAY COMPARISON — SQuAD v2 QA — 500 test examples — seed=42')\n"
    "print('='*70)\n"
    "ExperimentTracker('experiments/results.csv').print_summary()\n"
    "try:\n"
    "    df = pd.read_csv('experiments/results.csv')\n"
    "    qa = df[df['task']=='qa'].dropna(subset=['exact_match','f1'])\n"
    "    if len(qa) >= 2:\n"
    "        fig, axes = plt.subplots(1,2,figsize=(11,4))\n"
    "        fig.suptitle('SQuAD v2 — 3-Way Comparison', fontweight='bold')\n"
    "        colors = ['#6b7280','#3b82f6','#7c3aed']\n"
    "        labels = qa['method'].tolist()\n"
    "        for ax,metric,title in zip(axes,['exact_match','f1'],['Exact Match (%)','F1 (%)']):\n"
    "            ax.bar(labels, qa[metric].tolist(), color=colors[:len(labels)])\n"
    "            ax.set_title(title); ax.set_ylabel('%')\n"
    "            ax.tick_params(axis='x', rotation=15)\n"
    "        plt.tight_layout()\n"
    "        plt.savefig('experiments/qa_results.png', dpi=150, bbox_inches='tight')\n"
    "        plt.show()\n"
    "        print('\\n✅ Chart saved to experiments/qa_results.png')\n"
    "except Exception as e:\n"
    "    print(f'Chart skipped: {e}')"
))
cells.append(experiment_cell("Cell 7A: Base SLM zero-shot — XL-Sum summarization (~15-20 min)",
                              "configs/pilot_xlsum_summarization.yaml"))
cells.append(code_cell(
    "# ── Cell 7B: LoRA fine-tuning — XL-Sum (~2 hours) ────────────────────────\n"
    "from src.training.trainer import run_training\n"
    "print('='*60)\n"
    "print('Cell 7B: LoRA fine-tuning | XL-Sum | ~2 hours')\n"
    "print('='*60)\n"
    "metrics = run_training('configs/pilot_xlsum_summarization.yaml')\n"
    "print('\\n📊 XL-Sum LoRA Results:')\n"
    "for k,v in metrics.items():\n"
    "    if not k.startswith('_'): print(f'  {k}: {v}')\n"
    "print('\\n✅ Cell 7B complete')"
))
cells.append(code_cell(
    "# ── Cell 8: View all results + download ──────────────────────────────────\n"
    "import pandas as pd, os\n"
    "from google.colab import files\n"
    "print('='*70)\n"
    "print('FULL EXPERIMENT RESULTS')\n"
    "print('='*70)\n"
    "ExperimentTracker('experiments/results.csv').print_summary()\n"
    "try: display(pd.read_csv('experiments/results.csv'))\n"
    "except: pass\n"
    "if os.path.exists('experiments/results.csv'):\n"
    "    files.download('experiments/results.csv')\n"
    "    print('\\n✅ Downloading results.csv')\n"
    "else:\n"
    "    print('\\n❌ No results yet — run experiment cells first')"
))
cells.append(md_cell(
    "## 💡 Troubleshooting\n\n"
    "| Problem | Fix |\n"
    "|---|---|\n"
    "| No GPU | Runtime → Change runtime type → T4 GPU |\n"
    "| `IndentationError` | Re-run **Cell 2** |\n"
    "| `ModuleNotFoundError` | Re-run **Cell 3** |\n"
    "| OOM during 0.5B training | Run **Cell 6B-OOM**, then re-run 6B |\n"
    "| OOM during 7B inference | Already handled — config has `load_in_bits: 4` |\n"
    "| Session expired | Run **Cell 8** to download results.csv before it expires |\n\n"
    "### Expected approximate results\n"
    "```\n"
    "SQuAD v2 QA:\n"
    "  Base SLM (0.5B zero-shot) : EM  5-15%,  F1 15-25%\n"
    "  LoRA fine-tuned   (ours)  : EM 40-55%,  F1 55-70%\n"
    "  LLM  (7B  zero-shot)      : EM 20-35%,  F1 35-50%\n\n"
    "XL-Sum Summarization:\n"
    "  Base SLM (zero-shot)      : ROUGE-1  5-15%\n"
    "  LoRA fine-tuned   (ours)  : ROUGE-1 20-35%\n"
    "```"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
        "colab": {"provenance":[],"gpuType":"T4"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = pathlib.Path("notebooks/05_colab_ready.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
size_kb = out.stat().st_size // 1024
print(f"Written: {out}  ({size_kb} KB,  {len(cells)} cells)")
