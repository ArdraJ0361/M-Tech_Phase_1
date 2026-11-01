# Training Errors Log — Vast.ai Session 2026-11-01

Instance: 58643084 | GPU: RTX 5060 Ti 16GB | Cost: ~$0.028 (~₹2.3)

---

## Error 1 — Wrong workspace path

**Command run:**
```bash
cd /workspaces
```

**Error:**
```
bash: cd: /workspaces: No such file or directory
```

**Cause:** Vast.ai PyTorch template uses `/workspace` (no 's').

**Fix:**
```bash
cd /workspace
```

---

## Error 2 — Zip extracted into nested folder

**Command run:**
```bash
cd /workspace/my-project
pip install -r requirements.txt
python src/training/run_pilot.py ...
```

**Error:**
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
python: can't open file '.../run_pilot.py': [Errno 2] No such file or directory
```

**Cause:** Mac's `git archive` created `my-project.zip` with an inner `my-project/` folder.
Unzip created: `/workspace/my-project/my-project/` (double nested).

**Fix:**
```bash
cd /workspace/my-project/my-project
pip install -r requirements.txt
python src/training/run_pilot.py --config configs/pilot_docfinqa.yaml
```

**Permanent fix for next time:** Create zip without the outer folder:
```bash
cd /Users/joelvarghese/Documents/my-project
zip -r my-project.zip . --exclude "*.git*" --exclude "__pycache__*"
```
This puts files at root of zip — unzips cleanly into target folder.

---

## Error 3 — DocFinQA PyArrow OverflowError (MAIN BLOCKER)

**Command run:**
```bash
python src/training/run_pilot.py --config configs/pilot_docfinqa.yaml
```

**Error:**
```
File "pyarrow/_json.pyx", line 79, in pyarrow._json.ReadOptions.block_size.__set__
OverflowError: value too large to convert to int32_t

datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset
```

**Root cause:**
- DocFinQA `train.json` = **3.62 GB**
- PyArrow JSON reader has an internal `int32` block size limit = max ~2.1 GB
- Any JSON file > 2.1 GB fails regardless of PyArrow version
- This is a C++ level bug in PyArrow's JSON reader — upgrading PyArrow does NOT fix it

**Attempted fixes that did NOT work:**
```bash
pip install -q "pyarrow>=14.0.0"   # ❌ same error
pip install -q "pyarrow>=16.0.0"   # ❌ same error
```

**Correct fix (to implement on laptop before next GPU session):**

Option A — Load DocFinQA via `trust_remote_code=True` to use its custom loader:
```python
load_dataset("kensho/DocFinQA", trust_remote_code=True)
```

Option B — Load the JSON files manually in chunks:
```python
import json
import datasets

def load_docfinqa_manually(split="train"):
    # Download the file first
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="kensho/DocFinQA",
        filename=f"{split}.json",
        repo_type="dataset"
    )
    with open(path) as f:
        data = json.load(f)  # json.load handles large files fine
    return datasets.Dataset.from_list(data)
```

Option C — Switch to FinanceBench (smaller, same financial QA domain):
```python
load_dataset("PatronusAI/financebench", split="train[:5000]")
```

---

## Error 4 — ibm/finqa dataset script error

**Command run:**
```bash
load_dataset('ibm/finqa', split='train[:5]')
```

**Error:**
```
RuntimeError: Dataset scripts are no longer supported, but found finqa.py
```

**Cause:** `ibm/finqa` uses an old-style dataset loading script (`.py` file) which
newer versions of `datasets` library no longer support.

**Fix:** Use a different financial QA dataset. Candidates to test:
- `TheFinAI/finqa` — check if it works
- `Linq-AI-Research/FinQA` — check if it works
- `PatronusAI/financebench` — smaller, modern format

---

## Error 5 — HuggingFaceH4/FinQA does not exist

**Error:**
```
DatasetNotFoundError: Dataset 'HuggingFaceH4/FinQA' doesn't exist on the Hub
```

**Fix:** Wrong dataset ID. Correct ones to try next session (see Error 4 above).

---

## Summary — What to Fix Before Next GPU Session

| Fix | File | Action |
|---|---|---|
| Zip without nesting | Terminal | Use `zip -r my-project.zip .` from inside project dir |
| DocFinQA loading | `src/data/dataset_loader.py` | Add `json.load()` fallback for large JSON files |
| Test correct FinQA dataset ID | Terminal | Check `TheFinAI/finqa` and `PatronusAI/financebench` locally |

---

## Time & Cost for This Session

```
Instance rented:    RTX 5060 Ti, $0.112/hr
Time running:       ~15 min
Cost:               ~$0.028  (~₹2.3)
Credit remaining:   ~$4.97
Training completed: ❌ blocked by DocFinQA PyArrow bug
```

---

## Next Session Plan

```
1. Fix zip creation on laptop (no nesting)
2. Fix dataset_loader.py to handle DocFinQA
3. Test on CPU locally (DRY_RUN=True) — confirm it works
4. Rent GPU again (~$0.056 for 30 min)
5. Upload fixed zip → training starts cleanly
```
