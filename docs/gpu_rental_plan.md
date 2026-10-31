# GPU Rental Plan — Cheap Training for PEFT-SLM

**Goal:** rent the cheapest GPU, train the LoRA adapter (~21 min), download
it, destroy the instance. Total cost ~₹7.

---

## Do You Need a GPU?

| Task | GPU needed? |
|---|---|
| LoRA fine-tune (3 epochs, 5k examples) | ✅ Yes — ~6 GB VRAM minimum |
| Run the endpoint / answer questions | ❌ No — CPU works |
| Dry run to check everything works | ❌ No — CPU fine for 64 examples |

Your peak GPU memory from Phase 1B: **4,978 MB (~5 GB)**. Any GPU with
≥ 6 GB VRAM is sufficient.

---

## Free Options (₹0)

| Platform | GPU | VRAM | Free Limit | Notes |
|---|---|---|---|---|
| **Kaggle Notebooks** | T4 / P100 | 16 GB | 30 h/week | ✅ Best free option — no disconnects |
| **Google Colab Free** | T4 | 16 GB | ~3-4 h/day | Already proven for Phase 1B |
| **Lightning.ai** | T4 | 16 GB | 22 GPU-h/month | Good backup |

**Recommended free path:** Kaggle — 30 free GPU hours/week, doesn't disconnect
mid-run, upload `simple_pipeline.ipynb`, run all, download `adapter/`.

---

## Paid Options — India (₹ pricing)

| Platform | GPU | Price | Your run cost (~30 min) | Notes |
|---|---|---|---|---|
| **Vast.ai** | RTX 3090 | ~₹14–20/h | ~₹7–10 | Cheapest per hour globally |
| **Jarvislabs.ai** | RTX 5000 | ₹15–25/h | ~₹12 | Indian platform, UPI accepted |
| **Jarvislabs.ai** | A100 | ₹60/h | ~₹30 | Fastest option |
| **E2E Networks** | V100 | ₹35–45/h | ~₹22 | Indian company, data in India |
| **RunPod** | RTX 3090 | ₹18–25/h | ~₹12 | Reliable, easy UI |

---

## Recommended Vast.ai Listing (as of 2026-10-31)

```
Instance:   m:153150  (host:483833, China, CN)
GPU:        1x RTX 3090
VRAM:       24 GB
Price:      $0.166/hr  (~₹14/hr)
Reliability: 99.02%
Max duration: 12 months

Your cost:  $0.166 × 0.5 h = ~$0.08  (~₹7)
```

If China upload/download speed is slow, use:
```
Instance:   m:84216  (host:443329, California, US)
GPU:        1x RTX 3090
VRAM:       24 GB
Price:      $0.193/hr
Reliability: 99.30%
Max duration: 5 months
```

---

## Step-by-Step: Vast.ai Training Run

### 1. Rent the instance
- Go to vast.ai → search RTX 3090
- Click **RENT** on the cheapest verified listing
- Template: **PyTorch** (pre-installed)
- Disk: **16 GB**

### 2. In the terminal — install deps

```bash
pip install -q peft bitsandbytes transformers datasets accelerate
export HF_HUB_DISABLE_XET=1
```

### 3. Upload your code

```bash
# Option A — upload simple_pipeline.ipynb via file browser

# Option B — clone from GitHub
git clone https://github.com/YOUR_USERNAME/YOUR_REPO
cd my-project
pip install -r requirements.txt
```

### 4. Run training

```bash
# Notebook route
jupyter nbconvert --to notebook --execute notebooks/simple_pipeline.ipynb

# Script route (cleaner)
python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml
```

### 5. Download the adapter

```bash
zip -r adapter.zip experiments/simple/adapter/
# Download via Vast.ai file browser or scp
```

### 6. Destroy the instance immediately

- Vast.ai → My Instances → **DESTROY**
- Billing stops the moment you destroy

---

## After Training — No GPU Needed

```
YOUR LAPTOP (CPU only, forever after)
├── adapter/         ← downloaded from Vast/Kaggle (~3-5 MB)
├── Qwen1.5-0.5B     ← auto-downloads from HuggingFace on first run (~1 GB)
└── pip install -r requirements.txt

→ python -m src.serving.server --config configs/pilot_squad_qa.yaml
→ Open localhost:8000/docs → type question → get answer (~1-2 sec on CPU)
```

---

## Total Cost Estimate — Full Phase 1

| What | Platform | Cost |
|---|---|---|
| SQuAD QA training run | Kaggle free | ₹0 |
| XL-Sum summarization run | Kaggle free | ₹0 |
| If Kaggle quota runs out | Vast.ai RTX 3090 | ~₹7/run |
| Entire Phase 1 (2 runs) | Vast.ai worst case | ~₹15 |

---

## Key Facts (from Phase 1B results)

```
Training time:    21 min  (Colab T4, 3 epochs, 5000 examples)
Peak GPU memory:  4,978 MB
Adapter size:     ~3-5 MB  (only LoRA weights, not the full model)
Minimum VRAM:     6 GB
```
