#!/usr/bin/env bash
# Full financial QA comparison on a rented GPU. Stops at the first failure, before
# spending time on later runs. Usage from the repo root, inside tmux: bash scripts/run_vast.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1   # Xet downloads have stalled on rented/Colab machines

if ! python -c "import sys; sys.exit(sys.version_info < (3, 11))"; then
  echo "Need Python 3.11+ (the pinned packages do not install on $(python --version)). Use a newer template image."
  exit 1
fi

free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 40 ]; then
  echo "Only ${free_gb} GB free, need 40 GB (the 7B model alone is ~15 GB). Rent with a 50 GB disk."
  exit 1
fi

pip install -q -r requirements.txt

if ! python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.zeros(1).cuda() + 1)"; then
  nvidia-smi || true
  echo "PyTorch cannot use this GPU. Destroy this instance and rent a different one."
  exit 1
fi

run() { python src/training/run_pilot.py "$@" 2>&1 | tee -a experiments/vast_run.log; }

echo "=== 1/4 Dry run (a few minutes) ==="
rm -rf experiments/dry_run
run --config configs/dry_run_fin10k.yaml
rm -rf experiments/dry_run

echo "=== 2/4 Base 0.5B ==="
run --config configs/pilot_fin10k_qa.yaml --base_only
echo "=== 3/4 0.5B + LoRA ==="
run --config configs/pilot_fin10k_qa.yaml
echo "=== 4/4 7B baseline (4-bit) ==="
run --config configs/llm_baseline_fin10k_qwen7b.yaml --base_only

tar czf fin10k_results.tgz --exclude='checkpoint-*' experiments/results.csv experiments/vast_run.log \
  experiments/fin10k experiments/fin10k_llm7b
echo "Done. Download fin10k_results.tgz, then DESTROY the instance."
