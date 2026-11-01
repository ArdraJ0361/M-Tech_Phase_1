#!/usr/bin/env bash
# Install everything on first run, then host the fine-tuned model on localhost.
# Run from anywhere:  bash scripts/start_server.sh     (PORT=9000 picks another start port)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f experiments/fin10k/adapter/adapter_model.safetensors ]; then
  echo "ERROR: experiments/fin10k/adapter is missing. Run: git pull" >&2
  exit 1
fi

case "$PWD" in
  /mnt/*|*OneDrive*)
    echo "WARNING: this folder is on Windows or OneDrive, which makes the install slow and syncs several GB." >&2
    echo "         Better: run 'cd ~' in Ubuntu and clone the project there." >&2
    ;;
esac

if [ ! -x .venv/bin/python ]; then
  echo "==> Creating Python environment"
  if command -v uv >/dev/null; then
    uv venv --seed --python 3.12 .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  elif command -v python3.11 >/dev/null; then
    python3.11 -m venv .venv
  else
    echo "ERROR: Python 3.12 is required. Install it from https://www.python.org/downloads/ and run this again." >&2
    exit 1
  fi
fi

if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "==> Installing packages (first run only, can take several minutes)"
  if command -v uv >/dev/null; then
    INSTALL=(uv pip install --python .venv/bin/python)
  else
    .venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade
    INSTALL=(.venv/bin/python -m pip install)
  fi
  # Without an NVIDIA GPU, skip the several GB of CUDA packages torch pulls in on Linux.
  if [ "$(uname)" = Linux ] && ! command -v nvidia-smi >/dev/null; then
    "${INSTALL[@]}" "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu \
      || echo "==> CPU-only torch not available, using the default package"
  fi
  "${INSTALL[@]}" -r requirements.txt
  touch .venv/.installed
fi

PORT="${PORT:-8765}"
port_free() { .venv/bin/python -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$1))==0)"; }
until port_free "$PORT"; do PORT=$((PORT + 1)); done

cat <<EOF

==> Starting the model. The first start downloads it (1.2 GB), which can take several minutes.
    Ready when you see: Uvicorn running on http://127.0.0.1:$PORT
    Try it in a browser: http://127.0.0.1:$PORT/docs
    Stop it: press Ctrl+C

EOF
exec .venv/bin/python -m src.serving.server --config configs/pilot_fin10k_qa.yaml --port "$PORT"
