#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"

echo "[1/4] Ensuring virtual environment at $VENV_DIR"
if [[ -L "$VENV_DIR" ]]; then
  rm "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" || ! -f "$VENV_DIR/bin/activate" ]]; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[2/4] Installing dependencies"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "[3/4] Ensuring .env exists"
if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "Created .env from .env.example"
else
  echo ".env already exists, keeping current file"
fi

echo "[4/4] Setup complete"
cat <<'EOF'

Next steps:
1) Generate dev TLS certs:
   ./scripts/generate_dev_certs.sh

2) Bootstrap auth users:
   python -m scripts.manage_users init --admin silver
   python -m scripts.manage_users add-user --username <name> --role user

3) Start Streamlit (HTTPS on port 9001):
   source venv/bin/activate
   streamlit run app.py

4) Open:
   https://localhost:9001
EOF
