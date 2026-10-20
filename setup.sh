#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Installing dependencies with uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with one of:" >&2
  echo "  brew install uv" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
uv sync --project "$ROOT_DIR"

echo "[2/3] Ensuring .env exists"
if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "Created .env from .env.example"
else
  echo ".env already exists, keeping current file"
fi

echo "[3/3] Setup complete"
cat <<'EOF'

Next steps:
1) Generate dev TLS certs:
   ./scripts/generate_dev_certs.sh

2) Bootstrap auth users:
   uv run python -m scripts.manage_users init --admin silver
   uv run python -m scripts.manage_users add-user --username <name> --role user

3) Start Streamlit (HTTPS on port 9001):
   ./start.sh
   (or: uv run streamlit run app.py)

4) Open:
   https://localhost:9001
EOF
