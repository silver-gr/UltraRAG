#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CERT_FILE="$ROOT_DIR/certs/frontend.pem"
KEY_FILE="$ROOT_DIR/certs/frontend-key.pem"
SERVER_ADDRESS="${ULTRARAG_SERVER_ADDRESS:-0.0.0.0}"
BROWSER_HOST="${ULTRARAG_BROWSER_HOST:-localhost}"
SERVER_PORT="${ULTRARAG_SERVER_PORT:-9001}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed (brew install uv)" >&2
  echo "Then run: ./setup.sh" >&2
  exit 1
fi

if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
  echo "TLS certs not found under: $ROOT_DIR/certs" >&2
  echo "Generating local self-signed dev certs..." >&2
  bash "$ROOT_DIR/scripts/generate_dev_certs.sh"
fi

exec uv run --project "$ROOT_DIR" streamlit run "$ROOT_DIR/app.py" \
  --server.address "$SERVER_ADDRESS" \
  --server.port "$SERVER_PORT" \
  --browser.serverAddress "$BROWSER_HOST"
