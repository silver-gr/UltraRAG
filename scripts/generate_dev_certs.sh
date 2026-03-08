#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$ROOT_DIR/certs"
CERT_FILE="$CERT_DIR/frontend.pem"
KEY_FILE="$CERT_DIR/frontend-key.pem"

mkdir -p "$CERT_DIR"

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required but not found in PATH" >&2
  exit 1
fi

# Generate a localhost certificate with SAN entries for browser compatibility.
openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days 825 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1

chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "Generated: $CERT_FILE"
echo "Generated: $KEY_FILE"
