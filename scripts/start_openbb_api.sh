#!/usr/bin/env bash
# Start OpenBB Platform API with HTTPS for OpenBB Workspace (avoids mixed-content errors).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SSL_CERT_FILE="$("$ROOT/.venv/bin/python" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

CERT_DIR="$ROOT/data/openbb-ssl"
if [[ ! -f "$CERT_DIR/localhost.crt" || ! -f "$CERT_DIR/localhost.key" ]]; then
  echo "Missing TLS certs in $CERT_DIR — generate with:"
  echo "  mkdir -p $CERT_DIR"
  echo "  openssl req -x509 -newkey rsa:4096 -keyout $CERT_DIR/localhost.key -out $CERT_DIR/localhost.crt -days 365 -nodes -subj '/CN=127.0.0.1'"
  exit 1
fi

exec "$ROOT/.venv/bin/uvicorn" openbb_platform_api.main:app \
  --host 127.0.0.1 \
  --port 6900 \
  --ssl-keyfile "$CERT_DIR/localhost.key" \
  --ssl-certfile "$CERT_DIR/localhost.crt"
