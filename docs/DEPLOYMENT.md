# Deployment Guide

UltraRAG runs Streamlit with HTTPS enabled by default on port `9001`.

## Streamlit TLS Expectations

`.streamlit/config.toml` is expected to reference:

- `sslCertFile = "certs/frontend.pem"`
- `sslKeyFile = "certs/frontend-key.pem"`
- `address = "0.0.0.0"`
- `port = 9001`

Create the `certs/` directory in repo root and place the files at those exact paths.

## Development (Self-Signed)

Use the built-in helper script:

```bash
cd /Users/silver/Projects/UltraRAG
./scripts/generate_dev_certs.sh
```

Then run:

```bash
source venv/bin/activate
streamlit run app.py
```

Open: `https://localhost:9001`

## VPS / Production Certificates

Use your CA-issued certs (for example Let's Encrypt) and copy/symlink them into:

- `/Users/silver/Projects/UltraRAG/certs/frontend.pem`
- `/Users/silver/Projects/UltraRAG/certs/frontend-key.pem`

Example:

```bash
sudo cp /etc/letsencrypt/live/<domain>/fullchain.pem /Users/silver/Projects/UltraRAG/certs/frontend.pem
sudo cp /etc/letsencrypt/live/<domain>/privkey.pem /Users/silver/Projects/UltraRAG/certs/frontend-key.pem
sudo chown $USER /Users/silver/Projects/UltraRAG/certs/frontend.pem /Users/silver/Projects/UltraRAG/certs/frontend-key.pem
chmod 644 /Users/silver/Projects/UltraRAG/certs/frontend.pem
chmod 600 /Users/silver/Projects/UltraRAG/certs/frontend-key.pem
```

## Auth Bootstrap

Authentication is controlled with:

- `ULTRARAG_AUTH_ENABLED=true`
- `ULTRARAG_USERS_PATH=data/auth/users.json`

Create the initial admin account:

```bash
python -m scripts.manage_users init --admin silver
```

Then add user accounts:

```bash
python -m scripts.manage_users add-user --username <name> --role user
```

## Optional Reverse Proxy

You can place Nginx/Caddy in front for:

- IP allow-lists / SSO
- Additional security headers
- Certificate automation

Even with a reverse proxy, this project currently expects TLS files for Streamlit at the paths above (TLS termination remains in Streamlit by default).
