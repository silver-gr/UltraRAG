# UltraRAG Quick Start

## Goal
Get UltraRAG running with HTTPS (`:9001`) and authentication enabled.

## Prerequisites
- Python 3.10+
- OpenSSL (for local dev cert generation)
- Obsidian vault path
- API keys (Voyage + Google recommended)

## 1) Setup

```bash
cd /Users/silver/Projects/UltraRAG
chmod +x setup.sh
./setup.sh
```

This script:
- Creates/uses `/Users/silver/Projects/UltraRAG/venv`
- Installs `requirements.txt`
- Creates `.env` from `.env.example` if missing

## 2) Configure `.env`

Set at minimum:

```bash
OBSIDIAN_VAULT_PATH=/Users/your-name/Documents/ObsidianVault
VOYAGE_API_KEY=pa-...
GOOGLE_API_KEY=AIza...
```

Auth and multi-user defaults are included:

```bash
ULTRARAG_AUTH_ENABLED=true
ULTRARAG_USERS_PATH=data/auth/users.json
ULTRARAG_SESSION_TIMEOUT_MINUTES=720
ULTRARAG_MAX_CONCURRENT_JOBS=1
ULTRARAG_MIN_SECONDS_BETWEEN_QUERIES=2
```

## 3) TLS certificates (required for default Streamlit config)

Expected files:
- `/Users/silver/Projects/UltraRAG/certs/frontend.pem`
- `/Users/silver/Projects/UltraRAG/certs/frontend-key.pem`

Generate local self-signed certs:

```bash
./scripts/generate_dev_certs.sh
```

## 4) Bootstrap authentication

Create the first admin user:

```bash
python -m scripts.manage_users init --admin silver
```

Optional additional user:

```bash
python -m scripts.manage_users add-user --username alice --role user
```

## 5) Start the app

```bash
source venv/bin/activate
streamlit run app.py
```

Open:
- `https://localhost:9001`

## First run notes
- Admin can initialize/index/reindex and access Settings/LLM Costs.
- Non-admin users are query-only.
- Query history and content-research exports are per-user under `data/users/<username>/`.

## Troubleshooting
- Missing cert files: run `./scripts/generate_dev_certs.sh`.
- Login fails: run `python -m scripts.manage_users list-users` and reset password if needed.
- `.env` missing keys: copy from `.env.example` and fill required values.
