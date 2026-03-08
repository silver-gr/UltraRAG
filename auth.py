"""Simple in-app authentication and role helpers for Streamlit."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

PBKDF2_ITERATIONS = 390_000
AUTH_SESSION_KEY = "ultrarag_auth"


@dataclass(frozen=True)
class AuthContext:
    """Authenticated user context."""
    username: str
    role: str


def _auth_enabled() -> bool:
    return os.getenv("ULTRARAG_AUTH_ENABLED", "true").lower() == "true"


def _allow_signup() -> bool:
    return os.getenv("ULTRARAG_ALLOW_SIGNUP", "false").lower() == "true"


def _users_path() -> Path:
    return Path(os.getenv("ULTRARAG_USERS_PATH", "data/auth/users.json"))


def _session_timeout_seconds() -> int:
    minutes = int(os.getenv("ULTRARAG_SESSION_TIMEOUT_MINUTES", "720"))
    return max(1, minutes) * 60


def load_users(path: Path) -> dict:
    """Load users database from disk, returning empty structure if absent."""
    if not path.exists():
        return {"version": 1, "users": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Users DB must be a JSON object")

    users = data.get("users", [])
    if not isinstance(users, list):
        raise ValueError("Users DB 'users' must be a list")

    normalized = []
    for user in users:
        if not isinstance(user, dict):
            continue
        username = str(user.get("username", "")).strip()
        role = str(user.get("role", "user")).strip().lower()
        password_hash = str(user.get("password_hash", "")).strip()
        if not username or role not in {"admin", "user"} or not password_hash:
            continue
        normalized.append({
            "username": username,
            "role": role,
            "password_hash": password_hash,
        })

    return {"version": int(data.get("version", 1)), "users": normalized}


def save_users(path: Path, data: dict) -> None:
    """Persist users database to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def hash_password(plain: str) -> str:
    """Create PBKDF2-SHA256 password hash in portable string format."""
    if not plain:
        raise ValueError("Password cannot be empty")

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    hash_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${hash_b64}"


def verify_password(plain: str, stored: str) -> bool:
    """Verify a plaintext password against stored PBKDF2 hash."""
    try:
        algo, iterations_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False

        iterations = int(iterations_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        actual = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _find_user(users_db: dict, username: str) -> dict | None:
    for user in users_db.get("users", []):
        if user.get("username") == username:
            return user
    return None


def _clear_session() -> None:
    st.session_state.pop(AUTH_SESSION_KEY, None)


def logout() -> None:
    """Log out current user from Streamlit session."""
    _clear_session()


def _set_session(ctx: AuthContext) -> None:
    now = int(time.time())
    st.session_state[AUTH_SESSION_KEY] = {
        "username": ctx.username,
        "role": ctx.role,
        "issued_at": now,
        "expires_at": now + _session_timeout_seconds(),
    }


def _get_session_context() -> AuthContext | None:
    raw = st.session_state.get(AUTH_SESSION_KEY)
    if not isinstance(raw, dict):
        return None

    try:
        username = str(raw["username"])
        role = str(raw["role"]).lower()
        expires_at = int(raw["expires_at"])
    except Exception:
        _clear_session()
        return None

    if role not in {"admin", "user"}:
        _clear_session()
        return None

    if int(time.time()) >= expires_at:
        _clear_session()
        return None

    return AuthContext(username=username, role=role)


def login_ui() -> AuthContext | None:
    """Render login form and return context when session is authenticated."""
    session_ctx = _get_session_context()
    if session_ctx:
        return session_ctx

    users_path = _users_path()

    st.title("UltraRAG Login")
    st.caption("Sign in to access UltraRAG.")

    if not users_path.exists() and _auth_enabled():
        st.error(
            "Authentication is enabled but users DB is missing. "
            "Run: python -m scripts.manage_users init --admin <username>"
        )
        st.stop()

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        users_db = load_users(users_path)
        user = _find_user(users_db, username.strip())
        if not user or not verify_password(password, user.get("password_hash", "")):
            st.error("Invalid username or password")
            return None

        ctx = AuthContext(username=user["username"], role=user["role"])
        _set_session(ctx)
        st.rerun()

    if _allow_signup():
        st.caption("Signup is enabled by configuration but not implemented in-app.")

    return None


def require_auth() -> AuthContext:
    """Require auth when enabled, otherwise return local admin context."""
    if not _auth_enabled():
        return AuthContext(username="local", role="admin")

    ctx = _get_session_context()
    if ctx:
        return ctx

    ctx = login_ui()
    if ctx:
        return ctx

    st.stop()


def is_admin(ctx: AuthContext) -> bool:
    return ctx.role == "admin"


def require_admin(ctx: AuthContext) -> None:
    if not is_admin(ctx):
        st.error("Admin role required for this action.")
        st.stop()
