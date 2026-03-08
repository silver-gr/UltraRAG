"""Manage local UltraRAG users JSON file.

Usage examples:
  python -m scripts.manage_users init --admin silver
  python -m scripts.manage_users add-user --username alice --role user
  python -m scripts.manage_users set-password --username alice
  python -m scripts.manage_users set-role --username alice --role admin
  python -m scripts.manage_users list-users
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from auth import hash_password, load_users, save_users


VALID_ROLES = {"admin", "user"}


def users_path() -> Path:
    return Path(os.getenv("ULTRARAG_USERS_PATH", "data/auth/users.json"))


def _prompt_password_twice() -> str:
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if not password:
        raise ValueError("Password cannot be empty")
    if password != confirm:
        raise ValueError("Passwords do not match")
    return password


def _save(db: dict) -> None:
    path = users_path()
    save_users(path, db)
    print(f"Saved users DB: {path}")


def _find(db: dict, username: str) -> dict | None:
    for user in db.get("users", []):
        if user.get("username") == username:
            return user
    return None


def cmd_init(args: argparse.Namespace) -> None:
    path = users_path()
    if path.exists():
        db = load_users(path)
        if db.get("users"):
            print(f"Users DB already exists with {len(db['users'])} users: {path}")
            return
    else:
        db = {"version": 1, "users": []}

    username = args.admin
    if _find(db, username):
        print(f"User already exists: {username}")
        return

    password = _prompt_password_twice()
    db["users"].append({
        "username": username,
        "role": "admin",
        "password_hash": hash_password(password),
    })
    _save(db)
    print(f"Initialized users DB with admin: {username}")


def cmd_add_user(args: argparse.Namespace) -> None:
    role = args.role.lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")

    path = users_path()
    db = load_users(path)
    username = args.username
    if _find(db, username):
        raise ValueError(f"User already exists: {username}")

    password = _prompt_password_twice()
    db.setdefault("users", []).append({
        "username": username,
        "role": role,
        "password_hash": hash_password(password),
    })
    _save(db)
    print(f"Added user: {username} ({role})")


def cmd_set_password(args: argparse.Namespace) -> None:
    path = users_path()
    db = load_users(path)
    user = _find(db, args.username)
    if not user:
        raise ValueError(f"User not found: {args.username}")

    password = _prompt_password_twice()
    user["password_hash"] = hash_password(password)
    _save(db)
    print(f"Password updated for: {args.username}")


def cmd_set_role(args: argparse.Namespace) -> None:
    role = args.role.lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")

    path = users_path()
    db = load_users(path)
    user = _find(db, args.username)
    if not user:
        raise ValueError(f"User not found: {args.username}")

    user["role"] = role
    _save(db)
    print(f"Role updated: {args.username} -> {role}")


def cmd_list_users(args: argparse.Namespace) -> None:
    path = users_path()
    db = load_users(path)
    users = db.get("users", [])
    if not users:
        print(f"No users found in {path}")
        return

    for user in users:
        print(f"- {user.get('username')} ({user.get('role')})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage UltraRAG users")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create users DB with initial admin")
    p_init.add_argument("--admin", required=True, help="Initial admin username")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add-user", help="Add a user")
    p_add.add_argument("--username", required=True)
    p_add.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    p_add.set_defaults(func=cmd_add_user)

    p_pwd = sub.add_parser("set-password", help="Set user password")
    p_pwd.add_argument("--username", required=True)
    p_pwd.set_defaults(func=cmd_set_password)

    p_role = sub.add_parser("set-role", help="Set user role")
    p_role.add_argument("--username", required=True)
    p_role.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    p_role.set_defaults(func=cmd_set_role)

    p_list = sub.add_parser("list-users", help="List users")
    p_list.set_defaults(func=cmd_list_users)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
