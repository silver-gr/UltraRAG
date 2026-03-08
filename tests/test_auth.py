from __future__ import annotations

import pytest

from auth import hash_password, verify_password


@pytest.mark.unit
def test_hash_password_round_trip_verifies() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password("correct horse battery staple", hashed)


@pytest.mark.unit
def test_verify_password_wrong_password_fails() -> None:
    hashed = hash_password("topsecret")
    assert not verify_password("wrong-password", hashed)


@pytest.mark.unit
def test_verify_password_malformed_hash_rejected() -> None:
    assert not verify_password("anything", "not-a-valid-hash")
    assert not verify_password("anything", "pbkdf2_sha256$bad$format")
