"""Security regression tests for fail-closed API-key configuration."""

import pytest
from fastapi import HTTPException

from src.callintel.api.dependencies import verify_api_key
from src.callintel.core.config import settings


def test_auth_enabled_without_key_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_key", None)

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(x_api_key=None, authorization=None)

    assert exc_info.value.status_code == 503


def test_configured_api_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_key", "test-only-secret")

    assert verify_api_key(x_api_key="test-only-secret", authorization=None) == "test-only-secret"


def test_invalid_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_key", "test-only-secret")

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(x_api_key="wrong", authorization=None)

    assert exc_info.value.status_code == 401
