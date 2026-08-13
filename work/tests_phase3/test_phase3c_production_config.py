from __future__ import annotations

import pytest

from cloud_control.production_config import (
    inspect_production_environment,
    validate_production_environment,
)


PRODUCTION_ENV = {
    "DEPLOYMENT_MODE": "production",
    "CLOUD_CONTROL_ENABLED": "1",
    "CONTROL_PLANE_REQUIRE_POSTGRES": "1",
    "CONTROL_PLANE_DATABASE_URL": "postgresql://db.example/ruichuang",
    "CONTROL_PLANE_TOKEN_PEPPER": "p" * 48,
    "CONTROL_PLANE_BOOTSTRAP_TOKEN": "b" * 48,
    "API_AUTH_REQUIRED": "1",
    "KAFU_API_TOKEN": "k" * 48,
    "API_CORS_ORIGINS": "https://support.example.com",
    "API_DEBUG": "0",
    "KNOWLEDGE_OBJECT_STORE_BACKEND": "s3",
    "KNOWLEDGE_OBJECT_STORE_BUCKET": "ruichuang-production",
    "REDIS_URL": "rediss://redis.example/0",
    "API_TRUST_PROXY_HEADERS": "1",
    "API_TRUSTED_PROXY_IPS": "10.0.0.10,10.0.0.11",
}


def apply_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in PRODUCTION_ENV.items():
        monkeypatch.setenv(key, value)


def test_development_mode_does_not_require_production_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEPLOYMENT_MODE", "development")
    report = inspect_production_environment()
    assert report.valid is True
    assert report.errors == ()


def test_complete_production_configuration_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_environment(monkeypatch)
    report = validate_production_environment()
    assert report.valid is True
    assert report.mode == "production"


def test_production_wildcard_cors_and_placeholder_secrets_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_environment(monkeypatch)
    monkeypatch.setenv("API_CORS_ORIGINS", "*")
    monkeypatch.setenv(
        "CONTROL_PLANE_TOKEN_PEPPER",
        "replace-with-at-least-32-random-bytes",
    )
    report = inspect_production_environment()
    assert report.valid is False
    assert "API_CORS_ORIGINS must be an explicit allowlist" in report.errors
    assert "CONTROL_PLANE_TOKEN_PEPPER is missing or unsafe" in report.errors
    with pytest.raises(RuntimeError, match="unsafe production configuration"):
        validate_production_environment()
