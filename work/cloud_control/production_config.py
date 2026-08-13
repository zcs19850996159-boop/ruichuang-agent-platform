from __future__ import annotations

import os
from dataclasses import dataclass


PLACEHOLDER_PARTS = (
    "replace-me",
    "replace-with",
    "changeme",
    "example",
)


def _value(name: str) -> str:
    return os.environ.get(name, "").strip()


def _is_secret_ready(name: str, minimum_bytes: int = 32) -> bool:
    value = _value(name)
    lowered = value.lower()
    return (
        len(value.encode("utf-8")) >= minimum_bytes
        and not any(part in lowered for part in PLACEHOLDER_PARTS)
    )


@dataclass(frozen=True, slots=True)
class ProductionConfigurationReport:
    mode: str
    valid: bool
    errors: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "valid": self.valid,
            "errors": list(self.errors),
        }


def inspect_production_environment() -> ProductionConfigurationReport:
    mode = _value("DEPLOYMENT_MODE").lower() or "development"
    if mode != "production":
        return ProductionConfigurationReport(mode=mode, valid=True, errors=())

    errors: list[str] = []
    database_url = _value("CONTROL_PLANE_DATABASE_URL")
    cors = {part.strip() for part in _value("API_CORS_ORIGINS").split(",") if part.strip()}
    trusted_proxies = {
        part.strip()
        for part in _value("API_TRUSTED_PROXY_IPS").split(",")
        if part.strip()
    }
    if _value("CLOUD_CONTROL_ENABLED") != "1":
        errors.append("CLOUD_CONTROL_ENABLED must be 1")
    if _value("CONTROL_PLANE_REQUIRE_POSTGRES") != "1":
        errors.append("CONTROL_PLANE_REQUIRE_POSTGRES must be 1")
    if not database_url.startswith(("postgresql://", "postgres://")):
        errors.append("CONTROL_PLANE_DATABASE_URL must use PostgreSQL")
    if not _is_secret_ready("CONTROL_PLANE_TOKEN_PEPPER"):
        errors.append("CONTROL_PLANE_TOKEN_PEPPER is missing or unsafe")
    if not _is_secret_ready("CONTROL_PLANE_BOOTSTRAP_TOKEN"):
        errors.append("CONTROL_PLANE_BOOTSTRAP_TOKEN is missing or unsafe")
    if _value("API_AUTH_REQUIRED") != "1":
        errors.append("API_AUTH_REQUIRED must be 1")
    if not _is_secret_ready("KAFU_API_TOKEN"):
        errors.append("KAFU_API_TOKEN is missing or unsafe")
    if not cors or "*" in cors:
        errors.append("API_CORS_ORIGINS must be an explicit allowlist")
    if _value("API_DEBUG") == "1":
        errors.append("API_DEBUG must be disabled")
    if _value("KNOWLEDGE_OBJECT_STORE_BACKEND").lower() != "s3":
        errors.append("KNOWLEDGE_OBJECT_STORE_BACKEND must be s3")
    if not _value("KNOWLEDGE_OBJECT_STORE_BUCKET"):
        errors.append("KNOWLEDGE_OBJECT_STORE_BUCKET is required")
    if not _value("REDIS_URL").startswith(("redis://", "rediss://")):
        errors.append("REDIS_URL is required for production coordination")
    if _value("API_TRUST_PROXY_HEADERS") != "1":
        errors.append("API_TRUST_PROXY_HEADERS must be 1 behind the ingress")
    if not trusted_proxies or "*" in trusted_proxies:
        errors.append("API_TRUSTED_PROXY_IPS must be an explicit allowlist")
    return ProductionConfigurationReport(
        mode=mode,
        valid=not errors,
        errors=tuple(errors),
    )


def validate_production_environment() -> ProductionConfigurationReport:
    report = inspect_production_environment()
    if not report.valid:
        raise RuntimeError(
            "unsafe production configuration: " + "; ".join(report.errors)
        )
    return report
