"""Adapters that preserve frozen competition behavior during migration."""

from .legacy_agent import LegacyAgentAdapter, create_legacy_runtime

__all__ = ["LegacyAgentAdapter", "create_legacy_runtime"]
