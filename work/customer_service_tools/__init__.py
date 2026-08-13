"""Versioned customer-service Tool API shared by REST, MCP and WorkBuddy."""

from .service import CustomerServiceToolService, create_tool_service

__all__ = ["CustomerServiceToolService", "create_tool_service"]
