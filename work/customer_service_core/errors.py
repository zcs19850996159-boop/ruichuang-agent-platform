from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INPUT_INVALID = "input_invalid"
    PRODUCT_UNCLEAR = "product_unclear"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    KNOWLEDGE_NOT_FOUND = "knowledge_not_found"
    MODEL_TIMEOUT = "model_timeout"
    VISION_FAILED = "vision_failed"
    INDEX_UNAVAILABLE = "index_unavailable"
    ANSWER_VALIDATION_FAILED = "answer_validation_failed"
    PERMISSION_DENIED = "permission_denied"
    INTERNAL_ERROR = "internal_error"


class RefusalType(StrEnum):
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    PRODUCT_UNCLEAR = "product_unclear"
    IMAGE_UNCERTAIN = "image_uncertain"
    VIDEO_UNCERTAIN = "video_uncertain"
    KNOWLEDGE_NOT_COVERED = "knowledge_not_covered"
    REALTIME_DATA_NOT_CONNECTED = "realtime_data_not_connected"


def normalize_refusal_payload(result: dict[str, Any]) -> dict[str, Any]:
    route = result.get("route") or {}
    reason = str(route.get("reason") or "")
    refusal_type = str(result.get("refusal_type") or "")
    if not refusal_type and reason == "unsupported_realtime_query":
        refusal_type = RefusalType.REALTIME_DATA_NOT_CONNECTED.value
    if not refusal_type and reason in {"knowledge_not_found", "knowledge_not_covered"}:
        refusal_type = RefusalType.KNOWLEDGE_NOT_COVERED.value
    allowed = {item.value for item in RefusalType}
    if refusal_type and refusal_type not in allowed:
        refusal_type = RefusalType.EVIDENCE_INSUFFICIENT.value
    result["refusal_type"] = refusal_type
    result["refusal"] = {
        "active": bool(refusal_type),
        "type": refusal_type,
        "reason": reason,
        "requires_clarification": refusal_type
        in {
            RefusalType.EVIDENCE_INSUFFICIENT.value,
            RefusalType.PRODUCT_UNCLEAR.value,
            RefusalType.IMAGE_UNCERTAIN.value,
            RefusalType.VIDEO_UNCERTAIN.value,
        },
        "escalation_recommended": refusal_type
        in {
            RefusalType.KNOWLEDGE_NOT_COVERED.value,
            RefusalType.REALTIME_DATA_NOT_CONNECTED.value,
        },
    }
    return result


class CustomerServiceError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        http_status: int = 500,
        retryable: bool = False,
        escalation_required: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.escalation_required = escalation_required
        self.details = details or {}

    def public_payload(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "escalation_required": self.escalation_required,
        }
