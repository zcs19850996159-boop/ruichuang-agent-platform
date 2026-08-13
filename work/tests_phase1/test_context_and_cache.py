from __future__ import annotations

import unittest

from customer_service_core.application import CustomerServiceApplication
from customer_service_core.cache import versioned_cache_key
from customer_service_core.context import RequestContextFactory, bind_request_context
from customer_service_core.errors import CustomerServiceError, ErrorCode
from customer_service_core.profiles import CompetitionPatchRegistry


class FakeLegacyService:
    def answer(self, payload, stream_callback=None):
        return {"answer": payload["question"], "images": []}


class ContextAndCacheTests(unittest.TestCase):
    def make_context(self, knowledge_version: str, profile: str = "competition"):
        return RequestContextFactory.from_request(
            payload={"question": "test"},
            headers={
                "x-request-id": "req-test",
                "x-trace-id": "trace-test",
                "x-profile": profile,
                "x-tenant-id": "tenant-a",
                "x-knowledge-space-id": "space-a",
                "x-knowledge-version": knowledge_version,
            },
        )

    def test_cache_key_changes_with_knowledge_version(self):
        with bind_request_context(self.make_context("knowledge-v1")):
            first = versioned_cache_key("answer", {"question": "same"})
        with bind_request_context(self.make_context("knowledge-v2")):
            second = versioned_cache_key("answer", {"question": "same"})
        self.assertNotEqual(first, second)

    def test_application_attaches_context_metadata(self):
        service = CustomerServiceApplication(
            legacy_service=FakeLegacyService(),
            patch_registry=CompetitionPatchRegistry(),
        )
        result = service.answer(
            {"question": "hello"},
            context=self.make_context("knowledge-v1"),
        )
        self.assertEqual(result["answer"], "hello")
        self.assertEqual(result["request_context"]["knowledge_version"], "knowledge-v1")
        self.assertEqual(result["trace_id"], "trace-test")

    def test_enterprise_cannot_force_competition_patch(self):
        service = CustomerServiceApplication(
            legacy_service=FakeLegacyService(),
            patch_registry=CompetitionPatchRegistry(),
        )
        with self.assertRaises(CustomerServiceError) as caught:
            service.answer(
                {"question": "hello", "_force_competition_patch": True},
                context=self.make_context("knowledge-v1", profile="enterprise"),
            )
        self.assertEqual(caught.exception.code, ErrorCode.PERMISSION_DENIED)


if __name__ == "__main__":
    unittest.main()

