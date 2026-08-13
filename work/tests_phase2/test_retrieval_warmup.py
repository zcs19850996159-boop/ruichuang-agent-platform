from __future__ import annotations

from build_own_evidence import EvidenceBuilder


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str):
        self.queries.append(query)
        return [1.0]


def test_retrieval_warmup_encodes_one_query() -> None:
    builder = EvidenceBuilder.__new__(EvidenceBuilder)
    builder.hybrid_retriever = FakeRetriever()
    builder.hybrid_status = {
        "device": "cpu",
        "embedding_model": "test-model",
    }

    result = builder.warmup("warm me")

    assert result["status"] == "ready"
    assert result["device"] == "cpu"
    assert builder.hybrid_retriever.queries == ["warm me"]


def test_retrieval_warmup_skips_without_hybrid_retriever() -> None:
    builder = EvidenceBuilder.__new__(EvidenceBuilder)
    builder.hybrid_retriever = None

    result = builder.warmup()

    assert result["status"] == "skipped"
