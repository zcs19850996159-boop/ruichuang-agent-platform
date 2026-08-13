from __future__ import annotations

import pytest
from PIL import Image

from remote_media import (
    PublicPeerHTTPAdapter,
    RemoteMediaResolver,
    _PublicPeerConnectionMixin,
    validate_public_response_peer,
)


def gray(value: int) -> Image.Image:
    return Image.new("L", (32, 18), color=value)


def resolver() -> RemoteMediaResolver:
    item = object.__new__(RemoteMediaResolver)
    item.video_max_frames = 8
    item.video_change_threshold = 0.10
    item.video_dedup_threshold = 0.035
    item.video_min_clarity = 0.015
    return item


def candidate(timestamp: float, value: int, change: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "thumb": gray(value),
        "change": change,
        "clarity": 1.0,
        "score": change,
    }


def test_static_video_keeps_three_anchors() -> None:
    items = [candidate(float(index), 20) for index in range(11)]
    selected = resolver()._select_adaptive_candidates(
        items,
        anchor_targets=[1.0, 5.0, 9.0],
        duration=10.0,
    )
    assert [item["timestamp"] for item in selected] == [1.0, 5.0, 9.0]
    assert all(item["reason"] == "time_anchor" for item in selected)


def test_scene_changes_add_frames_and_remove_near_duplicates() -> None:
    items = [candidate(float(index), 20) for index in range(11)]
    items[3] = candidate(3.0, 230, 0.60)
    items[4] = candidate(3.2, 230, 0.55)
    items[7] = candidate(7.0, 120, 0.40)
    selected = resolver()._select_adaptive_candidates(
        items,
        anchor_targets=[1.0, 5.0, 9.0],
        duration=10.0,
    )
    assert [item["timestamp"] for item in selected] == [1.0, 3.0, 5.0, 7.0, 9.0]
    assert [item["reason"] for item in selected].count("scene_change") == 2


def test_difference_and_anchor_fractions() -> None:
    assert RemoteMediaResolver._anchor_fractions(3) == [0.1, 0.5, 0.9]
    assert RemoteMediaResolver._anchor_fractions(2) == [0.25, 0.75]
    assert RemoteMediaResolver._frame_difference(gray(0), gray(255)) == 1.0
    assert RemoteMediaResolver._frame_difference(gray(80), gray(80)) == 0.0


class FakeSocket:
    def __init__(self, address: str) -> None:
        self.address = address

    def getpeername(self) -> tuple[str, int]:
        return self.address, 443


class FakeResponse:
    def __init__(self, address: str) -> None:
        connection = type("Connection", (), {"sock": FakeSocket(address)})()
        self.raw = type("Raw", (), {"_connection": connection})()


class ReleasedSocketResponse:
    def __init__(self, connection: object) -> None:
        self.raw = type("Raw", (), {})()
        self.connection = connection


class FakeConnectBase:
    def __init__(self, address: str) -> None:
        self.sock = FakeSocket(address)
        self.closed = False

    def connect(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class ConnectHarness(_PublicPeerConnectionMixin, FakeConnectBase):
    pass


def test_connected_peer_must_still_be_public_after_dns_validation() -> None:
    validate_public_response_peer(FakeResponse("93.184.216.34"))
    with pytest.raises(ValueError, match="non_public_peer"):
        validate_public_response_peer(FakeResponse("127.0.0.1"))
    with pytest.raises(ValueError, match="non_public_peer"):
        validate_public_response_peer(FakeResponse("169.254.169.254"))


def test_released_redirect_socket_requires_validating_transport() -> None:
    validating = ReleasedSocketResponse(PublicPeerHTTPAdapter())
    validate_public_response_peer(validating, allow_validated_transport=True)

    unguarded = ReleasedSocketResponse(object())
    with pytest.raises(ValueError, match="peer_address_unavailable"):
        validate_public_response_peer(unguarded, allow_validated_transport=True)


def test_connection_guard_rejects_private_peer() -> None:
    public = ConnectHarness("93.184.216.34")
    public.connect()
    assert public.closed is False

    private = ConnectHarness("127.0.0.1")
    with pytest.raises(ValueError, match="non_public_peer"):
        private.connect()
    assert private.closed is True


def test_remote_media_session_ignores_environment_proxies() -> None:
    session = RemoteMediaResolver().session
    assert session.trust_env is False
    assert isinstance(session.get_adapter("https://example.com"), PublicPeerHTTPAdapter)


if __name__ == "__main__":
    test_static_video_keeps_three_anchors()
    test_scene_changes_add_frames_and_remove_near_duplicates()
    test_difference_and_anchor_fractions()
    print("adaptive remote media tests passed")
