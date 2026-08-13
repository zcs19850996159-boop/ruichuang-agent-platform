from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image, ImageChops, ImageFilter, ImageStat
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool


URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
HTML_COVER_PATTERNS = (
    re.compile(r'"urlDefault"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
    re.compile(r'"cover"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
    re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', re.I),
)


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def strip_remote_urls(text: str) -> str:
    cleaned = URL_RE.sub(" ", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip(" ,，;；")


def _decode_json_string(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except Exception:
        return value.replace(r"\u002F", "/").replace(r"\/", "/")


def _is_public_host(hostname: str) -> bool:
    if not hostname or hostname.lower() in {"localhost", "localhost.localdomain"}:
        return False
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def validate_public_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "invalid_url"
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "unsupported_scheme"
    if parsed.username or parsed.password:
        return False, "credentials_in_url"
    if parsed.port not in {None, 80, 443}:
        return False, "non_standard_port"
    hostname = parsed.hostname or ""
    if not _is_public_host(hostname):
        return False, "non_public_host"
    return True, ""


def _validate_public_peer_address(address: str) -> None:
    try:
        peer = ipaddress.ip_address(str(address or "").split("%", 1)[0])
    except ValueError as exc:
        raise ValueError("invalid_peer_address") from exc
    if not peer.is_global:
        raise ValueError("non_public_peer")


class _PublicPeerConnectionMixin:
    """Reject a private peer immediately after TCP/TLS connection setup."""

    def connect(self) -> None:
        super().connect()
        candidate = getattr(self, "sock", None)
        try:
            peer = candidate.getpeername() if candidate is not None else None
            if not isinstance(peer, tuple) or not peer:
                raise ValueError("peer_address_unavailable")
            _validate_public_peer_address(str(peer[0]))
        except Exception:
            self.close()
            raise


class _PublicPeerHTTPConnection(_PublicPeerConnectionMixin, HTTPConnection):
    pass


class _PublicPeerHTTPSConnection(_PublicPeerConnectionMixin, HTTPSConnection):
    pass


class _PublicPeerHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PublicPeerHTTPConnection


class _PublicPeerHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PublicPeerHTTPSConnection


class _PublicPeerPoolManager(PoolManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = {
            "http": _PublicPeerHTTPConnectionPool,
            "https": _PublicPeerHTTPSConnectionPool,
        }


class PublicPeerHTTPAdapter(HTTPAdapter):
    """Requests adapter whose connections enforce public peer addresses."""

    peer_validation_enforced = True

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = _PublicPeerPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


def _response_peer_address(response: requests.Response) -> str:
    raw = response.raw
    sockets = [
        getattr(getattr(raw, "_connection", None), "sock", None),
        getattr(getattr(raw, "connection", None), "sock", None),
    ]
    nested = getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None)
    sockets.append(getattr(nested, "_sock", None))
    for candidate in sockets:
        if candidate is None:
            continue
        try:
            peer = candidate.getpeername()
        except OSError:
            continue
        if isinstance(peer, tuple) and peer:
            return str(peer[0]).split("%", 1)[0]
    raise ValueError("peer_address_unavailable")


def validate_public_response_peer(
    response: requests.Response,
    *,
    allow_validated_transport: bool = False,
) -> None:
    try:
        address = _response_peer_address(response)
    except ValueError as exc:
        transport = getattr(response, "connection", None)
        if (
            allow_validated_transport
            and str(exc) == "peer_address_unavailable"
            and bool(getattr(transport, "peer_validation_enforced", False))
        ):
            # Empty redirect responses can release their urllib3 socket before
            # requests builds the Response. The mounted transport validated the
            # actual peer at connect time, so the missing post-response socket is
            # expected and does not weaken the fail-closed guarantee.
            return
        raise
    _validate_public_peer_address(address)


@dataclass
class RemoteMediaItem:
    source_url: str
    resolved_url: str
    media_kind: str
    mime_type: str = ""
    byte_count: int = 0
    data_url: str = ""
    page_context: str = ""
    note: str = ""
    video_sha256: str = ""
    frame_timestamp: float | None = None
    frame_sequence: int | None = None
    frame_count: int | None = None

    def as_image_item(self, index: int) -> dict[str, Any]:
        encoded = self.data_url.split(",", 1)[1] if "," in self.data_url else ""
        return {
            "index": index,
            "mime_type": self.mime_type,
            "bytes": self.byte_count,
            "base64": encoded,
            "data_url": self.data_url,
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "media_kind": self.media_kind,
            "video_sha256": self.video_sha256,
            "frame_timestamp": self.frame_timestamp,
            "frame_sequence": self.frame_sequence,
            "frame_count": self.frame_count,
        }

    def public_metadata(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "media_kind": self.media_kind,
            "mime_type": self.mime_type,
            "bytes": self.byte_count,
            "usable": bool(self.data_url),
            "page_context": self.page_context,
            "note": self.note,
            "video_sha256": self.video_sha256,
            "frame_timestamp": self.frame_timestamp,
            "frame_sequence": self.frame_sequence,
            "frame_count": self.frame_count,
        }


@dataclass
class RemoteMediaResult:
    original_question: str
    cleaned_question: str
    urls: list[str]
    items: list[RemoteMediaItem] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def image_items(self, start_index: int = 1) -> list[dict[str, Any]]:
        return [item.as_image_item(start_index + idx) for idx, item in enumerate(self.items) if item.data_url]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "detected": len(self.urls),
            "usable": len([item for item in self.items if item.data_url]),
            "items": [item.public_metadata() for item in self.items],
            "errors": self.errors,
        }

    def context_text(self) -> str:
        values = [item.page_context.strip() for item in self.items if item.page_context.strip()]
        return re.sub(r"\s+", " ", " ".join(dict.fromkeys(values))).strip()[:1200]


class RemoteMediaResolver:
    """Resolve public image URLs and social-video covers embedded in a question."""

    def __init__(self) -> None:
        self.enabled = bool_env("REMOTE_MEDIA_ENABLED", True)
        self.timeout = float(os.environ.get("REMOTE_MEDIA_TIMEOUT", "15"))
        self.max_urls = max(0, int(os.environ.get("REMOTE_MEDIA_MAX_URLS", "3")))
        self.max_bytes = max(1024, int(os.environ.get("REMOTE_MEDIA_MAX_BYTES", str(8 * 1024 * 1024))))
        self.max_html_bytes = max(1024, int(os.environ.get("REMOTE_MEDIA_MAX_HTML_BYTES", str(2 * 1024 * 1024))))
        self.max_video_bytes = max(
            1024,
            int(os.environ.get("REMOTE_MEDIA_MAX_VIDEO_BYTES", str(32 * 1024 * 1024))),
        )
        self.video_enabled = bool_env("REMOTE_MEDIA_VIDEO_ENABLED", True)
        self.video_frame_count = max(
            2,
            min(6, int(os.environ.get("REMOTE_MEDIA_VIDEO_FRAME_COUNT", "3"))),
        )
        self.video_adaptive_enabled = bool_env("REMOTE_MEDIA_VIDEO_ADAPTIVE_ENABLED", True)
        self.video_max_frames = max(
            self.video_frame_count,
            min(12, int(os.environ.get("REMOTE_MEDIA_VIDEO_MAX_FRAMES", "8"))),
        )
        self.video_scan_fps = max(
            0.5,
            min(5.0, float(os.environ.get("REMOTE_MEDIA_VIDEO_SCAN_FPS", "2"))),
        )
        self.video_change_threshold = max(
            0.01,
            min(0.80, float(os.environ.get("REMOTE_MEDIA_VIDEO_CHANGE_THRESHOLD", "0.10"))),
        )
        self.video_dedup_threshold = max(
            0.005,
            min(0.30, float(os.environ.get("REMOTE_MEDIA_VIDEO_DEDUP_THRESHOLD", "0.035"))),
        )
        self.video_min_clarity = max(
            0.0,
            min(1.0, float(os.environ.get("REMOTE_MEDIA_VIDEO_MIN_CLARITY", "0.015"))),
        )
        self.video_max_duration = max(
            1.0,
            float(os.environ.get("REMOTE_MEDIA_VIDEO_MAX_DURATION_SECONDS", "120")),
        )
        self.video_frame_max_edge = max(
            320,
            int(os.environ.get("REMOTE_MEDIA_VIDEO_FRAME_MAX_EDGE", "1280")),
        )
        self.max_redirects = max(0, int(os.environ.get("REMOTE_MEDIA_MAX_REDIRECTS", "5")))
        self.user_agent = os.environ.get(
            "REMOTE_MEDIA_USER_AGENT",
            "Mozilla/5.0 (compatible; CustomerAgentMediaResolver/1.0)",
        )
        self.session = requests.Session()
        self.session.trust_env = False
        peer_adapter = PublicPeerHTTPAdapter()
        self.session.mount("http://", peer_adapter)
        self.session.mount("https://", peer_adapter)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_urls": self.max_urls,
            "max_bytes": self.max_bytes,
            "max_redirects": self.max_redirects,
            "video_mode": (
                "adaptive_keyframes_with_cover_fallback"
                if self.video_enabled and self.video_adaptive_enabled
                else "keyframes_with_cover_fallback"
                if self.video_enabled
                else "page_cover_fallback"
            ),
            "video_frame_count": self.video_frame_count,
            "video_frame_range": [self.video_frame_count, self.video_max_frames],
            "video_scan_fps": self.video_scan_fps,
            "video_change_threshold": self.video_change_threshold,
            "video_min_clarity": self.video_min_clarity,
            "max_video_bytes": self.max_video_bytes,
            "video_max_duration_seconds": self.video_max_duration,
            "ssrf_public_hosts_only": True,
        }

    def resolve(self, question: str) -> RemoteMediaResult:
        urls = list(dict.fromkeys(URL_RE.findall(str(question or ""))))[: self.max_urls]
        result = RemoteMediaResult(
            original_question=str(question or ""),
            cleaned_question=strip_remote_urls(question),
            urls=urls,
        )
        if not self.enabled or not urls:
            return result
        for url in urls:
            try:
                items = self._resolve_one(url)
            except Exception as exc:
                result.errors.append({"url": url, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            usable_items = [item for item in items if item.data_url]
            if usable_items:
                result.items.extend(usable_items)
            else:
                note = next((item.note for item in items if item.note), "unsupported_media")
                result.errors.append({"url": url, "reason": note})
        return result

    def _resolve_one(self, source_url: str) -> list[RemoteMediaItem]:
        response, final_url = self._get(
            source_url,
            max_bytes=max(self.max_bytes, self.max_html_bytes, self.max_video_bytes),
        )
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        body = response.content
        if content_type in SUPPORTED_IMAGE_TYPES:
            if len(body) > self.max_bytes:
                return [RemoteMediaItem(source_url, final_url, "image", note="image_too_large")]
            encoded = base64.b64encode(body).decode("ascii")
            return [
                RemoteMediaItem(
                    source_url=source_url,
                    resolved_url=final_url,
                    media_kind="image",
                    mime_type=content_type,
                    byte_count=len(body),
                    data_url=f"data:{content_type};base64,{encoded}",
                    note="public_image_url",
                )
            ]
        if content_type in {"text/html", "application/xhtml+xml"}:
            if len(body) > self.max_html_bytes:
                return [RemoteMediaItem(source_url, final_url, "html", note="html_too_large")]
            html = body.decode(response.encoding or "utf-8", errors="replace")
            page_context = self._extract_page_context(html)
            video_url = self._extract_video_url(html, final_url)
            if self.video_enabled and video_url:
                try:
                    video_response, video_final_url = self._get(
                        video_url,
                        max_bytes=self.max_video_bytes,
                        referer=final_url,
                    )
                    video_type = (
                        video_response.headers.get("Content-Type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if video_type.startswith("video/") or ".mp4" in video_final_url.lower():
                        frames = self._extract_video_frames(
                            source_url=source_url,
                            resolved_url=video_final_url,
                            video_body=video_response.content,
                            page_context=page_context,
                        )
                        if frames:
                            return frames
                except Exception:
                    # A cover is still useful when a platform changes its video
                    # schema, blocks the media URL, or ships an unsupported codec.
                    pass
            cover_url = self._extract_cover_url(html, final_url)
            if not cover_url:
                return [RemoteMediaItem(source_url, final_url, "video_page", note="video_page_without_cover")]
            cover_response, cover_final_url = self._get(cover_url, max_bytes=self.max_bytes)
            cover_type = cover_response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if cover_type not in SUPPORTED_IMAGE_TYPES:
                return [
                    RemoteMediaItem(
                        source_url,
                        cover_final_url,
                        "video_cover",
                        note="unsupported_cover_type",
                    )
                ]
            cover_body = cover_response.content
            encoded = base64.b64encode(cover_body).decode("ascii")
            return [
                RemoteMediaItem(
                    source_url=source_url,
                    resolved_url=cover_final_url,
                    media_kind="video_cover",
                    mime_type=cover_type,
                    byte_count=len(cover_body),
                    data_url=f"data:{cover_type};base64,{encoded}",
                    page_context=page_context,
                    note="video_cover_fallback",
                )
            ]
        if content_type.startswith("video/"):
            if self.video_enabled:
                frames = self._extract_video_frames(
                    source_url=source_url,
                    resolved_url=final_url,
                    video_body=body,
                    page_context="",
                )
                if frames:
                    return frames
            return [RemoteMediaItem(source_url, final_url, "video", note="video_frame_extraction_failed")]
        return [
            RemoteMediaItem(
                source_url,
                final_url,
                "unknown",
                note=f"unsupported_content_type:{content_type or 'missing'}",
            )
        ]

    def _get(self, url: str, *, max_bytes: int, referer: str = "") -> tuple[requests.Response, str]:
        current = url
        for redirect_count in range(self.max_redirects + 1):
            valid, reason = validate_public_url(current)
            if not valid:
                raise ValueError(reason)
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "image/*,video/*,text/html;q=0.9,*/*;q=0.5",
            }
            if referer:
                headers["Referer"] = referer
            response = self.session.get(
                current,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
                stream=True,
            )
            try:
                validate_public_response_peer(response, allow_validated_transport=True)
            except ValueError:
                response.close()
                raise
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise ValueError("redirect_without_location")
                if redirect_count >= self.max_redirects:
                    raise ValueError("too_many_redirects")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    response.close()
                    raise ValueError("response_too_large")
                chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
            return response, current
        raise ValueError("too_many_redirects")

    @staticmethod
    def _extract_cover_url(html: str, page_url: str) -> str:
        for pattern in HTML_COVER_PATTERNS:
            match = pattern.search(html)
            if not match:
                continue
            candidate = _decode_json_string(match.group(1)).strip()
            if candidate.startswith("//"):
                candidate = "https:" + candidate
            candidate = urljoin(page_url, candidate)
            if candidate.startswith("http://"):
                candidate = "https://" + candidate[len("http://") :]
            return candidate
        return ""

    @staticmethod
    def _extract_video_url(html: str, page_url: str) -> str:
        note_match = re.search(r'"noteData"\s*:\s*\{', html)
        segment = html[note_match.start() :] if note_match else html
        for pattern in (
            re.compile(r'"masterUrl"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
            re.compile(r'"backupUrls"\s*:\s*\[\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
            re.compile(r'<meta[^>]+property=["\']og:video(?::url)?["\'][^>]+content=["\']([^"\']+)', re.I),
        ):
            match = pattern.search(segment)
            if not match:
                continue
            candidate = _decode_json_string(match.group(1)).strip()
            if candidate.startswith("//"):
                candidate = "https:" + candidate
            candidate = urljoin(page_url, candidate)
            if candidate.startswith("http://"):
                candidate = "https://" + candidate[len("http://") :]
            return candidate
        return ""

    def _extract_video_frames(
        self,
        *,
        source_url: str,
        resolved_url: str,
        video_body: bytes,
        page_context: str,
    ) -> list[RemoteMediaItem]:
        import av

        container = av.open(io.BytesIO(video_body))
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return []
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            if duration <= 0 and stream.frames and stream.average_rate:
                duration = float(stream.frames / stream.average_rate)
            if duration <= 0 or duration > self.video_max_duration:
                return []

            fractions = self._anchor_fractions(self.video_frame_count)
            anchor_targets = [duration * fraction for fraction in fractions]
            if self.video_adaptive_enabled:
                candidates = self._scan_video_candidates(container, stream, duration)
                selected_specs = self._select_adaptive_candidates(
                    candidates,
                    anchor_targets=anchor_targets,
                    duration=duration,
                )
            else:
                selected_specs = [
                    {"timestamp": target, "reason": "time_anchor"}
                    for target in anchor_targets
                ]
            if not selected_specs:
                return []
        finally:
            container.close()

        selected = self._decode_selected_frames(video_body, selected_specs)
        if not selected:
            return []

        digest = hashlib.sha256(video_body).hexdigest()
        frame_count = len(selected)
        times = ",".join(f"{timestamp:.2f}s" for timestamp, _, _ in selected)
        mode = "adaptive" if self.video_adaptive_enabled else "fixed"
        temporal_context = (
            f"{page_context} Chronological video keyframes ({mode}): {times}; "
            f"video_duration={duration:.2f}s; frame_count={frame_count}."
        ).strip()
        items: list[RemoteMediaItem] = []
        for index, (timestamp, image, reason) in enumerate(selected, start=1):
            if max(image.size) > self.video_frame_max_edge:
                image.thumbnail(
                    (self.video_frame_max_edge, self.video_frame_max_edge),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
            body = buffer.getvalue()
            encoded = base64.b64encode(body).decode("ascii")
            items.append(
                RemoteMediaItem(
                    source_url=source_url,
                    resolved_url=resolved_url,
                    media_kind="video_frame",
                    mime_type="image/jpeg",
                    byte_count=len(body),
                    data_url=f"data:image/jpeg;base64,{encoded}",
                    page_context=temporal_context,
                    note=f"adaptive_keyframe_{index}_of_{frame_count}:{reason}",
                    video_sha256=digest,
                    frame_timestamp=round(timestamp, 3),
                    frame_sequence=index,
                    frame_count=frame_count,
                )
            )
        return items

    @staticmethod
    def _anchor_fractions(count: int) -> list[float]:
        if count == 2:
            return [0.25, 0.75]
        return [0.10 + (0.80 * index / (count - 1)) for index in range(count)]

    @staticmethod
    def _analysis_image(image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        gray.thumbnail((160, 90), Image.Resampling.BILINEAR)
        return gray

    @staticmethod
    def _frame_difference(left: Image.Image, right: Image.Image) -> float:
        if left.size != right.size:
            right = right.resize(left.size, Image.Resampling.BILINEAR)
        difference = ImageChops.difference(left, right)
        return float(ImageStat.Stat(difference).mean[0] / 255.0)

    @staticmethod
    def _frame_clarity(image: Image.Image) -> float:
        edges = image.filter(ImageFilter.FIND_EDGES)
        variance = float(ImageStat.Stat(edges).var[0])
        return min(1.0, variance / 2500.0)

    def _scan_video_candidates(self, container: Any, stream: Any, duration: float) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        sample_interval = 1.0 / self.video_scan_fps
        next_sample = 0.0
        previous: Image.Image | None = None
        last_time = 0.0
        frame_index = 0
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            frame_time = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else (frame_index / rate if rate > 0 else last_time)
            )
            frame_index += 1
            last_time = frame_time
            if frame_time + 1e-6 < next_sample:
                continue
            image = self._analysis_image(frame.to_image())
            change = self._frame_difference(previous, image) if previous is not None else 0.0
            clarity = self._frame_clarity(image)
            candidates.append(
                {
                    "timestamp": min(max(0.0, frame_time), duration),
                    "thumb": image,
                    "change": change,
                    "clarity": clarity,
                    "score": change * (0.65 + 0.35 * clarity),
                }
            )
            previous = image
            next_sample = frame_time + sample_interval
        return candidates

    def _select_adaptive_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        anchor_targets: list[float],
        duration: float,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        selected: list[dict[str, Any]] = []
        selected_indexes: set[int] = set()
        for target in anchor_targets:
            index = min(
                range(len(candidates)),
                key=lambda item: abs(float(candidates[item]["timestamp"]) - target),
            )
            if index not in selected_indexes:
                selected.append({**candidates[index], "reason": "time_anchor"})
                selected_indexes.add(index)

        minimum_gap = max(0.60, min(2.0, duration * 0.04))
        ranked = sorted(
            (
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if index not in selected_indexes
                and float(candidate.get("change") or 0.0) >= self.video_change_threshold
                and float(candidate.get("clarity") or 0.0) >= self.video_min_clarity
                and float(candidate.get("score") or 0.0)
                >= float(candidates[index - 1].get("score") or 0.0)
                and (
                    index + 1 >= len(candidates)
                    or float(candidate.get("score") or 0.0)
                    >= float(candidates[index + 1].get("score") or 0.0)
                )
            ),
            key=lambda item: float(item[1].get("score") or 0.0),
            reverse=True,
        )
        for index, candidate in ranked:
            if len(selected) >= self.video_max_frames:
                break
            timestamp = float(candidate["timestamp"])
            if any(abs(timestamp - float(item["timestamp"])) < minimum_gap for item in selected):
                continue
            if any(
                self._frame_difference(candidate["thumb"], item["thumb"])
                < self.video_dedup_threshold
                for item in selected
            ):
                continue
            selected.append({**candidate, "reason": "scene_change"})
            selected_indexes.add(index)

        selected.sort(key=lambda item: float(item["timestamp"]))
        return [
            {"timestamp": float(item["timestamp"]), "reason": str(item["reason"])}
            for item in selected[: self.video_max_frames]
        ]

    @staticmethod
    def _decode_selected_frames(
        video_body: bytes,
        selected_specs: list[dict[str, Any]],
    ) -> list[tuple[float, Image.Image, str]]:
        import av

        targets = sorted(selected_specs, key=lambda item: float(item["timestamp"]))
        container = av.open(io.BytesIO(video_body))
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return []
            selected: list[tuple[float, Image.Image, str]] = []
            target_index = 0
            last_image: Image.Image | None = None
            last_time = 0.0
            frame_index = 0
            rate = float(stream.average_rate) if stream.average_rate else 0.0
            for frame in container.decode(stream):
                frame_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else (frame_index / rate if rate > 0 else last_time)
                )
                frame_index += 1
                image = frame.to_image().convert("RGB")
                last_image, last_time = image, frame_time
                while (
                    target_index < len(targets)
                    and frame_time >= float(targets[target_index]["timestamp"])
                ):
                    selected.append(
                        (
                            frame_time,
                            image.copy(),
                            str(targets[target_index].get("reason") or "keyframe"),
                        )
                    )
                    target_index += 1
                if target_index >= len(targets):
                    break
            while target_index < len(targets) and last_image is not None:
                selected.append(
                    (
                        last_time,
                        last_image.copy(),
                        str(targets[target_index].get("reason") or "keyframe"),
                    )
                )
                target_index += 1
            return selected
        finally:
            container.close()

    @staticmethod
    def _extract_page_context(html: str) -> str:
        values: list[str] = []
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if title_match:
            values.append(re.sub(r"<[^>]+>", " ", title_match.group(1)))
        desc_matches = re.findall(r'"desc"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', html)
        for value in desc_matches[:5]:
            decoded = _decode_json_string(value)
            if decoded and decoded not in values:
                values.append(decoded)
        cleaned = re.sub(r"\s+", " ", " ".join(values)).strip()
        return cleaned[:1200]
