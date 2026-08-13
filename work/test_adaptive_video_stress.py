from __future__ import annotations

import io
from pathlib import Path

import av
from PIL import Image, ImageDraw

from remote_media import RemoteMediaResolver


def frame_image(cue: bool) -> Image.Image:
    image = Image.new("RGB", (640, 360), (45, 48, 52))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "NORMAL STATE", fill=(210, 210, 210))
    if cue:
        draw.rectangle((120, 90, 520, 280), fill=(235, 225, 30))
        draw.text((230, 155), "ERROR E01", fill=(180, 0, 0), stroke_width=2)
    return image


def build_video(cue_start: float, cue_end: float, duration: float = 10.0, fps: int = 10) -> bytes:
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = "yuv420p"
    for index in range(int(duration * fps)):
        timestamp = index / fps
        frame = av.VideoFrame.from_image(frame_image(cue_start <= timestamp <= cue_end))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def main() -> None:
    resolver = RemoteMediaResolver()
    resolver.video_adaptive_enabled = True
    cases = [(2.2, 3.0), (3.2, 4.0), (6.1, 6.9), (7.4, 8.2)]
    output_dir = Path("outputs/blind_media_benchmark_v1/video_stress")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (start, end) in enumerate(cases, start=1):
        body = build_video(start, end)
        path = output_dir / f"stress_{index:02d}.mp4"
        path.write_bytes(body)
        frames = resolver._extract_video_frames(
            source_url=f"stress://{index}",
            resolved_url=str(path),
            video_body=body,
            page_context="Synthetic keyframe recall test; not a real customer benchmark.",
        )
        times = [float(item.frame_timestamp or 0.0) for item in frames]
        assert any(start <= timestamp <= end + 0.55 for timestamp in times), (
            index,
            start,
            end,
            times,
        )
        assert 3 <= len(frames) <= resolver.video_max_frames
        print({"case": index, "cue": [start, end], "selected": times})
    print("adaptive video stress tests passed")


if __name__ == "__main__":
    main()
