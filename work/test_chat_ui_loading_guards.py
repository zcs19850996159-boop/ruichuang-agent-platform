from pathlib import Path


def test_manual_image_loading_has_terminal_states() -> None:
    html = (Path(__file__).resolve().parent / "chat_ui.html").read_text(encoding="utf-8")

    assert "AbortController" in html
    assert "图片解码超时" in html
    assert "图片加载失败，请稍后重试" in html
    assert 'event==="answer_delta"' in html
    answer_delta = html.split('event==="answer_delta"', 1)[1].split("else if(event===", 1)[0]
    assert "loadManualFigures" not in answer_delta
