from __future__ import annotations

from agent_api import (
    direct_image_answer,
    is_direct_image_understanding_query,
    requires_current_visual_input,
)


def main() -> None:
    assert is_direct_image_understanding_query("请描述这张图片里有什么")
    assert is_direct_image_understanding_query("Read the text visible in this screenshot")
    assert is_direct_image_understanding_query("这是什么？")
    assert is_direct_image_understanding_query("What's this?")
    assert not is_direct_image_understanding_query(
        "请识别图片中的部件，并根据对应手册说明它的用途或正确使用方法。"
    )
    assert not is_direct_image_understanding_query("图中这个错误码怎么处理？")
    assert not is_direct_image_understanding_query("How should I repair the component in this image?")
    assert requires_current_visual_input("这张图片里的部件是什么？")
    assert requires_current_visual_input("图中这个错误码怎么处理？")
    assert requires_current_visual_input("How should I repair the component in this image?")
    assert not requires_current_visual_input("如何上传图片？")
    assert not requires_current_visual_input("手册中有哪些产品图片？")
    answer = direct_image_answer(
        "这是什么？",
        "product=鼠标; product_model=Logitech MX Master 3",
        True,
        {
            "product": "鼠标",
            "product_model": "Logitech MX Master 3",
            "visual_description": "一个黑色无线鼠标",
            "ocr_text": "Logi",
        },
    )
    assert "Logitech MX Master 3 鼠标" in answer
    assert "product=" not in answer
    print("image RAG routing tests passed")


if __name__ == "__main__":
    main()
