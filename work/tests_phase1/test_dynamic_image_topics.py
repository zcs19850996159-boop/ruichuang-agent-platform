from dynamic_image_selector import _topic_images


def test_manual27_generic_battery_replacement_uses_installation_figures() -> None:
    decision = _topic_images(
        "Manual27",
        "如何按照手册更换电池？",
        {},
        {"Manual27_1", "Manual27_2", "Manual27_3"},
    )

    assert decision == (
        ["Manual27_1", "Manual27_2", "Manual27_3"],
        "topic:bluetooth_mouse_battery_install",
    )


def test_manual27_battery_status_keeps_status_figures() -> None:
    decision = _topic_images(
        "Manual27",
        "如何查看鼠标电量状态？",
        {},
        {"Manual27_14", "Manual27_15", "Manual27_16", "Manual27_17"},
    )

    assert decision == (
        ["Manual27_14", "Manual27_15", "Manual27_16", "Manual27_17"],
        "topic:bluetooth_mouse_battery_status",
    )
