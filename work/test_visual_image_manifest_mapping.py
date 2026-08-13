from __future__ import annotations

from visual_image_retrieval import VisualImageMatcher


def main() -> None:
    matcher = object.__new__(VisualImageMatcher)
    matcher._manual_by_image_id = {
        "air_fryer_16": "Manual08",
        "toothbrush0_11": "Manual37",
        "oven_13": "Manual28",
    }
    matcher._evidence_by_image_id = {}
    assert matcher._manual_id("air_fryer_16") == "Manual08"
    assert matcher._manual_id("toothbrush0_11") == "Manual37"
    assert matcher._manual_id("oven_13") == "Manual28"
    assert matcher._manual_id("Manual14_39") == "Manual14"
    assert matcher._manual_id("unknown_1") == ""
    print("visual image manifest mapping tests passed")


if __name__ == "__main__":
    main()
