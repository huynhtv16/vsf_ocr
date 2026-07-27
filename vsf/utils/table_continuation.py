# Copyright (c) Opendatalab. All rights reserved.

from mineru.utils.char_utils import full_to_half


CONTINUATION_END_MARKERS = [
    "(\u7eed)",
    "(\u7eed\u8868)",
    "(\u7eed\u4e0a\u8868)",
    "(continued)",
    "(cont.)",
    "(cont\u2019d)",
    "(…continued)",
    "continued",
    "\u7eed\u8868",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]


def is_table_continuation_text(text: str) -> bool:
    """Validate the current value."""
    continuation_text = full_to_half((text or "").strip()).lower()
    if not continuation_text:
        return False

    return (
        any(
            _matches_continuation_end_marker(continuation_text, marker.lower())
            for marker in CONTINUATION_END_MARKERS
        )
        or any(marker.lower() in continuation_text for marker in CONTINUATION_INLINE_MARKERS)
    )


def _matches_continuation_end_marker(text: str, marker: str) -> bool:
    """Validate the current value."""
    if not text.endswith(marker):
        return False

    if marker == "continued":
        marker_start = len(text) - len(marker)
        return marker_start == 0 or not text[marker_start - 1].isalpha()

    return True
