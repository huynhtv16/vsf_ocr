# Copyright (c) Opendatalab. All rights reserved.
import re


# Process text content.
# Validate the current value.
LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"
LINE_END_HYPHEN_RE = re.compile(
    rf"[A-Za-z]+[{re.escape(LINE_END_HYPHEN_CHARS)}]\s*$"
)


def is_hyphen_at_line_end(line):
    """Validate the current value.

    Process the current item.
    """
    return bool(LINE_END_HYPHEN_RE.search(line))


def full_to_half_exclude_marks(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)
