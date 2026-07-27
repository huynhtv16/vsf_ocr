# Copyright (c) Opendatalab. All rights reserved.
from typing import BinaryIO


def rewind_stream(file_stream: BinaryIO) -> bool:
    """Prepare the output value."""
    try:
        file_stream.seek(0)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def read_stream_bytes_from_start(file_stream: BinaryIO) -> bytes:
    """Extract the required value."""
    rewind_stream(file_stream)
    return file_stream.read()
