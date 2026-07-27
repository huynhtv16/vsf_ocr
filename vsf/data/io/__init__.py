# Copyright (c) Opendatalab. All rights reserved.

from .base import IOReader, IOWriter
from .http import HttpReader, HttpWriter

__all__ = ['IOReader', 'IOWriter', 'HttpReader', 'HttpWriter', 'S3Reader', 'S3Writer']


def __getattr__(name):
    """Implementation detail."""
    if name in {'S3Reader', 'S3Writer'}:
        from .s3 import S3Reader, S3Writer

        s3_exports = {'S3Reader': S3Reader, 'S3Writer': S3Writer}
        globals().update(s3_exports)
        return s3_exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
