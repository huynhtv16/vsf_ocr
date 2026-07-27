"""Rule-based Intelligent Document Processing for HR documents."""

from vsf.idp.engine import HRIDPProcessor, process_hr_document
from vsf.idp.schemas import DOCUMENT_TYPE_AUTO, DOCUMENT_TYPES

__all__ = [
    "DOCUMENT_TYPE_AUTO",
    "DOCUMENT_TYPES",
    "HRIDPProcessor",
    "process_hr_document",
]
