"""Schemas supported by the built-in HR IDP processor."""

from __future__ import annotations

from dataclasses import dataclass

DOCUMENT_TYPE_AUTO = "auto"
DOCUMENT_TYPE_UNKNOWN = "unknown"
DOCUMENT_TYPE_OTHER = "other_document"


@dataclass(frozen=True)
class FieldSchema:
    """Definition of one business field."""

    name: str
    data_type: str = "string"
    required: bool = False
    sensitive: bool = False


@dataclass(frozen=True)
class DocumentSchema:
    """Definition and classification hints for one HR document type."""

    document_type: str
    label: str
    keywords: tuple[tuple[str, float], ...]
    fields: tuple[FieldSchema, ...]


DOCUMENT_SCHEMAS: dict[str, DocumentSchema] = {
    "identity_card": DocumentSchema(
        document_type="identity_card",
        label="Căn cước công dân/CMND",
        keywords=(
            ("căn cước công dân", 1.0),
            ("căn cước", 0.65),
            ("chứng minh nhân dân", 1.0),
            ("identity card", 0.85),
            ("quốc tịch", 0.2),
            ("quê quán", 0.2),
            ("nơi thường trú", 0.25),
        ),
        fields=(
            FieldSchema("full_name", required=True, sensitive=True),
            FieldSchema("identity_number", required=True, sensitive=True),
            FieldSchema("date_of_birth", "date", required=True, sensitive=True),
            FieldSchema("gender"),
            FieldSchema("nationality"),
            FieldSchema("place_of_origin", sensitive=True),
            FieldSchema("address", sensitive=True),
            FieldSchema("expiry_date", "date", sensitive=True),
        ),
    ),
    "cv": DocumentSchema(
        document_type="cv",
        label="CV/Sơ yếu lý lịch",
        keywords=(
            ("curriculum vitae", 1.0),
            ("sơ yếu lý lịch", 1.0),
            ("professional summary", 0.45),
            ("work experience", 0.45),
            ("summary", 0.2),
            ("experience", 0.3),
            ("education", 0.3),
            ("skills", 0.25),
            ("projects", 0.2),
            ("linkedin.com/in/", 0.25),
            ("kinh nghiệm làm việc", 0.45),
            ("kinh nghiệm công tác", 0.45),
            ("mục tiêu nghề nghiệp", 0.4),
            ("trình độ học vấn", 0.35),
            ("kỹ năng", 0.2),
        ),
        fields=(
            FieldSchema("full_name", required=True, sensitive=True),
            FieldSchema("date_of_birth", "date", sensitive=True),
            FieldSchema("gender"),
            FieldSchema("phone", sensitive=True),
            FieldSchema("email", sensitive=True),
            FieldSchema("address", sensitive=True),
            FieldSchema("education"),
            FieldSchema("experience"),
            FieldSchema("skills"),
        ),
    ),
    "labor_contract": DocumentSchema(
        document_type="labor_contract",
        label="Hợp đồng lao động",
        keywords=(
            ("hợp đồng lao động", 1.0),
            ("labor contract", 1.0),
            ("người lao động", 0.35),
            ("người sử dụng lao động", 0.35),
            ("loại hợp đồng", 0.25),
            ("thời hạn hợp đồng", 0.3),
            ("mức lương", 0.2),
        ),
        fields=(
            FieldSchema("contract_number", required=True),
            FieldSchema("employee_name", required=True, sensitive=True),
            FieldSchema("employee_id"),
            FieldSchema("identity_number", sensitive=True),
            FieldSchema("contract_type"),
            FieldSchema("start_date", "date", required=True),
            FieldSchema("end_date", "date"),
            FieldSchema("position", required=True),
            FieldSchema("department"),
            FieldSchema("salary", "money", sensitive=True),
        ),
    ),
    "hr_decision": DocumentSchema(
        document_type="hr_decision",
        label="Quyết định nhân sự",
        keywords=(
            ("quyết định", 0.25),
            ("bổ nhiệm", 0.45),
            ("điều chuyển", 0.45),
            ("tăng lương", 0.45),
            ("khen thưởng", 0.4),
            ("kỷ luật", 0.4),
            ("thôi việc", 0.45),
            ("nghỉ việc", 0.4),
        ),
        fields=(
            FieldSchema("decision_number", required=True),
            FieldSchema("decision_date", "date", required=True),
            FieldSchema("decision_type", required=True),
            FieldSchema("employee_name", required=True, sensitive=True),
            FieldSchema("employee_id"),
            FieldSchema("position"),
            FieldSchema("department"),
            FieldSchema("effective_date", "date"),
            FieldSchema("salary", "money", sensitive=True),
        ),
    ),
    "leave_request": DocumentSchema(
        document_type="leave_request",
        label="Đơn xin nghỉ phép",
        keywords=(
            ("đơn xin nghỉ phép", 1.0),
            ("đơn xin nghỉ", 0.8),
            ("leave request", 1.0),
            ("thời gian nghỉ", 0.3),
            ("lý do nghỉ", 0.3),
            ("nghỉ phép năm", 0.35),
        ),
        fields=(
            FieldSchema("employee_name", required=True, sensitive=True),
            FieldSchema("employee_id"),
            FieldSchema("department"),
            FieldSchema("leave_type", required=True),
            FieldSchema("from_date", "date", required=True),
            FieldSchema("to_date", "date", required=True),
            FieldSchema("number_of_days", "number"),
            FieldSchema("reason"),
        ),
    ),
    "degree_certificate": DocumentSchema(
        document_type="degree_certificate",
        label="Bằng cấp/Chứng chỉ",
        keywords=(
            ("bằng tốt nghiệp", 1.0),
            ("chứng chỉ", 0.8),
            ("chứng nhận", 0.55),
            ("diploma", 0.85),
            ("degree certificate", 0.85),
            ("certificate of", 0.35),
            ("certificate", 0.1),
            ("xếp loại", 0.25),
            ("ngành đào tạo", 0.3),
        ),
        fields=(
            FieldSchema("employee_name", required=True, sensitive=True),
            FieldSchema("qualification", required=True),
            FieldSchema("major"),
            FieldSchema("institution"),
            FieldSchema("graduation_year", "number"),
            FieldSchema("classification"),
            FieldSchema("certificate_number"),
        ),
    ),
    DOCUMENT_TYPE_OTHER: DocumentSchema(
        document_type=DOCUMENT_TYPE_OTHER,
        label="Tài liệu khác (không giới hạn mẫu)",
        keywords=(),
        fields=(
            FieldSchema("document_title", required=True),
            FieldSchema("document_number"),
            FieldSchema("issued_date", "date"),
            FieldSchema("organization"),
            FieldSchema("content_summary"),
        ),
    ),
}

DOCUMENT_TYPES = (DOCUMENT_TYPE_AUTO, *DOCUMENT_SCHEMAS.keys())


def get_document_schema(document_type: str) -> DocumentSchema | None:
    """Return a schema by its public document type."""

    return DOCUMENT_SCHEMAS.get(document_type)


def validate_document_type(document_type: str) -> str:
    """Validate and normalize the document type requested by a caller."""

    normalized = (document_type or DOCUMENT_TYPE_AUTO).strip().lower()
    if normalized not in DOCUMENT_TYPES:
        allowed = ", ".join(DOCUMENT_TYPES)
        raise ValueError(
            f"Unsupported IDP document type '{document_type}'. Allowed values: {allowed}"
        )
    return normalized
