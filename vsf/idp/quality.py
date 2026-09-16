"""Quality scoring, semantic validation, and human-review decisions for IDP."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable


def normalized_classification_scores(
    scores: dict[str, float],
) -> dict[str, float]:
    """Convert positive rule scores into a comparable distribution."""

    positive = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {key: 0.0 for key in positive}
    return {key: round(value / total, 4) for key, value in positive.items()}


def classification_margin(scores: dict[str, float]) -> float:
    ranked = sorted((float(value) for value in scores.values()), reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return round(ranked[0], 4)
    return round(max(0.0, ranked[0] - ranked[1]), 4)


def _append_issue(
    issues: list[dict[str, str]],
    *,
    code: str,
    field: str,
    severity: str,
    message: str,
) -> None:
    if any(issue.get("code") == code and issue.get("field") == field for issue in issues):
        return
    issues.append(
        {
            "code": code,
            "field": field,
            "severity": severity,
            "message": message,
        }
    )


def add_semantic_validation_issues(
    document_type: str,
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    """Add format and cross-field checks that are independent of extraction."""

    email = fields.get("email", {}).get("value")
    if email and not re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", str(email)):
        _append_issue(
            issues,
            code="INVALID_EMAIL",
            field="email",
            severity="high",
            message="Email không đúng định dạng.",
        )

    phone = fields.get("phone", {}).get("value")
    if phone:
        digits = re.sub(r"\D", "", str(phone))
        if not 9 <= len(digits) <= 15:
            _append_issue(
                issues,
                code="INVALID_PHONE",
                field="phone",
                severity="high",
                message="Số điện thoại phải có từ 9 đến 15 chữ số.",
            )

    for field_name in ("full_name", "employee_name"):
        value = fields.get(field_name, {}).get("value")
        if value and (any(char.isdigit() for char in str(value)) or len(str(value)) < 3):
            _append_issue(
                issues,
                code="INVALID_PERSON_NAME",
                field=field_name,
                severity="medium",
                message=f"Giá trị {field_name} không giống tên người hợp lệ.",
            )

    birth_value = fields.get("date_of_birth", {}).get("value")
    if isinstance(birth_value, str):
        try:
            birth_date = date.fromisoformat(birth_value)
        except ValueError:
            _append_issue(
                issues,
                code="INVALID_DATE",
                field="date_of_birth",
                severity="high",
                message="Ngày sinh không đúng định dạng hoặc không tồn tại.",
            )
        else:
            today = date.today()
            if birth_date > today:
                _append_issue(
                    issues,
                    code="FUTURE_DATE_OF_BIRTH",
                    field="date_of_birth",
                    severity="high",
                    message="Ngày sinh không được nằm trong tương lai.",
                )
            elif (today - birth_date).days > 125 * 366:
                _append_issue(
                    issues,
                    code="IMPLAUSIBLE_DATE_OF_BIRTH",
                    field="date_of_birth",
                    severity="medium",
                    message="Ngày sinh nằm ngoài khoảng hợp lý.",
                )

    for field_name in ("salary", "base_salary", "net_salary"):
        value = fields.get(field_name, {}).get("value")
        if isinstance(value, dict):
            amount = value.get("amount")
            if isinstance(amount, (int, float)) and amount <= 0:
                _append_issue(
                    issues,
                    code="INVALID_MONEY_AMOUNT",
                    field=field_name,
                    severity="high",
                    message=f"{field_name} phải lớn hơn 0.",
                )

    graduation_year = fields.get("graduation_year", {}).get("value")
    if isinstance(graduation_year, (int, float)):
        if not 1950 <= int(graduation_year) <= date.today().year + 1:
            _append_issue(
                issues,
                code="INVALID_GRADUATION_YEAR",
                field="graduation_year",
                severity="medium",
                message="Năm tốt nghiệp nằm ngoài khoảng hợp lý.",
            )

    if document_type == "leave_request":
        number_of_days = fields.get("number_of_days", {}).get("value")
        from_value = fields.get("from_date", {}).get("value")
        to_value = fields.get("to_date", {}).get("value")
        if isinstance(number_of_days, (int, float)) and from_value and to_value:
            try:
                inclusive_days = (
                    date.fromisoformat(str(to_value)) - date.fromisoformat(str(from_value))
                ).days + 1
            except ValueError:
                pass
            else:
                if inclusive_days > 0 and abs(float(number_of_days) - inclusive_days) > 1:
                    _append_issue(
                        issues,
                        code="LEAVE_DAYS_MISMATCH",
                        field="number_of_days",
                        severity="medium",
                        message=(
                            "Số ngày nghỉ không khớp với khoảng ngày; "
                            "cần kiểm tra lịch làm việc và ngày nghỉ lễ."
                        ),
                    )


def _text_quality_score(blocks: Iterable[Any]) -> float:
    text = "\n".join(str(getattr(block, "text", "") or "") for block in blocks)
    if not text.strip():
        return 0.0
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    control_count = sum(
        1 for char in text if ord(char) < 32 and char not in "\n\r\t"
    )
    control_ratio = control_count / max(1, len(text))
    alphanumeric_ratio = sum(char.isalnum() for char in text) / max(1, len(text))
    score = 1.0 - min(0.7, replacement_ratio * 25 + control_ratio * 25)
    if alphanumeric_ratio < 0.15:
        score -= 0.25
    return round(max(0.0, min(1.0, score)), 4)


def build_quality_report(
    *,
    document_type: str,
    classification_confidence: float,
    schema: Any,
    fields: dict[str, dict[str, Any]],
    blocks: list[Any],
    issues: list[dict[str, str]],
    structured_tables: list[dict[str, Any]],
) -> dict[str, Any]:
    schema_fields = list(getattr(schema, "fields", ()) or ())
    required_names = [field.name for field in schema_fields if field.required]
    optional_names = [field.name for field in schema_fields if not field.required]
    required_present = sum(
        fields.get(name, {}).get("value") not in (None, "") for name in required_names
    )
    optional_present = sum(
        fields.get(name, {}).get("value") not in (None, "") for name in optional_names
    )
    required_score = (
        required_present / len(required_names) if required_names else 1.0
    )
    optional_score = (
        optional_present / len(optional_names) if optional_names else 1.0
    )
    completeness_score = 0.75 * required_score + 0.25 * optional_score

    extracted = [
        result for result in fields.values() if result.get("value") not in (None, "")
    ]
    field_confidence_score = (
        sum(float(result.get("confidence", 0.0)) for result in extracted) / len(extracted)
        if extracted
        else 0.0
    )
    evidence_score = (
        sum(
            1.0
            if result.get("evidence")
            else 0.5
            if result.get("extraction_method") == "filename_rule"
            else 0.0
            for result in extracted
        )
        / len(extracted)
        if extracted
        else 0.0
    )
    text_quality_score = _text_quality_score(blocks)
    table_structure_score = (
        sum(float(table.get("confidence", 0.0)) for table in structured_tables)
        / len(structured_tables)
        if structured_tables
        else 1.0
    )
    classification_score = float(classification_confidence)
    if document_type == "other_document":
        classification_score = min(classification_score, 0.65)

    overall_score = (
        0.24 * classification_score
        + 0.25 * completeness_score
        + 0.21 * field_confidence_score
        + 0.15 * text_quality_score
        + 0.10 * evidence_score
        + 0.05 * table_structure_score
    )
    high_issue_count = sum(issue.get("severity") == "high" for issue in issues)
    medium_issue_count = sum(issue.get("severity") == "medium" for issue in issues)
    overall_score -= min(0.35, high_issue_count * 0.15 + medium_issue_count * 0.05)
    overall_score = round(max(0.0, min(1.0, overall_score)), 4)
    grade = "high" if overall_score >= 0.85 else "medium" if overall_score >= 0.65 else "low"

    return {
        "overall_score": overall_score,
        "grade": grade,
        "classification_score": round(classification_score, 4),
        "required_field_completeness": round(required_score, 4),
        "field_completeness": round(completeness_score, 4),
        "field_confidence_score": round(field_confidence_score, 4),
        "evidence_coverage": round(evidence_score, 4),
        "text_quality_score": text_quality_score,
        "table_structure_score": round(table_structure_score, 4),
        "issue_counts": {
            "high": high_issue_count,
            "medium": medium_issue_count,
            "low": sum(issue.get("severity") == "low" for issue in issues),
        },
    }


def build_review_decision(
    *,
    document_type: str,
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    reasons = [issue.get("message", issue.get("code", "")) for issue in issues]
    fields_to_review = {
        str(issue.get("field"))
        for issue in issues
        if issue.get("field")
    }
    for field_name, result in fields.items():
        if result.get("value") not in (None, "") and float(result.get("confidence", 0)) < 0.8:
            fields_to_review.add(field_name)

    if document_type == "other_document":
        reasons.append(
            "Tài liệu đang dùng schema chung; cần chọn hoặc xây schema chuyên sâu "
            "nếu muốn tự động hóa nghiệp vụ."
        )
    overall_score = float(quality.get("overall_score", 0.0))
    required = bool(issues) or document_type == "other_document" or overall_score < 0.78
    has_high_issue = any(issue.get("severity") == "high" for issue in issues)
    priority = (
        "high"
        if has_high_issue or overall_score < 0.55
        else "medium"
        if required
        else "low"
    )
    if required:
        action = (
            "Kiểm tra các trường được liệt kê và đối chiếu với evidence trước khi "
            "ghi vào hệ thống nghiệp vụ."
        )
    else:
        action = "Có thể tiếp tục quy trình tự động; vẫn lưu audit trail để truy vết."
    return {
        "required": required,
        "priority": priority,
        "reasons": list(dict.fromkeys(filter(None, reasons))),
        "fields_to_review": sorted(fields_to_review),
        "recommended_action": action,
    }
