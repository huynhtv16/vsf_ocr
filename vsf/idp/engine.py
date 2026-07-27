"""Deterministic HR document classification, extraction, and validation."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from vsf.idp.schemas import (
    DOCUMENT_SCHEMAS,
    DOCUMENT_TYPE_AUTO,
    DOCUMENT_TYPE_OTHER,
    DOCUMENT_TYPE_UNKNOWN,
    DocumentSchema,
    FieldSchema,
    get_document_schema,
    validate_document_type,
)

IDP_SCHEMA_VERSION = "1.0"
_SPACE_RE = re.compile(r"[ \t]+")
_TAG_RE = re.compile(r"<[^>]+>")
_DATE_VALUE = (
    r"(?:\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}"
    r"|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})"
)
_MONEY_VALUE = r"(\d[\d\s.,]{2,}(?:\s*(?:vnđ|vnd|đồng|đ))?)"


@dataclass(frozen=True)
class SourceBlock:
    """Text plus its location in the parsed document."""

    text: str
    page: int | None
    bbox: list[int | float] | None
    index: int


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", str(value)))
    return _SPACE_RE.sub(" ", text).strip()


def _item_text(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type in {"text", "equation"}:
        return _plain_text(item.get("text"))
    if item_type == "table":
        parts = [
            *(item.get("table_caption") or []),
            item.get("table_body"),
            *(item.get("table_footnote") or []),
        ]
        return "\n".join(filter(None, (_plain_text(part) for part in parts)))
    if item_type == "image":
        parts = [
            *(item.get("image_caption") or []),
            *(item.get("image_footnote") or []),
        ]
        return "\n".join(filter(None, (_plain_text(part) for part in parts)))
    return _plain_text(item.get("text") or item.get("content"))


def build_source_blocks(content_list: Iterable[dict[str, Any]]) -> list[SourceBlock]:
    """Convert backend-neutral content-list items into searchable blocks."""

    blocks: list[SourceBlock] = []
    for index, item in enumerate(content_list):
        if not isinstance(item, dict):
            continue
        text = _item_text(item)
        if not text:
            continue
        raw_page = item.get("page_idx")
        page = raw_page + 1 if isinstance(raw_page, int) else None
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = None
        blocks.append(SourceBlock(text=text, page=page, bbox=bbox, index=index))
    return blocks


def _search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _classification(
    blocks: list[SourceBlock],
    requested_type: str,
    document_name: str | None = None,
) -> tuple[str, float, str, dict[str, float]]:
    if requested_type != DOCUMENT_TYPE_AUTO:
        return requested_type, 1.0, "user_selected", {requested_type: 1.0}

    text = _search_text("\n".join(block.text for block in blocks))
    raw_scores: dict[str, float] = {}
    for document_type, schema in DOCUMENT_SCHEMAS.items():
        score = sum(
            weight
            for keyword, weight in schema.keywords
            if _search_text(keyword) in text
        )
        raw_scores[document_type] = round(score, 4)

    normalized_name = _search_text(document_name or "")
    filename_is_cv = bool(
        re.search(r"\b(?:cv|resume|curriculum vitae)\b", normalized_name)
    )
    if filename_is_cv:
        raw_scores["cv"] = round(raw_scores["cv"] + 1.0, 4)
    elif (
        " - " in (document_name or "")
        and re.search(
            r"\b(?:engineer|developer|intern|manager|specialist|executive)\b",
            normalized_name,
        )
    ):
        filename_is_cv = True
        raw_scores["cv"] = round(raw_scores["cv"] + 0.55, 4)

    if not raw_scores:
        return DOCUMENT_TYPE_UNKNOWN, 0.0, "rules", {}

    # Spreadsheets used as dictionaries/training material can contain isolated
    # HR words such as "experience", "education", and "certificate". Do not
    # force those unrelated documents into an HR schema.
    non_hr_signals = (
        "word bank",
        "tu vung",
        "vocabulary",
        "quizlet",
        "toeic",
    )
    non_hr_signal_count = sum(signal in text for signal in non_hr_signals)
    filename_is_non_hr = any(
        signal in normalized_name for signal in non_hr_signals
    )
    title_anchors = (
        "curriculum vitae",
        "so yeu ly lich",
        "can cuoc cong dan",
        "chung minh nhan dan",
        "identity card",
        "hop dong lao dong",
        "labor contract",
        "don xin nghi phep",
        "leave request",
        "bang tot nghiep",
        "degree certificate",
        "quyet dinh bo nhiem",
        "quyet dinh dieu chuyen",
        "quyet dinh tang luong",
        "quyet dinh khen thuong",
        "quyet dinh ky luat",
        "quyet dinh thoi viec",
    )
    has_hr_title = any(
        anchor in _search_text(block.text)
        for block in blocks[:8]
        if len(block.text) <= 200
        for anchor in title_anchors
    )
    if (
        not filename_is_cv
        and not has_hr_title
        and (filename_is_non_hr or non_hr_signal_count >= 2)
    ):
        return DOCUMENT_TYPE_OTHER, 0.95, "generic_document_rules", raw_scores

    best_type = max(raw_scores, key=raw_scores.get)
    best_score = raw_scores[best_type]
    if best_score < 0.45:
        if blocks:
            return DOCUMENT_TYPE_OTHER, 0.75, "generic_fallback", raw_scores
        return DOCUMENT_TYPE_UNKNOWN, 0.0, "rules", raw_scores

    second_score = sorted(raw_scores.values(), reverse=True)[1]
    margin = max(0.0, best_score - second_score)
    confidence = min(0.99, 0.55 + min(best_score, 1.5) * 0.2 + min(margin, 1.0) * 0.2)
    return best_type, round(confidence, 4), "rules", raw_scores


COMMON_PATTERNS: dict[str, tuple[str, ...]] = {
    "full_name": (
        r"(?:họ\s*(?:và\s*)?tên(?:\s*/\s*full\s*name)?|họ tên|full\s*name)\s*[:\-]\s*([^\n|;]{3,80})",
        r"(?:cấp cho|họ tên người được cấp)\s*[:\-]\s*([^\n|;]{3,80})",
    ),
    "employee_name": (
        r"(?:họ\s*(?:và\s*)?tên|họ tên|người lao động|ông\s*/\s*bà|ông|bà)\s*[:\-]\s*([^\n|;]{3,80})",
        r"(?:cấp cho|họ tên người được cấp)\s*[:\-]\s*([^\n|;]{3,80})",
    ),
    "identity_number": (
        r"(?:số(?:\s*/\s*no\.?)?|số\s*(?:cccd|cmnd)|cccd|cmnd|"
        r"số định danh|identity\s*(?:no|number))\s*[:\-]?\s*(\d[\d\s]{7,14}\d)",
        r"\b(\d{12})\b",
    ),
    "date_of_birth": (
        rf"(?:ngày sinh(?:\s*/\s*date of birth)?|sinh ngày|date of birth|dob)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "gender": (
        r"(?:giới tính(?:\s*/\s*sex)?|sex|gender)\s*[:\-]?\s*(nam|nữ|male|female)",
    ),
    "nationality": (
        r"(?:quốc tịch(?:\s*/\s*nationality)?|nationality)\s*[:\-]?\s*([^\n|;]{2,40})",
    ),
    "place_of_origin": (
        r"(?:quê quán(?:\s*/\s*place of origin)?|place of origin)\s*[:\-]?\s*([^\n|;]{3,160})",
    ),
    "address": (
        r"(?:địa chỉ(?: thường trú)?|"
        r"nơi thường trú(?:\s*/\s*place of residence)?|address)"
        r"\s*[:\-]?\s*([^\n|]{5,200})",
    ),
    "expiry_date": (
        rf"(?:có giá trị đến|ngày hết hạn|expiry date|date of expiry)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "phone": (
        r"(?:điện thoại|số điện thoại|phone|mobile)\s*[:\-]?\s*(\+?\d[\d .-]{7,14}\d)",
        r"\b(0\d(?:[\s.-]?\d){8,10})\b",
    ),
    "email": (
        r"(?:email|e-mail)\s*[:\-]?\s*([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",
        r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b",
    ),
    "employee_id": (
        r"(?:mã nhân viên|mã nv|employee\s*id)\s*[:\-]?\s*([A-Za-z0-9._/-]{2,30})",
    ),
    "department": (
        r"(?:phòng ban|đơn vị|bộ phận|department)\s*[:\-]?\s*([^\n|;]{2,100})",
    ),
    "position": (
        r"(?:chức danh|chức vụ|vị trí công việc|position|job title)\s*[:\-]?\s*([^\n|;]{2,100})",
    ),
    "contract_number": (
        r"(?:số hợp đồng|hợp đồng số|số|no\.?)\s*[:\-]\s*([^\s|;]{2,50})",
    ),
    "contract_type": (
        r"(?:loại hợp đồng|hình thức hợp đồng)\s*[:\-]?\s*([^\n|;]{3,100})",
    ),
    "start_date": (
        rf"(?:ngày bắt đầu|từ ngày|có hiệu lực từ)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "end_date": (
        rf"(?:ngày kết thúc|đến ngày|tới ngày)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "salary": (
        rf"(?:mức lương|lương cơ bản|tiền lương)\s*[:\-]?\s*{_MONEY_VALUE}",
    ),
    "decision_number": (
        r"(?:số|quyết định số)\s*[:\-]\s*([^\s|;]{2,50})",
    ),
    "decision_date": (
        rf"(?:ban hành ngày|ngày ký|ngày)\s*[:\-]?\s*({_DATE_VALUE})",
        rf"\b({_DATE_VALUE})\b",
    ),
    "effective_date": (
        rf"(?:ngày hiệu lực|có hiệu lực(?: kể)? từ ngày)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "from_date": (
        rf"(?:từ ngày|nghỉ từ)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "to_date": (
        rf"(?:đến ngày|tới ngày|nghỉ đến)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "number_of_days": (
        r"(?:số ngày nghỉ|tổng số ngày|thời gian nghỉ)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)",
    ),
    "reason": (
        r"(?:lý do(?: nghỉ)?|reason)\s*[:\-]\s*([^\n|]{3,300})",
    ),
    "education": (
        r"(?:trình độ học vấn|học vấn|education)\s*[:\-]\s*([^\n|]{3,300})",
    ),
    "experience": (
        r"(?:kinh nghiệm(?: làm việc| công tác)?|experience)\s*[:\-]\s*([^\n|]{3,500})",
    ),
    "skills": (
        r"(?:kỹ năng|skills?)\s*[:\-]\s*([^\n|]{2,500})",
    ),
    "qualification": (
        r"(?:tên văn bằng|văn bằng|chứng chỉ|trình độ|degree)\s*[:\-]?\s*([^\n|;]{3,150})",
    ),
    "major": (
        r"(?:ngành(?: đào tạo)?|chuyên ngành|major)\s*[:\-]?\s*([^\n|;]{2,120})",
    ),
    "institution": (
        r"(?:trường|cơ sở đào tạo|đơn vị cấp|institution)\s*[:\-]?\s*([^\n|;]{3,180})",
    ),
    "graduation_year": (
        r"(?:năm tốt nghiệp|graduation year)\s*[:\-]?\s*((?:19|20)\d{2})",
    ),
    "classification": (
        r"(?:xếp loại|classification)\s*[:\-]?\s*([^\n|;]{2,80})",
    ),
    "certificate_number": (
        r"(?:số hiệu|số vào sổ|certificate\s*(?:no|number))\s*[:\-]?\s*([^\s|;]{2,60})",
    ),
    "document_number": (
        r"(?:số văn bản|văn bản số|document\s*(?:no|number)|reference\s*(?:no|number))\s*[:\-]?\s*([^\s|;]{2,80})",
    ),
    "issued_date": (
        rf"(?:ngày ban hành|ban hành ngày|issued date|date issued)\s*[:\-]?\s*({_DATE_VALUE})",
    ),
    "organization": (
        r"(?:cơ quan|đơn vị|tổ chức|organization|company)\s*[:\-]\s*([^\n|;]{2,180})",
    ),
}


def _document_specific_value(field_name: str, text: str) -> str | None:
    searchable = _search_text(text)
    if field_name == "decision_type":
        choices = (
            ("bổ nhiệm", "appointment"),
            ("điều chuyển", "transfer"),
            ("tăng lương", "salary_adjustment"),
            ("khen thưởng", "reward"),
            ("kỷ luật", "discipline"),
            ("thôi việc", "termination"),
            ("nghỉ việc", "termination"),
        )
        for keyword, value in choices:
            if _search_text(keyword) in searchable:
                return value
    if field_name == "leave_type":
        choices = (
            ("nghỉ phép năm", "annual"),
            ("nghỉ ốm", "sick"),
            ("nghỉ không lương", "unpaid"),
            ("thai sản", "maternity"),
        )
        for keyword, value in choices:
            if _search_text(keyword) in searchable:
                return value
        if "nghi phep" in searchable:
            return "annual"
    if field_name == "qualification":
        choices = (
            ("bằng tốt nghiệp", "Bằng tốt nghiệp"),
            ("chứng chỉ", "Chứng chỉ"),
            ("chứng nhận", "Giấy chứng nhận"),
            ("diploma", "Diploma"),
            ("certificate", "Certificate"),
        )
        for keyword, value in choices:
            if _search_text(keyword) in searchable:
                return value
    return None


_CV_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "tom tat", "muc tieu nghe nghiep"),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "kinh nghiem",
        "kinh nghiem lam viec",
        "kinh nghiem cong tac",
    ),
    "skills": (
        "skills",
        "skills & background knowledge",
        "technical skills",
        "ky nang",
    ),
    "education": (
        "education",
        "academic background",
        "hoc van",
        "trinh do hoc van",
    ),
    "projects": ("projects", "personal projects", "du an"),
    "certificates": (
        "certificates",
        "certifications",
        "chung chi",
        "chung nhan",
    ),
    "awards": (
        "awards",
        "awards & activities",
        "activities",
        "honors",
        "giai thuong",
        "hoat dong",
    ),
}
_CV_SECTION_TITLES = {
    alias
    for aliases in _CV_SECTION_ALIASES.values()
    for alias in aliases
}


def _normalized_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9&]+", " ", _search_text(text)).strip()


def _find_cv_section(
    field_name: str,
    blocks: list[SourceBlock],
) -> tuple[str | None, SourceBlock | None]:
    aliases = _CV_SECTION_ALIASES.get(field_name, ())
    start_index: int | None = None
    for index, block in enumerate(blocks):
        if _normalized_heading(block.text) in aliases:
            start_index = index
            break
    if start_index is None:
        return None, None

    section_blocks: list[SourceBlock] = []
    for block in blocks[start_index + 1 :]:
        if _normalized_heading(block.text) in _CV_SECTION_TITLES:
            break
        section_blocks.append(block)
    value = "\n".join(block.text for block in section_blocks).strip()
    if not value:
        return None, None
    return value[:12000], section_blocks[0]


def _find_cv_contact_address(
    blocks: list[SourceBlock],
) -> tuple[str | None, SourceBlock | None]:
    for block in blocks[:8]:
        if "@" not in block.text:
            continue
        parts = [
            _plain_text(part).strip(" -")
            for part in re.split(r"\s*[•|]\s*", block.text)
        ]
        for part in reversed(parts):
            if (
                "," in part
                and "@" not in part
                and "http://" not in part.lower()
                and "https://" not in part.lower()
                and "www." not in part.lower()
                and not any(char.isdigit() for char in part)
            ):
                return part, block
    return None, None


def _find_field(
    field: FieldSchema,
    blocks: list[SourceBlock],
) -> tuple[str | None, SourceBlock | None, str, float]:
    patterns = COMMON_PATTERNS.get(field.name, ())
    for block in blocks:
        for pattern_index, pattern in enumerate(patterns):
            match = re.search(pattern, block.text, re.IGNORECASE)
            if match:
                value = _plain_text(match.group(1)).strip(" :-|")
                if value:
                    confidence = 0.94 if pattern_index == 0 else 0.82
                    return value, block, "regex_label", confidence

    if field.name == "document_title":
        source = next(
            (
                block
                for block in blocks
                if 2 <= len(block.text.strip()) <= 300
            ),
            blocks[0] if blocks else None,
        )
        if source is not None:
            return source.text, source, "first_content_rule", 0.75

    if field.name == "content_summary":
        summary_blocks = blocks[:8]
        value = "\n".join(block.text for block in summary_blocks).strip()
        if value:
            return value[:3000], summary_blocks[0], "content_summary_rule", 0.72

    if field.name in {"experience", "skills", "education"}:
        value, source = _find_cv_section(field.name, blocks)
        if value is not None:
            return value, source, "section_rule", 0.86

    if field.name == "address":
        value, source = _find_cv_contact_address(blocks)
        if value is not None:
            return value, source, "contact_line_rule", 0.76

    all_text = "\n".join(block.text for block in blocks)
    derived = _document_specific_value(field.name, all_text)
    if derived is not None:
        evidence = next(
            (
                block
                for block in blocks
                if _search_text(derived.replace("_", " ")) in _search_text(block.text)
            ),
            blocks[0] if blocks else None,
        )
        return derived, evidence, "keyword_rule", 0.82
    return None, None, "not_found", 0.0


def _normalize_date(value: str) -> str:
    numbers = [int(number) for number in re.findall(r"\d+", value)]
    if len(numbers) < 3:
        return value
    day, month, year = numbers[:3]
    if year < 100:
        year += 2000 if year < 50 else 1900
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return value


def _normalize_value(field: FieldSchema, value: str) -> Any:
    value = _SPACE_RE.sub(" ", value).strip()
    if field.data_type == "date":
        return _normalize_date(value)
    if field.data_type == "number":
        numeric = value.replace(",", ".")
        try:
            return int(numeric) if "." not in numeric else float(numeric)
        except ValueError:
            return value
    if field.data_type == "money":
        digits = re.sub(r"\D", "", re.sub(r"([.,])\d{1,2}$", "", value))
        return {
            "amount": int(digits) if digits else value,
            "currency": "VND",
        }
    if field.name == "identity_number":
        return re.sub(r"\D", "", value)
    if field.name == "phone":
        prefix = "+" if value.strip().startswith("+") else ""
        return prefix + re.sub(r"\D", "", value)
    if field.name == "email":
        return value.lower()
    return value


def _field_result(field: FieldSchema, blocks: list[SourceBlock]) -> dict[str, Any]:
    raw_value, source, method, confidence = _find_field(field, blocks)
    result: dict[str, Any] = {
        "value": _normalize_value(field, raw_value) if raw_value is not None else None,
        "raw_value": raw_value,
        "confidence": confidence,
        "required": field.required,
        "sensitive": field.sensitive,
        "extraction_method": method,
        "evidence": None,
    }
    if source is not None:
        result["evidence"] = {
            "page": source.page,
            "bbox": source.bbox,
            "source_text": source.text[:500],
            "content_index": source.index,
        }
    return result


def _name_from_document_name(document_name: str | None) -> str | None:
    if not document_name or " - " not in document_name:
        return None
    candidate = document_name.rsplit(" - ", 1)[-1].strip(" _-.")
    words = candidate.split()
    if not 2 <= len(words) <= 7 or any(char.isdigit() for char in candidate):
        return None
    if not all(
        all(char.isalpha() or char in "'’-" for char in word)
        for word in words
    ):
        return None
    return candidate


def _apply_document_fallbacks(
    document_type: str,
    fields: dict[str, dict[str, Any]],
    document_name: str | None,
) -> None:
    if document_type != "cv":
        return
    full_name = fields.get("full_name")
    if full_name is None or full_name["value"] not in (None, ""):
        return
    candidate = _name_from_document_name(document_name)
    if candidate is None:
        return
    full_name.update(
        {
            "value": candidate,
            "raw_value": candidate,
            "confidence": 0.78,
            "extraction_method": "filename_rule",
        }
    )


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _validate_fields(
    schema: DocumentSchema | None,
    fields: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if schema is None:
        issues.append(
            {
                "code": "DOCUMENT_TYPE_UNKNOWN",
                "field": "document_type",
                "severity": "high",
                "message": "Không xác định được loại tài liệu HCNS.",
            }
        )
        return "needs_review", issues

    for field in schema.fields:
        result = fields[field.name]
        if field.required and result["value"] in (None, ""):
            issues.append(
                {
                    "code": "REQUIRED_FIELD_MISSING",
                    "field": field.name,
                    "severity": "high",
                    "message": f"Thiếu trường bắt buộc: {field.name}.",
                }
            )
        elif result["value"] not in (None, "") and result["confidence"] < 0.7:
            issues.append(
                {
                    "code": "LOW_CONFIDENCE",
                    "field": field.name,
                    "severity": "medium",
                    "message": f"Trường {field.name} cần được kiểm tra lại.",
                }
            )

    if schema.document_type == "cv":
        if not any(
            fields.get(name, {}).get("value")
            for name in ("phone", "email")
        ):
            issues.append(
                {
                    "code": "CV_CONTACT_MISSING",
                    "field": "phone_or_email",
                    "severity": "high",
                    "message": "CV thiếu cả số điện thoại và email.",
                }
            )
        if not any(
            fields.get(name, {}).get("value")
            for name in ("education", "experience")
        ):
            issues.append(
                {
                    "code": "CV_PROFILE_SECTION_MISSING",
                    "field": "education_or_experience",
                    "severity": "high",
                    "message": "CV thiếu cả thông tin học vấn và kinh nghiệm.",
                }
            )

    identity = fields.get("identity_number", {}).get("value")
    if identity and (not str(identity).isdigit() or len(str(identity)) not in (9, 12)):
        issues.append(
            {
                "code": "INVALID_IDENTITY_NUMBER",
                "field": "identity_number",
                "severity": "high",
                "message": "Số CCCD/CMND phải gồm 9 hoặc 12 chữ số.",
            }
        )

    date_pairs = (
        ("start_date", "end_date"),
        ("from_date", "to_date"),
    )
    for start_field, end_field in date_pairs:
        start = _parse_iso_date(fields.get(start_field, {}).get("value"))
        end = _parse_iso_date(fields.get(end_field, {}).get("value"))
        if start and end and end < start:
            issues.append(
                {
                    "code": "INVALID_DATE_RANGE",
                    "field": end_field,
                    "severity": "high",
                    "message": f"{end_field} không được trước {start_field}.",
                }
            )

    return ("valid" if not issues else "needs_review"), issues


class HRIDPProcessor:
    """Process OCR content lists into structured HR business data."""

    def process(
        self,
        content_list: Iterable[dict[str, Any]],
        document_type: str = DOCUMENT_TYPE_AUTO,
        document_name: str | None = None,
    ) -> dict[str, Any]:
        requested_type = validate_document_type(document_type)
        blocks = build_source_blocks(content_list)
        detected_type, confidence, method, scores = _classification(
            blocks,
            requested_type,
            document_name,
        )
        schema = get_document_schema(detected_type)
        fields = (
            {field.name: _field_result(field, blocks) for field in schema.fields}
            if schema is not None
            else {}
        )
        _apply_document_fallbacks(detected_type, fields, document_name)
        status, issues = _validate_fields(schema, fields)
        requires_review = (
            status != "valid"
            or detected_type == DOCUMENT_TYPE_UNKNOWN
            or confidence < 0.75
        )
        return {
            "schema_version": IDP_SCHEMA_VERSION,
            "domain": (
                "general_documents"
                if detected_type == DOCUMENT_TYPE_OTHER
                else "human_resources"
            ),
            "document_name": document_name,
            "classification": {
                "document_type": detected_type,
                "label": schema.label if schema else "Không xác định",
                "confidence": confidence,
                "method": method,
                "scores": scores,
            },
            "fields": fields,
            "validation": {
                "status": status,
                "requires_review": requires_review,
                "issues": issues,
            },
            "metadata": {
                "source_block_count": len(blocks),
                "processor": "vsf_hr_rules_v1",
            },
        }


def process_hr_document(
    content_list: Iterable[dict[str, Any]],
    document_type: str = DOCUMENT_TYPE_AUTO,
    document_name: str | None = None,
) -> dict[str, Any]:
    """Convenience entry point for HR IDP."""

    return HRIDPProcessor().process(
        content_list=content_list,
        document_type=document_type,
        document_name=document_name,
    )
