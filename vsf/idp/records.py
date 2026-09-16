"""Normalize table content into row records for downstream IDP workflows."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any, Iterable

from bs4 import BeautifulSoup


HEADER_ALIASES = {
    "ma nv": "employee_id",
    "ma nhan vien": "employee_id",
    "employee id": "employee_id",
    "id nhan vien": "employee_id",
    "ho va ten": "employee_name",
    "ho ten": "employee_name",
    "ten nhan vien": "employee_name",
    "employee name": "employee_name",
    "ngay sinh": "date_of_birth",
    "date of birth": "date_of_birth",
    "sdt": "phone",
    "so dien thoai": "phone",
    "dien thoai": "phone",
    "phone": "phone",
    "email": "email",
    "phong ban": "department",
    "bo phan": "department",
    "department": "department",
    "chuc vu": "position",
    "chuc danh": "position",
    "position": "position",
    "luong co ban": "base_salary",
    "base salary": "base_salary",
    "phu cap": "allowance",
    "allowance": "allowance",
    "khau tru": "deduction",
    "deduction": "deduction",
    "thuc linh": "net_salary",
    "luong thuc nhan": "net_salary",
    "net salary": "net_salary",
    "ngay cong": "work_days",
    "so ngay cong": "work_days",
    "working days": "work_days",
    "gio tang ca": "overtime_hours",
    "tang ca": "overtime_hours",
    "overtime hours": "overtime_hours",
}
HEADER_HINTS = {
    "stt",
    "ma",
    "id",
    "ten",
    "ho va ten",
    "ngay",
    "thang",
    "nam",
    "noi dung",
    "ghi chu",
    "trang thai",
    "phong ban",
    "chuc vu",
    "so luong",
    "don gia",
    "thanh tien",
    "link",
    "deadline",
    "tuan",
    "buoi",
}


def _search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _key(value: str, fallback_index: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", _search_text(value)).strip()
    if normalized in HEADER_ALIASES:
        return HEADER_ALIASES[normalized]
    slug = normalized.replace(" ", "_")
    return slug[:80] or f"column_{fallback_index + 1}"


def _deduplicate_keys(keys: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
        result.append(key if counts[key] == 1 else f"{key}_{counts[key]}")
    return result


def _normalize_record_value(key: str, value: str) -> Any:
    stripped = value.strip()
    if key == "phone":
        prefix = "+" if stripped.startswith("+") else ""
        digits = re.sub(r"\D", "", stripped)
        return prefix + digits if digits else stripped
    if key in {
        "base_salary",
        "allowance",
        "deduction",
        "net_salary",
    }:
        digits = re.sub(r"\D", "", stripped)
        if digits:
            return {"amount": int(digits), "currency": "VND"}
    if key in {"work_days", "overtime_hours"}:
        numeric = stripped.replace(",", ".")
        try:
            return int(numeric) if "." not in numeric else float(numeric)
        except ValueError:
            return stripped
    if key == "date_of_birth":
        numbers = [int(number) for number in re.findall(r"\d+", stripped)]
        if len(numbers) >= 3:
            day, month, year = numbers[:3]
            if year < 100:
                year += 2000 if year < 50 else 1900
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass
    return stripped


def _html_table_grid(table_html: str, max_columns: int) -> tuple[list[list[str]], list[int]]:
    soup = BeautifulSoup(table_html, "lxml")
    rows: list[list[str]] = []
    header_counts: list[int] = []
    active_spans: dict[int, tuple[int, str]] = {}

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells and tr.find_all(["th", "td"]):
            cells = tr.find_all(["th", "td"])
        row_values: dict[int, str] = {
            column: value for column, (_, value) in active_spans.items()
        }
        next_spans: dict[int, tuple[int, str]] = {
            column: (remaining - 1, value)
            for column, (remaining, value) in active_spans.items()
            if remaining > 1
        }
        column = 0
        header_count = 0
        for cell in cells:
            while column in row_values:
                column += 1
            if column >= max_columns:
                break
            value = " ".join(cell.stripped_strings)
            colspan = max(1, int(cell.get("colspan", 1) or 1))
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            if cell.name == "th":
                header_count += min(colspan, max_columns - column)
            for offset in range(colspan):
                target = column + offset
                if target >= max_columns:
                    break
                row_values[target] = value
                if rowspan > 1:
                    next_spans[target] = (rowspan - 1, value)
            column += colspan
        active_spans = next_spans
        if row_values:
            width = min(max(row_values) + 1, max_columns)
            rows.append([row_values.get(index, "") for index in range(width)])
            header_counts.append(header_count)
    return rows, header_counts


def _select_header_index(rows: list[list[str]], header_counts: list[int]) -> int:
    def score(index: int) -> tuple[float, int]:
        values = [value.strip() for value in rows[index] if value.strip()]
        normalized = [_search_text(value) for value in values]
        unique_values = set(normalized)
        duplicate_count = len(normalized) - len(unique_values)
        short_count = sum(len(value) <= 48 for value in values)
        long_count = sum(len(value) > 100 for value in values)
        alias_count = sum(value in HEADER_ALIASES for value in normalized)
        hint_count = sum(
            value in HEADER_HINTS
            or any(
                value.startswith(f"{hint} ")
                or value.endswith(f" {hint}")
                for hint in HEADER_HINTS
            )
            for value in normalized
        )
        weighted = (
            alias_count * 8
            + hint_count * 4
            + len(unique_values) * 1.5
            + short_count
            + header_counts[index] * 0.15
            - duplicate_count * 3
            - long_count * 2
        )
        return weighted, index

    candidates = range(min(8, len(rows)))
    return max(candidates, key=score, default=0)


def _normalize_table(
    item: dict[str, Any],
    table_index: int,
    *,
    max_rows: int,
    max_columns: int,
) -> dict[str, Any] | None:
    body = str(item.get("table_body") or "")
    if "<table" not in body.lower():
        return None
    rows, header_counts = _html_table_grid(body, max_columns)
    if not rows:
        return None

    header_index = _select_header_index(rows, header_counts)
    headers = rows[header_index]
    width = min(max(len(row) for row in rows), max_columns)
    headers = headers + [""] * (width - len(headers))
    keys = _deduplicate_keys([_key(header, index) for index, header in enumerate(headers)])
    records = []
    for row in rows[header_index + 1 :]:
        padded = row + [""] * (width - len(row))
        values = padded[:width]
        if not any(value.strip() for value in values):
            continue
        if all(
            _search_text(value.strip()) == _search_text(header.strip())
            for value, header in zip(values, headers)
            if value.strip() or header.strip()
        ):
            continue
        record = {
            key: _normalize_record_value(key, value)
            for key, value in zip(keys, values)
            if value not in ("", None)
        }
        if record:
            records.append(record)
        if len(records) >= max_rows:
            break

    nonempty_headers = sum(bool(header.strip()) for header in headers)
    confidence = 0.45
    if nonempty_headers >= 2:
        confidence += 0.2
    if records:
        confidence += 0.15
    if any(key in HEADER_ALIASES.values() for key in keys):
        confidence += 0.15
    return {
        "table_index": table_index,
        "page": item.get("page_idx", 0) + 1
        if isinstance(item.get("page_idx"), int)
        else None,
        "bbox": item.get("bbox") if isinstance(item.get("bbox"), list) else None,
        "headers": headers,
        "normalized_headers": keys,
        "source_row_count": max(0, len(rows) - header_index - 1),
        "record_count": len(records),
        "records": records,
        "truncated": len(rows) - header_index - 1 > max_rows,
        "confidence": round(min(confidence, 0.95), 4),
    }


def extract_structured_tables(
    content_list: Iterable[dict[str, Any]],
    *,
    max_tables: int = 50,
    max_rows_per_table: int = 500,
    max_columns: int = 256,
) -> list[dict[str, Any]]:
    """Extract bounded, normalized row records from HTML tables."""

    tables = []
    for item in content_list:
        if not isinstance(item, dict) or item.get("type") != "table":
            continue
        normalized = _normalize_table(
            item,
            len(tables),
            max_rows=max_rows_per_table,
            max_columns=max_columns,
        )
        if normalized is not None:
            tables.append(normalized)
        if len(tables) >= max_tables:
            break
    return tables
