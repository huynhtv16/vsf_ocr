from vsf.idp import HRIDPProcessor, process_hr_document


def text_item(text, page_idx=0, bbox=None):
    item = {
        "type": "text",
        "text": text,
        "page_idx": page_idx,
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def test_classifies_and_extracts_labor_contract():
    result = process_hr_document(
        [
            text_item("HỢP ĐỒNG LAO ĐỘNG\nSố hợp đồng: 12/2026/HĐLĐ"),
            text_item(
                "Người lao động: Nguyễn Văn An\n"
                "CCCD: 079095001234\n"
                "Chức danh: Kỹ sư phần mềm\n"
                "Phòng ban: Công nghệ",
                bbox=[100, 200, 800, 500],
            ),
            text_item(
                "Từ ngày: 01/08/2026\n"
                "Đến ngày: 31/07/2027\n"
                "Mức lương: 15.000.000 VNĐ"
            ),
        ],
        document_name="hop_dong_an",
    )

    assert result["classification"]["document_type"] == "labor_contract"
    assert result["classification"]["confidence"] >= 0.75
    assert result["fields"]["contract_number"]["value"] == "12/2026/HĐLĐ"
    assert result["fields"]["employee_name"]["value"] == "Nguyễn Văn An"
    assert result["fields"]["identity_number"]["value"] == "079095001234"
    assert result["fields"]["start_date"]["value"] == "2026-08-01"
    assert result["fields"]["salary"]["value"] == {
        "amount": 15000000,
        "currency": "VND",
    }
    assert result["fields"]["employee_name"]["evidence"]["page"] == 1
    assert result["fields"]["employee_name"]["evidence"]["bbox"] == [
        100,
        200,
        800,
        500,
    ]
    assert result["validation"]["status"] == "valid"


def test_leave_request_reports_invalid_date_range():
    result = HRIDPProcessor().process(
        [
            text_item("ĐƠN XIN NGHỈ PHÉP"),
            text_item(
                "Họ và tên: Trần Thị Bình\n"
                "Loại nghỉ: nghỉ phép năm\n"
                "Từ ngày: 10/09/2026\n"
                "Đến ngày: 08/09/2026"
            ),
        ]
    )

    assert result["classification"]["document_type"] == "leave_request"
    assert result["fields"]["leave_type"]["value"] == "annual"
    assert result["validation"]["requires_review"] is True
    assert any(
        issue["code"] == "INVALID_DATE_RANGE"
        for issue in result["validation"]["issues"]
    )


def test_forced_identity_schema_reports_missing_required_fields():
    result = process_hr_document(
        [text_item("Họ và tên: Lê Văn C")],
        document_type="identity_card",
    )

    assert result["classification"] == {
        "document_type": "identity_card",
        "label": "Căn cước công dân/CMND",
        "confidence": 1.0,
        "method": "user_selected",
        "scores": {"identity_card": 1.0},
    }
    assert result["fields"]["full_name"]["value"] == "Lê Văn C"
    missing = {
        issue["field"]
        for issue in result["validation"]["issues"]
        if issue["code"] == "REQUIRED_FIELD_MISSING"
    }
    assert {"identity_number", "date_of_birth"} <= missing


def test_unrecognized_document_uses_generic_schema():
    result = process_hr_document(
        [text_item("Biên bản kiểm kê thiết bị tại kho số 3")]
    )

    assert result["classification"]["document_type"] == "other_document"
    assert result["domain"] == "general_documents"
    assert result["classification"]["label"] == "Tài liệu khác (không giới hạn mẫu)"
    assert result["fields"]["document_title"]["value"] == (
        "Biên bản kiểm kê thiết bị tại kho số 3"
    )
    assert result["fields"]["content_summary"]["value"] == (
        "Biên bản kiểm kê thiết bị tại kho số 3"
    )
    assert result["validation"]["status"] == "valid"
    assert result["validation"]["requires_review"] is False


def test_extracts_bilingual_identity_card_labels():
    result = process_hr_document(
        [
            text_item("CĂN CƯỚC CÔNG DÂN\nSố / No.: 079095001234"),
            text_item(
                "Họ và tên / Full name: NGUYỄN VĂN AN\n"
                "Ngày sinh / Date of birth: 20/06/1995\n"
                "Giới tính / Sex: Nam\n"
                "Quốc tịch / Nationality: Việt Nam"
            ),
        ]
    )

    assert result["classification"]["document_type"] == "identity_card"
    assert result["fields"]["full_name"]["value"] == "NGUYỄN VĂN AN"
    assert result["fields"]["identity_number"]["value"] == "079095001234"
    assert result["fields"]["date_of_birth"]["value"] == "1995-06-20"


def test_extracts_hr_appointment_decision():
    result = process_hr_document(
        [
            text_item("QUYẾT ĐỊNH BỔ NHIỆM\nSố: 28/2026/QĐ-VSF"),
            text_item(
                "Ngày ký: 27/07/2026\n"
                "Ông: Phạm Minh Đức\n"
                "Chức vụ: Trưởng phòng Nhân sự\n"
                "Đơn vị: Phòng Nhân sự"
            ),
        ]
    )

    assert result["classification"]["document_type"] == "hr_decision"
    assert result["fields"]["decision_type"]["value"] == "appointment"
    assert result["fields"]["decision_number"]["value"] == "28/2026/QĐ-VSF"
    assert result["fields"]["decision_date"]["value"] == "2026-07-27"
    assert result["fields"]["employee_name"]["value"] == "Phạm Minh Đức"
    assert result["validation"]["status"] == "valid"


def test_extracts_degree_certificate_fallback():
    result = process_hr_document(
        [
            text_item("BẰNG TỐT NGHIỆP ĐẠI HỌC"),
            text_item(
                "Cấp cho: Đỗ Thu Hà\n"
                "Ngành đào tạo: Công nghệ thông tin\n"
                "Trường: Đại học Ví dụ\n"
                "Năm tốt nghiệp: 2025"
            ),
        ]
    )

    assert result["classification"]["document_type"] == "degree_certificate"
    assert result["fields"]["employee_name"]["value"] == "Đỗ Thu Hà"
    assert result["fields"]["qualification"]["value"] == "Bằng tốt nghiệp"
    assert result["fields"]["major"]["value"] == "Công nghệ thông tin"
    assert result["validation"]["status"] == "valid"


def test_english_cv_with_certificate_word_is_not_a_degree():
    result = process_hr_document(
        [
            text_item("0386 917 776 • huynhtv.vn@gmail.com • linkedin.com/in/huynhtv"),
            text_item("Summary"),
            text_item("Backend Engineer with experience in Java and Spring Boot."),
            text_item("Experience"),
            text_item("Configured HTTPS certificate and deployment health checks."),
            text_item("Education"),
            text_item("Skills"),
        ],
        document_name="MLOps Engineer Intern - Tran Van Huynh",
    )

    assert result["classification"]["document_type"] == "cv"
    assert result["fields"]["full_name"]["value"] == "Tran Van Huynh"
    assert result["fields"]["full_name"]["extraction_method"] == "filename_rule"
    assert result["fields"]["phone"]["value"] == "0386917776"
    assert result["fields"]["email"]["value"] == "huynhtv.vn@gmail.com"
    assert result["validation"]["status"] == "valid"


def test_vocabulary_workbook_uses_generic_schema():
    result = process_hr_document(
        [
            text_item("Mục lục"),
            {
                "type": "table",
                "table_body": (
                    "Beginner's TOEIC - Word Bank | Quizlet | Lesson 1 | "
                    "summary | experience | education | certificate | chứng chỉ"
                ),
                "page_idx": 0,
            },
            text_item("Danh sách từ vựng (Official)"),
        ],
        document_name="Be_Danh sách từ vựng",
    )

    assert result["classification"]["document_type"] == "other_document"
    assert result["domain"] == "general_documents"
    assert result["classification"]["method"] == "generic_document_rules"
    assert result["fields"]["document_title"]["value"] == "Mục lục"
    assert "Beginner's TOEIC" in result["fields"]["content_summary"]["value"]
    assert result["validation"]["status"] == "valid"


def test_extracts_cv_contact_address_and_sections():
    result = process_hr_document(
        [
            text_item(
                "0386 917 776 • huynhtv.vn@gmail.com • "
                "www.linkedin.com/in/huynhtv • Cau Giay, Hanoi, Vietnam"
            ),
            text_item("Summary"),
            text_item("Backend Engineer pursuing a career in MLOps."),
            text_item("Experience"),
            text_item("VinUni - Applied AI Talent Program"),
            text_item("MLOps Engineer / AI Engineer"),
            text_item("Skills & Background Knowledge"),
            text_item("Docker, Kubernetes, Python, Java"),
            text_item("Education"),
            text_item("East Asia University of Technology"),
            text_item("Engineering, Software technology"),
            text_item("Certificates"),
            text_item("TOEIC 530"),
        ],
        document_name="MLOps Engineer Intern - Tran Van Huynh",
    )

    assert result["fields"]["address"]["value"] == "Cau Giay, Hanoi, Vietnam"
    assert result["fields"]["address"]["extraction_method"] == "contact_line_rule"
    assert "VinUni - Applied AI Talent Program" in result["fields"]["experience"]["value"]
    assert "Docker, Kubernetes" in result["fields"]["skills"]["value"]
    assert (
        "East Asia University of Technology"
        in result["fields"]["education"]["value"]
    )
    assert "TOEIC 530" not in result["fields"]["education"]["value"]
    assert result["fields"]["education"]["extraction_method"] == "section_rule"
    assert result["validation"]["status"] == "valid"


def test_cv_without_contact_or_profile_sections_requires_review():
    result = process_hr_document(
        [text_item("Summary"), text_item("A short professional profile.")],
        document_name="CV - Nguyen Van An",
    )

    issue_codes = {issue["code"] for issue in result["validation"]["issues"]}
    assert "CV_CONTACT_MISSING" in issue_codes
    assert "CV_PROFILE_SECTION_MISSING" in issue_codes
    assert result["validation"]["status"] == "needs_review"
