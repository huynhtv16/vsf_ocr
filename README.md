# VSF OCR

> Công cụ phân tích tài liệu thông minh được duy trì bởi **Tran Van Huynh**.

Dự án chuyển đổi PDF, hình ảnh và tài liệu Office thành Markdown hoặc JSON có
cấu trúc. Kết quả có thể được sử dụng để xây dựng hệ thống RAG, tìm kiếm tài
liệu, trích xuất thông tin và các quy trình Intelligent Document Processing
(IDP).

## Mục tiêu

Dự án tập trung vào lớp đọc và chuẩn hóa tài liệu:

```text
PDF / Hình ảnh / PPTX / XLSX
                    │
                    ▼
        OCR và phân tích bố cục
                    │
                    ▼
    Văn bản / Bảng / Hình / Công thức
                    │
                    ▼
          Markdown và JSON có cấu trúc
                    │
                    ▼
       RAG / LLM / Search / IDP / ETL
```

VSF OCR không phải là một nền tảng IDP nghiệp vụ hoàn
chỉnh. Dự án cung cấp lõi đọc tài liệu; các hệ thống sử dụng nó có thể bổ sung
schema nghiệp vụ, validation, human review và tích hợp ERP/CRM.

## Tính năng

- Đọc PDF thông thường và PDF scan.
- Hỗ trợ PNG, JPG, JPEG, TIFF, WebP, GIF, BMP và JP2.
- Đọc trực tiếp PPTX và XLSX.
- Nhận diện bố cục một cột, nhiều cột và bố cục phức tạp.
- Sắp xếp nội dung theo thứ tự đọc tự nhiên.
- Loại bỏ header, footer, số trang và một số nội dung thừa.
- Nhận diện tiêu đề, đoạn văn, danh sách và mục lục.
- Trích xuất hình ảnh, biểu đồ, chú thích và footnote.
- Chuyển công thức toán học thành LaTeX.
- Chuyển bảng thành HTML và JSON có cấu trúc.
- OCR đa ngôn ngữ.
- Chạy bằng CPU, NVIDIA CUDA hoặc Apple Silicon.
- Cung cấp CLI, REST API, Gradio WebUI và router đa GPU.

## Các backend

| Backend | Mô tả | Phần cứng phù hợp |
|---|---|---|
| `pipeline` | Kết hợp các model chuyên biệt cho layout, OCR, bảng và công thức | CPU hoặc GPU có VRAM thấp |
| `vlm-engine` | Dùng Vision-Language Model để hiểu toàn bộ trang | GPU có VRAM lớn |
| `hybrid-engine` | Kết hợp pipeline và VLM | GPU mạnh, ưu tiên độ chính xác |
| `vlm-http-client` | Gửi suy luận VLM tới server tương thích OpenAI | Máy client cấu hình thấp |
| `hybrid-http-client` | Pipeline cục bộ kết hợp VLM từ server từ xa | Client có PyTorch và server VLM |

Với NVIDIA GTX 1650 4 GB, nên sử dụng `pipeline`. `vlm-engine` và
`hybrid-engine` có thể vượt quá dung lượng VRAM.

## Yêu cầu hệ thống

- Python từ 3.10 đến 3.13.
- Linux, Windows hoặc macOS.
- RAM khuyến nghị từ 8 GB.
- Dung lượng trống dành cho dependency và model.
- NVIDIA driver hoạt động nếu muốn sử dụng CUDA.

Kiểm tra GPU:

```bash
nvidia-smi
```

## Cài đặt

### 1. Clone repository

```bash
git clone <URL_REPOSITORY_CUA_BAN>
cd MinerU
```

### 2. Tạo virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Trên Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Cài pipeline và WebUI

```bash
pip install -e ".[pipeline,gradio]"
```

Để cài toàn bộ thành phần tùy chọn:

```bash
pip install -e ".[all]"
```

### 4. Tải model

Tải model cho backend `pipeline`:

```bash
mineru-models-download \
  --source huggingface \
  --model_type pipeline
```

Nếu cần VLM và máy có đủ VRAM:

```bash
mineru-models-download \
  --source huggingface \
  --model_type vlm
```

Sau khi tải, đường dẫn model được lưu trong `mineru.json` tại thư mục người
dùng.

## Chạy WebUI bằng GPU

```bash
MINERU_MODEL_SOURCE=local \
MINERU_DEVICE_MODE=cuda \
MINERU_API_MAX_CONCURRENT_REQUESTS=1 \
mineru-gradio \
  --server-name 127.0.0.1 \
  --server-port 7860
```

Mở trình duyệt tại:

```text
http://127.0.0.1:7860
```

Trong giao diện:

1. Tải tài liệu lên.
2. Chọn backend `pipeline` nếu GPU có 4 GB VRAM.
3. Để parsing method là `auto`.
4. Bật hoặc tắt nhận diện bảng và công thức tùy nhu cầu.
5. Bắt đầu chuyển đổi và tải kết quả.

## Sử dụng CLI

Phân tích một tài liệu:

```bash
mineru \
  -p input/document.pdf \
  -o output \
  -b pipeline
```

Phân tích toàn bộ tài liệu trong một thư mục:

```bash
mineru \
  -p input \
  -o output \
  -b pipeline
```

Chỉ xử lý trang đầu tiên:

```bash
mineru \
  -p input/document.pdf \
  -o output \
  -b pipeline \
  --start 0 \
  --end 0
```

Buộc sử dụng CUDA:

```bash
MINERU_MODEL_SOURCE=local \
MINERU_DEVICE_MODE=cuda \
mineru \
  -p input/document.pdf \
  -o output \
  -b pipeline
```

## Chạy REST API

Khởi động FastAPI:

```bash
MINERU_MODEL_SOURCE=local \
MINERU_DEVICE_MODE=cuda \
MINERU_API_MAX_CONCURRENT_REQUESTS=1 \
mineru-api \
  --host 127.0.0.1 \
  --port 8000
```

Tài liệu Swagger:

```text
http://127.0.0.1:8000/docs
```

Các endpoint chính:

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái service |
| `POST` | `/tasks` | Gửi tác vụ bất đồng bộ |
| `GET` | `/tasks/{task_id}` | Kiểm tra trạng thái tác vụ |
| `GET` | `/tasks/{task_id}/result` | Lấy kết quả tác vụ |
| `POST` | `/file_parse` | Phân tích đồng bộ |

Ví dụ gửi tài liệu:

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -F "files=@input/document.pdf" \
  -F "backend=pipeline" \
  -F "return_md=true"
```

## Cấu trúc kết quả

Tùy backend và tham số, thư mục kết quả có thể chứa:

| Tệp | Nội dung |
|---|---|
| `document.md` | Nội dung Markdown cuối cùng |
| `document_middle.json` | Cấu trúc trung gian chi tiết theo trang và block |
| `document_content_list.json` | Danh sách nội dung theo thứ tự đọc |
| `document_content_list_v2.json` | Định dạng danh sách nội dung phiên bản mới |
| `document_model.json` | Kết quả suy luận thô của model |
| `document_layout.pdf` | PDF trực quan hóa vùng bố cục |
| `document_span.pdf` | PDF trực quan hóa text span |
| `images/` | Hình ảnh và vùng nội dung được trích xuất |

Ví dụ cấu trúc:

```text
output/
└── document/
    └── auto/
        ├── document.md
        ├── document_middle.json
        ├── document_content_list.json
        ├── document_content_list_v2.json
        ├── document_model.json
        ├── document_layout.pdf
        ├── document_span.pdf
        └── images/
```

## Ứng dụng IDP

Dự án có thể làm parsing engine cho một hệ thống IDP:

```text
Tài liệu đầu vào
→ VSF OCR
→ Markdown/JSON
→ Trích xuất theo schema
→ Validation
→ Human review
→ Database/ERP/CRM
```

Ví dụ với hóa đơn, lớp nghiệp vụ phía sau có thể lấy:

```json
{
  "invoice_number": "INV-2026-001",
  "invoice_date": "2026-07-24",
  "vendor": "Example Company",
  "total": 1250000,
  "currency": "VND"
}
```

MinerU chịu trách nhiệm đọc và chuẩn hóa tài liệu. Schema, quy tắc kiểm tra và
quy trình phê duyệt cần được xây dựng theo từng bài toán.

## Khắc phục lỗi thường gặp

### `Local path for repo_mode 'vlm' is not configured`

Tác vụ đang sử dụng `hybrid-engine` hoặc `vlm-engine`, nhưng model VLM chưa
được tải.

Với GPU 4 GB, đổi backend thành:

```text
pipeline
```

Nếu máy có đủ VRAM và thực sự cần VLM:

```bash
mineru-models-download \
  --source huggingface \
  --model_type vlm
```

### CUDA không khả dụng

Kiểm tra:

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, kiểm tra lại NVIDIA driver và phiên bản PyTorch CUDA.

### CUDA out of memory

- Chuyển sang backend `pipeline`.
- Giới hạn số request đồng thời:

  ```bash
  export MINERU_API_MAX_CONCURRENT_REQUESTS=1
  ```

- Tắt nhận diện công thức hoặc bảng nếu không cần.
- Chỉ xử lý một khoảng trang trong mỗi tác vụ.

### Lần chạy đầu tiên chậm

Lần đầu MinerU cần tải và khởi tạo model. Sử dụng
`mineru-models-download` trước khi khởi động service để tránh tải model trong
lúc xử lý tài liệu.

## Cấu trúc mã nguồn

```text
mineru/
├── backend/
│   ├── pipeline/    # Pipeline OCR, layout, bảng và công thức
│   ├── vlm/         # Vision-Language Model
│   ├── hybrid/      # Kết hợp pipeline và VLM
│   └── office/      # PPTX và XLSX
├── cli/             # CLI, FastAPI, Gradio và router
├── data/            # Lớp đọc/ghi dữ liệu
├── model/           # Model và inference adapter
├── resources/       # Tài nguyên giao diện và ngôn ngữ
└── utils/           # Hàm tiện ích dùng chung
```

## Kiểm thử

Chạy test:

```bash
pytest
```

Chạy bài test end-to-end:

```bash
pytest tests/unittest/test_e2e.py
```

## Người duy trì

**Tran Van Huynh**

Bản fork này được duy trì và tùy chỉnh cho mục đích học tập, nghiên cứu và xây
dựng các hệ thống xử lý tài liệu thông minh.
