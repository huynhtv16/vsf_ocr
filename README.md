# VSF OCR

VSF OCR là một hệ thống xử lý tài liệu hướng tới OCR, nhận diện bố cục, trích xuất bảng và công thức, và chuyển đổi tài liệu đầu vào thành Markdown/JSON có cấu trúc.

Dự án này không chỉ dừng ở việc "đọc chữ" trên trang PDF hoặc ảnh. Nó cố gắng hiểu từng phần của tài liệu: tiêu đề, đoạn văn, danh sách, bảng, hình ảnh, chú thích, footnote, công thức toán học, và thứ tự đọc tự nhiên của trang. Từ đó, tài liệu được chuẩn hóa thành định dạng dễ làm nghiệp vụ, tìm kiếm, lọc, hay đưa vào hệ thống OCR/IDP/ETL.

## Mục tiêu của dự án

VSF tập trung vào tầng xử lý tài liệu thực tế:

```text
PDF / ảnh / PPTX / XLSX
        │
        ▼
   OCR + layout detection
        │
        ▼
  text / table / image / formula
        │
        ▼
   Markdown + JSON có cấu trúc
        │
        ▼
  Search / Automation / IDP / review workflows
```

Điểm quan trọng là: dự án này xây dựng một pipeline tài liệu từ đầu vào đến đầu ra, không phải một hệ thống RAG đơn thuần. Mỗi trang được phân tích như một "bộ dữ liệu tài liệu" có structure, bounding box, block ordering, và metadata.

---

## Tính năng chính

- Hỗ trợ PDF thông thường và PDF scan.
- Hỗ trợ ảnh: PNG, JPG, JPEG, TIFF, WebP, GIF, BMP, JP2.
- Hỗ trợ đọc trực tiếp các tài liệu Office: PPTX, XLSX.
- Nhận diện layout trang: 1 cột, nhiều cột, bố cục phức tạp.
- Sắp xếp nội dung theo thứ tự đọc tự nhiên.
- Loại bỏ header, footer, số trang và các nội dung không cần thiết.
- Nhận diện tiêu đề, đoạn văn, danh sách, mục lục.
- Trích xuất hình ảnh, biểu đồ, chú thích và footnote.
- Chuyển công thức toán học sang LaTeX.
- Chuyển bảng sang HTML/JSON có cấu trúc.
- Hỗ trợ OCR đa ngôn ngữ.
- Có thể chạy trên CPU, CUDA, Apple Silicon hoặc môi trường không GPU.
- Cung cấp CLI, REST API, Gradio UI, và router.

---

## Kiến trúc xử lý

### 1. Input layer

Dự án chấp nhận các đầu vào phổ biến:

- PDF
- Hình ảnh
- PPTX
- XLSX

### 2. Tải và chuẩn hóa trang

Đối với PDF, hệ thống mở tài liệu bằng PDFium, xác định số trang, và chuyển từng trang sang hình ảnh để xử lý. Nếu tài liệu là scan hoặc có nhiều trang, hệ thống có thể xử lý từng trang hoặc theo từng window để giảm áp lực bộ nhớ.

### 3. Phát hiện loại tài liệu

Hệ thống kiểm tra liệu tài liệu có cần OCR hay không bằng cách phân loại PDF/image:

- `auto`: tự động phát hiện nếu cần OCR
- `ocr`: buộc OCR
- `layout-only`: không OCR, chỉ xử lý bố cục nếu có thể

Điều này rất quan trọng với các tài liệu scan hoặc ảnh có chữ mờ, chữ bị xoay, hoặc bản quét chất lượng thấp.

### 4. Layout detection

Mỗi trang được phân tích theo vùng nội dung:

- text blocks
- title
- paragraph
- list
- table
- formula
- figure
- footnote
- header/footer

Các block này được gắn metadata như vị trí, page index, và mức độ tin cậy. Sau đó, hệ thống sắp xếp lại theo thứ tự đọc hợp lý.

### 5. OCR và trích xuất văn bản

Sau khi xác định vùng cần OCR, hệ thống dùng OCR model để lấy text, bounding box và confidence score. Với bảng, text được OCR theo vùng ô. Với công thức, hệ thống ưu tiên tách ra khỏi text thông thường để nhận diện đúng dạng LaTeX.

### 6. Bảng và công thức

VSF không chỉ OCR text đơn thuần. Với bảng, nó cố gắng:

- xác định vùng bảng
- phân tích hàng/cột
- nhận diện ô dữ liệu
- chuẩn hóa thành HTML hoặc JSON

Với công thức toán học:

- phát hiện vùng công thức
- OCR hoặc trích xuất thành dạng LaTeX
- tách khỏi văn bản bình thường để tránh lẫn lộn

### 7. Tổng hợp output

Sau khi phân tích xong, hệ thống sinh các output chính:

- Markdown cuối cùng
- JSON trung gian theo trang/block
- JSON danh sách nội dung theo thứ tự đọc
- JSON mô tả layout / OCR raw
- PDF minh họa bbox
- ảnh trích xuất và các vùng nội dung

### 8. IDP optional

Nếu bật IDP, hệ thống sẽ tiếp tục làm các bước sau:

- phân loại loại tài liệu
- trích xuất trường dữ liệu quan trọng
- chuẩn hóa giá trị
- kiểm tra độ tin cậy
- đánh dấu cần review nếu cần

Tuy nhiên, đây là lớp nghiệp vụ bổ sung; lõi của dự án vẫn là OCR và chuẩn hóa tài liệu.

---

## Xử lý tài liệu lớn như thế nào?

Với tài liệu nhiều trang, hệ thống không xử lý toàn bộ doc bằng một lần tải hết vào bộ nhớ. Dự án triển khai mô hình "window-based processing" để hỗ trợ các PDF lớn.

### Cách hoạt động

- Mở PDF bằng PDFium
- Lấy total page count
- Chia thành các batch theo window size
- Với mỗi batch, lấy một nhóm trang, render ra ảnh, và xử lý song song/tuần tự theo cấu hình
- Sau khi batch xong, gom kết quả vào `middle_json`
- Tiếp tục xử lý batch tiếp theo cho đến hết tài liệu

Trong code, đây là cách xử lý thực tế:

- `doc_analyze_streaming(...)` đọc file đầu vào
- `window_size = get_processing_window_size(default=64)`
- `while processed_pages < total_pages:`
- `batch_images` và `batch_slices` được xây dựng theo từng batch
- `append_batch_results_to_middle_json(...)` dùng để ghép kết quả từng phần

### Lợi ích

- giảm RAM/VRAM tiêu thụ
- tránh overflow khi PDF có hàng trăm trang
- xử lý an toàn hơn trên GPU 4GB/8GB
- tăng khả năng xử lý hàng loạt và API service

### Biến môi trường quan trọng

```bash
export VSF_PROCESSING_WINDOW_SIZE=32
export VSF_API_MAX_CONCURRENT_REQUESTS=1
export VSF_DEVICE_MODE=cuda
```

- `VSF_PROCESSING_WINDOW_SIZE`: số trang xử lý mỗi window
- `VSF_API_MAX_CONCURRENT_REQUESTS`: số request đồng thời tối đa
- `VSF_DEVICE_MODE`: chọn `cuda`, `cpu`, `mps`, ...

---

## Các backend hỗ trợ

| Backend | Mô tả | Mức độ phù hợp |
|---|---|---|
| `pipeline` | Pipeline OCR + layout + bảng + công thức | Dành cho GPU 4GB/8GB hoặc CPU |
| `vlm-engine` | Dùng VLM hiểu cả trang | GPU mạnh, VRAM lớn |
| `hybrid-engine` | Kết hợp pipeline + VLM | GPU mạnh, cần độ chính xác cao |
| `vlm-http-client` | Gửi inference tới server OpenAI-compatible | Client nhẹ |
| `hybrid-http-client` | Pipeline local + VLM remote | Cân bằng giữa local và remote |

Khuyến nghị:

- Với GPU 4GB, ưu tiên `pipeline`
- Với GPU lớn và cần độ chính xác cao, có thể dùng `hybrid-engine`
- Với máy yếu, dùng client HTTP hoặc backend nhẹ hơn

---

## Yêu cầu hệ thống

- Python: 3.10 - 3.13
- Linux / Windows / macOS
- RAM khuyến nghị: 8GB+
- Dung lượng đĩa: đủ cho model và cache
- Nếu dùng CUDA: cần driver NVIDIA hợp lệ

Kiểm tra GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Cài đặt

### 1. Clone repository

```bash
git clone <repo_url>
cd vsf_ocr
```

### 2. Tạo môi trường ảo

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Trên Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Cài đặt gói

```bash
pip install -e ".[pipeline,gradio]"
```

Hoặc cài đầy đủ:

```bash
pip install -e ".[all]"
```

### 4. Tải model

```bash
vsf-models-download \
  --source huggingface \
  --model_type pipeline
```

Nếu cần VLM:

```bash
vsf-models-download \
  --source huggingface \
  --model_type vlm
```

---

## Chạy WebUI

```bash
VSF_MODEL_SOURCE=local \
VSF_DEVICE_MODE=cuda \
VSF_API_MAX_CONCURRENT_REQUESTS=1 \
vsf-gradio \
  --server-name 127.0.0.1 \
  --server-port 7860
```

Mở:

```text
http://127.0.0.1:7860
```

Trong giao diện:

1. Tải tài liệu lên
2. Chọn backend phù hợp
3. Chọn `parse_method=auto`
4. Bật/tắt table/formula extraction nếu cần
5. Bắt đầu xử lý và tải output

---

## Sử dụng CLI

### Xử lý một file

```bash
vsf \
  -p input/document.pdf \
  -o output \
  -b pipeline
```

### Xử lý một thư mục

```bash
vsf \
  -p input \
  -o output \
  -b pipeline
```

### Chỉ xử lý một số trang đầu

```bash
vsf \
  -p input/document.pdf \
  -o output \
  -b pipeline \
  --start 0 \
  --end 5
```

### Buộc dùng CUDA

```bash
VSF_MODEL_SOURCE=local \
VSF_DEVICE_MODE=cuda \
vsf \
  -p input/document.pdf \
  -o output \
  -b pipeline
```

---

## API REST

Khởi động service:

```bash
VSF_MODEL_SOURCE=local \
VSF_DEVICE_MODE=cuda \
VSF_API_MAX_CONCURRENT_REQUESTS=1 \
vsf-api \
  --host 127.0.0.1 \
  --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Các endpoint chính:

- `GET /health`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/result`
- `POST /file_parse`

Ví dụ gửi tác vụ:

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -F "files=@input/document.pdf" \
  -F "backend=pipeline" \
  -F "enable_idp=true" \
  -F "idp_document_type=auto" \
  -F "return_md=true"
```

---

## Cấu trúc output

Khi xử lý xong, thường sẽ có cấu trúc như sau:

```text
output/
└── document/
    └── auto/
        ├── document.md
        ├── document_middle.json
        ├── document_content_list.json
        ├── document_content_list_v2.json
        ├── document_model.json
        ├── document_idp.json
        ├── document_layout.pdf
        ├── document_span.pdf
        └── images/
```

### Ý nghĩa các file:

- `document.md`: nội dung Markdown cuối cùng
- `document_middle.json`: dữ liệu trung gian theo trang và block
- `document_content_list.json`: danh sách nội dung theo thứ tự đọc
- `document_content_list_v2.json`: định dạng nội dung mới hơn
- `document_model.json`: output gốc từ model
- `document_idp.json`: kết quả phân loại/trích xuất dữ liệu
- `document_layout.pdf`: minh họa bbox layout
- `document_span.pdf`: minh họa bbox text span
- `images/`: hình ảnh và vùng trích xuất

---

## Dòng xử lý OCR thực tế trong dự án

Mô hình OCR của hệ thống được triển khai theo hướng pipeline document understanding, gồm các bước sau:

```text
input document
    ↓
PDF/image loading
    ↓
OCR enable detection
    ↓
page rendering
    ↓
layout detection
    ↓
text / table / formula region extraction
    ↓
OCR on region
    ↓
block ordering and cleanup
    ↓
Markdown & structured JSON generation
```

Đặc điểm quan trọng:

- OCR không chạy trên toàn bộ trang một cách bừa bãi; nó chạy trên từng vùng cần nhận dạng.
- Layout detection giúp phân tách các vùng độc lập hơn, tránh nhầm text với header, footer, figure, table.
- Table và formula có xử lý riêng, vì chúng cần logic khác với đoạn văn thơng thường.
- Output cuối cùng là dữ liệu đã được sắp xếp theo nghĩa đọc, không phải là raw OCR dump thô.

---

## IDP (tùy chọn)

Sau khi pipeline OCR + cấu trúc hóa xong, hệ thống có thể tiếp tục phân tích dữ liệu theo schema công việc.

Ví dụ:

- hợp đồng lao động
- thẻ căn cước
- CV
- quyết định nhân sự
- bảng lương
- bảng chấm công
- tài liệu khác

IDP ở đây tập trung vào:

- phân loại tài liệu
- trích xuất trường dữ liệu
- chuẩn hóa định dạng
- xác định điểm cần review
- đánh giá độ tin cậy

Lưu ý: IDP là lớp bổ trợ; phần cốt lõi của dự án vẫn là OCR và biến tài liệu thành cấu trúc.

---

## Cấu trúc mã nguồn

```text
vsf/
├── backend/
│   ├── pipeline/     # OCR, layout, table, formula
│   ├── vlm/          # Vision-language model
│   ├── hybrid/       # Kết hợp pipeline + VLM
│   └── office/       # PPTX / XLSX processing
├── cli/              # CLI, API, Gradio, router
├── data/             # Reader/writer cho dữ liệu
├── idp/              # Classification + field extraction + validation
├── model/            # models and adapter
├── resources/        # UI resources, language data
├── utils/            # shared helpers
└── __init__.py
```

---

## Kiểm thử

```bash
pytest
```

Test end-to-end:

```bash
pytest tests/unittest/test_e2e.py
```

Test IDP riêng:

```bash
pytest tests/unittest/idp/test_hr_idp.py
```

---

## Lưu ý vận hành

- Với GPU 4GB, nên ưu tiên backend `pipeline`.
- Với PDF lớn, hãy dùng `VSF_PROCESSING_WINDOW_SIZE` hợp lý.
- Nếu tài liệu scan và chữ mờ, nên dùng OCR mode hoặc `auto`.
- Nếu gặp OOM, giảm số request đồng thời và xử lý theo từng batch.
- Lần đầu chạy thường chậm vì cần tải model và setup môi trường.

---

## Người duy trì

**Tran Van Huynh**

Dự án này được phát triển cho mục đích nghiên cứu, học tập và xây dựng hệ thống xử lý tài liệu văn bản có cấu trúc.

---

## Tóm tắt ngắn

VSF OCR là một hệ thống OCR + document understanding tập trung vào việc đọc, hiểu và chuẩn hóa tài liệu. Nó không chỉ chụp chữ, mà còn phát hiện bố cục, tách bảng, phân tách công thức, sắp xếp nội dung theo thứ tự đọc, và sinh ra Markdown/JSON để phục vụ các workflow xử lý tài liệu thực tế.
