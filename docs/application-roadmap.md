# Kế hoạch phát triển BLIP Vision Assistant

## Mục tiêu

Phát triển ứng dụng thành trợ lý hình ảnh đa phương thức: người dùng có thể tải ảnh, đặt câu hỏi bằng giọng nói, nghe câu trả lời, nhận cảnh báo khi mô hình không đủ bằng chứng và dùng mô hình đã thích nghi với miền dữ liệu nội thất.

```text
Ảnh ───────────────┬─> BLIP Caption ─┬─> Confidence/Abstention ─> Kết quả ─> TTS
                   └─> BLIP VQA ─────┘
Microphone ─> STT ─────────────────────> Câu hỏi
Dataset ─> Chuẩn hóa ─> Fine-tune/LoRA ─> Model Registry ─> Inference
```

## Giai đoạn 1 — Speech-to-text và text-to-speech (đã triển khai)

### Phạm vi

- Thêm microphone trong tab VQA.
- Chuyển âm thanh thành văn bản tiếng Việt/Anh; cho phép sửa transcript trước khi gửi.
- Chuyển caption và câu trả lời thành âm thanh, có tùy chọn tự động đọc.
- Không lưu audio mặc định sau khi phiên kết thúc.

### Kỹ thuật đề xuất

- STT local: `faster-whisper` với model `base` hoặc `small`.
- TTS: `edge-tts` khi có mạng; Piper nếu cần chạy hoàn toàn offline.
- Đóng gói adapter trong `src/speech.py`; cache model STT thay vì tải lại mỗi lượt.

### Tiêu chí nghiệm thu

- Ghi âm, xem/sửa transcript và gửi câu hỏi trong một luồng.
- Hỗ trợ tối thiểu `vi` và `en`.
- Đo latency cho audio 10 giây và hiển thị trạng thái xử lý rõ ràng.

## Giai đoạn 2 — Confidence và abstention (đã triển khai, chờ duyệt validation)

BLIP sinh văn bản nên xác suất token không phải xác suất câu trả lời đúng. Không hiển thị phần trăm “độ tin cậy” nếu chưa hiệu chỉnh. Hệ thống kết hợp nhiều tín hiệu và chọn ngưỡng trên validation:

- Log-probability trung bình của chuỗi sinh.
- Độ chênh điểm giữa các beam tốt nhất.
- Độ ổn định qua nhiều cấu hình decoding hoặc seed.
- Tương đồng ảnh–câu trả lời bằng CLIP.
- Mức đồng thuận giữa các lần sinh đối với VQA.

Giao diện có ba trạng thái: `Tin cậy`, `Cần kiểm tra`, `Không đủ bằng chứng`. Trạng thái cuối trả lời mềm: “Tôi chưa thể xác định điều này từ ảnh.”

### Đánh giá

- Tạo tập validation có nhãn `answerable/unanswerable`, gồm câu hỏi ngoài nội dung ảnh.
- Đo accuracy–coverage, selective risk, AUROC và Expected Calibration Error.
- Chọn ngưỡng trên validation, không dùng test.
- Mục tiêu đầu: giảm ít nhất 30% lỗi nghiêm trọng trong khi giữ coverage từ 70%.

## Giai đoạn 3 — Fine-tune nhẹ theo miền dữ liệu

### Chuẩn bị

- Kiểm tra thủ công ánh xạ 300 ảnh trong `annotations_300.json`.
- Chia train/validation/test theo ảnh để QA hoặc caption của cùng ảnh không bị rò rỉ.
- Chuẩn hóa nhãn và giữ cố định test set trước thí nghiệm.

### Chiến lược

1. Đo baseline bằng hai model pretrained hiện tại.
2. Fine-tune language decoder hoặc dùng LoRA/PEFT, đóng băng phần lớn vision encoder.
3. Huấn luyện caption và VQA thành hai thí nghiệm riêng.
4. Dùng early stopping và lưu checkpoint tốt nhất theo validation.
5. Chỉ mở thêm layer nếu phương án nhẹ không vượt baseline.

### Chỉ số

- Caption: BLEU-4, ROUGE-L, BERTScore và đánh giá thủ công tính đúng nội dung.
- VQA: Exact Match, token F1 và accuracy theo nhóm câu hỏi.
- Hệ thống: latency, RAM/VRAM, kích thước checkpoint và tỷ lệ abstention.

Tập 300 ảnh khá nhỏ và dễ overfit; ưu tiên LoRA, augmentation hợp lý và cross-validation. Không dùng test để chọn prompt, ngưỡng hay epoch.

## Cấu trúc mã nguồn dự kiến

```text
src/
  speech.py
  confidence.py
  model_registry.py
training/
  prepare_dataset.py
  train_caption_lora.py
  train_vqa_lora.py
  calibrate_confidence.py
tests/
  test_speech.py
  test_confidence.py
  test_abstention.py
```

## Lộ trình

| Mốc | Thời lượng | Bàn giao |
|---|---:|---|
| UI và baseline | 2–3 ngày | UI mới, test và benchmark gốc |
| STT/TTS | 4–6 ngày | Luồng voice, hỗ trợ vi/en |
| Confidence/abstention | 6–8 ngày | Dataset answerability, calibration, cảnh báo UI |
| Fine-tune nhẹ | 8–12 ngày | Checkpoint LoRA, báo cáo so sánh, model registry |
| Kiểm thử và tài liệu | 3–5 ngày | Test hồi quy, hướng dẫn triển khai, demo |

Tổng ước tính: 23–34 ngày công cho một người, phụ thuộc phần cứng và chất lượng nhãn.
