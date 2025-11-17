# Video RAG Chat Interface

Ứng dụng chat với video sử dụng Retrieval Augmented Generation (RAG) và LLM.

## 🚀 Cách chạy

### Web Interface (Gradio) - **Khuyến nghị dùng Ubuntu**

```bash
cd /home/thienta/HUST_20235839/AI/video/memory
./run_web_app.sh
```

Sau đó mở browser và truy cập: **http://localhost:7860**


## 📱 Web Interface Features

### Video Loading
- 📁 Upload your video file
- 🔄 Load & Initialize - tải video và khởi tạo embedding model
- ✅ Status indicator - hiển thị trạng thái load

### Chat Interface
- 💬 Chat History - hiển thị toàn bộ lịch sử chat
- ❓ Question Input - nhập và gửi câu hỏi về video
- 🤖 Real-time Streaming - xem từng token được sinh ra

### Features
- ⚡ **Real-time Streaming**: Xem LLM sinh từng token khi đang trả lời
- 🎬 **Multi-modal**: Kết hợp ASR (audio), OCR (text), và visual (images)
- 🔍 **RAG-based**: Tìm kiếm context liên quan trước khi sinh câu trả lời
- 🎨 **Beautiful UI**: Giao diện hiện đại với Gradio

## 🏗️ Architecture

### Components

```
src/
├── main/
│   ├── embedding.py      # Xử lý embedding
│   └── video_rag.py      # VideoRAG
├── app/
│   ├── styles.css        # Css for UI
│   └── web_app.py        # Gradio Web Interface
└── utils/
    ├── asr.py            # Speech-to-text
    ├── ocr.py            # Optical Character Recognition
    ├── video_processing.py
    └── choose_frame.py
```

### Workflow

1. **Video Loading** → `EmbeddingManager`
   - Xử lý video frames
   - Thực hiện ASR (transcribe audio)
   - Thực hiện OCR (extract text from frames)
   - Embedding all text dùng `SentenceTransformer`
   - Lưu vào FAISS databases

2. **Question Answering** → `VideoRAG`
   - Retrieve context - xác định thông tin cần lấy
   - Search - tìm kiếm relevant transcriptions/OCR texts
   - Answer - sinh câu trả lời dùng LLM
   - Streaming - in từng token real-time

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (optional, nhưng khuyến nghị)
- Video files: MP4, AVI, MOV, MKV

### Installed Packages

- `llama-cpp-python` - LLM inference
- `sentence-transformers` - Embedding model
- `faiss-cpu` - Vector search
- `gradio` - Web UI
- `PyQt6` - Desktop UI (optional)
- `opencv-python` - Video processing
- `openai-whisper` - Speech recognition
- `easyocr` - Text recognition

## 📝 Usage Examples

### Web Interface

1. **Load Video**
   ```
   Video Path: /path/to/your/video.mp4
   Click: Load & Initialize
   Wait for: ✅ Video loaded successfully!
   ```

2. **Ask Questions**
   ```
   Question: What did the speaker say about AI?
   Click: Send Question
   Watch: Answer streams in real-time
   ```

## 🎯 Tips

- **First load**: Sẽ mất 10-20s tùy video length
- **GPU Memory**: Nếu hết GPU memory, hãy giảm `n_gpu_layers` trong `VideoRAG`
- **Large Videos**: Chia video thành các phần nhỏ hơn
- **Accuracy**: Prompt engineering ảnh hưởng đến chất lượng câu trả lời

## 🐛 Troubleshooting

### GPU Out of Memory
- Giảm `n_gpu_layers` từ 12 xuống 8-10
- Giảm context length `n_ctx` từ 4096 xuống 2048

### Video load không thành công
- Kiểm tra format video: MP4, AVI, MOV, MKV
- Thử convert video: `ffmpeg -i input.video -c:v libx264 output.mp4`

## 📊 Performance

| Task | Time | GPU Memory |
|------|------|-----------|
| Load Video (8 frames) | 8-10 sec | ~4GB |
| First Answer | 6-9 sec | ~5-6GB |
| Subsequent Answers | 3-6 sec | ~5-6GB |

## 📚 References

- [Gradio Docs](https://www.gradio.app/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
