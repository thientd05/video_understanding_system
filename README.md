# Video RAG Chat Interface

Chat with videos using Retrieval Augmented Generation (RAG) and an LLM.

### Demo link: https://youtu.be/ZiRbgEmONZU

## 🚀 How to Run

### Web Interface (Gradio) — **Recommended on Ubuntu**

```bash
cd ~your_dir # when cloned my repo
./run_web_app.sh
```

Then open your browser and visit: **http://localhost:7860**


## 📱 Web Interface Features

### Video Loading
- 📁 Upload your video file
- 🔄 Load & Initialize - load the video and initialize the embedding model
- ✅ Status indicator - display the load status

### Chat Interface
- 💬 Chat History - display the entire conversation
- ❓ Question Input - enter and send questions about the video
- 🤖 Real-time Streaming - watch each generated token in real time

### Features
- ⚡ **Real-time Streaming**: Watch the LLM emit tokens as it responds
- 🎬 **Multi-modal**: Combine ASR (audio), OCR (text), and visuals (frames)
- 🔍 **RAG-based**: Retrieve relevant context before generating the answer
- 🎨 **Beautiful UI**: Modern interface built with Gradio

## 🏗️ Architecture

### Components

```
src/
├── main/
│   ├── embedding.py      # Embedding processing
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
### Sample Video
```
├── src/
└── Is_the_future_of_AI_physical_Ian_Bremmer_Explains.mp4      # Sample video
```

### Workflow

1. **Video Loading** → `EmbeddingManager`
   - Process video frames
   - Run ASR (transcribe audio)
   - Run OCR (extract text from frames)
   - Embed all text with `SentenceTransformer`
   - Store everything inside FAISS databases

2. **Question Answering** → `VideoRAG`
   - Retrieve context — determine what information to fetch
   - Search — find relevant transcriptions/OCR texts
   - Answer — generate the reply with the LLM
   - Streaming — print each token in real time

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (optional but recommended)
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
Check the requirements.txt!

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

- **First load**: Takes roughly 10–20 seconds depending on video length
- **GPU Memory**: If you hit GPU memory limits, reduce `n_gpu_layers` in `VideoRAG`
- **Large Videos**: Split very large videos into smaller chunks
- **Accuracy**: Prompt engineering has a big impact on answer quality

## 🐛 Troubleshooting

### GPU Out of Memory
- Reduce `n_gpu_layers` from 32 to around under 30
- Lower the context length `n_ctx` from 4096 to 2048

### Video failed to load
- Verify the video format: MP4, AVI, MOV, MKV
- Try converting the file: `ffmpeg -i input.video -c:v libx264 output.mp4`

## 📊 Performance

| Task | Time | GPU Memory |
|------|------|-----------|
| Load Video (47 frames) | 10-12 sec | ~4GB |
| Answer | 12-14 sec | ~4GB |

## 📚 References

- [Gradio Docs](https://www.gradio.app/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
