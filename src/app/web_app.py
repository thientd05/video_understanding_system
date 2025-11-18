#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np

from src.main.embedding import EmbeddingManager
from src.main.video_rag import VideoRAG


class VideoRAGInterface:
    
    def __init__(self):
        self.video_rag = None
        self.embedding_manager = None
    
    def load_video(self, video_path: str) -> str:
        if not video_path or not video_path.strip():
            return "🔴 Please enter a video path"
        
        video_path = video_path.strip()
        
        if not os.path.exists(video_path):
            return f"🔴 File does not exist: {video_path}"
        
        try:
            project_root = Path(__file__).resolve().parent.parent  # .../src
            main_dir = project_root / "main"  # .../src/main

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            trans_index_path = main_dir / f"{base_name}_transcriptions.index"
            texts_index_path = main_dir / f"{base_name}_texts.index"
            meta_path = main_dir / f"{base_name}_meta.json"
            frames_path = main_dir / f"{base_name}_frames.npz"

            if (
                trans_index_path.exists()
                and texts_index_path.exists()
                and meta_path.exists()
                and frames_path.exists()
            ):
                index_paths = {
                    "transcriptions_index": str(trans_index_path),
                    "texts_index": str(texts_index_path),
                    "meta": str(meta_path),
                    "frames": str(frames_path),
                }

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                frames_npz = np.load(frames_path)
                frames_array = frames_npz["frames"]

                self.embedding_manager = None
                self.video_rag = VideoRAG(index_paths=index_paths)

                return (
                    "🟢 Video loaded successfully\n"
                    f"Frames: {len(frames_array)}\n"
                    f"Transcriptions: {len(meta.get('transcriptions', []))}\n"
                    f"OCR texts: {len(meta.get('texts', []))}"
                )

            # If indexes do not exist yet -> create embeddings and save vector databases + metadata + frames
            self.embedding_manager = EmbeddingManager(video_path)
            index_paths = self.embedding_manager.save_vector_databases()

            # Initialize VideoRAG from the saved files
            self.video_rag = VideoRAG(index_paths=index_paths)
            
            return (
                "🟢 Video loaded successfully!\n"
                f"Frames: {len(self.embedding_manager.frames)}\n"
                f"Transcriptions: {len(self.embedding_manager.transcriptions)}\n"
                f"OCR texts: {len(self.embedding_manager.texts)}"
            )
            
        except Exception as e:
            return f"🔴 Error while loading video: {str(e)}"
    
    def answer_question(self, question: str) -> str: # type: ignore
        if self.video_rag is None:
            return "🔴 Please load a video first!"
        
        if not question or not question.strip():
            return "🔴 Please enter a question"
        
        question = question.strip()
        
        try:
            response_text = ""
            for token in self.video_rag.answer_question(question, streaming=True):
                response_text += token
                yield response_text
            
        except Exception as e:
            yield f"🔴 Error while answering: {str(e)}"


def create_interface():
    
    rag_interface = VideoRAGInterface()
    
    css_file_path = Path(__file__).parent / "styles.css"
    custom_css = ""
    
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
    except FileNotFoundError:
        print(f"[WARNING] Could not find styles.css at {css_file_path}. Using default styles.")
    except Exception as e:
        print(f"[ERROR] Error loading CSS: {e}")
    
    with gr.Blocks(title="Video RAG Chat", css=custom_css, theme=gr.themes.Soft()) as interface:
        
        with gr.Row(equal_height=False):
            
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                <div class="header">
                    <h2 style="margin: 0;">🎬 Video RAG</h2>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">Chat with Video</p>
                </div>
                """)
                
                gr.Markdown("#### 📁 Load Video")
                
                video_file_picker = gr.File(
                    label="📂 Select File",
                    file_count="single",
                    file_types=["video"]
                )
                
                load_btn = gr.Button("Load Video", variant="primary", scale=1)
                
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                    elem_classes="status-box",
                    value="Ready to load..."
                )
            
            with gr.Column(scale=3):
                
                chat_history = gr.Markdown(
                    value="🧠 **Assistant ready!** Load a video and ask your questions.\n\n---\n",
                    label="💬 Chat History",
                    elem_id="chat-history"
                )
            
        
        with gr.Group(elem_id="input-container"):
            with gr.Row():
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the video (Press Enter to send)",
                    interactive=True,
                    lines=2,
                    scale=4,
                    max_lines=5
                )
        
        def load_video_handler(video_file):
            if video_file is None:
                return "🔴 Please select a file"
            
            final_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
            
            status = rag_interface.load_video(final_path)
            return status
        
        def answer_question_handler(question, current_history):
            if rag_interface.video_rag is None:
                return current_history + "\n🔴 **Error:** Video not loaded. Please load a video first.\n\n---\n", ""
            
            if not question or not question.strip():
                return current_history + "\n🔴 **Error:** Please enter a question.\n\n---\n", ""
            
            new_history = current_history + f"\n**👤 You:**\n> {question}\n\n"
            
            new_history += "**🧠 Assistant:**\n\n"
            
            # Store the final response
            final_response = ""
            
            # Stream the response
            for streamed_response in rag_interface.answer_question(question):
                final_response = streamed_response  # Keep updating with latest
                display_history = new_history + streamed_response
                yield display_history, ""
            
            # Add separator after complete response
            final_history = new_history + final_response + "\n\n---\n"
            yield final_history, ""
            
        
        load_btn.click(
            fn=load_video_handler,
            inputs=[video_file_picker],
            outputs=[load_status],
            #show_progress="hidden"
        )
        
        question_input.submit(
            fn=answer_question_handler,
            inputs=[question_input, chat_history],
            outputs=[chat_history, question_input],
            #show_progress="hidden"
        )
    
    return interface


def main():
    print("[INFO] Starting Video RAG Web Interface...")
    print("[INFO] Opening browser at http://localhost:7860")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False
    )


if __name__ == "__main__":
    main()