#!/usr/bin/env python3
"""
Video RAG Web Interface using Gradio
Giao diện web đẹp cho Video RAG Chat
"""

import gradio as gr
import os
import sys
from pathlib import Path

from src.main.embedding import EmbeddingManager
from src.main.video_rag import VideoRAG


class VideoRAGInterface:
    """Wrapper class cho VideoRAG interface"""
    
    def __init__(self):
        self.video_rag = None
        self.embedding_manager = None
    
    def load_video(self, video_path: str) -> str:
        """
        Load video và khởi tạo embedding
        
        Args:
            video_path: Đường dẫn tới file video
            
        Returns:
            Status message
        """
        if not video_path or not video_path.strip():
            return "❌ Vui lòng nhập đường dẫn video"
        
        video_path = video_path.strip()
        
        if not os.path.exists(video_path):
            return f"❌ File không tồn tại: {video_path}"
        
        try:
            print(f"[INFO] Loading video: {video_path}")
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(video_path)
            print("[INFO] EmbeddingManager initialized")
            
            # Initialize video RAG
            self.video_rag = VideoRAG(self.embedding_manager)
            print("[INFO] VideoRAG initialized")
            
            return f"✅ Video loaded successfully!\nFrames: {len(self.embedding_manager.get_frames())}\nTranscriptions: {len(self.embedding_manager.get_transcriptions())}\nOCR texts: {len(self.embedding_manager.get_texts())}"
            
        except Exception as e:
            print(f"[ERROR] Failed to load video: {str(e)}")
            return f"❌ Lỗi khi load video: {str(e)}"
    
    def answer_question(self, question: str) -> str:
        """
        Trả lời câu hỏi với streaming output
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Câu trả lời từ LLM
        """
        if self.video_rag is None:
            return "❌ Vui lòng load video trước!"
        
        if not question or not question.strip():
            return "❌ Vui lòng nhập câu hỏi"
        
        question = question.strip()
        
        try:
            print(f"\n[INFO] Processing question: {question}")
            
            # Get streaming response
            response_text = ""
            for token in self.video_rag.answer_question(question, streaming=True):
                response_text += token
                yield response_text  # Yield progressively for real-time display
            
            print("[INFO] Answer completed")
            
        except Exception as e:
            print(f"[ERROR] Failed to answer question: {str(e)}")
            yield f"❌ Lỗi khi trả lời: {str(e)}"


def create_interface():
    """Tạo Gradio interface"""
    
    rag_interface = VideoRAGInterface()
    
    # Custom CSS
    custom_css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .section {
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
        border-left: 4px solid #667eea;
    }
    
    .status-box {
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    """
    
    with gr.Blocks(title="Video RAG Chat", css=custom_css, theme=gr.themes.Soft()) as interface:
        
        # Header
        with gr.Group():
            gr.HTML("""
            <div class="header">
                <h1>🎬 Video RAG Chat Interface</h1>
                <p>Hỏi đáp về nội dung video bằng AI</p>
            </div>
            """)
        
        # Video Loading Section
        gr.Markdown("### 📁 Video Loading")
        with gr.Group():
            with gr.Row():
                video_path_input = gr.Textbox(
                    label="Video Path",
                    placeholder="Nhập đường dẫn tới file video (e.g., /path/to/video.mp4)",
                    interactive=True,
                    scale=4
                )
                load_btn = gr.Button("🔄 Load & Initialize", scale=1, variant="primary")
            
            load_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_classes="status-box"
            )
        
        # Chat Section
        gr.Markdown("### 💬 Chat Interface")
        with gr.Group():
            # Chat history
            chat_history = gr.Textbox(
                label="Chat History",
                interactive=False,
                lines=15,
                max_lines=30,
                elem_classes="status-box",
                value="🤖 Assistant ready. Load a video and ask your questions!\n" + "-"*80 + "\n"
            )
            
            # Question input
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Hỏi bất cứ điều gì về video...",
                interactive=True,
                lines=2
            )
            
            # Send button
            send_btn = gr.Button("📤 Send Question", variant="primary", scale=1)
        
        # ===== INTERACTIONS =====
        
        def load_video_handler(video_path):
            """Handle video loading"""
            status = rag_interface.load_video(video_path)
            return status
        
        def answer_question_handler(question, current_history):
            """Handle question answering with streaming"""
            if rag_interface.video_rag is None:
                return current_history + "\n❌ Error: Video not loaded. Please load a video first.\n" + "-"*80 + "\n"
            
            if not question or not question.strip():
                return current_history + "\n❌ Error: Please enter a question.\n" + "-"*80 + "\n"
            
            # Add user question to history
            new_history = current_history + f"\n👤 You:\n{question}\n\n"
            
            # Start assistant response
            new_history += "🤖 Assistant:\n"
            
            # Stream response
            for streamed_response in rag_interface.answer_question(question):
                new_history = current_history + f"\n👤 You:\n{question}\n\n" + "🤖 Assistant:\n" + streamed_response
                yield new_history
            
            # Add separator
            new_history += "\n" + "-"*80 + "\n"
            yield new_history
        
        # Connect load button
        load_btn.click(
            fn=load_video_handler,
            inputs=[video_path_input],
            outputs=[load_status]
        )
        
        # Connect send button
        send_btn.click(
            fn=answer_question_handler,
            inputs=[question_input, chat_history],
            outputs=[chat_history]
        )
        
        # Allow Enter key to submit question
        question_input.submit(
            fn=answer_question_handler,
            inputs=[question_input, chat_history],
            outputs=[chat_history]
        )
        
        # Clear question after sending
        def clear_question():
            return ""
        
        send_btn.click(
            fn=clear_question,
            inputs=[],
            outputs=[question_input]
        )
        
        question_input.submit(
            fn=clear_question,
            inputs=[],
            outputs=[question_input]
        )
    
    return interface


def main():
    """Main function"""
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
