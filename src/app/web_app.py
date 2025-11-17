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
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .status-box {
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    /* Chat history - scrollable */
    #chat-history {
        overflow-y: auto;
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 8px;
        min-height: 400px;
        max-height: calc(100vh - 250px);
    }
    
    /* Input container - FIXED at bottom, only in chat area */
    #input-container {
        position: fixed;
        bottom: 0;
        right: 20px;
        padding: 15px 20px;
        background-color: white;
        border-top: 1px solid #ddd;
        z-index: 100;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        width: calc(100% - 340px);
        max-width: 1100px;
        box-sizing: border-box;
    }
    """
    
    with gr.Blocks(title="Video RAG Chat", css=custom_css, theme=gr.themes.Soft()) as interface:
        
        # Main layout: Left sidebar + Right chat
        with gr.Row(equal_height=False):
            # ===== LEFT SIDEBAR =====
            with gr.Column(scale=1, min_width=300):
                # Header
                gr.HTML("""
                <div class="header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 20px; text-align: center; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">🎬 Video RAG</h2>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">Chat with Video</p>
                </div>
                """)
                
                # Video Loading Section
                gr.Markdown("#### 📁 Load Video")
                
                # File picker
                video_file_picker = gr.File(
                    label="📂 Select File",
                    file_count="single",
                    file_types=["video"]
                )
                
                # Or use text input
                video_path_input = gr.Textbox(
                    label="📝 Or Enter Path",
                    placeholder="/path/to/video.mp4",
                    interactive=True
                )
                
                load_btn = gr.Button("🔄 Load Video", variant="primary", scale=1)
                
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                    elem_classes="status-box",
                    value="Ready to load..."
                )
            
            # ===== RIGHT CHAT AREA =====
            with gr.Column(scale=3):
                # Chat history - using Markdown for proper formatting
                chat_history = gr.Markdown(
                    value="🤖 **Assistant ready!** Load a video and ask your questions.\n\n---\n",
                    label="💬 Chat History",
                    elem_id="chat-history"
                )
            
            # ===== INPUT BOX - OUTSIDE MAIN ROW, FIXED AT BOTTOM =====
        
        # Input section - completely separate, fixed at bottom
        with gr.Group(elem_id="input-container"):
            with gr.Row():
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the video... (Enter to send, Shift+Enter for new line)",
                    interactive=True,
                    lines=2,
                    scale=4
                )
                
                # Send button
                send_btn = gr.Button("📤 Send", variant="primary", scale=1, visible=False)
        
        # ===== INTERACTIONS =====
        
        def load_video_handler(video_file, video_path):
            """Handle video loading from file picker or text path"""
            # Priority: file picker > text input
            final_path = None
            
            if video_file is not None:
                # File was selected from picker
                final_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
            elif video_path and video_path.strip():
                # Text path was entered
                final_path = video_path.strip()
            
            if not final_path:
                return "❌ Vui lòng chọn file hoặc nhập đường dẫn video"
            
            status = rag_interface.load_video(final_path)
            return status
        
        def answer_question_handler(question, current_history):
            """Handle question answering with streaming"""
            if rag_interface.video_rag is None:
                return current_history + "\n❌ **Error:** Video not loaded. Please load a video first.\n\n---\n", ""
            
            if not question or not question.strip():
                return current_history + "\n❌ **Error:** Please enter a question.\n\n---\n", ""
            
            # Add user question to history in markdown format
            new_history = current_history + f"\n**👤 You:**\n> {question}\n\n"
            
            # Start assistant response
            new_history += "**🤖 Assistant:**\n\n"
            
            # Stream response
            for streamed_response in rag_interface.answer_question(question):
                new_history = current_history + f"\n**👤 You:**\n> {question}\n\n" + "**🤖 Assistant:**\n\n" + streamed_response
                yield new_history, ""
            
            # Add separator
            new_history += "\n\n---\n"
            yield new_history, ""
        
        # Connect load button
        load_btn.click(
            fn=load_video_handler,
            inputs=[video_file_picker, video_path_input],
            outputs=[load_status]
        )
        
        # Allow Enter key to submit question
        question_input.submit(
            fn=answer_question_handler,
            inputs=[question_input, chat_history],
            outputs=[chat_history, question_input]
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
