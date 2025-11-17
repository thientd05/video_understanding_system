import argparse
from src.main.embedding import EmbeddingManager
from src.main.video_rag import VideoRAG

def main():
    parser = argparse.ArgumentParser(description="Video answering tool")
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input video file"
    )

    parser.add_argument(
        "--question",
        type=str,
        required=False,
        help="question you want to ask the chatbot, default will be set to: Summarize the video"
    )
    
    parser.add_argument(
        "--streaming",
        type=bool,
        default=False,
        help="Enable streaming output (default: False)"
    )
    
    args = parser.parse_args()

    # Khởi tạo EmbeddingManager để xử lý embedding
    embedding_manager = EmbeddingManager(args.path)
    
    # Khởi tạo VideoRAG để trả lời câu hỏi
    video_rag = VideoRAG(embedding_manager)
    
    # Trả lời câu hỏi
    answer = video_rag.answer_question(args.question, streaming=args.streaming)
    
    # Nếu streaming, in từng chunk
    if args.streaming:
        for chunk in answer:
            print(chunk, end="", flush=True)
        print()  # Xuống dòng cuối cùng
    else:
        print(answer) 



    
if __name__ == "__main__":
    main()