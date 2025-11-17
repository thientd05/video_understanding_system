import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from src.utils.video_processing import video_processing
from src.utils.asr import transcribe
from src.utils.ocr import ocr_frames


class EmbeddingManager:
    """
    Class để quản lý embedding cho audio, text OCR và lưu vào các cơ sở dữ liệu tương ứng
    """
    
    def __init__(self, video_path):
        """
        Khởi tạo EmbeddingManager
        
        Args:
            video_path (str): Đường dẫn đến file video
        """
        self.video_path = video_path
        # Load embedding model trên CPU để tiết kiệm GPU memory cho LLM
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        
        # Xử lý video để lấy frames
        self.frames = video_processing(video_path)
        while len(self.frames) < 5:
            self.frames.append(np.zeros(self.frames[0].shape, dtype=np.uint8))
        
        # Lấy transcriptions từ audio
        self.transcriptions = transcribe(video_path)
        
        # Lấy texts từ OCR
        self.texts = ocr_frames(frames=self.frames)
        
        # Tạo embeddings
        self.transcriptions_embed = None
        self.texts_embed = None
        self.dimension = None
        
        # Tạo databases
        self.transcriptions_database = None
        self.texts_database = None
        
        self._create_embeddings()
    
    def _create_embeddings(self):
        """
        Tạo embeddings cho transcriptions và texts, sau đó thêm vào cơ sở dữ liệu
        """
        # Embedding transcriptions
        self.transcriptions_embed = self.embed_model.encode(
            self.transcriptions, 
            convert_to_numpy=True
        )
        self.transcriptions_embed = self.transcriptions_embed.astype("float32")
        
        # Embedding texts
        self.texts_embed = self.embed_model.encode(
            self.texts, 
            convert_to_numpy=True
        )
        self.texts_embed = self.texts_embed.astype("float32")
        
        # Lấy dimension từ embeddings
        self.dimension = self.texts_embed.shape[1]
        
        # Tạo FAISS databases
        self.transcriptions_database = faiss.IndexFlatL2(self.dimension)
        self.transcriptions_database.add(self.transcriptions_embed)
        
        self.texts_database = faiss.IndexFlatL2(self.dimension)
        self.texts_database.add(self.texts_embed)
        
        # Giải phóng GPU memory sau khi embedding xong
        # Di chuyển model sang CPU (nếu chưa) để giải phóng GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_frames(self):
        """Lấy danh sách frames từ video"""
        return self.frames
    
    def get_transcriptions(self):
        """Lấy danh sách transcriptions"""
        return self.transcriptions
    
    def get_texts(self):
        """Lấy danh sách texts từ OCR"""
        return self.texts
    
    def get_transcriptions_database(self):
        """Lấy FAISS database cho transcriptions"""
        return self.transcriptions_database
    
    def get_texts_database(self):
        """Lấy FAISS database cho texts"""
        return self.texts_database
    
    def get_embed_model(self):
        """Lấy embedding model"""
        return self.embed_model
