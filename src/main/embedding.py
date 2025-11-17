import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from src.utils.video_processing import video_processing
from src.utils.asr import transcribe
from src.utils.ocr import ocr_frames


class EmbeddingManager:
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        
        self.frames = video_processing(video_path)
        while len(self.frames) < 5:
            self.frames.append(np.zeros(self.frames[0].shape, dtype=np.uint8))
        
        self.transcriptions = transcribe(video_path)
        self.texts = ocr_frames(frames=self.frames)
        
        self.transcriptions_embed = None
        self.texts_embed = None
        self.dimension = None
        
        self.transcriptions_database = None
        self.texts_database = None
        
        self._create_embeddings()
    
    def _create_embeddings(self):
        self.transcriptions_embed = self.embed_model.encode(
            self.transcriptions, 
            convert_to_numpy=True
        )
        self.transcriptions_embed = self.transcriptions_embed.astype("float32")
        
        self.texts_embed = self.embed_model.encode(
            self.texts, 
            convert_to_numpy=True
        )
        self.texts_embed = self.texts_embed.astype("float32")
        
        self.dimension = self.texts_embed.shape[1]
        
        self.transcriptions_database = faiss.IndexFlatL2(self.dimension)
        self.transcriptions_database.add(self.transcriptions_embed)
        
        self.texts_database = faiss.IndexFlatL2(self.dimension)
        self.texts_database.add(self.texts_embed)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_frames(self):
        return self.frames
    
    def get_transcriptions(self):
        return self.transcriptions
    
    def get_texts(self):
        return self.texts
    
    def get_transcriptions_database(self):
        return self.transcriptions_database
    
    def get_texts_database(self):
        return self.texts_database
    
    def get_embed_model(self):
        return self.embed_model