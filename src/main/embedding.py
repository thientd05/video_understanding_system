import json
import os

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
        
        self.transcriptions_database = None
        self.texts_database = None
        
        self._create_embeddings()
        self.save_vector_databases()

    def save_vector_databases(self, output_dir: str | None = None) -> dict:
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]

        trans_index_path = os.path.join(output_dir, f"{base_name}_transcriptions.index")
        texts_index_path = os.path.join(output_dir, f"{base_name}_texts.index")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
        frames_path = os.path.join(output_dir, f"{base_name}_frames.npz")

        faiss.write_index(self.transcriptions_database, trans_index_path)
        faiss.write_index(self.texts_database, texts_index_path)

        # Save all frames (image data) so that VideoRAG can simply load them
        # without calling video_processing again.
        # frames: list[np.ndarray] with the same shape -> stack into (N, H, W, C)
        frames_array = np.stack(self.frames, axis=0)
        np.savez_compressed(frames_path, frames=frames_array)

        meta = {
            "video_path": self.video_path,
            "transcriptions": self.transcriptions,
            "texts": self.texts,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        return {
            "transcriptions_index": trans_index_path,
            "texts_index": texts_index_path,
            "meta": meta_path,
            "frames": frames_path,
        }
    
    def _create_embeddings(self):
        self.transcriptions_embed = self.embed_model.encode(
            self.transcriptions, 
            convert_to_numpy=True
        ).astype("float32")
        
        self.texts_embed = self.embed_model.encode(
            self.texts, 
            convert_to_numpy=True
        ).astype("float32")
        
        self.transcriptions_database = faiss.IndexFlatL2(self.transcriptions_embed.shape[1])
        self.transcriptions_database.add(self.transcriptions_embed)
        
        self.texts_database = faiss.IndexFlatL2(self.texts_embed.shape[1])
        self.texts_database.add(self.texts_embed)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 