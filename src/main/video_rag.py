import json
import os

import faiss
import numpy as np
import torch
from llama_cpp import Llama
from PIL import Image

from src.utils.choose_frame import choose_frame


class VideoRAG:
    
    def __init__(self, index_paths: dict = None):
        self._init_from_files(index_paths)
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, "gemma-3-4b-it-Q4_K_M.gguf")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=32
        )

    def _init_from_files(self, index_paths: dict):
        meta_path = index_paths["meta"]
        trans_index_path = index_paths["transcriptions_index"]
        texts_index_path = index_paths["texts_index"]
        frames_path = index_paths["frames"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.video_path = meta["video_path"]
        self.transcriptions = meta["transcriptions"]
        self.texts = meta["texts"]

        self.transcriptions_database = faiss.read_index(trans_index_path)
        self.texts_database = faiss.read_index(texts_index_path)

        # Load previously saved frames (without calling video_processing again)
        frames_npz = np.load(frames_path)
        frames_array = frames_npz["frames"]
        # Convert to list[np.ndarray] for compatibility with existing code
        self.frames = [frame for frame in frames_array]

        from sentence_transformers import SentenceTransformer

        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    
    def _rewrite_user_query(self, question):
        system_prompt_retrieve = "You are an helpful assistant, always follow my instructions. To answer the question step by step, you can provide your retrieve request to assist you by the following json format:\n"
        
        system_prompt_retrieve += '''{
    "ASR": Optional[str]. The abstract information that people in the video may discuss, or just the summary of the question, in two sentences. If you don't need this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "OCR": Optional[list]. (The output must be specified as null or a list containing detailed texts in video that may relevant to the answer of the question. (The information that you want to know more about.)
    }
    ## Example 1: 
    Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
    Your retrieve can be:
    {
        "ASR": "The location and the color of balloons, the number of the blue balloons.",
        "DET": ["blue ballons", "long table"],
        "OCR": null
    }
    ## Example 2: 
    Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
    Your retrieve can be:
    {
        "ASR": null,
        "DET": ["the man in black", "woman"],
        "OCR": null
    }
    ## Example 3: 
    Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
    Your retrieve can be:
    {
        "ASR": "The country recognized worldwide for its comedy.",
        "DET": null,
        "OCR": ["China", "UK", "Germany", "USA"]
    }
    Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''
        
        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_retrieve
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        raw = output["choices"][0]["message"]["content"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        rewritten_info = json.loads(clean)
        
        return rewritten_info
    
    def _retrieval_information(self, rewritten_info):
        asr_prompt = ""
        ocr_prompt = ""
        frame_step = max(1, len(self.frames) // 5) if len(self.frames) > 0 else 1
        chosen_frame = self.frames[::frame_step]
        
        if rewritten_info.get("ASR") is not None:
            embed = self.embed_model.encode(
                rewritten_info["ASR"], 
                convert_to_numpy=True
            ).astype("float32")
            embed = embed.reshape(1, -1)
            distances, indices = self.transcriptions_database.search(embed, 3)
            for i in indices[0]:
                asr_prompt += self.transcriptions[i] + "\n"
        
        if rewritten_info.get("OCR") is not None:
            for text in rewritten_info["OCR"]:
                embed = self.embed_model.encode(
                    text, 
                    convert_to_numpy=True
                ).astype("float32")
                embed = embed.reshape(1, -1)
                distances, indices = self.texts_database.search(embed, 2)
                for i in indices[0]:
                    ocr_prompt += self.texts[i] + "\n"
        
        det_objects = rewritten_info.get("DET") or []
        if len(det_objects) > 0:
            chosen_frame = choose_frame(frames=self.frames, objects=det_objects)
            if len(chosen_frame) > 0:
                det_step = max(1, len(chosen_frame) // 5)
                chosen_frame = chosen_frame[::det_step]
            else:
                chosen_frame = self.frames[::frame_step]
            chosen_frame = chosen_frame[:5]
            
        frames_dir = os.path.dirname(os.path.abspath(__file__))
        for idx, frame in enumerate(chosen_frame[:5]):
            img = Image.fromarray(frame)
            img.save(os.path.join(frames_dir, f"{idx}.jpg"), format="JPEG")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return asr_prompt, ocr_prompt, chosen_frame
    
    def answer_question(self, question, streaming=False):
        formatted_question = "Question: " + question
        
        rewritten_info = self._rewrite_user_query(formatted_question)
        
        asr_prompt, ocr_prompt, chosen_frame = self._retrieval_information(rewritten_info)
        
        answer_system_prompt = "You are an helpful assistant, always follow my instructions. The users are attempting to ask you some questions relevant to the video. The information about the question is retrieved as follows:\n"
        
        if len(asr_prompt) > 0:
            answer_system_prompt += "Here are some speeches in the video that may include the information you need to answer the question: " + asr_prompt + "\n"
        if len(ocr_prompt) > 0:
            answer_system_prompt += "Here are some texts that are included in the video that are retrieved base on the question: " + ocr_prompt + "\n"
        
        answer_system_prompt += "You got some images in the video that will help you get more information. Read all the information carefully and think step by step, and then anwser the question."
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": answer_system_prompt},
                    {"type": "image_url", "image_url": {"url": "0.jpg"}},
                    {"type": "image_url", "image_url": {"url": "1.jpg"}},
                    {"type": "image_url", "image_url": {"url": "2.jpg"}},
                    {"type": "image_url", "image_url": {"url": "3.jpg"}},
                    {"type": "image_url", "image_url": {"url": "4.jpg"}},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_question}
                ]
            }
        ]
        
        if streaming:
            response = self.llm.create_chat_completion(
                messages=messages,
                stream=True
            )
            
            def stream_generator():
                for chunk in response:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
            
            return stream_generator()
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                stream=False
            )
            
            return response["choices"][0]["message"]["content"] 