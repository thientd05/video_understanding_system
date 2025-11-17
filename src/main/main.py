import argparse
import transformers
from llama_cpp import Llama
from src.utils.video_processing import video_processing
from src.utils.asr import transcribe
from src.utils.choose_frame import choose_frame
from src.utils.ocr import ocr_frames
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

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
    
    args = parser.parse_args()

    frames = video_processing(args.path)

    while len(frames) < 5:
        frames.append(np.zeros(frames[0].shape, dtype=np.uint8))


    transcriptions = transcribe(args.path)

    texts = ocr_frames(frames=frames)


    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    transcriptions_embed = embed_model.encode(transcriptions, convert_to_numpy=True)
    transcriptions_embed = transcriptions_embed.astype("float32")

    texts_embed = embed_model.encode(texts, convert_to_numpy=True)
    texts_embed = texts_embed.astype("float32")

    dimension = texts_embed.shape[1]

    transcriptions_database = faiss.IndexFlatL2(dimension)
    transcriptions_database.add(transcriptions_embed)

    texts_database = faiss.IndexFlatL2(dimension)
    texts_database.add(texts_embed)

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


    question = args.question

    question = "Question: " + question


    llm = Llama(
        model_path="gemma-3-4b-it-qat-Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1
    )
    
    output = llm.create_chat_completion(
        messages = [
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
    info = json.loads(clean)

    asr_prompt = ""
    ocr_prompt = ""
    chosen_frame = frames[::len(frames) // 5]
    if info["ASR"] is not None:
        embed = embed_model.encode(info["ASR"], convert_to_numpy=True).astype("float32")
        embed = embed.reshape(1, -1)
        distances, indices = transcriptions_database.search(embed, 3)
        for i in indices[0]:
            asr_prompt += transcriptions[i] + "\n"
    if info["OCR"] is not None:
        for text in info["OCR"]:
            embed = embed_model.encode(text, convert_to_numpy=True).astype("float32")
            embed = embed.reshape(1, -1)
            distances, indices = texts_database.search(embed, 2)
            for i in indices[0]:
                ocr_prompt += texts[i] + "\n"
    
    if info["DET"] is not None:
        chosen_frame = choose_frame(frames=frames, objects=info["DET"])
        chosen_frame = chosen_frame[::len(chosen_frame)//5]
        chosen_frame = chosen_frame[:5]
    
    for i, frame in enumerate(chosen_frame):
        image = Image.fromarray(frame)
        image.save(f"{i}.jpg")

    system_prompt = "You are an helpful assistant, always follow my instructions. The users are attempting to ask you some questions relevant to the video. The information about the question is retrieved as follows:\n"

    if len(asr_prompt) > 0:
        system_prompt += "Here are some speeches in the video that may include the information you need to answer the question: " + asr_prompt + "\n"
    if len(ocr_prompt) > 0:
        system_prompt += "Here are some texts that are included in the video that are retrieved base on the question: " + ocr_prompt + "\n"
    
    system_prompt += "You got some images in the video that will help you get more information. Read all the information carefully and think step by step, and then anwser the question."

    output = llm.create_chat_completion(
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url",
                     "image_url": {
                         "url" : "0.jpg"
                        }
                    },
                    {"type": "image_url",
                     "image_url": {
                         "url" : "1.jpg"
                        }
                    },
                    {"type": "image_url",
                     "image_url": {
                         "url" : "2.jpg"
                        }
                    },
                    {"type": "image_url",
                     "image_url": {
                         "url" : "3.jpg"
                        }
                    },
                    {"type": "image_url",
                     "image_url": {
                         "url" : "4.jpg"
                        }
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
    )

    print(output["choices"][0]["message"]["content"]) 



    
if __name__ == "__main__":
    main()