from moviepy import VideoFileClip
import librosa
import soundfile as sf
from transformers import pipeline

def chunking_audio(audio_path: str, chunk_sec=30) -> list:
    chunks = []
    
    video = VideoFileClip(audio_path)
    
    audio = video.audio
    audio.write_audiofile("temp.wav", fps=16000, nbytes=2, codec="pcm_s16le")
    
    audio, sr = librosa.load("temp.wav", sr=16000)
    
    chunk_samples = chunk_sec * sr
    
    for i in range(0, len(audio), chunk_samples):
        chunks.append(audio[i:i+chunk_samples])
    
    return chunks

def transcribe(audio_path: str) -> list:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
    )
    
    chunks = chunking_audio(audio_path)
    
    ans = []
    
    for chunk in chunks:
        ans.append(pipe(chunk)["text"])
    
    return ans
    