import easyocr
from src.utils.video_processing import video_processing

def ocr_frames(frames: list) -> list:
    ans =[]
    reader = easyocr.Reader(['en'])
    for frame in frames:
        texts = reader.readtext(frame, detail=0)
        for text in texts:
            ans.append(text)
        
    return ans