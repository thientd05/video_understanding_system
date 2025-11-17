import torch
import clip
import cv2
from PIL import Image
from src.utils.video_processing import video_processing

print(clip.__file__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def choose_frame(frames: list, objects: list, threshold=0.2, batch_size=32):
    ans = []

    # Encode text 1 lần
    text_tokens = clip.tokenize(objects).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Chia batch cho frames
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        # Preprocess batch → tensor [B, 3, 224, 224]
        batch_tensors = []
        for frame in batch_frames:
            # BGR → RGB nếu cần
            if frame.shape[2] == 3:
                frame = frame[..., ::-1]  # nhanh hơn cv2.cvtColor
            
            pil_image = Image.fromarray(frame)
            tensor = preprocess(pil_image)
            batch_tensors.append(tensor)

        batch_input = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            image_features = model.encode_image(batch_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # similarity shape: [B, num_text]
            similarity = image_features @ text_features.T
            print(similarity)

        # Lấy max mỗi ảnh trong batch
        max_vals = similarity.max(dim=1).values  # shape [B]

        # Filter frame theo threshold
        for f, score in zip(batch_frames, max_vals):
            if score.item() > threshold:
                ans.append(f)

    return ans