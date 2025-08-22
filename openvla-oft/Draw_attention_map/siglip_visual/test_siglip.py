import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float32, attn_implementation="eager").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

image_path='/home/Better-oft/openvla-oft/Draw_attention_map/My_image/pipeline-cat-chonk.jpeg'
image = Image.open(image_path).convert('RGB')
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")