from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_id = "google/medgemma-1.5-4b-it"

print("Chargement du processeur...")
processor = AutoProcessor.from_pretrained(model_id)

print("Chargement du modèle...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Modèle chargé sur : {model.device}")
print("OK")