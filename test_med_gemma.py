"""Test de chargement + inférence MedGemma-4B sur une radio de test.

Objectif : valider que le VLM se charge et produit une sortie texte.

ATTENTION : MedGemma est un modèle 'gated' -> il faut au préalable :
  1. Accepter les conditions sur https://huggingface.co/google/medgemma-4b-it
  2. S'authentifier :  huggingface-cli login   (ou variable d'env HF_TOKEN)
"""

from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "google/medgemma-1.5-4b-it"

# --- Détection du matériel ---------------------------------------------------
has_cuda = torch.cuda.is_available()
print(f"GPU CUDA disponible : {has_cuda}")

# --- Construction des arguments de chargement --------------------------------
# Si GPU dispo -> on tente la quantification 4 bits (passe ~8 Go a ~3 Go).
# Sinon (CPU seul) -> bfloat16 + offload disque pour limiter le pic memoire
#                    (corrige l'OSError 1455 'fichier de pagination insuffisant').
load_kwargs = {
    "low_cpu_mem_usage": True,
    "device_map": "auto",
}

if has_cuda:
    from transformers import BitsAndBytesConfig

    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    load_kwargs["dtype"] = torch.bfloat16       # 'dtype' remplace 'torch_dtype'
    load_kwargs["offload_folder"] = "offload"   # decharge sur disque si RAM insuffisante

# --- Chargement du processeur et du modèle (UNE seule fois) ------------------
print("Chargement du processeur...")
processor = AutoProcessor.from_pretrained(model_id)

print("Chargement du modèle... (peut prendre plusieurs minutes)")
model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
print(f"Modèle chargé sur : {model.device}")

# --- Inférence sur une image de test ----------------------------------------
sample = Path("data/sample_images/CXR_SYN_002_suspected_opacity.png")
image = Image.open(sample).convert("RGB")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a careful radiology assistant. This is an educational prototype, not a medical device."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this frontal chest X-ray. Is there any visible opacity?"},
            {"type": "image", "image": image},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

print("Génération...")
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]   # on ne garde que la reponse generee

answer = processor.decode(generation, skip_special_tokens=True)
print("\n----- Réponse MedGemma -----")
print(answer)
print("OK")
