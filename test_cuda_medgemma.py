import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image


# --- 1. Vérification CUDA ---
print("=== Vérification CUDA ===")
print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ATTENTION : pas de GPU détecté, le modèle tournera sur CPU (très lent)")

# --- 2. Chargement modèle ---
model_id = "google/medgemma-4b-it"
print(f"\n=== Chargement de {model_id} ===")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Modèle chargé sur : {model.device}")
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM utilisée : {used:.1f} GB")

# --- 3. Image de test (radio thoracique publique NIH) ---
print("\n=== Test d'inférence ===")
print("Création image de test (gris neutre 512x512)...")
image = Image.new("RGB", (512, 512), color=(128, 128, 128))

messages = [
    {"role": "system", "content": [{"type": "text", "text": "Tu es un assistant pédagogique d'analyse de radiographies thoraciques. Tu n'es pas un dispositif médical."}]},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Décris brièvement ce que tu vois sur cette radiographie thoracique."},
    ]},
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

print("Génération en cours...")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=200)

input_len = inputs["input_ids"].shape[-1]
decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

print("\n=== Réponse MedGemma ===")
print(decoded)
print("\n=== Test terminé avec succès ===")
