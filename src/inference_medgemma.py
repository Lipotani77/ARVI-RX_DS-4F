from pathlib import Path
from typing import Any
import json, re, torch
from PIL import Image
from transformers import pipeline

_PIPE = None
def _get_pipe():
    """ cette fonction permet de ne charger qu'une fois medgemma qu'une seule fois

    Returns:
        object: objet pipeline de medgemma
    """
    global _PIPE
    if _PIPE is None:
        _PIPE = pipeline("image-text-to-text", model="google/medgemma-4b-it", torch_dtype=torch.bfloat16, device="cuda")
    return _PIPE

# ======================================================================================
# Prompt et fonction de prédiction pour MedGemma
# ======================================================================================

SYSTEM = ("Tu es un assistant pédagogique d'analyse de radiographies thoraciques. "
          "Tu n'es pas un dispositif médical.")


# le paramètre 'mode' sert à comparer baseline vs prompt amélioré
PROMPTS = {
    "baseline": """Analyse cette radiographie thoracique frontale.
Réponds UNIQUEMENT par un JSON valide, sans aucun texte autour, au format exact :
{"image_quality":"bonne|moyenne|mauvaise","predicted_class":"normal|suspected_opacity|uncertain",
"confidence":0.0,"visual_evidence":"...","justification":"...","limitations":"...","warning":"..."}
En cas de doute, utilise la classe "uncertain".""",
    # "improved": "...prompt renforcé : contrôle qualité image, pas d'invention, seuil d'incertitude..."
}

# ======================================================================================
# pipeline de prédiction pour MedGemma
# ======================================================================================
def medgemma_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """ Cette fonction permet d'obtenir la prédiction de MedGemma pour une image donnée

    Args:
        image_path (str | Path): path vers l'image à analyser
        mode (str, optional): mode : choix entre baseline et improved. Defaults to "baseline"

    Returns:
        dict[str, Any]: dictionnaire contenant la prédiction de MedGemma
    """
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[mode]},
        ]},
    ]
    out = _get_pipe()(text=messages, max_new_tokens=512)
    raw = out[0]["generated_text"][-1]["content"]
    return _coerce_json(raw)

def _coerce_json(raw: str) -> dict[str, Any]:
    """ Cette fonction tente d'extraire un JSON valide de la sortie brute du modèle

    Args:
        raw (str): _description_

    Returns:
        dict[str, Any]: _description_
    """
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}
    if "predicted_class" not in data:   # contrat non respecté → garde-fou
        return {"image_quality": "mauvaise", "predicted_class": "uncertain",
                "confidence": 0.0, "visual_evidence": "", "justification": "Sortie non parsable.",
                "limitations": "JSON invalide renvoyé par le modèle.",
                "warning": "Résultat non exploitable, relecture nécessaire.", "_raw": raw}
    return data
