from pathlib import Path
from typing import Any
import json, re, torch
from PIL import Image
from transformers import pipeline

_PIPE = None
_PIPE_DEVICE = None

MIN_VRAM_BYTES = 8 * 1024 ** 3  # 8 Go minimum pour medgemma-4b en bfloat16

def _resolve_device(device: str | None = None) -> str:
    """ Détermine le device à utiliser selon la config de la machine

    Args:
        device (str | None): "cuda", "cpu" ou None pour auto-détection

    Returns:
        str: "cuda" si GPU disponible avec assez de VRAM, sinon "cpu"
    """
    if device is None:
        if torch.cuda.is_available():
            free_vram, _ = torch.cuda.mem_get_info()
            return "cuda" if free_vram >= MIN_VRAM_BYTES else "cpu"
        return "cpu"
    return device

def _get_pipe(device: str | None = None):
    """ cette fonction permet de ne charger medgemma qu'une seule fois

    Args:
        device (str | None, optional): "cuda" ou "cpu". None = auto-détection. Defaults to None.

    Returns:
        object: objet pipeline de medgemma
    """
    global _PIPE, _PIPE_DEVICE
    resolved = _resolve_device(device) # on résout le device demandé
    # on recharge si le pipeline n'existe pas ou si le device demandé a changé
    if _PIPE is None or _PIPE_DEVICE != resolved:
        # bfloat16 partout : ~2x moins de RAM que float32 (~8 Go au lieu de ~16 Go),
        _PIPE = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device=resolved,
            model_kwargs={"low_cpu_mem_usage": True}, # low_cpu_mem_usage réduit le pic de mémoire pendant le chargement
        )
        _PIPE_DEVICE = resolved
    return _PIPE

# ======================================================================================
# Prompt et fonction de prédiction pour MedGemma
# ======================================================================================

SYSTEM = ("Tu es un assistant pédagogique d'analyse de radiographies thoraciques. "
          "Tu n'es pas un dispositif médical.")


# le paramètre 'mode' sert à comparer baseline vs prompt amélioré
# le prompt cidessous est un prompt test
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
def medgemma_predict(image_path: str | Path, mode: str = "baseline", device: str | None = None) -> dict[str, Any]:
    """ Cette fonction permet d'obtenir la prédiction de MedGemma pour une image donnée

    Args:
        image_path (str | Path): path vers l'image à analyser
        mode (str, optional): mode : choix entre baseline et improved. Defaults to "baseline"
        device (str | None, optional): "cuda" ou "cpu" selon la config de la machine.
            None = auto-détection (cuda si disponible, sinon cpu). Defaults to None.

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
    resolved = _resolve_device(device)
    # en CPU on génère moins de tokens : plus rapide et moins de RAM sur les petits PC.
    # Le JSON attendu reste court, 256 tokens suffisent largement.
    max_new_tokens = 256 if resolved == "cpu" else 512
    out = _get_pipe(device)(text=messages, max_new_tokens=max_new_tokens)
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
