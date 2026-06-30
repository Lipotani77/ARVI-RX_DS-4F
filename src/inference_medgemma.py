from pathlib import Path
from typing import Any
import json, os, re, time
# Sous Anaconda (Windows), MKL et PyTorch embarquent chacun leur copie de
# `libiomp5md.dll` → la 2e init du runtime OpenMP avorte le process (OMP: Error #15).
# On autorise la coexistence AVANT d'importer torch (sinon trop tard).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig

_PIPE = None
_PIPE_KEY = None  # (device, précision CPU) → on recharge si la config change

# En bfloat16, medgemma-4b pèse ~8 Go → il faut un GPU avec ≥ 8 Go de VRAM.
# En 4-bit (NF4), il tombe à ~3 Go → un petit GPU (ex. RTX 3050 4 Go) suffit.
MIN_VRAM_BYTES_4BIT = int(3.2 * 1024 ** 3)
# Sur un GPU plus costaud, on bascule en 16-bit (bfloat16) : ~8 Go de poids + la
# marge pour les activations / le cache (~10 Go au total). Aucune déquantification
# par couche → débit maximal. En dessous de ce seuil, on reste en 4-bit.
MIN_VRAM_BYTES_BF16 = int(10 * 1024 ** 3)


def _resolve_device(device: str | None = None) -> str:
    """ cette fonction permet de choisir sur quel device on va faire tourner le modèle (GPU ou CPU)

    Args:
        device (str | None): "cuda", "cpu" ou None pour auto-détection

    Returns:
        str: "cuda" si GPU avec assez de VRAM pour le 4-bit, sinon "cpu"
    """
    if device is not None: # si on indique explicitement le device, on ne fait pas d'auto-détection
        return device
    if torch.cuda.is_available(): # si CUDA est dispo, on regarde la VRAM libre pour savoir si on peut charger le modèle en 4-bit
        free_vram, _ = torch.cuda.mem_get_info()
        return "cuda" if free_vram >= MIN_VRAM_BYTES_4BIT else "cpu"
    return "cpu"


def _cuda_precision() -> str:
    """ si un gpu est dispo, on choisit la précision des poids côté GPU selon les ressources disponibles

    16-bit (bfloat16) dès qu'un GPU dispose d'assez de VRAM (≥ MIN_VRAM_BYTES_BF16) :
    pas de déquantification à chaque couche → performances maximales. Sinon 4-bit
    (NF4) pour tenir dans un petit GPU. À n'appeler que si CUDA est disponible.

    Returns:
        str: "bf16" sur un gros GPU, sinon "4bit"
    """
    free_vram, _ = torch.cuda.mem_get_info()
    return "bf16" if free_vram >= MIN_VRAM_BYTES_BF16 else "4bit"


def _cpu_precision() -> torch.dtype:
    """ Précision des poids en mode CPU

    Par défaut bfloat16 (~8 Go, peu de bande passante mémoire). Mais les CPU Intel
    sans unité bf16 native (ex. Alder Lake / i7-12650H) *émulent* le bf16 et sont
    souvent plus rapides en float32 — au prix de ~16 Go de RAM. À activer seulement
    si la RAM libre le permet, via la variable d'environnement
    `ARVI_CPU_PRECISION=float32` (sinon risque de swap = très lent).

    Returns:
        torch.dtype: torch.float32 si demandé explicitement, sinon torch.bfloat16
    """
    pref = os.environ.get("ARVI_CPU_PRECISION", "bfloat16").lower()
    return torch.float32 if pref in {"float32", "fp32"} else torch.bfloat16


def _get_pipe(device: str | None = None):
    """ cette fonction permet de ne charger medgemma qu'une seule fois. Si c'est la première fois, 3 cas de figure :
    1. GPU costaud : quantification 16-bit (bfloat16) pour débit maximal (≥ 10 Go de VRAM)
    2. Petit GPU : quantification 4-bit (NF4) pour tenir dans un petit GPU (≥ 3 Go de VRAM)
    3. CPU : bfloat16 par défaut (précision configurable, threads optionnels)

    Args:
        device (str | None, optional): "cuda" ou "cpu". None = auto-détection. Defaults to None.

    Returns:
        tuple[object, str, str]: (pipeline medgemma, device résolu, précision) — on
            renvoie aussi device/précision pour éviter de les recalculer côté appelant.
    """
    global _PIPE, _PIPE_KEY
    # Déjà chargé et aucun device explicitement imposé : on réutilise tel quel.
    # (On ne re-déduit PAS le backend depuis la VRAM libre, qui a chuté après le
    #  chargement — sinon la clé change et le modèle se recharge inutilement.)
    if _PIPE is not None and device is None:
        resolved, precision = _PIPE_KEY
        return _PIPE, resolved, precision

    resolved = _resolve_device(device)  # on résout le device (cuda ou cpu) selon la config de la machine
    cpu_dtype = _cpu_precision() # on résout la précision côté CPU
    precision = _cuda_precision() if resolved == "cuda" else str(cpu_dtype).replace("torch.", "")
    key = (resolved, precision)
    # on recharge si le pipeline n'existe pas ou si la config a changé
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE, resolved, precision

    if resolved == "cuda" and precision == "bf16":
        # GPU costaud : on charge les poids en 16-bit (bfloat16). Pas de
        # déquantification par couche → débit maximal. Coût : ~8 Go de VRAM.
        _PIPE = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            dtype=torch.bfloat16,
            model_kwargs={
                "device_map": "cuda",
                "low_cpu_mem_usage": True,
            },
        )
    elif resolved == "cuda":
        # Petit GPU : 4-bit NF4 → ~3 Go de VRAM (au lieu de ~8 en bf16). Le calcul
        # se fait en bfloat16, nativement géré par les GPU Ampere (RTX 30xx).
        # double_quant grappille encore un peu de VRAM sur les constantes de quant.
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _PIPE = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            model_kwargs={
                "quantization_config": quant,
                "device_map": "cuda",
                "low_cpu_mem_usage": True,
            },
        )
    else:
        # CPU : on peut limiter les threads aux cœurs physiques pour éviter la
        # contention hyper-threading (à régler/mesurer via ARVI_CPU_THREADS).
        threads = os.environ.get("ARVI_CPU_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        _PIPE = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            dtype=cpu_dtype,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": True},  # réduit le pic mémoire au chargement
        )
    _PIPE_KEY = key
    return _PIPE, resolved, precision

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
Chaque champ texte (visual_evidence, justification, limitations, warning) : 5 mots maximum.
En cas de doute, utilise la classe "uncertain".""",
    "improved": """Analyse cette radiographie thoracique frontale.
Réponds UNIQUEMENT par un JSON valide, sans aucun texte autour, au format exact :
{"image_quality":"bonne|moyenne|mauvaise","predicted_class":"normal|suspected_opacity|uncertain",
"confidence":0.0,"visual_evidence":"...","justification":"...","limitations":"...","warning":"..."}
En cas de doute, utilise la classe "uncertain".""",
    "advanced": """Analyse cette radiographie thoracique frontale.
Réponds UNIQUEMENT par un JSON valide, sans aucun texte autour, au format exact :
{"image_quality":"bonne|moyenne|mauvaise","predicted_class":"normal|suspected_opacity|uncertain",
"confidence":0.0,"visual_evidence":"...","justification":"...","limitations":"...","warning":"..."}
En cas de doute, utilise la classe "uncertain"."""
}

# Plafond de tokens par mode : baseline cherche la vitesse maximale (sortie
# courte, parfois tronquée → JSON parfois invalide, assumé) ; improved reste
# généreux pour produire un JSON complet et valide. latence ≈ nb tokens générés.
MAX_NEW_TOKENS = {"baseline": 96, "improved": 512, "advanced": 512}

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
    # On charge le pipeline AVANT de démarrer le chrono : le premier chargement
    # (téléchargement + mise en mémoire des poids) est un coût unique de plusieurs
    # minutes qui ne doit pas polluer la latence d'inférence *par image*. Le pipeline
    # nous renvoie aussi device/précision résolus → pas de recalcul redondant ensuite.
    pipe, resolved, precision = _get_pipe(device)
    # La latence ≈ nb de tokens générés × coût/token. On plafonne donc la sortie,
    # par mode : baseline serre fort (vitesse max, troncature/JSON invalide assumés),
    # improved reste généreux (JSON complet et valide). Cf. MAX_NEW_TOKENS.
    max_new_tokens = MAX_NEW_TOKENS.get(mode, 512)

    start = time.perf_counter()
    # IMPORTANT : les paramètres de génération doivent passer par `generate_kwargs`,
    # sinon le pipeline les route vers le *processor* qui les ignore silencieusement.
    # NB : on garde l'échantillonnage par défaut du modèle (la génération greedy
    # forcée dégradait la validité du JSON sur les images de test). La reproductibilité
    # déterministe sera traitée plus tard via le prompt (phase S3/S4), pas ici.
    out = pipe(
        text=messages,
        generate_kwargs={"max_new_tokens": max_new_tokens},
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    raw = out[0]["generated_text"][-1]["content"]
    result = _coerce_json(raw)
    # json_valid = le modèle a-t-il produit un JSON parsable ? `_coerce_json` n'ajoute
    # la clé `_raw` que sur son chemin de repli (parsing échoué) → marqueur fiable.
    # Sans ce champ, summarize_metrics retombe sur son défaut True → json_valid_rate
    # toujours à 1.0, aveugle aux sorties tronquées/non parsables.
    result["json_valid"] = "_raw" not in result
    # Champs runtime attendus par le pipeline (cf. schéma JSON du projet et eval/).
    backend = f"{resolved}/{precision}"
    result["model_name"] = f"google/medgemma-4b-it ({backend})"
    result["prompt_version"] = f"{mode}_v1"
    result["latency_ms"] = latency_ms
    return result

def _coerce_json(raw: str) -> dict[str, Any]:
    """ Cette fonction tente d'extraire un JSON valide de la sortie brute du modèle

    Args:
        raw (str): _description_

    Returns:
        dict[str, Any]: _description_
    """
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
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
