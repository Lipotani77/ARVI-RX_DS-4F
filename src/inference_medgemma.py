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

# Enum des classes autorisées : source unique de vérité, partagée avec le garde-fou
# (src/guardrails.py). On normalise la classe prédite contre cet ensemble pour qu'un
# « Normal » ou « suspected opacity » du modèle ne soit pas traité comme invalide.
from src.guardrails import ALLOWED_CLASSES

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


# Les prompts (un par mode) sont maintenus dans des fichiers .txt à part, sous
# `prompts/`. On les charge à l'import pour que toute modification de ces fichiers
# soit prise en compte sans toucher au code. Le module est dans `src/`, donc la
# racine du dépôt = parents[1].
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_FILES = {
    "baseline": "baseline_prompt.txt",
    "improved": "improved_prompt.txt",
    "advanced": "advanced_prompt.txt",
}


def _load_prompt(filename: str) -> str:
    """Lit le contenu d'un fichier de prompt sous `prompts/`."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# le paramètre 'mode' sert à comparer baseline vs prompt amélioré
PROMPTS = {mode: _load_prompt(fname) for mode, fname in _PROMPT_FILES.items()}

# Plafond de tokens identique pour tous les modes : le schéma JSON (7 champs)
# doit pouvoir se fermer entièrement quel que soit le prompt. En égalisant le
# budget, l'écart de json_valid_rate / accuracy entre baseline et improved
# mesure la qualité du PROMPT et non la troncature (un baseline trop serré
# rendait le JSON systématiquement invalide, un artefact de budget, pas de
# prompt). latence ≈ nb de tokens générés. Dict conservé pour permettre une
# future modulation par mode si besoin.
MAX_NEW_TOKENS = {"baseline": 512, "improved": 512, "advanced": 512}

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
    # La latence ≈ nb de tokens générés × coût/token. On plafonne la sortie au
    # même budget pour tous les modes : le JSON (7 champs) doit avoir la place de
    # se fermer, sinon json_valid_rate refléterait la troncature et non le prompt.
    # Cf. MAX_NEW_TOKENS.
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
    # json_valid = le modèle a-t-il produit un JSON *réellement* parsable et complet ?
    # `_coerce_json` pose `_json_valid` (booléen explicite) : il vaut False dès qu'on a dû
    # récupérer la classe d'une sortie tronquée, même si la prédiction est correcte. Ainsi
    # json_valid_rate expose la troncature (la bonne métrique) sans pour autant pénaliser
    # l'accuracy (cf. récupération de la classe dans `_coerce_json`).
    result["json_valid"] = result.pop("_json_valid", False)
    # Champs runtime attendus par le pipeline (cf. schéma JSON du projet et eval/).
    backend = f"{resolved}/{precision}"
    result["model_name"] = f"google/medgemma-4b-it ({backend})"
    result["prompt_version"] = f"{mode}_v1"
    result["latency_ms"] = latency_ms
    return result

# Repli sentinelle quand la classe prédite est *réellement* irrécupérable (ni un objet
# JSON parsable, ni même un `predicted_class` exploitable dans le texte tronqué).
_UNPARSABLE = {
    "image_quality": "mauvaise", "predicted_class": "uncertain", "confidence": 0.0,
    "visual_evidence": "", "justification": "Sortie non parsable.",
    "limitations": "JSON invalide renvoyé par le modèle.",
    "warning": "Résultat non exploitable, relecture nécessaire.",
}


def _normalize_class(value: Any) -> str | None:
    """Normalise une classe prédite contre l'enum `ALLOWED_CLASSES`.

    Tolère la casse et les séparateurs : « Normal », « suspected opacity »,
    « suspected-opacity » → `normal` / `suspected_opacity`. Renvoie None si la
    valeur ne correspond à aucune classe autorisée (→ le garde-fou prendra le relais).
    """
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    return key if key in ALLOWED_CLASSES else None


def _salvage_fields(text: str) -> dict[str, Any]:
    """Récupère les champs scalaires d'un objet JSON *tronqué* (sans accolade fermante).

    Le plafond `MAX_NEW_TOKENS` coupe souvent la sortie avant le `}` final : `json.loads`
    échoue alors complètement alors que `predicted_class` (2e champ du schéma) a déjà été
    émis. On l'extrait par regex ciblée, avec `image_quality` et `confidence` (présents tôt
    eux aussi) pour éviter que le garde-fou ne dégrade la prédiction. Les champs descriptifs
    coupés en plein milieu ne sont pas récupérés (ils n'influent pas sur le scoring).
    """
    salvaged: dict[str, Any] = {}
    mc = re.search(r'"predicted_class"\s*:\s*"([^"]+)"', text)
    if mc:
        salvaged["predicted_class"] = mc.group(1)
    mq = re.search(r'"image_quality"\s*:\s*"([^"]*)"', text)
    if mq:
        salvaged["image_quality"] = mq.group(1)
    mf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if mf:
        try:
            salvaged["confidence"] = float(mf.group(1))
        except ValueError:
            pass
    return salvaged


def _coerce_json(raw: str) -> dict[str, Any]:
    """Extrait une prédiction exploitable de la sortie brute du modèle.

    Parser durci : (1) strip des fences markdown, (2) parsing du 1er objet `{...}` bien
    fermé, (3) à défaut — sortie tronquée — récupération ciblée des champs scalaires,
    (4) normalisation de la classe contre l'enum, (5) repli sentinelle seulement si la
    classe reste irrécupérable. Pose `_json_valid` (validité JSON *réelle*, False dès qu'on
    a dû récupérer) pour que l'accuracy mesure le modèle et json_valid_rate la troncature.
    """
    # 1) Strip des fences markdown (```json ... ```), en tête comme en queue.
    cleaned = re.sub(r"^\s*```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # 2) Chemin nominal : 1er objet {...} bien fermé → json.loads.
    data: dict[str, Any] = {}
    json_ok = False
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                data = parsed
                json_ok = True
        except json.JSONDecodeError:
            pass

    # 3) Repli sur sortie tronquée : on complète depuis le texte brut sans écraser ce qui
    #    a été parsé proprement. Champs descriptifs manquants → défauts (le scoring n'en
    #    dépend pas ; ils servent juste à satisfaire le schéma du garde-fou).
    if "predicted_class" not in data:
        salvaged = _salvage_fields(cleaned)
        if "predicted_class" in salvaged:
            data = {**_UNPARSABLE, **salvaged, **data}

    # 4) Normalisation de la classe contre l'enum, y compris sur le chemin nominal
    #    (un JSON parfaitement valide peut contenir « Normal »).
    norm = _normalize_class(data.get("predicted_class"))

    # 5) Classe irrécupérable → repli sentinelle (le garde-fou forcera `uncertain`).
    if norm is None:
        return {**_UNPARSABLE, "_json_valid": False, "_raw": raw}

    data["predicted_class"] = norm
    # 6) Marqueur de validité JSON *réelle* : une sortie tronquée mais récupérée reste
    #    invalide (visible dans json_valid_rate), même si la classe prédite est correcte.
    data["_json_valid"] = json_ok
    if not json_ok:
        data["_raw"] = raw
    return data
