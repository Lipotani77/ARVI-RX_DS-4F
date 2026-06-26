from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Rend la racine du projet importable, quel que soit le dossier de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")
st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning("Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.")


# MedGemma tire torch + transformers (lourd). On le charge UNE fois et on le garde
# vivant entre les reruns Streamlit. `st.cache_resource` sert le spinner « 1er
# chargement » ; le pipeline lui-même est mémoïsé côté src.inference_medgemma.
@st.cache_resource(
    show_spinner="Chargement de MedGemma — au 1er lancement : téléchargement + mise en mémoire des poids (plusieurs minutes)…"
)
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()


# --- Barre latérale : choix du moteur d'inférence + infos matériel --------------
with st.sidebar:
    st.header("Moteur d'inférence")
    engine = st.radio(
        "Backend",
        ("MedGemma (réel)", "Jouet (rapide)"),
        help=(
            "MedGemma : vrai VLM médical (google/medgemma-4b-it), lent, GPU recommandé. "
            "Jouet : validateur de tuyauterie, lit le label dans le nom de fichier."
        ),
    )
    use_medgemma = engine.startswith("MedGemma")

    # Si MedGemma est demandé mais que les dépendances lourdes manquent, on retombe
    # proprement sur le jouet au lieu de faire planter toute l'app au démarrage.
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:  # torch/transformers absents ou cassés
            st.error(f"MedGemma indisponible ({type(exc).__name__}). Bascule sur le mode jouet.")
            use_medgemma = False

    # Le jouet gère baseline ET improved ; MedGemma n'expose que les prompts
    # réellement implémentés (PROMPTS) — `improved` arrivera en S3/S4.
    mode_options = list(mg.PROMPTS.keys()) if use_medgemma else ["baseline", "improved"]
    mode = st.selectbox("Mode (prompt)", mode_options)

    if use_medgemma:
        device = mg._resolve_device()
        st.caption(f"Device détecté : **{device}**")
        if device == "cpu":
            st.warning("Pas de GPU exploitable : l'inférence CPU peut prendre **plusieurs minutes** par image.")
        else:
            st.caption("Comptez ~20 s/image (GPU 4-bit). Voir docs/latence_et_materiel.md.")


uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
    with col2:
        try:
            if use_medgemma:
                _warmup_medgemma()  # chargement unique, avec son propre spinner
                from src.inference_medgemma import medgemma_predict
                with st.spinner("Analyse par MedGemma en cours…"):
                    raw = medgemma_predict(tmp_path, mode=mode)
            else:
                raw = toy_predict(tmp_path, mode=mode)
            pred = apply_safety_guardrails(raw)
        except Exception as exc:  # démo : une sortie modèle imprévue ne doit pas tuer l'app
            st.error(f"Échec de l'inférence : {type(exc).__name__} — {exc}")
            st.stop()

        st.metric("Classe", pred["predicted_class"])
        st.metric("Confiance", pred["confidence"])
        st.write("**Observations**", pred["visual_evidence"])
        st.write("**Justification**", pred["justification"])
        st.write("**Limites**", pred["limitations"])
        # Traçabilité attendue par le sujet (preuves de soutenance).
        st.caption(
            f"Modèle : `{pred.get('model_name', '?')}` · prompt : `{pred.get('prompt_version', '?')}` · "
            f"latence : {pred.get('latency_ms', '?')} ms"
        )
        if pred.get("guardrail_errors"):
            st.warning(f"Garde-fous déclenchés : {pred['guardrail_errors']}")
        st.json(pred)
else:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux (mode jouet).")
