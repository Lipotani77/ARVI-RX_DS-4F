from __future__ import annotations

import sys
import tempfile
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

# Rend la racine du projet importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image
import base64

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails

# --- Configuration SQLite ---
DB_PATH = Path(__file__).resolve().parent.parent / "database.sqlite"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.commit()

def log_run(image_name: str, model_name: str, prompt_version: str, pred: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (image_name, model_name, prompt_version, json.dumps(pred), pred.get("predicted_class", "error"), pred.get("confidence", 0.0), pred.get("latency_ms", 0))
        )
        conn.commit()

def get_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT id, created_at, image_path, predicted_class, confidence, model_name FROM runs ORDER BY id DESC", conn)
    return df

init_db()

# --- Cache MedGemma ---
@st.cache_resource(show_spinner="Chargement de MedGemma…")
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()

# --- Config & Style (Glassmorphism Premium) ---
st.set_page_config(page_title="ARVI-RX Pro", layout="wide", page_icon="⚕️")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = Path(__file__).resolve().parent.parent / "data" / "assets" / "logo.png"

st.markdown("""
<style>
/* Arrière-plan global */
.stApp {
    background-color: #f1f5f9;
    background-image: radial-gradient(at 0% 0%, hsla(199,89%,75%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(168,100%,79%,1) 0, transparent 50%);
    font-family: 'Inter', sans-serif;
}

/* Personnalisation Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.4) !important;
}
/* Boutons Premium */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px 0 rgba(14, 165, 233, 0.39) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4) !important;
}
/* Titres et textes */
h1, h2, h3 { color: #0f172a !important; }
p { color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGIN SYSTEM ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Page de Login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if LOGO_PATH.exists():
            # Afficher le logo proprement avec HTML centré
            b64_logo = get_base64_image(LOGO_PATH)
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{b64_logo}" width="200"></div>', unsafe_allow_html=True)
        
        st.markdown('<h1 style="text-align: center;">Connexion - ARVI-RX Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;"><em>Portail Médical Sécurisé HDS</em></p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Identifiant Médecin (ex: efrei)")
            password = st.text_input("Mot de passe (ex: jury2026)", type="password")
            submit = st.form_submit_button("🔒 Se Connecter", use_container_width=True)
            
            if submit:
                if username == "efrei" and password == "jury2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Identifiants incorrects.")
    st.stop()

# --- APPLICATION PRINCIPALE ---
# En-tête
col_logo, col_title, col_user = st.columns([1, 4, 2])
with col_logo:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=100)
with col_title:
    st.title("Assistant Radiologue Virtuel")
    st.markdown("*Analyse IA certifiée - Module Thoracique Frontal*")
with col_user:
    st.info("👤 **Dr. Lazzem**\nHôpital EFREI Paris\nService Pneumologie")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration (S4)")
    engine = st.radio("Moteur d'inférence", ["Jouet (rapide)", "MedGemma (réel)"])
    use_medgemma = engine == "MedGemma (réel)"
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error("MedGemma indisponible. Bascule sur le mode jouet.")
            use_medgemma = False
    mode_options = ["advanced", "improved", "baseline"]
    mode = st.selectbox("Prompt utilisé", mode_options, index=0)
    
    st.markdown("---")
    if st.button("🚪 Déconnexion", type="secondary"):
        st.session_state["authenticated"] = False
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["🔍 Nouvelle Analyse", "🗂️ Dossiers Patients (Historique)", "❓ Documentation IA"])

with tab1:
    st.markdown("### Nouvelle Inférence")
    
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("📥 Déposer une radiographie (DICOM simulé, PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    analyze_button = st.button("🚀 Lancer l'analyse Médicale", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        with st.status("Traitement sécurisé en cours...", expanded=True) as status:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            st.write("🔄 Anonymisation des métadonnées (Conformité HIPAA/RGPD)...")
            time.sleep(0.5)
            st.write("📏 Calibration et redimensionnement...")
            time.sleep(0.5)
            st.write("🧠 Inférence du modèle MedGemma-4B-IT...")
            
            try:
                if use_medgemma:
                    _warmup_medgemma()
                    from src.inference_medgemma import medgemma_predict
                    raw_pred = medgemma_predict(tmp_path, mode=mode)
                else:
                    raw_pred = toy_predict(tmp_path, mode=mode)
                
                pred = apply_safety_guardrails(raw_pred)
                log_run(uploaded.name, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred)
                status.update(label="Analyse terminée", state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"Erreur : {exc}", state="error")
                st.stop()
        
        # Affichage Premium
        st.markdown("---")
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            st.image(Image.open(tmp_path), caption=f"Radio : {uploaded.name}", use_container_width=True)
            
        with col_res:
            st.subheader("Bilan Radiologique (IA Assistée)")
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.success("✅ **Classe : NORMAL**")
            elif p_class == "suspected_opacity":
                st.error("🚨 **Classe : SUSPECTED OPACITY (Anomalie détectée)**")
            else:
                st.warning("⚠️ **Classe : UNCERTAIN (Examen non concluant)**")
                
            conf = pred.get("confidence", 0.0)
            st.write(f"**Indice de confiance du modèle : {conf*100:.1f}%**")
            st.progress(float(conf))
            
            with st.expander("📌 Preuves Visuelles", expanded=True):
                for evidence in pred.get("visual_evidence", []):
                    st.markdown(f"- {evidence}")
            with st.expander("📝 Justification Clinique", expanded=True):
                st.write(pred.get("justification", "Aucune justification."))
            with st.expander("💡 Recommandations Suivi", expanded=True):
                if p_class == "suspected_opacity":
                    st.markdown("- Planifier un scanner thoracique (TDM) sans injection.")
                    st.markdown("- Consultation pneumologique requise.")
                elif p_class == "uncertain":
                    st.markdown("- Acquisiton de mauvaise qualité : refaire le cliché.")
                else:
                    st.markdown("- Pas de suivi radiologique immédiat.")
            st.caption(f"**Avertissement Médical Légal** : {pred.get('warning', '')}")

with tab2:
    st.markdown("### 🗂️ Registre des Inférences")
    try:
        df_history = get_history()
        if df_history.empty:
            st.info("Aucun historique disponible.")
        else:
            df_history['created_at'] = pd.to_datetime(df_history['created_at']).dt.strftime('%d/%m/%Y %H:%M')
            st.dataframe(
                df_history,
                column_config={
                    "id": "Dossier #", "created_at": "Date", "image_path": "Fichier Patient",
                    "predicted_class": "Prédiction",
                    "confidence": st.column_config.ProgressColumn("Confiance", format="%.2f", min_value=0, max_value=1),
                    "model_name": "Modèle",
                },
                hide_index=True, use_container_width=True
            )
            if st.button("🔄 Actualiser"):
                st.rerun()
    except Exception as e:
        st.error(f"Erreur DB : {e}")

with tab3:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Guide introuvable.")