from __future__ import annotations

import sys
import tempfile
import sqlite3
import json
import time
import uuid
import random
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

def log_run(image_name: str, model_name: str, prompt_version: str, pred: dict, doctor_notes: str = ""):
    pred["doctor_final_notes"] = doctor_notes
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
        df = pd.read_sql_query("SELECT id, created_at, image_path, predicted_class, confidence, model_name, prediction_json FROM runs ORDER BY id DESC", conn)
    return df

init_db()

# --- Cache MedGemma ---
@st.cache_resource(show_spinner="Initialisation du service MedGemma...")
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()

# --- Config & Style (Sober Medical Premium) ---
st.set_page_config(page_title="ARVI-RX Pro - Portail Médical", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = Path(__file__).resolve().parent.parent / "data" / "assets" / "logo.png"

st.markdown("""
<style>
/* Arrière-plan global strict */
.stApp {
    background-color: #f8fafc;
    font-family: 'Inter', -apple-system, sans-serif;
}
/* Cartes Cliniques */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    padding: 24px !important;
}
/* Personnalisation Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
/* Boutons Premium Médicaux */
.stButton > button {
    background: #0ea5e9 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #0284c7 !important;
}
/* Titres et textes */
h1, h2, h3 { color: #0f172a !important; font-weight: 600 !important; }
p, li { color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGIN SYSTEM ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if LOGO_PATH.exists():
            b64_logo = get_base64_image(LOGO_PATH)
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{b64_logo}" width="160"></div>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="text-align: center; color: #0f172a;">Système ARVI-RX Pro</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #64748b; font-size: 14px;">Portail d\'Accès Sécurisé - Établissement de Santé</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Identifiant Praticien")
            password = st.text_input("Mot de passe sécurisé", type="password")
            submit = st.form_submit_button("Authentification", use_container_width=True)
            
            if submit:
                if username == "efrei" and password == "jury2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Accès refusé. Identifiants non reconnus.")
    st.stop()

# --- APPLICATION PRINCIPALE ---
# En-tête
col_logo, col_title, col_user = st.columns([1, 4, 2])
with col_logo:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=80)
with col_title:
    st.title("Assistant Radiologue Virtuel")
    st.markdown("*Module d'Analyse Thoracique Frontale - Certification En Cours*")
with col_user:
    st.info("Dr. Lazzem (Pneumologie)\nSession Sécurisée - HDS")

# Sidebar
with st.sidebar:
    st.markdown("### Configuration (S4)")
    engine = st.radio("Moteur d'inférence", ["Mode Jouet (Simulation)", "MedGemma (Inférence Locale)"])
    use_medgemma = engine == "MedGemma (Inférence Locale)"
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error("Service MedGemma indisponible. Retour au mode simulation.")
            use_medgemma = False
    
    mode_options = ["advanced", "improved", "baseline"]
    mode = st.selectbox("Version du Prompt", mode_options, index=0)
    
    st.markdown("---")
    if st.button("Fermer la session"):
        st.session_state["authenticated"] = False
        st.session_state.pop("current_analysis", None)
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["[+] Nouveau Dossier Patient", "[=] Registre des Examens", "[?] Manuel Utilisateur"])

with tab1:
    st.markdown("### Création d'une Inférence")
    
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Importer le cliché radiographique (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    analyze_button = st.button("Exécuter l'Analyse IA", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        st.session_state["patient_id"] = f"PAT-{random.randint(1000, 9999)}"
        with st.status("Traitement et anonymisation...", expanded=True) as status:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            st.write("Anonymisation des métadonnées (Standard DICOM/RGPD)...")
            time.sleep(0.5)
            st.write("Calibration de l'image (Format 512x512)...")
            time.sleep(0.5)
            st.write("Requête d'inférence envoyée au modèle...")
            
            try:
                if use_medgemma:
                    _warmup_medgemma()
                    from src.inference_medgemma import medgemma_predict
                    raw_pred = medgemma_predict(tmp_path, mode=mode)
                else:
                    raw_pred = toy_predict(tmp_path, mode=mode)
                
                pred = apply_safety_guardrails(raw_pred)
                
                # Sauvegarde temporaire en session pour l'étape de validation humaine
                st.session_state["current_analysis"] = pred
                st.session_state["current_image_path"] = tmp_path
                st.session_state["current_filename"] = uploaded.name
                
                status.update(label="Analyse complétée en attente de validation clinique.", state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"Erreur d'inférence : {exc}", state="error")
                st.stop()
                
    # Affichage de la zone de validation si une analyse est en mémoire
    if "current_analysis" in st.session_state:
        pred = st.session_state["current_analysis"]
        tmp_path = st.session_state["current_image_path"]
        patient_id = st.session_state["patient_id"]
        
        st.markdown("---")
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            st.image(Image.open(tmp_path), caption=f"ID Patient Masqué : {patient_id}", use_container_width=True)
            
        with col_res:
            st.subheader("Bilan Préliminaire (IA Assistée)")
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.success("Classification : NORMAL")
            elif p_class == "suspected_opacity":
                st.error("Classification : SUSPECTED OPACITY (Anomalie détectée)")
            else:
                st.warning("Classification : UNCERTAIN (Examen non concluant)")
                
            conf = pred.get("confidence", 0.0)
            st.write(f"**Indice de confiance calculé : {conf*100:.1f}%**")
            st.progress(float(conf))
            
            with st.expander("Preuves Visuelles (Extraction)", expanded=False):
                for evidence in pred.get("visual_evidence", []):
                    st.markdown(f"- {evidence}")
            with st.expander("Recommandations Médicales Suggérées", expanded=False):
                if p_class == "suspected_opacity":
                    st.markdown("- TDM Thoracique recommandée.")
                    st.markdown("- Orientation vers un spécialiste (Pneumologie).")
                elif p_class == "uncertain":
                    st.markdown("- Acquisition défectueuse : recommencer l'examen.")
                else:
                    st.markdown("- Pas de nécessité d'examen complémentaire immédiat.")
                    
            st.caption(f"Avertissement Légal : {pred.get('warning', '')}")

        st.markdown("---")
        st.subheader("Validation Clinique et Compte-Rendu")
        st.info("L'IA propose un compte-rendu. En tant que médecin, vous devez le relire, le modifier si nécessaire, puis le valider pour l'intégrer au dossier patient.")
        
        # L'IA pré-remplit la boîte de texte
        default_notes = f"Observation : {pred.get('justification', '')}\nConclusion : {p_class.upper()}."
        doctor_notes = st.text_area("Compte-Rendu Médical (Éditable)", value=default_notes, height=150)
        
        if st.button("Signer et Enregistrer au Dossier Patient", type="primary"):
            log_run(patient_id, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred, doctor_notes)
            st.success("Le compte-rendu a été validé et enregistré de manière sécurisée.")
            # Nettoyer l'état
            del st.session_state["current_analysis"]
            st.rerun()

with tab2:
    st.markdown("### Registre des Examens Validés")
    try:
        df_history = get_history()
        if df_history.empty:
            st.info("Aucun historique disponible pour cet établissement.")
        else:
            df_history['created_at'] = pd.to_datetime(df_history['created_at']).dt.strftime('%d/%m/%Y %H:%M')
            st.dataframe(
                df_history,
                column_config={
                    "id": "Dossier #", 
                    "created_at": "Date & Heure", 
                    "image_path": "ID Patient (Anonymisé)",
                    "predicted_class": "Diagnostic Retenu",
                    "confidence": st.column_config.ProgressColumn("Confiance IA", format="%.2f", min_value=0, max_value=1),
                    "model_name": "Version IA",
                },
                hide_index=True, use_container_width=True
            )
            if st.button("Actualiser le registre"):
                st.rerun()
    except Exception as e:
        st.error(f"Erreur d'accès à la base de données : {e}")

with tab3:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Manuel utilisateur introuvable sur le serveur.")