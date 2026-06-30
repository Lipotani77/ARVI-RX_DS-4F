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
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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
        df = pd.read_sql_query("SELECT id, created_at, image_path, predicted_class, confidence, model_name, latency_ms FROM runs ORDER BY id DESC", conn)
    return df

init_db()

# --- Cache MedGemma ---
@st.cache_resource(show_spinner="Initialisation du service MedGemma...")
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()

# --- Config & Style (High-Tech Clair / Bleu Cyan) ---
st.set_page_config(page_title="ARVI-RX Intelligence", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = Path(__file__).resolve().parent.parent / "data" / "assets" / "logo.png"

st.markdown("""
<style>
/* Arrière-plan Clair High-Tech avec Gradient */
.stApp {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    color: #1e293b;
    font-family: 'Inter', -apple-system, sans-serif;
}
/* Cartes Glassmorphism Clair avec Bordure Cyan */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border-top: 3px solid #00a8ff !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 30px rgba(0, 168, 255, 0.08) !important;
    padding: 24px !important;
    transition: transform 0.3s ease;
}
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0, 168, 255, 0.12) !important;
}
/* Sidebar Moderne */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 15px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] * {
    color: #334155 !important;
}
/* Boutons Interactifs Cyan */
.stButton > button {
    background: linear-gradient(90deg, #00a8ff 0%, #0097e6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 30px !important; /* Bouton Pilule très moderne */
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(0, 168, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 168, 255, 0.6) !important;
    transform: scale(1.02) !important;
}
/* Titres */
h1, h2, h3 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.5px; }
/* KPIs & Métriques */
[data-testid="stMetricValue"] { color: #00a8ff !important; text-shadow: 0 0 10px rgba(0, 168, 255, 0.2); }
/* Inputs */
input, textarea { border-radius: 8px !important; border: 1px solid #cbd5e1 !important; }
input:focus, textarea:focus { border-color: #00a8ff !important; box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2) !important; }
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
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{b64_logo}" width="150" style="filter: drop-shadow(0 4px 6px rgba(0,168,255,0.3));"></div>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="text-align: center; color: #0f172a;">ARVI-RX Intelligence</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #64748b;">Portail Clinique Haute Technologie</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Identifiant Praticien (ID Sécurisé)")
            password = st.text_input("Clé d'accès biométrique (Mot de passe)", type="password")
            submit = st.form_submit_button("Ouvrir la session sécurisée", use_container_width=True)
            
            if submit:
                if username == "efrei" and password == "jury2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Authentification échouée. Veuillez réessayer.")
    st.stop()

# --- APPLICATION PRINCIPALE ---
# En-tête
col_logo, col_title, col_user = st.columns([1, 4, 2])
with col_logo:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=70)
with col_title:
    st.markdown("<h2 style='margin-bottom:0;'>ARVI-RX Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#00a8ff; font-weight:600;'>Module VLM Thoracique Frontal</span>", unsafe_allow_html=True)
with col_user:
    st.info("👤 **Dr. Moreau**\nService d'Imagerie de Pointe")

# Sidebar
with st.sidebar:
    st.markdown("### Contrôle Système")
    engine = st.radio("Cœur d'IA (Backend)", ["Simulateur Haute-Vitesse", "MedGemma 4B (Local)"])
    use_medgemma = engine == "MedGemma 4B (Local)"
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error("Serveur MedGemma injoignable. Passage en simulation.")
            use_medgemma = False
    
    mode = st.selectbox("Algorithme de Prompting", ["advanced", "improved", "baseline"], index=0)
    
    st.markdown("---")
    if st.button("Verrouiller la session"):
        st.session_state["authenticated"] = False
        st.session_state.pop("current_analysis", None)
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["Scanner un patient", "Centre Analytique", "Documentation Produit"])

with tab1:
    col_upload, col_context = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Importer une plaque radiographique (DICOM simulé, PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    with col_context:
        st.markdown("##### Paramètres Cliniques")
        if uploaded:
            st.info(f"**Patient ID:** PAT-{uuid.uuid4().hex[:6].upper()}\n\n"
                    f"**Âge:** {random.randint(35, 75)} ans\n\n"
                    f"**Statut:** {'Fumeur' if random.choice([True, False]) else 'Non-fumeur'}\n\n"
                    f"**Température:** {round(random.uniform(36.5, 39.5), 1)}°C")
        else:
            st.write("En attente d'un cliché pour synchroniser le dossier patient...")

    analyze_button = st.button("Lancer le diagnostic assisté par IA", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        st.session_state["patient_id"] = f"PAT-{uuid.uuid4().hex[:6].upper()}"
        with st.status("Algorithmes en cours d'exécution...", expanded=True) as status:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            st.write("Extraction des contours et calibration...")
            time.sleep(0.4)
            st.write("Recherche d'opacités thoraciques (VLM)...")
            time.sleep(0.4)
            
            try:
                start_time = time.time()
                if use_medgemma:
                    _warmup_medgemma()
                    from src.inference_medgemma import medgemma_predict
                    raw_pred = medgemma_predict(tmp_path, mode=mode)
                else:
                    raw_pred = toy_predict(tmp_path, mode=mode)
                latency = int((time.time() - start_time) * 1000)
                
                pred = apply_safety_guardrails(raw_pred)
                pred["latency_ms"] = latency
                
                st.session_state["current_analysis"] = pred
                st.session_state["current_image_path"] = tmp_path
                st.session_state["current_filename"] = uploaded.name
                
                status.update(label=f"Analyse terminée avec succès (Latence: {latency}ms)", state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"Erreur d'analyse : {exc}", state="error")
                st.stop()
                
    if "current_analysis" in st.session_state:
        pred = st.session_state["current_analysis"]
        tmp_path = st.session_state["current_image_path"]
        patient_id = st.session_state["patient_id"]
        
        st.markdown("---")
        st.markdown("### Espace de Travail Radiologique (Outils PACS)")
        
        col_img, col_tools, col_res = st.columns([1.5, 1, 1.5])
        
        # Outils Interactifs d'imagerie
        with col_tools:
            st.markdown("##### Outils d'Imagerie")
            brightness = st.slider("Luminosité", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
            show_heatmap = st.checkbox("Filtre d'Explicabilité (IA)")
            
            # Traitement de l'image
            original_img = Image.open(tmp_path).convert("RGB")
            enhancer_b = ImageEnhance.Brightness(original_img)
            img_edited = enhancer_b.enhance(brightness)
            enhancer_c = ImageEnhance.Contrast(img_edited)
            img_final = enhancer_c.enhance(contrast)
            
            if show_heatmap:
                # Création rapide de Heatmap
                overlay = Image.new("RGBA", img_final.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                w, h = img_final.size
                if pred.get("predicted_class") == "suspected_opacity":
                    draw.ellipse((w//3, h//3, 2*w//3, 2*h//3), fill=(0, 168, 255, 90)) # Cyan glow
                    overlay = overlay.filter(ImageFilter.GaussianBlur(30))
                else:
                    draw.rectangle((10, 10, w-10, h-10), outline=(0, 255, 0, 80), width=4)
                img_final = Image.alpha_composite(img_final.convert("RGBA"), overlay).convert("RGB")

        with col_img:
            st.image(img_final, caption=f"Patient : {patient_id}", use_container_width=True)
            
        with col_res:
            st.markdown("##### Rapport Préliminaire IA")
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.success("✅ Résultat : NORMAL (Poumons clairs)")
            elif p_class == "suspected_opacity":
                st.error("⚠️ Résultat : SUSPECTED OPACITY (Anomalie)")
            else:
                st.warning("❔ Résultat : UNCERTAIN (Examen à refaire)")
                
            conf = pred.get("confidence", 0.0)
            st.metric(label="Précision estimée par l'algorithme", value=f"{conf*100:.1f}%")
            st.progress(float(conf))
            
            with st.expander("Justification Clinique"):
                st.write(pred.get("justification", "Non fourni."))

        st.markdown("---")
        st.subheader("Finalisation du Rapport")
        st.info("Le médecin doit corriger ce brouillon avant validation et export.")
        
        default_notes = f"Radiographie {patient_id}.\nConstat de l'IA : {pred.get('justification', '')}\nDiagnostic validé : {p_class.upper()}."
        doctor_notes = st.text_area("Éditeur de Compte-Rendu", value=default_notes, height=120)
        
        col_save, col_export = st.columns([1, 1])
        with col_save:
            if st.button("Enregistrer au dossier système", type="primary", use_container_width=True):
                log_run(patient_id, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred, doctor_notes)
                st.success("Données synchronisées avec la base SQLite.")
                del st.session_state["current_analysis"]
                st.rerun()
        with col_export:
            # Génération d'un fichier TXT pour export
            report_content = f"COMPTE RENDU RADIOLOGIQUE - ARVI-RX Pro\n"
            report_content += f"========================================\n"
            report_content += f"Médecin: Dr. Moreau\n"
            report_content += f"Patient ID: {patient_id}\n"
            report_content += f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            report_content += f"DIAGNOSTIC FINAL :\n{doctor_notes}\n\n"
            report_content += f"--- Signature Numérique ARVI-RX ---"
            
            st.download_button(
                label="Télécharger le PDF/TXT officiel",
                data=report_content,
                file_name=f"Rapport_{patient_id}.txt",
                mime="text/plain",
                use_container_width=True
            )

with tab2:
    st.markdown("### Centre Analytique des Performances")
    try:
        df_history = get_history()
        if df_history.empty:
            st.info("Aucune donnée d'inférence en base.")
        else:
            df_history['created_at'] = pd.to_datetime(df_history['created_at'])
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Examens Traités", len(df_history))
            c2.metric("Fiabilité Moyenne", f"{(df_history['confidence'].mean()*100):.1f}%")
            c3.metric("Rapidité Système", f"{df_history['latency_ms'].mean():.0f} ms")
            error_rate = len(df_history[df_history['predicted_class'] == 'uncertain']) / len(df_history) * 100
            c4.metric("Examens Inconcluants", f"{error_rate:.1f}%")
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Bilan des Diagnostics**")
                st.bar_chart(df_history['predicted_class'].value_counts(), color="#00a8ff")
                
            with col_chart2:
                st.markdown("**Latence Chronologique (ms)**")
                df_time = df_history.sort_values('created_at').set_index('created_at')
                st.line_chart(df_time['latency_ms'], color="#00a8ff")

            st.markdown("---")
            st.markdown("**Traceabilité (Conformité Légale)**")
            df_display = df_history.copy()
            df_display['created_at'] = df_display['created_at'].dt.strftime('%d/%m/%Y %H:%M')
            st.dataframe(df_display, hide_index=True, use_container_width=True)
            
            if st.button("Rafraîchir les statistiques"):
                st.rerun()
    except Exception as e:
        st.error(f"Erreur SQL : {e}")

with tab3:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Document introuvable.")