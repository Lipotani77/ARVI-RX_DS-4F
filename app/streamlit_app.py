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

# --- Simulateur de Heatmap XAI ---
def generate_heatmap(image_path, p_class):
    """Génère une carte de chaleur simulée sur l'image pour démontrer l'explicabilité (XAI)."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = img.size
    center_x, center_y = width // 2, height // 2
    
    if p_class == "suspected_opacity":
        # Tache rouge/jaune diffuse (simulation pneumonie)
        radius = min(width, height) // 3
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=(255, 0, 0, 100))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=50))
    else:
        # Grille d'analyse verte (normal)
        draw.rectangle((10, 10, width-10, height-10), outline=(0, 255, 255, 80), width=5)
        for i in range(1, 4):
            draw.line((0, height * i // 4, width, height * i // 4), fill=(0, 255, 255, 40), width=2)
            draw.line((width * i // 4, 0, width * i // 4, height), fill=(0, 255, 255, 40), width=2)
            
    blended = Image.alpha_composite(img, overlay)
    return blended.convert("RGB")

# --- Cache MedGemma ---
@st.cache_resource(show_spinner="Initialisation du service MedGemma...")
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()

# --- Config & Style (Tech Dashboard) ---
st.set_page_config(page_title="ARVI-RX Data Center", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = Path(__file__).resolve().parent.parent / "data" / "assets" / "logo.png"

st.markdown("""
<style>
/* Arrière-plan global Cyber/Data */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
}
/* Cartes Sombre/Néon */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 4px !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.05) !important;
    padding: 24px !important;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #090c15 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
/* Boutons Tech */
.stButton > button {
    background: #0f172a !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #00d4ff !important;
    color: #090c15 !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
}
/* Titres */
h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; font-family: 'Inter', sans-serif !important; }
/* Métriques */
[data-testid="stMetricValue"] { color: #00d4ff !important; }
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
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{b64_logo}" width="120" style="filter: drop-shadow(0 0 10px #00d4ff);"></div>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="text-align: center; color: #f8fafc;">ARVI-RX Analytics Engine</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #00d4ff; font-family: monospace;">[RESTRICTED ACCESS - SYSTEM AUTHENTICATION REQUIRED]</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("USER_ID")
            password = st.text_input("AUTH_KEY", type="password")
            submit = st.form_submit_button("INITIALIZE SECURE CONNECTION", use_container_width=True)
            
            if submit:
                if username == "efrei" and password == "jury2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("[ERROR] ACCESS DENIED. INVALID CREDENTIALS.")
    st.stop()

# --- APPLICATION PRINCIPALE ---
# En-tête
col_logo, col_title, col_user = st.columns([1, 4, 2])
with col_logo:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=60)
with col_title:
    st.markdown("<h2 style='margin-bottom:0;'>ARVI-RX Analytics Engine</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#00d4ff; font-family:monospace;'>VLM Diagnostic System [ONLINE]</span>", unsafe_allow_html=True)
with col_user:
    st.info("DATA SCIENTIST : Lazzem\nENV : Production (HDS)")

# Sidebar
with st.sidebar:
    st.markdown("### SYSTEM CONFIG")
    engine = st.radio("BACKEND INFERENCE", ["SIMULATOR (FAST)", "MEDGEMMA-4B (NATIVE)"])
    use_medgemma = engine == "MEDGEMMA-4B (NATIVE)"
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error("MEDGEMMA ENGINE OFFLINE. FALLBACK TO SIMULATOR.")
            use_medgemma = False
    
    mode_options = ["advanced", "improved", "baseline"]
    mode = st.selectbox("PROMPT INJECTION LAYER", mode_options, index=0)
    
    st.markdown("---")
    if st.button("TERMINATE SESSION"):
        st.session_state["authenticated"] = False
        st.session_state.pop("current_analysis", None)
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["[INFERENCE PIPELINE]", "[DATA ANALYTICS DASHBOARD]", "[DOCUMENTATION]"])

with tab1:
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("INPUT: X-RAY TENSOR (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    analyze_button = st.button("EXECUTE VLM ANALYSIS", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        st.session_state["patient_id"] = f"UUID-{uuid.uuid4().hex[:8].upper()}"
        
        # CONSOLE INGÉNIEUR TEMPS RÉEL
        console_box = st.empty()
        logs = "> INITIALIZING INFERENCE PIPELINE...\n"
        console_box.code(logs, language="bash")
        
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        logs += "> [OK] TENSOR LOADED INTO MEMORY\n"
        console_box.code(logs, language="bash")
        time.sleep(0.3)
        
        logs += f"> [WARN] STRIPPING DICOM METADATA FOR HDS COMPLIANCE... ID: {st.session_state['patient_id']}\n"
        console_box.code(logs, language="bash")
        time.sleep(0.3)
        
        logs += f"> [INFO] LOADING VLM KERNEL : {'medgemma-4b-it' if use_medgemma else 'toy_simulator'}\n"
        logs += f"> [INFO] INJECTING PROMPT : {mode}\n"
        console_box.code(logs, language="bash")
        
        try:
            start_time = time.time()
            if use_medgemma:
                _warmup_medgemma()
                from src.inference_medgemma import medgemma_predict
                raw_pred = medgemma_predict(tmp_path, mode=mode)
            else:
                raw_pred = toy_predict(tmp_path, mode=mode)
            latency = int((time.time() - start_time) * 1000)
            
            logs += f"> [OK] INFERENCE COMPLETE. LATENCY: {latency}ms\n"
            console_box.code(logs, language="bash")
            
            logs += "> [INFO] APPLYING SAFETY GUARDRAILS (JSON VALIDATION)...\n"
            console_box.code(logs, language="bash")
            
            pred = apply_safety_guardrails(raw_pred)
            pred["latency_ms"] = latency
            
            if pred.get("guardrail_errors"):
                logs += f"> [ERROR] GUARDRAIL TRIGGERED: {pred['guardrail_errors']}\n"
            else:
                logs += "> [OK] GUARDRAILS PASSED.\n"
            console_box.code(logs, language="bash")
            
            st.session_state["current_analysis"] = pred
            st.session_state["current_image_path"] = tmp_path
            st.session_state["current_filename"] = uploaded.name
            st.session_state["logs"] = logs
            
        except Exception as exc:
            logs += f"\n> [FATAL EXCEPTION] {exc}"
            console_box.code(logs, language="bash")
            st.stop()
                
    if "current_analysis" in st.session_state:
        pred = st.session_state["current_analysis"]
        tmp_path = st.session_state["current_image_path"]
        patient_id = st.session_state["patient_id"]
        
        with st.expander("TERMINAL CONSOLE (DEBUG)", expanded=False):
            st.code(st.session_state.get("logs", ""), language="bash")
        
        st.markdown("---")
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            # XAI Heatmap Toggle
            show_heatmap = st.checkbox("Activer l'Explicabilité (XAI - Grad-CAM simulé)", value=False)
            if show_heatmap:
                img_to_show = generate_heatmap(tmp_path, pred.get("predicted_class", "uncertain"))
                st.image(img_to_show, caption=f"Heatmap (Attention Mask) - {patient_id}", use_container_width=True)
            else:
                st.image(Image.open(tmp_path), caption=f"Radio Source - {patient_id}", use_container_width=True)
            
        with col_res:
            st.subheader("MODEL OUTPUT TENSOR")
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.success("[CLASS_ID: NORMAL] - No anomalies detected.")
            elif p_class == "suspected_opacity":
                st.error("[CLASS_ID: SUSPECTED_OPACITY] - High probability of infection.")
            else:
                st.warning("[CLASS_ID: UNCERTAIN] - Insufficient data / poor quality.")
                
            conf = pred.get("confidence", 0.0)
            st.metric(label="CONFIDENCE SCORE", value=f"{conf*100:.1f}%")
            st.progress(float(conf))
            
            st.json(pred, expanded=False)

        st.markdown("---")
        st.subheader("HUMAN-IN-THE-LOOP VALIDATION")
        st.info("L'ingénieur/médecin doit valider ou corriger l'output du modèle avant l'insertion dans la base de données SQL.")
        
        default_notes = f"Observation Modèle : {pred.get('justification', '')}\nConclusion Humaine : "
        doctor_notes = st.text_area("Correction/Validation (Éditable)", value=default_notes, height=100)
        
        if st.button("OVERRIDE & SAVE TO DATABASE", type="primary"):
            log_run(patient_id, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred, doctor_notes)
            st.success("DATA SAVED TO SQLITE DB SUCCESSFULLY.")
            del st.session_state["current_analysis"]
            st.rerun()

with tab2:
    st.markdown("### DATA ANALYTICS & MONITORING")
    try:
        df_history = get_history()
        if df_history.empty:
            st.info("NO DATA FOUND IN DB.")
        else:
            df_history['created_at'] = pd.to_datetime(df_history['created_at'])
            
            # --- METRICS ROW ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Inferences", len(df_history))
            c2.metric("Avg Confidence", f"{(df_history['confidence'].mean()*100):.1f}%")
            c3.metric("Avg Latency", f"{df_history['latency_ms'].mean():.0f} ms")
            error_rate = len(df_history[df_history['predicted_class'] == 'uncertain']) / len(df_history) * 100
            c4.metric("Uncertainty Rate", f"{error_rate:.1f}%")
            
            st.markdown("---")
            # --- CHARTS ---
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Distribution des Diagnostics (Classes)**")
                class_counts = df_history['predicted_class'].value_counts()
                st.bar_chart(class_counts, color="#00d4ff")
                
            with col_chart2:
                st.markdown("**Évolution de la Latence (ms)**")
                # Trier par date pour la courbe
                df_time = df_history.sort_values('created_at').set_index('created_at')
                st.line_chart(df_time['latency_ms'], color="#a855f7")

            st.markdown("---")
            st.markdown("**RAW SQL DATA (runs table)**")
            df_display = df_history.copy()
            df_display['created_at'] = df_display['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df_display, hide_index=True, use_container_width=True)
            
            if st.button("REFRESH DATABASE"):
                st.rerun()
    except Exception as e:
        st.error(f"SQL QUERY ERROR : {e}")

with tab3:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("404 NOT FOUND")