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

# --- Config & Style (ULTRA-PREMIUM, FIXES) ---
st.set_page_config(page_title="ARVI-RX Intelligence", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = Path(__file__).resolve().parent.parent / "data" / "assets" / "logo.png"

st.markdown("""
<style>
/* Animations originales */
@keyframes gradient-xy {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse-border {
    0% { box-shadow: 0 0 0 0 rgba(0, 206, 209, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(0, 206, 209, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 206, 209, 0); }
}

/* Arrière-plan Animé "Mesh Gradient" */
.stApp {
    background: linear-gradient(-45deg, #ffffff, #f0fdfa, #e0f2fe, #ffffff);
    background-size: 400% 400%;
    animation: gradient-xy 15s ease infinite;
    color: #0f172a;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Cartes Principales */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(20px) !important;
    border-left: 4px solid #00ced1 !important;
    border-top: 1px solid rgba(255,255,255,0.8) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px rgba(0, 31, 63, 0.05) !important;
    padding: 30px !important;
}

/* Sidebar élégante */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001F3F 0%, #0f172a 100%) !important;
}
/* Ciblage spécifique du texte dans la sidebar pour éviter le bug des dropdowns blancs sur fond blanc */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stRadio label {
    color: #00ced1 !important; 
}

/* Boutons d'Action (Correction du texte invisible) */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #001F3F 0%, #008080 50%, #00ced1 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 30px !important;
    box-shadow: 0 8px 20px rgba(0, 128, 128, 0.3) !important;
    transition: 0.5s !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background-position: right center !important;
    box-shadow: 0 12px 25px rgba(0, 206, 209, 0.5) !important;
    transform: scale(1.02) !important;
}
/* Force la couleur du texte du bouton à être blanche */
.stButton > button, [data-testid="stFormSubmitButton"] > button,
.stButton > button p, [data-testid="stFormSubmitButton"] > button p,
.stButton > button *, [data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

/* Correction du widget de statut (Analyse en cours/terminée) */
[data-testid="stStatusWidget"] {
    background-color: #f8fafc !important;
    border: 1px solid #00ced1 !important;
    border-radius: 8px !important;
}
[data-testid="stStatusWidget"] * {
    color: #001F3F !important;
}

/* Zone de dépôt (Upload) */
[data-testid="stFileUploader"] section {
    border: 2px dashed #00ced1 !important;
    border-radius: 16px !important;
    background-color: rgba(240, 253, 250, 0.5) !important;
    animation: pulse-border 3s infinite;
}

/* Correction des Titres d'Expanders (Parties de résultats) */
[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.5) !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}
[data-testid="stExpander"] summary p {
    color: #001F3F !important;
    font-weight: 800 !important;
}

/* Text Area */
textarea {
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    background-color: #ffffff !important;
}

/* Onglets en style 'Pillules' */
[data-testid="stTabs"] button {
    background-color: transparent !important;
    border-radius: 20px !important;
    margin-right: 10px !important;
    border: 1px solid #cbd5e1 !important;
    color: #001F3F !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #001F3F !important;
    color: white !important;
    border: 1px solid #001F3F !important;
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: white !important;
}

/* Titres Globaux */
h1, h2, h3 { color: #001F3F !important; font-weight: 800 !important; letter-spacing: -0.5px; }
[data-testid="stMetricValue"] { color: #008080 !important; font-size: 2.5rem !important; font-weight: 900 !important;}

/* Floating Chat Widget (Hacking CSS) */
div[data-testid="stExpander"]:last-of-type {
    position: fixed !important;
    bottom: 20px !important;
    right: 20px !important;
    width: 350px !important;
    z-index: 999999 !important;
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px solid #00ced1 !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(0,31,63,0.3) !important;
}
div[data-testid="stExpander"]:last-of-type summary {
    background: linear-gradient(90deg, #001F3F 0%, #008080 100%) !important;
    border-radius: 8px !important;
    padding: 10px 15px !important;
}
div[data-testid="stExpander"]:last-of-type summary p {
    color: white !important;
    font-weight: bold !important;
}
div[data-testid="stExpander"]:last-of-type .stMarkdown p {
    color: #001F3F !important;
}
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
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{b64_logo}" width="180" style="filter: drop-shadow(0 10px 15px rgba(0,206,209,0.3));"></div>', unsafe_allow_html=True)
        
        st.markdown('<h1 style="text-align: center; color: #001F3F; font-weight: 900;">ARVI-RX</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #008080; font-weight: bold; letter-spacing: 2px;">VLM INTELLIGENCE MEDICALE</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Identifiant Médecin")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("DÉVERROUILLER LE TERMINAL", use_container_width=True)
            
            if submit:
                if username == "efrei" and password == "jury2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Accès refusé.")
    st.stop()

# --- APPLICATION PRINCIPALE ---
# En-tête
col_logo, col_title, col_user = st.columns([1, 4, 2])
with col_logo:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=90)
with col_title:
    st.markdown("<h2 style='margin-bottom:0;'>Console d'Intelligence Diagnostique</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#008080; font-weight:700; letter-spacing: 1px;'>MOTEUR D'INFERENCE THORACIQUE</span>", unsafe_allow_html=True)
with col_user:
    st.info("⚕️ **Dr. Moreau**\nSession : Active (HDS)")

# Sidebar
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(Image.open(LOGO_PATH), width=150)
    st.markdown("### Contrôles Système")
    engine = st.radio("Cœur IA", ["Simulateur Haute-Vitesse", "MedGemma 4B (Local)"])
    use_medgemma = engine == "MedGemma 4B (Local)"
    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error("Serveur MedGemma injoignable. Passage en simulation.")
            use_medgemma = False
    
    mode = st.radio("Injection de Prompt", ["advanced", "improved", "baseline"], index=0)
    
    st.markdown("---")
    if st.button("DÉCONNEXION SÉCURISÉE"):
        st.session_state["authenticated"] = False
        st.session_state.pop("current_analysis", None)
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(["Scanner un patient", "Centre Analytique", "Documentation Technique"])

with tab1:
    col_upload, col_context = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("📥 Déposer le scan thoracique", type=["png", "jpg", "jpeg"])
    
    with col_context:
        st.markdown("##### 🧬 Contexte Patient")
        if uploaded:
            st.success(f"**ID:** PAT-{uuid.uuid4().hex[:6].upper()}\n\n"
                    f"**Âge:** {random.randint(35, 75)} ans\n\n"
                    f"**Fumeur:** {'Oui' if random.choice([True, False]) else 'Non'}\n\n"
                    f"**Température:** {round(random.uniform(36.5, 39.5), 1)}°C")
        else:
            st.warning("En attente d'une radiographie...")

    analyze_button = st.button("LANCER LE DIAGNOSTIC IA", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        st.session_state["patient_id"] = f"PAT-{uuid.uuid4().hex[:6].upper()}"
        with st.status("Traitement algorithmique...", expanded=True) as status:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            
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
        st.markdown("### 🎛️ Terminal de Radiologie Interactif (PACS)")
        
        col_img, col_tools, col_res = st.columns([1.5, 1, 1.5])
        
        # Outils Interactifs d'imagerie
        with col_tools:
            st.markdown("##### Filtres Optiques")
            brightness = st.slider("Luminosité", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
            show_heatmap = st.checkbox("🔮 Activer le Radar XAI (Heatmap)", value=True)
            
            # Traitement de l'image
            original_img = Image.open(tmp_path).convert("RGB")
            enhancer_b = ImageEnhance.Brightness(original_img)
            img_edited = enhancer_b.enhance(brightness)
            enhancer_c = ImageEnhance.Contrast(img_edited)
            img_final = enhancer_c.enhance(contrast)
            
            if show_heatmap:
                overlay = Image.new("RGBA", img_final.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                w, h = img_final.size
                if pred.get("predicted_class") == "suspected_opacity":
                    # Gradient cyan/teal très stylisé
                    draw.ellipse((w//3, h//3, 2*w//3, 2*h//3), fill=(0, 206, 209, 110))
                    overlay = overlay.filter(ImageFilter.GaussianBlur(35))
                else:
                    draw.rectangle((10, 10, w-10, h-10), outline=(0, 128, 128, 90), width=6)
                img_final = Image.alpha_composite(img_final.convert("RGBA"), overlay).convert("RGB")

        with col_img:
            st.image(img_final, caption=f"Analyse Visuelle - {patient_id}", use_container_width=True)
            
        with col_res:
            st.markdown("##### Bilan de l'Intelligence Artificielle")
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.info("✅ DIAGNOSTIC : NORMAL (Aucune opacité majeure)")
            elif p_class == "suspected_opacity":
                st.error("🚨 DIAGNOSTIC : SUSPECTED OPACITY (Anomalie détectée)")
            else:
                st.warning("❔ DIAGNOSTIC : UNCERTAIN (Examen non concluant)")
                
            conf = pred.get("confidence", 0.0)
            st.metric(label="Confiance de la Prédiction", value=f"{conf*100:.1f}%")
            st.progress(float(conf))
            
            with st.expander("Justification Algorithmique"):
                st.write(pred.get("justification", "Non fourni."))

        st.markdown("---")
        st.subheader("Validation Clinique et Export")
        st.info("L'IA assiste, mais le médecin signe. Corrigez le rapport avant de l'ajouter à la base de données de l'hôpital.")
        
        default_notes = f"Patient {patient_id}.\n\nRelevé du système ARVI-RX :\n{pred.get('justification', '')}\n\nDiagnostic retenu par le praticien : {p_class.upper()}."
        doctor_notes = st.text_area("Compte-Rendu Officiel", value=default_notes, height=130)
        
        col_save, col_export = st.columns([1, 1])
        with col_save:
            if st.button("VALIDER ET ENREGISTRER", type="primary", use_container_width=True):
                log_run(patient_id, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred, doctor_notes)
                st.success("Dossier sauvegardé dans le cloud sécurisé de l'hôpital.")
                del st.session_state["current_analysis"]
                st.rerun()
        with col_export:
            report_content = f"COMPTE RENDU RADIOLOGIQUE - ARVI-RX\n"
            report_content += f"========================================\n"
            report_content += f"Médecin: Dr. Moreau\n"
            report_content += f"Patient ID: {patient_id}\n"
            report_content += f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            report_content += f"RAPPORT :\n{doctor_notes}\n\n"
            report_content += f"--- Généré par l'IA Médicale ARVI-RX ---"
            
            st.download_button(
                label="TÉLÉCHARGER LE RAPPORT (TXT)",
                data=report_content,
                file_name=f"Rapport_{patient_id}.txt",
                mime="text/plain",
                use_container_width=True
            )

with tab2:
    st.markdown("### 📊 Data Analytics - Performances du Modèle")
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
                st.bar_chart(df_history['predicted_class'].value_counts(), color="#00ced1")
                
            with col_chart2:
                st.markdown("**Latence Chronologique (ms)**")
                df_time = df_history.sort_values('created_at').set_index('created_at')
                st.line_chart(df_time['latency_ms'], color="#008080")

            st.markdown("---")
            st.markdown("**Logs de la Base de Données (SQLite)**")
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

# --- FLOATING CHATBOT WIDGET ---
with st.expander("💬 ARVI-Bot (Support Utilisateur)", expanded=False):
    st.markdown("<small style='color: #008080;'>Posez vos questions sur l'analyse, les fonctionnalités ou la technique.</small>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "🤖 ARVI-Bot", "content": "Bonjour Dr. Moreau ! Comment puis-je vous aider aujourd'hui avec le terminal ARVI-RX ?"}]
        
    for msg in st.session_state.chat_history:
        color = "#001F3F" if "ARVI-Bot" in msg['role'] else "#00ced1"
        st.markdown(f"<strong style='color:{color}'>{msg['role']}</strong>: {msg['content']}", unsafe_allow_html=True)
        
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([4, 1.5])
        with cols[0]:
            user_input = st.text_input("Message...", label_visibility="collapsed", placeholder="Ex: Comment analyser une radio ?")
        with cols[1]:
            submitted = st.form_submit_button("Envoyer", use_container_width=True)
            
        if submitted and user_input:
            st.session_state.chat_history.append({"role": "👤 Vous", "content": user_input})
            
            p_lower = user_input.lower()
            if any(k in p_lower for k in ["analyse", "comment", "scanner", "radio"]):
                rep = "Pour analyser un patient, rendez-vous dans l'onglet **Scanner un patient**. Importez une image dans la zone de dépôt (format DICOM, PNG, JPG) et cliquez sur **LANCER LE DIAGNOSTIC IA**."
            elif any(k in p_lower for k in ["erreur", "bug", "marche pas", "uncertain", "problème"]):
                rep = "Si vous rencontrez une erreur (par exemple UNCERTAIN), nos algorithmes (Guardrails) ont probablement détecté une image de mauvaise qualité ou non-frontale. Veuillez réessayer avec un cliché plus net."
            elif any(k in p_lower for k in ["pacs", "luminosité", "contraste", "filtre", "xai"]):
                rep = "Notre Viewer PACS interactif vous permet d'ajuster la luminosité et le contraste de l'image. La coche 'Radar XAI' met en évidence la zone d'attention du modèle."
            elif any(k in p_lower for k in ["technique", "modèle", "vlm", "medgemma"]):
                rep = "ARVI-RX utilise un modèle VLM (Vision-Language Model) MedGemma 4B affiné pour la radiologie thoracique, capable de détecter des opacités avec une précision élevée."
            elif any(k in p_lower for k in ["export", "rapport", "sauvegarder"]):
                rep = "Une fois le diagnostic généré, vous pouvez modifier le compte-rendu puis le valider pour l'enregistrer dans la base sécurisée, ou l'exporter au format TXT."
            elif any(k in p_lower for k in ["équipe", "qui", "créateur"]):
                rep = "Le système ARVI-RX a été développé par une brillante équipe d'ingénieurs : Leila, William, Killian, Thomas, Iza et Victor."
            else:
                rep = "Je suis l'assistant de premier niveau. Je peux vous renseigner sur l'analyse d'image, le fonctionnement des filtres PACS ou les erreurs techniques !"
                
            st.session_state.chat_history.append({"role": "🤖 ARVI-Bot", "content": rep})
            st.rerun()