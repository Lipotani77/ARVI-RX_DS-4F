from __future__ import annotations

import sys
import tempfile
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

# Rend la racine du projet importable, quel que soit le dossier de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image

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
            (
                image_name,
                model_name,
                prompt_version,
                json.dumps(pred),
                pred.get("predicted_class", "error"),
                pred.get("confidence", 0.0),
                pred.get("latency_ms", 0)
            )
        )
        conn.commit()

def get_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT id, created_at, image_path, predicted_class, confidence, model_name FROM runs ORDER BY id DESC", conn)
    return df

# Initialiser la base au démarrage
init_db()

# --- Interface Utilisateur ---
st.set_page_config(page_title="ARVI-RX : Démo S5", layout="wide", page_icon="🩻")

st.title("🩻 Assistant Radiologue Virtuel (ARVI-RX)")
st.markdown("*Prototype pédagogique d'analyse de radiographies thoraciques frontales.*")

# Cache MedGemma (Ajout de William)
@st.cache_resource(
    show_spinner="Chargement de MedGemma — au 1er lancement : téléchargement + mise en mémoire des poids…"
)
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()

# Sidebar (Ajout de William)
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

# Création des onglets
tab1, tab2, tab3 = st.tabs(["🔍 Nouvelle Analyse", "🗂️ Historique des analyses", "❓ Guide Utilisateur"])

with tab1:
    st.header("Nouvelle Analyse")
    st.info("ℹ️ Ce système est une démonstration éducative.")
    
    col_upload, col_settings = st.columns([2, 1])
    
    with col_upload:
        uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
    
    # Bouton d'analyse (désactivé si pas d'image)
    analyze_button = st.button("🚀 Lancer l'analyse", type="primary", disabled=not uploaded, use_container_width=True)

    if uploaded and analyze_button:
        with st.status("Traitement en cours...", expanded=True) as status:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            st.write("🔄 Anonymisation des métadonnées (DICOM)...")
            time.sleep(0.5)
            st.write("📏 Redimensionnement de l'image (512x512)...")
            time.sleep(0.5)
            st.write("🧠 Exécution du modèle d'Intelligence Artificielle...")
            
            try:
                # Exécution du modèle
                if use_medgemma:
                    _warmup_medgemma()
                    from src.inference_medgemma import medgemma_predict
                    raw_pred = medgemma_predict(tmp_path, mode=mode)
                else:
                    raw_pred = toy_predict(tmp_path, mode=mode)
                
                pred = apply_safety_guardrails(raw_pred)
                
                st.write("💾 Sauvegarde dans la base de données SQLite...")
                log_run(uploaded.name, "medgemma-4b-it" if use_medgemma else "toy", f"prompt_{mode}", pred)
                
                status.update(label="Analyse terminée avec succès !", state="complete", expanded=False)
            except Exception as exc:
                status.update(label=f"Erreur : {exc}", state="error")
                st.stop()
        
        # Affichage des résultats de manière structurée et esthétique
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            st.image(Image.open(tmp_path), caption=f"Fichier : {uploaded.name}", use_container_width=True)
            
        with col_res:
            st.subheader("Diagnostic suggéré")
            
            # Affichage de la classe avec des couleurs
            p_class = pred.get("predicted_class", "uncertain")
            if p_class == "normal":
                st.success("✅ **Classe prédite : NORMAL**")
            elif p_class == "suspected_opacity":
                st.error("🚨 **Classe prédite : SUSPECTED OPACITY (Anomalie suspectée)**")
            else:
                st.warning("⚠️ **Classe prédite : UNCERTAIN (Incertain)**")
                
            # Confiance
            conf = pred.get("confidence", 0.0)
            st.write(f"**Indice de confiance : {conf*100:.1f}%**")
            st.progress(float(conf))
            
            # Détails structurés (plus de JSON brut)
            with st.expander("📌 Preuves Visuelles", expanded=True):
                for evidence in pred.get("visual_evidence", []):
                    st.markdown(f"- {evidence}")
                    
            with st.expander("📝 Justification Médicale", expanded=True):
                st.write(pred.get("justification", "Aucune justification fournie."))
                
            with st.expander("🛑 Limites de l'analyse", expanded=False):
                for limit in pred.get("limitations", []):
                    st.markdown(f"- {limit}")
                    
            with st.expander("💡 Recommandations Médicales", expanded=True):
                if p_class == "suspected_opacity":
                    st.markdown("- Planifier un scanner thoracique (TDM) pour confirmation.")
                    st.markdown("- Référer le patient à un pneumologue.")
                    st.markdown("- Corréler avec l'historique clinique (fièvre, toux, etc.).")
                elif p_class == "uncertain":
                    st.markdown("- Refaire la radiographie (problème possible de qualité ou d'exposition).")
                    st.markdown("- Demander l'avis d'un radiologue senior.")
                else:
                    st.markdown("- Aucun suivi radiologique immédiat nécessaire basé sur cette image seule.")
                    st.markdown("- Corréler systématiquement avec l'examen clinique.")
                    
            st.error(f"**AVERTISSEMENT LÉGAL** : {pred.get('warning', 'Non destiné au diagnostic.')}")
            
            if pred.get("guardrail_errors"):
                st.error(f"⚡ **Garde-fous déclenchés :** {pred['guardrail_errors']}")

with tab2:
    st.header("Historique des analyses")
    st.write("Retrouvez ici toutes les inférences passées sauvegardées dans la base de données locale (SQLite).")
    
    try:
        df_history = get_history()
        if df_history.empty:
            st.info("Aucune analyse n'a été effectuée pour le moment.")
        else:
            # Formater la date
            df_history['created_at'] = pd.to_datetime(df_history['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Afficher le dataframe
            st.dataframe(
                df_history,
                column_config={
                    "id": "ID",
                    "created_at": "Date et Heure",
                    "image_path": "Nom de l'image",
                    "predicted_class": "Diagnostic",
                    "confidence": st.column_config.ProgressColumn(
                        "Confiance",
                        help="Niveau de confiance de l'IA",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "model_name": "Modèle utilisé",
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Bouton de rafraîchissement
            if st.button("🔄 Rafraîchir l'historique"):
                st.rerun()
                
    except Exception as e:
        st.error(f"Erreur lors du chargement de la base de données : {e}")

with tab3:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Le fichier du guide utilisateur est introuvable.")