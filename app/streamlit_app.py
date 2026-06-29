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

def _as_bullet_list(value) -> list[str]:
    """Normalise un champ JSON (liste OU chaîne) en liste de puces.

    MedGemma renvoie parfois `visual_evidence`/`limitations` sous forme de
    chaîne alors que le schéma attend une liste : itérer dessus afficherait
    une puce par caractère. On enveloppe donc une chaîne dans une liste à un
    seul élément.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


# --- Configuration SQLite ---
DB_PATH = Path(__file__).resolve().parent.parent / "database.sqlite"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval" / "outputs"

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

def get_runs_df() -> pd.DataFrame:
    """Récupère les colonnes utiles aux statistiques d'exécution (avec latence)."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, created_at, predicted_class, confidence, latency_ms, model_name "
            "FROM runs ORDER BY id",
            conn,
        )
    return df

def load_eval_metrics(mode: str) -> dict | None:
    """Lit eval/outputs/{mode}_metrics.json (déjà produit par run_evaluation.py).

    Renvoie le dict de métriques (contenant le bloc `targets` + `all_targets_met`),
    ou None si le fichier n'existe pas / est illisible.
    """
    path = EVAL_DIR / f"{mode}_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Nouvelle Analyse",
    "🗂️ Historique des analyses",
    "📊 Métriques",
    "❓ Guide Utilisateur",
])

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
                log_run(uploaded.name, pred.get("model_name", "toy"), f"prompt_{mode}", pred)
                
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

            # Métriques simples de cette analyse (sans graphique)
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Temps d'inférence", f"{pred.get('latency_ms', 0)} ms")
            m_col2.metric("Confiance", f"{conf*100:.1f}%")
            m_col3.metric("Qualité d'image", pred.get("image_quality", "—"))

            # Détails structurés (plus de JSON brut)
            with st.expander("📌 Preuves Visuelles", expanded=True):
                evidences = _as_bullet_list(pred.get("visual_evidence"))
                if evidences:
                    for evidence in evidences:
                        st.markdown(f"- {evidence}")
                else:
                    st.caption("Aucune preuve visuelle fournie.")
                    
            with st.expander("📝 Justification Médicale", expanded=True):
                st.write(pred.get("justification", "Aucune justification fournie."))
                
            with st.expander("🛑 Limites de l'analyse", expanded=False):
                limits = _as_bullet_list(pred.get("limitations"))
                if limits:
                    for limit in limits:
                        st.markdown(f"- {limit}")
                else:
                    st.caption("Aucune limite renseignée.")
                    
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
    st.header("Métriques")

    # --- Section A : métriques d'évaluation (jeu de test, vérité-terrain) ---
    st.subheader("Métriques d'évaluation (jeu de test)")
    st.caption(
        "Calculées hors-ligne par `eval/run_evaluation.py` et lues depuis "
        "`eval/outputs/`. Comparées aux objectifs du projet."
    )

    eval_mode = st.selectbox("Mode évalué", ["improved", "baseline"], index=0)
    m = load_eval_metrics(eval_mode)

    if m is None:
        st.info(
            "Aucun fichier de métriques trouvé pour ce mode. "
            "Générez-les avec :\n\n"
            "```bash\npython eval/run_evaluation.py --mode toy\n```"
        )
    else:
        targets = m.get("targets", {})

        def _target_caption(key: str) -> str:
            t = targets.get(key)
            if not t:
                return ""
            badge = "✅" if t.get("pass") else "❌"
            return f"🎯 Objectif {t.get('operator', '')} {t.get('threshold', '')} {badge}"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{m.get('accuracy', 0):.2f}")
            st.caption(_target_caption("accuracy"))
        with c2:
            st.metric("Macro-F1", f"{m.get('macro_f1', 0):.2f}")
            st.caption(_target_caption("macro_f1"))
        with c3:
            st.metric("Latence moy. (ms)", f"{m.get('avg_latency_ms', 0)}")
            st.caption(_target_caption("avg_latency_ms"))
        with c4:
            st.metric("JSON valide", f"{m.get('json_valid_rate', 0):.2f}")
            st.caption(_target_caption("json_valid_rate"))

        c5, c6, c7 = st.columns(3)
        c5.metric("Nb. de cas", m.get("n", 0))
        c6.metric("Taux d'avertissement", f"{m.get('warning_rate', 0):.2f}")
        c7.metric("Taux 'uncertain'", f"{m.get('uncertain_rate', 0):.2f}")

        if m.get("all_targets_met"):
            st.success("✅ Tous les objectifs du projet sont atteints pour ce mode.")
        else:
            st.warning("⚠️ Certains objectifs du projet ne sont pas atteints.")

        st.caption(
            "ℹ️ Avec le backend *jouet*, la latence d'évaluation est ~0 ms "
            "(inférence instantanée). La latence réaliste apparaît dans les "
            "statistiques d'exécution ci-dessous."
        )

    st.divider()

    # --- Section B : statistiques d'exécution live (database.sqlite) ---
    st.subheader("Statistiques d'exécution (analyses de l'application)")
    st.caption(
        "Dérivées des prédictions réellement effectuées dans l'application "
        "(pas de vérité-terrain → pas d'accuracy/F1 ici)."
    )

    df_runs = get_runs_df()
    if df_runs.empty:
        st.info("Aucune analyse enregistrée pour le moment. Lancez une analyse dans l'onglet « Nouvelle Analyse ».")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Analyses effectuées", len(df_runs))
        k2.metric("Temps moyen / image (ms)", f"{df_runs['latency_ms'].mean():.0f}")
        k3.metric("Confiance moyenne", f"{df_runs['confidence'].mean():.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Distribution des classes prédites**")
            st.bar_chart(df_runs["predicted_class"].value_counts())
        with col_b:
            st.markdown("**Répartition des confiances**")
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
            conf_binned = pd.cut(
                df_runs["confidence"], bins=bins, labels=labels, include_lowest=True
            )
            st.bar_chart(conf_binned.value_counts().sort_index())

        st.markdown("**Latence dans le temps (ms)**")
        df_lat = df_runs.copy()
        df_lat["created_at"] = pd.to_datetime(df_lat["created_at"])
        st.line_chart(df_lat.set_index("created_at")["latency_ms"])

        st.markdown("**Latence moyenne par modèle (ms)**")
        st.bar_chart(df_runs.groupby("model_name")["latency_ms"].mean())

        if st.button("🔄 Rafraîchir les métriques"):
            st.rerun()

with tab4:
    guide_path = Path(__file__).resolve().parent.parent / "docs" / "guide_utilisateur.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Le fichier du guide utilisateur est introuvable.")