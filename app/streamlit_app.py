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

#Config
st.set_page_config(
    page_title="ARVI-RX · Assistant Radiologue Virtuel",
    page_icon="⊕",
    layout="wide",
    initial_sidebar_state="expanded",
)

#CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {
    background-color: #0B1220;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0F1923 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    color: #e2e8f0 !important;
    letter-spacing: 0.3px !important;
    padding-bottom: 0 !important;
}

h2, h3 {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #00B4D8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] {
    background: #0f172a !important;
    border: 1.5px dashed #1e293b !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: #00B4D8 !important;
    color: #0B1220 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

[data-testid="stJson"] {
    background: #0a0f1a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

.stCaption {
    color: #334155 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}

[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
    filter: grayscale(100%) contrast(1.05) !important;
}

hr { border-color: #1e293b !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0B1220; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

CLASSE_CONFIG = {
    "normal": {
        "icon": "✓", "label": "Normal",
        "color": "#22c55e", "bg": "#052e16", "border": "#14532d",
    },
    "suspected_opacity": {
        "icon": "⚠", "label": "Opacité suspectée",
        "color": "#f59e0b", "bg": "#451a03", "border": "#78350f",
    },
    "uncertain": {
        "icon": "?", "label": "Incertain",
        "color": "#94a3b8", "bg": "#1e293b", "border": "#334155",
    },
}


def render_diagnostic_card(pred: dict) -> None:
    cls_key = pred.get("predicted_class", "uncertain")
    cls = CLASSE_CONFIG.get(cls_key, CLASSE_CONFIG["uncertain"])
    conf_pct = int(pred.get("confidence", 0) * 100)

    st.markdown(f"""
    <div style="background:{cls['bg']};border:1px solid {cls['border']};
        border-radius:10px;padding:16px 20px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <div style="width:40px;height:40px;border-radius:50%;
                background:{cls['color']}22;border:2px solid {cls['color']};
                display:flex;align-items:center;justify-content:center;
                font-size:18px;color:{cls['color']};flex-shrink:0;">{cls['icon']}</div>
            <div>
                <div style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
                    letter-spacing:1px;">DIAGNOSTIC JOUET</div>
                <div style="color:{cls['color']};font-size:20px;font-weight:700;">{cls['label']}</div>
            </div>
        </div>
        <div style="margin-bottom:6px;display:flex;justify-content:space-between;">
            <span style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
                letter-spacing:1px;">CONFIANCE</span>
            <span style="color:{cls['color']};font-size:13px;font-family:'JetBrains Mono',monospace;
                font-weight:700;">{conf_pct}%</span>
        </div>
        <div style="height:4px;background:#0B1220;border-radius:2px;overflow:hidden;">
            <div style="height:100%;width:{conf_pct}%;background:{cls['color']};border-radius:2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section(title: str, content: str | list) -> None:
    if isinstance(content, list):
        body = "".join(
            f'<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;">'
            f'<span style="color:#00B4D8;margin-top:1px;flex-shrink:0;">›</span>'
            f'<span style="color:#cbd5e1;font-size:13px;">{item}</span></div>'
            for item in content
        )
    else:
        body = f'<p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">{content}</p>'

    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
        padding:12px 14px;margin-bottom:8px;">
        <div style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
            letter-spacing:1px;margin-bottom:8px;">{title.upper()}</div>
        {body}
    </div>
    """, unsafe_allow_html=True)


def render_tags(items: list) -> None:
    tags = " ".join(
        f'<span style="background:#1e293b;border:1px solid #334155;border-radius:3px;'
        f'padding:2px 8px;font-size:10px;color:#64748b;'
        f'font-family:\'JetBrains Mono\',monospace;">{t}</span>'
        for t in items
    )
    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
        padding:12px 14px;margin-bottom:8px;">
        <div style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
            letter-spacing:1px;margin-bottom:8px;">LIMITES</div>
        <div style="display:flex;flex-wrap:wrap;gap:5px;">{tags}</div>
    </div>
    """, unsafe_allow_html=True)


def render_tracabilite(pred: dict) -> None:
    st.markdown(f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;background:#0B1220;
        border:1px solid #1e293b;border-radius:6px;padding:8px 12px;margin-bottom:8px;">
        <span style="color:#334155;font-size:10px;font-family:'JetBrains Mono',monospace;">
            MODÈLE <span style="color:#38bdf8;">{pred.get('model_name', '?')}</span>
        </span>
        <span style="color:#1e293b;">·</span>
        <span style="color:#334155;font-size:10px;font-family:'JetBrains Mono',monospace;">
            PROMPT <span style="color:#38bdf8;">{pred.get('prompt_version', '?')}</span>
        </span>
        <span style="color:#1e293b;">·</span>
        <span style="color:#334155;font-size:10px;font-family:'JetBrains Mono',monospace;">
            LATENCE <span style="color:#38bdf8;">{pred.get('latency_ms', '?')} ms</span>
        </span>
        <span style="color:#1e293b;">·</span>
        <span style="color:#334155;font-size:10px;font-family:'JetBrains Mono',monospace;">
            QUALITÉ <span style="color:#38bdf8;">{pred.get('image_quality', '?')}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


#Tête
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        <div style="width:32px;height:32px;border-radius:7px;background:#00B4D820;
            border:1px solid #00B4D840;display:flex;align-items:center;
            justify-content:center;font-size:16px;">⊕</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:#e2e8f0;">ARVI-RX</div>
            <div style="font-size:9px;color:#334155;font-family:'JetBrains Mono',monospace;
                letter-spacing:1px;">ASSISTANT RADIOLOGUE VIRTUEL · PROTOTYPE PÉDAGOGIQUE EFREI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="background:#052e16;border:1px solid #14532d40;border-radius:4px;
        padding:4px 10px;font-size:10px;color:#16a34a;
        font-family:'JetBrains Mono',monospace;letter-spacing:0.5px;
        text-align:center;margin-top:6px;">● MODE JOUET</div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:8px 0 12px;'>", unsafe_allow_html=True)

#Bandeau avertissement
st.markdown("""
<div style="background:#1c1400;border:1px solid #78350f50;border-radius:6px;
    padding:8px 14px;font-size:11px;color:#92400e;
    font-family:'JetBrains Mono',monospace;letter-spacing:0.3px;margin-bottom:16px;">
    ⚠&nbsp; Prototype pédagogique. Non destiné au diagnostic.
    Validation par un professionnel qualifié requise.
</div>
""", unsafe_allow_html=True)


#Cache MedGemma
@st.cache_resource(
    show_spinner="Chargement de MedGemma — au 1er lancement : téléchargement + mise en mémoire des poids…"
)
def _warmup_medgemma():
    from src.inference_medgemma import _get_pipe
    return _get_pipe()


#Sidebar
with st.sidebar:
    st.markdown("""
    <div style="font-size:10px;color:#334155;font-family:'JetBrains Mono',monospace;
        letter-spacing:1px;margin-bottom:16px;padding-bottom:12px;
        border-bottom:1px solid #1e293b;">CONFIGURATION</div>
    """, unsafe_allow_html=True)

    st.markdown("**Moteur d'inférence**")
    engine = st.radio(
        "Backend",
        ("MedGemma (réel)", "Jouet (rapide)"),
        help=(
            "MedGemma : vrai VLM médical (google/medgemma-4b-it), lent, GPU recommandé. "
            "Jouet : validateur de tuyauterie, lit le label dans le nom de fichier."
        ),
    )
    use_medgemma = engine.startswith("MedGemma")

    if use_medgemma:
        try:
            from src import inference_medgemma as mg
        except Exception as exc:
            st.error(f"MedGemma indisponible ({type(exc).__name__}). Bascule sur le mode jouet.")
            use_medgemma = False

    mode_options = list(mg.PROMPTS.keys()) if use_medgemma else ["baseline", "improved"]
    mode = st.selectbox("Mode (prompt)", mode_options)

    if use_medgemma:
        device = mg._resolve_device()
        st.caption(f"Device détecté : **{device}**")
        if device == "cpu":
            st.warning("Pas de GPU : inférence CPU peut prendre **plusieurs minutes** par image.")
        else:
            st.caption("~20 s/image (GPU 4-bit). Voir docs/latence_et_materiel.md.")

    st.markdown("<hr style='margin:16px 0;border-color:#1e293b;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;color:#334155;font-family:'JetBrains Mono',monospace;
        letter-spacing:1px;margin-bottom:10px;">CLASSES DE SORTIE</div>
    """, unsafe_allow_html=True)

    for key, cfg in CLASSE_CONFIG.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 8px;
            border-radius:5px;margin-bottom:4px;
            background:{cfg['bg']};border:1px solid {cfg['border']}30;">
            <span style="color:{cfg['color']};font-size:13px;">{cfg['icon']}</span>
            <span style="color:{cfg['color']};font-size:11px;
                font-family:'JetBrains Mono',monospace;">{key}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0;border-color:#1e293b;'>", unsafe_allow_html=True)
    st.caption("Images test : data/sample_images/")
    st.caption("Schéma DB : sql/schema.sql")
    st.caption("API : uvicorn api.main:app --reload")


#Zone principale
uploaded = st.file_uploader(
    "Déposer une radiographie thoracique frontale",
    type=["png", "jpg", "jpeg"],
    help="Utiliser les images synthétiques dans data/sample_images pour tester le flux (mode jouet).",
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("""
        <div style="font-size:10px;color:#475569;font-family:'JetBrains Mono',monospace;
            letter-spacing:1px;margin-bottom:8px;">RADIOGRAPHIE</div>
        """, unsafe_allow_html=True)
        st.image(Image.open(tmp_path), caption=uploaded.name, use_container_width=True)
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;
            padding:8px 12px;margin-top:8px;font-family:'JetBrains Mono',monospace;
            font-size:10px;color:#334155;">
            Fichier&nbsp;: <span style="color:#64748b;">{uploaded.name}</span><br>
            Taille&nbsp;: <span style="color:#64748b;">{uploaded.size // 1024} Ko</span>
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        try:
            if use_medgemma:
                _warmup_medgemma()
                from src.inference_medgemma import medgemma_predict
                with st.spinner("Analyse par MedGemma en cours…"):
                    raw = medgemma_predict(tmp_path, mode=mode)
            else:
                raw = toy_predict(tmp_path, mode=mode)

            pred = apply_safety_guardrails(raw)

        except Exception as exc:
            st.error(f"Échec de l'inférence : {type(exc).__name__} — {exc}")
            st.stop()

        render_diagnostic_card(pred)
        render_tracabilite(pred)
        render_section("Observations visuelles", pred["visual_evidence"])
        render_section("Justification", pred["justification"])
        render_tags(pred["limitations"])

        if pred.get("guardrail_errors"):
            st.markdown(f"""
            <div style="background:#2d1515;border:1px solid #7f1d1d40;border-radius:6px;
                padding:8px 12px;font-size:11px;color:#f87171;
                font-family:'JetBrains Mono',monospace;margin-bottom:8px;">
                ⚡ Garde-fous déclenchés : {pred['guardrail_errors']}
            </div>
            """, unsafe_allow_html=True)

        with st.expander("{ }  Sortie JSON brute", expanded=False):
            st.json(pred)

else:
    st.markdown("""
    <div style="border:1.5px dashed #1e293b;border-radius:10px;
        padding:60px 24px;text-align:center;background:#0f172a;margin-top:8px;">
        <div style="font-size:36px;opacity:0.2;margin-bottom:12px;">⊕</div>
        <div style="color:#334155;font-family:'JetBrains Mono',monospace;
            font-size:13px;letter-spacing:0.5px;">Déposer une radiographie ci-dessus</div>
        <div style="color:#1e293b;font-size:11px;margin-top:6px;
            font-family:'JetBrains Mono',monospace;">
            ou utiliser les images synthétiques dans data/sample_images/
        </div>
    </div>
    """, unsafe_allow_html=True)