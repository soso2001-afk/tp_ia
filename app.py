"""
app.py
======
Point d'entrée de l'application Streamlit.
Détection du cancer pulmonaire — M2 ESIC 2025-2026

Lancement :
  streamlit run app.py
"""

import streamlit as st

# ─── Configuration de la page (doit être le premier appel Streamlit) ──────────
st.set_page_config(
    page_title="Détection Cancer Pulmonaire",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Bannière principale ───────────────────────────────────────────────────────
st.title("🫁 Détection du Cancer Pulmonaire par IA")
st.markdown(
    """
    **M2 ESIC — Intelligence Artificielle, Machine Learning et Deep Learning**  
    *TP Noté 2025-2026*

    ---

    Bienvenue dans l'application de détection du cancer pulmonaire.  
    Ce système exploite deux modèles complémentaires :

    | Modèle | Données | Sortie |
    |--------|---------|--------|
    | **Modèle 1 — ML Tabulaire** | Données cliniques du patient | Risque de malignité (Faible / Intermédiaire / Élevé) |
    | **Modèle 2 — CNN Multimodal** | Radio thoracique + prob. Modèle 1 | Cancer probable (Oui / Non) |

    ---
    """
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.info("📊 **Page 1**\nAnalyse Exploratoire\n(EDA)")
with col2:
    st.info("🤖 **Page 2**\nModèle Tabulaire\n(ML)")
with col3:
    st.info("🧠 **Page 3**\nModèle Image\n(CNN + Fusion)")
with col4:
    st.info("📈 **Page 4**\nAnalyse &\nInterprétation")
with col5:
    st.success("🔬 **Page 5**\nPrédiction\nPatient")

st.markdown("---")
st.markdown("### Navigation")
st.markdown(
    "Utilisez le **menu latéral** (☰) pour naviguer entre les différentes sections de l'application."
)

# ─── Statut des modèles ───────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_models_dir

models_dir = get_models_dir()
m1_ok = os.path.exists(os.path.join(models_dir, "model1_tabular.pkl"))
m2a_ok = os.path.exists(os.path.join(models_dir, "model2a_image_only.keras"))
m2b_ok = os.path.exists(os.path.join(models_dir, "model2b_multimodal.keras"))

st.markdown("### Statut des modèles")
s1, s2, s3 = st.columns(3)
s1.metric("Modèle 1 — ML Tabulaire", "✅ Chargé" if m1_ok else "❌ Non entraîné")
s2.metric("Modèle 2a — Image seul",  "✅ Chargé" if m2a_ok else "❌ Non entraîné")
s3.metric("Modèle 2b — Multimodal",  "✅ Chargé" if m2b_ok else "❌ Non entraîné")

if not (m1_ok and m2a_ok and m2b_ok):
    st.warning(
        "⚠️ Certains modèles ne sont pas encore entraînés.  \n"
        "Exécutez `python train_models.py` depuis le dossier `lung_cancer_app/` "
        "pour entraîner et sauvegarder les modèles."
    )
