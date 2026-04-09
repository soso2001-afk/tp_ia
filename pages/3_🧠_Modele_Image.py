"""
pages/3_🧠_Modele_Image.py
============================
Partie 2 — CNN Image + Fusion Multimodale (6 pts)
"""

import os, sys, json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    get_models_dir, get_csv_path, get_jsrt_root,
    plot_confusion_matrix_fig, plot_learning_curves, plot_roc_curves, cancer_label
)

st.set_page_config(page_title="Modèle Image", page_icon="🧠", layout="wide")
st.title("🧠 Partie 2 — CNN Image + Fusion Multimodale")
st.markdown(
    "Classification binaire **cancer pulmonaire probable** (0/1)  \n"
    "Deux variantes : CNN image seul **vs** CNN multimodal (image + probabilités Modèle 1)."
)

MODELS_DIR = get_models_dir()
meta2_path = os.path.join(MODELS_DIR, "model2_meta.json")

# ─── Architecture des modèles ─────────────────────────────────────────────────
st.header("1. Architectures CNN")

tab_arch1, tab_arch2 = st.tabs(["Modèle 2a — Image seul", "Modèle 2b — Multimodal"])

with tab_arch1:
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────┐
        │  Input : Radio thoracique (128×128×1)        │
        ├─────────────────────────────────────────────┤
        │  Conv2D(32)  → BN → MaxPool(2×2)            │
        │  Conv2D(64)  → BN → MaxPool(2×2)            │
        │  Conv2D(128) → BN → MaxPool(2×2)            │
        │  GlobalAveragePooling                        │
        │  Dense(256) → Dropout(0.4)                  │
        │  Dense(64)  → Dropout(0.3)                  │
        │  Dense(1, sigmoid) → P(cancer)              │
        └─────────────────────────────────────────────┘
        ```
        """
    )
    st.markdown(
        "**Justification** : Architecture CNN progressivement plus profonde pour "
        "capturer des patterns locaux (nodules) puis globaux (contexte pulmonaire). "
        "GlobalAveragePooling réduit le surapprentissage vs Flatten."
    )

with tab_arch2:
    st.markdown(
        """
        ```
        ┌─────────────────────────────────┐   ┌─────────────────────────┐
        │ Branche Image                   │   │ Branche Tabulaire       │
        │ Input: Radio (128×128×1)        │   │ Input: Prob M1 (3,)     │
        │ Conv→BN→Pool (×3)               │   │ Dense(32) → Dense(16)   │
        │ GlobalAvgPool → Dense(256)      │   │                         │
        └────────────────┬────────────────┘   └──────────┬──────────────┘
                         │                               │
                         └──────── Concatenate ──────────┘
                                       │
                               Dense(128) → Dropout(0.3)
                               Dense(32)
                               Dense(1, sigmoid) → P(cancer)
        ```
        """
    )
    st.markdown(
        """
        **Justification (Fusion Multimodale)** :  
        La fusion tardive (*late fusion*) par concaténation permet à chaque branche
        d'apprendre ses propres représentations avant de les combiner.  
        Les **3 probabilités du Modèle 1** (risque 0/1/2) fournissent un signal clinique
        structuré qui enrichit la décision basée uniquement sur l'image.
        """
    )

# ─── Résultats ────────────────────────────────────────────────────────────────
st.header("2. Résultats et comparaison")

if os.path.exists(meta2_path):
    with open(meta2_path) as f:
        meta2 = json.load(f)

    m_a = meta2["image_only"]
    m_b = meta2["multimodal"]

    st.subheader("Tableau comparatif")
    comp_df = pd.DataFrame({
        "Modèle": ["CNN Image Seul (2a)", "CNN Multimodal (2b)"],
        "Accuracy": [m_a["accuracy"], m_b["accuracy"]],
        "F1-Score": [m_a["f1"], m_b["f1"]],
        "AUC ROC": [m_a["auc"], m_b["auc"]],
    })
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "F1-Score", "AUC ROC"], color="#1a5f1a"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    col1.metric("AUC — Image Seul",  f"{m_a['auc']:.4f}")
    col2.metric("AUC — Multimodal",  f"{m_b['auc']:.4f}",
                delta=f"{(m_b['auc']-m_a['auc'])*100:+.2f}%")

    st.subheader("Comparaison AUC ROC")
    fig_roc = plot_roc_curves({"Image Seul": m_a["auc"], "Multimodal": m_b["auc"]})
    st.pyplot(fig_roc, use_container_width=True)

else:
    st.warning(
        "⚠️ Les modèles ne sont pas encore entraînés.  \n"
        "Exécutez `python train_models.py` depuis le dossier `lung_cancer_app/`."
    )

# ─── Préparation des données ──────────────────────────────────────────────────
st.header("3. Préparation des données image")
st.markdown(
    """
    | Étape | Détail |
    |-------|--------|
    | **Format** | JPEG niveaux de gris (1 canal) |
    | **Redimensionnement** | 128×128 pixels (LANCZOS) |
    | **Normalisation** | Division par 255 → valeurs dans [0, 1] |
    | **Split** | 80% train / 20% test (stratifié sur `cancer_image`) |
    | **Augmentation** | Flip horizontal aléatoire + bruit gaussien (σ=0.02) |
    | **Déséquilibre** | Pondération des classes (`class_weight='balanced'`) |
    """
)

# ─── Courbes d'apprentissage ──────────────────────────────────────────────────
st.header("4. Courbes d'apprentissage")
st.info(
    "Les courbes d'apprentissage sont générées lors de l'entraînement "
    "et sauvegardées automatiquement. Relancez l'entraînement pour les voir ici."
)

# Tenter de charger depuis les fichiers npy si disponibles
hist_a_path = os.path.join(MODELS_DIR, "history_a.npy")
hist_b_path = os.path.join(MODELS_DIR, "history_b.npy")

st.markdown(
    """
    **Paramètres d'entraînement :**
    - Optimiseur : Adam (lr=1e-3)
    - Loss : Binary CrossEntropy
    - Callbacks : EarlyStopping (patience=10), ReduceLROnPlateau (patience=5)
    - Batch size : 16
    - Epochs max : 60 (arrêt anticipé possible)
    """
)

# ─── Interprétation ───────────────────────────────────────────────────────────
st.header("5. Interprétation")
st.markdown(
    """
    **Analyse comparative :**
    
    Le modèle multimodal incorpore les **probabilités de risque clinique** issues du Modèle 1
    (données tabulaires) en complément de l'analyse visuelle de la radio thoracique.
    
    Cette fusion permet de :
    - **Contextualiser** l'interprétation de l'image avec le profil clinique du patient
    - **Réduire les faux négatifs** (cancer non détecté) en cas d'image ambiguë
    - **Améliorer la robustesse** sur les cas limites (nodules de petite taille)
    
    **Point central du TP** : la complémentarité données tabulaires / image est la
    principale valeur ajoutée par rapport à un CNN classique.
    """
)
