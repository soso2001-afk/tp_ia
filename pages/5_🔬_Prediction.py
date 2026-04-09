"""
pages/5_🔬_Prediction.py
==========================
Partie 4 — Interface de Prédiction Patient (MLOps) — 4 pts

Interface utilisateur permettant :
  - Saisie des données cliniques d'un patient
  - Upload d'une radio thoracique
  - Prédiction Modèle 1 (risque de malignité)
  - Prédiction Modèle 2 multimodal (cancer probable/non)
  - Affichage du résultat de manière claire et exploitable
"""

import os, sys
import streamlit as st
import numpy as np
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import get_models_dir, get_csv_path, risk_label, cancer_label
from src.data_loader import get_tabular_features, load_tabular_data, load_single_image
from src.preprocessing import load_scaler, preprocess_single_patient
from src.model1_tabular import load_model, get_probabilities
try:
    from src.model2_image import load_keras_model, predict_single_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Prédiction Patient", page_icon="🔬", layout="wide"
)
st.title("🔬 Partie 4 — Interface de Prédiction")
st.markdown(
    """
    Saisissez les données cliniques d'un patient et chargez sa radio thoracique
    pour obtenir une **estimation du risque de cancer pulmonaire**.

    > ⚠️ *Cet outil est à des fins académiques uniquement. Il ne substitue pas un avis médical.*
    """
)

MODELS_DIR = get_models_dir()

# ─── Vérification des modèles ─────────────────────────────────────────────────
m1_path     = os.path.join(MODELS_DIR, "model1_tabular.pkl")
m2b_path    = os.path.join(MODELS_DIR, "model2b_multimodal.keras")
m2a_path    = os.path.join(MODELS_DIR, "model2a_image_only.keras")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

models_ok = all(os.path.exists(p) for p in [m1_path, scaler_path])

if not models_ok:
    st.error(
        "❌ Le Modèle 1 (ML tabulaire) n'est pas encore entraîné.  \n"
        "Lancez d'abord `python train_models.py` dans le dossier `lung_cancer_app/`."
    )
    st.stop()

# ─── Chargement des modèles (cache) ───────────────────────────────────────────
@st.cache_resource
def load_all_models():
    m1 = load_model(m1_path)
    sc = load_scaler(scaler_path)
    m2b = load_keras_model(m2b_path) if os.path.exists(m2b_path) else None
    m2a = load_keras_model(m2a_path) if os.path.exists(m2a_path) else None
    return m1, sc, m2b, m2a

df_ref  = load_tabular_data(get_csv_path())
features = get_tabular_features(df_ref)
model1, scaler, model2b, model2a = load_all_models()

# ─── Formulaire patient ───────────────────────────────────────────────────────
st.header("📋 Données cliniques du patient")

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Profil général")
        age = st.slider("Âge", 30, 90, 60)
        sexe = st.radio("Sexe", ["Féminin", "Masculin"], horizontal=True)
        sexe_m = 1 if sexe == "Masculin" else 0
        tabagisme = st.number_input("Tabagisme (paquets-année)", 0.0, 100.0, 20.0, step=0.5)
        antecedent = st.checkbox("Antécédent familial cancer pulmonaire")

    with c2:
        st.subheader("Nodule pulmonaire")
        presence = st.checkbox("Nodule détecté", value=True)
        subtilite = st.slider("Subtilité du nodule (1=très subtil → 5=évident)", 1, 5, 3,
                              disabled=not presence)
        taille = st.number_input("Taille du nodule (px)", 0, 10, 1, disabled=not presence)
        x_norm = st.slider("Position X normalisée [0-1]", 0.0, 1.0, 0.5, disabled=not presence)
        y_norm = st.slider("Position Y normalisée [0-1]", 0.0, 1.0, 0.5, disabled=not presence)

    with c3:
        st.subheader("Symptômes & Biologie")
        spo2 = st.slider("SpO2 (%)", 85, 100, 95)
        toux = st.checkbox("Toux chronique")
        dyspnee = st.checkbox("Dyspnée")
        douleur = st.checkbox("Douleur thoracique")
        perte_poids = st.checkbox("Perte de poids inexpliquée")

    # ── Radio thoracique ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📷 Radio thoracique")
    uploaded = st.file_uploader(
        "Chargez une radiographie thoracique (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    submitted = st.form_submit_button("🔍 Analyser ce patient", use_container_width=True)

# ─── Traitement et prédiction ─────────────────────────────────────────────────
if submitted:
    # Construire le dict patient
    patient = {
        "age": float(age),
        "sexe_masculin": float(sexe_m),
        "presence_nodule": float(presence),
        "subtilite_nodule": float(subtilite if presence else 0),
        "taille_nodule_px": float(taille if presence else 0),
        "x_nodule_norm": float(x_norm if presence else 0.0),
        "y_nodule_norm": float(y_norm if presence else 0.0),
        "tabagisme_paquets_annee": float(tabagisme),
        "toux_chronique": float(toux),
        "dyspnee": float(dyspnee),
        "douleur_thoracique": float(douleur),
        "perte_poids": float(perte_poids),
        "spo2": float(spo2),
        "antecedent_familial": float(antecedent),
    }

    st.markdown("---")
    st.header("📊 Résultats de l'analyse")

    # ── Modèle 1 ─────────────────────────────────────────────────────────────
    X_patient = preprocess_single_patient(patient, scaler, features)
    probs_m1  = model1.predict_proba(X_patient)[0]      # (3,)
    pred_m1   = int(np.argmax(probs_m1))

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("🤖 Modèle 1 — Risque de malignité")

        risk_colors = {0: "green", 1: "orange", 2: "red"}
        risk_emojis = {0: "🟢", 1: "🟡", 2: "🔴"}
        color = risk_colors[pred_m1]
        emoji = risk_emojis[pred_m1]

        st.markdown(
            f"<h2 style='text-align:center; color:{color}'>"
            f"{emoji} Risque {risk_label(pred_m1).upper()}</h2>",
            unsafe_allow_html=True
        )

        # Barres de probabilité
        import matplotlib.pyplot as plt
        fig_bar, ax = plt.subplots(figsize=(5, 3))
        classes = ["Faible (0)", "Intermédiaire (1)", "Élevé (2)"]
        colors  = ["green", "orange", "red"]
        bars = ax.bar(classes, probs_m1, color=colors, alpha=0.8)
        ax.bar_label(bars, fmt="%.2%", padding=3)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Probabilité")
        ax.set_title("Probabilités par classe de risque")
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)

        st.markdown(
            f"""
            | Classe | Probabilité |
            |--------|-------------|
            | Faible (0)          | **{probs_m1[0]:.1%}** |
            | Intermédiaire (1)   | **{probs_m1[1]:.1%}** |
            | Élevé (2)           | **{probs_m1[2]:.1%}** |
            """
        )

    # ── Modèle 2 ─────────────────────────────────────────────────────────────
    with col_m2:
        st.subheader("🧠 Modèle 2 — Cancer pulmonaire")

        if uploaded is not None:
            # Affichage de la radio
            img_pil = Image.open(uploaded)
            st.image(img_pil, caption="Radio chargée", use_container_width=True)

            # Prétraitement
            img_arr = load_single_image(io.BytesIO(uploaded.getvalue()), img_size=(128, 128))

            # Prédiction
            use_multimodal = model2b is not None
            if use_multimodal:
                label_m2, proba_m2 = predict_single_image(
                    model2b, img_arr, multimodal=True,
                    tab_probs=probs_m1.reshape(1, -1)
                )
                model_used = "Multimodal"
            elif model2a is not None:
                label_m2, proba_m2 = predict_single_image(model2a, img_arr)
                model_used = "Image seul"
            else:
                st.warning("Le Modèle 2 n'est pas encore entraîné.")
                label_m2, proba_m2, model_used = None, None, None

            if label_m2 is not None:
                can_color = "red" if label_m2 == 1 else "green"
                can_emoji = "🔴" if label_m2 == 1 else "🟢"
                can_text  = "CANCER PROBABLE" if label_m2 == 1 else "NON PROBABLE"

                st.markdown(
                    f"<h2 style='text-align:center; color:{can_color}'>"
                    f"{can_emoji} {can_text}</h2>",
                    unsafe_allow_html=True
                )
                st.metric(f"Probabilité de cancer ({model_used})", f"{proba_m2:.1%}")

                # Gauge visuelle
                fig_g, ax_g = plt.subplots(figsize=(5, 1.5))
                ax_g.barh([0], [proba_m2], color="red" if proba_m2 > 0.5 else "green",
                          height=0.5, alpha=0.8)
                ax_g.barh([0], [1 - proba_m2], left=[proba_m2],
                          color="#ddd", height=0.5, alpha=0.5)
                ax_g.axvline(0.5, color="black", linestyle="--", linewidth=1)
                ax_g.set_xlim(0, 1)
                ax_g.set_yticks([])
                ax_g.set_xlabel("Probabilité cancer")
                ax_g.set_title(f"Confiance : {proba_m2:.1%}")
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)
        else:
            st.info("📷 Chargez une radio thoracique pour obtenir la prédiction du Modèle 2.")

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📋 Résumé du bilan")

    risk_txt = risk_label(pred_m1)
    summary_color = "error" if pred_m1 == 2 else ("warning" if pred_m1 == 1 else "success")

    getattr(st, summary_color)(
        f"""
        **Bilan clinique assisté par IA**

        - **Risque de malignité (données cliniques)** : {risk_txt} 
          (P={probs_m1[pred_m1]:.1%})
        - **Cancer pulmonaire (imagerie)** : {"Prédiction disponible" if uploaded else "Radio non fournie"}

        ⚠️ *Ce résultat est produit par un modèle de recherche académique.*  
        *Toute décision clinique doit être prise par un médecin qualifié.*
        """
    )
