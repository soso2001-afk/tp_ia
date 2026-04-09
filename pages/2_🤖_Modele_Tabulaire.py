"""
pages/2_🤖_Modele_Tabulaire.py
================================
Partie 1 — Modèle ML Tabulaire (5 pts)
Classification du risque de malignité : 0 (Faible) / 1 (Intermédiaire) / 2 (Élevé)
"""

import os, sys, json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_tabular_data, prepare_tabular_Xy
from src.preprocessing import handle_missing_values, split_tabular, scale_tabular
from src.model1_tabular import (
    cross_validate_models, train_best_model, evaluate_model,
    get_probabilities, save_model, load_model, train_all_models, MODELS
)
from src.utils import (
    get_csv_path, get_models_dir,
    plot_confusion_matrix_fig, plot_feature_importance,
    plot_class_distribution, risk_label
)

st.set_page_config(page_title="Modèle Tabulaire", page_icon="🤖", layout="wide")
st.title("🤖 Partie 1 — Modèle ML Tabulaire")
st.markdown(
    "Classification du **risque de malignité** (3 classes) à partir des données "
    "cliniques du patient. Comparaison de 3 algorithmes."
)

MODELS_DIR = get_models_dir()

# ─── Chargement données ───────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = load_tabular_data(get_csv_path())
    X, y = prepare_tabular_Xy(df)
    X = handle_missing_values(X)
    X_tr, X_te, y_tr, y_te = split_tabular(X, y)
    X_tr_sc, X_te_sc, scaler = scale_tabular(X_tr, X_te,
                                              os.path.join(MODELS_DIR, "scaler.pkl"))
    return df, X, y, X_tr_sc, X_te_sc, y_tr, y_te, scaler, list(X.columns)

df, X, y, X_tr, X_te, y_tr, y_te, scaler, feat_names = load_and_prepare()

# ─── Section 1 : Pipeline ─────────────────────────────────────────────────────
st.header("1. Pipeline de traitement")
st.markdown(
    """
    | Étape | Description |
    |-------|-------------|
    | **Chargement** | Lecture du CSV, sélection des 14 features cliniques |
    | **Nettoyage** | Remplacement des valeurs manquantes par la médiane |
    | **Split** | 80% entraînement / 20% test (stratifié) |
    | **Normalisation** | `StandardScaler` ajusté sur le train uniquement |
    | **Validation** | 5-fold stratifié, métrique F1-macro |
    | **Sélection** | Meilleur modèle selon F1-macro moyen |
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train", len(y_tr))
col2.metric("Test",  len(y_te))
col3.metric("Features", len(feat_names))
col4.metric("Classes", 3)

# ─── Section 2 : Comparaison des modèles ─────────────────────────────────────
st.header("2. Comparaison des 3 algorithmes")

# Charger ou entraîner
model1_path = os.path.join(MODELS_DIR, "model1_tabular.pkl")
meta_path   = os.path.join(MODELS_DIR, "model1_meta.json")

if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    cv_data = meta["cv_results"]
    best_name = meta["best_model"]

    cv_df = pd.DataFrame({
        "Modèle": list(cv_data.keys()),
        "F1-macro moyen (CV)": [v["mean"] for v in cv_data.values()],
        "Écart-type": [v["std"] for v in cv_data.values()],
    }).sort_values("F1-macro moyen (CV)", ascending=False)

    st.dataframe(cv_df.style.highlight_max(subset=["F1-macro moyen (CV)"],
                                           color="#1a5f1a"), use_container_width=True)
    st.success(f"✅ **Meilleur modèle sélectionné : {best_name}**")
else:
    st.warning("⚠️ Modèles non encore entraînés. Cliquez ci-dessous pour lancer l'entraînement.")
    if st.button("🚀 Lancer l'entraînement (peut prendre quelques minutes)"):
        with st.spinner("Entraînement en cours..."):
            cv_results = cross_validate_models(X_tr, y_tr.values)
            best_name, best_model = train_best_model(X_tr, y_tr.values, cv_results)
            save_model(best_model, model1_path)
            meta = {
                "best_model": best_name,
                "features": feat_names,
                "cv_results": {k: {"mean": v["mean"], "std": v["std"]}
                               for k, v in cv_results.items()},
            }
            import json
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        st.success("✅ Entraînement terminé !")
        st.rerun()
    st.stop()

# ─── Section 3 : Évaluation sur test ─────────────────────────────────────────
st.header("3. Évaluation sur le jeu de test")

@st.cache_resource
def get_best_model():
    return load_model(model1_path)

model = get_best_model()
metrics = evaluate_model(model, X_te, y_te.values)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
c2.metric("F1-macro", f"{metrics['f1_macro']:.4f}")
c3.metric("F1-weighted", f"{metrics['f1_weighted']:.4f}")

st.subheader("Rapport de classification")
st.text(metrics["report"])

st.subheader("Matrice de confusion")
fig_cm = plot_confusion_matrix_fig(
    metrics["confusion_matrix"],
    display_labels=["Faible (0)", "Intermédiaire (1)", "Élevé (2)"],
    title=f"Matrice de confusion — {best_name}"
)
st.pyplot(fig_cm, use_container_width=True)

st.markdown(
    """
    **Interprétation** : 
    - La majorité des erreurs concernent la frontière entre classe 1 (Intermédiaire) 
      et classe 2 (Élevé), ce qui est cliniquement attendu.
    - Le F1-macro pondère équitablement les 3 classes et est plus représentatif
      que l'accuracy dans ce contexte de déséquilibre.
    """
)

# ─── Section 4 : Feature Importance ──────────────────────────────────────────
st.header("4. Importance des variables")
fig_imp = plot_feature_importance(model, feat_names)
if fig_imp:
    st.pyplot(fig_imp, use_container_width=True)
    st.markdown(
        """
        **Interprétation** : Le tabagisme (`tabagisme_paquets_annee`) et la 
        saturation en oxygène (`spo2`) sont les features les plus discriminantes,
        confirmant leur rôle clinique dans le diagnostic du cancer pulmonaire.
        """
    )
else:
    st.info("L'importance des features n'est pas disponible pour la Régression Logistique.")

# ─── Section 5 : Probabilités ─────────────────────────────────────────────────
st.header("5. Probabilités de prédiction (20 premiers patients du test)")
probs = get_probabilities(model, X_te)
prob_df = pd.DataFrame(
    probs, columns=["P(Faible)", "P(Intermédiaire)", "P(Élevé)"]
).round(4)
prob_df["Vraie classe"] = [risk_label(int(c)) for c in y_te.values]
prob_df["Classe prédite"] = [risk_label(int(c)) for c in metrics["y_pred"]]
prob_df["Correct"] = prob_df["Vraie classe"] == prob_df["Classe prédite"]
st.dataframe(prob_df.head(20).style.applymap(
    lambda v: "color: green" if v is True else ("color: red" if v is False else ""),
    subset=["Correct"]
), use_container_width=True)
