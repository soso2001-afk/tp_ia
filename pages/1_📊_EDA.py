"""
pages/1_📊_EDA.py
=================
Partie 0 — Analyse Exploratoire des Données (2 pts)
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_tabular_data, get_tabular_features, load_image_dataset_from_folders
from src.utils import (
    get_csv_path, get_jsrt_root,
    plot_class_distribution, plot_age_distribution,
    plot_correlation_heatmap, plot_images_grid,
)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Partie 0 — Analyse Exploratoire des Données")
st.markdown(
    "Exploration complète du jeu de données patients (données tabulaires) "
    "et visualisation des radios thoraciques."
)

# ─── Chargement ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_tabular_data(get_csv_path())
    return df

df = load_data()
features = get_tabular_features(df)

# ─── Vue générale ─────────────────────────────────────────────────────────────
st.header("1. Vue générale du jeu de données")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nombre de patients", len(df))
c2.metric("Nombre de variables", df.shape[1])
c3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))
c4.metric("Classes de risque", df["risque_malignite"].nunique())

st.subheader("Aperçu des données (5 premières lignes)")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Types des variables")
dtype_df = pd.DataFrame({"Type": df.dtypes, "Non-nuls": df.notnull().sum(),
                          "Nuls": df.isnull().sum(), "Unique": df.nunique()})
st.dataframe(dtype_df, use_container_width=True)

# ─── Statistiques descriptives ────────────────────────────────────────────────
st.header("2. Statistiques descriptives")
st.dataframe(df[features].describe().T.round(3), use_container_width=True)

# ─── Visualisation 1 : Distribution classes ───────────────────────────────────
st.header("3. Visualisations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution risque de malignité (Modèle 1)")
    labels_risk = {0: "Faible", 1: "Intermédiaire", 2: "Élevé"}
    fig1 = plot_class_distribution(
        df, "risque_malignite",
        title="Distribution du risque de malignité",
        labels=labels_risk
    )
    st.pyplot(fig1, use_container_width=True)
    st.markdown(
        """
        **Interprétation** : La classe "Élevé" (risque=2) est majoritaire dans ce dataset clinique,
        reflétant un biais de sélection typique des études sur nodules pulmonaires.
        """
    )

with col2:
    st.subheader("Distribution cancer pulmonaire (Modèle 2)")
    fig2 = plot_class_distribution(
        df, "cancer_image",
        title="Distribution cancer pulmonaire (binaire)",
        labels={0: "Non-cancer", 1: "Cancer"}
    )
    st.pyplot(fig2, use_container_width=True)
    st.markdown(
        """
        **Interprétation** : ~54% des patients présentent un cancer probable.
        Un déséquilibre modéré qui sera géré via la pondération des classes.
        """
    )

# ─── Visualisation 2 : Âge par risque ────────────────────────────────────────
st.subheader("Distribution de l'âge par risque de malignité")
fig3 = plot_age_distribution(df)
st.pyplot(fig3, use_container_width=True)
st.markdown(
    """
    **Interprétation** : Les patients à risque élevé (rouge) ont tendance à être
    légèrement plus âgés, ce qui est cohérent avec la littérature médicale sur
    le cancer pulmonaire (pic d'incidence entre 65-70 ans).
    """
)

# ─── Visualisation 3 : Corrélation ───────────────────────────────────────────
st.subheader("Matrice de corrélation des variables cliniques")
fig4 = plot_correlation_heatmap(df, features + ["risque_malignite", "cancer_image"])
st.pyplot(fig4, use_container_width=True)
st.markdown(
    """
    **Interprétation** : 
    - `tabagisme_paquets_annee` est positivement corrélé au risque de malignité.
    - `spo2` (saturation en oxygène) présente une corrélation négative — 
      une SpO2 basse est associée à un risque accru.
    - Les symptômes (`toux_chronique`, `dyspnee`, `douleur_thoracique`) 
      sont fortement inter-corrélés.
    """
)

# ─── Visualisation 4 : Distribution par sexe ─────────────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.subheader("Répartition par sexe")
    sex_counts = df["sexe_masculin"].value_counts()
    fig5, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sex_counts.values, labels=["Masculin" if i == 1 else "Féminin"
                                       for i in sex_counts.index],
           autopct="%1.1f%%", colors=["#4a90d9", "#e91e8c"])
    ax.set_title("Répartition Hommes / Femmes")
    st.pyplot(fig5, use_container_width=True)

with col4:
    st.subheader("Tabagisme par risque (boxplot)")
    fig6, ax = plt.subplots(figsize=(5, 4))
    df.boxplot(column="tabagisme_paquets_annee", by="risque_malignite",
               ax=ax, grid=False)
    ax.set_xlabel("Risque de malignité (0=Faible, 1=Intermédiaire, 2=Élevé)")
    ax.set_ylabel("Paquets-année")
    ax.set_title("Tabagisme par niveau de risque")
    plt.suptitle("")
    st.pyplot(fig6, use_container_width=True)

# ─── Images ───────────────────────────────────────────────────────────────────
st.header("4. Radios thoraciques représentatives")

@st.cache_data
def load_images_eda():
    jsrt = get_jsrt_root()
    imgs, lbls, paths = load_image_dataset_from_folders(jsrt, img_size=(128, 128))
    return imgs, lbls, paths

with st.spinner("Chargement des images..."):
    imgs, lbls, paths = load_images_eda()

n_per_cls = st.slider("Nombre d'images par classe", 2, 5, 3)
fig_img = plot_images_grid(imgs, lbls, n_per_class=n_per_cls)
st.pyplot(fig_img, use_container_width=True)

st.markdown(
    f"""
    **Dataset images** :
    - 🟢 **Sain** : {lbls.count('sain')} images (pas de nodule)
    - 🟡 **Bénin** : {lbls.count('benin')} images (nodule bénin)
    - 🔴 **Malin** : {lbls.count('malin')} images (nodule malin → cancer)
    
    Les images sont des radiographies thoraciques en niveaux de gris redimensionnées
    en 128×128 pixels et normalisées dans [0, 1].
    """
)

# ─── Conclusion EDA ───────────────────────────────────────────────────────────
st.header("5. Synthèse")
st.success(
    """
    **Points clés de l'EDA :**
    - 185 patients au total, aucune valeur manquante.
    - Variable cible Modèle 1 : `risque_malignite` (0/1/2) — classes déséquilibrées.
    - Variable cible Modèle 2 : `cancer_image` (0/1) — ~54% positifs.
    - Features les plus discriminantes : tabagisme, SpO2, symptômes cliniques.
    - 184 radios thoraciques réparties en 3 classes (sain/bénin/malin).
    """
)
