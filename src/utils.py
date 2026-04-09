"""
utils.py
========
Fonctions utilitaires partagées : visualisations, métriques, chemins.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# ─────────────────────────────────────────────
# Chemins standards
# ─────────────────────────────────────────────

def get_project_root() -> str:
    """Retourne le dossier racine du projet (lung_cancer_app/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_root() -> str:
    """Retourne le dossier contenant les données.
    Priorité : data/ dans le projet (pour Streamlit Cloud),
    puis le dossier parent (pour usage local hors repo).
    """
    project_root = get_project_root()
    local_data = os.path.join(project_root, "data")
    if os.path.isdir(local_data):
        return local_data
    return os.path.dirname(project_root)


def get_csv_path() -> str:
    data_root = get_data_root()
    # In data/ folder
    candidate = os.path.join(data_root, "patients_cancer_poumon.csv")
    if os.path.isfile(candidate):
        return candidate
    # Legacy: parent folder
    return os.path.join(os.path.dirname(get_project_root()), "patients_cancer_poumon.csv")


def get_jsrt_root() -> str:
    data_root = get_data_root()
    # In data/jsrt_subset/jsrt_subset (nested structure)
    nested = os.path.join(data_root, "jsrt_subset", "jsrt_subset")
    if os.path.isdir(nested):
        return nested
    # In data/jsrt_subset
    single = os.path.join(data_root, "jsrt_subset")
    if os.path.isdir(single):
        return single
    # Legacy: parent folder
    return os.path.join(os.path.dirname(get_project_root()), "jsrt_subset", "jsrt_subset")


def get_models_dir() -> str:
    return os.path.join(get_project_root(), "models")


# ─────────────────────────────────────────────
# Visualisations EDA
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, col: str,
                             title: str = None, labels: dict = None):
    """Barplot de la distribution d'une colonne catégorielle."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[col].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#4CAF50", "#FF9800", "#F44336"][:len(counts)])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel(col)
    ax.set_ylabel("Nombre de patients")
    ax.set_title(title or f"Distribution de {col}")
    if labels:
        ax.set_xticklabels([labels.get(int(x), str(x))
                            for x in counts.index], rotation=0)
    plt.tight_layout()
    return fig


def plot_age_distribution(df: pd.DataFrame):
    """Histogramme de l'âge coloré par risque de malignité."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
    labels_txt = {0: "Faible", 1: "Intermédiaire", 2: "Élevé"}
    for risk, grp in df.groupby("risque_malignite"):
        ax.hist(grp["age"], bins=15, alpha=0.6,
                color=colors[risk], label=labels_txt[risk])
    ax.set_xlabel("Âge")
    ax.set_ylabel("Effectif")
    ax.set_title("Distribution de l'âge par risque de malignité")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    """Heatmap de corrélation des features numériques."""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matrice de corrélation des variables")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 10):
    """Barplot horizontal des importances de features (RF / GB)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features les plus importantes")
    plt.tight_layout()
    return fig


def plot_confusion_matrix_fig(cm: np.ndarray, display_labels: list,
                               title: str = "Matrice de confusion"):
    """Retourne une figure matplotlib de la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_learning_curves(history, title: str = "Courbes d'apprentissage"):
    """Retourne une figure avec loss et accuracy (train vs val)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    # Accuracy
    if "accuracy" in history.history:
        axes[1].plot(history.history["accuracy"], label="Train")
        axes[1].plot(history.history["val_accuracy"], label="Validation")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Époque")
        axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_images_grid(images: np.ndarray, labels: list,
                     n_per_class: int = 3) -> plt.Figure:
    """
    Affiche une grille de radios thoraciques par classe.

    Parameters
    ----------
    images : np.ndarray (N, H, W, 1) — valeurs [0,1]
    labels : list de str ('sain', 'benin', 'malin')
    """
    classes = ["sain", "benin", "malin"]
    class_colors = {"sain": "green", "benin": "orange", "malin": "red"}

    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(n_per_class * 3, len(classes) * 3))

    for row_i, cls in enumerate(classes):
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        sample = cls_idx[:n_per_class]

        for col_i in range(n_per_class):
            ax = axes[row_i, col_i]
            if col_i < len(sample):
                ax.imshow(images[sample[col_i], :, :, 0], cmap="gray")
                ax.set_title(cls.capitalize(), color=class_colors[cls], fontsize=9)
            ax.axis("off")

    fig.suptitle("Radios thoraciques représentatives par classe", fontsize=13)
    plt.tight_layout()
    return fig


def plot_roc_curves(models_auc: dict):
    """Barplot comparatif des AUC ROC."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(models_auc.keys())
    aucs = list(models_auc.values())
    bars = ax.bar(names, aucs, color=["#1f77b4", "#ff7f0e"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC ROC")
    ax.set_title("Comparaison AUC — Image Only vs Multimodal")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Utilitaires divers
# ─────────────────────────────────────────────

def risk_label(risk_id: int) -> str:
    return {0: "Faible", 1: "Intermédiaire", 2: "Élevé"}.get(risk_id, "?")


def cancer_label(cancer_id: int) -> str:
    return {0: "Non probable", 1: "Probable"}.get(cancer_id, "?")
