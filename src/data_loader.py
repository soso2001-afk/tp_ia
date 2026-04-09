"""
data_loader.py
==============
Chargement et préparation des données tabulaires et images.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
IMG_SIZE = (128, 128)          # Taille de redimensionnement des images
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Données tabulaires
# ─────────────────────────────────────────────

def load_tabular_data(csv_path: str) -> pd.DataFrame:
    """Charge le fichier CSV et retourne un DataFrame propre."""
    df = pd.read_csv(csv_path)
    return df


def get_tabular_features(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des variables prédictives pour le Modèle 1."""
    return [
        "age",
        "sexe_masculin",
        "presence_nodule",
        "subtilite_nodule",
        "taille_nodule_px",
        "x_nodule_norm",
        "y_nodule_norm",
        "tabagisme_paquets_annee",
        "toux_chronique",
        "dyspnee",
        "douleur_thoracique",
        "perte_poids",
        "spo2",
        "antecedent_familial",
    ]


def prepare_tabular_Xy(df: pd.DataFrame):
    """
    Prépare X et y pour le Modèle 1 (classification 3 classes).

    Returns
    -------
    X : pd.DataFrame  — features
    y : pd.Series     — cible (risque_malignite : 0, 1, 2)
    """
    features = get_tabular_features(df)
    X = df[features].copy()
    y = df["risque_malignite"].copy()
    return X, y


# ─────────────────────────────────────────────
# Données images
# ─────────────────────────────────────────────

def load_single_image(image_path: str, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Charge une image, la convertit en niveaux de gris et la normalise.

    Returns
    -------
    np.ndarray de shape (H, W, 1), valeurs dans [0, 1]
    """
    img = Image.open(image_path).convert("L")      # Niveaux de gris
    img = img.resize(img_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # Normalisation [0,1]
    return arr[..., np.newaxis]                     # (H, W, 1)


def load_image_dataset(df: pd.DataFrame, data_root: str, img_size: tuple = IMG_SIZE):
    """
    Charge toutes les images référencées dans le DataFrame.

    Parameters
    ----------
    df        : DataFrame contenant la colonne 'image_path'
    data_root : dossier racine contenant jsrt_subset/
    img_size  : tuple (H, W)

    Returns
    -------
    images : np.ndarray de shape (N, H, W, 1)
    labels : np.ndarray de shape (N,)   — cancer_image (0 ou 1)
    valid_idx : liste des index du DataFrame correspondants
    """
    images = []
    labels = []
    valid_idx = []

    for idx, row in df.iterrows():
        # Construire le chemin absolu
        rel_path = row["image_path"]            # ex: jsrt_subset/malin/JPCLN001.jpg
        abs_path = os.path.join(data_root, rel_path)

        if not os.path.exists(abs_path):
            # Essai avec sous-dossier redondant jsrt_subset/jsrt_subset/...
            alt_path = os.path.join(data_root, "jsrt_subset", rel_path)
            if os.path.exists(alt_path):
                abs_path = alt_path
            else:
                continue  # Image introuvable, on passe

        try:
            img_arr = load_single_image(abs_path, img_size)
            images.append(img_arr)
            labels.append(int(row["cancer_image"]))
            valid_idx.append(idx)
        except Exception:
            continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels, valid_idx


def load_image_dataset_from_folders(jsrt_root: str, img_size: tuple = IMG_SIZE):
    """
    Charge les images directement depuis les dossiers benin/malin/sain/.
    Utilisé pour la visualisation EDA indépendante du CSV.

    Returns
    -------
    images : np.ndarray (N, H, W, 1)
    labels : list de str ('benin', 'malin', 'sain')
    paths  : list de str
    """
    images, labels, paths = [], [], []
    for cls in ["sain", "benin", "malin"]:
        folder = os.path.join(jsrt_root, cls)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                img = load_single_image(fpath, img_size)
                images.append(img)
                labels.append(cls)
                paths.append(fpath)
            except Exception:
                continue
    return np.array(images, dtype=np.float32), labels, paths
