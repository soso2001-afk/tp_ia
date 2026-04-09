"""
preprocessing.py
================
Prétraitement des données tabulaires et des images.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Tabulaire
# ─────────────────────────────────────────────

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs manquantes par la médiane de chaque colonne."""
    return X.fillna(X.median(numeric_only=True))


def split_tabular(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Divise X et y en ensembles d'entraînement et de test.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=RANDOM_STATE, stratify=y)


def scale_tabular(X_train: pd.DataFrame, X_test: pd.DataFrame,
                  scaler_path: str = None):
    """
    Normalise X_train et X_test avec StandardScaler.

    Parameters
    ----------
    scaler_path : si fourni, sauvegarde le scaler à cet emplacement

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def load_scaler(scaler_path: str) -> StandardScaler:
    """Charge un scaler sauvegardé."""
    return joblib.load(scaler_path)


def preprocess_single_patient(patient_dict: dict, scaler: StandardScaler,
                               feature_order: list) -> np.ndarray:
    """
    Prétraite un patient unique pour la prédiction en production.

    Parameters
    ----------
    patient_dict : {feature_name: value}
    scaler       : StandardScaler ajusté sur train
    feature_order: liste ordonnée des features

    Returns
    -------
    np.ndarray de shape (1, n_features)
    """
    row = [patient_dict[f] for f in feature_order]
    X = np.array(row, dtype=np.float32).reshape(1, -1)
    return scaler.transform(X)


# ─────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────

def split_image_data(images: np.ndarray, labels: np.ndarray,
                     valid_idx: list, df,
                     test_size: float = 0.2):
    """
    Divise le dataset image en train/test, en conservant les index du DataFrame
    pour récupérer les probabilités du Modèle 1.

    Returns
    -------
    X_img_train, X_img_test,
    y_train, y_test,
    idx_train, idx_test
    """
    all_idx = list(range(len(images)))
    tr, te = train_test_split(all_idx, test_size=test_size,
                              random_state=RANDOM_STATE, stratify=labels)
    return (images[tr], images[te],
            labels[tr], labels[te],
            [valid_idx[i] for i in tr],
            [valid_idx[i] for i in te])


def augment_image_batch(images: np.ndarray) -> np.ndarray:
    """
    Augmentation légère : flip horizontal aléatoire + bruit gaussien.
    Appliqué uniquement au train.

    Returns
    -------
    np.ndarray de même shape
    """
    augmented = images.copy()
    for i in range(len(augmented)):
        # Flip horizontal aléatoire
        if np.random.rand() > 0.5:
            augmented[i] = augmented[i, :, ::-1, :]
        # Bruit gaussien faible
        noise = np.random.normal(0, 0.02, augmented[i].shape).astype(np.float32)
        augmented[i] = np.clip(augmented[i] + noise, 0.0, 1.0)
    return augmented
