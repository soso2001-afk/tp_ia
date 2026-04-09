"""
model1_tabular.py
=================
Modèle 1 : Classification du risque de malignité (0, 1, 2) à partir des
données tabulaires patients.

Trois algorithmes sont comparés :
  - Régression Logistique (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost-like via sklearn)

Le meilleur modèle est sélectionné sur la base du F1-score macro moyen
en validation croisée, puis sauvegardé.
"""

import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────
# Définition des modèles candidats
# ─────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, multi_class="multinomial", random_state=42, C=1.0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
}


# ─────────────────────────────────────────────
# Entraînement et sélection
# ─────────────────────────────────────────────

def cross_validate_models(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5):
    """
    Évalue chaque modèle par validation croisée stratifiée.

    Returns
    -------
    dict : {nom_modele: {"mean": float, "std": float, "scores": np.ndarray}}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, model in MODELS.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="f1_macro", n_jobs=-1)
        results[name] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores,
        }
        print(f"  [{name}] F1-macro CV: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def train_best_model(X_train: np.ndarray, y_train: np.ndarray,
                     cv_results: dict):
    """
    Entraîne le meilleur modèle (sélectionné par F1-macro CV) sur tout X_train.

    Returns
    -------
    best_name  : str
    best_model : modèle scikit-learn entraîné
    """
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    best_model = MODELS[best_name]
    best_model.fit(X_train, y_train)
    print(f"\n  ✓ Meilleur modèle : {best_name} "
          f"(F1={cv_results[best_name]['mean']:.4f})")
    return best_name, best_model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Évalue le modèle sur le jeu de test.

    Returns
    -------
    metrics dict : accuracy, f1_macro, f1_weighted, report, confusion_matrix
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred,
                                        target_names=["Faible (0)",
                                                      "Intermédiaire (1)",
                                                      "Élevé (2)"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def get_probabilities(model, X: np.ndarray) -> np.ndarray:
    """
    Retourne les probabilités prédites par classe.

    Returns
    -------
    np.ndarray de shape (N, 3)  — une colonne par classe de risque
    """
    return model.predict_proba(X)


# ─────────────────────────────────────────────
# Sauvegarde / Chargement
# ─────────────────────────────────────────────

def save_model(model, path: str):
    """Sauvegarde le modèle avec joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  ✓ Modèle sauvegardé : {path}")


def load_model(path: str):
    """Charge un modèle sauvegardé."""
    return joblib.load(path)


def train_all_models(X_train, y_train):
    """
    Entraîne tous les modèles candidats et retourne un dict
    {nom: modèle entraîné}.

    Utile pour l'affichage comparatif dans Streamlit.
    """
    trained = {}
    for name, model in MODELS.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        trained[name] = m
    return trained
