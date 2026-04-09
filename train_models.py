"""
train_models.py
===============
Script d'entraînement complet — à exécuter UNE FOIS avant de lancer l'app.

Étapes :
  1. Chargement et prétraitement des données
  2. Modèle 1 : comparaison de 3 algorithmes, sélection et sauvegarde
  3. Modèle 2a : CNN image seul, entraînement et sauvegarde
  4. Modèle 2b : CNN multimodal (image + prob. Modèle 1), entraînement et sauvegarde
  5. Résumé des performances

Usage :
  cd lung_cancer_app/
  python train_models.py
"""

import os
import sys
import json
import numpy as np

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_tabular_data,
    get_tabular_features,
    prepare_tabular_Xy,
    load_image_dataset,
)
from src.preprocessing import (
    handle_missing_values,
    split_tabular,
    scale_tabular,
    split_image_data,
)
from src.model1_tabular import (
    cross_validate_models,
    train_best_model,
    evaluate_model,
    get_probabilities,
    save_model,
    train_all_models,
)
try:
    from src.model2_image import (
        build_cnn_image_only,
        build_cnn_multimodal,
        train_image_only,
        train_multimodal,
        evaluate_binary,
        save_keras_model,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from src.utils import get_csv_path, get_jsrt_root, get_models_dir


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = (128, 128)
EPOCHS      = 60          # EarlyStopping arrêtera avant si nécessaire
BATCH_SIZE  = 16
TEST_RATIO  = 0.20
MODELS_DIR  = get_models_dir()

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 1 — Modèle tabulaire
# ─────────────────────────────────────────────────────────────────────────────

def train_model1():
    print("\n" + "="*60)
    print("  PARTIE 1 — Modèle ML Tabulaire")
    print("="*60)

    # Chargement
    csv_path = get_csv_path()
    print(f"\n[1/5] Chargement des données : {csv_path}")
    df = load_tabular_data(csv_path)
    print(f"  → {len(df)} patients, {df.shape[1]} variables")

    # Préparation
    print("[2/5] Préparation X, y")
    X, y = prepare_tabular_Xy(df)
    X = handle_missing_values(X)
    print(f"  → X shape : {X.shape}, classes : {sorted(y.unique())}")

    # Split
    print("[3/5] Séparation train/test (80/20)")
    X_train, X_test, y_train, y_test = split_tabular(X, y, TEST_RATIO)
    print(f"  → Train : {len(X_train)}, Test : {len(X_test)}")

    # Normalisation
    print("[4/5] Normalisation StandardScaler")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    X_train_sc, X_test_sc, scaler = scale_tabular(X_train, X_test, scaler_path)

    # Validation croisée
    print("[5/5] Validation croisée (5-fold, F1-macro)")
    cv_results = cross_validate_models(X_train_sc, y_train.values)

    # Sélection + entraînement final
    best_name, best_model = train_best_model(X_train_sc, y_train.values, cv_results)

    # Évaluation sur test
    metrics = evaluate_model(best_model, X_test_sc, y_test.values)
    print(f"\n  Test Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Test F1-macro  : {metrics['f1_macro']:.4f}")
    print(f"  Test F1-weight : {metrics['f1_weighted']:.4f}")
    print("\n" + metrics["report"])

    # Sauvegarde
    model1_path = os.path.join(MODELS_DIR, "model1_tabular.pkl")
    save_model(best_model, model1_path)

    # Sauvegarde des métadonnées
    meta = {
        "best_model": best_name,
        "features": get_tabular_features(df),
        "cv_results": {k: {"mean": v["mean"], "std": v["std"]}
                       for k, v in cv_results.items()},
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        },
    }
    with open(os.path.join(MODELS_DIR, "model1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return df, best_model, scaler, X_train_sc, X_test_sc, y_train, y_test, cv_results


# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 2 — Modèles image
# ─────────────────────────────────────────────────────────────────────────────

def train_model2(df, model1, scaler, y_train_tab, y_test_tab):
    print("\n" + "="*60)
    print("  PARTIE 2 — CNN Image (image seul + multimodal)")
    print("="*60)

    jsrt_root = get_jsrt_root()
    data_root = os.path.dirname(os.path.dirname(jsrt_root))  # tp/

    # Chargement des images
    print("\n[1/5] Chargement des images")
    from src.preprocessing import handle_missing_values
    X_tab_full, _ = prepare_tabular_Xy_filtered(df)
    images, labels_img, valid_idx = load_image_dataset(df, data_root, IMG_SIZE)
    print(f"  → {len(images)} images chargées, cancer=1 : {labels_img.sum()}")

    # Probabilités Modèle 1 pour TOUTES les images
    print("[2/5] Génération des probabilités Modèle 1")
    from src.data_loader import get_tabular_features
    features = get_tabular_features(df)
    X_tab_for_images = df.loc[valid_idx, features].copy()
    X_tab_for_images = handle_missing_values(X_tab_for_images)
    X_tab_scaled = scaler.transform(X_tab_for_images)
    probs_m1 = model1.predict_proba(X_tab_scaled)   # (N, 3)

    # Split images
    print("[3/5] Séparation train/test images (80/20)")
    from src.preprocessing import split_image_data
    (X_img_tr, X_img_te,
     y_tr, y_te,
     idx_tr, idx_te) = split_image_data(images, labels_img, valid_idx, df, TEST_RATIO)

    # Probabilités train/test
    probs_all = {vi: probs_m1[i] for i, vi in enumerate(valid_idx)}
    X_tab_tr = np.array([probs_all[vi] for vi in idx_tr], dtype=np.float32)
    X_tab_te = np.array([probs_all[vi] for vi in idx_te], dtype=np.float32)
    print(f"  → Train : {len(X_img_tr)}, Test : {len(X_img_te)}")

    # ── Modèle 2a : image seul ────────────────────────────────────────────
    print("\n[4/5] Entraînement CNN Image Seul")
    model2a = build_cnn_image_only(IMG_SIZE)
    model2a.summary(line_length=80)

    ckpt_a = os.path.join(MODELS_DIR, "model2a_best.keras")
    history_a = train_image_only(
        model2a, X_img_tr, y_tr, X_img_te, y_te,
        epochs=EPOCHS, batch_size=BATCH_SIZE, checkpoint_path=ckpt_a
    )
    metrics_a = evaluate_binary(model2a, X_img_te, y_te)
    print(f"  Image Only  → Accuracy={metrics_a['accuracy']:.4f}  "
          f"F1={metrics_a['f1']:.4f}  AUC={metrics_a['auc']:.4f}")
    save_keras_model(model2a, os.path.join(MODELS_DIR, "model2a_image_only.keras"))

    # ── Modèle 2b : multimodal ────────────────────────────────────────────
    print("\n[5/5] Entraînement CNN Multimodal")
    model2b = build_cnn_multimodal(IMG_SIZE, n_tabular_features=3)
    model2b.summary(line_length=80)

    ckpt_b = os.path.join(MODELS_DIR, "model2b_best.keras")
    history_b = train_multimodal(
        model2b,
        X_img_tr, X_tab_tr, y_tr,
        X_img_te, X_tab_te, y_te,
        epochs=EPOCHS, batch_size=BATCH_SIZE, checkpoint_path=ckpt_b
    )
    metrics_b = evaluate_binary(model2b, X_img_te, y_te,
                                 multimodal=True, X_tab_test=X_tab_te)
    print(f"  Multimodal  → Accuracy={metrics_b['accuracy']:.4f}  "
          f"F1={metrics_b['f1']:.4f}  AUC={metrics_b['auc']:.4f}")
    save_keras_model(model2b, os.path.join(MODELS_DIR, "model2b_multimodal.keras"))

    # Sauvegarde métriques
    meta2 = {
        "image_only": {
            "accuracy": metrics_a["accuracy"],
            "f1": metrics_a["f1"],
            "auc": metrics_a["auc"],
        },
        "multimodal": {
            "accuracy": metrics_b["accuracy"],
            "f1": metrics_b["f1"],
            "auc": metrics_b["auc"],
        },
    }
    with open(os.path.join(MODELS_DIR, "model2_meta.json"), "w") as f:
        json.dump(meta2, f, indent=2)

    return history_a, history_b, metrics_a, metrics_b


def prepare_tabular_Xy_filtered(df):
    """Version locale pour éviter les imports circulaires."""
    from src.data_loader import get_tabular_features
    features = get_tabular_features(df)
    X = df[features].copy()
    y = df["risque_malignite"].copy()
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  ENTRAÎNEMENT — Détection Cancer Pulmonaire")
    print("  M2 ESIC — TP Noté 2025-2026")
    print("█"*60)

    # Partie 1
    df, model1, scaler, X_tr, X_te, y_tr, y_te, cv_res = train_model1()

    # Partie 2
    h_a, h_b, m_a, m_b = train_model2(df, model1, scaler, y_tr, y_te)

    # Résumé final
    print("\n" + "="*60)
    print("  RÉSUMÉ FINAL")
    print("="*60)
    print(f"  Modèle 1 (tabulaire)   sauvegardé dans models/")
    print(f"  Modèle 2a (img seul)   — AUC : {m_a['auc']:.4f}")
    print(f"  Modèle 2b (multimodal) — AUC : {m_b['auc']:.4f}")
    improvement = (m_b['auc'] - m_a['auc']) * 100
    print(f"  Amélioration AUC (multimodal vs image-only) : "
          f"{improvement:+.2f}%")
    print("\n  ✓ Entraînement terminé. Lancez l'app avec :")
    print("    streamlit run app.py")
