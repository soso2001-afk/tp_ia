"""
predict.py
==========
Script de prédiction standalone — utilisable en ligne de commande.

Charge les modèles sauvegardés et prédit pour un patient donné
via des arguments ou en mode interactif.

Usage (exemple) :
  python predict.py --image jsrt_subset/jsrt_subset/malin/JPCLN001.jpg \
                    --age 53 --sexe 1 --tabagisme 34.9 --spo2 92 \
                    --toux 1 --dyspnee 1 --douleur 1 --perte_poids 1 \
                    --antecedent 0 --presence_nodule 1 --subtilite 5 \
                    --taille 1 --x_norm 0.7979 --y_norm 0.3379

Mode interactif (sans arguments) :
  python predict.py
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import get_models_dir, risk_label, cancer_label
from src.data_loader import get_tabular_features, load_tabular_data, load_single_image
from src.preprocessing import load_scaler, preprocess_single_patient
from src.model1_tabular import load_model, get_probabilities
try:
    from src.model2_image import load_keras_model, predict_single_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─── Chemins modèles ──────────────────────────────────────────────────────────
MODELS_DIR  = get_models_dir()
M1_PATH     = os.path.join(MODELS_DIR, "model1_tabular.pkl")
M2B_PATH    = os.path.join(MODELS_DIR, "model2b_multimodal.keras")
M2A_PATH    = os.path.join(MODELS_DIR, "model2a_image_only.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def load_models():
    """Charge tous les modèles sauvegardés."""
    if not os.path.exists(M1_PATH):
        raise FileNotFoundError(
            f"Modèle 1 introuvable : {M1_PATH}\n"
            "Lancez d'abord : python train_models.py"
        )
    model1 = load_model(M1_PATH)
    scaler = load_scaler(SCALER_PATH)

    model2 = None
    multimodal = False
    if os.path.exists(M2B_PATH):
        model2 = load_keras_model(M2B_PATH)
        multimodal = True
        print("[✓] Modèle 2b (multimodal) chargé")
    elif os.path.exists(M2A_PATH):
        model2 = load_keras_model(M2A_PATH)
        print("[✓] Modèle 2a (image seul) chargé")
    else:
        print("[!] Modèle 2 non disponible — prédiction tabulaire uniquement")

    print("[✓] Modèle 1 (tabulaire) chargé")
    return model1, scaler, model2, multimodal


def predict(patient_dict: dict, image_path: str = None):
    """
    Effectue la prédiction complète pour un patient.

    Parameters
    ----------
    patient_dict : dictionnaire {feature: valeur}
    image_path   : chemin vers la radio thoracique (optionnel)

    Returns
    -------
    dict contenant les prédictions et probabilités
    """
    from src.data_loader import load_tabular_data, get_tabular_features
    from src.utils import get_csv_path

    df_ref   = load_tabular_data(get_csv_path())
    features = get_tabular_features(df_ref)

    model1, scaler, model2, multimodal = load_models()

    # ── Modèle 1 ─────────────────────────────────────────────────────────────
    X = preprocess_single_patient(patient_dict, scaler, features)
    probs_m1 = model1.predict_proba(X)[0]
    pred_m1  = int(np.argmax(probs_m1))

    result = {
        "modele1": {
            "prediction": pred_m1,
            "label": risk_label(pred_m1),
            "probabilites": {
                "faible (0)": float(probs_m1[0]),
                "intermediaire (1)": float(probs_m1[1]),
                "eleve (2)": float(probs_m1[2]),
            },
        }
    }

    # ── Modèle 2 ─────────────────────────────────────────────────────────────
    if model2 is not None and image_path is not None:
        if not os.path.exists(image_path):
            print(f"[!] Image introuvable : {image_path}")
        else:
            img_arr  = load_single_image(image_path, img_size=(128, 128))
            label_m2, proba_m2 = predict_single_image(
                model2, img_arr,
                multimodal=multimodal,
                tab_probs=probs_m1 if multimodal else None
            )
            result["modele2"] = {
                "prediction": label_m2,
                "label": cancer_label(label_m2),
                "probabilite_cancer": float(proba_m2),
                "type": "multimodal" if multimodal else "image_seul",
            }

    return result


def print_results(result: dict):
    """Affiche les résultats de façon lisible."""
    sep = "─" * 55
    print(f"\n{'#'*55}")
    print("  RÉSULTAT — Détection Cancer Pulmonaire")
    print(f"{'#'*55}")

    m1 = result["modele1"]
    RISK_ICONS = {0: "🟢 FAIBLE", 1: "🟡 INTERMÉDIAIRE", 2: "🔴 ÉLEVÉ"}
    print(f"\n  ┌{sep}┐")
    print(f"  │ MODÈLE 1 — Risque de malignité (données cliniques)   │")
    print(f"  │  → {RISK_ICONS[m1['prediction']].ljust(50)}│")
    for k, v in m1["probabilites"].items():
        print(f"  │    P({k}) = {v:.1%}".ljust(57) + "│")
    print(f"  └{sep}┘")

    if "modele2" in result:
        m2 = result["modele2"]
        icon = "🔴 CANCER PROBABLE" if m2["prediction"] == 1 else "🟢 NON PROBABLE"
        print(f"\n  ┌{sep}┐")
        print(f"  │ MODÈLE 2 — Cancer pulmonaire ({m2['type']:12s})        │")
        print(f"  │  → {icon.ljust(50)}│")
        print(f"  │    Probabilité : {m2['probabilite_cancer']:.1%}".ljust(57) + "│")
        print(f"  └{sep}┘")

    print(f"\n  ⚠️  Usage académique uniquement — avis médical requis.")
    print()


def interactive_mode():
    """Saisie interactive des données patient."""
    print("\n" + "="*55)
    print("  MODE INTERACTIF — Prédiction patient")
    print("="*55)

    patient = {}
    print("\n[Profil général]")
    patient["age"]                    = float(input("  Âge : "))
    patient["sexe_masculin"]          = float(input("  Sexe masculin (0=F, 1=M) : "))
    patient["tabagisme_paquets_annee"]= float(input("  Tabagisme (paquets-année) : "))
    patient["antecedent_familial"]    = float(input("  Antécédent familial (0/1) : "))

    print("\n[Nodule]")
    patient["presence_nodule"]  = float(input("  Présence nodule (0/1) : "))
    patient["subtilite_nodule"] = float(input("  Subtilité nodule (1-5) : "))
    patient["taille_nodule_px"] = float(input("  Taille nodule (px) : "))
    patient["x_nodule_norm"]    = float(input("  Position X normalisée [0-1] : "))
    patient["y_nodule_norm"]    = float(input("  Position Y normalisée [0-1] : "))

    print("\n[Symptômes & biologie]")
    patient["spo2"]             = float(input("  SpO2 (%) : "))
    patient["toux_chronique"]   = float(input("  Toux chronique (0/1) : "))
    patient["dyspnee"]          = float(input("  Dyspnée (0/1) : "))
    patient["douleur_thoracique"] = float(input("  Douleur thoracique (0/1) : "))
    patient["perte_poids"]      = float(input("  Perte de poids (0/1) : "))

    image_path = input("\n[Radio thoracique] Chemin de l'image (Entrée pour ignorer) : ").strip()
    if not image_path:
        image_path = None

    return patient, image_path


# ─── Argparse ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prédiction cancer pulmonaire — TP ESIC 2025-2026"
    )
    parser.add_argument("--image", type=str, default=None, help="Chemin radio thoracique")
    parser.add_argument("--age",              type=float, default=None)
    parser.add_argument("--sexe",             type=float, default=None)
    parser.add_argument("--tabagisme",        type=float, default=None)
    parser.add_argument("--spo2",             type=float, default=None)
    parser.add_argument("--toux",             type=float, default=None)
    parser.add_argument("--dyspnee",          type=float, default=None)
    parser.add_argument("--douleur",          type=float, default=None)
    parser.add_argument("--perte_poids",      type=float, default=None)
    parser.add_argument("--antecedent",       type=float, default=None)
    parser.add_argument("--presence_nodule",  type=float, default=None)
    parser.add_argument("--subtilite",        type=float, default=None)
    parser.add_argument("--taille",           type=float, default=None)
    parser.add_argument("--x_norm",           type=float, default=None)
    parser.add_argument("--y_norm",           type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Déterminer si mode interactif
    if args.age is None:
        patient_dict, image_path = interactive_mode()
    else:
        patient_dict = {
            "age":                     args.age,
            "sexe_masculin":           args.sexe or 0,
            "presence_nodule":         args.presence_nodule or 0,
            "subtilite_nodule":        args.subtilite or 0,
            "taille_nodule_px":        args.taille or 0,
            "x_nodule_norm":           args.x_norm or 0.0,
            "y_nodule_norm":           args.y_norm or 0.0,
            "tabagisme_paquets_annee": args.tabagisme or 0,
            "toux_chronique":          args.toux or 0,
            "dyspnee":                 args.dyspnee or 0,
            "douleur_thoracique":      args.douleur or 0,
            "perte_poids":             args.perte_poids or 0,
            "spo2":                    args.spo2 or 95,
            "antecedent_familial":     args.antecedent or 0,
        }
        image_path = args.image

    result = predict(patient_dict, image_path)
    print_results(result)
