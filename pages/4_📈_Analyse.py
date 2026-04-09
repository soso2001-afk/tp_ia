"""
pages/4_📈_Analyse.py
======================
Partie 3 — Analyse et Interprétation (3 pts)
"""

import os, sys, json
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import get_models_dir

st.set_page_config(page_title="Analyse & Interprétation", page_icon="📈", layout="wide")
st.title("📈 Partie 3 — Analyse et Interprétation")
st.markdown("Discussion critique des résultats, limites du système et perspectives d'amélioration.")

MODELS_DIR = get_models_dir()
meta1_path = os.path.join(MODELS_DIR, "model1_meta.json")
meta2_path = os.path.join(MODELS_DIR, "model2_meta.json")

# ─── Q1 : Multimodal vs Image seul ───────────────────────────────────────────
st.header("1. Le modèle multimodal est-il meilleur que le modèle image seul ?")

if os.path.exists(meta2_path):
    with open(meta2_path) as f:
        m2 = json.load(f)
    auc_a = m2["image_only"]["auc"]
    auc_b = m2["multimodal"]["auc"]
    delta = (auc_b - auc_a) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC Image seul", f"{auc_a:.4f}")
    col2.metric("AUC Multimodal", f"{auc_b:.4f}", delta=f"{delta:+.2f}%")
    col3.metric("Amélioration", f"{delta:+.2f}%")

    if delta > 0:
        conclusion = f"""
        ✅ **Le modèle multimodal est supérieur** (AUC +{delta:.2f}%).
        
        L'ajout des probabilités du Modèle 1 améliore la discriminabilité du CNN,
        ce qui confirme que les données cliniques apportent une information complémentaire
        à l'analyse visuelle de la radio thoracique.
        """
    elif delta == 0:
        conclusion = "🟡 Les deux modèles sont équivalents sur ce jeu de données de taille réduite."
    else:
        conclusion = f"""
        ⚠️ Le modèle image seul est légèrement supérieur sur ce run.
        Cela peut s'expliquer par la taille limitée du dataset (184 images) 
        et la variabilité de l'initialisation aléatoire des réseaux de neurones.
        Sur un dataset plus large, la fusion multimodale serait attendue comme bénéfique.
        """
    st.info(conclusion)
else:
    st.warning("Entraînez les modèles pour voir les résultats ici.")

st.markdown(
    """
    **Justification théorique :**
    La littérature médicale (Ardila et al., Nature Medicine 2019 ; Shen et al. 2015)
    montre systématiquement qu'un système de détection de cancer pulmonaire bénéficie
    de la combinaison données cliniques + imagerie. La fusion multimodale permet de
    surmonter les ambiguïtés visuelles (nodules calcifiés vs malins de taille similaire).
    """
)

# ─── Q2 : Apport des données tabulaires ───────────────────────────────────────
st.header("2. En quoi les données tabulaires améliorent-elles la décision ?")

st.markdown(
    """
    Les données tabulaires apportent **quatre types d'information complémentaires** :

    | Catégorie | Variables | Apport |
    |-----------|-----------|--------|
    | **Profil de risque** | Tabagisme, antécédents familiaux | Contextualise la probabilité a priori |
    | **Symptômes** | Toux chronique, dyspnée, douleur thoracique | Indicateurs cliniques directs |
    | **Physiologie** | SpO2, perte de poids | Signes de gravité systémique |
    | **Morphologie** | Taille/position du nodule | Corrèle avec la malignité |

    **Mécanisme de fusion** :  
    Les 3 probabilités de classe du Modèle 1 (P(Faible), P(Inter.), P(Élevé)) forment un
    vecteur de 3 valeurs qui est concaténé avec la représentation image dans le CNN multimodal.
    Ce vecteur encode de façon compacte tout le contexte clinique du patient.
    
    **Limite** : si le Modèle 1 fait une erreur de classification clinique, cette erreur
    se propage au Modèle 2. Une incertitude calibrée serait préférable à une simple probabilité.
    """
)

# ─── Q3 : Limites du dataset ──────────────────────────────────────────────────
st.header("3. Limites du jeu de données")

tabs = st.tabs(["Taille", "Biais", "Labels", "Modalités"])

with tabs[0]:
    st.markdown(
        """
        **Taille insuffisante :**
        - 185 patients tabulaires, 184 images — très insuffisant pour entraîner un CNN robuste.
        - Les réseaux de neurones nécessitent typiquement ≥10 000 images pour généraliser.
        - Risque élevé de **surapprentissage** (overfitting) malgré EarlyStopping et Dropout.
        - **Solution** : Transfer Learning (VGG16, ResNet, etc. pré-entraînés sur CheXpert/NIH-CXR).
        """
    )
with tabs[1]:
    st.markdown(
        """
        **Biais de sélection :**
        - Le dataset JSRT original ne contient que des patients avec nodules détectés.
        - Les cas "sain" (30 images) sont sous-représentés → classe déséquilibrée.
        - Les patients sans cancer mais avec symptômes respiratoires sont absents.
        - **Impact** : le modèle peut avoir un biais vers la détection de nodules visibles.
        """
    )
with tabs[2]:
    st.markdown(
        """
        **Qualité des labels :**
        - Le label `cancer_image` est dérivé de `classe_jsrt_source` (malin vs bénin/sain).
        - `risque_malignite` = variable semi-synthétique construite à partir des données JSRT.
        - L'absence de confirmation histologique pour chaque cas limite la valeur clinique réelle.
        - **Standard** : les annotations devraient être faites par des radiologues experts.
        """
    )
with tabs[3]:
    st.markdown(
        """
        **Modalités manquantes :**
        - PET-scan, TDM thoracique (plus précis que la radio pour les nodules < 1cm) absents.
        - Biomarqueurs sanguins (CEA, CYFRA 21-1) non inclus dans les données tabulaires.
        - Évolution temporelle (comparaison avec radios antérieures) non disponible.
        - **Impact** : en pratique clinique, ces données sont essentielles au diagnostic.
        """
    )

# ─── Q4 : Améliorations ───────────────────────────────────────────────────────
st.header("4. Améliorations proposées pour un contexte médical réel")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Améliorations techniques :**
        
        🔬 **Transfer Learning**  
        Pre-training sur CheXpert (224 226 images) ou NIH CXR-14 (112 120 images)
        avec fine-tuning sur JSRT.
        
        📊 **Dataset augmenté**  
        Data augmentation médicalement pertinente (rotation légère ±10°,
        ajustement de contraste/luminosité). Utilisation de LUNA16 ou LIDC-IDRI.
        
        🎯 **Calibration des probabilités**  
        Platt scaling ou isotonic regression pour que les probabilités prédites
        soient bien calibrées (Hosmer-Lemeshow test).
        
        🔍 **Explicabilité (XAI)**  
        Grad-CAM pour visualiser les régions activées par le CNN (localisation du
        nodule suspect), SHAP pour les features tabulaires.
        """
    )

with col2:
    st.markdown(
        """
        **Améliorations cliniques :**
        
        👨‍⚕️ **Validation externe**  
        Évaluation sur des cohortes indépendantes (multi-centres, multi-équipements).
        
        📏 **Métriques cliniques**  
        Sensibilité ≥ 90% (minimiser les faux négatifs) avant spécificité.
        Calcul du NNR (number needed to read).
        
        🔄 **Apprentissage continu**  
        MLOps : retraining automatique avec accumulation de nouveaux cas annotés.
        Détection de drift de distribution input.
        
        ⚖️ **Aspects réglementaires**  
        Certification CE médical (classe IIa), FDA 510(k) pour les États-Unis.
        Conformité RGPD pour les données patients.
        """
    )

# ─── Synthèse ─────────────────────────────────────────────────────────────────
st.header("5. Synthèse générale")
st.success(
    """
    **Ce TP démontre la faisabilité d'un système d'aide au diagnostic cancer pulmonaire** 
    associant ML tabulaire et CNN, avec une fusion multimodale comme différenciateur clé.
    
    Pour un déploiement clinique réel, les priorités seraient :
    1. **Transfer Learning** sur un large dataset radiologique
    2. **Validation externe** multi-sites avec radiologues référents
    3. **Explicabilité** (Grad-CAM + SHAP) pour la confiance des cliniciens
    4. **Certification médicale** (CE/FDA)
    """
)
