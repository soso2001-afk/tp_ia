# 🫁 Détection du Cancer Pulmonaire par IA

**M2 ESIC — Intelligence Artificielle, Machine Learning et Deep Learning**  
**TP Noté 2025-2026 — Date de remise : 12/04/2026**

Application complète de détection du cancer pulmonaire combinant :
- **Modèle 1** : Classification ML tabulaire (risque de malignité 3 classes)
- **Modèle 2** : CNN multimodal image + données cliniques (cancer probable/non)
- **Interface** : Application Streamlit multipage
- **Déploiement** : Streamlit Cloud (ou local)

---

## Structure du projet

```
lung_cancer_app/
├── app.py                      ← Point d'entrée Streamlit
├── train_models.py             ← Script d'entraînement (à exécuter 1 fois)
├── predict.py                  ← Script de prédiction CLI standalone
├── requirements.txt            ← Dépendances Python
│
├── pages/                      ← Pages Streamlit
│   ├── 1_📊_EDA.py             ← Partie 0 : Analyse exploratoire
│   ├── 2_🤖_Modele_Tabulaire.py ← Partie 1 : ML tabulaire
│   ├── 3_🧠_Modele_Image.py    ← Partie 2 : CNN + fusion
│   ├── 4_📈_Analyse.py         ← Partie 3 : Analyse & interprétation
│   └── 5_🔬_Prediction.py      ← Partie 4 : Interface prédiction
│
├── src/                        ← Modules Python
│   ├── __init__.py
│   ├── data_loader.py          ← Chargement données (CSV + images)
│   ├── preprocessing.py        ← Prétraitement et split
│   ├── model1_tabular.py       ← Modèle ML tabulaire (3 algos)
│   ├── model2_image.py         ← CNN image seul + multimodal
│   └── utils.py                ← Visualisations + utilitaires
│
├── models/                     ← Modèles sauvegardés (générés par train_models.py)
│   ├── model1_tabular.pkl
│   ├── scaler.pkl
│   ├── model2a_image_only.keras
│   ├── model2b_multimodal.keras
│   ├── model1_meta.json
│   └── model2_meta.json
│
├── docs/
│   └── rapport.md              ← Rapport complet du TP
│
└── .streamlit/
    └── config.toml             ← Configuration thème Streamlit
```

**Données** (dans le dossier parent `tp/`) :
```
tp/
├── patients_cancer_poumon.csv
└── jsrt_subset/jsrt_subset/
    ├── benin/     (54 images)
    ├── malin/     (100 images)
    └── sain/      (30 images)
```

---

## Installation

### Prérequis

- Python 3.10 ou 3.11
- pip

### Installation des dépendances

```bash
cd lung_cancer_app/
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Entraîner les modèles

Cette étape est **obligatoire** avant de lancer l'application.

```bash
cd lung_cancer_app/
python train_models.py
```

**Durée estimée** : 5-15 minutes (selon CPU/GPU)

Le script :
1. Charge et prétraite les données tabulaires
2. Compare 3 algorithmes ML (LogReg, Random Forest, Gradient Boosting)
3. Sélectionne le meilleur modèle par validation croisée
4. Entraîne le CNN image seul
5. Entraîne le CNN multimodal
6. Sauvegarde tous les modèles dans `models/`

### Étape 2 — Lancer l'application Streamlit

```bash
cd lung_cancer_app/
streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**

### Prédiction via ligne de commande (optionnel)

```bash
cd lung_cancer_app/

# Mode interactif
python predict.py

# Mode avec arguments
python predict.py \
  --image ../jsrt_subset/jsrt_subset/malin/JPCLN001.jpg \
  --age 53 --sexe 1 --tabagisme 34.9 --spo2 92 \
  --toux 1 --dyspnee 1 --douleur 1 --perte_poids 1 \
  --antecedent 0 --presence_nodule 1 --subtilite 5 \
  --taille 1 --x_norm 0.7979 --y_norm 0.3379
```

---

## Déploiement sur Streamlit Cloud

### Configuration préalable

1. Créer un compte sur [share.streamlit.io](https://share.streamlit.io) (gratuit)
2. Pousser le projet sur GitHub :

```bash
# Depuis le dossier lung_cancer_app/
git init
git add .
git commit -m "Initial commit — TP Cancer Pulmonaire"
git remote add origin https://github.com/[username]/[repo].git
git push -u origin main
```

3. **Inclure les modèles** dans le commit (si < 100 MB par fichier) :
   - Les fichiers `.pkl` et `.keras` du dossier `models/` doivent être présents dans le repo
   - Supprimer `.gitignore` si les modèles sont ignorés

### Déploiement

1. Se connecter sur [share.streamlit.io](https://share.streamlit.io)
2. Cliquer **"New app"**
3. Sélectionner votre repository GitHub
4. **Main file path** : `lung_cancer_app/app.py` (ou `app.py` si le repo est `lung_cancer_app/`)
5. Cliquer **"Deploy!"**

L'application sera accessible sur : `https://[username]-[repo]-app-[hash].streamlit.app`

### Variables d'environnement (si nécessaire)

Si les modèles sont stockés externes (S3/Azure), configurer dans Streamlit Cloud :  
*Settings → Secrets* → ajouter les credentials

---

## Déploiement alternatif — Render (gratuit)

1. Créer un compte sur [render.com](https://render.com)
2. "New Web Service" → connecter le repo GitHub
3. **Build Command** : `pip install -r requirements.txt`
4. **Start Command** : `streamlit run app.py --server.port $PORT`
5. Plan : **Free** (suffisant pour le TP)

---

## Docker (optionnel)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.headless=true", \
            "--server.address=0.0.0.0"]
```

```bash
# Build
docker build -t cancer-detection .

# Run
docker run -p 8501:8501 cancer-detection
```

---

## Description des modèles

### Modèle 1 — Classification tabulaire

- **Objectif** : Prédire le risque de malignité (0, 1, 2) à partir de 14 features cliniques
- **Algorithme** : Random Forest (meilleur CV F1-macro)
- **Features** : âge, sexe, tabagisme, SpO2, symptômes, nodule (taille/position/subtilité)
- **Sortie** : vecteur de 3 probabilités → input du Modèle 2

### Modèle 2 — CNN Multimodal

- **Objectif** : Prédire cancer pulmonaire probable (0/1)
- **Version a** (image seul) : CNN 3 couches conv → GlobalAvgPool → Dense
- **Version b** (multimodal) : CNN + branche Dense sur prob. Modèle 1 → Fusion (Concatenate)
- **Entraînement** : EarlyStopping, ReduceLROnPlateau, class_weight balanced

---

## Barème couvert

| Partie | Points | Éléments implémentés |
|--------|--------|---------------------|
| Partie 0 — EDA | 2 | ✅ Stats descriptives, 5+ visualisations, radios par classe |
| Partie 1 — ML Tabulaire | 5 | ✅ 3 algos, pipeline, normalisation, CV, métriques, feature importance |
| Partie 2 — CNN + Fusion | 6 | ✅ CNN image seul + multimodal, comparaison, courbes apprentissage |
| Partie 3 — Analyse | 3 | ✅ Discussion complète, limites, améliorations |
| Partie 4 — MLOps | 4 | ✅ Interface Streamlit, prédiction standalone, déploiement cloud |
| **Total** | **20** | |

---

## Rapport

Le rapport complet est disponible dans [docs/rapport.md](docs/rapport.md).

---

## Auteur

TP Noté — M2 ESIC  
Détection du Cancer Pulmonaire par Machine Learning, Deep Learning et MLOps  
Année académique 2025-2026

> ⚠️ *Ce projet est à des fins académiques uniquement. Il ne constitue pas un dispositif médical certifié.*
