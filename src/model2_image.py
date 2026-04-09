"""
model2_image.py
===============
Modèle 2 : Classification binaire cancer pulmonaire (0 / 1)
à partir des radios thoraciques.

Deux architectures :
  - CNN image seul     (build_cnn_image_only)
  - CNN multimodal     (build_cnn_multimodal) : image + prob. Modèle 1

Entraînement, évaluation et sauvegarde sont gérés ici.
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)


# ─────────────────────────────────────────────
# Architectures CNN
# ─────────────────────────────────────────────

def build_cnn_image_only(img_size: tuple = (128, 128), num_classes: int = 1):
    """
    CNN convolutif simple opérant uniquement sur l'image.

    Architecture :
      Conv → BN → Pool → Conv → BN → Pool → Conv → BN → Pool
      → Flatten → Dense → Dropout → Output (sigmoid)

    Parameters
    ----------
    img_size    : (H, W) de l'image (1 canal niveaux de gris)
    num_classes : 1 pour classification binaire (sigmoid)

    Returns
    -------
    model Keras compilé
    """
    inp = Input(shape=(*img_size, 1), name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, out, name="CNN_ImageOnly")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_cnn_multimodal(img_size: tuple = (128, 128),
                          n_tabular_features: int = 3):
    """
    CNN multimodal : fusion de la branche image et des probabilités
    du Modèle 1 (n_tabular_features = 3 probabilités de classe).

    La fusion est réalisée par concaténation (late fusion) :
      ImageBranch → Feature vector
      TabularBranch → Dense
      Concat → Dense → Output

    Parameters
    ----------
    img_size           : (H, W)
    n_tabular_features : taille du vecteur tabulaire (ex : 3 probabilités)

    Returns
    -------
    model Keras compilé (2 inputs : image + prob_vecteur)
    """
    # ── Branche image ─────────────────────────────────────────────────────
    img_input = Input(shape=(*img_size, 1), name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    img_feat = layers.Dense(256, activation="relu", name="img_features")(x)
    img_feat = layers.Dropout(0.4)(img_feat)

    # ── Branche tabulaire (probabilités Modèle 1) ─────────────────────────
    tab_input = Input(shape=(n_tabular_features,), name="tabular_input")
    tab_feat = layers.Dense(32, activation="relu")(tab_input)
    tab_feat = layers.Dense(16, activation="relu")(tab_feat)

    # ── Fusion ────────────────────────────────────────────────────────────
    merged = layers.Concatenate(name="fusion")([img_feat, tab_feat])
    merged = layers.Dense(128, activation="relu")(merged)
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(32, activation="relu")(merged)

    out = layers.Dense(1, activation="sigmoid", name="output")(merged)

    model = models.Model([img_input, tab_input], out, name="CNN_Multimodal")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ─────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────

def get_callbacks(checkpoint_path: str = None):
    """Retourne les callbacks standards pour l'entraînement."""
    cbs = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        cbs.append(ModelCheckpoint(
            checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=0
        ))
    return cbs


def compute_class_weights(y: np.ndarray) -> dict:
    """Calcule les poids de classe pour gérer le déséquilibre."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def train_image_only(model, X_train, y_train, X_val, y_val,
                     epochs: int = 50, batch_size: int = 16,
                     checkpoint_path: str = None):
    """
    Entraîne le CNN image seul.

    Returns
    -------
    history : History Keras
    """
    cw = compute_class_weights(y_train)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(checkpoint_path),
        verbose=1,
    )
    return history


def train_multimodal(model, X_img_train, X_tab_train, y_train,
                     X_img_val, X_tab_val, y_val,
                     epochs: int = 50, batch_size: int = 16,
                     checkpoint_path: str = None):
    """
    Entraîne le CNN multimodal.

    Returns
    -------
    history : History Keras
    """
    cw = compute_class_weights(y_train)
    history = model.fit(
        [X_img_train, X_tab_train], y_train,
        validation_data=([X_img_val, X_tab_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(checkpoint_path),
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────

def evaluate_binary(model, X_test, y_test, multimodal: bool = False,
                    X_tab_test=None, threshold: float = 0.5):
    """
    Évalue un modèle binaire (image-only ou multimodal).

    Returns
    -------
    metrics dict
    """
    if multimodal:
        proba = model.predict([X_test, X_tab_test], verbose=0).ravel()
    else:
        proba = model.predict(X_test, verbose=0).ravel()

    y_pred = (proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score(y_test, proba),
        "report": classification_report(y_test, y_pred,
                                        target_names=["Non-cancer (0)", "Cancer (1)"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "proba": proba,
        "y_pred": y_pred,
    }


# ─────────────────────────────────────────────
# Sauvegarde / Chargement
# ─────────────────────────────────────────────

def save_keras_model(model, path: str):
    """Sauvegarde un modèle Keras au format .keras."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"  ✓ Modèle sauvegardé : {path}")


def load_keras_model(path: str):
    """Charge un modèle Keras depuis le disque."""
    return tf.keras.models.load_model(path)


def predict_single_image(model, image_arr: np.ndarray,
                         multimodal: bool = False,
                         tab_probs: np.ndarray = None,
                         threshold: float = 0.5):
    """
    Prédit sur une image unique (prétraitée).

    Parameters
    ----------
    image_arr : np.ndarray de shape (H, W, 1)
    tab_probs : np.ndarray de shape (3,) — probabilités Modèle 1

    Returns
    -------
    label     : int (0 ou 1)
    proba     : float (probabilité de cancer)
    """
    X = image_arr[np.newaxis, ...]  # (1, H, W, 1)
    if multimodal:
        tab = tab_probs.reshape(1, -1)
        proba = float(model.predict([X, tab], verbose=0)[0, 0])
    else:
        proba = float(model.predict(X, verbose=0)[0, 0])
    label = int(proba >= threshold)
    return label, proba
