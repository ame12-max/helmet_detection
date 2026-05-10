# train_cnn.py  ← FINAL FIXED VERSION
# ======================================
# Fix: replaced LearningRateSchedule with plain float lr=0.001
#      so ReduceLROnPlateau can adjust it at runtime.
#
# DATASET LAYOUT:
#   dataset/helmet/
#     with_helmet/      ← images WITH helmet
#     without_helmet/   ← images WITHOUT helmet
#
# RUN (from backend/ folder):
#   python train_cnn.py
#
# OUTPUT:
#   models/helmet_cnn_best.keras   ← best model (used by app)
#   models/helmet_cnn.keras        ← final model
#   models/training_history.png    ← curves

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks as keras_callbacks   # alias — avoids name clash

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = '../dataset/helmet'
MODEL_DIR    = 'models'
IMG_SIZE     = 64
BATCH_SIZE   = 32
EPOCHS       = 50

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: str):
    """
    Load images from:
        data_dir/without_helmet/  → label 0
        data_dir/with_helmet/     → label 1
    Returns normalised float32 arrays X (N,64,64,3) and y (N,).
    """
    X, y = [], []
    class_map = {'without_helmet': 0, 'with_helmet': 1}

    for cls, label in class_map.items():
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            print(f"  ⚠️  Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls:>20}: {len(files)} images")

        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    if len(X) == 0:
        return np.array([]), np.array([])

    return np.array(X, dtype=np.float32) / 255.0, np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> keras.Model:
    """
    4-block CNN with in-graph augmentation.
    Output: Dense(1, sigmoid) — 1.0=helmet, 0.0=no helmet
    ✅ Uses plain float lr=0.001 so ReduceLROnPlateau can modify it.
    """
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.10),
    ], name="augmentation")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = keras.Model(inputs, outputs, name="HelmetCNN")

    # ✅ FIX: plain float — required for ReduceLROnPlateau to work
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(model, X_tr, y_tr, X_val, y_val, class_weight_dict):
    cb_list = [
        keras_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ✅ Now works because lr is a plain float, not a Schedule object
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras_callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'helmet_cnn_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    return model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history):
    pairs = [
        ('accuracy',  'val_accuracy',  'Accuracy'),
        ('loss',      'val_loss',      'Loss'),
        ('precision', 'val_precision', 'Precision'),
        ('recall',    'val_recall',    'Recall'),
    ]
    # Only plot metrics that actually exist in history
    pairs = [(a, b, t) for a, b, t in pairs if a in history.history]

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (train_k, val_k, title) in zip(axes, pairs):
        ax.plot(history.history[train_k], label='Train')
        ax.plot(history.history[val_k],   label='Validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(out, dpi=150)
    print(f"\n📈 Training curves → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Helmet CNN — Training Script")
    print("=" * 60)

    print(f"\n[1/4] Loading data from: {DATASET_PATH}")
    X, y = load_data(DATASET_PATH)

    if len(X) == 0:
        print("\n❌ No images found!")
        print(f"   Expected folders:")
        print(f"     {DATASET_PATH}/with_helmet/")
        print(f"     {DATASET_PATH}/without_helmet/")
        exit(1)

    print(f"\n  Total images   : {len(X)}")
    print(f"  Without helmet : {int(np.sum(y == 0))}")
    print(f"  With helmet    : {int(np.sum(y == 1))}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train / Val    : {len(X_tr)} / {len(X_val)}")

    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"  Class weights  : {class_weight_dict}")

    print("\n[2/4] Building model...")
    model = build_model()
    model.summary()

    print("\n[3/4] Training...")
    history = train(model, X_tr, y_tr, X_val, y_val, class_weight_dict)

    print("\n[4/4] Evaluating on validation set...")
    res = model.evaluate(X_val, y_val, verbose=0)
    for name, val in zip(model.metrics_names, res):
        print(f"  {name:<15}: {val:.4f}")

    final = os.path.join(MODEL_DIR, 'helmet_cnn.keras')
    model.save(final)
    print(f"\n✅ Final model  → {final}")
    print(f"✅ Best model   → {os.path.join(MODEL_DIR, 'helmet_cnn_best.keras')}")

    plot_history(history)
    print("\nDone! Restart app.py — it will load the new model automatically.")