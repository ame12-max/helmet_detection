# train_cnn.py — TRANSFER LEARNING VERSION (MobileNetV2)
# =========================================================
# WHY THIS REPLACES THE PREVIOUS VERSION:
#   The custom CNN from scratch was getting ~50% accuracy (random chance)
#   because 764 images is too small for a CNN to learn from scratch.
#
#   This version uses MobileNetV2 pretrained on ImageNet — the lower
#   layers already know how to detect edges, textures, shapes, colours.
#   We only train the top classification layers on our helmet data.
#   This reliably achieves 85–95% accuracy even on small datasets.
#
# DATASET LAYOUT:
#   dataset/helmet/
#     with_helmet/      ← rider images WITH helmet
#     without_helmet/   ← rider images WITHOUT helmet
#
# RUN from backend/ folder:
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
from tensorflow.keras import callbacks as keras_callbacks
from tensorflow.keras.applications import MobileNetV2

from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_PATH = '../dataset/helmet'
MODEL_DIR    = 'models'

# MobileNetV2 requires minimum 96×96; 128 gives better accuracy
IMG_SIZE   = 96
BATCH_SIZE = 16    # smaller batch = more stable with small dataset
EPOCHS     = 30

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: str):
    """
    Load images from:
        data_dir/without_helmet/  → label 0
        data_dir/with_helmet/     → label 1
    Returns uint8 arrays (before normalisation) for oversampling.
    """
    X, y = [], []
    class_map = {'without_helmet': 0, 'with_helmet': 1}

    for cls, label in class_map.items():
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            print(f"  ⚠️  Missing: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls:>20}: {len(files)} images")

        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNet expects RGB
            X.append(img)
            y.append(label)

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVERSAMPLE MINORITY CLASS
# ─────────────────────────────────────────────────────────────────────────────
def oversample(X: np.ndarray, y: np.ndarray) -> tuple:
    """Balance classes by augmenting the minority class."""
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [X], [y]

    for cls, count in zip(classes, counts):
        deficit = max_count - count
        if deficit == 0:
            continue

        minority_X = X[y == cls]
        print(f"  Oversampling class {'without_helmet' if cls==0 else 'with_helmet'}: "
              f"+{deficit} synthetic samples")

        indices = np.random.choice(len(minority_X), size=deficit, replace=True)
        extras  = []
        for idx in indices:
            img = minority_X[idx].copy()
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            angle = np.random.uniform(-20, 20)
            M     = cv2.getRotationMatrix2D(
                (IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
            # brightness / contrast jitter
            alpha = np.random.uniform(0.7, 1.3)  # contrast
            beta  = np.random.randint(-30, 30)    # brightness
            img   = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            extras.append(img)

        X_parts.append(np.array(extras, dtype=np.uint8))
        y_parts.append(np.full(deficit, cls, dtype=np.int32))

    X_bal = np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)
    perm  = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD MODEL  (Transfer Learning with MobileNetV2)
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> keras.Model:
    """
    MobileNetV2 base (frozen) + custom classification head.

    Phase 1: Train only the head (base frozen) for fast initial learning.
    Phase 2: Fine-tune the top 30 layers of MobileNetV2 at a low LR.

    Output: Dense(1, sigmoid) — 1.0=helmet, 0.0=no helmet
    """
    # Preprocessing built into the model (no manual normalisation needed)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Light augmentation at the model input
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)

    # MobileNetV2 preprocessing (scales to [-1, 1])
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # Pretrained base — locked during Phase 1
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # Phase 1: frozen
    x = base(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64,  activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = keras.Model(inputs, outputs, name="HelmetMobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def phase1_train(model, X_tr, y_tr, X_val, y_val):
    """Phase 1: Train head only — base is frozen."""
    print("\n── Phase 1: Training classification head (base frozen) ──")
    cb = [
        keras_callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1),
        keras_callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'helmet_cnn_best.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    return model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=15,
        validation_data=(X_val, y_val),
        callbacks=cb, verbose=1
    )


def phase2_finetune(model, base, X_tr, y_tr, X_val, y_val):
    """Phase 2: Unfreeze top layers and fine-tune at low LR."""
    print("\n── Phase 2: Fine-tuning top layers ──")

    # Unfreeze top 30 layers of MobileNetV2
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # very low LR
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    cb = [
        keras_callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-8, verbose=1),
        keras_callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'helmet_cnn_best.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    return model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=cb, verbose=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def threshold_analysis(model, X_val, y_val):
    print("\n📊 Threshold analysis (pick threshold for detect_production.py):")
    print(f"  {'thresh':>8} | {'acc':>5} | {'prec':>5} | {'rec':>5} | "
          f"{'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 65)

    preds = model.predict(X_val, verbose=0).flatten()
    best_thresh = 0.5

    for thresh in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        predicted = (preds >= thresh).astype(int)
        tp = int(np.sum((predicted == 1) & (y_val == 1)))
        tn = int(np.sum((predicted == 0) & (y_val == 0)))
        fp = int(np.sum((predicted == 1) & (y_val == 0)))
        fn = int(np.sum((predicted == 0) & (y_val == 1)))
        acc  = (tp + tn) / len(y_val)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        marker = " ← recommended" if abs(thresh - 0.50) < 0.01 else ""
        print(f"  {thresh:>8.2f} | {acc:>5.2f} | {prec:>5.2f} | {rec:>5.2f} | "
              f"{tp:>5} {tn:>5} {fp:>5} {fn:>5}{marker}")

    # Find best F1 threshold
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.9, 0.01):
        p = (preds >= t).astype(int)
        tp = int(np.sum((p == 1) & (y_val == 1)))
        fp = int(np.sum((p == 1) & (y_val == 0)))
        fn = int(np.sum((p == 0) & (y_val == 1)))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n  ✅ Best F1={best_f1:.3f} at threshold={best_t:.2f}")
    print(f"  → Set self.helmet_thresh = {best_t:.2f} in detect_production.py")
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(h1, h2=None):
    pairs = [('accuracy','val_accuracy','Accuracy'),
             ('loss','val_loss','Loss')]
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))

    for ax, (tk, vk, title) in zip(axes, pairs):
        tr_vals = h1.history[tk]
        vl_vals = h1.history[vk]
        if h2:
            tr_vals += h2.history[tk]
            vl_vals += h2.history[vk]
        ax.plot(tr_vals, label='Train')
        ax.plot(vl_vals, label='Validation')
        if h2:
            ax.axvline(len(h1.history[tk])-1, color='gray',
                       linestyle='--', alpha=0.5, label='Fine-tune start')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(out, dpi=150)
    print(f"\n📈 Curves → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Helmet CNN — Transfer Learning (MobileNetV2)")
    print("=" * 65)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from: {DATASET_PATH}")
    X, y = load_data(DATASET_PATH)

    if len(X) == 0:
        print(f"\n❌ No images found!")
        print(f"   Put images in:")
        print(f"     {DATASET_PATH}/with_helmet/")
        print(f"     {DATASET_PATH}/without_helmet/")
        print(f"\n   Run: python download_dataset.py  for instructions")
        exit(1)

    print(f"\n  Raw dataset:")
    print(f"  Without helmet : {int(np.sum(y==0))}")
    print(f"  With helmet    : {int(np.sum(y==1))}")

    # Warn if dataset too small
    if len(X) < 400:
        print("\n⚠️  WARNING: Only {} images. Transfer learning helps but".format(len(X)))
        print("   results will be much better with 1000+ images per class.")
        print("   See: python download_dataset.py")

    # ── Oversample ───────────────────────────────────────────────────────────
    print("\n[2/5] Balancing via oversampling...")
    X_bal, y_bal = oversample(X, y)
    print(f"  Balanced → {len(X_bal)} total ({int(np.sum(y_bal==0))} each class)")

    # Normalise and split
    X_bal = X_bal.astype(np.float32)  # keep [0,255]; MobileNet preprocess handles scaling
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_bal, y_bal, test_size=0.20, random_state=42, stratify=y_bal)
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    # ── Build ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Building MobileNetV2 model...")
    model, base = build_model()
    trainable   = sum(1 for l in model.layers if l.trainable)
    total       = len(model.layers)
    print(f"  Trainable layers: {trainable}/{total}")
    model.summary(print_fn=lambda s: print(" ", s))

    # ── Train Phase 1 ─────────────────────────────────────────────────────────
    print("\n[4/5] Training...")
    h1 = phase1_train(model, X_tr, y_tr, X_val, y_val)

    # ── Fine-tune Phase 2 ─────────────────────────────────────────────────────
    h2 = phase2_finetune(model, base, X_tr, y_tr, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/5] Final evaluation...")
    res = model.evaluate(X_val, y_val, verbose=0)
    for name, val in zip(model.metrics_names, res):
        print(f"  {name:<15}: {val:.4f}")

    best_thresh = threshold_analysis(model, X_val, y_val)

    # ── Save ──────────────────────────────────────────────────────────────────
    final = os.path.join(MODEL_DIR, 'helmet_cnn.keras')
    model.save(final)
    print(f"\n✅ Final model  → {final}")
    print(f"✅ Best model   → {os.path.join(MODEL_DIR, 'helmet_cnn_best.keras')}")
    print(f"\n💡 Update detect_production.py:")
    print(f"   self.helmet_thresh = {best_thresh:.2f}")

    plot_history(h1, h2)
    print("\nDone! Restart app.py to load the new model.")