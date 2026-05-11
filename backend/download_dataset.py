# download_dataset.py
# =====================================================================
# Downloads a proper helmet dataset from Kaggle and organises it into
# the correct folder structure for train_cnn.py
#
# STEP 1 — Install Kaggle API (once):
#   pip install kaggle
#
# STEP 2 — Get your Kaggle API key:
#   Go to https://www.kaggle.com → Account → Create API Token
#   This downloads kaggle.json — place it in:
#     Windows: C:\Users\<YourName>\.kaggle\kaggle.json
#
# STEP 3 — Run this script from the backend/ folder:
#   python download_dataset.py
#
# OUTPUT structure:
#   dataset/helmet/
#     with_helmet/      (≥1000 images)
#     without_helmet/   (≥1000 images)
# =====================================================================

import os
import shutil
import zipfile
import random

DATASET_OUT = '../dataset/helmet'


def download_via_kaggle():
    """Download helmet dataset using Kaggle API."""
    print("Downloading dataset from Kaggle...")

    # Best helmet dataset on Kaggle — 7000+ images, well balanced
    os.system(
        'kaggle datasets download -d andrewmvd/helmet-detection '
        '--unzip -p ../dataset/raw_kaggle'
    )

    raw_dir  = '../dataset/raw_kaggle'
    with_dir = os.path.join(DATASET_OUT, 'with_helmet')
    wo_dir   = os.path.join(DATASET_OUT, 'without_helmet')
    os.makedirs(with_dir, exist_ok=True)
    os.makedirs(wo_dir,   exist_ok=True)

    # The kaggle dataset uses subfolders: helmet / no_helmet (or similar)
    # Scan all images and copy to correct target folder
    moved = {0: 0, 1: 0}
    for root, dirs, files in os.walk(raw_dir):
        folder_lower = root.lower()
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src = os.path.join(root, fname)
            # Determine class from folder name
            if any(k in folder_lower for k in ['no_helmet', 'without', 'nohelmet', 'no-helmet']):
                dst = os.path.join(wo_dir, fname)
                moved[0] += 1
            elif any(k in folder_lower for k in ['with_helmet', 'helmet', 'withhelmet']):
                dst = os.path.join(with_dir, fname)
                moved[1] += 1
            else:
                continue
            shutil.copy2(src, dst)

    print(f"Copied {moved[1]} with_helmet images")
    print(f"Copied {moved[0]} without_helmet images")
    return moved[0], moved[1]


def check_current_dataset():
    """Check what's currently in the dataset folder."""
    with_dir = os.path.join(DATASET_OUT, 'with_helmet')
    wo_dir   = os.path.join(DATASET_OUT, 'without_helmet')

    w_count  = len([f for f in os.listdir(with_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(with_dir) else 0
    wo_count = len([f for f in os.listdir(wo_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(wo_dir) else 0

    print(f"\nCurrent dataset:")
    print(f"  with_helmet    : {w_count} images")
    print(f"  without_helmet : {wo_count} images")
    print(f"  Total          : {w_count + wo_count} images")
    return wo_count, w_count


if __name__ == '__main__':
    print("=" * 60)
    print("  Helmet Dataset Setup")
    print("=" * 60)

    wo, w = check_current_dataset()

    if wo < 500 or w < 500:
        print("\n⚠️  Dataset too small for reliable training!")
        print("   You need at least 500 images per class.")
        print("\n📥 Option 1 — Kaggle (recommended, 7000+ images):")
        print("   1. pip install kaggle")
        print("   2. Get kaggle.json from kaggle.com/account")
        print("   3. Place in C:\\Users\\<Name>\\.kaggle\\kaggle.json")
        print("   4. Run: python download_dataset.py --kaggle")
        print("\n📥 Option 2 — Manual download:")
        print("   Download from: https://www.kaggle.com/datasets/andrewmvd/helmet-detection")
        print("   Extract and put images in:")
        print(f"     {os.path.abspath(DATASET_OUT)}/with_helmet/")
        print(f"     {os.path.abspath(DATASET_OUT)}/without_helmet/")
        print("\n📥 Option 3 — Use roboflow (free, no account needed):")
        print("   https://universe.roboflow.com/joseph-nelson/hard-hat-workers")
        print("   Download as 'folder' format, then sort images by class.")
    else:
        print("\n✅ Dataset looks OK for training.")
        print("   Run: python train_cnn.py")

    import sys
    if '--kaggle' in sys.argv:
        download_via_kaggle()
