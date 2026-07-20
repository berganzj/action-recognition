"""
POC: binary classifier — Sepultura (214S) vs Neutral.

Prerequisites:
    1. Record sepultura_session.mp4 and neutral_session.mp4 into data/
    2. Run: python scripts/extract_data.py
    3. Run: python scripts/train_poc.py   (on the Windows PC w/ GPU)

Output:
    models/action_model.pkl  — works directly with scripts/predict.py

How it works:
    Copies just the sepultura + neutral folders into data/dataset_poc/,
    then trains a ResNet18 with fastai (same recipe as the notebook).
    This keeps the exported model free of custom code, so predict.py
    can load it without issues.
"""

import os
import shutil
from pathlib import Path

from fastai.vision.all import (
    ImageDataLoaders, Resize, aug_transforms, vision_learner,
    resnet18, accuracy, ClassificationInterpretation,
)
import torch

POC_CLASSES = ["sepultura", "neutral"]

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "dataset"
POC_DIR = ROOT / "data" / "dataset_poc"
MODEL_PATH = ROOT / "models" / "action_model.pkl"
EPOCHS = 5


def stage_poc_dataset():
    """Copy only the POC classes into a separate dataset folder."""
    if POC_DIR.exists():
        shutil.rmtree(POC_DIR)

    for split in ("train", "val"):
        for cls in POC_CLASSES:
            src = DATASET_DIR / split / cls
            dst = POC_DIR / split / cls
            if not src.exists() or not any(src.iterdir()):
                raise SystemExit(
                    f"ERROR: no frames at {src}\n"
                    f"Run scripts/extract_data.py first (needs "
                    f"{cls}_session.mp4 in data/)."
                )
            shutil.copytree(src, dst)
            print(f"  {split}/{cls}: {len(list(dst.iterdir()))} frames")


def train():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nStaging POC dataset (sepultura vs neutral)...")
    stage_poc_dataset()

    dls = ImageDataLoaders.from_folder(
        POC_DIR, train="train", valid="val",
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        bs=32,
        num_workers=0,
    )
    print(f"\nClasses: {dls.vocab}")

    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(EPOCHS)

    # Results
    interp = ClassificationInterpretation.from_learner(learn)
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print(f"  classes: {list(dls.vocab)}")
    print(interp.confusion_matrix())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    learn.export(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Try it: python scripts/predict.py <image_or_video>")


if __name__ == "__main__":
    train()
