"""
Run predictions on new images or video clips using a trained model.

Usage:
    python scripts/predict.py path/to/image_or_video.mp4
"""

import sys
import os
from pathlib import Path
from fastai.vision.all import load_learner, PILImage

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "action_model.pkl")


def predict_image(image_path):
    """Predict action from a single image."""
    learn = load_learner(MODEL_PATH)
    pred, pred_idx, probs = learn.predict(PILImage.create(image_path))
    print(f"Prediction: {pred}")
    print(f"Confidence: {probs[pred_idx]:.2%}")
    print(f"All probabilities: {dict(zip(learn.dls.vocab, [f'{p:.2%}' for p in probs]))}")
    return pred


def predict_video(video_path, every_n_frames=10):
    """Extract frames from a video and predict each one."""
    import cv2

    learn = load_learner(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred, pred_idx, probs = learn.predict(frame_rgb)
            predictions.append(pred)
            print(f"Frame {count}: {pred} ({probs[pred_idx]:.2%})")
        count += 1

    cap.release()

    if predictions:
        from collections import Counter
        counts = Counter(predictions)
        print(f"\nSummary: {dict(counts)}")
        print(f"Most frequent action: {counts.most_common(1)[0][0]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_or_video_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)

    ext = Path(input_path).suffix.lower()
    if ext in (".mp4", ".mkv", ".avi", ".mov"):
        predict_video(input_path)
    else:
        predict_image(input_path)
