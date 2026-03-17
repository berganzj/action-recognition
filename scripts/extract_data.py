"""
Extract motion frames from OBS recordings and organize into a labeled dataset.

Usage:
    1. Place your MP4 recordings in the data/ folder:
       - data/punch_session.mp4
       - data/kick_session.mp4
       - data/throw_session.mp4
       - data/jump_session.mp4
    2. Run: python scripts/extract_data.py
    3. Output: data/dataset/train/<action>/ and data/dataset/val/<action>/
"""

import cv2
import os
import random
import shutil

# === CONFIGURE THESE ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

VIDEOS = {
    "punch": os.path.join(DATA_DIR, "punch_session.mp4"),
    "kick":  os.path.join(DATA_DIR, "kick_session.mp4"),
    "throw": os.path.join(DATA_DIR, "throw_session.mp4"),
    "jump":  os.path.join(DATA_DIR, "jump_session.mp4"),
}

OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")
TARGET_SIZE = (224, 224)
VAL_SPLIT = 0.2
MOTION_THRESHOLD = 25.0   # Minimum motion to count as "doing something"
FRAME_INTERVAL = 5        # Check every 5th frame
# =======================


def has_motion(prev_frame, curr_frame, threshold):
    """Compare two frames — return True if enough movement detected."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    motion_score = gray_diff.mean()
    return motion_score > threshold


def extract_and_organize():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for action, video_path in VIDEOS.items():
        print(f"Processing: {action}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  ERROR: Cannot open {video_path}")
            print(f"  Make sure the file exists in the data/ folder.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Video: {fps:.0f} fps, {total} frames")

        frames = []
        prev_frame = None
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % FRAME_INTERVAL == 0:
                frame_resized = cv2.resize(frame, TARGET_SIZE)

                if prev_frame is not None:
                    if has_motion(prev_frame, frame_resized, MOTION_THRESHOLD):
                        frames.append(frame_resized)

                prev_frame = frame_resized
            count += 1

        cap.release()
        print(f"  Extracted {len(frames)} motion frames (skipped idle)")

        if len(frames) == 0:
            print(f"  WARNING: No motion frames found. Try lowering MOTION_THRESHOLD.")
            continue

        # Shuffle and split into train/val
        random.shuffle(frames)
        split_idx = int(len(frames) * (1 - VAL_SPLIT))
        splits = {"train": frames[:split_idx], "val": frames[split_idx:]}

        for split_name, split_frames in splits.items():
            out_dir = os.path.join(OUTPUT_DIR, split_name, action)
            os.makedirs(out_dir, exist_ok=True)
            for i, f in enumerate(split_frames):
                cv2.imwrite(os.path.join(out_dir, f"{action}_{i:04d}.png"), f)
            print(f"  {split_name}: {len(split_frames)} frames → {out_dir}")

    print(f"\nDone! Dataset ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_and_organize()
