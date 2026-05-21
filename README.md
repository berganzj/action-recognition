# Action Recognition MVP

Train a model to classify basic actions: punch, kick, throw, jump.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. Record actions in OBS (MP4, 720p, 30fps, NVENC encoder)
2. Place MP4s in `data/` folder (one per action: `punch_session.mp4`, etc.)
3. Run `python scripts/extract_data.py` to extract motion frames
4. Open `notebooks/02_train_model.ipynb` to train with fastai
5. Trained model saved to `models/`

## Project Structure

```
action-recognition/
├── configs/         # Class definitions
├── data/            # Raw recordings and extracted dataset (not in git)
├── models/          # Trained model files (not in git)
├── notebooks/       # Jupyter notebooks for training and evaluation
├── scripts/         # Data extraction and prediction scripts
└── requirements.txt
```

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA support
- OBS Studio for recording
