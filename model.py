import os
import numpy as np

# ── Placeholder model ──────────────────────────────────────────────────────
# Replace this entire file with your real Kaggle model loading + inference

EMOTIONS = ["calm", "focused", "stressed", "excited", "sad", "angry", "relaxed", "fatigued"]

def load_model(model_path: str = "model.pth"):
    """
    Load your trained model here.
    Example for PyTorch:
        import torch
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model
    """
    if os.path.exists(model_path):
        # import torch
        # return torch.load(model_path, map_location="cpu")
        pass
    return None  # placeholder


def predict(model, features: np.ndarray) -> dict:
    """
    Run inference on the feature vector.
    Replace with your real model inference.
    """
    import random

    # ── Placeholder: random scores ─────────────────────────────────────────
    scores = [round(random.uniform(0, 1), 4) for _ in EMOTIONS]
    total  = sum(scores)
    scores = [round(s / total, 4) for s in scores]
    top    = int(np.argmax(scores))

    return {
        "emotion":    EMOTIONS[top],
        "confidence": scores[top],
        "scores":     dict(zip(EMOTIONS, scores)),
    }
