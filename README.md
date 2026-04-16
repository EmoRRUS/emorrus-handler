# Emorrus RunPod Handler

Serverless emotion inference endpoint for the Emorrus Flutter app.

## Structure
| File | Purpose |
|---|---|
| `handler.py` | RunPod entry point |
| `preprocessing.py` | EEG bandpass filter + BVP normalisation |
| `features.py` | Band power + HRV feature extraction |
| `model.py` | Model loading + inference (replace with real model) |
| `model.pth` | Trained model weights (not committed — add manually) |

## Deployment
Push to `main` branch → GitHub Actions builds Docker image → pushes to Docker Hub → RunPod pulls it.

## GitHub Secrets Required
| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

## Input Format (from Flutter)
```json
{
  "input": {
    "eeg": [[ch0...], [ch1...], [ch2...], [ch3...]],
    "bvp": [sample1, sample2, ...],
    "duration_seconds": 12
  }
}
```

## Output Format (to Flutter)
```json
{
  "output": {
    "emotion": "calm",
    "confidence": 0.72,
    "scores": { "calm": 0.72, "focused": 0.10, ... },
    "meta": { "eeg_samples": 3072, "bvp_samples": 768 }
  }
}
```
