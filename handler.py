import runpod
from preprocessing import preprocess_eeg, preprocess_bvp
from features import extract_features
from model import load_model, predict

# Load model once at container startup
model = load_model("model.pth")

def handler(job):
    inp = job.get("input", {})
    eeg = inp.get("eeg", [])
    bvp = inp.get("bvp", [])

    # Preprocess
    eeg_clean = preprocess_eeg(eeg)
    bvp_clean = preprocess_bvp(bvp)

    # Extract features
    features = extract_features(eeg_clean, bvp_clean)

    # Inference
    result = predict(model, features)

    result["meta"] = {
        "eeg_samples": eeg_clean.shape[1] if eeg_clean.ndim > 1 else 0,
        "bvp_samples": len(bvp_clean),
    }

    return result

runpod.serverless.start({"handler": handler})
