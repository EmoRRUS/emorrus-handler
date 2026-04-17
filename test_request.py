"""
test_request.py  - Simulates exactly what the Flutter app sends to RunPod:
  - Raw EEG CSV (RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10) base64-encoded
  - Raw PPG CSV (time_s, ppg_green) base64-encoded
  - No band powers (handler computes them)

Run with:
    python test_request.py --endpoint https://api.runpod.ai/v2/<ID>/runsync --api-key <KEY>
"""

import io, json, base64, argparse
import numpy as np
import pandas as pd
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", required=True)
parser.add_argument("--api-key",  required=True)
parser.add_argument("--duration", type=int, default=12)
args = parser.parse_args()

EEG_SR = 256
BVP_SR = 25
n_eeg  = EEG_SR * args.duration
n_bvp  = BVP_SR * args.duration

np.random.seed(42)
t     = np.linspace(0, args.duration, n_eeg)
t_bvp = np.linspace(0, args.duration, n_bvp)

eeg = np.array([
    5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20,
    5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20,
    5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20,
    5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_eeg) * 20,
], dtype=np.float32)

bvp = (
    np.sin(2 * np.pi * 1.17 * t_bvp) +
    0.3 * np.sin(2 * np.pi * 2.34 * t_bvp) +
    np.random.randn(n_bvp) * 0.05
).astype(np.float32)

# Build raw EEG CSV -- exactly like Flutter EegPreprocessor.buildEegCsv
eeg_buf = io.BytesIO()
pd.DataFrame({"RAW_TP9": eeg[0], "RAW_AF7": eeg[1], "RAW_AF8": eeg[2], "RAW_TP10": eeg[3]}).to_csv(eeg_buf, index=False)
eeg_b64 = base64.b64encode(eeg_buf.getvalue()).decode("utf-8")

# Build PPG CSV -- exactly like Flutter EegPreprocessor.buildPpgCsv
ppg_buf = io.BytesIO()
pd.DataFrame({"time_s": np.arange(n_bvp, dtype=np.float32) / BVP_SR, "ppg_green": bvp}).to_csv(ppg_buf, index=False)
ppg_b64 = base64.b64encode(ppg_buf.getvalue()).decode("utf-8")

payload = {
    "input": {
        "eeg_csv":   eeg_b64,
        "ppg_csv":   ppg_b64,
        "trial_key": "test_trial",
    }
}

print(f"Sending {args.duration}s synthetic data (app format) to RunPod ...")
print(f"  EEG CSV : {n_eeg} samples, {len(eeg_b64)/1024:.1f} KB base64")
print(f"  PPG CSV : {n_bvp} samples, {len(ppg_b64)/1024:.1f} KB base64\n")

resp = requests.post(
    args.endpoint,
    headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
    json=payload,
    timeout=120,
)
print(f"HTTP status: {resp.status_code}")
print("=" * 50)
print(json.dumps(resp.json(), indent=2))