import numpy as np

SAMPLE_RATE_EEG = 256

def extract_features(eeg: np.ndarray, bvp: np.ndarray) -> np.ndarray:
    """
    Extract features from preprocessed EEG and BVP.
    Replace this with whatever features your Kaggle model expects.
    
    Input:  eeg (4, N), bvp (N,)
    Output: 1D feature vector
    """
    features = []

    # ── EEG band power per channel ─────────────────────────────────────────
    bands = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  13),
        'beta':  (13, 30),
        'gamma': (30, 40),
    }

    for ch in range(eeg.shape[0]):
        freqs = np.fft.rfftfreq(eeg.shape[1], d=1.0 / SAMPLE_RATE_EEG)
        fft   = np.abs(np.fft.rfft(eeg[ch])) ** 2
        for band, (lo, hi) in bands.items():
            idx   = np.where((freqs >= lo) & (freqs <= hi))
            power = fft[idx].mean() if len(idx[0]) > 0 else 0.0
            features.append(power)

    # ── BVP statistical features ───────────────────────────────────────────
    if len(bvp) > 1:
        features.append(float(bvp.mean()))
        features.append(float(bvp.std()))
        features.append(float(np.percentile(bvp, 25)))
        features.append(float(np.percentile(bvp, 75)))
        # HRV proxy — std of differe
        features.append(float(np.diff(bvp).std()))
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)
