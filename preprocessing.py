import numpy as np
from scipy.signal import butter, filtfilt

SAMPLE_RATE_EEG = 256  # Hz
SAMPLE_RATE_BVP = 64   # Hz

def preprocess_eeg(eeg: list) -> np.ndarray:
    """
    Input:  eeg — list of 4 channels, each a list of raw samples
    Output: filtered numpy array of shape (4, N)
    """
    eeg_np = np.array(eeg, dtype=np.float32)  # (4, N)

    if eeg_np.shape[1] < 10:
        return eeg_np

    # Bandpass filter 1–40 Hz
    nyq = SAMPLE_RATE_EEG / 2
    b, a = butter(4, [1 / nyq, 40 / nyq], btype='band')
    eeg_filtered = filtfilt(b, a, eeg_np, axis=1)

    return eeg_filtered.astype(np.float32)


def preprocess_bvp(bvp: list) -> np.ndarray:
    """
    Input:  bvp — list of raw BVP samples
    Output: normalised numpy array of shape (N,)
    """
    bvp_np = np.array(bvp, dtype=np.float32)

    if bvp_np.std() < 1e-6:
        return bvp_np

    # Normalise to zero mean, unit variance
    bvp_np = (bvp_np - bvp_np.mean()) / bvp_np.std()

    return bvp_np
