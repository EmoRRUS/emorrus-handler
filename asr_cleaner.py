"""
asr_cleaner.py
--------------
Artifact Subspace Reconstruction (ASR) for the RunPod inference pipeline.

Workflow:
  1. At container startup, load the pre-recorded clean calibration JSON,
     fit the ASR model once, and keep it in memory.
  2. For every inference request, call clean_eeg(raw_4ch) to remove
     motion/eye/muscle artifacts before band-power computation.

Channel order expected by this module (matches Muse 2):
    [TP9, AF7, AF8, TP10]  →  shape (4, N_samples)

Sampling rate: 256 Hz (EEG_SR from preprocess.py)
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EEG_SR          = 256          # Muse 2 EEG sampling rate (Hz)
EEG_CHANNELS    = ["TP9", "AF7", "AF8", "TP10"]
CAL_FILE_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "data", "clean_calibrations", "clean_cal_2026-04-28_15-01-58.json"
)

# Raw-column names inside the calibration JSON (Muse app format)
_JSON_KEYS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

# Bandpass applied before fitting ASR (same as offline script)
_BP_LOW  = 1.0   # Hz
_BP_HIGH = 40.0  # Hz

# Global: fitted ASR object (None until _fit() is called)
_asr: "ASR | None" = None
_asr_available: bool = False   # False if mne/asrpy not installed


# ── Private helpers ────────────────────────────────────────────────────────────

def _import_deps():
    """Lazy-import MNE and ASRPy so the container can still start if missing."""
    global _asr_available
    try:
        import mne                    # noqa: F401
        from asrpy.asr import ASR    # noqa: F401
        _asr_available = True
    except ImportError as e:
        logger.warning(f"[ASR] mne/asrpy not available — ASR will be skipped. ({e})")
        _asr_available = False


def _load_cal_json(cal_path: str) -> np.ndarray:
    """
    Load a Muse-format calibration JSON and return shape (4, N).
    Keys expected: RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10.
    """
    with open(cal_path, "r") as f:
        data = json.load(f)

    arrays = []
    for key in _JSON_KEYS:
        if key not in data:
            raise KeyError(f"Calibration JSON missing key: {key}")
        arrays.append(np.asarray(data[key], dtype=np.float64))

    # Trim to shortest channel length (rare timestamp drift)
    min_len = min(a.shape[0] for a in arrays)
    cal = np.stack([a[:min_len] for a in arrays], axis=0)  # (4, N)

    # Convert µV → V (MNE convention)
    cal = cal * 1e-6
    return cal


def _bandpass_mne(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Apply zero-phase FIR bandpass using MNE (operates on (n_ch, n_samples))."""
    from mne.filter import filter_data
    return filter_data(
        data.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def init_asr(cal_path: str = CAL_FILE_DEFAULT) -> bool:
    """
    Fit the ASR model from a clean calibration file.
    Must be called once at container startup before any clean_eeg() calls.

    Parameters
    ----------
    cal_path : str
        Path to the calibration JSON file.

    Returns
    -------
    bool  True if ASR was fitted successfully, False otherwise.
    """
    global _asr, _asr_available

    _import_deps()
    if not _asr_available:
        return False

    if not os.path.isfile(cal_path):
        logger.error(f"[ASR] Calibration file not found: {cal_path}")
        _asr_available = False
        return False

    try:
        from asrpy.asr import ASR

        logger.info(f"[ASR] Loading calibration from {cal_path} ...")
        cal_data = _load_cal_json(cal_path)               # (4, N) in Volts
        cal_bp   = _bandpass_mne(cal_data, EEG_SR, _BP_LOW, _BP_HIGH)

        _asr = ASR(sfreq=EEG_SR)
        _asr.fit(cal_bp)

        n_sec = cal_data.shape[1] / EEG_SR
        logger.info(
            f"[ASR] Fitted successfully on {n_sec:.1f}s of calibration data "
            f"({cal_data.shape[1]} samples, 4 channels)."
        )
        return True

    except Exception as e:
        logger.error(f"[ASR] Failed to fit: {e}", exc_info=True)
        _asr = None
        _asr_available = False
        return False


def clean_eeg(raw_4ch: np.ndarray) -> np.ndarray:
    """
    Apply ASR artifact removal to a 4-channel EEG array.

    Parameters
    ----------
    raw_4ch : np.ndarray, shape (4, N)
        Raw EEG in **microvolts** (µV), channel order [TP9, AF7, AF8, TP10].

    Returns
    -------
    np.ndarray, shape (4, N)
        Cleaned EEG in **microvolts** (µV).
        If ASR is unavailable or fails, returns the input unchanged.
    """
    if not _asr_available or _asr is None:
        return raw_4ch   # graceful fallback — pipeline continues unaffected

    try:
        # Convert µV → V for MNE/ASRPy
        data_v = raw_4ch.astype(np.float64) * 1e-6

        # Bandpass first (same as calibration)
        data_bp = _bandpass_mne(data_v, EEG_SR, _BP_LOW, _BP_HIGH)

        # Run ASR transform
        cleaned_v = _asr.transform(data_bp)

        # Convert V → µV and preserve original dtype
        cleaned_uv = (cleaned_v * 1e6).astype(raw_4ch.dtype)

        n_ch, n_samp = raw_4ch.shape
        logger.debug(f"[ASR] Cleaned {n_samp} samples across {n_ch} channels.")
        return cleaned_uv

    except Exception as e:
        # Never crash the inference pipeline — just skip ASR
        logger.warning(f"[ASR] transform failed (skipping): {e}")
        return raw_4ch
