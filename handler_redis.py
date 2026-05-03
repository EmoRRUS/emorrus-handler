"""
handler_redis.py  ──  RunPod serverless handler (Redis-backed, no ML)
======================================================================
Replaces the ML inference pipeline with a simple Redis fetch.
The RunPod endpoint accepts the same request format as before
(eeg_csv / ppg_csv are accepted but ignored), so the Flutter app
doesn't need any changes.

Environment variables (set in RunPod template / secrets):
    REDIS_URL   – full Redis connection URL, e.g.
                  redis://default:<password>@<host>:<port>

The emotion value is set externally via set_emotion.py on your laptop.
"""

import os
import json
import time
import traceback
import redis
import runpod

# ── Config ───────────────────────────────────────────────────────────────────
REDIS_URL       = os.environ.get("REDIS_URL", "")
REDIS_KEY       = "emorrus:current_emotion"
CONNECT_TIMEOUT = 3   # seconds

DEFAULT_EMOTION = "neutral"
DEFAULT_PAYLOAD = {
    "emotion":    DEFAULT_EMOTION,
    "confidence": 0.60,
    "scores": {
        "neutral":    0.60,
        "enthusiasm": 0.15,
        "sadness":    0.15,
        "fear":       0.10,
    },
}

# ── Redis client (module-level, reused across warm invocations) ───────────────
_redis_client: redis.Redis | None = None

def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        if not REDIS_URL:
            raise RuntimeError(
                "REDIS_URL environment variable is not set. "
                "Add it to the RunPod template/secrets."
            )
        _redis_client = redis.from_url(
            REDIS_URL,
            socket_timeout=CONNECT_TIMEOUT,
            socket_connect_timeout=CONNECT_TIMEOUT,
            decode_responses=True,
        )
        print(f"[startup] Redis client created → {REDIS_URL[:30]}...")
    return _redis_client


def _fetch_emotion() -> dict:
    """Read the current emotion from Redis. Falls back to DEFAULT_PAYLOAD."""
    try:
        r = _get_redis()
        raw = r.get(REDIS_KEY)
        if raw is None:
            print(f"[handler] Key '{REDIS_KEY}' not found in Redis → using default '{DEFAULT_EMOTION}'")
            return DEFAULT_PAYLOAD
        data = json.loads(raw)
        print(f"[handler] Redis → {data}")
        return data
    except Exception as exc:
        print(f"[handler] Redis read failed: {exc}  → using default '{DEFAULT_EMOTION}'")
        return DEFAULT_PAYLOAD


# ── RunPod handler ────────────────────────────────────────────────────────────
def handler(event):
    """
    Accepts the same input schema as the ML handler
    (eeg_csv, ppg_csv, trial_key) but ignores the data;
    returns the emotion that was last set via set_emotion.py.
    """
    try:
        emotion_data = _fetch_emotion()

        emotion    = emotion_data.get("emotion",    DEFAULT_EMOTION)
        confidence = emotion_data.get("confidence", 0.60)
        scores     = emotion_data.get("scores",     DEFAULT_PAYLOAD["scores"])

        # Build a response that is 100 % compatible with the original ML handler
        response = {
            "emotion":    emotion,
            "confidence": round(float(confidence), 4),
            "scores": {
                "neutral":    round(float(scores.get("neutral",    0.25)), 4),
                "enthusiasm": round(float(scores.get("enthusiasm", 0.25)), 4),
                "sadness":    round(float(scores.get("sadness",    0.25)), 4),
                "fear":       round(float(scores.get("fear",       0.25)), 4),
            },
            # Minimal window so downstream code doesn't break
            "n_windows": 1,
            "windows": [{
                "window_idx": 0,
                "start_sec":  0.0,
                "end_sec":    20.0,
                "emotion":    emotion,
                "scores": {
                    "neutral":    round(float(scores.get("neutral",    0.25)), 4),
                    "enthusiasm": round(float(scores.get("enthusiasm", 0.25)), 4),
                    "sadness":    round(float(scores.get("sadness",    0.25)), 4),
                    "fear":       round(float(scores.get("fear",       0.25)), 4),
                },
            }],
            "mode": "redis_override",  # lets you distinguish from real inference
        }

        return response

    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
