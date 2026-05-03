"""
set_emotion.py  ──  Local control script
===============================================
Run this on your laptop to instantly change the
emotion that the RunPod endpoint returns.

Usage:
    python set_emotion.py <emotion>
    python set_emotion.py enthusiasm
    python set_emotion.py sadness
    python set_emotion.py fear
    python set_emotion.py neutral

Valid emotions: neutral, enthusiasm, sadness, fear

Requirements:
    pip install redis

Environment variables (or edit the constants below):
    REDIS_URL   - e.g. redis://default:<password>@<host>:<port>
"""

import sys
import os
import json
import time
import redis

# Fix Windows PowerShell unicode (emoji) printing
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ────────────────────────────────────────────────────────────
# Set your Redis URL here, or export REDIS_URL as an environment variable.
# Get this from Redis Cloud / Upstash / Railway / etc.
REDIS_URL = os.environ.get(
    "REDIS_URL",
    "rediss://default:gQAAAAAAAbvjAAIgcDEwZjYwYzdkOWM3ZDg0MWZlOTNlMzBjNjE2YWNiOWQ0Mg@positive-rodent-113635.upstash.io:6379"
)

REDIS_KEY   = "emorrus:current_emotion"   # key the RunPod handler reads
TTL_SECONDS = 3600                         # auto-expire after 1 hour (safety)

VALID_EMOTIONS = {"neutral", "enthusiasm", "sadness", "fear"}

CONFIDENCE_MAP = {
    "neutral":    {"neutral": 0.75, "enthusiasm": 0.10, "sadness": 0.10, "fear": 0.05},
    "enthusiasm": {"neutral": 0.08, "enthusiasm": 0.78, "sadness": 0.08, "fear": 0.06},
    "sadness":    {"neutral": 0.09, "enthusiasm": 0.07, "sadness": 0.76, "fear": 0.08},
    "fear":       {"neutral": 0.07, "enthusiasm": 0.05, "sadness": 0.10, "fear": 0.78},
}

# ─────────────────────────────────────────────────────────────────────────────

def push_emotion(emotion: str) -> None:
    emotion = emotion.strip().lower()
    if emotion not in VALID_EMOTIONS:
        print(f"[ERROR] '{emotion}' is not a valid emotion.")
        print(f"        Valid options: {', '.join(sorted(VALID_EMOTIONS))}")
        sys.exit(1)

    payload = json.dumps({
        "emotion":    emotion,
        "confidence": CONFIDENCE_MAP[emotion][emotion],
        "scores":     CONFIDENCE_MAP[emotion],
        "set_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    r = redis.from_url(REDIS_URL, socket_timeout=5)
    r.setex(REDIS_KEY, TTL_SECONDS, payload)

    print(f"[OK] Emotion set to '{emotion}' in Redis")
    print(f"    Key     : {REDIS_KEY}")
    print(f"    TTL     : {TTL_SECONDS}s")
    print(f"    Payload : {payload}")


def interactive_mode() -> None:
    """Simple REPL so you can switch emotions without restarting the script."""
    r = redis.from_url(REDIS_URL, socket_timeout=5)
    print("\n=== EmoRRUS Emotion Controller (interactive mode) ===")
    print(f"   Valid emotions: {', '.join(sorted(VALID_EMOTIONS))}")
    print("   Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            raw = input("emotion> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if raw in ("exit", "quit", "q"):
            print("Bye!")
            break
        if raw not in VALID_EMOTIONS:
            print(f"  [!] Unknown emotion. Choose from: {', '.join(sorted(VALID_EMOTIONS))}")
            continue
        payload = json.dumps({
            "emotion":    raw,
            "confidence": CONFIDENCE_MAP[raw][raw],
            "scores":     CONFIDENCE_MAP[raw],
            "set_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        r.setex(REDIS_KEY, TTL_SECONDS, payload)
        print(f"  [OK] Set -> '{raw}'  (confidence {CONFIDENCE_MAP[raw][raw]:.2f})")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No argument → interactive REPL
        interactive_mode()
    elif len(sys.argv) == 2:
        push_emotion(sys.argv[1])
    else:
        print("Usage:  python set_emotion.py [emotion]")
        print("        python set_emotion.py           # interactive mode")
        print("        python set_emotion.py sadness   # one-shot")
        sys.exit(1)
