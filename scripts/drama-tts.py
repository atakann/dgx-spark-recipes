#!/usr/bin/env python3
"""Drama 3 on the recipe's SGLang-Omni server: one short sentence and one paragraph, one request at
a time, non-streaming. Prints the real-time factor (seconds of speech per second of compute).
Runs on the host, standard library only.

  ./run-recipe.sh drama-3 --setup
  python3 scripts/drama-tts.py              # server on http://127.0.0.1:8000
"""

import io
import json
import pathlib
import sys
import time
import urllib.request
import wave
from datetime import UTC, datetime

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
TEXTS = {
    "sentence": "The kettle clicked off, and for a moment the kitchen was completely quiet.",
    "paragraph": (
        "Every morning the old ferry left the harbor at seven, whatever the weather. "
        "The captain had made the same crossing for thirty years, and he still checked the ropes "
        "twice before he started the engine. Most passengers slept or looked at their phones. "
        "A few stood at the rail, watching the town get smaller, the gulls following the wake "
        "until the open water began. On clear days you could see the island long before you "
        "reached it, a low green line that slowly turned into houses, a church and a pier."
    ),
}


MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]


def speak(text):
    body = {"model": MODEL, "input": text, "response_format": "wav"}
    req = urllib.request.Request(
        URL + "/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    audio = urllib.request.urlopen(req).read()
    elapsed = time.perf_counter() - start
    with wave.open(io.BytesIO(audio)) as w:
        seconds = w.getnframes() / w.getframerate()
    return audio, seconds, elapsed


out = pathlib.Path(__file__).resolve().parent.parent / "out"
out.mkdir(exist_ok=True)
stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
speak(TEXTS["sentence"])  # warm-up: the first request after a start is slower
rows = []
for name, text in TEXTS.items():
    audio, seconds, elapsed = speak(text)
    (out / f"{stamp}_drama-3_{name}.wav").write_bytes(audio)
    rows.append({"text": name, "speech_s": round(seconds, 2), "compute_s": round(elapsed, 2)})
    print(f"{name}: {seconds:.1f} s of speech in {elapsed:.1f} s, {seconds / elapsed:.2f}x real time")
(out / f"{stamp}_drama-3.json").write_text(json.dumps(rows, indent=1))
