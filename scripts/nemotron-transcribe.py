"""Transcribe one recording with a running Nemotron 3.5 Transcribe recipe: the transcript, word timestamps
and speaker tags (NVIDIA's "who said what, and when" prompt), saved as .json and .srt next to the input.
Standard library only; the input is a WAV file (any rate, the server resamples), up to one hour.

  ffmpeg -i film.mp4 -vn -ac 1 -ar 16000 film.wav
  python3 scripts/nemotron-transcribe.py film.wav
"""

import argparse
import base64
import json
import re
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--speakers", type=int, default=4, help="speaker tags offered in the prompt")
a = ap.parse_args()

tags = " ".join(f"<spk:{i}>" for i in range(a.speakers))
prompt = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
    f"Write numbers in their standard written form. Put speaker tags {tags} on the transcription."
)
with open(a.wav, "rb") as f:
    audio = base64.b64encode(f.read()).decode()
body = {
    "model": "nemotron-transcribe",
    "temperature": 0,
    "skip_special_tokens": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
    ]}],
}
req = urllib.request.Request(
    f"{a.url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=3600) as r:
    reply = json.load(r)
wall = time.time() - t0
base = a.wav.rsplit(".", 1)[0]
with open(base + ".json", "w") as f:
    json.dump(reply, f, indent=1)

ts = reply.get("ctc_timestamps") or {}
words = ts.get("words") or []
if ts.get("error"):
    print("timestamps:", ts["error"])
spoken = len(re.sub(r"<[^>]+>", " ", reply["choices"][0]["message"]["content"]).split())
if spoken and len(words) < 0.9 * spoken:  # NeMo's aligner drops words in long one-speaker stretches
    note = f"timestamps cover {len(words)} of {spoken} words; for full timestamps, cut the file into parts of a few minutes"
else:
    note = ""


def stamp(s):
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# one cue per speaker turn, cut after a sentence once it has 12 words
cues = []
for w in words:
    c = cues[-1] if cues else None
    if not c or c["spk"] != w["speaker"] or (len(c["w"]) >= 12 and c["w"][-1]["word"][-1] in ".?!"):
        cues.append({"spk": w["speaker"], "w": []})
    cues[-1]["w"].append(w)
with open(base + ".srt", "w") as f:
    for i, c in enumerate(cues, 1):
        text = " ".join(w["word"] for w in c["w"])
        f.write(f"{i}\n{stamp(c['w'][0]['start'])} --> {stamp(c['w'][-1]['end'])}\nSpeaker {int(c['spk']) + 1}: {text}\n\n")

print(reply["choices"][0]["message"]["content"])
print(f"\n{wall:.1f} s, {len(words)} timed words, {len(cues)} cues -> {base}.json, {base}.srt")
if note:
    print(note)
