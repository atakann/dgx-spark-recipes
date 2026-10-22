#!/usr/bin/env python3
"""FLUX 3 Klein BF16 vs NVFP4 on the recipe's ComfyUI server: fixed prompts and seeds,
times from ComfyUI's own history. Runs on the host, standard library only.

  ./run-recipe.sh flux-3-klein --setup
  python3 scripts/klein-bench.py              # server on http://127.0.0.1:8000
"""

import itertools
import json
import pathlib
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parent.parent
# BFL's flux3_klein_bf16_t2i graph in API form, prompt upsampler off, alpha 6.93
GRAPH = json.loads((ROOT / "workflows/flux-3-klein-t2i-api.json").read_text())
PROMPTS = json.loads((ROOT / "workflows/klein-prompts.json").read_text())
SEEDS = [1, 2]
MODELS = {
    "bf16": "flux3-image-klein-bf16.safetensors",
    "nvfp4": "flux3-image-klein-nvfp4.safetensors",
}
MP = [1.0, 4.0]


def call(path, body=None):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))


def run(prompt, seed, model, mp, name):
    g = json.loads(json.dumps(GRAPH))
    g["28"]["inputs"]["value"] = prompt
    g["128:107"]["inputs"]["seed"] = seed
    g["128:101"]["inputs"]["unet_name"] = MODELS[model]
    g["19"]["inputs"]["megapixels"] = mp
    g["9"]["inputs"]["filename_prefix"] = f"bench/{name}"
    pid = call("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = call(f"/history/{pid}").get(pid)
        if h and h["status"]["completed"]:
            break
        if h and h["status"]["status_str"] == "error":
            sys.exit(json.dumps(h["status"]))
        time.sleep(1)
    ts = {m[0]: m[1]["timestamp"] for m in h["status"]["messages"]}
    return (ts["execution_success"] - ts["execution_start"]) / 1000


rows = []
for model in MODELS:  # model outer: one load per model
    for mp, (pname, p), seed in itertools.product(MP, PROMPTS.items(), SEEDS):
        name = f"{pname}_s{seed}_{mp:g}mp_{model}"
        t = run(p, seed, model, mp, name)
        rows.append({"model": model, "mp": mp, "prompt": pname, "seed": seed, "s": round(t, 2)})
        print(rows[-1], flush=True)

out = ROOT / "out"
out.mkdir(exist_ok=True)
stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
(out / f"{stamp}_klein-bench.json").write_text(json.dumps(rows, indent=1))
for model, mp in itertools.product(MODELS, MP):
    s = sorted(r["s"] for r in rows if r["model"] == model and r["mp"] == mp)
    print(f"{model} {mp:g} MP: median {(s[len(s) // 2 - 1] + s[len(s) // 2]) / 2:.1f} s")
