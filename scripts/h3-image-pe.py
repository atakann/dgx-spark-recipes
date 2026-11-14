#!/usr/bin/env python3
"""MiniMax-H3-Image with its prompt expansion model: a short request -> a long prompt and an aspect ratio -> one image.

Needs both servers on this Spark: image/minimax-h3-image-mxfp8-sglang (port 8000) and
llm/minimax-h3-image-pe-sglang (port 8001). The task prompt, the system prompt and the request
parameters are MiniMax's, read from the model's folder in the Hugging Face cache.

  python3 scripts/h3-image-pe.py "An old watchmaker at his workbench." watchmaker.png
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PE = os.environ.get("PE_URL", "http://127.0.0.1:8001")
IMAGE = os.environ.get("IMAGE_URL", "http://127.0.0.1:8000")
# about 16 MP for every ratio that the model can answer with; both sides are multiples of 32
SIZES = {
    "1:1": "4096x4096", "5:4": "4576x3648", "4:3": "4736x3552", "3:2": "4992x3328", "16:9": "5440x3072",
    "21:9": "6208x2656", "4:5": "3648x4576", "3:4": "3552x4736", "2:3": "3328x4992", "9:16": "3072x5440",
}  # fmt: skip


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    request, out = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "h3-image.png"
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    found = sorted((hf / "hub/models--MiniMaxAI--MiniMax-H3-Image-PE/snapshots").glob("*/prompt/t2i"))
    if not found:
        sys.exit("MiniMax-H3-Image-PE is not in the Hugging Face cache: run its recipe with --setup first")
    prompts = found[-1]

    body = json.loads((prompts / "request_params.json").read_text())
    body |= {
        "model": "MiniMax-H3-Image-PE",
        "stream": False,
        "messages": [
            {"role": "system", "content": (prompts / "system_prompt.txt").read_text().strip()},
            {"role": "user", "content": (prompts / "task_prompt.txt").read_text().format(request)},
        ],
    }
    start = time.time()
    reply = post(PE + "/v1/chat/completions", body)
    answer = json.loads(reply["choices"][0]["message"]["content"])
    ratio, prompt = answer["aspect_ratio"], answer["expanded_caption"]
    tokens = reply["usage"]["completion_tokens"]
    print(f"prompt: {time.time() - start:.1f} s, {tokens} tokens, {len(prompt.split())} words, ratio {ratio}\n")
    print(prompt + "\n")

    size = SIZES.get(ratio, "4096x4096")
    start = time.time()
    image = post(
        IMAGE + "/v1/images/generations",
        {"prompt": prompt, "size": size, "seed": int(os.environ.get("SEED", 42)), "response_format": "b64_json"},
    )
    Path(out).write_bytes(base64.b64decode(image["data"][0]["b64_json"]))
    print(f"image: {time.time() - start:.1f} s, {size} -> {out}")


if __name__ == "__main__":
    main()
