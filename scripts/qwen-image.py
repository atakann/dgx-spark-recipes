"""Qwen-Image 2.1 (base or Turbo), BF16, diffusers, timed per step. Runs inside the recipe image.

  ./run-recipe.sh qwen-image-2.1-turbo -- --prompt "a red fox in fresh snow" --seed 7
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import diffusers
import torch
from diffusers import QwenImage21Pipeline

SIGN = (
    "Photograph of a small machine workshop at night, one work lamp on. On the brick wall, "
    "a rectangular enamel sign hangs slightly tilted, seen from the side at about 30 degrees, "
    "so the text runs at an angle. The sign has three lines of painted lettering: the first "
    'line large in cream capitals "DGX SPARK", the second line medium "GB10 · 128 GB", '
    'the third line small in the bottom-right corner "recipe #1". Chipped enamel, a little '
    "rust at the screws. A workbench with tools below, shallow depth of field, 35mm film look, "
    "natural grain."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--prompt", default=SIGN)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--images", type=int, default=2, help="the first one is the cold run")
    args = ap.parse_args()

    # the local snapshot, no hub calls: the cache is mounted read-only
    snap = Path("/hf/hub") / f"models--{args.model.replace('/', '--')}" / "snapshots" / args.rev
    t0 = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(snap, dtype=torch.bfloat16).to("cuda")
    load_s = time.perf_counter() - t0
    print(f"load {load_s:.1f} s")

    name = args.model.split("/")[-1]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    rows = []
    for i in range(args.images):
        stamps = []

        def on_step(_pipe, _step, _t, kw, stamps=stamps):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            return kw

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        # no steps, no cfg: each checkpoint's own defaults (base 40 steps, Turbo its saved 8)
        image = pipe(
            prompt=args.prompt,
            width=args.size,
            height=args.size,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            callback_on_step_end=on_step,
        ).images[0]
        torch.cuda.synchronize()
        end = time.perf_counter()

        steps = [b - a for a, b in zip([start, *stamps], stamps)]
        png = Path("/out") / f"{stamp}_{name}_{i}.png"
        image.save(png)
        row = {
            "image": i,
            "steps": len(stamps),
            "first_step_s": round(steps[0], 2),  # includes the text encoder
            "s_per_step": round(sum(steps[1:]) / max(len(steps) - 1, 1), 2),
            "after_steps_s": round(end - stamps[-1], 2),  # VAE decode + png
            "total_s": round(end - start, 2),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "png": png.name,
        }
        rows.append(row)
        print(json.dumps(row))

    record = {
        "model": args.model,
        "revision": args.rev,
        "size": args.size,
        "seed": args.seed,
        "prompt": args.prompt,
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "load_s": round(load_s, 1),
        "images": rows,
    }
    (Path("/out") / f"{stamp}_{name}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
