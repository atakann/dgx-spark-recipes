"""Ming-Image-0.2-Design (inclusionAI), text to image, BF16, one DGX Spark. Runs inside the recipe image.

Loads the model once (about 6 min), then about 2 min for each image at 3-4 MP (12 steps).

  ./run-recipe.sh image/ming-image-0.2-design -- --prompt "..." --width 2048 --height 2048
  ./run-recipe.sh image/ming-image-0.2-design -- --jobs /out/jobs.json

jobs.json is a list of {"name", "prompt", "width", "height", "seed" (optional)}; images go to out/<name>.png.
Write every visible word into the prompt: text it does not spell out comes out garbled.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/ming")
import infer  # Ming-Image's own CLI module, from the image
from inference_profile import load_checkpoint_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--prompt")
    ap.add_argument("--jobs", help="a JSON list of jobs, instead of --prompt")
    ap.add_argument("--name", default="ming")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.jobs:
        jobs = json.loads(Path(a.jobs).read_text())
    elif a.prompt:
        jobs = [
            {
                "name": a.name,
                "prompt": a.prompt,
                "width": a.width,
                "height": a.height,
                "seed": a.seed,
            }
        ]
    else:
        ap.error("give --prompt or --jobs")

    model_dir = Path(f"/hf/hub/models--{a.model.replace('/', '--')}/snapshots/{a.rev}")
    # Ming's loader takes its CLI namespace; build it with the CLI's own defaults
    sys.argv = [
        "infer.py",
        "--model",
        str(model_dir),
        "--local-files-only",
        "--task",
        "text-to-image",
        "--prompt",
        "-",
        "--output-dir",
        "/out",
    ]
    args = infer.parse_args()
    profile = load_checkpoint_capabilities(model_dir)
    sampling = profile.resolve_sampling_parameters(steps=args.steps, cfg=args.cfg)

    t0 = time.time()
    model, processor = infer.load_model_and_processor(model_dir, args)
    print(f"load {time.time() - t0:.0f} s", flush=True)

    out = Path("/out")
    for job in jobs:
        w, h = job["width"], job["height"]
        _, _, resolution = infer.resolve_t2i_size("text-to-image", None, w, h)
        t0 = time.time()
        images = infer.run_generation(
            model,
            processor,
            profile,
            task="text-to-image",
            prompt=job["prompt"],
            input_image=None,
            resolution=resolution,
            width=w,
            height=h,
            sampling=sampling,
            seed=job.get("seed", 42),
            num_layers=1,
            dtype=infer._dtype(args.dtype),
        )
        tmp = out / f".{job['name']}"
        saved = Path(infer._save_outputs(images, tmp, "text-to-image")[0])
        saved.rename(out / f"{job['name']}.png")
        tmp.rmdir()
        print(f"{job['name']}.png {w}x{h} {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
