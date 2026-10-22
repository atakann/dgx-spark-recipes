"""FastH3-QAD-v2 on FastVideo with FastVideo's own DGX Spark config. Runs inside the recipe image.

  ./run-recipe.sh fasth3-qad                  # 10 s of 1344x768 video with sound, the prompt below
  ./run-recipe.sh fasth3-qad -- --ref         # the config as shipped: 832x480, 124 frames
  ./run-recipe.sh fasth3-qad -- --prompt-file out/my-prompt.txt --seed 7

The video and FastVideo's stage log (time per stage) go to ./out.
"""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

CONFIG = "examples/inference/basic/basic_fasth3_spark_qad_v2_nvfp4.yaml"
TOKYO = """\
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot of a narrow back \
street in Tokyo at night in the rain. A red neon sign reading "RAMEN 24H" glows above a small ramen \
shop, its light streaking across the wet asphalt. A person with a clear umbrella walks toward the \
shop's fabric curtain as the camera pushes in with small amplitude at slow speed. [Shot 2] At \
00:04.000, the camera cuts to a medium close-up inside the warm, steamy shop. An elderly ramen chef \
with a white headband, deep smile lines and a low, warm voice (S1) lifts a full bowl from the \
counter, steam rising past his face, and says: <d>[Japanese] いらっしゃい。今夜最後の一杯だよ。</d> He \
sets the bowl down in front of the camera and nods once.

overall_soundscape: Steady rain falls on the umbrella and the street while a distant train passes. \
Inside, broth simmers in a large pot, a ladle clinks against metal, and the fabric curtain rustles \
as it is pushed aside.

non_diegetic_music: A slow lo-fi beat with soft electric piano chords and light vinyl crackle, \
fading out at the end."""

# the env vars the config's header lists: FP4 sparse attention, Triton VSA kernel, no FA4
ENV = {
    "FASTVIDEO_H3_VSA_FP4": "1",
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_NVFP4_MM_BACKEND": "cudnn",
    "FASTVIDEO_NVFP4_MM_ROW_SPLIT": "table",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "12",
    "FASTVIDEO_H3_VAE_INT8_OVERLAY": "0",
    "FASTVIDEO_VSA_TRITON": "1",
    "FASTVIDEO_VSA_PTX_BLACKWELL": "0",
    "FASTVIDEO_FA4": "0",
    "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
    "FASTVIDEO_STAGE_LOGGING": "1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--ref", action="store_true", help="run the config's own request unchanged")
    ap.add_argument("--prompt-file", help="a path under ./out, seen as /out/... in the container")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(CONFIG).read_text())
    # downloaded by sha, so the read-only offline cache has no refs/main
    cfg["generator"]["model_path"] = args.model
    cfg["generator"]["revision"] = args.rev
    req = cfg["request"]
    name = "ref" if args.ref else "clip"
    if not args.ref:  # spark-009's clip: 10 s of 1344x768 at 24 fps
        prompt = TOKYO
        if args.prompt_file:
            prompt = Path("/out", Path(args.prompt_file).name).read_text().strip()
        req["prompt"] = prompt
        req["sampling"].update(seed=args.seed, height=768, width=1344, num_frames=243)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = Path("/out") / f"{stamp}_fasth3-{name}"
    out.mkdir(parents=True)
    req["output"]["output_path"] = f"{out}/"
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    snap = Path("/hf/hub") / f"models--{args.model.replace('/', '--')}" / "snapshots" / args.rev
    env = os.environ | ENV | {
        "HF_HOME": "/hf",  # the read-only cache, offline
        # the model card's AdaLN table replaces 26 GB of BF16 timestep layers
        "FASTVIDEO_H3_ADALN_TABLE": str(snap / "transformer/adaln_tables.pt"),
    }
    with open(out / "generate.log", "w") as log:
        rc = subprocess.run(
            ["fastvideo", "generate", "--config", str(out / "config.yaml")],
            check=False,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    print(f"exit {rc}, video and log in out/{out.name}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
