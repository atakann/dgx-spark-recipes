"""ModelOpt post-training quantization of one HF checkpoint with NVIDIA's hf_ptq.py.
Runs inside the recipe image; the layer map comes from the recipe's config (mounted at /config.yaml).

  ./run-recipe.sh qwen3.8-27b-nvfp4-quant -- --name my-nvfp4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--name", default="nvfp4", help="export folder under ./out")
    ap.add_argument("--dataset", default="cnn_dailymail,nemotron-post-training-dataset-v1")
    ap.add_argument("--calib-size", default="512,512", help="samples per dataset")
    args = ap.parse_args()

    snap = Path("/hf/hub") / f"models--{args.model.replace('/', '--')}" / "snapshots" / args.rev
    cmd = [
        sys.executable, "hf_ptq.py",
        "--pyt_ckpt_path", str(snap),
        "--recipe", "/config.yaml",
        "--dataset", args.dataset,
        "--calib_size", args.calib_size,
        "--export_path", f"/out/{args.name}",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, cwd="/opt/modelopt/examples/hf_ptq", check=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min, export in out/{args.name}")


if __name__ == "__main__":
    main()
