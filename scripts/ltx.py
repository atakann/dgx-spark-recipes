"""LTX-2.5 distilled (BF16) with Lightricks' own pipeline, ltx_pipelines.distilled. Runs inside the recipe image.

  ./run-recipe.sh video/ltx-2.5-distilled                      # 10 s of 1344x768 video with sound, the prompt below
  ./run-recipe.sh video/ltx-2.5-distilled -- --prompt-file out/my-prompt.txt --seconds 5 --seed 7

The video and the pipeline's log go to ./out.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

TOKYO = (
    "Live-action, cinematic. A wide shot of a narrow back street in Tokyo at night in the rain. A red neon sign "
    'reading "RAMEN 24H" glows above a small ramen shop, its light streaking across the wet asphalt. A person with a '
    "clear umbrella walks toward the shop's fabric curtain as the camera slowly pushes in. The scene cuts to a "
    "medium close-up inside the warm, steamy shop. An elderly ramen chef with a white headband and deep smile lines "
    "lifts a full bowl from the counter, steam rising past his face, and says in a low, warm voice in Japanese: "
    '"いらっしゃい。今夜最後の一杯だよ。" He sets the bowl down in front of the camera and nods once. Steady rain '
    "falls on the street, a distant train passes, broth simmers in a large pot and a ladle clinks against metal. A "
    "slow lo-fi beat with soft electric piano chords and light vinyl crackle plays in the background."
)
FILES = {
    "--transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "--text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "--video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "--audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "--spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--prompt-file", help="a text file under ./out (seen as out/... from the recipe directory)")
    ap.add_argument("--seconds", type=int, default=10, help="video length at 24 fps")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    prompt = Path("/out", Path(a.prompt_file).name).read_text().strip() if a.prompt_file else TOKYO
    snap = Path("/hf/hub", "models--" + a.model.replace("/", "--"), "snapshots", a.rev)
    frames = 8 * round(a.seconds * 24 / 8) + 1  # the pipeline wants 8k + 1 frames
    out = Path("/out", f"{datetime.now(UTC):%Y%m%d-%H%M%S}_ltx-2.5.mp4")

    # The HF cache stores each file as a symlink to a blob without an extension; the pipeline resolves paths
    # and then rejects the text encoder for not ending in .safetensors. Keep the paths as given.
    from ltx_pipelines import distilled
    from ltx_pipelines.utils import args as ltx_args

    ltx_args.resolve_path = lambda p: str(Path(p).expanduser().absolute().as_posix())
    sys.argv = ["ltx_pipelines.distilled"]
    for flag, rel in FILES.items():
        sys.argv += [flag, str(snap / rel)]
    sys.argv += ["--num-frames", str(frames), "--width", str(a.width), "--height", str(a.height),
                 "--seed", str(a.seed), "--output-path", str(out), "--prompt", prompt]
    distilled.main()
    print(f"wrote {out.relative_to('/out')} ({frames} frames, {a.width}x{a.height})")


if __name__ == "__main__":
    main()
