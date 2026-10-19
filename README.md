# DGX Spark recipes

Recipes for the NVIDIA DGX Spark (GB10, 128 GB unified memory): image, video, audio,
quantization, and LLM serving. Each one was run and measured on a Spark.

Every recipe:
- pins the container image, the engine version and the model revision
- ships with the numbers it was measured at
- runs without host changes: read-only model cache, no added privileges, output in `./out`

Not affiliated with NVIDIA or any model maker. Each model keeps its own license.

## Quick start

```bash
git clone https://github.com/atakann/dgx-spark-recipes
cd dgx-spark-recipes
./run-recipe.sh qwen-image-2.1-turbo --setup
```

`--setup` builds the image and downloads the model at the pinned revision. Leave it out
on later runs. Script arguments go after `--`:

```bash
./run-recipe.sh qwen-image-2.1-turbo -- --prompt "a red fox in fresh snow" --seed 7
```

Needs Docker with the NVIDIA runtime (DGX OS has both) and the `hf` CLI for `--setup`.

## Recipes

| recipe | task | engine | one Spark |
|---|---|---|---|
| `qwen-image-2.1-turbo` | text to image, BF16, 8 steps | diffusers 0.41.0 | 55.5 s per 2048² image |
| `qwen-image-2.1-base` | text to image, BF16, 40 steps | diffusers 0.41.0 | 257 s per 2048² image |

## CHANGELOG

### 2026-10-19

#### Qwen-Image 2.1, base and Turbo

Added diffusers recipes for both checkpoints, BF16 as released.

At 2048×2048 a step takes 6.3 s on both, because they are the same 7B transformer.
Turbo needs 8 steps (55.5 s per image), base needs 40 (257 s). Peak memory is
56.6 GiB on both, in the untiled VAE decode. Loading takes about 4 min.

```bash
./run-recipe.sh qwen-image-2.1-turbo --setup
./run-recipe.sh qwen-image-2.1-base --setup
```
