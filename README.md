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
| `qwen3.8-27b-nvfp4-quant` | LLM quantization, NVFP4 | ModelOpt 0.47.0 | 50 min, KLD 0.0295 vs BF16 |
| `ornith-1.5-122b-q4km` | LLM server, Q4_K_M GGUF | llama.cpp v0.7.0, built for sm_121 | 28.2 tok/s, one user |

## CHANGELOG

### 2026-10-20

#### Ornith-1.5-122B on one Spark, llama.cpp built for GB10

Added a server recipe for Ornith-1.5-122B-A10B (released today, MIT) with the lab's own Q4_K_M GGUF
(76 GB). It serves an OpenAI-compatible API on port 8000, bound to 127.0.0.1 unless you set `HOST`.

One user, warm: 28.2 tok/s decode, an 8k-token prompt in 8.5 s, about 80 GiB in use.
GPQA Diamond 74.2% with thinking off (simple-evals format, 198 questions, 95% CI 67.7 to 79.8).
The lab's 85.5 is with thinking on.

The prebuilt llama.cpp CUDA image is compiled up to sm_120 and decodes this model at
1.8 tok/s on the GB10, so the recipe builds llama.cpp for sm_121. It also loads the weights
without mmap (one copy in unified memory) and turns off auto-fit, which reads page cache
as used memory and moves the layers to the CPU.

```bash
./run-recipe.sh ornith-1.5-122b-q4km --setup
```

#### Qwen3.8-27B NVFP4 quantization with ModelOpt

Added a recipe that makes an NVFP4 checkpoint on one Spark with NVIDIA ModelOpt 0.47.0:
MLP in NVFP4, attention in FP8, `lm_head` and embeddings in BF16, Local-Hessian calibration.

Calibration takes about 50 min; the export is 22 GiB. Against BF16 on WikiText-2 it loses
2.2% perplexity (KLD 0.0295, top-1 agreement 91.1%). NVIDIA's own NVFP4 of the same model
loses 2.9% (KLD 0.0398, 88.7%). The result is on Hugging Face as
`atakans/Qwen3.8-27B-NVFP4-BF16head`.

Calibration downloads two datasets, so log in with `hf auth login` first.

```bash
./run-recipe.sh qwen3.8-27b-nvfp4-quant --setup
```

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
