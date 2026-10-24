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
./run-recipe.sh image/qwen-image-2.1-turbo --setup
```

`--setup` builds the image and downloads the model at the pinned revision. Leave it out
on later runs. Script arguments go after `--`:

```bash
./run-recipe.sh image/qwen-image-2.1-turbo -- --prompt "a red fox in fresh snow" --seed 7
```

Needs Docker with the NVIDIA runtime (DGX OS has both) and the `hf` CLI for `--setup`.

## Recipes

A recipe is `recipes/<task>/<name>.yaml` and runs as `./run-recipe.sh <task>/<name>`.
LLM recipe names end in the engine. Server recipes start in the background and serve on port 8000
(`HOST`, `PORT` to change).

### LLM servers (`llm/`)

| recipe | model | engine | Sparks | one user |
|---|---|---|---|---|
| `ornith-1.5-122b-q4km-llamacpp` | Ornith-1.5-122B, Q4_K_M GGUF | llama.cpp v0.7.0, built for sm_121 | 1 | 28.2 tok/s |
| `ornith-1.5-122b-nvfp4-vllm` | Ornith-1.5-122B, NVFP4 | vLLM nightly 77154ca | 1 | 26.5 tok/s, prefill 2,447 tok/s |
| `nemotron-3.5-bolt-vllm` | Nemotron 3.5 Bolt 74B, 2-bit QAD | vLLM nightly 77154ca + b12x | 1 | 29 tok/s |
| `nemotron-3.5-bolt-lilicorr-vllm` | same, with NVIDIA's LiLiCorr draft | vLLM nightly 77154ca + b12x | 1 | 93-104 tok/s on code, JSON, rewrites |
| `qwen3.8-flash-next-fp4-vllm` | Qwen3.8-Flash-Next FP4-Local (NVIDIA's QAD NVFP4), MTP 4 | vLLM 8244d1c (eugr/spark-vllm-b12x) | 1 | 44-62 tok/s on JSON, code, rewrites |
| `qwen3.8-flash-next-fp4-flare` | same model, MTP 5 | NVIDIA Flare 5f37bd1 (research prototype) | 1 | 42-66 tok/s on JSON, rewrites, code, explanations |
| `qwen3.8-27b-nvfp4-bf16head-vllm` | Qwen3.8-27B, NVFP4 with a BF16 lm_head ([atakans](https://huggingface.co/atakans/Qwen3.8-27B-NVFP4-BF16head)) | vLLM 0.30.0 | 1 | 11.1 tok/s |

### Image (`image/`)

| recipe | model | engine | Sparks | time |
|---|---|---|---|---|
| `qwen-image-2.1-turbo` | Qwen-Image 2.1 Turbo, BF16, 8 steps | diffusers 0.41.0 | 1 | 55.5 s per 2048² image |
| `qwen-image-2.1-base` | Qwen-Image 2.1, BF16, 40 steps | diffusers 0.41.0 | 1 | 257 s per 2048² image |
| `flux-3-klein` | FLUX 3 Klein, BF16 and NVFP4 | ComfyUI v0.41.0 | 1 | 7.2 s per 1 MP image (NVFP4) |

### Video (`video/`)

| recipe | model | engine | Sparks | time |
|---|---|---|---|---|
| `fasth3-qad` | FastH3-QAD-v2, NVFP4, video with sound | FastVideo d4f1445 | 1 | 8.1 min for 10 s of 1344×768 |

### Speech (`speech/`)

| recipe | model | engine | Sparks | speed |
|---|---|---|---|---|
| `drama-3` | Fish Audio Drama 3, text to speech | SGLang-Omni 22d4ace | 1 | 1.1× real time, one request |

## CHANGELOG

### 2026-10-24 (2)

#### Recipes move into task folders; the quantization recipe becomes a server recipe

Recipes now sit in `recipes/llm`, `recipes/image`, `recipes/video` and `recipes/speech`, and run as
`./run-recipe.sh <task>/<name>` (for example `./run-recipe.sh llm/qwen3.8-flash-next-fp4-flare`).
LLM recipe names end in the engine: `ornith-1.5-122b-q4km` is now `llm/ornith-1.5-122b-q4km-llamacpp`,
`ornith-1.5-122b-nvfp4` is `llm/ornith-1.5-122b-nvfp4-vllm`, `nemotron-3.5-bolt` and
`nemotron-3.5-bolt-lilicorr` end in `-vllm`, and `qwen3.8-flash-next-fp4` is `llm/qwen3.8-flash-next-fp4-vllm`.
The tables gain a Sparks column.

The ModelOpt recipe that quantized Qwen3.8-27B is gone. In its place, `llm/qwen3.8-27b-nvfp4-bf16head-vllm`
downloads the finished checkpoint (`atakans/Qwen3.8-27B-NVFP4-BF16head`) and serves it on vLLM 0.30.0:
11.1 tok/s, one user, on every prompt type. A recipe without `build:` now pulls its published image.

### 2026-10-24

#### Qwen3.8-Flash-Next FP4-Local on NVIDIA Flare

Added a server recipe for the same `nvidia/Qwen3.8-Flash-Next-FP4-Local` checkpoint on NVIDIA's Flare research
prototype (`flare-local-inference` 5f37bd1, built from source in `images/flare`). `--setup` exports a Flare
package once, then the server starts in under a minute. Measured on one Spark, one user, temperature 0.7,
MTP 5, median of 5: JSON 66.2, rewrite 49.4, code 45.8, explain 42.4, story 27.6 tok/s. The recipe caps Flare's
memory at 25 GiB below the total, so the KV cache (430k tokens) leaves 32 GiB free.

`run-recipe.sh` gains two recipe keys: `home: true` keeps an engine's home directory (signing key, trust store,
prepared package) in `./cache/<recipe>`, and `prepare:` runs a one-time step during `--setup`. In server args,
`{dev_key}` becomes the fingerprint of the locally generated Flare signing key.

### 2026-10-23

#### Qwen3.8-Flash-Next FP4-Local on vLLM, NVIDIA's DGX Spark command

Added a server recipe for `nvidia/Qwen3.8-Flash-Next-FP4-Local` (NVIDIA's QAD build: routed experts
NVFP4, attention MXFP8, 96 GiB) with the command from NVIDIA's model card: the community image the card
names (`eugr/spark-vllm-b12x`, pinned by digest to nightly-20261015, vLLM 8244d1c), the card's
environment and flags, MTP 4. Port 8000, model name `qwen3.8-flash-next`.

One user, thinking off, temperature 0.7, 300 tokens, median of 5 (`spec.py`): JSON 62.4 tok/s,
code 58.0, explain 46.5, rewrite 44.2, story 33.6. On random text, where the MTP draft rarely
matches, decode is 21.3 tok/s.

The first start compiles vLLM's, Triton's and b12x's kernels into `./cache/qwen3.8-flash-next-fp4`
and takes 17 min; later starts take about 5 min.

```bash
./run-recipe.sh qwen3.8-flash-next-fp4 --setup
```

`run-recipe.sh` takes a new key, `io_uring: true`: b12x reads the weights and the PLE table through
io_uring with locked buffers, which Docker's default seccomp profile blocks.

### 2026-10-22

#### FastH3-QAD-v2 on FastVideo: 10 s of video with sound

Added a recipe for `FastVideo/FastH3-QAD-v2` (an NVFP4 DiT trained with QAD, with its NVFP4 text
encoder, light video VAE and audio VAE) on FastVideo d4f1445, with FastVideo's own DGX Spark config
and the environment variables its header lists. It runs in the foreground and writes the MP4 and
FastVideo's stage log to `./out`.

By default it makes 10 s of 1344×768 video (243 frames) from a two-shot prompt with speech, rain and
music, seed 0: 489 s (8.1 min) with the kernels cached, of which 392 s of denoising (8 DiT
steps) and 96 s of video decoding. Peak memory 46.6 GiB allocated. The first run also compiles FlashInfer's FP4 kernels (about 3.5 min) into
`./cache/fasth3-qad`; later runs reuse them. `-- --ref` runs FastVideo's own request (832×480,
124 frames); `-- --prompt-file out/<file>` takes your prompt from `./out`.

```bash
./run-recipe.sh fasth3-qad --setup
```

#### Drama 3 on SGLang-Omni: text to speech

Added a server recipe for `fishaudio/drama-3` (35B MoE, 3B active) on SGLang-Omni 22d4ace, built for
the GB10: upstream's Dockerfile without its amd64-only kernel cache, with OpenFst built from source
for arm64. The first start compiles FlashInfer's SM121 kernels into `./cache/drama-3`.

`scripts/drama-tts.py` sends a sentence and a paragraph to `/v1/audio/speech`, one at a time:
5.2 s of speech in 4.8 s (1.07× real time) and 30.4 s in 27.3 s (1.11×). The wav files go to `./out`.

SGLang's default memory share (0.70) gave the KV cache 47 GiB and left 1 GiB free; the recipe sets
`--tts_engine.gpu_memory_fraction 0.45`, and about 42 GiB stays available.

```bash
./run-recipe.sh drama-3 --setup
python3 scripts/drama-tts.py
```

`run-recipe.sh` now keeps FlashInfer's JIT kernels per recipe in `./cache/<recipe>` (`cache: true`)
and applies `memory` to script recipes too.

#### Nemotron 3.5 Bolt on vLLM, with and without NVIDIA's draft

Added two server recipes for `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (74B MoE, 9B active,
native 2-bit QAD, 23 GiB) with NVIDIA's flags, text only. Port 8000, model name `bolt`.

Without the draft: 29 tok/s decode, one user. With the LiLiCorr draft (DFlash, 7 tokens),
temperature 0, 300 tokens: code 94 tok/s, JSON 93, rewrite 104, story 48. Translation to Turkish
is slower with the draft (26 tok/s), because the draft is accepted less often.

```bash
./run-recipe.sh nemotron-3.5-bolt --setup
./run-recipe.sh nemotron-3.5-bolt-lilicorr --setup
```

#### FLUX 3 Klein on ComfyUI, BF16 and NVFP4

Added a ComfyUI server recipe with BFL's own files (`flux-3-image-klein-comfy`): BF16 and NVFP4
in one server, on NVIDIA's PyTorch for the GB10. Open `http://127.0.0.1:8000` for the UI.

`scripts/klein-bench.py` makes 24 images (2 formats, 1 and 4 MP, 3 prompts, 2 seeds, 16 steps)
through the API with BFL's text-to-image graph. Median time per image:

| | 1 MP | 4 MP |
|---|---|---|
| BF16 | 10.6 s | 55.9 s |
| NVFP4 | 7.2 s | 44.1 s |

Only the linear layers are 4-bit in NVFP4; attention stays BF16, so the gain is smaller at 4 MP.
NVFP4 keeps the scene but not every detail of the BF16 image.

```bash
./run-recipe.sh flux-3-klein --setup
python3 scripts/klein-bench.py
```

`run-recipe.sh` now takes a draft model (`draft`, `draft_revision`) and the engine's own
listen flags (`serve`) in a recipe.

### 2026-10-21

#### Ornith-1.5-122B NVFP4 on vLLM

Added a server recipe for the lab's own NVFP4 build (ModelOpt, 78 GB), text only, on vLLM
nightly 77154ca. Same API as the Q4_K_M recipe: port 8000, model name `ornith`.

One user, warm: 26.5 tok/s decode, prefill about 2,450 tok/s (a 7.6k-token prompt in 3.0 s;
Q4_K_M on llama.cpp: about 790 tok/s).
vLLM reserves about 110 GiB for weights, FP8 KV cache and graphs; the container is capped at 112g.
Quality with thinking off, against the Q4_K_M build on the same questions: GPQA Diamond 70.2%
(Q4_K_M 74.2%, difference within noise), IFEval prompt-strict 82.0% (Q4_K_M 76.5%, +5.5,
95% CI +0.5 to +10.5).

Server recipes now run with `--ipc host`, which vLLM needs for its shared memory.

```bash
./run-recipe.sh ornith-1.5-122b-nvfp4 --setup
```

### 2026-10-20

#### Ornith-1.5-122B on one Spark, llama.cpp built for GB10

Added a server recipe for Ornith-1.5-122B-A10B (released today, MIT) with the lab's own Q4_K_M GGUF
(76 GB). It serves an OpenAI-compatible API on port 8000, bound to 127.0.0.1 unless you set `HOST`.

One user, warm: 28.2 tok/s decode, prefill about 790 tok/s (a 6.7k-token prompt in 8.5 s), about 80 GiB in use.
(Corrected 2026-10-21: this line first said an 8k-token prompt; the benchmark had cut the prompts to ~6.7k.)
GPQA Diamond 74.2% with thinking off (simple-evals format, 198 questions, 95% CI 67.7 to 79.8).
An independent run on the lab's API with thinking on reported 85.5.

The prebuilt llama.cpp CUDA image (CUDA 12.8, compiled up to sm_120) decoded this model at
1.8 tok/s on the GB10, with the GPU mostly idle. The recipe builds llama.cpp v0.7.0 with CUDA 13
for sm_121 instead; which of those changes matters most was not isolated. It also loads the weights
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
