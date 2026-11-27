# vLLM for NVIDIA's MiMo-V2.6-Flash-RL NVFP4 on DGX Spark: a vLLM nightly (f2f9edcac, arm64), pinned by digest, plus
# the two model files of vllm-project/vllm#70196 (commit 9728c5ee9). The nightly's MiMo loader cannot read this build:
#   mimo_v2.py      loads the NVFP4 fused qkv_proj (pre-sharded at num_key_value_heads chunks, one per-tensor scale
#                   per Q, K and V) and puts the dense MLP's MiMo block-FP8 weight_scale_inv into ModelOpt's weight_scale
#   mimo_v2_mtp.py  the MTP draft gets its own quantization config and loads its FP8 scales the same way
# No weights inside.
FROM vllm/vllm-openai@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7
ARG V=/usr/local/lib/python3.14/dist-packages/vllm
COPY mimo_v2.py mimo_v2_mtp.py $V/model_executor/models/
# vLLM, Triton and FlashInfer keep their compile caches under $HOME/.cache; with "cache: true"
# that is ./cache/<recipe>, so only the first start compiles
ENV HOME=/cache
ENTRYPOINT ["vllm", "serve"]
