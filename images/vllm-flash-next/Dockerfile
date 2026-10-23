# vLLM for Qwen3.8-Flash-Next-FP4-Local: the community image NVIDIA's model card names for DGX Spark,
# pinned by digest (eugr/spark-vllm-b12x nightly-20261015), with the card's environment. No weights inside.
FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
ENV VLLM_PLE_TABLE_MEMORY=disk B12X_DISK_BACKEND=io_uring CUTE_DSL_ARCH=sm_121a \
    SAFETENSORS_FAST_GPU=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SSM_CONV_STATE_LAYOUT=DS \
    VLLM_USE_AOT_COMPILE=1 VLLM_USE_MEGA_AOT_ARTIFACT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    B12X_POLICY_MODE=auto VLLM_MXFP8_LM_HEAD=1 VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
    VLLM_B12X_MOE_FP4_FORCE_A16=1 B12X_W4A16_A4_PREFILL=1
# vLLM, Triton, FlashInfer and b12x keep their compile caches under $HOME/.cache; with "cache: true"
# that is ./cache/<recipe>, so only the first start compiles
ENV HOME=/cache
ENTRYPOINT ["vllm", "serve"]
