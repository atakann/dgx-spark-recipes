# vLLM nightly 77154ca (NVIDIA's pin for Nemotron 3.5 Bolt) + b12x kernels. No weights inside.
FROM vllm/vllm-openai@sha256:a7deff2c216def98c193352aca50fb03a2ee29231355b3033dc3789009f6d23e
RUN pip install --no-cache-dir b12x==1.5.1
# NVIDIA's "vllm download-kernels" step, at build time so the server starts offline
RUN python3 -m flashinfer download-kernels
# trust_remote_code copies the model code into HF_MODULES_CACHE; the HF cache is mounted read-only
ENV VLLM_USE_V2_MODEL_RUNNER=1 MAX_JOBS=4 HF_MODULES_CACHE=/tmp/hf_modules
ENTRYPOINT ["vllm", "serve"]
