# Nemotron 3.5 Transcribe on vLLM 0.32.0 with the NeMo Speech plugin, set up the way the model card says
# (uv sync --locked from NeMo Speech's own uv.lock, Python 3.13). arm64, CUDA 12.9 devel base for JIT kernels.
# Built by run-recipe.sh --setup as dgx-spark-recipes/nemotron-transcribe:nemo-2ed70ab
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:933d999bafbc7afa344e0e600fec13a6c3c0a191225fcd65db22185679b1476f

ARG UV_VERSION=0.13.1
ARG UV_SHA256=9041c73c99fdc5892cf128140beb354eca9231d502068cc36e6cb53a8e59ad14
ARG NEMO_SHA=2ed70aba1104df1940af0d6a0b811d3e255f3053

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sfL -o /tmp/uv.tgz \
      https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz \
    && echo "${UV_SHA256}  /tmp/uv.tgz" | sha256sum -c - \
    && tar -xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 && rm /tmp/uv.tgz

WORKDIR /opt/nemo-speech
RUN git init -q . && git remote add origin https://github.com/NVIDIA-NeMo/Speech.git \
    && git fetch -q --depth 1 origin ${NEMO_SHA} && git checkout -q FETCH_HEAD
RUN uv venv --python 3.13 .venv \
    && UV_PROJECT_ENVIRONMENT=/opt/nemo-speech/.venv uv sync --extra asr --extra vllm --locked --no-install-project \
    && uv pip install --python .venv/bin/python --no-deps -e . \
    && uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93

ENV PATH=/opt/nemo-speech/.venv/bin:$PATH \
    VLLM_PLUGINS=nemo_speechlm \
    VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180 \
    VLLM_USE_V2_MODEL_RUNNER=0
