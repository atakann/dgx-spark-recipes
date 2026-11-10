# Lightricks LTX-2 (LTX-2.5 pipelines) at a pinned commit, the README's install (uv sync --extra natten).
# torch 2.13 cu132 and natten have aarch64 wheels; uv's default Python is 3.15, which torchvision has no wheel for.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ARG LTX_SHA=4f521619ae0029e8fbb64d555030c2b121a33b4f
RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 && cd /opt/LTX-2 && git checkout "$LTX_SHA"
WORKDIR /opt/LTX-2
ENV UV_PYTHON=3.12 UV_PYTHON_INSTALL_DIR=/opt/python VIRTUAL_ENV=/opt/LTX-2/.venv PATH=/opt/LTX-2/.venv/bin:$PATH
RUN uv sync --extra natten
RUN python -c "import torch, natten; print(torch.__version__, torch.version.cuda, natten.__version__)"
