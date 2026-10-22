# FastVideo on the GB10, following its DGX Spark install guide (uv, CUDA 13, kernel built for sm_121).
# No weights inside.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ARG FASTVIDEO_SHA=d4f144581d6882d5bceb288ca13d767c054ca041
RUN git clone https://github.com/hao-ai-lab/FastVideo.git /opt/FastVideo \
    && cd /opt/FastVideo && git checkout "$FASTVIDEO_SHA" \
    && git submodule update --init --recursive fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk
WORKDIR /opt/FastVideo
ENV UV_PYTHON_INSTALL_DIR=/opt/python VIRTUAL_ENV=/opt/FastVideo/.venv PATH=/opt/FastVideo/.venv/bin:$PATH
RUN uv venv .venv --python 3.12 --python-preference only-managed --seed
# no GPU during the build, so name the arch and the torch backend (the guide's CI/Docker path)
# MAX_JOBS: the kernel build runs parallel nvcc jobs that can fill the 128 GB
RUN MAX_JOBS=4 UV_TORCH_BACKEND=cu130 TORCH_CUDA_ARCH_LIST=12.1a uv pip install -e .
# fastvideo itself needs a GPU to import (Triton probes the driver), so check it at run time
RUN python -c "import torch; print(torch.__version__, torch.version.cuda)"
