# Edge0 (Edge0-AI/Edge0) for the GB10: edge0-strata (the engine, CUDA, sm_121) and edge0-infer (the launcher, CPU),
# from source as edge0-strata/EDGE0.md "Build (Linux, NVIDIA)" and edge0-infer/docs/getting-started.md say.
# No weights inside. No prebuilt engine: setup.py is not used.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ARG EDGE0_SHA=257ff84ae8a89dbf34239dcf8803c91d07df94b8
RUN git clone https://github.com/Edge0-AI/Edge0.git /opt/edge0 && cd /opt/edge0 && git checkout "$EDGE0_SHA"
WORKDIR /opt/edge0/edge0-strata
# ggml for the engine: llama.cpp at the commit that EDGE0.md names
ARG LLAMACPP_SHA=3cf03257f219afbe7334045ff7c6a06ac68c627d
RUN mkdir -p third_party/llama.cpp \
    && curl -L https://github.com/ggml-org/llama.cpp/archive/$LLAMACPP_SHA.tar.gz \
       | tar -xz --strip-components=1 -C third_party/llama.cpp
# no GPU during the build, so the architecture is named: GB10 = compute capability 12.1 (as in Edge0 PR #199).
# lat_dense_d_tail_test is that PR's regression test for the prefill hang on the DGX Spark; it runs before a model.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
      -DCMAKE_CUDA_ARCHITECTURES=121 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
    && cmake --build build --target strata lat_dense_d_tail_test
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
WORKDIR /opt/edge0/edge0-infer
RUN cmake --preset cpu && cmake --build --preset cpu --target edge0-infer
RUN build/cpu/bin/edge0-infer info
