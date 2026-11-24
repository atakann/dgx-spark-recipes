# llama.cpp for Kolibri 1.1 on the GB10 (sm_121, CUDA 13): ggml-org/llama.cpp pull request #32891 at its pinned head,
# which adds the model (its sparse attention's lightning indexer included). Kolibri's own vLLM plugin
# (aleph-alpha-inference 2.0.0) accepts only SM90 and SM100 GPUs, so llama.cpp is the GB10 path until the PR merges.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    libcurl4-openssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
ARG LLAMA_PR=32891
ARG LLAMA_SHA=5f4d222898ddb12cf581edf1b7e841d80e0b2e17
WORKDIR /opt/llama.cpp
RUN git init -q && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch -q --depth 1 origin pull/${LLAMA_PR}/head && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${LLAMA_SHA}"
# 121a: the PR rounds to NVFP4 with the native E2M1 convert on arch >= 1000, which needs the "a" target.
# -j 6: each CUDA template unit takes a few GB to compile; six at once stays far below 121 GB.
# No GPU during the build, so libcuda.so.1 resolves at run time (llama.cpp's own .devops/cuda.Dockerfile flag).
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j 6 --target llama-server
ENTRYPOINT ["/opt/llama.cpp/build/bin/llama-server"]
