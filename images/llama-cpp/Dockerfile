# llama.cpp built for the GB10 (sm_121, CUDA 13). The prebuilt server-cuda image (CUDA 12.8, up to sm_120)
# decoded Ornith-1.5-122B at 1.8 tok/s on the GB10 with the GPU mostly idle; the cause was not isolated.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    libcurl4-openssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
ARG LLAMA_TAG=v0.7.0
RUN git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp /opt/llama.cpp
WORKDIR /opt/llama.cpp
# -j 6: each CUDA template unit takes a few GB to compile; six at once stays far below 121 GB.
# No GPU during the build, so libcuda.so.1 is missing; allow it to resolve at run time
# (the same linker flag as llama.cpp's own .devops/cuda.Dockerfile).
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j 6 --target llama-server
ENTRYPOINT ["/opt/llama.cpp/build/bin/llama-server"]
