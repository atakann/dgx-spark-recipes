# SGLang-Omni for Fish Audio Drama 3 on a DGX Spark (GB10, arm64, CUDA 13). No weights inside.
# Upstream docker/Dockerfile @22d4ace, five changes:
#  1. no flashinfer-cache stage: that image is amd64-only and holds only a SM89/SM90 JIT cache;
#     FlashInfer compiles for SM121 on the first start instead
#  2. sglang-omni cloned at a pinned commit at build time, not pulled from main at every start
#  3. only the fishaudio extra (no llama.cpp / CosyVoice builds)
#  4. OpenFst 1.8.3 built from source: pynini (pulled in by two TTS text frontends) has no
#     aarch64 wheel and compiles against it
#  5. the S2 Pro playground opens with Drama 3's sampling defaults (temperature 1.0, top_p 0.9;
#     sglang_omni/models/fishaudio_drama3/speech_request.py) instead of S2 Pro's 0.8 / 0.8
ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469
FROM ${SGLANG_IMAGE}
ARG OMNI_COMMIT=22d4ace88f9547777b819a69d7691204bb9ce847
ARG UCX_COMMIT=d8e50df6651b9ea5b76f23aee0aefbf053a4137a

RUN apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*

# upstream's UCX build, unchanged
RUN git clone --filter=blob:none https://github.com/openucx/ucx.git /tmp/ucx \
    && git -C /tmp/ucx checkout "${UCX_COMMIT}" \
    && cd /tmp/ucx \
    && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=/usr/local/cuda \
        --with-verbs \
        --with-dm \
        --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip \
    && ldconfig \
    && rm -rf /tmp/ucx

ARG OPENFST_VERSION=1.8.3
RUN curl -fsSL https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz \
      | tar -xz -C /tmp \
    && cd /tmp/openfst-${OPENFST_VERSION} \
    && ./configure --enable-grm --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/openfst-${OPENFST_VERSION}

RUN git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git /opt/sglang-omni \
    && git -C /opt/sglang-omni checkout "${OMNI_COMMIT}"
RUN cd /opt/sglang-omni \
    && python3 -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml","rb"))["tool"]["uv"]["override-dependencies"]))' > /tmp/ov.txt \
    && uv pip install --python /opt/sglang/bin/python3 --no-build-isolation --overrides /tmp/ov.txt \
         -r pyproject.toml --extra fishaudio \
    && uv pip install --python /opt/sglang/bin/python3 --no-deps --reinstall flashinfer-python==0.6.18 \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && python3 -m pip install --no-deps --no-build-isolation -e /opt/sglang-omni \
    && rm /tmp/ov.txt

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0
# llama.cpp must load the same NCCL as Torch, not the older system copy (upstream)
ENV LD_LIBRARY_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}
WORKDIR /opt/sglang-omni

# change 5: Drama 3's sampling defaults in the S2 Pro playground (Top K is 30 in both)
RUN sed -i -e '/label="Temperature"/,/value=/ s/value=0.8/value=1.0/' \
           -e '/label="Top P"/,/value=/ s/value=0.8/value=0.9/' \
    /opt/sglang-omni/playground/s2pro/ui.py
