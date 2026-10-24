# NVIDIA Flare for GB10 (SM121): Ignition prepares a runtime package, flare serves it.
# Upstream builds Ignition on the host; this puts the same locked environment in one image.
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98

# docs/model-preparation.md "Environment setup", plus nasm and libibverbs from the upstream Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev \
        python3-dev liburing-dev nasm libibverbs-dev git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# toolchains outside /root, so a mounted /root keeps only keys and caches
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:/opt/flare/.venv/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.94.1
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

ARG FLARE_REV=5f37bd14062334f7687d168d58ea00567b9aa7fe
RUN git clone https://github.com/NVIDIA/flare-local-inference /opt/flare \
    && git -C /opt/flare checkout $FLARE_REV
WORKDIR /opt/flare

# the env vars scripts/setup_preparation.py sets; 4 jobs to keep the build inside memory
RUN MAX_JOBS=4 CARGO_BUILD_JOBS=4 BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 \
    uv sync --locked --extra prepare --python /usr/bin/python3.12
RUN cargo build --locked --release -j 4 --manifest-path flare/Cargo.toml --bin flare \
    && install -m755 flare/target/release/flare /usr/local/bin/flare
