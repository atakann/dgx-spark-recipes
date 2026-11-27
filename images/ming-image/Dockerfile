# Ming-Image (inclusionAI) on DGX Spark: vLLM 0.32.0's arm64 image, pinned by digest, only for its torch with sm_121
# kernels; Ming-Image's code at d561511 and the packages its requirements.txt pins (torch and transformer-engine
# left out, the image's torch stays). No weights inside.
FROM vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
ARG MING=d5615110e3ae7d27f270eafe5556d669b2069971
RUN mkdir -p /opt/ming && curl -fsSL https://github.com/inclusionAI/Ming-Image/archive/$MING.tar.gz \
    | tar -xz --strip-components=1 -C /opt/ming
RUN pip install --no-cache-dir transformers==4.57.1 diffusers==0.36.0 accelerate==1.13.0 peft==0.17.0 \
    safetensors==0.7.0 tokenizers==0.22.2 \
 && python3 -c "import torch, transformers, diffusers; print(torch.__version__, transformers.__version__, diffusers.__version__)" \
 && ln -sf /usr/bin/python3 /usr/local/bin/python
ENTRYPOINT []
