# diffusers on NVIDIA's PyTorch for the GB10 (aarch64, CUDA 13.3). No weights inside.
FROM nvcr.io/nvidia/pytorch:26.07-py3
# NGC pins its own packages through PIP_CONSTRAINT; these three need newer ones
RUN PIP_CONSTRAINT= pip install --no-cache-dir diffusers==0.41.0 transformers==5.19.0 accelerate
# fail the build if pip replaced NVIDIA's torch (the GB10 build) with a generic wheel
RUN python -c "import torch; assert 'nv26.07' in torch.__version__, torch.__version__"
WORKDIR /out
