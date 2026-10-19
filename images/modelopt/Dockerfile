# NVIDIA ModelOpt 0.47.0 on NVIDIA's PyTorch for the GB10. No weights inside.
FROM nvcr.io/nvidia/pytorch:26.07-py3
# ModelOpt 0.47.0 wants transformers < 5.15; Qwen3.8 needs >= 5.8
RUN PIP_CONSTRAINT= pip install --no-cache-dir "nvidia-modelopt[hf]==0.47.0" transformers==5.14.1 \
    accelerate datasets compressed-tensors fire
# hf_ptq.py, the example NVIDIA's own checkpoints come from, at the same tag as the package
RUN git clone --depth 1 --branch 0.47.0 https://github.com/NVIDIA/Model-Optimizer /opt/modelopt
# fail the build if pip replaced NVIDIA's torch (the GB10 build) with a generic wheel
RUN python -c "import torch; assert 'nv26.07' in torch.__version__, torch.__version__"
