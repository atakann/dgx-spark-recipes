# ComfyUI v0.41.0 (FLUX 3 Image support) on NVIDIA's PyTorch for the GB10. No weights inside.
FROM nvcr.io/nvidia/pytorch:26.07-py3
RUN git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout 98cc0bf84b7cb1383654c5b402e7612d875d6d22
# NVIDIA's torch stays: drop torch/torchvision from the list, PIP_CONSTRAINT off for the rest
RUN grep -v -E '^(torch|torchvision)$' /opt/ComfyUI/requirements.txt > /tmp/req.txt \
 && PIP_CONSTRAINT= pip install --no-cache-dir -r /tmp/req.txt
# fail the build if pip replaced NVIDIA's torch (the GB10 build) with a generic wheel
RUN python -c "import torch; assert 'nv26.07' in torch.__version__, torch.__version__"
WORKDIR /opt/ComfyUI
ENTRYPOINT ["python", "main.py"]
