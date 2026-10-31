# Ultralytics 8.4.192 on NVIDIA's PyTorch for the GB10: the ultralytics/ultralytics:latest-nvidia-arm64
# image of 2026-10-29 (NGC PyTorch 26.08, TensorRT 11.2.1.2, ModelOpt 0.45.0). No weights inside.
FROM ultralytics/ultralytics@sha256:97baf8bf16742e6be7bfba5239bbe3b1ffbccf8d8fb434d54282ce15ddc466f7
# TensorRT 11 export bakes FP16 into the ONNX with ModelOpt, and the image lacks ModelOpt's onnx extra.
# The full extra would put onnxruntime over the image's onnxruntime-gpu, so add only the missing
# packages. lap: ByteTrack's matching.
RUN pip install --no-cache-dir --no-deps onnx-graphsurgeon==0.6.1 onnxconverter-common==1.16.0 \
    cppimport==26.4.17 lief==1.0.0 mako==1.4.3 \
    && pip install --no-cache-dir lap==0.5.13
# OpenCV here cannot write H.264; ffmpeg converts its MPEG-4 output so the video plays anywhere
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# The test clip of both vision recipes: DeepStream's 1080p 30 FPS H.264 sample (1443 frames),
# from the DeepStream image that vision/yolo26-deepstream builds on
COPY --from=ghcr.io/nvidia/deepstream@sha256:87014f626496737dc888eb80d5de9baf5a37c834033cabfbbcc50139c7183251 \
    /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /opt/samples/
# the recipe runs as your user, offline, with nothing to install at run time
ENV YOLO_AUTOINSTALL=false YOLO_OFFLINE=true
# The script runs as /run.py, so Python would put / first on its path and import the source checkout
# /ultralytics as an empty namespace package; PYTHONSAFEPATH keeps the script's folder off the path
ENV PYTHONSAFEPATH=1
