# NVIDIA DeepStream 9.2 for DGX Spark (9.2-triton-sbsa-dgx-spark: CUDA 13.2.1, TensorRT 10.16.1), with
# our nvinfer parser for YOLO26's NMS-free head and a CPU-only Ultralytics for the ONNX export.
# No weights inside.
FROM ghcr.io/nvidia/deepstream@sha256:87014f626496737dc888eb80d5de9baf5a37c834033cabfbbcc50139c7183251
# run-recipe.sh starts the script with "python"; the image has python3 only
RUN ln -s /usr/bin/python3 /usr/local/bin/python
# Ultralytics only writes the ONNX here and nvinfer builds the TensorRT engine, so CPU torch is enough.
# pip warns that the image's tritonserver Python package wants numpy<2: the recipe runs deepstream-app
# with nvinfer, not Triton's Python API.
RUN pip install --no-cache-dir --break-system-packages --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1 torchvision==0.29.1 \
    && pip install --no-cache-dir --break-system-packages ultralytics==8.4.192 onnx==1.21.0 onnxslim==0.1.98
COPY yolo26_parse.cpp /opt/yolo26/
RUN g++ -O2 -std=c++17 -shared -fPIC -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda/include -o /opt/yolo26/libyolo26_parse.so /opt/yolo26/yolo26_parse.cpp
ENV YOLO_AUTOINSTALL=false YOLO_OFFLINE=true
