// nvinfer bbox parser for YOLO26's NMS-free head (Ultralytics ONNX export with nms=False).
// One output per image, [300, 6]: x1, y1, x2, y2, score, class in network input pixels (640x640
// letterbox). The one-to-one head gives one box per object, so nvinfer runs no NMS (cluster-mode=4).
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo26(std::vector<NvDsInferLayerInfo> const& layers,
    NvDsInferNetworkInfo const& net, NvDsInferParseDetectionParams const& params,
    std::vector<NvDsInferParseObjectInfo>& objects)
{
    if (layers.size() != 1 || layers[0].dataType != FLOAT || layers[0].inferDims.d[1] != 6)
        return false;
    const float* p = static_cast<const float*>(layers[0].buffer);
    const float w = net.width, h = net.height;
    for (unsigned i = 0; i < layers[0].inferDims.d[0]; ++i, p += 6) {
        const unsigned c = static_cast<unsigned>(p[5]);
        if (c >= params.numClassesConfigured || p[4] < params.perClassPreclusterThreshold[c])
            continue;
        const float x1 = std::clamp(p[0], 0.f, w), y1 = std::clamp(p[1], 0.f, h);
        const float x2 = std::clamp(p[2], 0.f, w), y2 = std::clamp(p[3], 0.f, h);
        if (x2 - x1 < 1 || y2 - y1 < 1)
            continue;
        NvDsInferParseObjectInfo o{};
        o.classId = c;
        o.left = x1;
        o.top = y1;
        o.width = x2 - x1;
        o.height = y2 - y1;
        o.detectionConfidence = p[4];
        objects.push_back(o);
    }
    return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);
