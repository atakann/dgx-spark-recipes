"""YOLO26s on DeepStream 9.2: N video streams in one pipeline. Runs inside the recipe image.

  ./run-recipe.sh vision/yolo26-deepstream                          # 16 streams of the 1080p sample clip
  ./run-recipe.sh vision/yolo26-deepstream -- --streams 8
  ./run-recipe.sh vision/yolo26-deepstream -- --source out/my.mp4   # your H.264/H.265 video, under ./out

Pipeline: NVDEC decode -> nvstreammux (batch N) -> nvinfer (TensorRT FP16, batch N, YOLO26's NMS-free
head through our parser) -> NvSORT tracker. Two runs of deepstream-app:
1. speed: the streams loop with no display and sync off, each as fast as it can. deepstream-app prints
   FPS per stream every 5 s; the first 3 prints are warm-up, the next 12 (60 s) are the result.
2. video: one pass of the clip through the same engine, a mosaic of the N streams with boxes and
   track IDs, H.264 from NVENC. One pass, because deepstream-app stopped by a signal leaves the MP4 open.
The first run exports the ONNX and builds the engine for N into out/yolo26-deepstream (a few minutes,
before the first FPS print); later runs with the same N reuse it.
"""

import argparse
import collections
import math
import re
import shutil
import signal
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

DS = "/opt/nvidia/deepstream/deepstream"
CLIP = f"{DS}/samples/streams/sample_1080p_h264.mp4"
WORK = Path("/out/yolo26-deepstream")
WEIGHTS = "yolo26s.pt"
WARMUP, MEASURE = 3, 12  # FPS prints, 5 s each
SMI = ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.decoder", "--format=csv,noheader,nounits"]

# Letterbox as in Ultralytics (keep aspect, center), conf 0.1 as in model.track(). No NMS
# (cluster-mode=4): the one-to-one head gives one box per object.
INFER = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={work}/yolo26s-dyn.onnx
model-engine-file={work}/yolo26s-dyn.onnx_b{n}_gpu0_fp16.engine
infer-dims=3;640;640
labelfile-path={work}/labels.txt
batch-size={n}
network-mode=2
num-detected-classes=80
gie-unique-id=1
network-type=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
scaling-filter=1
parse-bbox-func-name=NvDsInferParseYolo26
custom-lib-path=/opt/yolo26/libyolo26_parse.so

[class-attrs-all]
pre-cluster-threshold=0.1
"""

APP = """[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[source0]
enable=1
type=3
uri=file://{video}
num-sources={n}
gpu-id=0

[streammux]
gpu-id=0
live-source=0
batch-size={n}
batched-push-timeout=40000
width=1920
height=1080

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
config-file={work}/infer-b{n}.txt
# box colors by COCO class (R;G;B;A): person blue, bicycle and motorcycle green, car orange,
# bus and truck yellow; the tiles are too small for text labels
bbox-border-color0=0.16;0.47;0.84;1
bbox-border-color1=0.11;0.69;0.48;1
bbox-border-color2=0.92;0.41;0.2;1
bbox-border-color3=0.11;0.69;0.48;1
bbox-border-color5=0.93;0.63;0;1
bbox-border-color7=0.93;0.63;0;1

[tracker]
enable=1
gpu-id=0
tracker-width=960
tracker-height=544
ll-lib-file={ds}/lib/libnvds_nvmultiobjecttracker.so
ll-config-file={ds}/samples/configs/deepstream-app/config_tracker_NvSORT.yml
display-tracking-id=1

[tiled-display]
enable=1
rows={side}
columns={side}
width={width}
height={height}
gpu-id=0

{tail}"""

# speed run: no on-screen display, frames discarded, the clip loops
SPEED = """[osd]
enable=0

[sink0]
enable=1
type=1
sync=0

[tests]
file-loop=1
"""

# video run: boxes and track IDs, an H.264 MP4 from NVENC, one pass so the MP4 is closed at the end
VIDEO = """[osd]
enable=1
gpu-id=0
border-width=2
text-size=10
text-color=1;1;1;1;
text-bg-color=0.2;0.2;0.2;0.8
font=Sans
show-clock=0

[sink0]
enable=1
type=3
container=1
codec=1
enc-type=0
sync=0
bitrate=16000000
profile=0
output-file={path}

[tests]
file-loop=0
"""


def export(model, rev):
    """ONNX with a dynamic batch axis, once. nvstreammux sends a partial batch when a stream is late,
    and a static-batch engine fails on it. Output [B, 300, 6]: x1, y1, x2, y2, score, class."""
    onnx = WORK / "yolo26s-dyn.onnx"
    if onnx.exists():
        return
    from ultralytics import YOLO  # CPU only, for the export

    WORK.mkdir(parents=True, exist_ok=True)
    snap = Path("/hf/hub") / f"models--{model.replace('/', '--')}" / "snapshots" / rev
    m = YOLO(shutil.copyfile(snap / WEIGHTS, WORK / WEIGHTS))
    (WORK / "labels.txt").write_text("\n".join(m.names.values()) + "\n")
    Path(m.export(format="onnx", nms=False, dynamic=True, imgsz=640)).rename(onnx)


def sample(stop, out):
    while not stop.wait(1):
        gpu, dec = subprocess.run(SMI, check=True, capture_output=True, text=True).stdout.split(",")
        out.append((int(gpu), int(dec)))


def deepstream(config, prints_needed=0):
    """Run deepstream-app to its end, or with prints_needed stop it after that many FPS prints.
    Returns the prints after the warm-up and the GPU and NVDEC load during them."""
    p = subprocess.Popen(
        ["deepstream-app", "-c", str(config)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    tail, prints, load, stop = collections.deque(maxlen=40), [], [], threading.Event()
    for line in p.stdout:
        if "GStreamer-WARNING" not in line and "gst-plugin-scanner" not in line:
            tail.append(line)  # GStreamer's plugin warnings would push the real error out
        if not prints_needed or not line.startswith("**PERF:") or "FPS 0" in line:  # "FPS 0 (Avg)": header
            continue
        prints.append([float(x) for x in re.findall(r"([\d.]+) \(", line)])
        if len(prints) == WARMUP:
            threading.Thread(target=sample, args=(stop, load), daemon=True).start()
        if len(prints) == prints_needed:
            break
    stop.set()
    if prints_needed:
        p.send_signal(signal.SIGINT)
    try:
        p.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
    if len(prints) < prints_needed or (not prints_needed and p.returncode):
        raise SystemExit(f"deepstream-app ended early (exit {p.returncode}):\n{''.join(tail)}")
    return prints[WARMUP:], load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--source", help="a video under ./out, seen as /out/... in the container")
    args = ap.parse_args()

    export(args.model, args.rev)
    n = args.streams
    out = Path("/out") / f"{datetime.now(UTC):%Y%m%d-%H%M%S}_yolo26-deepstream"
    out.mkdir(parents=True)
    (WORK / f"infer-b{n}.txt").write_text(INFER.format(work=WORK, n=n))
    video = f"/out/{Path(args.source).name}" if args.source else CLIP
    common = {"video": video, "n": n, "side": math.ceil(math.sqrt(n)), "work": WORK, "ds": DS}
    speed, mosaic = out / "speed.txt", out / "video.txt"
    speed.write_text(APP.format(**common, width=1280, height=720, tail=SPEED))
    mosaic.write_text(APP.format(**common, width=1920, height=1080, tail=VIDEO.format(path=out / "mosaic.mp4")))

    prints, load = deepstream(speed, WARMUP + MEASURE)
    fps = [sum(s) / len(s) for s in zip(*prints)]  # per stream, mean of the measured prints
    gpu, dec = zip(*load)
    line = (
        f"yolo26s DeepStream 9.2, {n} streams: {min(fps):.1f} FPS per stream at the slowest,"
        f" {sum(fps) / n:.1f} mean, {sum(fps):.0f} in total; GPU {sum(gpu) / len(gpu):.0f}%,"
        f" NVDEC {sum(dec) / len(dec):.0f}%"
    )
    print(line, flush=True)
    (out / "result.txt").write_text(line + "\n")
    deepstream(mosaic)
    print(f"mosaic video in out/{out.name}/mosaic.mp4")


if __name__ == "__main__":
    main()
