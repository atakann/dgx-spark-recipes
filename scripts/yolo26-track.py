"""YOLO26s on TensorRT FP16 with Ultralytics: detect and track one video. Runs inside the recipe image.

  ./run-recipe.sh vision/yolo26-tensorrt                          # DeepStream's 1080p 30 FPS sample clip
  ./run-recipe.sh vision/yolo26-tensorrt -- --source out/my.mp4   # your video, put under ./out

The first run exports yolo26s.pt to a TensorRT FP16 engine in out/yolo26-tensorrt (a few minutes);
later runs reuse it. The loop is the one in Ultralytics' tracking docs: read a frame, then
model.track(frame, persist=True) with ByteTrack. FPS counts decode and tracking, not the drawing and
writing of the output video. The video (H.264) and the result line go to ./out.
"""

import argparse
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

CLIP = "/opt/samples/sample_1080p_h264.mp4"
WEIGHTS = "yolo26s.pt"
WARMUP = 30  # frames before the clock starts: the first ones carry CUDA and TensorRT start-up


def engine(model, rev):
    """Export once into out/yolo26-tensorrt; the HF cache is read-only."""
    cache = Path("/out/yolo26-tensorrt")
    path = cache / "yolo26s-fp16.engine"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        snap = Path("/hf/hub") / f"models--{model.replace('/', '--')}" / "snapshots" / rev
        pt = shutil.copyfile(snap / WEIGHTS, cache / WEIGHTS)
        # half=True is TensorRT FP16 in this Ultralytics version (newer ones call it quantize=16)
        Path(YOLO(pt).export(format="engine", half=True, imgsz=640, batch=1)).rename(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True, help="the snapshot sha in the HF cache")
    ap.add_argument("--source", help="a video under ./out, seen as /out/... in the container")
    args = ap.parse_args()

    model = YOLO(engine(args.model, args.rev), task="detect")
    cap = cv2.VideoCapture(f"/out/{Path(args.source).name}" if args.source else CLIP)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path("/out") / f"{datetime.now(UTC):%Y%m%d-%H%M%S}_yolo26-track"
    out.mkdir(parents=True)
    raw = out / "mpeg4.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    frames, busy, objects = 0, 0.0, 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if frames >= WARMUP:
            busy += time.perf_counter() - t0
            objects += len(r.boxes)
        frames += 1
        writer.write(r.plot())
    writer.release()
    if frames <= WARMUP:
        raise SystemExit(f"read {frames} frames, need more than {WARMUP}")

    video = out / "yolo26s-track.mp4"
    cmd = ["ffmpeg", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"]
    subprocess.run([*cmd, str(video)], check=True)
    raw.unlink()
    n = frames - WARMUP
    line = (
        f"yolo26s TensorRT FP16 + ByteTrack, one stream: {n / busy:.1f} FPS (decode and track,"
        f" {n} frames after {WARMUP} warm-up), {objects / n:.1f} tracked objects per frame"
    )
    (out / "result.txt").write_text(line + "\n")
    print(f"{line}\nvideo in out/{out.name}/{video.name}")


if __name__ == "__main__":
    main()
