#!/usr/bin/env bash
# Run one recipe on a DGX Spark.
#   ./run-recipe.sh <recipe> [--setup] [-- script args]
#   --setup   build the image and download the model at the pinned revision first
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 <recipe> [--setup] [-- script args]"
  ls recipes | sed 's/\.yaml$//; s/^/  /'
  exit 1
}
[ $# -ge 1 ] || usage
RECIPE=recipes/$1.yaml
shift
[ -f "$RECIPE" ] || usage

SETUP=0
while [ $# -gt 0 ]; do
  case $1 in
    --setup) SETUP=1; shift ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

# recipes hold flat "key: value" lines only
get() { sed -n "s/^$1: *//p" "$RECIPE"; }
IMAGE=$(get image)
BUILD=$(get build)
MODEL=$(get model)
REV=$(get revision)
SCRIPT=$(get script)
CONFIG=$(get config)
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
OUT=${OUT:-$PWD/out}

# offline by default; "online: true" lets the script download data (e.g. calibration sets)
ENV=(-e HF_HUB_OFFLINE=1)
[ "$(get online)" = true ] && ENV=(-e HF_HOME=/out/.hf-cache -e HF_TOKEN_PATH=/hf/token)
MOUNTS=()
[ -n "$CONFIG" ] && MOUNTS=(-v "$PWD/$CONFIG:/config.yaml:ro")

if [ "$SETUP" = 1 ]; then
  docker build -t "$IMAGE" "$BUILD"
  HF_HOME=$HF_HOME hf download "$MODEL" --revision "$REV"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE, run with --setup"; exit 1; }

mkdir -p "$OUT"
docker run --rm --gpus all --ipc host \
  --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" -e HOME=/tmp "${ENV[@]}" ${MOUNTS[@]+"${MOUNTS[@]}"} \
  -v "$HF_HOME:/hf:ro" -v "$PWD/$SCRIPT:/run.py:ro" -v "$OUT:/out" \
  "$IMAGE" python /run.py --model "$MODEL" --rev "$REV" "$@"
