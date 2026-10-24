#!/usr/bin/env bash
# Run one recipe on a DGX Spark.
#   ./run-recipe.sh <recipe> [--setup] [-- script args]
#   --setup   build the image and download the model at the pinned revision first
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 <recipe> [--setup] [-- script args]"
  find recipes -name '*.yaml' | sort | sed 's|^recipes/||; s/\.yaml$//; s/^/  /'
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
TYPE=$(get type)
FILE=$(get file)
DRAFT=$(get draft)
DRAFT_REV=$(get draft_revision)
PREPARE=$(get prepare)
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
OUT=${OUT:-$PWD/out}
HOST=${HOST:-127.0.0.1}  # set HOST=0.0.0.0 to serve your LAN
PORT=${PORT:-8000}

# offline by default; "online: true" lets the script download data (e.g. calibration sets)
ENV=(-e HF_HUB_OFFLINE=1)
[ "$(get online)" = true ] && ENV=(-e HF_HOME=/out/.hf-cache -e HF_TOKEN_PATH=/hf/token)
MOUNTS=()
[ -n "$CONFIG" ] && MOUNTS=(-v "$PWD/$CONFIG:/config.yaml:ro")
NAME=$(basename "$RECIPE" .yaml)
# "cache: true" keeps FlashInfer's SM121 kernels in ./cache/<recipe>: they compile once,
# and MAX_JOBS=4 keeps the compile from running the box out of memory
CACHE=()
if [ "$(get cache)" = true ]; then
  mkdir -p "cache/$NAME"
  CACHE=(-v "$PWD/cache/$NAME:/cache" -e FLASHINFER_WORKSPACE_BASE=/cache -e MAX_JOBS=4)
fi
MEMORY=$(get memory)
# "io_uring: true": b12x reads the weights and the PLE table through io_uring with locked
# buffers, which Docker's default seccomp profile blocks
URING=()
[ "$(get io_uring)" = true ] && URING=(--security-opt seccomp=unconfined --ulimit memlock=-1:-1)
# "home: true": the engine's home (signing key, trust store, prepared package) lives in
# ./cache/<recipe>, with the HF hub mounted inside it read-only
HOMEDIR=()
if [ "$(get home)" = true ]; then
  mkdir -p "cache/$NAME"
  HOMEDIR=(-v "$PWD/cache/$NAME:/root" -v "$HF_HOME/hub:/root/.cache/huggingface/hub:ro" -e HF_HUB_OFFLINE=1)
fi

if [ "$SETUP" = 1 ]; then
  # a recipe with "build:" makes its image here; one without it runs a published image
  if [ -n "$BUILD" ]; then docker build -t "$IMAGE" "$BUILD"; else docker pull "$IMAGE"; fi
  HF_HOME=$HF_HOME hf download "$MODEL" ${FILE:+"$FILE"} --revision "$REV"
  if [ -n "$DRAFT" ]; then
    HF_HOME=$HF_HOME hf download "$DRAFT" --revision "$DRAFT_REV"
  fi
  # "prepare": a one-time step before the first start (Flare exports its package here)
  if [ -n "$PREPARE" ]; then
    # shellcheck disable=SC2086  # a flat command line from the recipe
    docker run --rm --gpus all --ipc host --memory "$MEMORY" \
      ${HOMEDIR[@]+"${HOMEDIR[@]}"} ${URING[@]+"${URING[@]}"} "$IMAGE" $PREPARE
  fi
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE, run with --setup"; exit 1; }

if [ "$TYPE" = server ]; then
  # long-running: detached, named after the recipe, model path substituted into the engine's args
  snap() { echo "/hf/hub/models--${1//\//--}/snapshots/$2"; }
  ARGS=$(get args)
  ARGS=${ARGS//\{model\}/$(snap "$MODEL" "$REV")${FILE:+/$FILE}}
  ARGS=${ARGS//\{draft\}/$(snap "$DRAFT" "$DRAFT_REV")}
  # Flare starts unattended only with its package's signing key approved: {dev_key} is the
  # fingerprint of the local development key the export made, the sha256 of its public key
  if [[ $ARGS == *"{dev_key}"* ]]; then
    KEY=$(docker run --rm ${HOMEDIR[@]+"${HOMEDIR[@]}"} "$IMAGE" python3 -c 'import hashlib, json
k = json.load(open("/root/.config/flare/development-publisher.json"))[0]["public_key"]
print(hashlib.sha256(bytes(k)).hexdigest())' | tail -1)
    ARGS=${ARGS//\{dev_key\}/$KEY}
  fi
  # vLLM and llama.cpp take --host/--port; ComfyUI sets its own in "serve"
  SERVE=$(get serve)
  # shellcheck disable=SC2086  # args are a flat flag list from the recipe
  docker run -d --init --name "$NAME" --gpus all --ipc host --memory "$MEMORY" \
    -p "$HOST:$PORT:8000" -v "$HF_HOME:/hf:ro" ${MOUNTS[@]+"${MOUNTS[@]}"} \
    ${CACHE[@]+"${CACHE[@]}"} ${URING[@]+"${URING[@]}"} ${HOMEDIR[@]+"${HOMEDIR[@]}"} \
    "$IMAGE" $ARGS ${SERVE:---host 0.0.0.0 --port 8000}
  echo "started $NAME on http://$HOST:$PORT  logs: docker logs -f $NAME  stop: docker rm -f $NAME"
  exit 0
fi

mkdir -p "$OUT"
docker run --rm --gpus all --ipc host ${MEMORY:+--memory "$MEMORY"} \
  ${CACHE[@]+"${CACHE[@]}"} --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" -e HOME=/tmp "${ENV[@]}" ${MOUNTS[@]+"${MOUNTS[@]}"} \
  -v "$HF_HOME:/hf:ro" -v "$PWD/$SCRIPT:/run.py:ro" -v "$OUT:/out" \
  "$IMAGE" python /run.py --model "$MODEL" --rev "$REV" "$@"
