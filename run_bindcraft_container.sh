#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the BindCraft Jupyter Docker image.
# Usage:
#   ./run_bindcraft_container.sh [--gpu] [PORT]
# Examples:
#   ./run_bindcraft_container.sh             # maps localhost:8888
#   ./run_bindcraft_container.sh 9999        # maps localhost:9999
#   ./run_bindcraft_container.sh --gpu 8888  # enable GPU and map 8888

PORT=8888
GPU_FLAG=""

if [[ "${1:-}" == "--gpu" ]]; then
  GPU_FLAG="--gpus all"
  shift || true
fi

if [[ $# -ge 1 ]]; then
  PORT="$1"
fi

IMAGE=${IMAGE:-muratshagirov/jupyter-bindcraft-full:v1.9.2}
NAME=${NAME:-bindcraft-jupyter}

# Note: We run as the image's default user; files created in /workspace may be owned by root on host.
# If you prefer host UID/GID mapping, add: -u "$(id -u):$(id -g)" (may break some images that expect root)
exec docker run --rm -it \
  -p "${PORT}:8888" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ${GPU_FLAG} \
  --name "${NAME}" \
  "${IMAGE}"
