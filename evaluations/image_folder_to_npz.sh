#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/root/autodl-tmp/imagenet/val"
OUTPUT_NPZ="/root/autodl-tmp/imagenet_val256.npz"
IMAGE_SIZE=256
# EXTRA_ARGS=()
EXTRA_ARGS=(--no_labels)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python evaluations/image_folder_to_npz.py \
  "${DATA_DIR}" \
  "${OUTPUT_NPZ}" \
  --image_size "${IMAGE_SIZE}" \
  "${EXTRA_ARGS[@]}"
