#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

DEFAULT_HF_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_CACHE_ROOT="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots"

resolve_default_model() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    printf '%s\n' "$MODEL_PATH"
    return 0
  fi

  if [[ -d "$DEFAULT_CACHE_ROOT" ]]; then
    local snapshot
    snapshot="$(find "$DEFAULT_CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [[ -n "$snapshot" ]]; then
      printf '%s\n' "$snapshot"
      return 0
    fi
  fi

  printf '%s\n' "$DEFAULT_HF_MODEL"
}

MODEL_PATH_RESOLVED="$(resolve_default_model)"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
QWEN_TTS_DISABLE_MLX="${QWEN_TTS_DISABLE_MLX:-1}"

echo "Starting Qwen3-TTS OpenAI API"
echo "ROOT_DIR=$ROOT_DIR"
echo "MODEL_PATH=$MODEL_PATH_RESOLVED"
echo "DEVICE=$DEVICE"
echo "DTYPE=$DTYPE"
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "REQUEST_TIMEOUT_SECONDS=$REQUEST_TIMEOUT_SECONDS"

exec env \
  PYTHONPATH="$ROOT_DIR" \
  UV_CACHE_DIR="$UV_CACHE_DIR" \
  QWEN_TTS_DISABLE_MLX="$QWEN_TTS_DISABLE_MLX" \
  uv run --active --no-sync python -m qwen_tts.openai_api \
  --model "$MODEL_PATH_RESOLVED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS" \
  --host "$HOST" \
  --port "$PORT"
