#!/usr/bin/env bash

# Run from project root, or set PROJECT_DIR explicitly.
PROJECT_DIR="/media/thanush/ubuntu_project/vla/vla_stack"

export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

VENV_PATH="$PROJECT_DIR/.venv"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Virtual env not found at $VENV_PATH"
  echo "Create it with: uv venv $VENV_PATH"
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return 1
  else
    exit 1
  fi
fi

# Activate venv in current shell
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "VIRTUAL_ENV=$VIRTUAL_ENV"