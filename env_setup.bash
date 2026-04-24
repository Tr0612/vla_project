#!/usr/bin/env bash

# By default, use ./vla_stack from the current directory.
# You can still override by exporting PROJECT_DIR before sourcing this file.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"

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
