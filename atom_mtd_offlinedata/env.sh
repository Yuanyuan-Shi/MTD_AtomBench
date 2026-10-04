#!/usr/bin/env bash
# Source this file from any directory. No credentials are loaded or printed.
ATOM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ATOM_ROOT
export ATOM_PYTHON="${ATOM_PYTHON:-$ATOM_ROOT/openpi/.venv/bin/python}"
export PYTHONPATH="$ATOM_ROOT/atom_mtd_offlinedata/runtime/python:$ATOM_ROOT/openpi/src:$ATOM_ROOT/openpi/packages/openpi-client/src:$ATOM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
