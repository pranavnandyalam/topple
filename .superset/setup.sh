#!/usr/bin/env bash
# Superset workspace setup: Python env via uv, web deps once web/ has a lockfile, local .env copy.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Apple Silicon: never let uv pick a Rosetta (x86_64) interpreter; onnxruntime has no Intel-mac wheels.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  export UV_PYTHON="cpython-3.12-macos-aarch64-none"
fi

uv sync --locked --all-extras

if [[ -f web/package-lock.json ]]; then
  npm --prefix web ci --no-audit --no-fund
fi

# .env is gitignored, so copy it from the main checkout when one exists there.
main_checkout="$(git worktree list --porcelain | awk 'NR == 1 && $1 == "worktree" { print $2 }')"
if [[ -n "$main_checkout" && "$main_checkout" != "$PWD" && -f "$main_checkout/.env" && ! -f .env ]]; then
  cp "$main_checkout/.env" .env
fi
