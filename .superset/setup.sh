#!/usr/bin/env bash
# Superset workspace setup: Python env via uv, web deps once web/ has a lockfile, local .env copy.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Apple Silicon: never let uv pick a Rosetta (x86_64) interpreter; onnxruntime has no Intel-mac
# wheels. pyproject's python-preference = "system" already covers this; this is a second guard.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  export UV_PYTHON="cpython-$(tr -d '[:space:]' < .python-version)-macos-aarch64-none"
fi

uv sync --locked --all-extras

if [[ -f web/package-lock.json ]]; then
  npm --prefix web ci --no-audit --no-fund
fi

# Gitignored per-developer files: copy them from the main checkout when present there.
main_checkout="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
if [[ -n "$main_checkout" && "$main_checkout" != "$PWD" ]]; then
  for f in .env .claude/settings.local.json; do
    if [[ -f "$main_checkout/$f" && ! -f "$f" ]]; then
      mkdir -p "$(dirname "$f")"
      cp "$main_checkout/$f" "$f"
    fi
  done
fi
