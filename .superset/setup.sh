#!/usr/bin/env bash
# Superset workspace setup: local guardrail/env files, Python env via uv, web deps once web/ exists.
set -euo pipefail

# Gitignored per-developer files (permission guardrails, .env) first, so a later failure never
# leaves a workspace without them. Never follow or overwrite symlinks.
main_checkout="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
if [[ -n "$main_checkout" && "$main_checkout" != "$PWD" && ! -L .claude ]]; then
  for f in .env .claude/settings.local.json; do
    src="$main_checkout/$f"
    if [[ -f "$src" && ! -L "$src" && ! -e "$f" && ! -L "$f" ]]; then
      mkdir -p "$(dirname "$f")"
      install -m 600 "$src" "$f"
    fi
  done
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Apple Silicon (including Rosetta shells): never let uv pick an x86_64 interpreter; onnxruntime has
# no Intel-mac wheels. pyproject's python-preference = "system" covers most setups; this is a guard.
if [[ "$(uname -s)" == "Darwin" && "$(sysctl -in hw.optional.arm64 2>/dev/null)" == "1" ]]; then
  export UV_PYTHON="cpython-$(tr -d '[:space:]' < .python-version)-macos-aarch64-none"
fi

uv sync --locked --all-extras

if [[ -f web/package-lock.json ]]; then
  npm --prefix web ci --no-audit --no-fund
fi
