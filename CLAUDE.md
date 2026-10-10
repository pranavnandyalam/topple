# Topple

Read PLAN.md first. It is the source of truth for scope, phases, the Controller contract and the trust model.

- Python 3.12 via uv (`uv sync`, `uv run pytest`). MuJoCo (pinned 3.15.x) + Gymnasium + Stable-Baselines3. Package `topple-sim`, import/CLI `topple`.
- Web arena in `web/` (Vite + TypeScript + three.js + `@mujoco/mujoco`).
- Work phase by phase. A phase is done only when its "Done when" check passes AND the review gate (code, security, done-when verifier agents) finds no CRITICAL/HIGH issues.
- TDD: tests first, 80%+ coverage. Seeded runs must be deterministic per platform.
- Every controller (PID, ONNX policy, submission, coach output, firmware) implements the `Controller` protocol.
- Policies ship as ONNX; eval and CI never need torch.
- Ranked scores come only from CI. Controllers never run in the scorer's process.
- Training runs on CPU (SubprocVecEnv); GPU only for Phase 13 (Great Lakes / cloud).
