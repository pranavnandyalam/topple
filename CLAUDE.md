# Robot Sim Lab

Read PLAN.md first. It is the source of truth for scope, phases, and the Controller contract.

- Python 3.12 via uv (`uv sync`, `uv run pytest`). MuJoCo + Gymnasium + Stable-Baselines3.
- Work phase by phase; a phase is done only when its "Done when" check passes.
- TDD: tests first, 80%+ coverage. Seeded runs must be deterministic.
- Every controller (PID, trained policy, submission) implements the `Controller` protocol.
- Training runs on CPU (SubprocVecEnv); GPU only for stretch phase S2.
