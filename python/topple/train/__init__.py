"""Training (docs/design/phase-2.md §2).

Torch-free (default CI, imported by the CLI and ``topple export``):

- ``config``: ``TrainConfig`` and ``configs/train/*.yaml`` loading (§2.1, §2.2);
- ``wrappers``: training seeds and the action-excess penalty (§2.1, §2.3);
- ``selection``: selection evals, eligibility, ranking, eval labels, parity rows and run flags
  (§2.4, §2.5), the pure logic the SB3 callbacks glue together.

The torch half (``learn``, ``snapshot``, ``callbacks``, ``resources``, ``surgery``; the
``train`` extra) is imported lazily, only by ``topple train``. Nothing in this package's
``__init__`` imports anything.
"""
