"""ONNX policies: the ``topple.policy/1`` format (docs/design/phase-2.md §3, contract §17).

- ``build``: the torch-free graph builder for first-party policies (§3.2);
- ``validate``: the acceptance policy, canonical form and smoke test for untrusted bytes (§3.4);
- ``manifest``: ``policy.json``, parsed as hostile input, and its checks against the model and
  the challenge (§3.1);
- ``runtime``: policy directories and bytes to a ``LoadedPolicy``, and the one place an ONNX
  Runtime session is created, from validated bytes (§3.5).

Nothing here imports torch: eval and CI never need it.
"""
