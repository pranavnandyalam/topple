"""ONNX policies: the ``topple.policy/1`` format (docs/design/phase-2.md §3).

- ``build``: the torch-free graph builder for first-party policies (§3.2);
- ``validate``: the acceptance policy, canonical form and smoke test for untrusted bytes (§3.4);
- ``runtime``: the one place an ONNX Runtime session is created, from validated bytes (§3.5).

Nothing here imports torch: eval and CI never need it.
"""
