# Export

Deployment utilities for saving learned inference functions.

!!! note
    ONNX export is for a single learned function, not a full solver. A solver
    contains constraints, samplers, losses, optimizer state, and logging behavior;
    ONNX should represent the inference boundary you want to deploy.

Use `phx.export.save_onnx(...)` directly for any array callable, or
`trained.save_onnx("u", ...)` as solver sugar for a named ansatz function.

Key points:

- Pass explicit `inputs`, using the shape spec expected by `jax2onnx`.
- Use `key=None` for deterministic inference/export.
- Use `vectorize=True` when exporting a pointwise `DomainFunction` over a
  leading batch axis.
- Optional `preprocess` and `postprocess` callables are included in the exported
  JAX graph, so they must be JAX-compatible.

```text
result = trained.save_onnx(
    "u",
    "u.onnx",
    inputs=[("B", 6)],
    input_names=["x"],
    output_names=["y"],
    vectorize=True,
    preprocess=x_scaler.transform,
    postprocess=y_scaler.inverse_transform,
)
```

::: phydrax.export.save_onnx

---

::: phydrax.export.OnnxExportResult

## Portable uncertainty results

`phydrax.uq.export_result` writes native UQ results as pickle-free, checksummed archives
whose arrays can be inspected without reconstructing the model. This is distinct from
ONNX deployment: the archive preserves inference output and provenance, not an
executable solver.

Bellman archives retain filtered modes, local covariances and information matrices,
curvature diagnostics, optimizer results, status masks, and cumulative
pseudo-log-likelihood. Rao--Blackwellized full-smoother archives retain nonlinear
paths, sampled particle indices, conditional linear means and covariances, lag-one
covariances, and the source filter/backward-simulation provenance.

```python
phx.uq.export_result(result, "inference.phxresult")
portable = phx.uq.read_result_archive("inference.phxresult")
```

::: phydrax.uq.export_result

---

::: phydrax.uq.read_result_archive

