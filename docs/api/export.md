# Export

Deployment utilities for saving learned inference functions.

## Mathematical complex parameter interchange

Phydrax holomorphic layers and potentials keep trainable state in explicit real
Cartesian leaves. `export_complex_parameters` presents those leaves as canonical
mathematical complex arrays; `import_complex_parameters` validates and splits a
compatible complex state back into the destination model without changing its
optimizer geometry or PyTree layout.

Complex interchange is not a training checkpoint. It excludes optimizer state, RNG
state, schedules, normalization progress, and loop position. Constrained-potential
exports contain derived full frame coefficients; import verifies affine-set
membership and never silently projects invalid coefficients.

::: phydrax.export.ComplexInterchangeEntry

---

::: phydrax.export.ComplexInterchangeState

---

::: phydrax.export.ComplexImportPolicy

---

::: phydrax.export.export_complex_parameters

---

::: phydrax.export.import_complex_parameters

---

::: phydrax.export.frame_coefficients_to_complex

---

::: phydrax.export.complex_coefficients_to_frame

## ONNX deployment

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

## StableHLO and IREE deployment

Install the matched optional compiler/runtime pair with `phydrax[iree]`.
`save_iree(...)` exports one deterministic array-valued callable through
`jax.export`, compiles the resulting StableHLO module in process, validates the
compiled function against native JAX, and publishes a pickle-free directory
containing a checksummed VMFB module and canonical JSON manifest.

```text
artifact = phx.export.save_iree(
    model,
    "model.phxiree",
    inputs=[sample],
    input_names=["x"],
)
deployed = phx.export.load_iree(artifact.path)
prediction = deployed(sample)
```

The manifest binds exact positional input shapes and dtypes, one output shape and
dtype, target backend, runtime driver, JAX calling-convention version, and the
identical IREE compiler/runtime release. Loading rejects checksum, version,
shape, and dtype mismatches. No implicit casting occurs.

The initial surface supports one array output and static concrete input shapes.
`key` must be `None`; stochastic inference must be converted to an explicitly
deterministic deployed function first. Like ONNX, IREE export is an inference
boundary, not serialization of a solver or training loop.

::: phydrax.export.save_iree

---

::: phydrax.export.load_iree

---

::: phydrax.export.IREEExportPolicy

---

::: phydrax.export.IREEArtifactManifest

---

::: phydrax.export.IREEExecutable

## Portable uncertainty results

`phydrax.uq.export_result` writes native UQ results as pickle-free, checksummed archives
whose arrays can be inspected without reconstructing the model. This is distinct from
ONNX deployment: the archive preserves inference output and provenance, not an
executable solver.

Finite MAP candidate archives use kind `map_candidate_search`. They retain selected
position/parameters when valid and always retain finite-space layout, signature,
batching, method identity, exact evaluation counts, and explicit all-invalid evidence.
The live posterior problem and search configuration object are listed as excluded.

Bellman archives retain filtered modes, local covariances and information matrices,
curvature diagnostics, optimizer results, status masks, and cumulative
pseudo-log-likelihood. Rao--Blackwellized full-smoother archives retain nonlinear
paths, sampled particle indices, conditional linear means and covariances, lag-one
covariances, and the source filter/backward-simulation provenance.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import phydrax as phx

sensor_x = jnp.linspace(0.1, 0.9, 8)
source_basis = 0.5 * sensor_x * (1.0 - sensor_x)
observed_field = 4.0 * source_basis
likelihood = phx.uq.GaussianLikelihood(0.02)
parameter_space = phx.uq.ParameterSpace(
    {"source_strength": jnp.asarray(3.5)},
    priors={"source_strength": phx.uq.Normal(0.0, 3.0)},
)
posterior = phx.uq.PosteriorProblem(
    parameter_space,
    lambda parameters: jnp.sum(
        likelihood.log_prob(
            parameters["source_strength"] * source_basis,
            observed_field,
        )
    ),
)
result = phx.uq.find_map(posterior)

with TemporaryDirectory() as directory:
    result_path = phx.uq.export_result(
        result,
        Path(directory) / "source-inference.phxresult",
    )
    portable = phx.uq.read_result_archive(result_path)

assert portable.kind == "map"
```

::: phydrax.uq.export_result

---

::: phydrax.uq.read_result_archive

