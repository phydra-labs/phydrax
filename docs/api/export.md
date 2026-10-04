# Export

Deployment utilities for saving learned inference functions.

## Mathematical complex parameter interchange

Phydrax holomorphic layers and potentials keep trainable state in explicit real
Cartesian leaves. `export_complex_parameters` presents those leaves as canonical
mathematical complex arrays; `import_complex_parameters` validates and splits a
compatible complex state back into the destination model without changing its
optimizer geometry or PyTree layout.

`export_complex_parameters` remains the intentionally parameter-only surface.
For continuation state, `prepare_complex_training_interchange` binds an exact
parameter architecture, optimizer treedef with explicit grouped leaf routes,
typed-key RNG tree, auxiliary-state tree, and training identity.
`export_complex_training_state` then retains mathematical complex first
moments, labeled Cartesian second-moment pairs, exact counters/discrete state,
typed key data and implementation, auxiliary arrays, and the update boundary.
No field-name heuristic, reseeding, projection, optimizer-geometry conversion,
or implicit precision narrowing occurs.

`write_complex_training_checkpoint` publishes this complete state as an atomic,
pickle-free, checksummed array archive with no executable/JAXPR/factor token.
`read_complex_training_checkpoint` requires the prepared destination templates
and rejects changed layouts, identities, paths, dtypes, or payload checksums.
Parameter-only interchange is not an alias for the full checkpoint surface.

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

## Complete complex training-state interchange

::: phydrax.export.ComplexOptimizerStateGroup

---

::: phydrax.export.ComplexOptimizerStateLayout

---

::: phydrax.export.RNGInterchangeState

---

::: phydrax.export.prepare_complex_training_interchange

---

::: phydrax.export.export_complex_training_state

---

::: phydrax.export.import_complex_training_state

---

::: phydrax.export.write_complex_training_checkpoint

---

::: phydrax.export.read_complex_training_checkpoint

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

## Host inference

External model tiers are declared by `phydrax.ExecutionCapabilities`. A host
runtime (ONNX Runtime, a compiled IREE module, or any framework-neutral callable)
executes outside JAX: it is host-only, so every call is admitted against its
capabilities before the runtime runs, and `jit`, `vmap`, `grad`, `jvp`, and `vjp`
raise `TypeError` without invoking it. Call such models eagerly with concrete
values. Host inference has no adjoint; `derivative_support` reports the
derivative-free alternatives (EKI, evolution strategies, POUNDERS, differential
evolution) and never selects one. Adjoint-capable providers use the staged
`phydrax.interchange.ExternalAdjointAction` instead.

`HostInferenceAdapter(runner, input_schema, output_schema, binding)` wraps a
runner that receives one host NumPy array per `ExternalTensorSpec` in
`input_schema` and returns one array per `output_schema` entry. Shapes and
dtypes must match exactly (nothing is cast), outputs must be finite, and results
are detached concrete JAX arrays. `transport="copy"` copies across the host
boundary; `transport="dlpack"` exchanges buffers without copying (inputs must
export `__dlpack__`, outputs alias the runtime's buffers). `binding` is the
`ArtifactBindingIdentity` of the loaded model.

Install `phydrax[onnx-inference]` for the ONNX Runtime loader. `load_onnx`
requires a caller-supplied model SHA-256 pin, reads fixed input and output
shapes and tensor types from the model as the schemas (symbolic dimensions are
refused), runs on the CPU execution provider, and binds the model digest,
schemas, and ONNX Runtime version. `phydrax[onnx-export]` stays export-only.

```text
exported = phx.export.save_onnx(model, "model.onnx", inputs=[(8, 3)])
runtime = phx.export.load_onnx(
    exported.path, trusted_model_sha256=trusted_digest, transport="dlpack"
)
prediction = runtime(features)  # eager only
```

::: phydrax.export.load_onnx

---

::: phydrax.export.HostInferenceAdapter

## StableHLO and IREE deployment

Install the matched optional compiler/runtime pair with `phydrax[iree]`.
`save_iree(...)` exports a deterministic callable returning one array or a
non-empty ordered tuple of arrays through `jax.export`, compiles the resulting
StableHLO module in process, validates each compiled output against native JAX,
and publishes a pickle-free directory containing a checksummed VMFB module and
canonical JSON manifest.

```text
artifact = phx.export.save_iree(
    model,
    "model.phxiree",
    inputs=[sample],
    input_names=["x"],
    output_names=["prediction"],
)
deployed = phx.export.load_iree(
    artifact.path,
    trusted_module_sha256=artifact.manifest.module_sha256,
)
prediction = deployed(sample)
```

The manifest binds exact positional input shapes and dtypes plus ordered output
names, shapes, and dtypes, target backend, runtime driver, JAX
calling-convention version, and the identical IREE compiler/runtime release.
Loading requires a caller-supplied module SHA-256 obtained from a trusted
out-of-band source; the bundle's self-declared checksum is not execution
authority. It also rejects checksum, version, input, and output ABI mismatches.
Runtime outputs are checked independently for shape, dtype, and finiteness;
validation records per-output absolute and relative native-parity errors. No implicit
casting or output packing occurs, so boolean and integer status arrays retain
their native dtypes.

A single-output executable returns one array, unchanged from the original
callable boundary. An executable with multiple outputs returns an ordered tuple
matching `output_names`. Outputs use static concrete shapes. `key` must be
`None`; stochastic inference must be converted to an explicitly deterministic
deployed function first. Like ONNX, IREE export is an inference boundary, not
serialization of a solver or training loop.

A loaded `IREEExecutable` is a host-only `"compiled-inference"` tier
(`IREEExecutable.capabilities`): it is refused under JAX transformations before
the module runs. `IREEArtifactManifest.binding_identity()` (also
`IREEExecutable.binding`) identifies the calling ABI, the module digest, and the
compiler, runtime, target, and driver.

`save_discrete_velocity_iree(...)` is the typed exception to the generic
model-only boundary: it compiles one frozen, fixed-shape smooth-compressible
D2V17 equilibrium, one-step, or fixed-horizon forward realization. Its ordered
outputs retain accepted f/g, boolean success and rollback, integer status and
first-failure step, and floating diagnostics as separate arrays. The contract
binds the spatial runtime, topology, material, quadrature, support, frozen
binding, numeric revision, step count, and backend. Training and reverse mode
remain unsupported.

::: phydrax.export.DiscreteVelocityIREEContract

---

::: phydrax.export.DiscreteVelocityIREEExportBundle

---

::: phydrax.export.save_discrete_velocity_iree

---

::: phydrax.export.save_iree

---

::: phydrax.export.load_iree

---

::: phydrax.export.IREEExportPolicy

---

::: phydrax.export.IREEArtifactManifest

---

::: phydrax.export.IREEExecutable

## Frozen atomistic energy, force, and stress

`save_atomistic_iree(plan, structure, units, path, ...)` freezes one native learned
atomistic model, prepared from a `phydrax.atomistic.NativeAtomisticProviderPlan`, into a
fixed-capacity IREE executable. `structure` is the reference geometry whose candidate
lifecycle epoch is frozen. Model weights, species, units, periodicity, capacities, and
the host-prepared candidate relation of that epoch (route identities, image shifts,
masks, and route topology) become immutable artifact data. Neighbor discovery and cache
lifecycles stay on the host; training and neighbor lifecycles are not exported.

The ABI, recorded in `AtomisticIREEContract`, is:

| Inputs | Outputs |
|---|---|
| `positions` (wrapped) | `energy`, `forces`, `atom_energy` |
| `cell_vectors`, `image_counts` (periodic systems only) | `stress` (only when the provider has stress) |
| | `status` (`phydrax.atomistic.AtomisticStatus`), `active_routes` |

The module evaluates the frozen routes in their reference frame
`positions + (image_counts - reference_image_counts) @ cell_vectors`, the same
whole-lattice re-expression as the native lifecycle, so rewrapped atoms keep their
physical routes. `AtomisticIREEContract.pack_inputs(request)` refuses a request from
another lifecycle epoch, provider, or failed host lifecycle; a rebuilt candidate graph
requires a new export. `active_routes` counts the frozen graph's stored candidate
routes, and returned values are NaN unless the status is `SUCCESS`.

Runtime geometry failures (non-finite coordinates, singular or deformed cells, image
certificate or capacity overflow) are reported through the traced status, never through
host callbacks. Before export the parent refuses any host callback, any Equinox guard
whose predicate depends on a runtime input, and host-runtime custom calls (such as
LAPACK) that IREE cannot compile. Remaining guards depend only on frozen data and are
proven inactive by a raise-mode evaluation; tracing then runs in a pinned isolated
worker with `EQX_ON_ERROR=nan` (failure policy `"equinox-nan-status"`), whose
environment carries only `EQX_ON_ERROR`, `JAX_ENABLE_X64`, and `JAX_PLATFORMS`. The
worker rebuilds model, plan, structure, and contract from one pickle-free archive.
`path` is published atomically only after the parent's frozen program matches the native
provider at the reference geometry and the loaded module reproduces the frozen
program's values and statuses there and at declared failure probes, within `rtol`/`atol`.

The only `AtomisticExportRoute` is `"native-jax"`, the ordinary JAX program lowered with
`jax.export`. Accelerated kernels are not exportable and are never silently
substituted. A float64 model needs `IREEExportPolicy(executable_format="system-library")`:
the default embedded ELF carries no C math library, so float64 transcendentals cannot
link. The system library requires the host system linker at export time and loads only
on the same OS and architecture.

`load_atomistic_iree(path, trusted_module_sha256=..., trusted_contract_id=...)` requires
both out-of-band pins; a different contract raises `PermissionError`, and an executable
whose ABI differs from its contract is refused. `LoadedAtomisticIREE` is a host-only
`"compiled-inference"` executable: it is called eagerly with a lifecycle request,
refuses JAX transformations, and has no derivatives.

```text
bundle = phx.export.save_atomistic_iree(
    provider_plan,
    structure,
    units,
    "water-mace.phxiree",
    policy=phx.export.IREEExportPolicy(executable_format="system-library"),
)
frozen = phx.export.load_atomistic_iree(
    bundle.path,
    trusted_module_sha256=bundle.module_sha256,
    trusted_contract_id=bundle.contract.contract_id,
)
# `native.request` comes from provider_plan.prepare(system).evaluate_state(...)
# in the frozen lifecycle epoch.
result = frozen(native.request)
```

This frozen export is an unreleased candidate. Float64 parity of an exported MACE
executable is not claimed: a full-float64 tiny-MACE mismatch after StableHLO scatter
legalization is still being fixed, and no exported MACE artifact is published or
qualified.

::: phydrax.export.save_atomistic_iree

---

::: phydrax.export.load_atomistic_iree

---

::: phydrax.export.prepare_atomistic_iree_contract

---

::: phydrax.export.AtomisticIREEContract

---

::: phydrax.export.AtomisticIREEExportBundle

---

::: phydrax.export.AtomisticIREEEvaluation

---

::: phydrax.export.LoadedAtomisticIREE

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

