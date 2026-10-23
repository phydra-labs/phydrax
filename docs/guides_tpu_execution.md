# TPU execution

`phydrax[tpu]` selects the upstream JAX TPU stack: JAX 0.12.0, JAXlib 0.12.0,
and libtpu 0.0.50. There is no Phydrax-owned TPU companion, patched native
runtime, or second array API. Scientific operations use their canonical JAX
routes. The supported wheel host is native Linux x86_64 with glibc 2.31 or newer
and standard-GIL CPython 3.13. WSL, free-threaded builds, and other Python minor
versions are not admitted, even if an upstream wheel is available.
Cloud TPU creation, permissions, network configuration, and provisioning are
outside this package's scope. Start on an already configured TPU worker.

An install, host assessment, or device probe is **not scientific hardware
qualification**. Hardware qualification remains pending until the complete
installed-wheel qualification campaign succeeds on the actual target TPU.
CPU runs and synthetic policy facts are not TPU execution evidence.

## Installation and explicit runtime selection

Install into a fresh environment on the supported worker:

```sh
python -m pip install 'phydrax[tpu]'
```

Select TPU before arrays, numerical imports, or device discovery initialize JAX:

```python
import phydrax as phx

phx.config.configure(phx.config.RuntimeConfig(platforms=("tpu",)))

from phydrax.backends import tpu_environment

host = tpu_environment()  # Host inspection only; the default never probes devices.
for issue in host.issues:
    print(issue.code, issue.severity, issue.message)
if not host.successful:
    raise RuntimeError("TPU installation was not admitted")

observed = tpu_environment(initialize_backend=True)
if not observed.successful or not observed.facts.device_probed:
    raise RuntimeError("No admitted TPU runtime was observed")
print(observed.facts.runtime_version)
print(observed.facts.devices)
print(observed.facts.local_device_count, observed.facts.global_device_count)
```

`JAX_PLATFORMS=tpu` is the equivalent launch selection. Keep CPU references in
a separate process with `JAX_PLATFORMS=cpu`; do not change platform selection
after a backend exists. Diagnostics never change selection. With an explicit
CPU selection, even `initialize_backend=True` leaves TPU unprobed and reports
`platform-not-selected`. With TPU selected, an unavailable TPU client or an
observed CPU device is a failure, not permission to silently fall back.

## Host evidence, overrides, and runtime identity

The public APIs separate collection, policy, and explicit discovery:

- `collect_tpu_environment()` reads host/ABI facts, canonical distribution/version
  pairs, effective x64 and matmul precision, `XLA_FLAGS`, `LIBTPU_INIT_ARGS`, and
  installed library identity. It does not call device discovery, import libtpu,
  or load a provider library/client.
  Kernel release and WSL evidence are recorded so WSL2 is not mistaken for an
  admitted native Linux host.
- `assess_tpu_environment(facts)` is pure: it assesses the supplied immutable
  record without consulting the machine, environment, metadata, or devices.
- `tpu_environment()` collects and assesses only.
  `tpu_environment(initialize_backend=True)` explicitly requests the TPU client
  after host admission and reports observed provider device identities,
  process-local/global device counts, process index/count, distributed readiness,
  client runtime version, and the mapped libtpu library path.

Runtime identity, devices, mapped libraries, counts, and backend errors require
`device_probed=True`; unprobed records cannot carry these observations.
`backends_initialized` and `distributed_initialized` may still be true in an
unprobed record: these report existing host lifecycle state, not a TPU probe.

JAX import has its own upstream Cloud TPU startup behavior. Device-inert
inspection promises no additional client initialization, not the absence of
JAX import-time environment or platform discovery effects. Flags are observed
as inputs; diagnostics do not assert that arbitrary flag combinations have
been qualified.

`TPU_LIBRARY_PATH` is honored as an explicit file selection, but a distribution
version is not the version of an unrelated shared library. Diagnostics record
the override, resolved selected file, SHA-256, and installed distribution path.
The wheel's `libtpu.so` must match its SHA-256 `RECORD` entry. An override whose
bytes match that wheel is attributable; different bytes have unknown version
and are refused. An invalid override is refused rather than silently accepting
upstream's default-library fallback. Unset the variable to use the wheel, or
point it to the exact wheel library. Phydrax does not bypass upstream JAX/libtpu
version checks.

The optional probe also requires the selected libtpu file to appear in Linux's
mapped-library evidence and records the client-provided runtime string. It does
not invent a native build ID, source ID, or owned build provenance for upstream
wheels. `.successful` means that the checks represented by `.facts` passed;
`.hardware_qualified` remains false even after a successful device probe.

## One host and multiple hosts

A one-host probe may create the local TPU client directly. Multiple devices on
that host do not by themselves mean multiple JAX processes. Before any device
discovery on **every** worker of an existing multi-host slice, let the caller
apply the repository's distributed bootstrap:

```python
import phydrax as phx

phx.config.configure(phx.config.RuntimeConfig(
    platforms=("tpu",),
    bootstrap=phx.config.RuntimeBootstrap(autodetect=True),
))

from phydrax.backends import tpu_environment

report = tpu_environment(initialize_backend=True)
if not report.successful or not report.facts.device_probed:
    raise RuntimeError("TPU runtime observation failed")
print(report.facts.process_index, report.facts.process_count)
print(report.facts.local_device_count, report.facts.global_device_count)
```

Use explicit `RuntimeBootstrap` coordinator fields instead when the launcher
requires them. Diagnostics never call `jax.distributed.initialize`, invent a
cluster, or reinitialize an existing cluster. Calling a probe before bootstrap
can initialize clients too early for later coordination; preserve that order.
Reuse native `ExecutionGroup` and JAX `NamedSharding` for distribution rather
than a TPU-specific execution abstraction. Local counts describe the current
process, global counts describe all devices visible to the TPU client, and
process count is separate from both. A one-process success does not qualify a
multi-host route or collective.

## Precision is a scientific contract

Phydrax already defaults matmul precision to `highest` when the caller has not
selected a precision. Do not replace it with a lower setting merely to admit a
TPU run. Effective precision is recorded, not silently rewritten. On TPU,
`highest` float32 contractions use **six BF16 passes** to approximate float32
products; this is not native IEEE float32 multiplication. Other TPU operations
and generations have their own rounding, underflow, accumulation, and supported
dtype behavior. Storage dtype alone is not an error bound.

`enable_x64=True` permits JAX wide values and protects exact wide indexing; it
is **not proof of native TPU FP64 arithmetic**, FP64 throughput, or scientific
FP64 qualification. Keep numerical roles explicit. Construct float32 geometry,
parameters, inputs, coefficients, optimizer state, and references deliberately
when selecting a float32 route; no diagnostic globally casts an existing
float64 model or artifact. Unsupported dtypes/operations must fail or remain
unqualified, not be silently downcast or moved to CPU. Preserve residual,
convergence, derivative-validity, masking, and resource evidence at consumer
boundaries.

Qualification must run a fresh installed wheel outside the source tree on
actual selected TPU devices, bind distribution/file/runtime identities, cover
the exact required cases without skipped passing cases, and compare against
independent references with declared tolerances. Model derivatives and higher
order derivatives, index width, FFT/complex arithmetic, solver status, replay,
and sharded results require their own evidence; host success does not imply
support for every operation. CUDA providers and CUDA/Pallas kernels are not
ported by this extra, nor does it claim new TPU Pallas kernels or optimization.

## Hardware-free lowering preflight

Before a TPU worker is available, check that bounded float32 scientific
kernels and their derivatives lower to TPU StableHLO. From the repository root,
in an environment with the pinned JAX (any host, CPU only):

```sh
python -m tools.check_tpu_lowering --output tpu-preflight.json
python -m tools.check_tpu_lowering --output sparse.json \
    --scenario sparse-route-gather-reduce
```

> **Boundary found by this preflight: bounded `linalg.solve` does not export.**
> Every jitted `phydrax.linalg.solve` route keeps fail-closed `eqx.error_if`
> runtime guards. These guards check the Krylov step bound against the prepared
> structural limit and require a non-negative operator numeric version. They
> lower to host callbacks, so the `bounded-pcg-solve` targets are declared
> `host-callback`. Their CPU references still run and must pass. The solver is
> not reported as TPU-exported or TPU-ready, and guards are not weakened to
> admit an export. Callbacks themselves are a TPU runtime feature, so this is an
> export limitation, not evidence that TPU cannot run the solve.

The tool selects `platforms=("cpu",)` before any backend exists and never
requests a TPU client. Each scenario builds its kernels with the native owner
and runs two stages over the **same** jitted functions:

- `cpu-reference` executes them on CPU and compares independent float64
  references and known results with declared tolerances: closed-form
  derivatives, central differences, Hessian-vector symmetry, Adam's first step,
  solver status, and exact masking or route identities. Every floating output
  must be finite float32.
- `tpu-export` calls `jax.export.export(jax.jit(f), platforms=("tpu",))` on
  abstract inputs. JVP, VJP, Hessian-vector, and mixed parameter/input
  derivatives are formed before export, not by transforming a rehydrated
  artifact. No export safety check is disabled. The record keeps the in/out
  avals, effects, custom-call targets, and the StableHLO byte count and SHA-256.

| Scenario | Native owner | Targets (expected outcome) |
| --- | --- | --- |
| `pinn-poisson-mlp` | `nn.models.MLP`, `laplacian`, `optim.adam` | Laplacian, value/gradient, JVP, HVP, Adam step (exported) |
| `measure-aware-attention` | `nn.layers.MeasureAwareAttention` | fused, blockwise, dense; gradients, query JVP/HVP, mixed (exported) |
| `prepared-contraction` | `tensor_network.prepare_contraction` | execute, reverse, JVP, JVP of the reverse schedule (exported) |
| `sparse-route-gather-reduce` | `sparse.EdgeRelation` | fast/deterministic apply, route max, VJP, JVP, mixed (exported) |
| `bounded-pcg-solve` | `linalg.solve` with `PCG` | success, step exhaustion status, right-hand-side JVP/VJP (`host-callback`) |
| `ordered-host-callback` | `jax.debug.callback` | callback loop (`host-callback`), callback-free loops (exported) |

Every target declares its expected outcome up front. Outcomes are `exported` or
a refusal: `host-callback`, `unstable-custom-call`, `lowering-unsupported`, or
`lowering-error`. A refusal is an export observation, not a TPU runtime
failure. Host callbacks need a live target client during lowering, and export
cannot serialize them. A refusal never satisfies a required export, and a
target that the CPU stage never built is reported as `not-attempted`.

`successful` means that every CPU check passed and every target had its declared
outcome. `fully_exported` is separate and is true only when every declared
target exported. It is false for any selection that includes
`bounded-pcg-solve` or `ordered-host-callback`. Report counts list `exported`,
`refused`, `required_exports`, `expected_refusals`, `matched_exports`,
`matched_refusals`, `not_attempted`, and `mismatched` targets, per scenario and
in total.

Every exported target must also be free of float64 work.
`strong_float64_primitives` lists the traced operations that produce strongly
typed float64 or complex128 values. Any entry fails a float32 route. Weakly
typed Python scalars are not float64 work: under x64, `jnp.where(mask, x, 0.0)`
carries an `f64` scalar constant that is immediately converted to float32. The
raw `module_f64_type_occurrences` count keeps that text evidence separately.
This audit does not prove that a module is legal for TPU at runtime.

The exit status is 0 when `successful` is true. It is 1 for a failed check,
execution error, unbuilt target, unexpected outcome, or strong float64 work.
It is 2 when CPU-only selection is refused or non-CPU devices are observed (the
report is still written), or when the arguments are invalid.

The JSON report (`kind: "phydrax-tpu-lowering-preflight"`) is published
atomically. It records Python, host, JAX/JAXlib/Equinox/Phydrax versions, the
Phydrax source path, the tool's SHA-256, the effective configuration, and the
frontend index patch identity. TPU fill-gather lowering is registered when
Phydrax is imported, so exported gathers use the same rule as a TPU run.

Preflight success is **not** TPU evidence, and `hardware_qualified`,
`tpu_compiled`, and `tpu_executed` are always false. These limits apply:

- StableHLO is not compiled by XLA:TPU. TPU compiler admission, layouts,
  memory, and performance remain unobserved.
- CPU tolerances do not carry over to TPU arithmetic. For example, `highest`
  float32 matmuls on TPU use six BF16 passes.
- Only one device is exported, with no sharding, collectives, or multi-host
  routes.
- Scenarios cover their fixed shapes and selected derivative orders only.

Qualification still requires `tools/qualify_accelerator_wheels.py` on actual
selected TPU devices.

## Upstream references

- [JAX installation: TPU](https://docs.jax.dev/en/latest/installation.html#tpu)
- [JAX distributed initialization](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)
- [JAX precision semantics](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision)
- [JAX export and serialization](https://docs.jax.dev/en/latest/export/export.html)
- [JAX 0.12.0 TPU library selection and Cloud TPU startup](https://github.com/jax-ml/jax/blob/jax-v0.12.0/jax/_src/cloud_tpu_init.py)
