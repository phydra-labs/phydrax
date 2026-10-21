# External solver backends

`phydrax.backends` is the explicit boundary between Phydrax problem contracts and
optional solver runtimes. Importing `phydrax` does not require MPAX, PETSc, SLEPc,
PyAMGCL, CUDA, or AmgX. Backend symbols load from their canonical owners on
first access; importing the facade does not initialize a JAX device.

External providers are never candidates of the native automatic linear policy.
Selecting one is an explicit user decision; an unavailable provider raises
`BackendUnavailableError` with the backend, capability, requirement, and import or
linker evidence. No external path silently materializes a matrix-free operator,
changes provider, or falls back to a native solve.

## Apple Metal environment

`metal_environment(initialize_backend=False)` inspects installed distribution,
entry-point, native-asset, source/build, JAX/JAXlib, and operating-system evidence
without loading the plugin or initializing devices. `assess_metal_environment(facts)`
is the pure equivalent for supplied immutable `MetalEnvironmentFacts`.

```python
from phydrax import config
from phydrax.backends import metal_environment

report = metal_environment(initialize_backend=False)
print(report.successful, report.facts.device_probed)
for issue in report.issues:
    print(issue.code, issue.severity, issue.message)
```

`successful` means the checks represented by those facts succeeded. It is not a
hardware or scientific qualification claim. The default explicit inspection,
`metal_environment()`, additionally probes MPS unless an explicit runtime
selection excludes it or installation assessment already failed.

Metal issues remain separate from the CUDA-specific
`BackendAvailability.environment_issues`. A matching `phydrax-metal` companion
must identify its exact `mps` runtime, source, assets, and JAX/JAXlib pair.
Explicit MPS selection through `config.RuntimeConfig` refuses a known-invalid
installation before applying settings. Competing Apple plugins are refused,
not selected by discovery order.

The Apple candidate profile is single-device float32/complex64 numerical
execution with exact 64-bit integer addressing. It does not authorize float64,
complex128, CUDA providers, Pallas Metal kernels, or distributed execution.
Keep x64 enabled for integer addresses and construct numerical arrays explicitly
in the scientifically admitted precision; x64 enablement alone is not a request
to execute float64 arithmetic.

::: phydrax.backends.MetalEnvironmentFacts

::: phydrax.backends.MetalEnvironmentIssue

::: phydrax.backends.MetalEnvironmentReport

::: phydrax.backends.metal_environment

::: phydrax.backends.assess_metal_environment

## Upstream oneAPI and TPU environments

`phydrax[oneapi]` and `phydrax[tpu]` use upstream JAX runtimes, not Phydrax-owned
native companions. The experimental oneAPI stack uses JAX 0.12.0 with JAXlib
0.11.2 and `jax-oneapi-plugin[with-oneapi]` 0.11.2. This mixed frontend/runtime
pair is **not Intel's documented matched release**. Intel's compute GPU driver
is an external prerequisite. The TPU stack uses JAX/JAXlib 0.12.0 and upstream
`libtpu` 0.0.50; Cloud TPU provisioning is outside the package's scope.
Both installation envelopes require native Linux x86_64, not macOS, Windows,
or ARM. TPU's selected wheel requires glibc 2.31 or newer.

`collect_oneapi_environment()` and `collect_tpu_environment()` collect host
facts without calling `jax.devices()` or loading a provider client.
`assess_oneapi_environment(facts)` and `assess_tpu_environment(facts)` are pure
assessments. `oneapi_environment(initialize_backend=False)` and
`tpu_environment(initialize_backend=False)` combine host collection and
assessment by default; request `initialize_backend=True` explicitly to probe
the runtime. Existing JAX import/startup side effects are not covered by the
no-client-initialization guarantee.

Reports expose immutable `facts` and `issues`, including canonical installed
version pairs, selected platforms, probe state, and observed runtime/device
facts when probed. `successful` means the represented assessment has no errors,
not that hardware or scientific qualification passed. An unprobed assessment
cannot establish device availability. CPU execution is never accepted as Intel
or TPU evidence. Upstream distribution, file, and runtime identities remain
upstream identities: no Phydrax native source/build ID is synthesized for them.

```python
from phydrax.backends import oneapi_environment, tpu_environment

for inspect in (oneapi_environment, tpu_environment):
    report = inspect()  # Host inspection only.
    print(report.successful, report.facts.device_probed)
    for issue in report.issues:
        print(issue.code, issue.severity, issue.message)
```

The evidence-free candidate catalogs contain bounded, single-device float32
MLP/PDE differentiation and parameter updates, exact wide integer addressing,
compiled control flow and callback errors, iterative linear solving, and real
Fourier resampling. They are **not release authorization**. Neither the
installation metadata nor software-only tests qualify an Intel GPU or TPU;
hardware qualification remains pending until the installed-wheel qualifier
passes every exact required case on the selected real runtime, without skips.
The support coordinate `execution_device_count=1` names the one selected local
device exercised by the qualifier, not the host's visible topology. Other
visible devices may be recorded but are not thereby qualified.

Keep x64 enabled for exact integer addressing, while selecting numerical dtypes
explicitly. x64 enablement is not evidence of native FP64 arithmetic. The Intel
candidate makes no float64/complex, multi-device, multi-host, MPI, CUDA-provider,
or Pallas-kernel qualification claim. TPU's candidate is float32 with
`matmul_precision="highest"`: on TPU this is six BF16 passes, **not native IEEE
float32 matrix multiplication**. TPU float64, emulated float64, and complex
coordinates are not qualified by these profiles. Native TPU meshes and JAX
collectives remain separate admission mechanisms; these single-device candidate
profiles do not qualify distributed execution.

::: phydrax.backends.oneapi_environment

::: phydrax.backends.assess_oneapi_environment

::: phydrax.backends.tpu_environment

::: phydrax.backends.assess_tpu_environment

## Spatial Pallas capability boundaries

`PALLAS_SPATIAL_CAPABILITIES` describes the CUDA/explicit-interpreter aggregate,
including float32/float64 distance leaves and the particle-FMM fragment route.
`PALLAS_TPU_SPATIAL_CAPABILITIES` retains implemented **float32 distance and
Morton neighbor/radius leaves only**. It does not declare TPU float64 or compiled
particle FMM: the bounded particle-FMM fragment kernel uses the CUDA-only route.
`pallas_spatial_availability()` returns the capabilities for the selected
runtime, so inspect `availability.capabilities` and require the specific
operation rather than treating any available Pallas device as supporting the
whole aggregate. Unsupported routes refuse rather than executing on CPU.
CPU interpreter tests establish semantics only, not accelerator qualification.

## Common inspection and transfer contract

Every backend publishes immutable `BackendCapabilities` and returns
`BackendAvailability`. Capabilities distinguish host-only from device execution,
assembled from matrix-free support, supported coordinate dtypes, lifecycle support,
and required explicit release. `BackendTransferEvidence` records host-to-device bytes,
device-to-host bytes, and explicit synchronization count for each result.

`BackendAvailability.environment_issues` is a tuple of `CudaEnvironmentIssue`
records (empty by default). cuDSS, AmgX, and QUDA load lazily from the matching
CUDA companion wheel, with version, component, handler, and CUDA-major checks.
NVIDIA dependencies are preloaded by absolute path from installed NVIDIA wheels,
not selected through `LD_LIBRARY_PATH`. GPU-direct mpi4jax remains user-provided
and must match JAX's CUDA major.

```python
import jax.numpy as jnp
import phydrax as phx

availability = phx.backends.petsc_availability()
if availability.available:
    availability.require("linear-system")
else:
    print(availability.requirement, availability.reason)
```

::: phydrax.backends.BackendUnavailableError

---

::: phydrax.backends.BackendCapabilities

---

::: phydrax.backends.BackendAvailability

---

::: phydrax.backends.BackendTransferEvidence

---

::: phydrax.backends.AbstractExternalBackend

## CUDA environment

`cuda_environment()` collects host facts, probes the JAX backend, and reports
mapped CUDA libraries on Linux. `assess_cuda_environment(facts)` assesses an
explicit immutable `CudaEnvironmentFacts` record without collecting host state
or initializing a backend. A `CudaEnvironmentReport` contains `facts` and
`issues`; `successful` means no issue has severity `"error"`.

Facts include host platform and machine, requested and installed plugin majors,
foreign NVIDIA distributions, `wheel_library_majors` (realpath/major pairs),
selected platforms, backend initialization and probe state, runtime and driver
versions, `backend_error`, CUDA library-path directories, and mapped libraries.
`MappedCudaLibrary` records `soname`, real `path`, and `major` (`None` when
unknown). `CudaEnvironmentIssue` records `code`, `severity`, `message`, and
related `paths`. `CudaIssueCode` is the closed set of issue codes;
`CudaIssueSeverity` is `"error"` or `"warning"`.

The mapped-library audit is diagnostic only; companion availability validates
installed identities and preloads NVIDIA wheel dependencies. Provider-specific issues appear in
`BackendAvailability.environment_issues`, not by probing every provider in
`cuda_environment()`. See [Installation](../installation.md#inspect-the-environment)
for usage, every issue code, and remedies.

Companion refusals use `companion-missing`, `companion-invalid`,
`companion-version-mismatch`, `companion-major-mismatch`, or
`native-dependency-missing`. Availability retains these typed issues without
changing the `BackendAvailability` record.

`cuda_native_resources()` snapshots loaded component identities and native memo
partitions without discovering or loading other components. Each partition
reports its component, device, partition key, live entries, declared capacity,
and counted rebuilds. Native state is released by the owned process-exit handler.

cuDSS and AmgX memo partitions are device-local. AmgX also owns its cuBLAS and
cuSPARSE handles, memory pools, allocation bookkeeping, and thread streams per
device; switching devices does not borrow another GPU's allocations.
Provider calls remain serialized and do not constitute a distributed solve.
Releasing a live preparation retires its named memo. Reusing a released or
evicted immutable preparation rebuilds under a fresh native ID without changing
the preparation's token; releasing that stale token cannot retire the rebuilt
entry. Reconstructed entries remain bounded by the device/plan cache capacity
and are reclaimed by eviction or component shutdown.

::: phydrax.backends.cuda_native_resources

---


::: phydrax.backends.cuda_environment

---

::: phydrax.backends.assess_cuda_environment

---

::: phydrax.backends.CudaEnvironmentReport

---

::: phydrax.backends.CudaEnvironmentIssue

---

::: phydrax.backends.CudaEnvironmentFacts

---

::: phydrax.backends.MappedCudaLibrary

---

::: phydrax.backends.CudaIssueCode

---

::: phydrax.backends.CudaIssueSeverity

## Distributed and vendor execution capabilities

`BackendDistributionCapabilities` distinguishes global JAX arrays from
rank-local and host-only providers. It records admissible execution scopes,
platforms, collectives, transformations, communicator ownership, process-local
input, distributed output, and topology-changing restart. These fields describe
a provider boundary; an exact `qualification.SupportTuple` is still required
for a hardware support claim.

`JaxCollectiveProvider` is the native named-axis route used inside
`shard_map`, using JAX/NCCL for NVIDIA multi-GPU and multi-node communication.
[Startup coordination](execution.md#bootstrap-before-device-initialization)
automatically handles `srun`/`mpirun` launches; MPI is only a launcher or optional
interop route. `Mpi4JaxCollectiveProvider` is an explicit rank-local MPI route
installed with `phydrax[mpi]`. Its communicator is caller-owned, and
automatic differentiation and batching remain operation-specific. Host
`mpi4py` traffic must not be interleaved on the same communicator with deferred
device MPI operations. mpi4jax selects GPU transport once per process from
`MPI4JAX_USE_CUDA_MPI`; set it to `1` before importing/using mpi4jax for
`gpu_direct=True`, or `0` for `gpu_direct=False` (host staging), and do not change
it afterward. Direct transport requires a CUDA-enabled mpi4jax build, a CUDA
device, and CUDA-aware MPI: the MPI library actually linked by mpi4py must
report `MPIX_Query_cuda_support() == 1`. Use a site CUDA-aware MPI and build both
mpi4py and mpi4jax against that installation's `mpicc`; use its matching
`mpirun`, not a launcher from another MPI installation. Ordinary PyPI MPI wheels
are built without CUDA support and may shadow the site's MPI. The mpi4jax CUDA
bridge must match the supported CUDA 13 JAX stack and is checked before import,
even for host-staged use. See the [CUDA 13 build recipe](../installation.md#gpu-direct-mpi-interoperability)
using a `CUDA_ROOT` toolkit view, an explicit NVIDIA-wheel RPATH, and
`uv pip install --no-cache --no-build-isolation --reinstall mpi4jax`.
MPI libraries must remain discoverable through RPATH or the deployment's library
path. Direct mode passes device buffers to MPI
without mpi4jax host staging; MPI may still copy internally. CUDA awareness does
not guarantee CUDA IPC or network GPUDirect RDMA. Single-node OpenMPI can use
its CUDA-aware shared-memory transport without UCX, including a copy route when
CUDA IPC export is unavailable. The provider refuses an unqualified direct
route instead of falling back.
Rank-local MPI does not require JAX multi-host initialization; avoid enabling
the independent launcher-detected Phydrax/JAX distributed bootstrap when both
ranks deliberately share one GPU. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for
that deployment. The CUDA MPI integration test accepts a matching launcher
(and deployment-specific flags) through `PHYDRAX_MPI_LAUNCHER`, and the requested
rank count through `PHYDRAX_MPI_RANKS` (default `2`).

`tools.qualify_cuda_mpi --visible-devices 0,1,2,3 ...` requests four ranks, not a
two-rank subset of four visible devices. Any explicit allocation of at least two
devices is supported by the qualification harness. Each rank must see the exact
allocation; the distinct-GPU cases require unique physical CUDA UUIDs and one
MPI shared-memory domain. The full distinct/shared placement × direct/staged
transport × float32/float64 matrix must pass without skips for that rank count.
The deliberately shared-GPU cases exercise compatibility and are never physical
multi-GPU evidence. Exchanges form a ring (receive from the previous rank, send
to the next); collective references include every rank. Both communicator
generations, supported AD paths, and unsupported-transform refusals are checked.

`benchmarks.cuda_mpi` requires an explicit matching `--ranks N`. Measurements
validate every rank's ownership and complete point/sample sets, then aggregate
the maximum of corresponding synchronized samples, never the maximum of rank
medians. Optional profiling is a separate workload with one bounded artifact
per requested rank (512 MiB each, `N × 512 MiB` total); instrumented timings are
not performance evidence. These are single-host CUDA-aware MPI checks, not
claims of network RDMA or absence of MPI-internal host staging.
The qualifier forwards and records an explicitly supplied `NCCL_NVLS_ENABLE`
deployment setting without selecting a default or retry fallback; MPI/UCX
measurements do not establish NCCL transport support.

The pinned mpi4jax transformation contract is narrower than a generic transpose
of stacked rank outputs: SUM JVP reduces tangents, while SUM VJP returns the
rank-local cotangent; mean VJP divides that local cotangent by communicator size.
The default same-shape `sendrecv` path supports VJP. Its receive argument is
read-only shape/dtype metadata, so Phydrax reuses the input rather than creating
an AD-inactive zero placeholder. Other explicitly supplied receive-buffer
differentiation combinations are not qualified by these checks.
MAX/MIN differentiation and forward-mode `sendrecv` remain unsupported and
refuse explicitly. Use the JIT dispatcher for ordered MPI effects; repeated
AOT `Compiled.__call__` execution loses an effect token in JAX 0.11.2 and is not
a qualified entry point.
The [two-A100 qualification](../installation.md#two-a100-results) distinguishes
large CUDA-IPC exchanges from SUM's internal host staging; `gpu_direct=True`
does not promise GPU-resident reduction.

Vendor profiles are admission envelopes rather than blanket qualification:

| Profile | Platform | Admissible scope |
| --- | --- | --- |
| `NVIDIA_JAX_PROFILE` | CUDA | single-device, multi-device, multi-host |
| `AMD_JAX_PROFILE` | ROCm | single-device, multi-device, multi-host |
| `TPU_JAX_PROFILE` | TPU | float32/bfloat16 admission on the upstream stable runtime; single-device, multi-device, multi-host; Phydrax qualification remains candidate |
| `INTEL_JAX_PROFILE` | oneAPI (`jax-oneapi`) | experimental single-device float32; no MPI or multi-host |
| `APPLE_METAL_PROFILE` | MPS (`phydrax-metal`) | experimental single-device, float32/complex64 |

Mixed vendors execute as separate workflow stages with explicit transfer.
Phydrax does not claim one heterogeneous global JAX mesh.

::: phydrax.backends.BackendDistributionCapabilities

---

::: phydrax.backends.JaxCollectiveProvider

---

::: phydrax.backends.Mpi4JaxCollectiveProvider

---

::: phydrax.backends.mpi4jax_availability

## Lattice kernel providers

`create_lattice_provider("jax-native-lattice" | "quda" | "gpt-grid" | "qlat")`
selects one canonical `LatticeKernelProvider`. The external availability functions
(`quda_availability`, `grid_availability`, `qlat_availability`, and
`optional_lattice_provider_availability`) admit executable adapter prerequisites,
not merely importable packages. Construction refuses unavailable runtimes; no
adapter silently substitutes native JAX execution.

Gauge operations take canonical `(site, direction, color, color)` links, `beta`,
an owned-output mask, and explicit `lattice_shape` and `periodic` keyword arguments.
Site ordering is C order. Action masks are `(site, plane)` with planes ordered
by ascending `mu < nu`; force masks are `(site, direction)`. Native JAX supports
periodic axes and bounded halo-extended blocks, excluding faces that leave bounded
axes. External adapters currently support single-process periodic 4D SU(3)
lattices; their action requires full ownership, while force supports output masks.
The QUDA and Grid adapters additionally require even lattice extents; QUDA
requires Wilson `r=1`.
No external adapter currently declares distributed halo exchange or MPI reductions.

Fermion operations consume canonical Wilson/clover-Wilson operators and
`(site, spin, color)` spinors (an optional final RHS axis denotes a block).
They execute the full Dirac action, including mass, spacing, boundary phases,
spin-basis conversion, and supplied local clover blocks—not just the library's
raw hopping term. The adjoint flag denotes the Hermitian adjoint.
QUDA uses Wilson hopping with faithful spin-basis conversion, including arbitrary
validated gamma bases, and applies the operator's arbitrary local clover blocks
on device through the native JAX clover owner; it does not replace those blocks
with QUDA's gauge-generated clover term.

| Provider | Operations | Dtypes | JAX transformations |
| --- | --- | --- | --- |
| Native JAX | Gauge action/force, Wilson/clover/block, projected halo | complex64/complex128 | jit, vmap, jvp, vjp, rematerialization |
| QUDA | Gauge action/force, Wilson/clover/block | complex64/complex128 | jit (device-resident public JAX FFI); no autodiff |
| Grid/gpt | Gauge action/force, Wilson/clover/block | complex64/complex128 | eager only |
| Qlat | Gauge action/force | complex128 | eager only |

QUDA executes through the companion's public JAX FFI handlers without host-staged
callbacks. Every call reloads device gauge links; only the scalar action result
crosses to the host. Calls synchronize XLA inputs and await QUDA completion, so
they are explicit synchronization points. A process mutex serializes QUDA's
global runtime. Only one JAX process is supported; the first call initializes one
CUDA device and later calls on another device are refused.

When unset, `QUDA_RESOURCE_PATH` defaults to
`$XDG_CACHE_HOME/phydrax/quda/<build hash>/<QUDA variant>` (using `~/.cache` when
`XDG_CACHE_HOME` is unset). The variant is QUDA's own tune-cache key (CPU
architecture, compiled GPU architectures, CUDA compiler version), so development
and release builds never share a tune cache that QUDA would reject. An explicit
resource path is preserved. The owned exit
handler calls `endQuda`.

CPU adapters explicitly transfer arrays to their host runtime. External adapters
declare no automatic differentiation; differentiating through them is refused
rather than returning an incorrect derivative. Provider status preserves runtime
and version evidence.

::: phydrax.backends.quda_availability

::: phydrax.backends.create_lattice_provider



## Clarabel conic programming

Clarabel 0.11.1 is an optional host interior-point backend for LP, QP, and the
public Phydrax cone product. Install `phydrax[clarabel]` and select
`ClarabelInteriorPoint` explicitly. The provider retains settings and version
evidence; the optimization adapter maps cone and bound layouts, restores rotated-SOC
coordinates, and independently audits original-coordinate KKT residuals and rays.

Clarabel execution transfers dense or canonical sparse program data to SciPy CSC on
the host. It is not JIT-compatible and exposes no PhydraX differentiation mode. No
native or other external method is selected after a Clarabel failure.

`NativeHomogeneousConic` is the separate accelerator-resident PhydraX method over
the same canonical coordinates, cone ordering, tolerances, result, and audit surface;
it is not Clarabel code running on an accelerator.

::: phydrax.backends.ClarabelBackend

---

::: phydrax.backends.ClarabelPlan

---

::: phydrax.backends.PreparedClarabel

---

::: phydrax.backends.clarabel_availability

---

::: phydrax.backends.prepare_clarabel

## SCIP mixed-integer programming

SCIP is an optional host MILP provider installed with `phydrax[scip]`.
`SCIPMixedInteger` consumes canonical `LinearProgram` data through PySCIPOpt;
the canonical program remains the source of truth and every returned incumbent
passes the same independent mixed-integer primal audit as native search.

The initial adapter is deliberately narrow: assembled MILP, full incumbent
starts, typed deterministic settings, and explicit model release. It does not
expose arbitrary SCIP parameters, Python constraint-handler plugins, conic
constraints, or nonlinear expressions. One thread is the deterministic
default.

SCIP's termination and dual bound are provider evidence rather than an
independently verified Phydrax branch proof. A result can therefore be
`provider_solved` and primal-feasible while `certified` and `successful` remain
false. Native audited branch-and-bound remains the independently certified
route until an external exact proof artifact is verified.

::: phydrax.backends.SCIPBackend

---

::: phydrax.backends.SCIPPlan

---

::: phydrax.backends.scip_availability

---

::: phydrax.backends.prepare_scip

---

---

## CVXPY canonicalization interoperability

CVXPY is an optional host modeling boundary, not a solver fallback.
`cvxpy_availability` inspects it lazily. Supported real continuous problems are
canonicalized to sparse LP/QP/product-cone data by `import_cvxpy_problem`; variable
slices and parameter topology live in `CVXPYProgramBinding`. Complex graphs, custom
atoms outside the supported canonical cones, graph-topology refresh, and mixed
integer models fail closed.

`refresh_cvxpy_program` re-canonicalizes parameter values and rejects any changed
cone ordering, sparse relation, variable slice, or canonical inverse map.
`restore_cvxpy_solution` sends canonical primal and cone-dual coordinates through
CVXPY's recorded solving-chain inverse data, restoring original variable and
constraint-dual values with their canonical signs.
::: phydrax.backends.cvxpy_availability

---

::: phydrax.backends.CVXPY_CAPABILITIES

## Optional prepared tensor CUDA execution

`phydrax[cuda13,cutensor]` and `phydrax[cuda13,cutensornet]` add explicitly
selected implementations beneath `phydrax.tensor_network`. Neither intercepts
ordinary `ein.contract`, matrix multiplication, or native symmetry-sector
execution. See the [prepared tensor contract](tensor_network.md).

`ContractionExecutionPolicy` selects the executor independently of an imported
path or `ContractionPathPlanner`. The latter performs bounded, shape-only
cuTensorNet planning, while its returned path still passes native schedule and
resource admission. Numerical arrays remain dynamic; current values are rebound
for every call. Descriptor reuse is not numerical-result caching.

The admitted numerical surface covers real/complex 32-/64-bit values, explicit
computation/accumulation precision, and native-owned derivative actions through
order two. Unsupported transformations, precision, layout, or resource requests
refuse explicitly. Vendor workspace/cache budgets count external device storage;
opaque vendor host descriptor bytes remain unknown.

Native host slice placement/checkpoint reduction remains separate from pure
device-local numerical execution. It is not differentiable orchestration and
does not silently enable vendor MPI or hidden slicing. Candidate declarations
are not hardware or release qualification.

## Optional cuEST molecular integrals

`phydrax[cuda13,cuest]` supplies a small companion wrapper and the separate
optional `phydrax-cuest` C-runtime wheel. Native Gaussian, molecular SCF,
convergence, units, result evidence, and PES owners remain authoritative.
The selected profile is nonperiodic all-electron real-FP64 integer DF-RHF with
analytic first nuclear forces—not exact ERIs, a vendor SCF engine, arbitrary
autodiff, or analytic Hessians.

The integral definition includes the actual orbital/auxiliary basis, fitting,
screening, and precision choices. Explicit bound geometry is reused through SCF
and associated force work; changing coordinates or a prepared recipe requires
the owning rebind/restore path. A density-factor fast path must certify the
actual current density, including damping, rather than substituting the newest
orbitals. Same-definition energy/Pulay/force and stationary checks are retained.
See [molecular electronic structure](../guides_computational_chemistry.md) and
the [typed production workflow](../guides_production_computational_chemistry.md).

## Site cuDecomp transposes

`cudecomp_availability` discovers the explicitly supplied site component.
`CudecompBackend` does not create an independent MPI world or imply actual
hardware qualification. The numerical owner is the distributed spectral
substrate: regular divisible three-dimensional slabs, one GPU per rank on one
host, explicit complex64/complex128, exact source/target layout validation, and
bounded known scratch.

Local JAX FFTs, normalization, mathematical adjoints, and native scientific
owners remain unchanged. Canonical pencil layouts, local guarded transforms,
unsupported dimensions, batching, and rematerialization refuse explicit
cuDecomp selection instead of falling back or replacing the native pencil
schedule. Payload packing and opaque resident bytes are reported separately.
See [execution ownership](execution.md#supervised-site-mpi-and-ordered-vendor-collectives).

::: phydrax.backends.CudecompBackend

::: phydrax.backends.cudecomp_availability

::: phydrax.backends.CUDECOMP_CAPABILITIES

## cuDSS (native)

The Linux x86-64 and aarch64 CUDA 13 companion provides cuDSS through
`phydrax[cuda13]`. Inspect `cudss_availability`, then select
`SparseLDLT(provider="cudss", factor_cache_capacity=8)` explicitly.
Importing Phydrax does not load the companion or initialize CUDA.

The local provider accepts sorted canonical 32-bit CSR structure and real
float32/float64 symmetric systems, including shared-pattern numeric value
batches. It retains symbolic analysis across `phydrax.linalg.refresh`, reuses
factors for true multiple RHS, and supports `phydrax.linalg.release` to retire
memo entries early. Refinement and original-coordinate residual certification
remain native Phydrax operations.

Each local/MG plan has a separate bounded native factor-cache partition; the positive
`factor_cache_capacity` participates in policy and plan identity. Evicted state
is rebuilt from call operands, so release or eviction does not invalidate a
prepared value. Solve provenance reports exact int32 `native_rebuilds`.
Use `cuda_native_resources()` to inspect live entries, capacities, and rebuilds.
Failed resident numeric refresh retires the affected memo before releasing its
operation lock. The error reaches the caller, and the original preparation's
next solve rebuilds from its original operands rather than reusing partial factors.

`SparseLDLT.execution` is a `CudssExecutionPolicy`:

| Mode | Ownership and admission |
| --- | --- |
| `local` | Existing device-local CSR and value-batch lifecycle |
| `mg` | Explicit ordered 2/4/8-GPU group in one process; one full CSR/RHS on the first initiating device, vendor-owned peer streams/scratch, one group solve, no distributed input or matrix batching |
| `mgmn` | Explicit unpadded rank-local CSR, one GPU per process on the same host, exact site-linked GPU-aware MPI/shim provenance and authenticated live collective lease; separate `collective` and `cudss_mgmn` DSOs, not universal companion-wheel components |

The current coupled supervisor authenticates same-host descendants; vendor
MGMN potential is not an admitted multi-host Phydrax execution profile.

`cudss_availability(execution_mode=...)` and
`sparse_provider_capabilities("cudss", execution_mode=...)` distinguish artifact
availability and mode capabilities. Neither allocates a group nor certifies
transport. MG refuses capture and conflicting live vendor reservations.
MGMN requires the lease owner's explicit invocation boundary: dynamic live
tickets, actual transformed all-rank schedule agreement, effect/token ordering,
stream completion, and explicit collective retirement. Numerical invocation
takes only an explicitly staged host-only `CollectiveProgram`; its owned graph
supplies the schedule and the exact associated executable. Invocation does not
retrace or execute arbitrary Python callbacks. User-code staging belongs to the
named preparation boundary. A side-effect flag or process-local registry
integer alone is not this contract. Site MPI is not bundled, and
`CUDSS_COMM_LIB`/arbitrary loader search paths are not accepted communication
library provenance.

The pinned NVIDIA CUDA 13
[cuDSS 0.8.0 redistribution index](https://developer.download.nvidia.com/compute/cudss/redist/redistrib_0.8.0.json)
supplies `src/cudss_commlayer_openmpi.cu` in its 0.8.0.10 archive.
`native/cuda/src/cudss_mgmn/commlayer/CMakeLists.txt` builds that unmodified,
SHA256-pinned source against an explicit GPU-aware site OpenMPI prefix and the
wheel-owned CUDA 13 toolchain. Both host and device communicators use MPI; the
site recipe records source/interface/MPI hashes and build identity. It neither
copies vendor source into the repository nor accepts a prebuilt shim merely
because its SONAME is compatible.

`memory_mode="hybrid"` remains the existing vendor hybrid-memory choice.
`CudssMemoryPolicy(hybrid_device_limit_bytes=...)` applies a checked
factorization/solve device limit after analysis; it does not bound analysis,
host factors, pinned pages, or other live states. Certified all-phase bounds
and host-limit requests are refused. `diagnostics.external_memory` retains
per-calling-device vendor estimates, with unknown host aggregation rather than
an inferred sum or RSS-derived bound. `provider_estimate` evidence cannot
satisfy a certified memory envelope.

Local cuDSS retains its partial positive/negative inertia report without
claiming reliable zero inertia. MG/MGMN inertia is unknown: the documented
diagonal-query reorderings are not supported in those modes. Consequently no
mode gains structured-IPM admission from multi-GPU execution. No unavailable or
insufficient-inertia execution falls back to another provider.

See the [linear-solve ownership contract](linalg.md#cudss-memory-and-execution-ownership)
and NVIDIA's pinned cuDSS 0.8
[advanced features](https://docs.nvidia.com/cuda/cudss/0.8.0/advanced_features.html),
[memory/data queries](https://docs.nvidia.com/cuda/cudss/0.8.0/types.html), and
[execution/distribution functions](https://docs.nvidia.com/cuda/cudss/0.8.0/functions.html).

::: phydrax.backends.CudssBackend

---

::: phydrax.backends.CUDSS_CAPABILITIES

---

::: phydrax.backends.cudss_availability

---

## MPAX mathematical programming

MPAX 0.2.4 is an optional device backend for assembled LPs and convex QPs. Install
the compatible `mpax==0.2.4` wheel separately, then select `MPAXraPDHG` for LP/QP
or `MPAXr2HPDHG` for LP only. MPAX's Darwin x86_64 wheel pins an incompatible JAX,
so Phydrax does not publish an unsatisfiable cross-platform `mpax` extra.
Phydrax maps inequality and dual signs explicitly, preserves native variable bounds,
and independently audits the original unscaled program before assigning an optimal,
infeasible, or unbounded status.

MPAX's default JIT loop is not reverse-mode differentiable. Algorithmic
differentiation requires `unroll=True`, a finite iteration capacity, and
`ConvexDifferentiationPolicy("algorithmic")`. It is not implicit KKT sensitivity.
Warm starts require an MPAX method configured with `warm_start=True`.

`MPAXPlan(representation="sparse")` requests the provider's assembled sparse LP/QP
surface; `"dense"` remains explicit. This does not accept arbitrary PhydraX callback
operators or general SOC/PSD/exponential/power cones. MPAX may use multiplication-only
iterations internally, but `supports_matrix_free` remains false for callback input.

::: phydrax.backends.MPAXBackend

---

::: phydrax.backends.MPAXPlan

---

::: phydrax.backends.PreparedMPAX

---

::: phydrax.backends.mpax_availability

---

::: phydrax.backends.prepare_mpax

---

## PETSc KSP and SNES

The PETSc KSP path accepts an unbatched square `LinearSystem` whose operator and
optional preconditioning operator expose sorted canonical CSR storage through
`AbstractSparseLinearOperator`. The two operators retain distinct identities and
PETSc handles: the action matrix is not silently replaced by the preconditioning
matrix. `PETScKSPPolicy` makes KSP type, PC type, tolerances, options, and
preconditioner reuse explicit. Convergence is determined from PETSc's reason and an
independently recomputed Phydrax residual, never from iteration count alone.

```python
matrix = jnp.asarray([[4.0, -1.0], [-1.0, 3.0]])
rows, columns = jnp.nonzero(matrix)
relation = phx.sparse.EdgeRelation(
    columns,
    rows,
    source_size=2,
    target_size=2,
)
space = phx.linalg.ArraySpace((2,), dtype=matrix.dtype)
operator = phx.sparse.SparseCoordinateOperator(
    relation,
    matrix[rows, columns],
    source=space,
    target=space,
    operator_id="external-backend-example",
)
system = phx.linalg.LinearSystem(operator, problem_id="external-backend-system")
right_hand_side = jnp.asarray([1.0, 2.0])

if phx.backends.petsc_availability().available:
    policy = phx.backends.PETScKSPPolicy(
        ksp_type="gmres",
        pc_type="ilu",
        reuse_preconditioner=True,
    )
    plan = phx.backends.plan_petsc_linear(system, policy)
    prepared = phx.backends.prepare_petsc_linear(plan)
    result = phx.backends.solve_petsc_linear(prepared, right_hand_side)
```

`PETScSNESPolicy(jacobian_mode="matrix-free")` is the nonlinear default and asks
PETSc for a matrix-free/JFNK Jacobian. `jacobian_mode="dense-autodiff"` is a separate,
user-declared mode guarded by maximum dimension and byte limits. Neither mode is
chosen as a fallback for the other. PETSc execution is host-only and is not a JIT or
autodiff boundary; use native `phydrax.nonlinear` methods when transformed execution
is required.

::: phydrax.backends.PETScBackend

---

::: phydrax.backends.PETScKSPPolicy

---

::: phydrax.backends.PETScSNESPolicy

---

::: phydrax.backends.PETScLinearPlan

---

::: phydrax.backends.PreparedPETScLinearSolve

---

::: phydrax.backends.PETScLinearResult

---

::: phydrax.backends.PETScLinearDiagnostics

---

::: phydrax.backends.PETScNonlinearPlan

---

::: phydrax.backends.PreparedPETScNonlinearSolve

---

::: phydrax.backends.PETScNonlinearResult

---

::: phydrax.backends.PETScNonlinearDiagnostics

---

::: phydrax.backends.PETScProvenance

---

::: phydrax.backends.petsc_availability

---

::: phydrax.backends.plan_petsc_linear

---

::: phydrax.backends.prepare_petsc_linear

---

::: phydrax.backends.solve_petsc_linear

---

::: phydrax.backends.refresh_petsc_linear

---

::: phydrax.backends.plan_petsc_nonlinear

---

::: phydrax.backends.prepare_petsc_nonlinear

---

::: phydrax.backends.solve_petsc_nonlinear

---

::: phydrax.backends.refresh_petsc_nonlinear

## SLEPc EPS

The SLEPc backend consumes `phydrax.linalg.eigen.GeneralEigenproblem`. Its default
`operator_mode="shell"` maps PETSc shell matrix actions directly to Phydrax operator
applications in canonical coordinates, including adjoint actions for two-sided
left/right solves. `operator_mode="csr"` instead requires canonical sparse storage.
Shift-invert and Cayley modes require the assembled CSR path plus explicit
`SLEPcSTOptions`; requesting either transform with a shell is rejected rather than
materialized.

The result retains the SLEPc convergence reason and partial count, then independently
checks selected count, original-pencil right and left residuals, pairing, and
biorthogonality. The prepared PETSc/SLEPc objects own host resources and must be
released explicitly.

```python
if phx.backends.slepc_availability().available:
    problem = phx.linalg.eigen.GeneralEigenproblem(operator)
    selection = phx.linalg.eigen.GeneralEigenSelection(
        "largest-real",
        count=1,
    )
    policy = phx.backends.SLEPcEigenPolicy(
        selection,
        operator_mode="shell",
    )
    plan = phx.backends.plan_slepc_eigensolve(problem, policy)
    prepared = phx.backends.prepare_slepc_eigensolve(problem, plan)
    try:
        result = phx.backends.slepc_eigensolve(prepared)
    finally:
        phx.backends.release_slepc_eigensolve(prepared)
```

::: phydrax.backends.SLEPcBackend

---

::: phydrax.backends.SLEPcEigenPolicy

---

::: phydrax.backends.SLEPcSTOptions

---

::: phydrax.backends.SLEPcEigenPlan

---

::: phydrax.backends.PreparedSLEPcEigenSolve

---

::: phydrax.backends.SLEPcEigenResult

---

::: phydrax.backends.SLEPcEigenDiagnostics

---

::: phydrax.backends.SLEPcEigenProvenance

---

::: phydrax.backends.slepc_availability

---

::: phydrax.backends.plan_slepc_eigensolve

---

::: phydrax.backends.prepare_slepc_eigensolve

---

::: phydrax.backends.slepc_eigensolve

---

::: phydrax.backends.refresh_slepc_eigensolve

---

::: phydrax.backends.release_slepc_eigensolve

## PyAMGCL and NVIDIA AmgX

Both AMG providers accept only square canonical-CSR `LinearSystem` problems with real
32- or 64-bit coordinates. The plan fingerprints the exact sparsity pattern and an
immutable, recursively canonicalized configuration. Numeric refresh preserves the
pattern and increments the numeric version; structural drift is an error.

PyAMGCL is a host provider. Its result reports iteration or reason data only when
the binding exposes them; missing counters remain absent. AmgX uses the CUDA
companion's public JAX FFI for device-resident, jittable execution, without
automatic differentiation. Its dDDI specialization promotes float32 buffers to
float64 and casts results back; `provenance.promoted_to_float64` records this.
The companion synchronizes the XLA stream before and after AmgX work, so these
calls are explicit synchronization points despite keeping numerical arrays on
device. Restricting compiled modes to dDDI does not prune solver algorithms.

`AmgXPolicy(hierarchy_cache_capacity=2)` declares a positive per-plan native
hierarchy-cache capacity and includes it in policy identity. Native refresh
replaces coefficients and resets the solver in place when resident, while
`refresh_amgx` returns an independent preparation. The old value remains usable:
its next solve rebuilds its original hierarchy once and counts that rebuild.
Evicted state is also rebuilt from complete operands. Diagnostics expose per-RHS
`native_status`, `iterations`, and `native_residual_norm`, plus `native_rebuilds`;
independent residual verification remains JAX-side. `release_amgx` is idempotent
and optional: it retires entries early without invalidating preparations. A
subsequent solve rebuilds state and records that work in `native_rebuilds`.
If coefficient replacement or resetup fails, native work is drained and the
affected memo is retired before the error returns. The original preparation
remains reconstructible; failed resetup never becomes a usable partial hierarchy.
AmgX initialization/finalization is reference counted inside the companion.
The owned exit handler releases native state and finalizes loaded components.

```python
if phx.backends.amgx_availability().available:
    plan = phx.backends.plan_amgx(system, phx.backends.AmgXPolicy())
    prepared = phx.backends.prepare_amgx(system, plan)
    try:
        result = phx.backends.solve_amgx(prepared, right_hand_side)
    finally:
        phx.backends.release_amgx(prepared)
```

::: phydrax.backends.PyAMGCLBackend

---

::: phydrax.backends.PyAMGCLPolicy

---

::: phydrax.backends.PyAMGCLPlan

---

::: phydrax.backends.PreparedPyAMGCL

---

::: phydrax.backends.AmgXBackend

---

::: phydrax.backends.AmgXPolicy

---

::: phydrax.backends.AmgXPlan

---

::: phydrax.backends.PreparedAmgX

---

::: phydrax.backends.AMGSolveResult

---

::: phydrax.backends.AMGSolveDiagnostics

---

::: phydrax.backends.AMGProvenance

---

::: phydrax.backends.AMGSolveStatus

---

::: phydrax.backends.pyamgcl_availability

---

::: phydrax.backends.amgx_availability

---

::: phydrax.backends.plan_pyamgcl

---

::: phydrax.backends.prepare_pyamgcl

---

::: phydrax.backends.solve_pyamgcl

---

::: phydrax.backends.refresh_pyamgcl

---

::: phydrax.backends.plan_amgx

---

::: phydrax.backends.prepare_amgx

---

::: phydrax.backends.solve_amgx

---

::: phydrax.backends.refresh_amgx

---

::: phydrax.backends.release_amgx

## Accelerated atomistic Pallas kernels

`phydrax.backends.atomistic` is the admission owner for the accelerated MACE
edge-coupling kernels (`phydrax.nn.atomistic.MACEAcceleratedCoupling`). These
kernels are written once against JAX Pallas Mosaic GPU, where one warpgroup of
`WARPGROUP_LANES = 128` lanes owns a channel tile. They have five
receiver/source/radial/harmonic/coefficient roles (`MACECouplingRole`), whose
backend problem kinds are given by `mace_coupling_problem_kind(role)`. A
streamed relation owns the fragment schedule and calls these kernels as its
whole-fragment aggregator. No target substitutes ordinary JAX for a kernel. The
ordinary-JAX streamed route remains the portable reference.

`pallas_atomistic_availability(target)` returns `BackendAvailability` for one
`AtomisticAccelerationTarget`:

| Target | Lowering | Admission |
| --- | --- | --- |
| `"cuda"` | `"pallas_mosaic_gpu"` | Active JAX GPU backend on the NVIDIA CUDA client with an admitted compute capability: exactly 8.0, 9.0, 10.0, 10.3, 12.0, or 12.1. Other GPU clients, such as ROCm, and other architectures are refused. |
| `"cpu_interpret"` | `"pallas_mosaic_gpu_interpret"` | Always available. The CPU Mosaic GPU interpreter executes the same kernel bodies for semantic reference verification. |

HIP/ROCm and TPU are not implemented targets. `AtomisticKernelTarget(target)`
resolves and refuses an unavailable target at construction. It records the
device kind, compute capability, platform version, and JAX/jaxlib versions in
`target_id`.

`AtomisticKernelAdmission(request, target, limits=ATOMISTIC_KERNEL_LIMITS)`
admits one `AtomisticKernelRequest` before execution. Each of the following
raises a refusal error:

- `accumulation="fast"`: only receiver-owned `"deterministic"` or
  `"compensated"` accumulation is implemented, so atomic scatter is refused;
- a channel tile that is not a power-of-two multiple of 128 lanes, or any
  channel/receiver/edge tile above `maximum_tile`;
- more reduction programs than `maximum_reduction_programs`;
- a coupling degree, path count, or coefficient count outside the admitted
  limits;
- a planned per-program workspace above `maximum_workspace_bytes`;
- a derivative order above `maximum_derivative_order`;
- declared fragment operand/result bytes above the caller's explicit
  `fragment_budget_bytes`.

`AtomisticKernelLimits.maximum_workspace_bytes` is a planning bound computed
from static structure and tiles, not a measured occupancy, register, or
shared-memory figure. `AtomisticKernelPrecision` admits `"float32"` and
`"float64"`.

Admission is not qualification. For `"cpu_interpret"`,
`qualification_scope` states that it is reference verification and not GPU
correctness, performance, or memory qualification; interpreter runs are not GPU
performance evidence. For `"cuda"`, admission names a runtime-admitted CUDA
candidate on the resolved device at the requested precision. Numerical,
derivative, and performance qualification of that `(device, precision)` tuple
is separate evidence. No CUDA tuple has been hardware-qualified, and
`PALLAS_ATOMISTIC_CAPABILITIES` describes the provider boundary only.

::: phydrax.backends.pallas_atomistic_availability

---

::: phydrax.backends.PALLAS_ATOMISTIC_CAPABILITIES

---

::: phydrax.backends.AtomisticKernelTarget

---

::: phydrax.backends.AtomisticKernelRequest

---

::: phydrax.backends.AtomisticKernelLimits

---

::: phydrax.backends.ATOMISTIC_KERNEL_LIMITS

---

::: phydrax.backends.AtomisticKernelAdmission

---

::: phydrax.backends.mace_coupling_problem_kind

---

::: phydrax.backends.MACECouplingRole

---

::: phydrax.backends.AtomisticAccelerationTarget

---

::: phydrax.backends.AtomisticKernelLowering

---

::: phydrax.backends.AtomisticKernelPrecision

---

::: phydrax.backends.WARPGROUP_LANES

## Polynomial geometry providers

The HomotopyContinuation boundary is a host-only numerical polynomial provider.
Callers supply a hash-pinned Julia executable and project containing the supported
package version. Preparation verifies the executable, project, manifest, worker, and
package identities; execution uses a fresh bounded no-shell process. Phydrax never
installs Julia packages, evaluates caller source, or silently falls back to a native
root method.

The Macaulay2 boundary is a separately installed exact symbolic provider. It accepts
only canonical integer, rational, or prime-field sparse polynomial data and a closed
operation enum. The fixed worker has no general evaluator, serialization entry point,
package loader, shell, socket, or numerical-provider route.

Both providers are external software with their own licenses. The process boundary is
operational isolation, not a sandbox.

::: phydrax.backends.HomotopyContinuationProvider

---

::: phydrax.backends.homotopy_continuation_availability

---

::: phydrax.backends.Macaulay2Provider

---

::: phydrax.backends.macaulay2_availability
