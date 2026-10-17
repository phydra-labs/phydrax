# External solver backends

`phydrax.backends` is the explicit boundary between Phydrax problem contracts and
optional solver runtimes. Importing `phydrax` does not require MPAX, PETSc, SLEPc,
PyAMGCL, CUDA, or AmgX. A provider is imported only when its availability or
preparation function is called.

External providers are never candidates of the native automatic linear policy.
Selecting one is an explicit user decision; an unavailable provider raises
`BackendUnavailableError` with the backend, capability, requirement, and import or
linker evidence. No external path silently materializes a matrix-free operator,
changes provider, or falls back to a native solve.

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
cuSPARSE handles per device; switching devices does not recreate those handles.
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
bridge must match JAX's CUDA major and is checked before import, even for
host-staged use. See the [verified CUDA 13 build recipe](../installation.md#gpu-direct-mpi-interoperability)
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
(and deployment-specific flags) through `PHYDRAX_MPI_LAUNCHER`.

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
| `TPU_JAX_PROFILE` | TPU | single-device, multi-device, multi-host |
| `INTEL_JAX_PROFILE` | XPU | experimental single/multi-device |
| `APPLE_METAL_PROFILE` | Metal | experimental single-device, float16/float32 |

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
architecture, compiled GPU architectures, CUDA compiler version), so CUDA 12 and
CUDA 13 companions, or development and release builds, never share a tune cache
that QUDA would reject. An explicit resource path is preserved. The owned exit
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

## cuDSS (native)

The Linux x86-64 CUDA companions provide cuDSS for both `phydrax[cuda12]` and
`phydrax[cuda13]`. Inspect `cudss_availability`, then select
`SparseLDLT(provider="cudss", factor_cache_capacity=8)` explicitly.
Importing Phydrax does not load the companion or initialize CUDA.

The provider accepts sorted canonical 32-bit CSR structure and real float32 or
float64 symmetric systems. It supports shared-pattern numeric value batches,
retains symbolic analysis across `phydrax.linalg.refresh`, reuses factors for
multiple right-hand sides, and supports optional `phydrax.linalg.release` to
retire memo entries early.
Refinement and original-coordinate residual certification remain JAX-side.

Each plan has a separate bounded native factor-cache partition; the positive
`factor_cache_capacity` participates in policy and plan identity. Evicted state
is rebuilt from call operands, so release or eviction does not invalidate a
prepared value. Solve provenance reports exact int32 `native_rebuilds`.
Use `cuda_native_resources()` to inspect live entries, capacities, and rebuilds.

cuDSS reports positive and negative LDLᵀ inertia, but does not claim reliable
zero inertia. Accordingly, the current nonconvex structured IPM rejects cuDSS as
its KKT provider until that capability is certified. No unavailable or
insufficient-inertia execution falls back to another provider.

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
| `"cuda"` | `"pallas_mosaic_gpu"` | Active JAX GPU backend on the NVIDIA CUDA client with compute capability 9.0 or newer. Other GPU clients, such as ROCm, and older devices are refused. |
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
