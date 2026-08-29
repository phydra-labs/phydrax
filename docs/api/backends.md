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

## Clarabel conic programming

Clarabel 0.11.1 is an optional host interior-point backend for LP, QP, and the
public Phydrax cone product. Install `phydrax[clarabel]` and select
`ClarabelInteriorPoint` explicitly. The provider retains settings and version
evidence; the optimization adapter maps cone and bound layouts, restores rotated-SOC
coordinates, and independently audits original-coordinate KKT residuals and rays.

Clarabel execution transfers program arrays to SciPy CSC on the host. It is not
JIT-compatible and exposes no Phydrax differentiation mode. No native or other
external method is selected after a Clarabel failure.

::: phydrax.backends.ClarabelBackend

---

::: phydrax.backends.ClarabelPlan

---

::: phydrax.backends.PreparedClarabel

---

::: phydrax.backends.clarabel_availability

---

::: phydrax.backends.prepare_clarabel

---

## Spineax cuDSS sparse direct execution

Spineax is an optional Linux x86-64 CUDA 13 bridge to NVIDIA cuDSS. Install the
`cudss` extra only on a supported NVIDIA system, inspect
`spineax_availability`, and select
`SparseLDLT(provider="spineax-cudss")` explicitly. Importing Phydrax does not
import Spineax or initialize CUDA.

The provider accepts sorted canonical 32-bit CSR structure, supports
shared-pattern numeric value batches, retains symbolic analysis across
`phydrax.linalg.refresh`, reuses factors for multiple right-hand sides, and
requires explicit `phydrax.linalg.release`. Original-coordinate residual
certification remains Phydrax-owned.

Spineax reports positive and negative LDLᵀ inertia. Its current cuDSS path does
not claim reliable zero inertia; this limitation is retained in capability and
KKT evidence rather than inferred away. No unavailable or insufficient-inertia
execution falls back to another provider.

Accordingly, the current nonconvex structured IPM rejects Spineax as its KKT
provider until that zero-inertia capability is certified. The provider remains
available for ordinary sparse linear systems.

::: phydrax.backends.SpineaxBackend

---

::: phydrax.backends.spineax_availability

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

PyAMGCL is a host provider. Its result reports iteration or reason data only when the
binding exposes them; missing counters remain absent. AmgX is a device provider. One
prepared hierarchy uploads CSR once, solves multiple right-hand sides without setup
repetition, downloads before independent residual verification, and reports exact
transfer and synchronization evidence. `release_amgx` is idempotent and releases the
solver, vectors, matrix, resources, configuration, and reference-counted global AmgX
runtime.

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
