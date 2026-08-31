# Linear algebra runtime

`phydrax.linalg` is the shared finite-dimensional linear algebra substrate for
Phydrax. Control, dynamics, interpolation, sparse derivatives, uncertainty
quantification, and PDE semidiscretizations use the same contracts rather than
maintaining subsystem-specific operator and solve APIs.

The design separates six concerns:

1. **spaces and pairings** define vectors, coordinates, and inner products;
2. **operators** define linear actions, certified properties, and executable
   capabilities;
3. **problems** state exact-system, least-squares, or minimum-norm semantics;
4. **preconditioning** separates setup operators, builders, and prepared
   approximate-inverse actions;
5. **policies and plans** select a feasible provider without probing traced
   values;
6. **prepared artifacts** own reusable numerical state and provenance.

A solve returns `LinearSolveResult`: value, status, diagnostics, and static
provenance. A failed numerical method never masquerades as a successful value.

## First workflow

```python
import jax.numpy as jnp
import phydrax as phx

properties = phx.linalg.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={
        "self_adjoint": "construction",
        "positive_definite": "construction",
        "positive_semidefinite": "construction",
    },
)
operator = phx.linalg.DenseLinearOperator(
    jnp.array([[4.0, 1.0], [1.0, 3.0]]),
    properties=properties,
)
problem = phx.linalg.LinearSystem(operator)
result = phx.linalg.solve(problem, jnp.array([1.0, 2.0]))

assert result.successful
solution = result.value
```

Auto planning chooses dense Cholesky here because the operator is explicit,
positive definite, and its factorization fits the declared resource budgets.
No numerical symmetry or definiteness test is run while tracing.

## Spaces, pairings, and axes

`ArraySpace`, `PyTreeSpace`, `CoordaxSpace`, `BlockSpace`,
`TensorProductSpace`, and `DualSpace` preserve mathematical structure while
exposing canonical coordinates to providers.

Every vector space owns a Riesz pairing. `EuclideanPairing` is the default.
`DiagonalPairing` represents a positive weighted inner product:

```python
space = phx.linalg.ArraySpace(
    (3,),
    dtype=jnp.float64,
    pairing=phx.linalg.DiagonalPairing(jnp.array([1.0, 2.0, 4.0])),
)
```

`transpose(operator)` is the algebraic transpose. `adjoint(operator)` is the
Hilbert adjoint induced by the source and target pairings. They coincide only
when the corresponding Riesz maps make them coincide.

Three axis classes remain distinct:

1. operator batch axes;
2. vector event axes declared by the space;
3. shared trailing right-hand-side axes.

For a batched dense operator with matrix shape `(batch, m, n)`, a target value
has shape `(batch, m)`. An unbatched target `(m,)` broadcasts across the
operator batch. `solve_many` accepts unbatched event axes followed by one or
more shared RHS axes and broadcasts them across an operator batch:

```python
prepared = phx.linalg.prepare(problem)
right_hand_sides = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # event (2,), RHS (2,)
result = phx.linalg.solve_many(prepared, right_hand_sides)
```

When an array shape could denote either an operator batch or trailing RHS axes,
pass `rhs_layout=RHSLayout(...)` to `solve`. Shape inference never overrides an
exact operator-batch match.

## Operators, properties, and capabilities

Common operators include:

- `DenseLinearOperator`, `DiagonalLinearOperator`, and
  `IdentityLinearOperator`;
- `FunctionLinearOperator` and `JacobianLinearOperator` for matrix-free
  actions;
- `BlockLinearOperator`, `StackedLinearOperator`, and
  `BlockDiagonalLinearOperator`;
- triangular, tridiagonal, banded, permutation, Kronecker, Kronecker-sum,
  low-rank, diagonal-plus-low-rank, arbitrary-base-plus-low-rank, and
  Schur-complement operators;
- `TwoSidedScaledLinearOperator` for explicit coordinate scalings and
  `TransformDiagonalLinearOperator` for FFT- or DCT-diagonal actions.
`LocalBlockDiagonalLinearOperator` stores one dense local block per disjoint
canonical block and applies all blocks without constructing a global matrix.
It supports exact structured solves and fixed-size block extraction.

Addition, scalar multiplication, and `@` construct lazy sum, scaled, and
composed operators. Materialization is explicit and bounded:

```python
dense = phx.linalg.materialize(
    operator,
    phx.linalg.MaterializationPolicy(max_entries=4, max_bytes=4096),
)
```
`assemble_diagonal` and `assemble_uniform_blocks` expose exact structural
extraction independently of dense materialization. An operator advertises
diagonal assembly through `OperatorCapabilities.diagonal_assembly`; Jacobi uses
that route directly and falls back to bounded materialization only when the
capability is absent. Uniform blocks use a retained sparse-assembly recipe when
one is supplied, otherwise the extraction is bounded by
`SparseAssemblyPolicy`.


`OperatorProperties` records mathematical claims such as self-adjointness,
positive definiteness, triangularity, and known rank. Evidence is attached per
claim: `construction`, `transformed`, `verified`, `asserted`, or `unknown`.
`properties.certifies(name)` is true only for a claimed property with
non-unknown evidence. Certified positive-definite or positive-semidefinite
evidence also certifies the implied self-adjoint claim.

Derived operators preserve the weakest input evidence: an `unknown` claim is
never upgraded merely by scaling, summing, transposing, taking an adjoint, or
forming a block/Kronecker operator. Certified-only methods, spectral routines,
matrix-function specializations, and backend tags reject such claims.

`OperatorCapabilities` separately records whether transpose, adjoint, and
materialization actions actually exist. A property does not synthesize a
missing executable action.

## Problem semantics

Choose the problem before choosing the algorithm:

- `LinearSystem(A)` means solve the exact square system `A x = b`;
- `LeastSquaresProblem(A, weights=..., regularizer=...)` means minimize a
  weighted residual with an optional zero-target regularizer;
- `MinimumNormProblem(A)` means satisfy an underdetermined exact system while
  minimizing the source-space norm.

Square direct methods may solve an equal-dimensional map between distinct
source and target spaces. Krylov recurrences repeatedly reapply the operator,
so square iterative methods require compatible source and target spaces rather
than silently identifying unrelated coordinate geometries.

Known left and right nullspaces are represented by `LinearSubspace` and
`NullspacePolicy`. Compatibility behavior (`error` or `project`) and gauge
behavior (`minimum-norm` or `project`) are explicit. The diagnostics report the
removed incompatible component, the remaining projected residual, and the
final gauge residual separately.

## Policies and deterministic planning

`LinearSolvePolicy` combines:

- numerical method;
- relative, absolute, and iteration tolerances;
- numerical-rank requirements;
- bounded materialization;
- a `PreconditioningPolicy` with a prepared action or setup-time builder;
- an optional fixed-capacity `RecyclingPolicy`;
- differentiation semantics;
- failure behavior;
- an optional, capability-checked `MixedPrecisionPolicy`;
- independent factorization, workspace, Krylov-basis, preconditioner, and recycling
  byte budgets.

`plan(problem, policy)` returns an immutable `LinearSolvePlan`. Its candidates
contain accepted or rejected cost estimates and reasons. Numerical state is
not stored in the plan.

`prepare_template(problem, policy)` freezes the symbolic selection independently
of coefficients; `bind_numeric(template, problem)` then creates the reusable
`PreparedLinearSolve`. This split is useful when many coefficient realizations
share spaces, operator identity, property evidence, and method configuration.
Binding rejects structural drift rather than silently replanning. The ordinary
`prepare` entry point performs both stages.

### Per-invocation Krylov controls

`LinearSolveControl` changes only dynamic tolerances and the effective step cap for
one prepared native-Krylov invocation:

```python
rhs = jnp.array([1.0, 2.0])
forcing = jnp.asarray(1e-5)
remaining_work = jnp.asarray(2, dtype=jnp.int32)
prepared = phx.linalg.prepare(
    problem,
    phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        tolerance=phx.linalg.TolerancePolicy(max_steps=100),
    ),
)
result = phx.linalg.solve(
    prepared,
    rhs,
    control=phx.linalg.LinearSolveControl(
        relative_tolerance=forcing,
        maximum_steps=remaining_work,
    ),
)
```

The control values may be traced scalars. They cannot increase
`TolerancePolicy.max_steps`, alter a plan, rebuild prepared state, or change
provenance identity. Nonempty controls are capability-rejected for providers whose
compiled loop cannot honor them. This is the solver-facing contract used by adaptive
inexact Newton methods.

### Mixed precision and refinement

`MixedPrecisionPolicy` names the arithmetic requested for operator storage,
factorization, preconditioning, Krylov work, residual certification, and correction
accumulation. A provider must either implement every requested stage or reject the
policy during planning.
Two routes are implemented. Explicit square dense `DenseLU` keeps the operator
and certified residual in stored precision, forms a lower same-kind LU factor,
screens the condition number before factorization, and accumulates iterative
refinement corrections in stored precision. Native `GMRES`/`FGMRES` keeps
operator actions, Arnoldi inner products, Hessenberg solves, convergence
decisions, and final residuals in coordinate precision while optionally storing
Krylov and preconditioned basis vectors in `krylov_dtype`. A Jacobi
preconditioner may independently store/apply its diagonal in
`preconditioner_dtype`; explicit adapters cast residuals into that space and
corrections back.

Planning rejects unsupported methods, providers, real/complex-kind changes, or
wider low-precision stages before execution. Resource estimates use the actual
factor, basis, and preconditioner item sizes, including cast workspace.
Provenance retains both requested and resolved effective dtypes. Other
preconditioner builders, compressed non-GMRES bases, half-precision LU, and
mixed residual/accumulation precision remain capability-rejected.

Standalone Hermitian eigendecompositions, matrix functions, and Sylvester
inverses use `HermitianPrecisionPolicy` rather than pretending they are ordinary
linear solves. It separates compute, eigendecomposition/factorization,
accumulation, certification, and output precision; spectra and Sylvester results
retain effective evidence.

`KernelCertificate` and `SpectralInterval` attach auditable evidence to one
operator structure or numerical value. Numerical certificates include an array
fingerprint and reject changed coefficients. Structural certificates explicitly
assert validity across coefficient refreshes. `ProjectedPCG` is available only
for a certified self-adjoint positive-semidefinite system with an explicit,
nonempty, complete kernel certificate and matching nullspace policy; it solves
on the certified quotient space instead of perturbing a singular operator.

`OperatorActionCostEstimate` reports deduplicated resident operator state and
structural apply scratch per right-hand side. `estimate_operator_action_cost`
walks composite, block, sparse, low-rank, and tensor operators without
materializing them. Its `exact` flag distinguishes fully modeled structural
actions from opaque callbacks, autodiff kernels, or nested inverse actions whose
implementation may require additional scratch.

`LinearCostEstimate` separates already-resident operator storage, additional
materialization, factorization storage, `preparation_workspace_bytes`,
`solve_workspace_bytes_per_rhs`, `operator_apply_workspace_bytes_per_rhs`,
and `krylov_basis_bytes_per_rhs`. Operator-batch multiplicity is included.
Planning checks that preparation and one canonical right-hand side are feasible;
`solve` and `solve_many` multiply all per-RHS estimates by the actual number of
shared right-hand sides before execution. Input and output arrays are
caller-owned and are not counted as solver scratch.

`PreconditionerCostEstimate` separately accounts for prepared action storage,
setup workspace, apply workspace per right-hand side, and setup operator
applications. Candidate costs propagate the last quantity as
`preconditioner_setup_matvec_count`; setup work is therefore visible beside
solve-time operator actions. `SolveResourcePolicy.preconditioner_bytes` bounds
persistent preconditioner state; preparation and apply scratch remain part of
the shared workspace budget. Planning rejects an over-budget preconditioner
before any numeric setup runs.


### Provider and method map

| Method | Provider | Main contract |
| --- | --- | --- |
| `StructuredDirect` | `jax-structured` | Recognized exact diagonal, triangular, tridiagonal, banded, block-diagonal, Kronecker, or diagonal-plus-low-rank structure; any dense fallback is included in materialization and resource checks |
| `DenseLU`, `DenseCholesky`, `DenseQR`, `DenseSVD` | `jax-dense` | Explicitly materializable operators within entry, byte, factor, and workspace budgets |
| `SparseQR(provider="jax-cuda")` | `jax-sparse` | Canonical unbatched CSR square system on CUDA through native JAX sparse QR |
| `SparseLDLT(provider="spineax-cudss")` | `spineax-cudss` | Optional Linux x86-64 CUDA 13 symmetric-indefinite factorization, shared-pattern value batches, numerical refactorization, multiple RHS, reported inertia, and explicit release |
| `SparseLU`, `SparseCholesky`, `SparseQR(provider="spqr")` | `host-sparse` | Explicit non-JIT host sparse direct providers |
| `GMRES`, `PCG`, `MINRES`, `FGMRES`, `GeneralizedLSMR` | `native-krylov` | Pairing-aware native JAX Krylov methods |
| `LSMR` | `matfree` | Real Euclidean unweighted least squares or minimum norm |
| `ConjugateGradient`, `BiCGStab` | `lineax` | Lineax-backed Euclidean or diagonal-metric methods; CG additionally requires real coordinates |

`DenseLU` solves canonical coordinates and accepts any declared Hilbert pairing.
`DenseCholesky`, `DenseQR`, and `DenseSVD` currently require the metric used by
their factor transformation to be Euclidean or coordinate-diagonal. The Lineax
adapter has the same metric restriction. Auto planning routes general pairings
to dense LU when materialization is feasible, or to native `FGMRES` and
`GeneralizedLSMR` otherwise.

`DiagonalPlusLowRankLinearOperator(..., nonsingular_diagonal=True)` validates
the declared diagonal as finite and nonzero and lets planning budget only the
Woodbury path, proportional to the diagonal, low-rank factors, and small core.
Without that certificate, planning reserves a dense fallback for a zero
diagonal; it never assumes nonsingularity from traced values.

`DenseQR` is a full-column-rank least-squares method. It estimates rank from
the singular values of its triangular factor and reports `RANK_DEFICIENT`
rather than presenting an unstable triangular solve as successful. Use
`DenseSVD` when rank-deficient minimizers or a numerical rank cutoff are part
of the intended contract.

Auto selection is deterministic:

1. recognized exact structure;
2. CUDA sparse direct for canonical CSR square systems;
3. dense Cholesky or LU when explicit storage and budgets permit;
4. PCG for certified positive-definite systems;
5. MINRES for certified self-adjoint indefinite systems;
6. FGMRES when a general square system has a preconditioner, otherwise GMRES;
7. dense SVD for explicit rectangular problems when budgets permit;
8. Matfree LSMR for its real-Euclidean envelope;
9. generalized pairing-aware LSMR otherwise.

An explicit infeasible method raises during planning. Phydrax does not silently
change the mathematical problem, discard weights, materialize an operator
outside policy, or reinterpret a failed factorization as another method.

## Preconditioning as a prepared subsystem

Preconditioning has its own plan, preparation, refresh, property, and
provenance contracts. A `PreconditioningPolicy` accepts either:

- an `AbstractPreconditioner`, meaning an already-prepared frozen action; or
- an `AbstractPreconditionerBuilder`, meaning a symbolic recipe that consumes
  a setup operator during `prepare`.

The system operator and setup operator are deliberately separate. This allows a
matrix-free or matrix-expensive operator to remain the true residual action
while a sparse approximation, assembled surrogate, block extraction, or
low-order discretization builds the approximate inverse:

```python
setup = phx.linalg.DiagonalLinearOperator(
    jnp.array([4.0, 3.0]),
    properties=properties,
    operator_id="assembled-setup",
)
policy = phx.linalg.LinearSolvePolicy(
    phx.linalg.PCG(),
    preconditioning=phx.linalg.PreconditioningPolicy(
        phx.linalg.JacobiPreconditionerBuilder(),
        setup_operator=setup,
    ),
)
prepared = phx.linalg.prepare(problem, policy)
```

### Randomized Nyström preconditioning

`RandomizedNystromPreconditionerBuilder` prepares a shifted fixed-rank inverse
from matrix-free actions of an unbatched, certified self-adjoint positive-semidefinite
operator. The setup operator may differ from the solved shifted system. Preparation
uses a deterministic Gaussian sketch, retains a fixed number of Ritz directions, and
records sketch rank, stabilization, captured sketch energy, core conditioning, and
exact setup matvec work.

```text
builder = phx.linalg.RandomizedNystromPreconditionerBuilder(
    24,
    oversampling=8,
    shift=1e-3,
    seed=0,
)
policy = phx.linalg.LinearSolvePolicy(
    phx.linalg.PCG(),
    preconditioning=phx.linalg.PreconditioningPolicy(
        builder,
        setup_operator=positive_semidefinite_operator,
    ),
)
```

The initial implementation requires Euclidean `ArraySpace` or `PyTreeSpace`
coordinates and a strictly positive shift. `probe_refresh="reuse"` recomputes
numeric factors from the same probes; `"redraw"` folds the refresh count into the
seed without changing capacity. It never retries an indefinite small core or falls
back to another preconditioner. Mixed preconditioner storage precision is not yet
supported.

::: phydrax.linalg.RandomizedNystromPreconditionerBuilder

::: phydrax.linalg.RandomizedNystromPreconditioner

::: phydrax.linalg.RandomizedNystromDiagnostics

`PreconditionerProperties` records `linear`, `stationary`, `self_adjoint`, and
`positive_definite` claims with the same evidence vocabulary as operator
properties. Planning uses only certified claims. It never infers solver safety
from a class name or a Boolean without evidence.

| Method | Side | Required preconditioner contract |
| --- | --- | --- |
| `PCG`, Lineax `ConjugateGradient` | left | fixed, linear, self-adjoint, and positive definite |
| `MINRES` | left | fixed, linear, self-adjoint, and positive definite |
| `GMRES` | right | fixed and linear |
| `BiCGStab` | right | fixed and linear |
| `FGMRES` | right | variable, iteration-dependent, or nonlinear actions are allowed |

An incompatible explicit side or property contract fails during planning.
Direct methods and the current least-squares providers reject solve
preconditioning rather than silently ignoring it.

`LinearSolvePlan.preconditioner_plan` records the setup identity, builder or
action identity, side, refresh policy, certified properties, and static cost
estimate. `PreparedLinearSolve.preconditioning_state` owns the prepared action
and its numeric version. Solve provenance reports the preconditioner
plan/action IDs, side, refresh outcome, current solve version, and version at
which the action was built.

Builders default to an explicit refresh policy. `JacobiPreconditionerBuilder`
and `DenseInversePreconditionerBuilder` refresh their numeric state.
`MultigridHierarchyBuilder` freezes the complete hierarchy by default because
its coarse operators are independently owned values; reuse remains visible in
provenance. Callers may request `refresh="rebuild"` only when the stored coarse
levels remain numerically valid, or construct a new hierarchy policy when
coarse coefficients change. If a builder uses a distinct setup operator, a
numerical solve refresh must pass the updated setup explicitly:

```python
changed_operator = phx.linalg.DenseLinearOperator(
    jnp.array([[5.0, 1.0], [1.0, 4.0]]),
    properties=properties,
    operator_id=operator.operator_id,
)
changed_problem = phx.linalg.LinearSystem(
    changed_operator,
    problem_id=problem.problem_id,
)
changed_setup = phx.linalg.DiagonalLinearOperator(
    jnp.array([5.0, 4.0]),
    properties=properties,
    operator_id=setup.operator_id,
)
refreshed = phx.linalg.refresh(
    prepared,
    changed_problem,
    setup_operator=changed_setup,
)
```

Omitting it is an error for a non-frozen distinct setup. A deliberately frozen
prepared action remains reusable, but provenance exposes that reuse.

### Sparse symbolic analysis and incomplete factors

Sparse triangular and factorization lifecycles consume canonical
`AbstractSparseLinearOperator` storage. `analyze_sparse_triangular` builds a fixed
level schedule from the sparsity pattern; `solve_sparse_triangular` changes only
numeric values and right-hand sides. The solve is JAX-native and returns explicit
zero-pivot, nonfinite, and dependency-invalid status rather than repairing a factor.

`SparseFactorizationPolicy` separates ordering and fill construction from numeric
factorization. `prepare_sparse_factorization` computes immutable symbolic LU or
Cholesky routes and binds the first values; `refresh_sparse_factorization` reuses
those routes only when the canonical pattern identity matches. Fill level, numerical
drop tolerance, maximum retained entries per row, diagonal shift, and pivot
replacement are declared policy. A replacement is never enabled implicitly, and
diagnostics report minimum pivot, dropped entries, replacement count, factor
nonzeros, and fill ratio.

`ILUPreconditionerBuilder`, `ILUTPreconditionerBuilder`, and
`IncompleteCholeskyPreconditionerBuilder` expose these factors through the ordinary
prepared-preconditioner lifecycle. Incomplete Cholesky requires certified
self-adjoint structure and only claims positive definiteness when the setup operator
provides that evidence. Symbolic analysis remains host-side; numeric refresh and
triangular actions have static storage under JIT.

`SPARSE_PROVIDER_CATALOG` is an immutable declaration of optional CUDA, SuperLU,
UMFPACK, CHOLMOD, and SPQR capabilities. Availability inspection is deterministic and
never imports or registers a runtime plugin. Host/device placement, transpose support,
complex support, and JIT support remain explicit, so an unavailable provider cannot
silently change the selected algorithm.

### Composable corrections, polynomials, and field splits

`SubspaceCorrectionTerm` couples a restriction, prolongation, and typed local
preconditioner source. `AdditiveSubspaceCorrectionBuilder` covers block Jacobi,
overlapping Schwarz, patch correction, and explicit coarse correction through
one sum of local actions. `MultiplicativeSubspaceCorrectionBuilder` executes a
forward, backward, or symmetric defect-correction sweep. Local setup operators
are derived as `R A P`; no local dense matrix is materialized unless its own
builder and materialization policy permit it.
`BlockJacobiPreconditionerBuilder(block_size, ...)` extracts fixed-size
canonical diagonal blocks, factors them once, and returns a
`LocalBlockPreconditioner`. Dense and structured operators use exact block
assembly; sparse and compositional operators use the same bounded symbolic
sparse recipe as Galerkin construction. Numeric refresh retains the symbolic
routes and refactors only changed coefficients.


`ChebyshevPreconditionerBuilder` prepares a fixed-degree inverse polynomial from
an explicit or estimated positive spectral interval. Optional symmetric Jacobi
scaling transforms the polynomial problem without claiming self-adjointness in
an incompatible pairing. The prepared action has fixed memory and operator
counts, so the same object can serve as a Krylov preconditioner or multigrid
smoother.

`BlockFactorizationPreconditionerBuilder` consumes typed pivot and Schur
preconditioner sources for an explicit 2 by 2 `BlockLinearOperator`. Diagonal,
lower, upper, and LDU forms preserve block PyTree structure. Pivot refresh
precedes Schur reconstruction and Schur-action refresh, preventing stale Schur
state. Triangular forms remain nonsymmetric unless the complete action has
independent valid evidence.

### Explicit multigrid hierarchy

`MultigridLevelBuilder` owns a level operator, restriction/prolongation pair,
smoother or coarse-solve builder, and pre/post smoothing counts.
`MultigridHierarchyBuilder` validates every transfer space before preparing
immutable `MultigridLevel` and `MultigridHierarchy` values. The resulting
`MultigridPreconditioner` executes one JAX-native V-cycle; hierarchy setup stays
outside the compiled iteration loop.

The coarsest level is an ordinary preconditioner source, so a small
`DenseInversePreconditionerBuilder` can provide an exact coarse solve without a
special solver path. Symmetry and positive-definiteness of a full V-cycle must
be supplied as explicit `PreconditionerProperties`; they are not inferred from
individual level names.
The hierarchy derives linearity and stationarity conservatively from every
level source. An explicit whole-cycle certificate cannot override an
incompatible variable or nonlinear level contract.

`GalerkinHierarchyBuilder` derives coarse operators as `R A P` around the same
immutable V-cycle. It plans canonical sparse products once, retains each
`PreparedSparseAssembly` in the hierarchy, and refreshes coarse coefficients
through those symbolic routes. Dense construction is an explicit bounded
fallback; a matrix-free route remains matrix-free at downstream levels.
`SmoothedAggregationHierarchyBuilder` constructs deterministic aggregates,
candidate-aware tentative interpolation, damped Jacobi smoothing,
pairing-aware restrictions, and the same planned sparse Galerkin products for
explicit dense or canonical sparse inputs. `MultigridSetupDiagnostics` reports
level dimensions, known nonzero counts, grid/operator complexity, prepared
bytes, peak setup workspace, construction mode, transfer identities, retained
aggregate assignments, the builder-dependency fingerprint, and every reuse
decision.

Hierarchy refresh distinguishes full rebuild, aggregate reuse, transfer reuse,
and symbolic sparse-product reuse. Every numeric refresh recomputes
fine-dependent coarse coefficients. Pattern or builder-dependency changes
invalidate structural reuse. A Galerkin route rejected by cumulative
materialization limits remains permanently matrix-free, so a downstream level
builder cannot rematerialize it. Symbolic sparse-product reuse requires the
retained route map; a mode name never licenses stale coarse values.


`multigrid_hierarchy_from_pyamg` is an optional host-only converter. It consumes
a supplied PyAMG multilevel solver, converts its SciPy level and transfer
matrices into canonical Phydrax sparse operators, prepares JAX Jacobi smoothers
and a dense coarse inverse, and returns ordinary Phydrax hierarchy values.
PyAMG is never called by the compiled V-cycle.

## Preparation, repeated solves, and refresh

Use the lifecycle API when structure is reused:

```python
policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseCholesky())
selected = phx.linalg.plan(problem, policy)
prepared = phx.linalg.prepare(problem, selected)
first = phx.linalg.solve(prepared, jnp.array([1.0, 2.0]))

changed_operator = phx.linalg.DenseLinearOperator(
    jnp.array([[5.0, 0.5], [0.5, 2.0]]),
    properties=properties,
    operator_id=operator.operator_id,
)
changed_problem = phx.linalg.LinearSystem(
    changed_operator,
    problem_id=problem.problem_id,
)
refreshed = phx.linalg.refresh(prepared, changed_problem)
second = phx.linalg.solve(refreshed, jnp.array([1.0, 2.0]))
```

`refresh` preserves the symbolic plan and increments `numeric_version`. It
rejects changed spaces, operator structure, weights, regularizers, nullspace
policies, method configuration, or property/capability evidence. Use stable,
meaningful `operator_id` and `problem_id` values when coefficients are expected
to change.

`factorize` exposes a reusable `PreparedFactorization` when users need rank,
singular values, determinant or pseudodeterminant, nullspaces, or transformed
solves in addition to `solve`. `refresh_factorization` applies the same symbolic
identity rule and numerical versioning.

## JIT behavior

Planning and preparation are host-side lifecycle operations. They validate
static structure and allocate reusable numerical state. Prepare before a hot
loop, then pass the artifact through `jax.jit`:

```python
import jax

prepared = phx.linalg.prepare(problem, policy)
compiled_solve = jax.jit(lambda rhs: phx.linalg.solve(prepared, rhs).value)
value = compiled_solve(jnp.array([1.0, 2.0]))
```

Device dense, structured, native Krylov, Matfree, Lineax, native CUDA sparse
QR, and optional Spineax/cuDSS execution are JIT-compatible. Host sparse
providers are intentionally non-JIT and require
`DifferentiationPolicy("none")`. Spineax preparation retains provider-owned
factor resources; call `phx.linalg.release(prepared)` when their lifetime ends.
Public dense factorizations and matrix-function/spectral artifacts are currently
unbatched; shared-pattern sparse LDLT values may carry explicit batch axes.

All iterative methods use fixed-capacity states with dynamic iteration counts
and breakdown status, so compiled shapes do not depend on convergence.
Diagnostic history and residual-check frequency are explicit policy costs.

`LinearSolveDiagnostics.matvec_count` and `adjoint_matvec_count` report the
provider algorithm's actual top-level forward and adjoint operator applications,
including its initialization and final checks. Direct methods report zero. The
separate provider-neutral residual verification performed after a solve is not
charged to these counters.

## Differentiation semantics

`DifferentiationPolicy` distinguishes four contracts:

| Mode | Meaning |
| --- | --- |
| `mathematical` | Differentiate the exact mathematical solution map with an implicit linear/root rule, independent of finite iteration history |
| `rhs-only` | Use the mathematical rule while stopping gradients through operator/problem coefficients |
| `algorithmic` | Differentiate the actually executed finite algorithm where the selected provider exposes that path |
| `none` | Stop gradients through the solve result |

Unsupported provider/mode combinations fail in planning. In particular, a
provider is never treated as algorithmically differentiable merely because its
forward pass is JIT-compatible. Transpose and adjoint rules use the declared
pairings and conjugation semantics.

The implicit rule supports one or many right-hand sides and operator-batched
dense solves. `rhs-only` stops every problem coefficient while retaining the
right-hand-side derivative. Minimum-norm differentiation uses the same
source-space pairing and active numerical-rank subspace as the forward solve.

Prepared nonlinear linearizations use `prepare_linearization`, then
`JacobianLinearOperator`. The artifact stores one primal evaluation plus JVP and
VJP actions, so repeated products do not retrace an opaque callable. Sparse
Jacobian and Hessian plans use the same prepared-linearization contract.

## Matrix-free Krylov, block methods, and recycling

`phydrax.linalg.krylov` exposes reusable Arnoldi, Lanczos, Golub–Kahan, and block
Arnoldi decompositions. Artifacts retain projected operators, dynamic effective
dimension, orthogonality error, matvec counts, and breakdown status. Block
Arnoldi performs rank deflation instead of inserting invalid normalized
columns.

`BlockGMRES` and `BlockCG` are explicit true-block methods. A nonempty
`RHSLayout` becomes part of the solve plan and prepared identity; execution must
match it exactly. True-block methods form one shared block Krylov space, use
rank-revealing deflation for dependent residual columns, and report effective
block rank separately from operator/problem rank. Ordinary scalar methods
continue to use the distinct pseudo-block path for multiple right-hand sides.
Auto planning does not select a block method without an explicit RHS contract.

`prepare_recycling_subspace(A, basis)` remains the low-level coarse-correction
primitive. Solver-integrated recycling uses the pure functional
`solve_recycled` API: it returns an ordinary solve result together with a new
immutable `RecyclingState`. The state stores source vectors and their current
orthonormal images, capacities, plan/operator identities, numeric version, and
update count.

GCRO-DR solves first apply the retained coarse correction, project the residual
away from the retained image space, build an augmented Krylov space, and
extract a bounded harmonic-Ritz recycle space. `refresh_recycling` may reuse
source vectors across a coefficient-only refresh, but always recomputes their
images. Structure or plan changes reject the state. Extraction and rank
decisions are algorithmic state and are stopped from mathematical
differentiation.
`solve_recycled` accepts the same `LinearSolveControl` as an ordinary native Krylov
solve, so nonlinear forcing and aggregate iteration limits remain dynamic while the
recycling capacity stays structural.
Planning charges both the retained recycling state and the transient augmented
Arnoldi, search-basis, pseudoinverse, and harmonic-Ritz eigensolve storage before
preparation.

Saddle-point helpers assemble the block operator `[[A, B*], [B, -C]]`, its
`LinearSystem`, and the matrix-free dual Schur complement. The block space and
pairings are retained, and self-adjoint evidence is inferred only when the
primal and stabilization blocks certify it.

## Eigen, SVD, Schur, and invariant subspaces

### Self-adjoint eigenproblems

`phydrax.linalg.eigen` provides a sibling plan/prepare/refresh/result lifecycle
for standard and generalized self-adjoint eigenproblems. `LOBPCG` handles
blocked smallest or largest modes, including an SPD generalized metric and an
excluded `LinearSubspace`. `RestartedLanczos` supplies a thick-restart
matrix-free alternative. `DenseEigh` provides the bounded Phydrax-native full
dense route for standard and generalized Hermitian problems. Generalized
problems use a certified Cholesky reduction before the JAX Hermitian solve.

Every route retains per-mode residuals, convergence masks, pairing-aware
orthogonality error, effective count, operator/metric application counts,
status, cost, and provenance. Generalized eigenvectors are normalized in the
declared metric; preparation rejects an uncertified or numerically invalid
positive-definite metric.

Eigenvalue differentiation is deliberately narrower than dense `eigh`.
`EigenSolvePolicy(differentiation="none")` stops all outputs. Eigenvalue-only
mode requires a certified converged isolated simple mode and uses the standard
or generalized pairing-aware derivative while stopping eigenvectors. Repeated
or unresolved clusters reject individual gradients rather than returning
values. Algorithmic differentiation through locking, deflation, ordering, and
restart decisions is not exposed.

Closure-converted `FunctionLinearOperator` actions may capture differentiable
arrays. Filtered derivative boundaries preserve their callable and static
leaves, so singular-value, isolated-eigenvalue, projector, density-kernel, and
smooth spectral-function derivatives support these matrix-free problems under
JIT, JVP, and reverse mode. The same convergence, isolation, selection-gap, and
domain requirements still apply.

### Prepared self-adjoint spectra and differentiable spectral calculus

`prepare_self_adjoint_spectrum` materializes one bounded dense standard or
generalized self-adjoint problem, solves its complete spectrum once, and retains
the result for repeated subspace, projector, density-kernel, and smooth
spectral-function evaluations. `refresh_self_adjoint_spectrum` rebuilds only
the numerical spectrum while preserving the symbolic problem and plan
identities. Full-spectrum preparation rejects excluded constraints, partial
dense capacity, nonconvergence, nonfinite output, an invalid generalized
metric, or a retained-state budget violation; it never switches to an
iterative or broadened-eigenvector fallback.

For a generalized problem `A x = λ B x`, let `R` be the source-space Riesz map,
`G = R B`, and let the retained coordinate basis satisfy `Vᴴ G V = I`. The
prepared inverse basis is therefore `V⁻¹ = Vᴴ G`. For selected columns `Vₛ`,
the invariant projector and covariant density kernel are
`P = Vₛ Vₛᴴ G` and `D = Vₛ Vₛᴴ`, with `P = D G`. This distinction is observable
for non-Euclidean pairings and generalized metrics.

`self_adjoint_spectral_subspace` applies a `SpectralSelection` to that prepared
spectrum and returns fixed-shape selected/complement eigenvalues and bases,
`P`, `D`, the certified selected/complement gap, residual evidence, status, and
provenance. `expected_dimension` is required so JIT output shapes do not depend
on data. Repeated eigenvalues inside either block are valid; a cluster crossing
the selection boundary is rejected. `self_adjoint_spectral_projector_derivative`
computes the exact selected/complement cross-block Sylvester derivative of
`P` and `D`, including perturbations of the generalized metric. It does not
differentiate individual eigenvectors.

```python
problem = phx.linalg.eigen.Eigenproblem(operator)
prepared = phx.linalg.eigen.prepare_self_adjoint_spectrum(problem)
selection = phx.linalg.eigen.SpectralSelection.real_below(
    3.0,
    expected_dimension=1,
)
subspace = phx.linalg.eigen.self_adjoint_spectral_subspace(
    prepared,
    selection,
    policy=phx.linalg.eigen.SelfAdjointSpectralSubspacePolicy(
        differentiation="projector",
    ),
)
```

Smooth functions use the basis-invariant Loewner Fréchet rule. In the prepared
eigenbasis, off-diagonal entries use
`(f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ)`, while coincident values use the declared
`f′(λᵢ)`. Exact repeated eigenvalues are therefore regular for matrix
exponentials, logarithms, square roots, inverse square roots, fractional
powers, resolvents, polynomials, and finite-temperature Fermi–Dirac
occupations. Trainable polynomial coefficients, chemical potential, and
temperature contribute their ordinary parameter tangents in addition to the
operator perturbation. Logarithm, square-root, inverse-square-root, fractional
power, and resolvent domain failures are explicit statuses; eigenvalues are
never clipped to manufacture a value.

`self_adjoint_spectral_operator` returns the operator matrix, density kernel,
trace, function values, residual/domain diagnostics, status, and provenance.
Its policy chooses `"none"` or exact `"frechet"` differentiation. Raw
`eigensolve(..., differentiation="eigenvalues")` remains the isolated
eigenvalue-only API: individual eigenvectors are still stopped, and batched raw
eigenvalue differentiation is rejected rather than silently applying a
different contract.

Dense `DenseEigh` preparation, solve, diagnostics, prepared-spectrum
consumers, and exact spectral derivatives support arbitrary leading operator
batch axes. Every batch member has independent residuals, convergence flags,
selection counts, gaps, domain validity, and status; selected dimensions and
all array shapes remain static across the batch. Resource estimates multiply
retained storage, workspace, materialization, and action counts by the batch
cardinality. Iterative `LOBPCG` and `RestartedLanczos` routes continue to reject
operator batches during planning.

### Pairing-aware singular values

`phydrax.linalg.svd` exposes `SVDProblem`, `SVDSolvePolicy`, and the same
plan/prepare/refresh/result lifecycle for an unbatched map between possibly
different source and target Hilbert spaces. Dense preparation applies the
declared Riesz maps, computes the requested largest or smallest triplets, maps
the vectors back to their spaces, and reports both residual directions and both
pairing-orthogonality errors. Singular-value-only differentiation requires
isolated retained values; vector derivatives are not exposed.

### General dense Schur problems and spectral projectors

`SchurEigenproblem` handles a general real or complex dense endomorphism. Its
host-side preparation computes an ordered complex Schur form with explicit
unitarity, residual, finite-value, separation, resource, refresh, and backend
evidence. `schur_spectral_observables` derives determinant, trace, spectral
radius/abscissa, numerical abscissa, nonnormality, and continuous/discrete-time
stability without pretending a nonnormal matrix has an orthogonal eigenbasis.

`SpectralSelection` defines a protected half-plane or disk cluster.
`prepare_spectral_subspace` constructs its right basis, left dual basis, Riesz
projector, orthogonal projector, exact-or-bounded Sylvester separation, and
projector-condition evidence. A Riesz projector for a nonnormal operator is not
generally orthogonal. `spectral_projector_derivative` solves the differentiated
Sylvester equations and returns commutator and projector-tangent residuals.
Refresh preserves the selected dimension and rejects eigenvalue crossings.

## Reusable projections, shifted systems, and rational actions

`prepare_krylov_projection(A, v, policy)` binds one fixed-capacity Arnoldi or
Lanczos basis to the complete numerical operator and starting-vector
fingerprints. Matrix-function calls can consume that projection repeatedly
without applying `A` again. `refresh_krylov_projection` rebuilds only the
numerical basis while preserving the symbolic plan and capacity.

`ShiftedLinearSystemFamily` represents all systems `(z_j I - A) x_j = b`.
`prepare_shifted_solve` builds one shared Krylov projection; `solve_shifted`
performs only the projected solves and returns per-shift residual, rank,
conditioning, status, and shared setup cost. The pole-minus-operator convention
is explicit and is also used by `PartialFractionRationalFunction`:

```python
right_hand_side = jnp.array([1.0, -1.0])
family = phx.linalg.ShiftedLinearSystemFamily(operator, jnp.array([2.0, 3.0]))
prepared = phx.linalg.prepare_shifted_solve(family, right_hand_side)
shifted = phx.linalg.solve_shifted(prepared)

rational = phx.linalg.PartialFractionRationalFunction(
    jnp.array([2.0, 3.0]),
    jnp.array([0.5, -0.25]),
    polynomial_coefficients=jnp.array([1.0]),
)
action = phx.linalg.rational_function_action(operator, right_hand_side, rational)
```

Rational preparation shares the shifted basis across all poles and accounts
separately for polynomial operator applications. Plans bound retained storage,
transient workspace, and total operator applications; refresh rejects changed
pole count, polynomial degree, spaces, or operator structure.

## Adaptive stochastic spectral estimation

`adaptive_stochastic_trace` and `adaptive_stochastic_log_determinant` execute
fixed-capacity probe batches under `jax.jit` while stopping dynamically at
batch boundaries. Their result distinguishes statistical uncertainty from
Lanczos projection error and reports the combined error, tolerance, active
probe count, per-probe status, iterations, and exact forward/adjoint matvec
counts. The policy fixes minimum and maximum probes, batch size, projection
dimension, confidence level, and absolute/relative stopping tolerances.
Self-adjoint evidence is mandatory; log determinants additionally require
positive-definite evidence.

## Linear matrix equations

`MatrixEquationTerm` represents `c A X B`; `MatrixEquationProblem` sums such
terms in row-major coordinates. Convenience constructors cover Sylvester and
continuous/discrete Lyapunov equations. Planning lowers the matrix equation to
an ordinary Phydrax linear problem, so provider selection, resources,
factorization reuse, status, provenance, JIT execution, and differentiation all
follow the shared runtime:

```python
A = jnp.array([[-1.0, 0.2], [0.0, -2.0]])
Q = jnp.eye(2)
problem = phx.linalg.continuous_lyapunov_equation(A, Q)
prepared = phx.linalg.prepare_matrix_equation(problem)
result = phx.linalg.solve_matrix_equation(prepared)
```

Residuals are evaluated against the original matrix equation. A declared
self-adjoint solution is checked explicitly and receives a separate structure
status rather than being symmetrized after the fact.

### Factored continuous Lyapunov equations

For a large equation `A X + X A* = -B B*`, the factored path accepts `B` directly
and returns `X` as a fixed-capacity `Z Z*` factor:

```python
factored_space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
stable_diagonal = jnp.asarray([-1.0, -2.0, -4.0, -8.0])
operator = phx.linalg.FunctionLinearOperator(
    lambda value: stable_diagonal * value,
    source=factored_space,
    target=factored_space,
    properties=phx.linalg.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    ),
    operator_id="stable-generator",
)
problem = phx.linalg.factored_continuous_lyapunov_equation(
    operator,
    jnp.ones((4, 1)),
    problem_id="factored-gramian",
)
policy = phx.linalg.FactoredMatrixEquationPolicy(
    (-1.0, -2.0, -4.0, -8.0),
    shifted=phx.linalg.ShiftedSolvePolicy("lanczos", max_dimension=4),
    maximum_rank=4,
)
prepared = phx.linalg.prepare_factored_matrix_equation(problem, policy)
result = phx.linalg.solve_factored_matrix_equation(prepared)
assert bool(result.successful)
```

The low-rank ADI lifecycle never forms `B B*`, materializes the operator, or
constructs dense `X`. It delegates each shift to the reusable shifted-Krylov
contract, compresses the accumulated factor, and reports effective/raw rank,
truncation loss, per-shift status/residual/iterations, factor storage, avoided
explicit-solution storage, and an original-equation Frobenius residual computed by
a low-rank Gram identity. Numeric refresh preserves the symbolic shifted plan.

This route currently supports only unbatched Euclidean vector spaces and continuous
Lyapunov structure with explicit open-left-half-plane shifts. Real-coordinate
operators require real shifts. Generalized, Sylvester, discrete, dense-forcing, and
general `U V*` equation solves are rejected without a dense fallback; the general
`FactoredMatrixSolution` type is a representation contract, not a claim that those
solvers exist.

## Low-rank updates, transformation, and resilience

`BasePlusLowRankLinearOperator` represents `B + U C V*` for any reusable base
operator. Its dedicated plan/prepare/refresh/solve lifecycle factors the base
once and then applies the Woodbury correction with explicit base
nonsingularity, correction conditioning, rank, storage, workspace, residual,
and provenance policies. It does not densify the base as an implicit fallback.

`EquilibrationPolicy` provides no scaling, explicit two-sided scaling, Ruiz
scaling, or symmetric Ruiz scaling. `ResilientSolvePolicy` composes that
transformation with an ordinary `LinearSolvePolicy` and bounded iterative
refinement. `solve_resilient` verifies the residual in the original coordinates
and reports the initial/final residual, backward error, refinement history,
condition estimates, scaling spread, base status, and terminal resilience
status.

`compile_linear_structure` is an explicit host-side optimizer for materializable
endomorphisms. In declared candidate order it can recognize exact diagonal,
permutation, tridiagonal, triangular, banded, DCT-diagonal, or FFT-diagonal
structure, otherwise applying the configured dense/error fallback. Approximate
projection requires explicit consent and nonzero tolerances. The result records
the projected error, route, exactness, operator identity, and numeric version;
`refresh_linear_structure` preserves the selected variant or rejects structural
drift.

## Spectral analytics and matrix functions

The spectral API provides:

- numerical range, operator norm, spectral-radius, condition-number, and
  self-adjoint spectral-bound estimates;
- fixed-count and adaptive stochastic trace/log-determinant estimates, plus
  diagonal and inverse-diagonal estimates;
- `exp`, `phi1`, `phi2`, trigonometric, logarithm, square root, inverse square
  root, fractional power, and resolvent actions;
- explicit spectral representations and reusable Krylov projections.

`MatrixFunctionResult` contains `.value`, convergence, residual and omitted-mode
error estimates, Krylov breakdown status, method, effective dimension, matvec
count, and provenance. Convergence requires finite evidence and an admissible
breakdown state; truncation is not reported as success. Logarithm, square root,
inverse square root, and fractional actions require positive-definite evidence,
explicit bounds, or an explicit spectral representation. Spectral
representations and reusable projections are bound to numerical operator
content, not only an `operator_id`. Matrix functions currently require an
unbatched endomorphism.

## Sparse interoperability

`phydrax.sparse.SparseLinearMap` implements the canonical sparse linear-operator
contract. COO-like relations are canonicalized to coalesced CSR storage for
providers. Provider preparation validates finite coefficients, pointer
monotonicity, bounds, ordering, and duplicate freedom, including under JAX
tracing. Invalid storage never reaches a factorization.
`plan_sparse_assembly` recognizes sparse leaves and exact algebraic recipes for
supported identity, diagonal, permutation, banded, local-block, scaled, summed,
composed, transpose, and adjoint operators. `SparseAssemblyPolicy` independently
bounds nonzeros, resident bytes, contribution count, symbolic workspace, and
any explicit fallback. `prepare_sparse_assembly` evaluates one immutable CSR
operator while retaining its symbolic recipe; `refresh_sparse_assembly`
accepts only the same operator identity, spaces, and structural pattern, then
updates coefficients without rebuilding routes.


`compile_sparse_jacobian` and `compile_sparse_hessian` return reusable derivative
plans. A supplied structural pattern uses the native compiler; omitting it can
invoke ASDEX once for global structure detection and optimized coloring.
Repeated coefficient evaluation and operator application remain native JAX.
The evaluated sparse operators enter the same exact-system, least-squares,
preconditioner, planning, and diagnostics APIs as every other operator.

See [Sparse derivatives](sparse_derivatives.md) for compiler-specific workflows.

## Reproducible benchmark harnesses

`tools/linalg_benchmarks.py` measures the core solve, preconditioner, sparse
derivative, sparse assembly, and structured-direct paths:

```console
python tools/linalg_benchmarks.py \
  --dense-size 256 \
  --right-hand-sides 8 \
  --iterative-size 4096 \
  --repeats 20 \
  --seed 0
```

`tools/linalg_advanced_benchmarks.py` measures the reusable Krylov, shared
shifted/rational, matrix-equation, spectral-projector derivative, arbitrary-base
low-rank, resilient equilibration/refinement, and adaptive stochastic paths:

```console
python tools/linalg_advanced_benchmarks.py \
  --size 64 \
  --shift-count 12 \
  --repeats 20 \
  --seed 0
```

Both commands emit one JSON document. `configuration` and `environment` record
dimensions, repeat count, seed, Python/JAX versions, backend, and device.
Preparation is timed separately from warmed execution. Every timed device
result is synchronized before the clock stops. Each advanced record includes a
dense, exact, finite-difference, or invariant reference as applicable, status
and diagnostics, and an individual `passed` flag. The top-level `passed` flag
requires every workload to satisfy its numerical contract. `--smoke` on the
advanced harness executes a small single-repeat end-to-end check.

Compare steady-state values only between reports with compatible `environment`
and `configuration`; preparation and cold compilation are intentionally not
presented as warmed execution.

## Capability limitations

Current boundaries are deliberate and reported before execution:

- iterative providers require an unbatched operator; use explicit outer
  `vmap`/batch policy rather than accidental batch semantics;
- native device sparse QR requires CUDA canonical CSR;
- Spineax/cuDSS requires an explicitly installed optional backend on Linux
  x86-64 with CUDA 13, uses 32-bit canonical CSR indices, and reports
  positive/negative inertia without claiming reliable zero inertia;
- host sparse direct providers are non-JIT and non-differentiable;
- provider-owned sparse LDLT resources require explicit release;
- public dense factorization artifacts remain unbatched;
- matrix functions and stochastic spectral estimators require unbatched
  endomorphisms;
- mixed-precision execution supports capability-checked dense `DenseLU`
  refinement plus native `GMRES`/`FGMRES` compressed bases and lower-precision
  Jacobi preconditioning; other stage/provider combinations are rejected;
- factored matrix-equation execution currently covers only unbatched continuous
  Lyapunov equations with explicit ADI shifts;
- Tensor-product spaces require factors with explicit positive coordinate-
  diagonal pairings;
- Schur-complement operators expose only the supplied forward inverse action,
  not an invented transpose or adjoint;
- distributed execution is not exposed without a reproducible multi-device
  workload and benchmark environment; the current runtime has no local/ghost
  vector abstraction;
- sparse-direct provider extensibility remains deliberately static until a
  second concrete backend is selected; current providers are device JAX sparse
  and host SciPy SuperLU;
- mathematical property claims require per-property evidence and are never
  inferred from sampled traced values.

## Migration from the former solver-local API

The linear algebra namespace is a clean cutover:

| Former usage | Current usage |
| --- | --- |
| `phx.solver.AbstractLinearOperator` and solver-local operators | `phx.linalg.AbstractLinearOperator` and `phx.linalg.*LinearOperator` |
| `phx.solver.matrix_exponential_action(...)` | `phx.linalg.matrix_exponential_action(...).value` |
| `phx.solver.MatrixFunctionPolicy(..., num_matvecs=n)` | `phx.linalg.MatrixFunctionPolicy(..., max_dimension=n)` |
| `update(prepared, problem)` | `phx.linalg.refresh(prepared, problem)` |
| one global `evidence="construction"` string | a mapping from each claimed property to its evidence |
| subsystem-specific sparse or Jacobian solve wrappers | one `LinearSystem`, `LeastSquaresProblem`, or `MinimumNormProblem` passed to `phx.linalg.solve` |

No deprecated aliases remain. Migrate call sites atomically rather than mixing
old and new plan identities.

## API reference

### Spaces and pairings

::: phydrax.linalg.AbstractVectorSpace

---

::: phydrax.linalg.ArraySpace

---

::: phydrax.linalg.PyTreeSpace

---

::: phydrax.linalg.BlockSpace

---

::: phydrax.linalg.TensorProductSpace

---

::: phydrax.linalg.CoordaxSpace

---

::: phydrax.linalg.DualSpace

---

::: phydrax.linalg.AbstractPairing

---

::: phydrax.linalg.EuclideanPairing

---

::: phydrax.linalg.DiagonalPairing

---

::: phydrax.linalg.RHSLayout

### Operators and structure

::: phydrax.linalg.AbstractLinearOperator

---

::: phydrax.linalg.DenseLinearOperator

---

::: phydrax.linalg.FunctionLinearOperator

---

::: phydrax.linalg.JacobianLinearOperator

---

::: phydrax.linalg.BlockLinearOperator

---

::: phydrax.linalg.KroneckerLinearOperator

---

::: phydrax.linalg.KroneckerSumLinearOperator

---

::: phydrax.linalg.LowRankLinearOperator

---

::: phydrax.linalg.DiagonalPlusLowRankLinearOperator

---

::: phydrax.linalg.BasePlusLowRankLinearOperator

---

::: phydrax.linalg.SymmetricLowRankLinearOperator

---

::: phydrax.linalg.TwoSidedScaledLinearOperator

---

::: phydrax.linalg.TransformDiagonalLinearOperator

---

::: phydrax.linalg.StructureCompilationPolicy

---

::: phydrax.linalg.StructureCompilationResult

---

::: phydrax.linalg.compile_linear_structure

---

::: phydrax.linalg.refresh_linear_structure

---

::: phydrax.linalg.LocalBlockDiagonalLinearOperator

---

::: phydrax.linalg.SchurComplementLinearOperator

---

::: phydrax.linalg.OperatorProperties

---

::: phydrax.linalg.OperatorCapabilities

---

::: phydrax.linalg.MaterializationPolicy

---

::: phydrax.linalg.materialize

---

::: phydrax.linalg.assemble_diagonal

---

::: phydrax.linalg.assemble_uniform_blocks

---

::: phydrax.linalg.OperatorActionCostEstimate

---

::: phydrax.linalg.estimate_operator_action_cost

### Empirical feature geometry

`EmpiricalGramLinearOperator` applies a weighted feature Gram or centered covariance
without materializing a parameter-by-parameter matrix. Its feature operator maps a
parameter tangent into an array with one leading sample axis. Zero weights are valid
masks; weights are normalized internally, and damping is applied in the declared
source-space pairing.

For centered features `J`, weights `W`, and sample-centering map `C`, the action is
`J† C† W C J + λI`. The result is an ordinary `AbstractLinearOperator` and therefore
uses the existing linear-system, minimum-norm, nullspace, planning, and diagnostic
runtime.

::: phydrax.linalg.EmpiricalGramLinearOperator

---


### Problems, policies, and runtime

::: phydrax.linalg.LinearSystem

---

::: phydrax.linalg.LeastSquaresProblem

---

::: phydrax.linalg.MinimumNormProblem

---

::: phydrax.linalg.LinearSubspace

---

::: phydrax.linalg.NullspacePolicy

---

::: phydrax.linalg.KernelCertificate

---

::: phydrax.linalg.SpectralInterval

---

::: phydrax.linalg.ProjectedPCG

---

::: phydrax.linalg.LinearSolveTemplate

---

::: phydrax.linalg.prepare_template

---

::: phydrax.linalg.bind_numeric

---

::: phydrax.linalg.LinearSolvePolicy

---


::: phydrax.linalg.LinearSolveControl

---

::: phydrax.linalg.MixedPrecisionPolicy

---
::: phydrax.linalg.HermitianPrecisionPolicy

---

::: phydrax.linalg.LinearPrecisionEvidence

---

---

::: phydrax.linalg.BlockGMRES

---

::: phydrax.linalg.BlockCG

---

::: phydrax.linalg.TolerancePolicy

---

::: phydrax.linalg.RankPolicy

---

::: phydrax.linalg.DifferentiationPolicy

---

::: phydrax.linalg.FailurePolicy

---

::: phydrax.linalg.SolveResourcePolicy

---

::: phydrax.linalg.LinearSolvePlan

---

::: phydrax.linalg.PreparedLinearSolve

---

::: phydrax.linalg.LinearSolveResult

---

::: phydrax.linalg.plan

---

::: phydrax.linalg.prepare

---

::: phydrax.linalg.refresh

---

::: phydrax.linalg.solve

---

::: phydrax.linalg.solve_many

---

::: phydrax.linalg.solve_transpose

---

::: phydrax.linalg.solve_adjoint

### Preconditioners and factorizations

::: phydrax.linalg.AbstractPreconditioner

---

::: phydrax.linalg.PreconditionerProperties

---

::: phydrax.linalg.PreconditioningPolicy

---

::: phydrax.linalg.AbstractPreconditionerBuilder

---

::: phydrax.linalg.PreconditionerPlan

---

::: phydrax.linalg.PreparedPreconditioner

---

::: phydrax.linalg.PreconditionerCostEstimate

---

::: phydrax.linalg.LinearCostEstimate

---

::: phydrax.linalg.ChebyshevPreconditionerBuilder

---

::: phydrax.linalg.ChebyshevPreconditioner

---

::: phydrax.linalg.SubspaceCorrectionTerm

---

::: phydrax.linalg.AdditiveSubspaceCorrectionBuilder

---

::: phydrax.linalg.AdditiveSubspaceCorrectionPreconditioner

---

::: phydrax.linalg.MultiplicativeSubspaceCorrectionBuilder

---

::: phydrax.linalg.MultiplicativeSubspaceCorrectionPreconditioner

---

::: phydrax.linalg.BlockFactorizationPreconditionerBuilder

---

::: phydrax.linalg.BlockFactorizationPreconditioner

---

::: phydrax.linalg.JacobiPreconditionerBuilder

---

::: phydrax.linalg.BlockJacobiPreconditionerBuilder

---

::: phydrax.linalg.LocalBlockPreconditioner

---

::: phydrax.linalg.DenseInversePreconditionerBuilder

---

::: phydrax.linalg.DiagonalPreconditioner

---

::: phydrax.linalg.BlockDiagonalPreconditioner

---

::: phydrax.linalg.IncompleteFactorizationPreconditioner

---

::: phydrax.linalg.SparseTriangularAnalysis

---

::: phydrax.linalg.SparseTriangularFactor

---

::: phydrax.linalg.analyze_sparse_triangular

---

::: phydrax.linalg.solve_sparse_triangular

---

::: phydrax.linalg.SparseFactorizationPolicy

---

::: phydrax.linalg.SparseFactorizationPlan

---

::: phydrax.linalg.PreparedSparseFactorization

---

::: phydrax.linalg.prepare_sparse_factorization

---

::: phydrax.linalg.refresh_sparse_factorization

---

::: phydrax.linalg.factorize_sparse

---

::: phydrax.linalg.SparseFactorizationPreconditioner

---

::: phydrax.linalg.SparseFactorizationPreconditionerBuilder

---

::: phydrax.linalg.ILUPreconditionerBuilder

---

::: phydrax.linalg.ILUTPreconditionerBuilder

---

::: phydrax.linalg.IncompleteCholeskyPreconditionerBuilder

---

::: phydrax.linalg.SparseProviderCapabilities

---

::: phydrax.linalg.SparseProviderAvailability

---

::: phydrax.linalg.available_sparse_providers

---

::: phydrax.linalg.LowRankWoodburyPreconditioner

---

::: phydrax.linalg.MultigridPreconditioner

---

::: phydrax.linalg.MultigridLevelBuilder

---

::: phydrax.linalg.MultigridLevel

---

::: phydrax.linalg.MultigridHierarchyBuilder

---

::: phydrax.linalg.MultigridHierarchy

---

::: phydrax.linalg.multigrid_hierarchy_from_pyamg

---

::: phydrax.linalg.GalerkinHierarchyBuilder

---

::: phydrax.linalg.SmoothedAggregationHierarchyBuilder

---

::: phydrax.linalg.MultigridSetupDiagnostics

---

::: phydrax.linalg.SparseAssemblyPolicy

---

::: phydrax.linalg.SparseAssemblyPlan

---

::: phydrax.linalg.PreparedSparseAssembly

---

::: phydrax.linalg.plan_sparse_assembly

---

::: phydrax.linalg.prepare_sparse_assembly

---

::: phydrax.linalg.refresh_sparse_assembly

---

::: phydrax.linalg.OperatorPreconditioner

---

::: phydrax.linalg.FactorizationPolicy

---

::: phydrax.linalg.PreparedFactorization

---

::: phydrax.linalg.factorize

---

::: phydrax.linalg.refresh_factorization

### Linearization, recycling, and saddle points

::: phydrax.linalg.LinearizationPolicy

---

::: phydrax.linalg.PreparedLinearization

---

::: phydrax.linalg.prepare_linearization

---

::: phydrax.linalg.RecyclingSubspace

---

::: phydrax.linalg.prepare_recycling_subspace

---

::: phydrax.linalg.saddle_point_operator

---

::: phydrax.linalg.saddle_point_system

---

::: phydrax.linalg.saddle_point_schur_complement

### Spectral and matrix-function APIs

::: phydrax.linalg.MatrixFunctionPolicy

---

::: phydrax.linalg.MatrixFunctionResult

---

::: phydrax.linalg.TransformDiagonalRepresentation

---

::: phydrax.linalg.matrix_function_action

---

::: phydrax.linalg.matrix_exponential_action

---

::: phydrax.linalg.matrix_phi1_action

---

::: phydrax.linalg.estimate_numerical_range

---

::: phydrax.linalg.estimate_operator_norm

---

::: phydrax.linalg.estimate_spectral_bounds

---

::: phydrax.linalg.estimate_spectral_radius

---

::: phydrax.linalg.estimate_condition_number

---

::: phydrax.linalg.stochastic_trace

---

::: phydrax.linalg.stochastic_log_determinant

---

::: phydrax.linalg.estimate_diagonal

---

::: phydrax.linalg.estimate_inverse_diagonal

### Reusable Krylov projections

::: phydrax.linalg.KrylovProjectionPolicy

---

::: phydrax.linalg.KrylovProjectionResourcePolicy

---

::: phydrax.linalg.KrylovProjectionPlan

---

::: phydrax.linalg.PreparedKrylovProjection

---

::: phydrax.linalg.plan_krylov_projection

---

::: phydrax.linalg.prepare_krylov_projection

---

::: phydrax.linalg.refresh_krylov_projection

### Shifted systems and rational functions

::: phydrax.linalg.ShiftedLinearSystemFamily

---

::: phydrax.linalg.ShiftedSolvePolicy

---

::: phydrax.linalg.ShiftedSolvePlan

---

::: phydrax.linalg.PreparedShiftedSolve

---

::: phydrax.linalg.ShiftedSolveResult

---

::: phydrax.linalg.plan_shifted_solve

---

::: phydrax.linalg.prepare_shifted_solve

---

::: phydrax.linalg.refresh_shifted_solve

---

::: phydrax.linalg.solve_shifted

---

::: phydrax.linalg.PartialFractionRationalFunction

---

::: phydrax.linalg.RationalFunctionPolicy

---

::: phydrax.linalg.RationalFunctionPlan

---

::: phydrax.linalg.PreparedRationalFunctionAction

---

::: phydrax.linalg.RationalFunctionResult

---

::: phydrax.linalg.plan_rational_function_action

---

::: phydrax.linalg.prepare_rational_function_action

---

::: phydrax.linalg.refresh_rational_function_action

---

::: phydrax.linalg.rational_function_action

### Adaptive stochastic estimators

::: phydrax.linalg.AdaptiveStochasticPolicy

---

::: phydrax.linalg.AdaptiveStochasticEstimate

---

::: phydrax.linalg.adaptive_stochastic_trace

---

::: phydrax.linalg.adaptive_stochastic_log_determinant

### Low-rank and resilient solves

::: phydrax.linalg.LowRankSolvePolicy

---

::: phydrax.linalg.LowRankSolvePlan

---

::: phydrax.linalg.PreparedLowRankSolve

---

::: phydrax.linalg.LowRankSolveResult

---

::: phydrax.linalg.plan_low_rank_solve

---

::: phydrax.linalg.prepare_low_rank_solve

---

::: phydrax.linalg.refresh_low_rank_solve

---

::: phydrax.linalg.solve_low_rank

---

::: phydrax.linalg.EquilibrationPolicy

---

::: phydrax.linalg.RefinementPolicy

---

::: phydrax.linalg.ResilientSolvePolicy

---

::: phydrax.linalg.ResilientSolvePlan

---

::: phydrax.linalg.PreparedResilientSolve

---

::: phydrax.linalg.ResilientSolveResult

---

::: phydrax.linalg.plan_resilient_solve

---

::: phydrax.linalg.prepare_resilient_solve

---

::: phydrax.linalg.refresh_resilient_solve

---

::: phydrax.linalg.solve_resilient

### Matrix equations

::: phydrax.linalg.MatrixEquationTerm

---

::: phydrax.linalg.MatrixEquationProblem

---

::: phydrax.linalg.MatrixEquationPolicy

---

::: phydrax.linalg.MatrixEquationPlan

---

::: phydrax.linalg.PreparedMatrixEquation

---

::: phydrax.linalg.MatrixEquationResult

---

::: phydrax.linalg.plan_matrix_equation

---

::: phydrax.linalg.prepare_matrix_equation

---

::: phydrax.linalg.refresh_matrix_equation

---

::: phydrax.linalg.solve_matrix_equation

---

::: phydrax.linalg.sylvester_equation

---

::: phydrax.linalg.continuous_lyapunov_equation

---

::: phydrax.linalg.discrete_lyapunov_equation

---

::: phydrax.linalg.FactoredMatrixSolution

---

::: phydrax.linalg.FactoredMatrixEquationProblem

---

::: phydrax.linalg.FactoredMatrixEquationPolicy

---

::: phydrax.linalg.FactoredMatrixEquationCostEstimate

---

::: phydrax.linalg.FactoredMatrixEquationPlan

---

::: phydrax.linalg.PreparedFactoredMatrixEquation

---

::: phydrax.linalg.FactoredMatrixEquationResidualCertificate

---

::: phydrax.linalg.FactoredMatrixEquationDiagnostics

---

::: phydrax.linalg.FactoredMatrixEquationProvenance

---

::: phydrax.linalg.FactoredMatrixEquationResult

---

::: phydrax.linalg.FactoredMatrixEquationStatus

---

::: phydrax.linalg.factored_continuous_lyapunov_equation

---

::: phydrax.linalg.plan_factored_matrix_equation

---

::: phydrax.linalg.prepare_factored_matrix_equation

---

::: phydrax.linalg.refresh_factored_matrix_equation

---

::: phydrax.linalg.solve_factored_matrix_equation

### Self-adjoint eigenproblems

::: phydrax.linalg.eigen.Eigenproblem

---

::: phydrax.linalg.eigen.GeneralizedEigenproblem

---

::: phydrax.linalg.eigen.DenseEigh

---

::: phydrax.linalg.eigen.LOBPCG

---

::: phydrax.linalg.eigen.RestartedLanczos

---

::: phydrax.linalg.eigen.EigenSolvePolicy

---

::: phydrax.linalg.eigen.EigenSolvePlan

---

::: phydrax.linalg.eigen.PreparedEigenSolve

---

::: phydrax.linalg.eigen.EigenSolveResult

---

::: phydrax.linalg.eigen.plan_eigensolve

---

::: phydrax.linalg.eigen.prepare_eigensolve

---

::: phydrax.linalg.eigen.refresh_eigensolve

---

::: phydrax.linalg.eigen.eigensolve

### Trial-subspace Rayleigh--Ritz and warm starts

`block_rayleigh_trace` evaluates a Hermitian block quotient with a native
Cholesky solve and explicit mass-rank, conditioning, Hermitian, and finite-value
evidence. `solve_reduced_ritz` feeds the same matrices to the ordinary dense
generalized eigensolver.

`rayleigh_ritz` accepts coordinate columns in any Phydrax vector space,
projects declared excluded constraints, applies the physical operator and
metric, solves the reduced pencil, lifts its modes, and reports full-space
absolute and relative residuals. `warm_started_eigensolve` first records that
trial-space certificate, then refines its Ritz vectors through the selected
authoritative eigensolver.

::: phydrax.linalg.eigen.BlockRayleighEvaluation

---

::: phydrax.linalg.eigen.block_rayleigh_trace

---

::: phydrax.linalg.eigen.solve_reduced_ritz

---

::: phydrax.linalg.eigen.TrialSubspaceRitzResult

---

::: phydrax.linalg.eigen.rayleigh_ritz

---

::: phydrax.linalg.eigen.WarmStartedEigenResult

---

::: phydrax.linalg.eigen.warm_started_eigensolve


### Self-adjoint spectra, invariant subspaces, and spectral functions

::: phydrax.linalg.eigen.SelfAdjointSpectrumPolicy

---

::: phydrax.linalg.eigen.SelfAdjointSpectrumPlan

---

::: phydrax.linalg.eigen.PreparedSelfAdjointSpectrum

---

::: phydrax.linalg.eigen.plan_self_adjoint_spectrum

---

::: phydrax.linalg.eigen.prepare_self_adjoint_spectrum

---

::: phydrax.linalg.eigen.refresh_self_adjoint_spectrum

---

::: phydrax.linalg.eigen.self_adjoint_spectrum

---

::: phydrax.linalg.eigen.SelfAdjointSpectralSubspacePolicy

---

::: phydrax.linalg.eigen.SelfAdjointSpectralSubspace

---

::: phydrax.linalg.eigen.SelfAdjointSpectralDerivativeResult

---

::: phydrax.linalg.eigen.self_adjoint_spectral_subspace

---

::: phydrax.linalg.eigen.self_adjoint_spectral_projector_derivative

---

::: phydrax.linalg.eigen.AbstractSpectralFunction

---

::: phydrax.linalg.eigen.PolynomialSpectralFunction

---

::: phydrax.linalg.eigen.FermiDiracSpectralFunction

---

::: phydrax.linalg.eigen.ExponentialSpectralFunction

---

::: phydrax.linalg.eigen.LogarithmSpectralFunction

---

::: phydrax.linalg.eigen.SquareRootSpectralFunction

---

::: phydrax.linalg.eigen.InverseSquareRootSpectralFunction

---

::: phydrax.linalg.eigen.FractionalPowerSpectralFunction

---

::: phydrax.linalg.eigen.ResolventSpectralFunction

---

::: phydrax.linalg.eigen.SelfAdjointSpectralOperatorPolicy

---

::: phydrax.linalg.eigen.SelfAdjointSpectralOperator

---

::: phydrax.linalg.eigen.self_adjoint_spectral_operator

### Dense Schur problems and spectral subspaces

::: phydrax.linalg.eigen.SchurEigenproblem

---

::: phydrax.linalg.eigen.SchurSolvePolicy

---

::: phydrax.linalg.eigen.PreparedSchurSolve

---

::: phydrax.linalg.eigen.SchurSolveResult

---

::: phydrax.linalg.eigen.prepare_schur_eigensolve

---

::: phydrax.linalg.eigen.refresh_schur_eigensolve

---

::: phydrax.linalg.eigen.schur_eigensolve

---

::: phydrax.linalg.eigen.schur_spectral_observables

---

::: phydrax.linalg.eigen.SpectralSelection

---

::: phydrax.linalg.eigen.SpectralSubspacePolicy

---

::: phydrax.linalg.eigen.PreparedSpectralSubspace

---

::: phydrax.linalg.eigen.SpectralSubspace

---

::: phydrax.linalg.eigen.prepare_spectral_subspace

---

::: phydrax.linalg.eigen.refresh_spectral_subspace

---

::: phydrax.linalg.eigen.spectral_subspace

---

::: phydrax.linalg.eigen.spectral_projector_derivative

### Singular value decompositions

::: phydrax.linalg.svd.SVDProblem

---

::: phydrax.linalg.svd.DenseSVD

---

::: phydrax.linalg.svd.SVDSolvePolicy

---

::: phydrax.linalg.svd.SVDSolvePlan

---

::: phydrax.linalg.svd.PreparedSVDSolve

---

::: phydrax.linalg.svd.SVDSolveResult

---

::: phydrax.linalg.svd.plan_svd

---

::: phydrax.linalg.svd.prepare_svd

---

::: phydrax.linalg.svd.refresh_svd

---

::: phydrax.linalg.svd.svd
