# Linear algebra runtime

`phydrax.linalg` is the shared finite-dimensional linear algebra substrate for
Phydrax. Control, dynamics, interpolation, sparse derivatives, uncertainty
quantification, and PDE semidiscretizations use the same contracts rather than
maintaining subsystem-specific operator and solve APIs.

The design separates five concerns:

1. **spaces and pairings** define vectors, coordinates, and inner products;
2. **operators** define linear actions, certified properties, and executable
   capabilities;
3. **problems** state exact-system, least-squares, or minimum-norm semantics;
4. **policies and plans** select a feasible provider without probing traced
   values;
5. **prepared artifacts** own reusable numerical state and provenance.

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
  low-rank, diagonal-plus-low-rank, and Schur-complement operators.

Addition, scalar multiplication, and `@` construct lazy sum, scaled, and
composed operators. Materialization is explicit and bounded:

```python
dense = phx.linalg.materialize(
    operator,
    phx.linalg.MaterializationPolicy(max_entries=4, max_bytes=4096),
)
```

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
- a preconditioner;
- differentiation semantics;
- failure behavior;
- independent factorization, workspace, and Krylov-basis byte budgets.

`plan(problem, policy)` returns an immutable `LinearSolvePlan`. Its candidates
contain accepted or rejected cost estimates and reasons. Numerical state is
not stored in the plan.

`LinearCostEstimate` separates already-resident operator storage, additional
materialization, factorization storage, `preparation_workspace_bytes`,
`solve_workspace_bytes_per_rhs`, and `krylov_basis_bytes_per_rhs`.
Operator-batch multiplicity is included. Planning checks that preparation and
one canonical right-hand side are feasible; `solve` and `solve_many` then
multiply both per-RHS estimates by the actual number of shared right-hand sides
before execution. Input and output arrays are caller-owned and are not counted
as solver scratch.

### Provider and method map

| Method | Provider | Main contract |
| --- | --- | --- |
| `StructuredDirect` | `jax-structured` | Recognized exact diagonal, triangular, tridiagonal, banded, block-diagonal, Kronecker, or diagonal-plus-low-rank structure; any dense fallback is included in materialization and resource checks |
| `DenseLU`, `DenseCholesky`, `DenseQR`, `DenseSVD` | `jax-dense` | Explicitly materializable operators within entry, byte, factor, and workspace budgets |
| `SparseDirect` | `jax-sparse` | Canonical unbatched CSR square system on CUDA; native device sparse QR |
| `HostSparseLU` | `host-sparse` | Explicit non-JIT SciPy SuperLU CPU fallback |
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

Device dense, structured, native Krylov, Matfree, Lineax, and supported CUDA
sparse execution are JIT-compatible. `HostSparseLU` is intentionally host-only,
non-JIT, and requires `DifferentiationPolicy("none")`. Public dense
factorizations and matrix-function/spectral artifacts are currently unbatched;
operator-batched dense solves remain supported through `solve`.

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

`prepare_recycling_subspace(A, basis)` applies `A` once to a coarse source
basis, rejects dependent images, and returns image-orthonormal source/image
bases. Its coarse correction is a JAX PyTree operation and can seed a later
solve:

```python
coarse_basis = jnp.eye(operator.source.size)[:, :1]
rhs = jnp.array([1.0, 2.0])
recycling = phx.linalg.prepare_recycling_subspace(operator, coarse_basis)
recycling_prepared = phx.linalg.prepare(
    problem,
    phx.linalg.LinearSolvePolicy(phx.linalg.PCG()),
)
initial_guess = recycling.correction(rhs)
result = phx.linalg.solve(
    recycling_prepared,
    rhs,
    initial_guess=initial_guess,
)
```

Recycling is explicit: Phydrax never carries a stale subspace across a changed
operator identity automatically.

Saddle-point helpers assemble the block operator `[[A, B*], [B, -C]]`, its
`LinearSystem`, and the matrix-free dual Schur complement. The block space and
pairings are retained, and self-adjoint evidence is inferred only when the
primal and stabilization blocks certify it.

## Spectral analytics and matrix functions

The spectral API provides:

- numerical range, operator norm, spectral-radius, condition-number, and
  self-adjoint spectral-bound estimates;
- stochastic trace, log-determinant, diagonal, and inverse-diagonal estimates
  with replayable keys, samples, standard errors, and matvec counts;
- `exp`, `phi1`, `phi2`, trigonometric, logarithm, square root, inverse square
  root, fractional power, and resolvent actions;
- explicit spectral representations and reusable Krylov decompositions.

`MatrixFunctionResult` contains `.value`, convergence, residual and omitted-mode
error estimates, Krylov breakdown status, method, effective dimension, matvec
count, and provenance. Convergence requires finite evidence and an admissible
breakdown state; truncation is not reported as success. Logarithm, square root,
inverse square root, and fractional actions require positive-definite evidence,
explicit bounds, or an explicit spectral representation. Spectral
representations are bound to numerical operator content, not only an
`operator_id`. Matrix functions currently require an unbatched endomorphism.

## Sparse interoperability

`phydrax.sparse.SparseLinearMap` implements the canonical sparse linear-operator
contract. COO-like relations are canonicalized to coalesced CSR storage for
providers. Provider preparation validates finite coefficients, pointer
monotonicity, bounds, ordering, and duplicate freedom, including under JAX
tracing. Invalid storage never reaches a factorization.

`compile_sparse_jacobian` and `compile_sparse_hessian` return reusable derivative
plans. A supplied structural pattern uses the native compiler; omitting it can
invoke ASDEX once for global structure detection and optimized coloring.
Repeated coefficient evaluation and operator application remain native JAX.
The evaluated sparse operators enter the same exact-system, least-squares,
preconditioner, planning, and diagnostics APIs as every other operator.

See [Sparse derivatives](sparse_derivatives.md) for compiler-specific workflows.

## Reproducible benchmark harness

`tools/linalg_benchmarks.py` compares lifecycle costs and steady-state execution
without mixing compilation into warm timings:

```console
python tools/linalg_benchmarks.py \
  --dense-size 256 \
  --right-hand-sides 8 \
  --iterative-size 4096 \
  --repeats 20 \
  --seed 0
```

The command emits one JSON document. `configuration` and `environment` record
the dimensions, repeat count, seed, Python/JAX versions, backend, device, and
64-bit mode. The timed fields have these contracts:

- `phydrax_cold_compile_and_execute_ms` is the first compiled dense solve;
- `prepare_ms` is symbolic planning plus reusable numerical preparation;
- `prepared_reuse_*_ms` is execution of an already prepared and compiled solve;
- `direct_jax_*_ms` is the corresponding warmed direct-JAX reference;
- sparse Jacobian and Hessian `*_compile_ms` measure structural compilation
  once, while `*_evaluation_*_ms` and `*_action_*_ms` are warmed coefficient or
  operator evaluations;
- matrix-free and sparse-solve timings report iterations, forward/adjoint
  operator counts, status, and relative residual with the wall time.

Every timed device result is synchronized before the clock stops. The top-level
`passed` flag requires finite outputs, successful solve statuses, tolerance-level
residuals, and agreement between Phydrax, direct JAX, native sparse, and ASDEX
paths. Compare steady-state values only between reports with compatible
`environment` and `configuration`; cold compilation is intentionally reported
separately.

## Capability limitations

Current boundaries are deliberate and reported before execution:

- iterative providers require an unbatched operator; use explicit outer
  `vmap`/batch policy rather than accidental batch semantics;
- device sparse direct execution currently requires CUDA canonical CSR;
- host sparse LU is non-JIT and non-differentiable;
- public factorization artifacts are unbatched and dense;
- matrix functions and stochastic spectral estimators require unbatched
  endomorphisms;
- Tensor-product spaces require factors with explicit positive coordinate-
  diagonal pairings;
- Schur-complement operators expose only the supplied forward inverse action,
  not an invented transpose or adjoint;
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

::: phydrax.linalg.SchurComplementLinearOperator

---

::: phydrax.linalg.OperatorProperties

---

::: phydrax.linalg.OperatorCapabilities

---

::: phydrax.linalg.MaterializationPolicy

---

::: phydrax.linalg.materialize

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

::: phydrax.linalg.LinearSolvePolicy

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

::: phydrax.linalg.DiagonalPreconditioner

---

::: phydrax.linalg.BlockDiagonalPreconditioner

---

::: phydrax.linalg.IncompleteFactorizationPreconditioner

---

::: phydrax.linalg.LowRankWoodburyPreconditioner

---

::: phydrax.linalg.MultigridPreconditioner

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

::: phydrax.linalg.SpectralMatrixRepresentation

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
