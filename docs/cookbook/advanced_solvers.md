# Advanced solver workflows

This page composes Phydrax's numerical contracts without hiding planning,
preparation, convergence, or transfer boundaries. The examples are deliberately
small; the same lifecycle applies to large matrix-free and sparse problems.

## 1. Reuse a native linear solve

Declare mathematical operator properties with evidence, then separate symbolic
planning from numerical preparation. Stable `operator_id` and `problem_id` values
allow a same-structure coefficient refresh.

```python
import jax
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
    jnp.asarray([[4.0, 1.0], [1.0, 3.0]]),
    properties=properties,
    operator_id="diffusion-block",
)
system = phx.linalg.LinearSystem(operator, problem_id="diffusion-system")
policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseCholesky())
plan = phx.linalg.plan(system, policy)
prepared = phx.linalg.prepare(system, plan)

compiled_solve = jax.jit(lambda rhs: phx.linalg.solve(prepared, rhs))
first = compiled_solve(jnp.asarray([1.0, 2.0]))
assert bool(first.successful)

changed_operator = phx.linalg.DenseLinearOperator(
    jnp.asarray([[5.0, 0.5], [0.5, 2.0]]),
    properties=properties,
    operator_id=operator.operator_id,
)
changed_system = phx.linalg.LinearSystem(
    changed_operator,
    problem_id=system.problem_id,
)
refreshed = phx.linalg.refresh(prepared, changed_system)
second = phx.linalg.solve(refreshed, jnp.asarray([1.0, 2.0]))
assert bool(second.successful)
assert int(second.provenance.operator_numeric_version) == 1
```

Planning uses only declared structure and policy. Refresh rejects changed spaces,
sparsity, problem semantics, or property evidence; it is not a request to re-plan.
For Krylov methods, use a `PreconditioningPolicy` to keep the physical action
operator distinct from the setup operator and prepared approximate inverse.

## 2. Solve a nonlinear PyTree system with JFNK

`NonlinearSystemProblem` owns the physical residual and accepted-state predicate.
`NewtonKrylov` uses matrix-free Jacobian products and a residual-merit line search by
default. The result preserves distinct residual, globalization, and inner-linear
failure states.

```python
problem = phx.nonlinear.NonlinearSystemProblem(
    lambda state, target: {
        "temperature": state["temperature"] ** 2 - target,
    },
    validity=lambda state, residual, auxiliary, target: (
        state["temperature"] >= 0.0
    ),
    problem_id="positive-square-root",
)
method = phx.nonlinear.NewtonKrylov(
    forcing_policy=phx.nonlinear.NewtonForcingPolicy("eisenstat-walker"),
    jacobian_refresh=phx.nonlinear.JacobianRefreshPolicy("stagnation"),
)
termination = phx.nonlinear.NonlinearTermination(
    absolute_residual=1e-8,
    relative_residual=0.0,
    maximum_linear_iterations=100,
)
prepared_root = phx.nonlinear.prepare_nonlinear(
    problem,
    {"temperature": jnp.asarray(1.0)},
    method=method,
    termination=termination,
    args=jnp.asarray(2.0),
)
root = phx.nonlinear.solve_prepared_nonlinear(prepared_root)
assert bool(root.successful)

changed_root = phx.nonlinear.refresh_nonlinear(
    prepared_root,
    problem,
    {"temperature": jnp.asarray(1.0)},
    args=jnp.asarray(3.0),
)
next_root = phx.nonlinear.solve_prepared_nonlinear(changed_root)
assert bool(next_root.successful)
```

Prepare linear or nonlinear structure outside transformed hot loops. A matrix-free
problem remains matrix-free unless the caller selects a resource-bounded explicit
Jacobian policy. Phydrax does not silently switch to a dense direct correction after
a Krylov failure.

### Compose bounded nonlinear work

Use a typed update when an inner method should propose progress without claiming
root convergence. The outer method still owns termination and certifies the
physical residual:

```python
half_step = phx.nonlinear.FunctionNonlinearUpdate(
    lambda state, target: jax.tree.map(
        lambda value, expected: value + 0.5 * (expected - value),
        state,
        target,
    ),
    update_id="half-correction",
)
composite = phx.nonlinear.CompositeNonlinearUpdate(
    (half_step, half_step),
    kind="multiplicative",
)
accelerated = phx.nonlinear.NonlinearGMRES(composite)
result = accelerated.solve(
    phx.nonlinear.NonlinearSystemProblem(
        lambda state, target: state - target
    ),
    jnp.asarray([0.0]),
    args=jnp.asarray([2.0]),
    termination=phx.nonlinear.NonlinearTermination(maximum_steps=10),
)
assert bool(result.successful)
```

For a complementarity operator that is undefined outside its declared bounds,
select strict trial projection explicitly:

```python
vi = phx.nonlinear.VariationalInequalityProblem(
    lambda state, _: jnp.where(state < 0.0, jnp.nan, state - 1.0),
    phx.nonlinear.Bounds(0.0, jnp.inf),
)
vi_result = phx.nonlinear.SemismoothNewton(
    feasibility="preserve-box"
).solve(
    vi,
    jnp.asarray([-1.0]),
)
assert bool(vi_result.successful)
```

### Select a root family from mathematical evidence

```python
scalar = phx.nonlinear.ScalarRootProblem(
    lambda x, target: x * x - target,
    bracket=(0.0, 2.0),
)
scalar_result = phx.nonlinear.scalar_root(
    scalar,
    method=phx.nonlinear.TOMS748(),
    args=2.0,
)

system = phx.nonlinear.NonlinearSystemProblem(
    lambda state, target: state * state - target
)
robust_result = phx.nonlinear.RobustRoot().solve(
    system,
    jnp.ones((2,)),
    args=jnp.asarray([4.0, 9.0]),
    termination=phx.nonlinear.NonlinearTermination(
        maximum_steps=100,
        maximum_evaluations=1000,
        maximum_linear_iterations=5000,
    ),
)
assert bool(scalar_result.successful)
assert bool(robust_result.successful)
```

### Build robust block least squares

```python
parameter = phx.optim.ParameterBlock(
    lambda values: values["location"],
    lambda values, replacement: {"location": replacement},
    block_id="location",
)
factor = phx.optim.ResidualBlock(
    lambda values, observations: values[0] - observations,
    ("location",),
    loss=phx.optim.HuberLoss(1.0),
    block_id="observations",
)
graph = phx.optim.ResidualGraphProblem((parameter,), (factor,))
fit = phx.optim.least_squares(
    graph.as_least_squares_problem(),
    {"location": jnp.asarray([0.0])},
    args=jnp.asarray([1.0]),
    method=phx.optim.LevenbergMarquardt(),
)
assert bool(fit.successful)
```

## 3. Select an optimization method by contract

Unconstrained second-order optimization, nonlinear least squares, bounds,
nonlinear constraints, proximal objectives, state/design systems, and stochastic
programs are different contracts. They share typed termination, diagnostics, status,
and provenance rather than a common bag of options.

A bound-constrained PyTree problem can use projected L-BFGS:

```python
bounded = phx.optim.MinimizationProblem(
    lambda value, target: jnp.sum((value - target) ** 2),
    bounds=phx.optim.Bounds(-2.0, 1.0),
    problem_id="bounded-fit",
)
fit = phx.optim.minimize(
    bounded,
    jnp.asarray([0.0, 0.0]),
    method=phx.optim.ProjectedLBFGS(),
    args=jnp.asarray([0.5, 2.0]),
)
assert bool(fit.successful)
assert fit.certificate is not None
```

For general equality and inequality constraints, declare each canonical source once:

```python
sum_to_one = phx.optim.NonlinearConstraint(
    lambda value, _: jnp.sum(value),
    lower=1.0,
    upper=1.0,
    constraint_id="sum-to-one",
)
nonnegative = phx.optim.Bounds(0.0, jnp.inf)
constrained = phx.optim.MinimizationProblem(
    lambda value, _: jnp.sum((value - jnp.asarray([0.2, 0.8])) ** 2),
    bounds=nonnegative,
    constraints=(sum_to_one,),
    problem_id="simplex-projection",
)
solution = phx.optim.minimize(
    constrained,
    jnp.asarray([0.5, 0.5]),
    method=phx.optim.SQP(),
)
assert bool(solution.successful)
assert solution.certificate is not None
```

Use `PrimalDualNewtonKrylov` or `PrimalDualPredictorCorrector` when a matrix-free KKT
solve and explicit primal, multiplier, slack, and complementarity evidence are
required. Use `ReducedAdjoint`, `ReducedNewtonKrylov`, or `SimultaneousKKT` for
state/design systems; the reduced methods require an explicit state solver and do
not differentiate through an opaque iterative history.

## 4. Compute paired general eigenvectors and target interior modes

A general eigenproblem is either standard, `A r = λ r`, or generalized,
`A r = λ B r`. Results keep right and left vectors paired and verify both residuals
in the original pencil. Dense Schur/QZ is the complete-spectrum path; restarted
Arnoldi is the partial matrix-free path.

```python
values = jnp.asarray([0.0, 2.0, 5.0, 8.0, 11.0])
eigen_operator = phx.linalg.DenseLinearOperator(
    jnp.diag(values),
    operator_id="interior-spectrum",
)
eigenproblem = phx.linalg.eigen.GeneralEigenproblem(eigen_operator)
selection = phx.linalg.eigen.GeneralEigenSelection.closest(5.2, 1)
policy = phx.linalg.eigen.GeneralEigenSolvePolicy(
    phx.linalg.eigen.DenseSchurQZ(),
    selection=selection,
)
eigen_result = phx.linalg.eigen.general_eigensolve(eigenproblem, policy=policy)
assert bool(eigen_result.successful)
assert jnp.allclose(eigen_result.eigenvalues, jnp.asarray([5.0]))
```

`ShiftInvertTransform` and `CayleyTransform` require a linear-solve policy for their
inner actions; they do not form an inverse. A partial Arnoldi result has an explicit
partial status and converged count. A simple-eigenpair derivative consumes a solved
and verified pair plus perturbation operators. Repeated-cluster derivatives instead
require the invariant-projector API; no individual derivative is fabricated for a
multiple eigenvalue.

## 5. Traverse and localize a fold

Natural parameter continuation cannot pass a turning point. Pseudo-arclength
continuation augments the residual with an arclength condition and uses a bordered
linear correction. It can detect and bracket a candidate event; that detection is
not itself a fold/Hopf/pitchfork certificate.

```python
fold_problem = phx.continuation.ParameterContinuationProblem(
    lambda state, coordinate, _: {
        "x": state["x"] ** 2 + coordinate - 1.0,
    },
    problem_id="quadratic-fold",
)
branch_result = phx.continuation.continue_branch(
    fold_problem,
    {"x": jnp.asarray(1.0)},
    jnp.asarray(0.0),
    num_steps=12,
    method=phx.continuation.PseudoArclengthContinuation(
        initial_step=0.18,
        maximum_step=0.24,
        residual_tolerance=1e-8,
    ),
)
assert branch_result.status == phx.continuation.ContinuationStatus.SUCCESS
bracket = branch_result.fold_brackets[0]

localized = phx.continuation.localize_event(
    fold_problem,
    branch_result.branch,
    bracket,
    lambda problem, state, coordinate, args: state["x"],
    indicator_id="quadratic-fold/state-zero",
    policy=phx.continuation.EventLocalizationPolicy(
        bracket_tolerance=1e-6,
        indicator_tolerance=1e-6,
        residual_tolerance=1e-8,
        maximum_steps=16,
    ),
)
assert localized.status == phx.continuation.EventLocalizationStatus.SUCCESS
```

For repeated same-structure runs, use `plan_continuation`, `prepare_continuation`,
`run_continuation`, and `refresh_continuation`. `ParameterPathContinuationProblem`
maps the scalar continuation coordinate to a validated physical-parameter PyTree and
stores both the physical parameters and tangent parameters at every branch point.
`GeneralKrylovStabilityAnalyzer` uses the public general-eigen Arnoldi contract for
nonsymmetric matrix-free Jacobians. Explicit fold, Hopf, and pitchfork workflows add
augmented solves and problem-specific assumptions/certificates.

## 6. Build one canonical sparse system for optional providers

External assembled providers consume the public sparse boundary. The following
constructs canonical coordinate storage without a private adapter:

```python
matrix = jnp.asarray(
    [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]]
)
rows, columns = jnp.nonzero(matrix)
relation = phx.sparse.EdgeRelation(
    columns,
    rows,
    source_size=3,
    target_size=3,
)
space = phx.linalg.ArraySpace((3,), dtype=matrix.dtype)
sparse_operator = phx.sparse.SparseCoordinateOperator(
    relation,
    matrix[rows, columns],
    source=space,
    target=space,
    operator_id="three-point-stencil",
)
sparse_system = phx.linalg.LinearSystem(
    sparse_operator,
    problem_id="three-point-system",
)
rhs = jnp.asarray([1.0, 0.0, 1.0])
```

The same `sparse_system` can be sent explicitly to PETSc, PyAMGCL, or AmgX. Probe
availability before preparing; package import remains dependency-free.

### PETSc KSP

```python
availability = phx.backends.petsc_availability()
if availability.available:
    petsc_plan = phx.backends.plan_petsc_linear(
        sparse_system,
        phx.backends.PETScKSPPolicy(
            ksp_type="gmres",
            pc_type="ilu",
            reuse_preconditioner=True,
        ),
    )
    petsc_prepared = phx.backends.prepare_petsc_linear(petsc_plan)
    petsc_result = phx.backends.solve_petsc_linear(petsc_prepared, rhs)
    print(petsc_result.status, petsc_result.diagnostics.residual_norm)
else:
    print(availability.requirement, availability.reason)
```

PETSc's action matrix and preconditioning matrix may be supplied independently. A
numeric refresh preserves symbolic identities and exposes whether PETSc reused or
rebuilt the preconditioner. SNES is a separate nonlinear lifecycle; its default
Jacobian mode is matrix-free, while dense autodiff is an explicit resource-guarded
choice.

### SLEPc EPS

```python
availability = phx.backends.slepc_availability()
if availability.available:
    slepc_problem = phx.linalg.eigen.GeneralEigenproblem(sparse_operator)
    slepc_policy = phx.backends.SLEPcEigenPolicy(
        phx.linalg.eigen.GeneralEigenSelection("largest-real", count=2),
        operator_mode="shell",
    )
    slepc_plan = phx.backends.plan_slepc_eigensolve(
        slepc_problem,
        slepc_policy,
    )
    slepc_prepared = phx.backends.prepare_slepc_eigensolve(
        slepc_problem,
        slepc_plan,
    )
    try:
        slepc_result = phx.backends.slepc_eigensolve(slepc_prepared)
        print(slepc_result.status, slepc_result.eigenvalues)
    finally:
        phx.backends.release_slepc_eigensolve(slepc_prepared)
```

The shell mode applies the Phydrax operator and its adjoint without materialization.
SLEPc shift-invert and Cayley instead require `operator_mode="csr"` and explicit
`SLEPcSTOptions`; an incompatible shell request is rejected.

### PyAMGCL and AmgX

```python
pyamgcl = phx.backends.pyamgcl_availability()
if pyamgcl.available:
    pyamgcl_plan = phx.backends.plan_pyamgcl(sparse_system)
    pyamgcl_prepared = phx.backends.prepare_pyamgcl(
        sparse_system,
        pyamgcl_plan,
    )
    pyamgcl_result = phx.backends.solve_pyamgcl(pyamgcl_prepared, rhs)

amgx = phx.backends.amgx_availability()
if amgx.available:
    amgx_plan = phx.backends.plan_amgx(
        sparse_system,
        phx.backends.AmgXPolicy(),
    )
    amgx_prepared = phx.backends.prepare_amgx(sparse_system, amgx_plan)
    try:
        amgx_result = phx.backends.solve_amgx(amgx_prepared, rhs)
        print(amgx_result.provenance.transfer)
    finally:
        phx.backends.release_amgx(amgx_prepared)
```

PyAMGCL is host-only. AmgX owns GPU resources and requires explicit idempotent
release. Its result records exact input upload bytes, output download bytes, and
synchronizations. All external results retain provider termination evidence and an
independently recomputed Phydrax residual; iteration count alone never certifies
success.

## 7. Keep a continuous Lyapunov solution factored

Large controllability and covariance equations should not require dense forcing or
solution matrices. Supply the source factor directly and keep the operator
matrix-free:

```python
gramian_space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
stable_diagonal = jnp.asarray([-1.0, -2.0, -4.0, -8.0])
stable_operator = phx.linalg.FunctionLinearOperator(
    lambda value: stable_diagonal * value,
    source=gramian_space,
    target=gramian_space,
    properties=phx.linalg.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    ),
    operator_id="stable-diagonal-generator",
)
factored_problem = phx.linalg.factored_continuous_lyapunov_equation(
    stable_operator,
    jnp.ones((4, 1)),
    problem_id="controllability-gramian",
)
factored_policy = phx.linalg.FactoredMatrixEquationPolicy(
    (-1.0, -2.0, -4.0, -8.0),
    shifted=phx.linalg.ShiftedSolvePolicy("lanczos", max_dimension=4),
    maximum_rank=4,
)
factored_result = phx.linalg.solve_factored_matrix_equation(
    factored_problem,
    policy=factored_policy,
)
assert bool(factored_result.successful)
```

The result retains a fixed-capacity factor with a dynamic effective rank. Inspect
its truncation loss, original-equation residual certificate, per-shift convergence,
and factor-versus-explicit storage evidence before using it downstream.

## Choosing the boundary

- Need JIT, `vmap`, autodiff, PyTrees, or matrix-free execution: use native Phydrax
  methods and prepare outside the hot loop.
- Need a host sparse direct/Krylov ecosystem: explicitly choose PETSc or PyAMGCL.
- Need distributed or established external eigensolver execution: explicitly choose
  SLEPc and manage its collective lifecycle.
- Need NVIDIA GPU AMG on an assembled CSR system: explicitly choose AmgX and retain
  transfer evidence.
- Need a numerical result to drive branch switching, sensitivities, or control:
  inspect its status and certificate/evidence first; do not consume a value solely
  because it is finite.
