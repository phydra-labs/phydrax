# Functional solver

`FunctionalSolver` optimizes named domain functions against one ordered collection
of scalar terms. Penalties and signed scalar functionals share that collection.
Its optional `enforcement` input accepts only a precompiled
`EnforcementProgram`; construct it with `phx.enforcement.compile(...)` before
creating the solver.

For a conceptual overview of loss evaluation, exact enforcement, and training-loop
behavior, see [Guides → Solvers and training](../../guides_solver.md). For the
four-layer model and integration-source choices, see
[Conditions, integration, terms, and enforcement](../../guides_conditions.md).

!!! note
    Key notes:

    - `loss(...)` evaluates the training `terms` and attached model losses at the
      current parameters. Signed terms such as `phx.terms.IntegralFunctional` are
      added directly rather than squared.
    - `evaluation_terms` are evaluated for diagnostics but do not contribute to
      optimizer updates.
    - `ansatz_functions()` applies the solver's precompiled `EnforcementProgram`,
      if supplied, before any term observes the fields.
    - Certified exact PDE trial fields reject generic hard enforcement because its
      correction need not preserve their solution space. Fit their boundary conditions
      with residual terms.
    - `partition_functions()` exposes the trainable/non-trainable state split used
      by `solve(...)`.
    - `solve(...)` accepts standard and line-search Optax transformations, Phydrax
      native scalar, least-squares, composite least-squares, and Riemannian methods,
      Evosax distribution-based algorithms, and `phydrax.optim.kfac(...)`. Native
      residual methods receive the same partitioned Equinox parameter tree returned by
      `partition_functions()`. Population-based Evosax algorithms require a separate
      finite search-space contract and are rejected.
    - `solve_linear_trial_space(...)` assembles directly bound
      `LinearTrefftzField` coefficients against fixed quadratic residual
      realizations, audits residual affinity, and routes the least-squares system
      through `phydrax.linalg`.
    - Optimizer choice changes the update backend, not objective semantics. All
      backends consume the same run controls and prepared objective lifecycle;
      `num_iter=0` is a no-op and negative values are rejected before dispatch.
    - `solve(..., evaluation_parameters=...)` keeps optimizer updates on raw
      training parameters while using the optimizer-prescribed view for diagnostics,
      selection, and returned functions.
    - Riemannian optimizers bind selected trainable leaves through
      [`ParameterGeometry`](../optim.md#parameter-pytree-binding); they reject
      `evaluation_parameters` transforms that could leave the manifold.
    - `solve(..., train_term_sample_size=k)` trains on an unbiased fixed-size
      subset of training terms per optimizer step.
    - Signed unbiased `RandomizedResidualTerm` and
      `RandomizedMomentPenalty` objectives require `keep_best=False`. Their
      individual estimates may be negative, so minimum sampled training loss is
      not a valid selection criterion.
    - Every optimizer update prepares one immutable objective realization. Sampled
      batches, per-step integration realizations, adaptive weights, evaluation
      keys, and the iteration value are reused by every candidate evaluation and
      by term diagnostics for that update.
    - A constitutive energy that is defined only for admissible states can use
      `IntegralFunctional(nonfinite_integrand="propagate")` with a tested Optax
      line search such as `optax.lbfgs`. The default remains strict. Propagation
      does not convert adaptive-budget or measure failures into optimizer trials.
    - Fixed sampled signed energies should normally use `keep_best=False`.
      Selection against the same quadrature realization rewards quadrature
      overfitting; evaluate the returned field on a larger independent fixed
      realization instead.
    - KFAC and least-squares methods are not energy minimizers. Use them only when
      the training objective supplies the residual-root contract they require.

    - `solve(..., log_terms=True)` logs the training and evaluation term breakdown;
      `tensorboard_log_dir` enables TensorBoard scalar logs.
    - KFAC accepts only `ResidualPenalty` training terms with a nonnegative
      quadratic residual reduction. It lowers the same prepared objective used by
      the other adapters to residual blocks, then reuses it across the gradient,
      curvature update, and line search; see
      [Optimization](../optim.md#structured-residual-optimization-kfac).
    - `GaussNewton` and `LevenbergMarquardt` require residual-only training objectives.
      `GeneralizedGaussNewton` combines those residual roots with signed
      `IntegralFunctional` terms and model-level scalar losses. It uses residual
      Gauss--Newton curvature plus exact scalar Hessian actions while replaying one
      frozen realization throughout the nonlinear solve.
    - With `jit=True`, native scalar, residual, and composite methods compile their
      per-update numerical kernel. The outer training controller remains responsible for
      logging, cancellation, immutable realization preparation, and best-model selection.
    - `save_onnx("u", ...)` exports one named ansatz function for deployment.
    - Data terms report data-fit diagnostics alongside their scalar values.
    - Enforcement compilation uses `gate_method="auto"` for the global,
      dimensionless CAD R-equivalence preservation gate. Select `"compact"` for
      the compact fallback and configure its transition with
      `gate_saturation_fraction` and `gate_linear_fraction`; a direct
      `enforce_dirichlet(...)` call accepts the same gate settings. These settings
      do not change derivative conditions.

## Eigen-PINN selection and strong residual refinement

For a self-adjoint eigenproblem, use separate terms for separate mathematical
roles:

1. `VariationalEigenspace` selects the lowest trial subspace through the block
   min--max objective.
2. `InvariantSubspaceResidual` applies the strong operator and minimizes
   `A U - B U H` with `H` projected from the same trial fields.
3. A distinct fixed realization evaluates the final Ritz values and relative
   strong residuals.

Both terms remove scale and trial-basis freedom through their mass matrix. Do
not add normalization or pairwise orthogonality penalties. Residual-only
training can converge to any invariant mode or cluster; it is not a low-mode
selection rule.

When both terms are active in one solver, pass the same `fixed_realization` if
their matrix entries must use identical quadrature. Staged selection and
refinement avoids duplicate strong-operator work and is the recommended
default.


## Typical usage

```python
import jax.random as jr
import optax
import phydrax as phx

space = phx.domain.Interval1d(0.0, 1.0)
interior = space.component()

model = phx.nn.models.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=16,
    depth=2,
    key=jr.key(0),
)
u = space.Model("x")(model)

condition = phx.conditions.Residual(
    "u",
    interior,
    lambda u: phx.operators.laplacian(u, var="x"),
)
source = phx.integration.per_step(
    phx.integration.mean_over(condition.on),
    phx.integration.MonteCarloPlan(128),
)
penalty = phx.terms.ResidualPenalty(condition, source)

solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(penalty,),
    evaluation_terms=(),
    enforcement=None,
)
loss_before = solver.loss(key=jr.key(1))
solver = solver.solve(
    num_iter=20,
    optim=optax.adam(1e-3),
    seed=0,
    log_terms=True,
    train_term_sample_size=1,
)
loss_after = solver.loss(key=jr.key(2))
```

`loss(...)` evaluates the sum of every term and model-attached loss at the
current parameters. `ansatz_functions()` returns the fields after exact
transforms, so all terms observe the same enforced field mapping.
`partition_functions()` exposes the trainable/non-trainable state split used by
`solve(...)`, and `save_onnx("u", ...)` exports one ansatz field.

## Experimental contraction precision

`solve(..., precision=FunctionalPrecisionPolicy(...))` scopes JAX matrix
contraction precision around every prepared objective evaluation, including
training loss, term diagnostics, data metrics, and final objective settling.
This experiment is deliberately restricted to standard Optax transformations;
line-search ExtraArgs transforms, native iterative/least-squares methods, KFAC,
Riemannian methods, and Evosax are rejected rather than receiving a partial
policy. Trainable inexact leaves must share one dtype.

The returned solver retains the policy and a content-addressed precision
evidence envelope. Its functional discretization records include that evidence
identity, and later `loss(...)` calls reuse the scoped contraction policy.

::: phydrax.solver.FunctionalPrecisionPolicy

---

::: phydrax.solver.FunctionalSolver
    options:
        members:
            - __init__
            - ansatz_functions
            - __getitem__
            - partition_functions
            - trainable_functions
            - loss
            - solve
            - save_onnx
