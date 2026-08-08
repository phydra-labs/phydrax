# Backward stochastic differential equations

## Markovian BSDE contract

For forward paths $X$ and terminal condition $g$, `BSDEProblem` represents

$$
dY_t = -f(t,X_t,Y_t,Z_t)\,dt + Z_t\,dW_t,
\qquad Y_T=g(X_T).
$$

`BSDEPathBatch` stores one explicit time grid, aligned state nodes and Wiener
increments, optional finite-activity jump events, validity masks, and realization
provenance. `evaluate_bsde` returns terminal, interval-local, and complete-trajectory
residuals. The local residual is

$$
Y_{i+1}-Y_i + f_i\,\Delta t_i - Z_i\,\Delta W_i.
$$

The quadrature rule is declared as `left`, `midpoint`, or `trapezoid`; it is never
selected from array shape. `control_mode="explicit"` evaluates a supplied $Z$ model.
`control_mode="autodiff"` computes $Z=\nabla_xu\,\sigma$ with JAX differentiation.
The latter requires a differentiable value predictor and is the direct bridge to the
semilinear PDE residual.

::: phydrax.stochastic.BSDEPathBatch

---

::: phydrax.stochastic.BSDEProblem

---

::: phydrax.stochastic.BSDEEvaluation

---

::: phydrax.stochastic.evaluate_bsde

---

::: phydrax.stochastic.autodiff_bsde_control

---

::: phydrax.stochastic.semilinear_pde_residual

---

::: phydrax.stochastic.bsde_objective_loss

---

::: phydrax.stochastic.bsde_diagnostics

## Functional training term

`BSDETerm` is an `AbstractScalarTerm` accepted by `FunctionalSolver`. Fixed
paths provide common-random-number optimization and deterministic replay. Resampled
paths request a fresh batch through the problem's forward sampler. Value and explicit
control models can be ordinary callables or labeled `DomainFunction` objects.

::: phydrax.terms.BSDETerm

## Global-in-time Feynman--Kac regression

`FeynmanKacSamplingPlan` turns a `BSDEProblem` into frozen supervised labels for one
global time-conditioned value field. The numerical policy is explicit: terminal
time, temporal quadrature, continuation count, antithetic coupling, path chunking,
control-target construction, and fixed-versus-resampled refresh all contribute to a
stable `plan_id`.

`FeynmanKacSamplingMode`, `FeynmanKacControlTargetMode`,
`FeynmanKacRefreshMode`, and `FeynmanKacTimeWeighting` expose the corresponding
literal policy types.

Two sampling modes have different estimands:

- `sampling_mode="trajectory_nodes"` reuses a `BSDEPathBatch`. Every valid
  \((t_i,X_{t_i})\) is a regression input, while future terminal and generator terms
  on that same path form its target. Nodes from one trajectory remain one dependence
  cluster.
- `sampling_mode="queries"` starts independent continuation ensembles at explicit
  \((t,x)\) queries. `num_paths_per_query` controls conditional Monte Carlo error;
  the query distribution controls where the learned global field is accurate.

`FeynmanKacLabelBatch` retains value and optional control targets, Monte Carlo
standard errors, validity, weights, dependence clusters, and problem/process/plan
identities. Martingale control labels estimate \(Z\) from aligned Wiener increments.
Malliavin labels require an explicit verified weight callable. Antithetic members are
not counted as independent paths in reported standard errors.

::: phydrax.stochastic.FeynmanKacSamplingPlan

---

::: phydrax.stochastic.FeynmanKacPathBatch

---

::: phydrax.stochastic.FeynmanKacLabelBatch

---

::: phydrax.stochastic.sample_feynman_kac_paths

---

::: phydrax.stochastic.trajectory_node_feynman_kac_labels

---

::: phydrax.stochastic.query_feynman_kac_labels

---

::: phydrax.stochastic.feynman_kac_label_diagnostics

`FeynmanKacRegressionTerm` applies `stop_gradient` to all generated targets and
performs weighted value regression, with an optional separately weighted control
term. Its sample provider is invoked once per optimizer update by
`FunctionalSolver`, outside the differentiated loss. Fixed labels therefore provide
common-random-number replay; resampled labels provide a fresh Monte Carlo target.

::: phydrax.terms.FeynmanKacRegressionTerm

## Deep Picard iteration

`solve_deep_picard` alternates between frozen Feynman--Kac target generation and
ordinary `FunctionalSolver` optimization of one global value field. Each outer step
records iterate contraction, target and terminal error, control error when requested,
target variance, and valid-path fraction. Target damping changes the iteration but
not its fixed point. Nonconvergence returns a valid result with `converged=False`;
`raise_on_failure` semantics are deliberately absent.

`DeepPicardInitialSource` selects zero or current-model initialization;
`StructuredSourceBuilder` is the callable contract for constructing a
`StructuredPicardSource` from each frozen context.

For semilinear problems, the frozen source is the `BSDEProblem.generator`. For
structured fully nonlinear sources, `StructuredPicardSource` receives a
`PicardSourceContext` exposing value, gradient, \(Z=\nabla u\,\sigma\), directional
Hessian contractions, and \(\operatorname{tr}(\sigma\sigma^\mathsf T\nabla^2u)\)
through factor-HVPs. There is no dense-Hessian callback. Supply a separate validation
query distribution when the training queries do not represent the desired error
measure.

::: phydrax.solver.solve_deep_picard

---

::: phydrax.solver.DeepPicardResult

---

::: phydrax.solver.DeepPicardDiagnostics

---

::: phydrax.solver.StructuredPicardSource

---

::: phydrax.solver.PicardSourceContext

## Deep BSDE terminal shooting

`DeepBSDEShootingTerm` implements the canonical forward rollout

$$
Y_{i+1}=Y_i-f(t_i,X_i,Y_i,Z_i)\Delta t_i+Z_i\Delta W_i
$$

and minimizes only the terminal mismatch against \(g(X_T)\). The initial-value
function and explicit control function are ordinary entries in a `FunctionalSolver`;
a `Domain.Parameter` is the direct representation of one learned \(Y_0\).
`deep_bsde_rollout` supports arbitrary declared output and noise event shapes and
masks invalid paths before they can contaminate gradients.

`solve_deep_bsde` appends the shooting term temporarily, trains both functions,
removes that temporary term, and evaluates the result on a separately sampled or
explicitly supplied validation batch. The returned object is a localized shooting
solution, not a global value field. Use a state-dependent initial-value function only
when intentionally amortizing over an initial-state distribution.

::: phydrax.terms.DeepBSDEShootingTerm

---

::: phydrax.terms.deep_bsde_rollout

---

::: phydrax.terms.DeepBSDEShootingDiagnostics

---

::: phydrax.solver.solve_deep_bsde

---

::: phydrax.solver.DeepBSDEResult

## Backward deep splitting

`deep_splitting_labels` constructs the explicit right-endpoint target

$$
\widehat U_i =
U_{i+1}(X_{i+1})
+\Delta t_i f(t_{i+1},X_{i+1},U_{i+1},Z_{i+1}),
\qquad
Z_{i+1}=\nabla_xU_{i+1}\,\sigma .
$$

Targets are stopped before differentiation. `solve_deep_splitting` trains one
conditional-expectation regression at a time in reverse temporal order, optionally
warm-starting each slice from its successor. Every resampled optimizer update receives
one newly materialized path batch; fixed paths provide common-random-number replay.

`DeepSplittingSolution` retains every learned slice plus the exact terminal condition.
It evaluates declared nodes directly and supports JAX-compatible nearest or linear
temporal interpolation. `as_domain_function` exposes that interpolant through the
normal labeled field API. A held-out one-step RMSE includes irreducible transition
noise; it is a regression diagnostic, not by itself a global solution-error estimate.

::: phydrax.terms.deep_splitting_labels

---

::: phydrax.terms.DeepSplittingRegressionTerm

---

::: phydrax.solver.solve_deep_splitting

---

::: phydrax.solver.DeepSplittingSolution

---

::: phydrax.solver.DeepSplittingResult

## Coupled forward-backward simulation

`solve_coupled_fbsde_explicit` performs Euler--Maruyama forward steps whose drift and
diffusion may depend on the current value and control predictions, then evaluates the
backward residuals on the exact same Wiener realization. It is an explicit scheme: the
control predictor must be supplied, and the time grid and Brownian-tree tolerance are
validated before simulation.

::: phydrax.solver.CoupledFBSDEProblem

---

::: phydrax.solver.CoupledFBSDEResult

---

::: phydrax.solver.solve_coupled_fbsde_explicit

## Finite-activity jump BSDEs

For a jump control $U$, the compensated interval increment is

$$
\sum_{t_i < \tau_k \le t_{i+1}} U(\tau_k,e_k)
- \int_{t_i}^{t_{i+1}}\!\int U(s,e)\,\nu_s(de)\,ds.
$$

`JumpBSDEProblem` requires an exact or user-controlled compensator-rate callable for
each named jump process. `evaluate_jump_bsde` subtracts the compensated increment from
the Brownian local residual. Event batches must retain pre-jump states. When realization
provenance is present, a composite realization must contain exactly one matching Wiener
component and each declared Poisson-clock component. Event-capacity or solver failure
invalidates the affected path and is available through status diagnostics.

::: phydrax.stochastic.JumpBSDEProblem

---

::: phydrax.stochastic.JumpBSDEEvaluation

---

::: phydrax.stochastic.evaluate_jump_bsde

---

::: phydrax.stochastic.jump_bsde_objective_loss

---

::: phydrax.stochastic.jump_bsde_diagnostics

## Least-squares backward schemes

`solve_bsde_least_squares` performs pathwise backward dynamic programming on a
`BSDEPathBatch`. A regression basis defines finite features without owning a training
loop; the built-in polynomial basis has an explicit feature cap. Regressions use
weighted masked least squares with ridge regularization, optional standardization, and
Picard iteration for nonlinear generators. Diagnostics retain rank, condition
estimates, valid sample counts, residuals, and convergence per time step.

::: phydrax.solver.AbstractBSDERegressionBasis

---

::: phydrax.solver.PolynomialBSDERegressionBasis

---

::: phydrax.solver.CallableBSDERegressionBasis

---

::: phydrax.solver.solve_bsde_least_squares

---

::: phydrax.solver.LeastSquaresBSDEResult

---

::: phydrax.solver.least_squares_bsde_diagnostics

## Reflected path-dependent BSDEs

`ReflectedPathDependentBSDEProblem` evaluates coefficients against complete causal
state histories and declares optional lower and upper obstacles. The solver projects
the continuation value at each backward node, records both reflection increments,
checks obstacle ordering, and keeps terminal compatibility separate from local
regression validity. This is a discrete reflected scheme, not a claim of automatic
continuous-time obstacle resolution.

::: phydrax.stochastic.ReflectedPathDependentBSDEProblem

---

::: phydrax.solver.solve_reflected_path_dependent_bsde

---

::: phydrax.solver.ReflectedPathDependentBSDEResult

---

::: phydrax.solver.reflected_path_dependent_bsde_diagnostics

## Empirical mean-field control

`EmpiricalMeanField` stores a weighted Lagrangian particle law over time and returns
interpolated `MeanFieldSnapshot` objects with the finite support, weights, mean,
covariance, and effective sample size still visible. `MeanFieldBSDEProblem` freezes
that measure flow into a canonical BSDE. The control adapter turns a declared policy,
running cost, and controlled drift into the corresponding Hamiltonian generator.

::: phydrax.stochastic.EmpiricalMeanField

---

::: phydrax.stochastic.MeanFieldSnapshot

---

::: phydrax.stochastic.MeanFieldBSDEControlAdapter

---

::: phydrax.stochastic.MeanFieldBSDEProblem

---

::: phydrax.stochastic.adapt_mean_field_control_bsde
