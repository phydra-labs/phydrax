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

## Functional training objective

`BSDEObjective` is an `AbstractObjectiveTerm` accepted by `FunctionalSolver`. Fixed
paths provide common-random-number optimization and deterministic replay. Resampled
paths request a fresh batch through the problem's forward sampler. Value and explicit
control models can be ordinary callables or labeled `DomainFunction` objects.

::: phydrax.objectives.BSDEObjective

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
