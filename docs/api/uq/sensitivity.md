# Sensitivity, information actions, and experiment design

## Variance-based global sensitivity

`sobol_indices` evaluates Saltelli first-order and Jansen total-order indices from one
declared joint QMC design. Parameter names and the reserved parameter axis remain
explicit, and array or `coordax.Field` outputs retain their physical output axes.
Optional masks and nonnegative weights apply only with an explicit mean or sum
reduction. Zero or nonfinite output variance is rejected rather than producing a
plausible-looking index.

::: phydrax.uq.sobol_indices

---

::: phydrax.uq.SobolResult

## Stochastic gradient estimators

`fixed_noise_pathwise_gradient` differentiates only the declared response at one fixed
noise realization. Its `noise_id` is required provenance; the result does not claim to
integrate over that noise. `likelihood_ratio_gradient` reports a score-function
estimate and Monte Carlo standard error, with an explicit baseline and estimator
method.

Discrete ancestry is nondifferentiable. `resampling_score_gradient` therefore computes
the categorical likelihood-ratio contribution of one supplied resampling operation,
including the softmax normalization term. It preserves the supplied ancestor indices,
normalized weights, `noise_id`, and required `resampling_id`; it is not a pathwise
derivative through those integer indices. Combine it deliberately with any continuous
pathwise terms rather than treating it as an automatic full particle-filter gradient.

Every result exposes `valid`, `status`, `estimator_id`, `method_id`, `approximation`,
sample count, and the applicable random-mechanism IDs. Nonfinite inputs or estimates
remain visible as `SENSITIVITY_NONFINITE`; there is no fallback estimator.

::: phydrax.uq.SensitivityGradientResult

---

::: phydrax.uq.ResamplingScoreResult

---

::: phydrax.uq.fixed_noise_pathwise_gradient

---

::: phydrax.uq.likelihood_ratio_gradient

---

::: phydrax.uq.resampling_score_gradient

## Matrix-free Fisher and Gauss--Newton actions

`fisher_information_action` applies the empirical score outer-product matrix to a
direction without forming that matrix. Optional sample weights must be finite,
nonnegative, and have positive total weight. `gauss_newton_action` applies `JᵀJ` to a
real-valued residual-model direction using one JVP and one transpose-VJP. Both add
only the explicitly requested `regularization * direction`; neither chooses a
regularizer or repairs an invalid result.

This complete local example does not materialize a Jacobian:

```python
import jax.numpy as jnp
import phydrax as phx

parameters = {"rate": jnp.asarray(1.0)}
direction = {"rate": jnp.asarray(1.0)}
target = jnp.asarray([0.8, 2.1, 3.2])


def residual_fn(value):
    return value["rate"] * jnp.asarray([1.0, 2.0, 3.0]) - target


curvature_direction = phx.uq.gauss_newton_action(
    residual_fn,
    parameters,
    direction,
    regularization=1e-3,
)

assert curvature_direction.operator_id == "gauss_newton"
assert curvature_direction.method_id == "jax_jvp_vjp"
```

The returned action records `operator_id`, `method_id`, approximation, explicit
regularization, sample count when applicable, validity, and status.

For a declared exponential family,
`exponential_family_fisher_action` applies the exact natural-coordinate Fisher as a
JVP of the mean map. `exponential_family_parameter_fisher_action` wraps this with the
natural-parameter JVP and transpose pullback to apply `Jηᵀ F(η) Jη`. These actions are
exact for the declared family geometry and do not require score samples.


::: phydrax.uq.SensitivityActionResult

---

::: phydrax.uq.fisher_information_action

---

::: phydrax.uq.exponential_family_fisher_action

---

::: phydrax.uq.exponential_family_parameter_fisher_action

---

::: phydrax.uq.gauss_newton_action

## Empirical observability and controllability directions

`empirical_observability_directions` computes dominant input/state directions of the
local `JᵀJ` Gramian. `empirical_controllability_directions` computes dominant response
directions of the local `J Jᵀ` Gramian. Derivative actions use JVPs and transpose-VJPs,
but the requested ambient action matrix is then materialized and diagonalized
densely. `max_dimension` (256 by default) is a hard guard, not a trigger for a hidden
iterative backend.

These are empirical local linearizations, not global system properties. Results
preserve the quantity name, ambient shape and dimension, requested rank, strengths,
directions, explicit regularization, validity/status, and the stable
`method_id="matrix_free_actions_dense_eigh"`.

::: phydrax.uq.EmpiricalDirectionsResult

---

::: phydrax.uq.empirical_observability_directions

---

::: phydrax.uq.empirical_controllability_directions

## Information-design objectives

`experiment_design_objective` evaluates D-optimal log determinant, negative
A-optimal inverse trace, E-optimal smallest eigenvalue, or Gaussian mutual
information. It accepts a dense information matrix or an action callable. A callable
is materialized on coordinate basis vectors and is therefore still guarded by
`max_dimension` (256 by default); it does not switch to a stochastic log-determinant
or iterative eigensolver.

The effective information matrix is exactly the supplied matrix plus the declared
diagonal `regularization`. It must be finite, symmetric, and positive semidefinite.
D- and A-optimal criteria additionally require positive definiteness; E-optimal and
mutual-information criteria preserve valid singular positive-semidefinite
information. Invalid information produces `value=nan`,
`SENSITIVITY_INVALID_INFORMATION`, and `valid=False`, with no clipping or repair.
The result records its criterion, eigenvalues, dimension, regularization, approximation,
and whether the source was `"dense_information"` or
`"matrix_free_actions_materialized"`.

::: phydrax.uq.ExperimentDesignResult

---

::: phydrax.uq.experiment_design_objective

## Status codes

::: phydrax.uq.SENSITIVITY_SUCCESS

---

::: phydrax.uq.SENSITIVITY_NONFINITE

---

::: phydrax.uq.SENSITIVITY_INVALID_INFORMATION
