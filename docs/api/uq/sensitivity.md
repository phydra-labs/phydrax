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

### Polynomial-chaos coefficient effects

For an orthonormal `PolynomialChaosExpansion`, `mean` is the constant coefficient
and `variance` is the pointwise sum of squared nonconstant coefficients. The
`first_order_sobol` and `total_order_sobol` mappings group that same coefficient
energy by each labeled factor. Array, Field, and PyTree output axes are preserved.
A zero-variance output receives zero coefficient effects; it is not divided by zero
or reported as evidence from a sampling estimator.

These effects are exact for the fitted expansion, not automatically exact for a
model outside the selected finite polynomial span. They require the independent
factor product measure declared by `PolynomialChaosBasis`.

::: phydrax.uq.PolynomialChaosExpansion
    options:
        members:
            - mean
            - variance
            - first_order_sobol
            - total_order_sobol


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

`fisher_information_action` applies the uncentered empirical outer product of real
score rows to a real direction. It delegates the numerical action to
`phydrax.linalg.EmpiricalGramLinearOperator` with `centered=False`; complex or
centered score geometry must construct that pairing-aware operator explicitly.
Optional sample weights must be finite, nonnegative, and have positive total weight.
`gauss_newton_action` applies `JᵀJ` to a real-valued residual-model direction using
one JVP and one transpose-VJP. Both add only the explicitly requested
`regularization * direction`; neither chooses a regularizer or repairs an invalid
result.

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

### Local information-matrix criteria

`experiment_design_objective` evaluates D-optimal log determinant, negative
A-optimal inverse trace, E-optimal smallest eigenvalue, or Gaussian mutual
information. It accepts a dense information matrix or an action callable. A
callable is materialized on coordinate basis vectors and is therefore still
guarded by `max_dimension` (256 by default); it does not switch to a stochastic
log-determinant or iterative eigensolver.

The effective information matrix is exactly the supplied matrix plus the
declared diagonal `regularization`. It must be finite, symmetric, and positive
semidefinite. D- and A-optimal criteria additionally require positive
definiteness; E-optimal and mutual-information criteria preserve valid singular
positive-semidefinite information. Invalid information produces `value=nan`,
`SENSITIVITY_INVALID_INFORMATION`, and `valid=False`, with no clipping or
repair. The result records its criterion, eigenvalues, dimension,
regularization, approximation, and whether the source was
`"dense_information"` or `"matrix_free_actions_materialized"`.

::: phydrax.uq.ExperimentDesignResult

---

::: phydrax.uq.experiment_design_objective

### Exact finite-outcome expected information gain

Finite-channel EIG is a different objective from the local Gaussian
information-matrix criteria above. `FiniteExperimentalDesignProblem` declares
finite `FiniteProductSpace` supports for parameters \(\Theta\), designs \(D\),
and outcomes \(Y\), and a callback
`log_conditional_probability(parameters, design, outcomes, context)`. For each
declared design \(d\), `ExpectedInformationGain` evaluates the finite mutual
information in nats,

\[
I(\Theta;Y\mid d)
=\sum_{\theta,y}\pi(\theta)\,p(y\mid\theta,d)
\log\frac{p(y\mid\theta,d)}
{\sum_{\theta'}\pi(\theta')p(y\mid\theta',d)}.
\]

This result is exact over the declared finite supports; it is not continuous
EIG, posterior integration, or a Gaussian information approximation. The
likelihood callback receives all parameter payloads and one outcome chunk and
must return a floating `(P, C)` array. Each active parameter row must be
normalized over the complete declared outcome support, not each chunk.
Impossible outcomes use `-inf`; `NaN`, `+inf`, or a complete-row normalization
error above `normalization_tolerance` rejects the design. No row is repaired or
renormalized.

`FiniteDesignBelief(parameters, log_masses, *, parameter_mask=None,
history=())` is constructed on the host and normalizes finite unnormalized
log masses once. `-inf` is exact zero mass. A finite log mass remains active
even if exponentiating it underflows, so activity must be read from
`active_parameters`, not reconstructed in probability space. Parameter masks
remove rows before validation; `design_mask` addresses row-major design flat
IDs and therefore preserves duplicate-payload identities.

```python
import jax.numpy as jnp
import phydrax as phx


def finite_space(values):
    return phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray(values))
    )


def channel(parameters, design, outcomes, context):
    del design, context
    return jnp.where(
        parameters[:, None] == outcomes[None, :],
        0.0,
        -jnp.inf,
    )


problem = phx.uq.FiniteExperimentalDesignProblem(
    finite_space([0, 1]),
    finite_space([0]),
    finite_space([0, 1]),
    channel,
    likelihood_id="perfect-binary-channel",
)
belief = phx.uq.FiniteDesignBelief(problem.parameters, jnp.asarray([0.0, 0.0]))
policy = phx.uq.ExpectedInformationGain(
    candidate_batch_size=1,
    outcome_batch_size=1,
)
selection = phx.uq.select_finite_experimental_design(
    problem, belief, policy=policy
)
experiment = phx.uq.bind_finite_design_experiment(
    problem,
    belief,
    selection,
    1,
    experiment_id="observed-one",
)
update = phx.uq.update_finite_design_belief(
    problem, belief, experiment, policy=policy
)
assert bool(update.accepted)
```

`evaluate_finite_experimental_design` scores one flat design ID and is
JIT-friendly. `select_finite_experimental_design` is a host-orchestrated exact
finite exhaustive reduction; invalid designs are excluded and the lowest flat
ID wins an EIG tie. Outcome chunking and candidate batching bound storage but
do not approximate the sum. No full likelihood tensor or score landscape is
retained.

`ExpectedInformationGain.preflight(...)` runs from shapes before payload
gathers, likelihood tracing, search allocation, or observation updating.
`maximum_bytes` bounds buffers owned by this API. Persistent inputs, compiler
and runtime storage, and callback temporaries are excluded;
`likelihood_workspace_bytes_per_candidate` reserves caller-declared callback
workspace. The estimate covers only batching performed by this API, not an
additional external `vmap`. Refusal raises `MemoryError` before the likelihood
is called, and `FiniteEIGResources` reports both bounded working storage and
the counterfactual dense likelihood size.

`bind_finite_design_experiment` is host-side and binds a real in-support
outcome flat ID; it never samples or invents an observation. The resulting
immutable `Experiment` records design payload and flat ID, outcome payload and
flat ID, context, likelihood identity, and prior belief identity.
`update_finite_design_belief` validates those identities and content before a
log-space Bayes update. Stale, replayed, or tampered records, inactive designs,
invalid likelihood rows, and zero-predictive-mass observations return a typed
failure with the original belief and history unchanged. A finite, extremely
small log predictive probability can still update without a probability-space
round trip.

::: phydrax.uq.FiniteExperimentalDesignProblem

---

::: phydrax.uq.FiniteDesignBelief

---

::: phydrax.uq.ExpectedInformationGain

---

::: phydrax.uq.FiniteEIGResources

---

::: phydrax.uq.FiniteDesignEvaluation

---

::: phydrax.uq.FiniteDesignSelection

---

::: phydrax.uq.FiniteDesignUpdate

---

::: phydrax.uq.FiniteDesignStatus

---

::: phydrax.uq.evaluate_finite_experimental_design

---

::: phydrax.uq.select_finite_experimental_design

---

::: phydrax.uq.bind_finite_design_experiment

---

::: phydrax.uq.update_finite_design_belief

## Status codes

::: phydrax.uq.SENSITIVITY_SUCCESS

---

::: phydrax.uq.SENSITIVITY_NONFINITE

---

::: phydrax.uq.SENSITIVITY_INVALID_INFORMATION
