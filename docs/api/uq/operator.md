# Neural-operator uncertainty

Operator uncertainty uses the same `PredictiveField` statistics as pointwise models,
but retains the complete `OperatorBatch` output contract: physical case axes, query
geometry, query masks, quadrature, and `OperatorOutputSpec` channels.

Tensor-grid outputs keep their `OperatorAxis` names. Point-cloud and channel axes use
reserved internal names so they cannot collide with physical case axes. Every
stochastic source remains an explicit `SampleAxis`; no reduction infers uncertainty
semantics from axis position.

Masked query padding is replaced by zero only in predictive storage. The attached
query mask remains authoritative, and statistics, proper scores, and calibration
exclude the padding. A non-finite value at a valid query location either records an
invalid whole-function draw or raises immediately according to `valid_policy`.

## Predictive results

::: phydrax.uq.OperatorPredictiveField
    options:
        members:
            - __init__
            - from_predictive
            - output_mask
            - output_weights
            - mean
            - variance
            - std
            - quantile
            - interval
            - epistemic_variance
            - input_variance
            - observation_variance
            - process_variance
            - numerical_variance
            - total_variance
            - decompose_variance

---

::: phydrax.uq.OperatorPredictionInterval
    options:
        members:
            - __init__

---

::: phydrax.uq.operator_prediction_field

---

::: phydrax.uq.operator_predictive_from_samples

---

::: phydrax.uq.operator_input_predictive

---

::: phydrax.uq.sample_operator_predictive

## Geometry-aware operator uncertainty

`OperatorPredictiveField` is architecture-independent: predictions from `GINO`,
`RIGNO`, and `GAOT` retain case-dependent point-cloud query coordinates, masks,
and quadrature exactly as predictions from grid operators do. The stochastic
contract is stricter than shape equality:

- every ensemble or posterior draw must use the same `OperatorOutputSpec`, case
  axes, and query geometry for the batch being summarized;
- source-coordinate, source-value, source-mask, and source-quadrature draws may
  vary together under declared input-sample axes;
- pointwise means, quantiles, CRPS, and coverage require common output query
  locations across draws; remesh first when queries differ;
- energy scores and functional conformal calibration treat one complete valid
  output field as one event and use output quadrature when available.

Deep ensembles are the default epistemic baseline for every geometry
architecture. MC dropout is meaningful only if the stochastic layer was active
during fitting: `RIGNO` exposes `processor_edge_dropout`; `GAOT` exposes
attention and feed-forward dropout. One root PRNG key denotes one coherent
whole-function draw. Use `phydrax.nn.inference_mode` before deterministic
deployment or posterior-density construction.

Weight-space inference should target an explicit `ParameterSubspace`. Useful
small choices include a GINO output projection, a RIGNO decoder/readout, or a
GAOT output projection. Never use global leaf order as a proxy for “last layer”
in an encode–process–decode model. A geometry ensemble captures parameter
uncertainty conditional on its observed mesh; uncertain domain shapes belong in
the input-sample axes instead.

## Complete-field probability distributions

`AbstractOperatorDistribution` defines a distribution over one complete output
field per physical case. Implementations expose a deterministic `location`, keyed
`sample`, normalized `log_prob`, the physical query/output contract, and an explicit
uncertainty source. `AbstractProbabilisticOperatorModel` makes that distribution the
model's primary prediction contract.

Use `GaussianFunctionOperator` first. It provides an exact masked
diagonal-plus-low-rank Gaussian density and coherent whole-field samples.
`scale_mode="fixed"` keeps a declared diagonal noise floor while learning only the
location and optional factors. Use `uncertainty_source="process"` for a stochastic
transition, not `"epistemic"`; uncertainty in learned weights remains a separate
posterior or ensemble axis.

`ConditionalFlowFunctionOperator` wraps a deterministic location operator with a
FlowJAX conditional residual flow. `OperatorBatchConditioner` builds its condition
from named branch encoders. The output query geometry and mask are fixed when the
flow is constructed because a finite FlowJAX event size is static. It supports
loader-broadcast copies of that same geometry, not arbitrary-query or
resolution-transfer inference.

`OperatorDistributionNLL` trains either implementation through `fit_operator`.
It scores the normalized execution-space density and respects valid query masks.
No `"space=\"physical\""` shortcut is exposed: normalization transforms and their
Jacobians must be part of a physical-space density explicitly.

::: phydrax.nn.AbstractOperatorDistribution

---

::: phydrax.nn.AbstractProbabilisticOperatorModel

---

::: phydrax.nn.GaussianOperatorDistribution

---

::: phydrax.nn.FlowJAXOperatorDistribution

---

::: phydrax.nn.OperatorDistributionNLL

---

::: phydrax.nn.operator_distribution_nll

## Process-consistent operator transitions

`OperatorTransitionSpec` identifies state, duration, optional source-time,
typed driver fields, and the output query on a canonical `OperatorBatch`;
unassigned inputs remain fixed forcing or parameter conditioning.
`OperatorDriverBinding` names each model input, realization component, driver
kind, and quantity. Wiener bindings expose increments. Jump bindings expose
event times, offsets, channels, marks, masks, or per-channel counts.

`OperatorMarginalTransition` adapts an
`AbstractProbabilisticOperatorModel` into an
`AbstractMarginalTransitionLaw`. Its complete-field distribution must declare
`uncertainty_source="process"`. `marginal_operator_rollout` generates a
replayable Markov chain but does not claim that independently sampled steps
share one driving path.

`OperatorPathwiseTransition` instead conditions an operator on one explicit
additive Wiener segment. `pathwise_operator_rollout` evaluates every segment
from one global `WienerRealization`. `OperatorJumpTransition` conditions on a
canonical `JumpEventBatch`. `OperatorProcessTransition` accepts multiple typed
Wiener and jump fields at once; `process_operator_rollout` derives all segments
from one `CompositeStochasticRealization` and explicit named event batches.
The resulting `StochasticOperatorRollout` wraps a provenance-preserving
`StochasticTrajectory`; `to_predictive()` retains process samples separately
from physical cases and every other uncertainty source.

Use `operator_markov_chain_nll` for adjacent teacher-forced likelihoods,
`direct_operator_horizon_nll` for direct initial-to-horizon likelihoods,
`semigroup_objective` for marginal Chapman--Kolmogorov consistency, and
`cocycle_objective` for pathwise composition.
`operator_weak_generator_objective` matches a general observable generator.
`jump_generator_observable` evaluates the nonlocal generator declared by a
jump process, and `operator_jump_generator_objective` compares it with a
learned marginal transition. `DistributionalSemigroupObjective` remains the
callback-based batch-training form of a complete-field energy-distance
semigroup loss.

::: phydrax.nn.OperatorTransitionSpec

---

::: phydrax.nn.OperatorDriverBinding

---

::: phydrax.nn.OperatorProcessDistribution

---

::: phydrax.nn.OperatorMarginalTransition

---

::: phydrax.nn.OperatorPathwiseTransition

---

::: phydrax.nn.OperatorProcessTransition

---

::: phydrax.nn.OperatorJumpTransition

---

::: phydrax.nn.StochasticOperatorRollout

---

::: phydrax.nn.marginal_operator_rollout

---

::: phydrax.nn.pathwise_operator_rollout

---

::: phydrax.nn.process_operator_rollout

---

::: phydrax.nn.operator_markov_chain_nll

---

::: phydrax.nn.direct_operator_horizon_nll

---

::: phydrax.stochastic.jump_generator_observable

---

::: phydrax.nn.operator_jump_generator_objective

---

::: phydrax.nn.operator_weak_generator_objective

---

::: phydrax.nn.DistributionalSemigroupObjective

---

::: phydrax.nn.conditioned_distributional_semigroup_loss

## Fixed observation likelihood

`FixedOperatorObservationLikelihood` defines a normalized finite-dimensional sensor
likelihood, not a continuum norm or a training objective. It fixes the complete batch,
combines the query and observation masks, sum-reduces likelihood elements within each
physical case, and then sums cases in `log_prob`. Quadrature weights are deliberately
absent unless a user derives a different stochastic observation model explicitly.

::: phydrax.uq.FixedOperatorObservationLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob
            - standardized_residual

## Operator minibatch likelihoods

`OperatorBatchObservationLikelihood` evaluates a normalized observation model on a
dynamically supplied `OperatorBatch`. It combines the batch query mask with the
finite-observation mask, reduces all query and channel elements within each physical
case, and returns exactly one factor per case.

`OperatorMinibatchSource` adapts deterministic `OperatorBatchLoader` epochs to
`LikelihoodBatch`. The loader must be shuffled, retain its final padded batch, and
keep a fixed batch capacity. Case subsampling is supported; query-anchor subsampling,
nonuniform factor weights, and stochastic geometry mutation within a case are not.

::: phydrax.uq.OperatorLikelihoodData

---

::: phydrax.uq.OperatorBatchObservationLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - __call__

---

::: phydrax.uq.OperatorMinibatchSource
    options:
        members:
            - __init__
            - num_factors
            - batch_capacity
            - batches_per_epoch
            - fingerprint
            - configuration
            - epoch

## Whole-function conformal calibration

`OperatorFunctionalConformal` treats one complete output field or trajectory as one
exchangeable calibration case. The maximum score produces simultaneous bands. The
quadrature-weighted L2 score produces a norm ball; it does not define pointwise lower
and upper bounds.

::: phydrax.uq.OperatorFunctionalConformal
    options:
        members:
            - __init__
            - calibrate
            - interval

## Operator-aware proper scores

::: phydrax.uq.operator_ensemble_crps

---

::: phydrax.uq.operator_energy_score

---
::: phydrax.uq.operator_ensemble_energy_distance

---


::: phydrax.uq.operator_interval_coverage

---

::: phydrax.uq.operator_interval_width
