# Continuous learned transport

Continuous learned transport is split into four contracts rather than one generic
"flow" object:

1. endpoint couplings select joint source/target samples;
2. endpoint interpolants provide state and conditional velocity targets;
3. `FlowMatchingTerm` trains a velocity field;
4. `ContinuousTransport` advances source-law samples through an existing evolution.

An interpolant is not a probability law or an invertible map. In particular, the
linear conditional endpoint path reaches one fixed target at its terminal coordinate
and is singular there. Flow matching regresses the conditional velocity so that the
learned marginal field can subsequently be integrated.

## Endpoint interpolation

::: phydrax.transport.AbstractEndpointInterpolant

---

::: phydrax.transport.LinearEndpointInterpolant

---

::: phydrax.transport.EndpointInterpolantEvaluation

## Endpoint coupling

`independent_endpoint_coupling` samples the empirical product law. A converged native
balanced transport plan can instead be sampled with
`transport_plan_endpoint_coupling`. The latter deliberately materializes
`dense_plan()`, so it is a minibatch-scale operation. It samples the joint plan; it
does not replace source endpoints by barycentric averages.

Physical cases must be coupled independently. A global plan must never mix endpoints
from different observations, geometries, controls, or parameter cases.

::: phydrax.transport.EndpointCouplingSample

---

::: phydrax.transport.independent_endpoint_coupling

---

::: phydrax.transport.transport_plan_endpoint_coupling

## Continuous transport sampling

`ContinuousTransport` binds an unbatched `AbstractProbabilityLaw` to one existing
`AbstractEvolution`. Sampling preserves leading sample axes and returns solver status,
backend status, and evolution provenance. The object does not claim a density,
stochastic-process realization, cocycle, or exact inverse.

::: phydrax.transport.ContinuousTransport

---

::: phydrax.transport.ContinuousTransportSample

## Exact continuous-flow density

`ContinuousFlowLaw` is intentionally narrow. It supports only finite-dimensional,
real, full-dimensional Lebesgue events on trivial Euclidean state geometry, using an
autonomous `DiffraxEvolution`. Exact Jacobian traces are capped explicitly because
their cost grows quadratically with event dimension.

The inverse solve evaluates

```text
log p_data(x) = log p_base(z) + log |det(dz/dx)|.
```

`sample_and_log_prob` integrates forward state and log volume together and therefore
does not perform a second inverse solve. Every density result retains accepted and
rejected step counts plus backend validity.

::: phydrax.transport.ContinuousFlowLaw

---

::: phydrax.transport.ContinuousFlowDensityResult

## Stochastic divergence estimates

`estimate_continuous_flow_log_prob` keeps one keyed Hutchinson probe ensemble fixed
through the complete reverse solve and integrates one log-volume accumulator per
probe. It reports the resulting mean and Monte Carlo standard error.

This estimate does not implement `AbstractProbabilityLaw.log_prob`. An unbiased trace
or log-volume estimate is not automatically an unbiased normalized density estimate,
and it is not an exact Metropolis acceptance target. Adaptive solver error and probe
error are distinct approximation sources.

::: phydrax.transport.estimate_continuous_flow_log_prob

## Deliberate exclusions

The initial contract rejects:

- Hausdorff or Riemannian densities;
- injective or rectangular maps;
- discrete, mixed, trajectory, or complex-coordinate events;
- input-driven continuous systems;
- state-shaped event callbacks in augmented density solves;
- batched source laws;
- arbitrary-query or mesh-independent field densities.

Use `OperatorFlowMatchingMetric` for a fixed query field when physical quadrature,
masks, and channel geometry must define velocity error. This still describes one
fixed-discretization event per solve.
