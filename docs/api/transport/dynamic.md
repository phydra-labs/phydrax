# Finite-state Schrödinger bridges

`phydrax.transport.dynamic` solves the exact finite-state Schrödinger bridge. It
finds the minimum-relative-entropy path law whose first and last marginals equal
caller-supplied finite measures while retaining an `AbstractTransitionKernel` as
the reference process. This is a dynamic transport family, not a mode of
`Sinkhorn`.

The endpoint inputs are existing `DiscreteMeasureTarget`, `WeightedSampleTarget`,
finite `DensityTarget`, or `IntegrationRealization` contracts. They must describe
the same ordered finite state support, event shape, case axes, and physical mass.
Zero endpoint probabilities remain valid states. Masks, case axes, physical mass,
and endpoint provenance remain present in the problem.

```py
import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

states = jnp.asarray([0.0, 1.0])
initial = phx.integration.discrete(
    states,
    cx.Field(jnp.asarray([0.8, 0.2]), dims=("state",)),
    axes="state",
    normalized=True,
    provenance="initial-population",
)
terminal = phx.integration.discrete(
    states,
    cx.Field(jnp.asarray([0.25, 0.75]), dims=("state",)),
    axes="state",
    normalized=True,
    provenance="controlled-terminal-population",
)
reference_matrix = jnp.asarray([[0.9, 0.1], [0.2, 0.8]])

def sample(key, state, t0, t1, context):
    del t0, t1, context
    row = reference_matrix[state.astype(jnp.int32)]
    return jr.categorical(key, jnp.log(row)).astype(float)

def log_prob(next_state, state, t0, t1, context):
    del t0, t1, context
    probability = reference_matrix[
        state.astype(jnp.int32), next_state.astype(jnp.int32)
    ]
    return jnp.where(probability > 0, jnp.log(probability), -jnp.inf)

reference = phx.stochastic.CallableTransitionKernel(
    sample,
    state_shape=(),
    process_id="two-state-reference",
    approximation_id="exact-matrix",
    log_prob_fn=log_prob,
)
problem = phx.transport.dynamic.SchrodingerBridgeProblem(
    initial,
    terminal,
    jnp.asarray([0.0, 0.5, 1.0]),
    reference,
    phx.stochastic.StateSpaceStepContext.empty(),
)
result = phx.transport.dynamic.require_converged_bridge(
    phx.transport.dynamic.SchrodingerBridgeSolver(tolerance=1e-10).solve(problem)
)
paths = result.sample_paths(jr.key(4), sample_shape=(1024,))
```

The reference kernel must provide a normalized finite-support `log_prob`; an exact
solve never accepts a sampler-only transition. At every grid interval the evaluated
rows must sum to one within `transition_tolerance`. Impossible endpoint support is
reported as `TransportStatus.INFEASIBLE_SUPPORT` rather than repaired. The solver
uses log-coordinate iterative proportional fitting and retains both endpoint
scalings, every forward/backward message, reference and controlled transitions,
endpoint coupling, marginal path law, path KL, fixed-length histories, status, case
axes, and reference-process provenance.

`ControlledTransitionKernel` is the Doob transform and directly implements the
stochastic `AbstractTransitionKernel` contract. Its rows are exact wherever the
backward potential is positive. Polar, never-visited rows use the normalized
reference row as a canonical extension and are identified by
`result.controlled_row_valid`; this convention never changes the solved path law.
Path keys fold in the case, sample member, and step identities, so replay is exact
and increasing a leading sample count preserves the existing prefix.

## Problem and solver

::: phydrax.transport.dynamic.SchrodingerBridgeProblem

---

::: phydrax.transport.dynamic.BridgeProblemProvenance

---

::: phydrax.transport.dynamic.SchrodingerBridgeSolver

---

::: phydrax.transport.dynamic.solve_schrodinger_bridge

## Result and controlled process

::: phydrax.transport.dynamic.SchrodingerBridgeResult

---

::: phydrax.transport.dynamic.SchrodingerBridgeDiagnostics

---

::: phydrax.transport.dynamic.BridgeProvenance

---

::: phydrax.transport.dynamic.ControlledTransitionKernel

---

::: phydrax.transport.dynamic.require_converged_bridge

## Sampling and path densities

::: phydrax.transport.dynamic.BridgePathSample

---

::: phydrax.transport.dynamic.sample_bridge

---

::: phydrax.transport.dynamic.sample_bridge_paths

---

::: phydrax.transport.dynamic.sample_bridge_state_indices

---

::: phydrax.transport.dynamic.bridge_path_log_prob

---

::: phydrax.transport.dynamic.reference_path_log_prob

## Scientific adapters

`BridgeInferenceAdapter` composes the solved Doob kernel with the existing
`CategoricalStatePrior` and transition interfaces. `TerminalDistributionControlAdapter`
exposes the controlled stochastic kernel, exact relative-entropy control cost, and
physical terminal residual. Both call `require_converged_bridge`; a failed IPF solve
cannot silently become a training or inference input. Path-law diagnostics consume
the keyed bridge sample instead of defining another sample or UQ hierarchy.

::: phydrax.transport.dynamic.BridgeInferenceAdapter

---

::: phydrax.transport.dynamic.TerminalDistributionControlAdapter

---

::: phydrax.transport.dynamic.BridgePathLawDiagnostics

---

::: phydrax.transport.dynamic.bridge_path_law_diagnostics
