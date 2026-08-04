# Stochastic processes and state space

`phydrax.stochastic` owns mathematical process contracts, stochastic realizations,
trajectory provenance, martingale decompositions, state-space models, transition adapters,
and backward stochastic equations. Numerical integration remains in `phydrax.solver`;
statistical filtering, smoothing, diagnostics, calibration, and portable results remain in
`phydrax.uq`.

The central invariants are explicit:

- physical case axes, process-realization axes, time axes, and state event axes are never
  inferred from array rank;
- a realization identifies the random object being queried, not merely a PRNG key;
- pathwise transition kernels preserve one driver segment across a complete transition;
- failed transitions and event streams remain masked and status-coded rather than being
  silently replaced by valid draws;
- weak, martingale, BSDE, and filtering diagnostics account for dependence clusters.

## API sections

- [State-space models and solver transition adapters](state_space.md)
- [Martingale problems and stopping](martingales.md)
- [Backward stochastic differential equations](bsde.md)
- [Filtering and smoothing](../uq/filtering.md)
- [Differential, jump, hybrid, and SPDE integration](../solver/differential.md)

## Realization provenance

::: phydrax.stochastic.WienerRealization

---

::: phydrax.stochastic.PoissonClockRealization

---

::: phydrax.stochastic.CompositeStochasticRealization

---

::: phydrax.stochastic.realization_path_labels

---

::: phydrax.stochastic.realization_independence_labels
