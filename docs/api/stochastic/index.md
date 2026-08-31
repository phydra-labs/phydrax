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

Finite-step learning can reuse canonical solver coefficients through
`EulerMaruyamaTransitionKernel` and masked `EulerMaruyamaQuasiLikelihood`.
`IsothermalPortHamiltonianDynamics` adds the complete state-dependent Itô
fluctuation--dissipation correction and an exact Gibbs-stationarity diagnostic;
it does not label the local Gaussian transition as an exact SDE likelihood.

## API sections

- [State-space models and solver transition adapters](state_space.md)
- [Gaussian score diffusions](diffusion.md)
- [Signatures and path kernels](signatures.md)
- [Martingale problems and stopping](martingales.md)
- [Backward stochastic differential equations](bsde.md)
- [Filtering and smoothing](../uq/filtering.md)
- [Differential, jump, hybrid, and SPDE integration](../solver/differential.md)

## Trajectory measures

`trajectory_measure` exposes either masked time marginals or complete path units
as an external weighted integration target. `time_measure` exposes each saved,
possibly irregular path schedule as deterministic left-point or trapezoid
quadrature. Neither adapter resamples the trajectory or consumes a key.

::: phydrax.stochastic.StochasticTrajectory

---

::: phydrax.stochastic.trajectory_measure

---

::: phydrax.stochastic.time_measure

## State-time particle adapters

`trajectory_state_time_samples` preserves every valid trajectory node as a structured
particle with explicit state, time, path, independence, and time-index fields.
`TrajectoryStateTimeMode` is `"global"` or `"per_time"` and controls the declared
normalization semantics of the adapter. The result can be consumed directly by score
matching or exposed as a generic `WeightedSampleTarget`; no resampling or density
reconstruction occurs.

::: phydrax.stochastic.TrajectoryStateTimeSamples

---

::: phydrax.stochastic.trajectory_state_time_samples

---

::: phydrax.stochastic.trajectory_state_time_measure

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


## Coupled stochastic hierarchies

`StochasticLevelSpec` names every approximation axis—time, space, retained noise,
surrogate fidelity, or another declared refinement—and records state, solver, problem,
and observable identities. `StochasticCouplingPlan` validates parent links, owns a
generic `DiscretizationHierarchy`, and prevents nominally adjacent levels with
incompatible contracts from being coupled accidentally. `NoiseCoupling` records the
shared realization and coarse/fine transformation rather than treating equal PRNG
keys as evidence of common randomness. Nested noise additionally requires explicit
coarse/fine basis IDs, one noise-family ID, and a passing `NoiseCouplingWitness`
whose covariance and increment projection residuals satisfy its tolerance.

These contracts are consumed by `solve_coupled_hierarchy` and multilevel integration.
Coarse and fine outputs retain pair IDs, validity, per-level cost, and coupling
provenance; a failed member is never silently replaced by an independent draw.

::: phydrax.stochastic.StochasticLevelSpec

---

::: phydrax.stochastic.StochasticCouplingPlan

---

::: phydrax.stochastic.NoiseCouplingWitness


## Path events and changes of measure

Path events have one score convention: success is `score >= 0`. Threshold crossings,
terminal sets, accumulated observables, and competing events share localization,
first-hit, terminal-score, and validity results. The same event objects drive ordinary
trajectory diagnostics and adaptive multilevel splitting.

`PathMeasureChange` objects retain interval log-density increments and cumulative
weights. Diffusion changes use the Girsanov drift shift; jump changes use compensator
rate ratios. `measure_changed_target` exposes the resulting weighted paths to the
ordinary integration API without claiming IID uncertainty for dependent trajectories.

::: phydrax.stochastic.ThresholdCrossingEvent

---

::: phydrax.stochastic.TerminalSetEvent

---

::: phydrax.stochastic.AccumulatedPathEvent

---

::: phydrax.stochastic.evaluate_path_event

---

::: phydrax.stochastic.DiffusionMeasureChange

---

::: phydrax.stochastic.JumpMeasureChange

---

::: phydrax.stochastic.measure_changed_target

## Lévy, fractional Gaussian, and rough paths

`LevyProcessRealization` owns a prefix-stable decreasing Poisson series. Increasing
`max_terms` preserves every represented jump. A cutoff is explicit, and Gaussian
small-jump closure uses a reserved, coupled `WienerRealization`; truncation and
Gaussian closure are therefore distinguishable approximations.

`FractionalGaussianRealization` samples the exact finite-grid law of a declared
fractional Gaussian process. The default `method="dense"` retains the covariance
eigendecomposition and its seeded path identity. Its read-only `covariance_factor`
exposes that dense factor. `method="davies-harte"` uses an exact, FFT-based
circulant embedding when the materialized grid begins exactly at `reference_time`
and has increasing uniform spacings; its `covariance_factor` is `None`. An explicit
request rejects an anchor mismatch, nonuniform grid, or numerically invalid
embedding. `method="auto"` keeps dense sampling for those grid mismatches and grids
below 256 increments, selects Davies–Harte for qualifying larger grids, and records
any dense fallback in `sampling_provenance`. The resolved `sampling_method` and
provenance are also included in stochastic-trajectory metadata.

Both methods support exact node queries or one declared global linear interpolant,
so overlapping increments remain additive. Sample batches are prefix-stable within
each resolved method. A `GeometricRoughPath` lifts either realization to levels one
and two and composes intervals by Chen's identity. The native rough solver is step
two and consequently requires a fractional Hurst exponent greater than one third.

::: phydrax.stochastic.SymmetricStableLevyProcess

---

::: phydrax.stochastic.LevyProcessRealization

---

::: phydrax.stochastic.FractionalGaussianProcess

---

::: phydrax.stochastic.FractionalGaussianRealization

---

::: phydrax.stochastic.GeometricRoughPath

---

::: phydrax.stochastic.compose_rough_path_segments