# Projector Monte Carlo

Native sparse signed/complex ground-state projection over packed configuration keys. Read the [scientific guide](../../guides_projector_monte_carlo.md) and [complete recipe](../../cookbook/projector_monte_carlo.md). This is not VMC, TDVP, quantum jumps, an excited-state solver, or a non-Hermitian physical eigensolver.

## Physical problem and resource plan

The Hamiltonian is an original self-adjoint `QuantumLatticeColumnOperator` with outgoing `H[target, source]` orientation. Initial/trial coefficients are physical `complex128` coefficients. `dt` has explicit native inverse-energy units reciprocal to `energy_unit`; each extra observable requires its own `observable_units` entry. An optional frozen positive guide changes the represented coefficients, not the physical observable convention.

`ProjectorMonteCarloPlan` separates physical spawning/compression/controller policy from resource capacities. `support_capacity`, `group_capacity`, `event_capacity`, `attempt_capacity`, `source_capacity`, `history_capacity`, and retained/workspace byte maxima do not define a qualified scale. The semistochastic threshold uses the operator's physical `raw_route_bound`, not an admitted maximum. Execution requires x64 on one owning device; distributed arrays and float32 execution refuse.

::: phydrax.solver.ProjectorMonteCarloProblem

::: phydrax.solver.ProjectorMonteCarloPlan

::: phydrax.solver.PreparedProjectorMonteCarlo

## Preparation and committed evolution

Initialize once with an explicit typed PRNG key. The same solve entry point accepts an existing validated state for continuation. A step returns `ProjectorMonteCarloStepResult`, not a bare state. Refusal preserves the complete previous commit for every replica, including controller/history/counter/key; it never means clipped or partially accepted evolution.

::: phydrax.solver.prepare_projector_monte_carlo

::: phydrax.solver.initialize_projector_monte_carlo

::: phydrax.solver.validate_projector_state

::: phydrax.solver.step_projector_monte_carlo

::: phydrax.solver.solve_projector_monte_carlo

::: phydrax.solver.ProjectorMonteCarloState

::: phydrax.solver.ProjectorMonteCarloHistory

::: phydrax.solver.ProjectorMonteCarloEvidence

::: phydrax.solver.ProjectorMonteCarloStepResult

::: phydrax.solver.ProjectorMonteCarloResult

::: phydrax.solver.ProjectorMonteCarloStatus

## Physical observations and host analysis

Raw projected and ordered replica-pair numerator/denominator histories remain aligned at every accepted step, including zero overlaps. Analysis aggregates shared replica pairs per time before joint correlated-ratio inference. `replicas` and `observable_units` in analysis are ordered Hamiltonian first, then requested observables. Real and complex physical observables use original columns and inverse-guide physical coefficients, never the Euclidean Rayleigh quotient of a guided operator.

Burn-in/cadence affect measurement selection, not retained incoming-shift history. Every requested reweighting depth uses complete accepted-step windows. Statistical ratio/weight status is distinct from propagation refusal and finite-population/history/timestep/sign assumptions. `deterministic_records` declares source evidence and must not be inferred from observed constants. See [correlated ratios](../uq/correlated_ratios.md) for the denominator and correlation gates.

::: phydrax.solver.ProjectorMonteCarloObservation

::: phydrax.solver.observe_projector_state

::: phydrax.solver.ProjectorEstimatorPolicy

::: phydrax.solver.ProjectorWeightStatus

::: phydrax.solver.ProjectorWeightDiagnostics

::: phydrax.solver.ProjectorReweightedEstimate

::: phydrax.solver.ProjectorSystematicRecord

::: phydrax.solver.ProjectorMonteCarloAnalysis

::: phydrax.solver.analyze_projector_monte_carlo

## Checkpoint, resource transport, and durable results

Checkpoint read requires `(path, prepared, template)` with matching scientific/storage structure and key implementation. Templates supply executable structure, not a replacement root stream. Write/read retain the full committed state and raw histories through the existing native runtime lifecycle.

Capacity-only transport requires the same scientific problem, replicas, and numerical policy, nondecreasing resource limits, and at least one actual enlargement. It returns `(state, RuntimeRestartRelation)` and performs no draws. Replay from that commit with the same logical step/root stream; do not retry with fresh randomness after overflow. Changed Hamiltonian numerical bindings, domain, guide, units, or timestep are not resource transports.

`write_projector_monte_carlo_result(path, prepared, result, *, run_id, analysis=None)` preserves typed result/history/evidence and optional analysis through the native durable result owner. Neither checkpoint nor result persistence serializes executable operator/guide providers or grants a release claim.

The result writer bounds container, aggregate, member, manifest, and element sizes
by the admitted retained-byte policy and the exact numerical record count. Later
`phydrax.lifecycle.open` calls use explicit `ArrayArchiveLimits` for a rich result
that exceeds the default member budget. The native archive format is unchanged.

::: phydrax.solver.write_projector_monte_carlo_checkpoint

::: phydrax.solver.read_projector_monte_carlo_checkpoint

::: phydrax.solver.transport_projector_monte_carlo_resources

::: phydrax.solver.write_projector_monte_carlo_result
