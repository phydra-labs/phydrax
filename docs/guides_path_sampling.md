# Transition-path and interface sampling

`phydrax.stochastic.path_sampling` provides prepared, fixed-capacity path-space Monte Carlo for transition-path sampling (TPS), transition-interface sampling (TIS), and replica-exchange TIS (RETIS). Its compiled state never grows Python containers or changes array shapes.

## Regions and path state

`StateRegionPlan.half_open(lower, upper)` uses the convention `lower <= x < upper` on every event coordinate. Regions compose with `&`, `|`, `^`, and `~`; composition preserves a canonical identity. A callable region must be constructed with `StateRegionPlan.from_predicate(..., region_id=...)` so the opaque callable has a caller-owned identity.

`PathBuffer.from_trajectory` pads a trajectory to a declared capacity. The active prefix is represented redundantly by `length` and the canonical mask `arange(capacity) < length`; inactive positions and times are canonical zeros and inactive lineage is `-1`. `lineage` records the source point for each active slot. `time_reversed()` reverses only the active positions and point lineage, preserves the increasing time schedule bitwise, and flips `direction`, making reversal an exact involution.

## Dynamics and actions

A `FunctionalDynamicsKernel` has an explicit `DynamicsKernelCapabilities` value. In particular, stochastic kernels must declare a normalized transition density, backward propagation is required for shooting, and path reversal requires reversible dynamics. Regrowth moves additionally require `fixed_step=True` and the bitwise-canonical active grid `times[i] = i * kernel.time_step`; shifted or nonuniform schedules fail at preparation. Each step returns `DynamicsStep(state, log_transition_density, valid, status)`, whose density must agree with the kernel's declared transition density. A failed, overflowing, or nonfinite step rejects that proposal immediately; the runtime never retries with a different random draw.

`DeterministicPathAction` represents a singular deterministic path law. `NormalizedStochasticPathAction` combines a normalized initial density with normalized transition densities. `SurrogatePathAction` is intentionally marked unnormalized.

The reweighting boundary is explicit: only `NormalizedStochasticPathAction` can construct `ReducedPathPotential`. `cross_evaluate_path_potentials` evaluates common-support reduced potentials and returns `PathCrossEvaluation`; its `.samples` field is the existing `uq.ReducedPotentialSamples` input for MBAR, while `path_fep_work` extracts finite source-conditioned work arrays for FEP or BAR. Deterministic and surrogate actions fail closed, as does a cross-evaluation that produces an infinite reduced potential.

## Moves and detailed balance evidence

One-way and two-way shooting use a normalized `AbstractShootingSelector` and `AbstractShootingModifier`. Uniform and state-weighted selectors and identity and Gaussian modifiers are included. Shifting regrows a discarded endpoint, path reversal is metropolized, and neighboring interface replicas exchange symmetrically.

Every `PathProposalEvaluation` exposes these additive log-ratio terms:

- `target_log_ratio`
- `selector_log_ratio`
- `modifier_log_ratio`
- `propagation_log_ratio`
- `length_log_ratio`
- `exchange_log_ratio`

Their sum is `log_acceptance_ratio` for a valid proposal. Separate validity flags identify target, selector, modifier, propagation, length, and exchange failures. Uniform index-cardinality changes are reported as the explicit length correction rather than being hidden in selector evidence.

## TPS

Construct `TPSPlan(ensemble, kernel, action, ...)`, call `prepare_tps(plan, initial_path)`, then `initialize_tps(prepared)`. `tps_step(prepared, state, key)` performs exactly one proposal and returns a `TPSStep`. The state carries fixed-capacity accepted and rejected proposal ancestry. `move_kind` selects `"one-way-shooting"`, `"two-way-shooting"`, `"shifting"`, or `"path-reversal"`.

The plan, prepared runtime, and initial trajectory have distinct identities. A compiled step updates only arrays; no host identity hashing occurs in a compiled path.

## TIS and RETIS

`InterfaceNetworkPlan` owns ordered interface values, endpoint regions, and an identified progress coordinate. It constructs interface and minus ensembles with consistent identities.

`TISPlan` prepares one replica per interface. `tis_step(..., replica_index=i)` updates exactly one statically selected replica. `RETISPlan` adds the minus ensemble. `retis_step(..., move_kind="shooting", replica_index=i)` updates a replica; `move_kind="exchange"` swaps neighboring assignments and reports the entire Metropolis ratio in `exchange_log_ratio`.

The fixed selection is deliberate: callers can schedule replicas and exchanges on the host or compile one executable per schedule position without introducing dynamic Python mutation.

## Rates, committors, and uncertainty

`estimate_reactive_flux` records crossing count and exposure. `factorize_tis_rate` evaluates the standard flux-times-conditional-crossing-probabilities factorization and retains log factors.

`CommittorFitPlan` defines a fixed-work weighted logistic fit. `fit_committor` reports convergence, gradient norm, and nonfinite validity; `predict_committor` evaluates the fitted probability.

For correlated path observations, use `block_mean_uncertainty`, `autocorrelation_uncertainty`, or `moving_block_bootstrap_uncertainty`. All report an effective sample size and integrated autocorrelation time. The moving-block bootstrap uses circular blocks and a fixed number of resamples.

## Portable restart

`write_tps_restart(path, prepared, state)` writes a canonical, non-pickle array archive. It includes the committed path, every counter and proposal-evidence leaf, and fixed-capacity lineage records containing parent, candidate, and committed trajectory serials for both accepted and rejected proposals. Rejected candidate coordinates are not retained. `read_tps_restart` requires the exact plan, prepared runtime, initial trajectory, array shapes, dtypes, payload digest, and current trajectory identity. `write_tis_restart`/`read_tis_restart` and `write_retis_restart`/`read_retis_restart` apply the same contract to every interface and minus replica, including exchange counters and both sides of rejected exchanges. A mismatch or corrupt archive fails closed rather than partially restoring state.

Differentiation through path moves is meaningful only while topology, capacities, event schedule, and accept/reject decisions are fixed. No score-function or discontinuous-event estimator is implied by the compiled array path.
