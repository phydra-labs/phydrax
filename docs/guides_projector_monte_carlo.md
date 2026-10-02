# Native projector Monte Carlo

The `phydrax.solver` projector evolves sparse signed or complex coefficients toward a ground state of an explicitly self-adjoint native lattice Hamiltonian. It is a bounded, single-device algorithm, not variational Monte Carlo (VMC), TDVP, quantum jumps, or a deterministic eigensolver. Start with the [complete cookbook](cookbook/projector_monte_carlo.md), [solver API](api/solver/projector_monte_carlo.md), and executable [public example](https://github.com/phydra-labs/phydrax/blob/dev/examples/projector_monte_carlo.py).

## Addresses, columns, and bindings

`QuantumConfigurationDomain` declares the ordered local spaces, explicit species/component IDs, fermion mode order, and optional exact integral or modular charges. `QuantumAddressCodec` packs local coordinates into exact `uint32` word vectors; canonical spare bits and range/charge predicates are checked. The all-zero key is a legitimate configuration, not an empty-slot sentinel. Activity is a separate mask. No hash, scalar Hilbert rank, ambient enumeration, or globally representable Hilbert dimension is required.

This differs from the finite `FixedCardinalityFermionBasis`, `FixedSpinProjectionBasis`, `FixedBosonNumberBasis`, and `FixedAbelianChargeBasis` APIs: those retain admitted dynamic-programming rank/unrank tables and an explicit finite sector dimension. They remain the direct route for finite native eigenproblems, symmetry sectors, TPQ, and response.

`prepare_quantum_lattice_columns(specification, resources)` prepares sparse local transition topologies and separate numerical bindings. Equal structural local-operator IDs do not mean equal matrix values: every ordered source-term/factor occurrence has its own binding slot. `refresh_quantum_lattice_columns` may replace values only with unchanged ordered terms and sparse support. Changed numerical values change operator identity; changed topology requires fresh preparation. Neither refresh nor equal array shapes authorizes continuing an old scientific run.

An outgoing column is **`H[target, source]`**. Repeated-site products, fermionic signs, duplicate paths, and all return-to-source contributions are included. Exact columns coalesce equal targets. The retained VMC adapter consumes **`H[current, connected]`**, conjugating the outgoing values only after self-adjoint certification; do not interchange these orientations.

## Probability-correct propagation

For physical coefficients, the finite-step equation is

$$c'_i=c_i+\Delta\tau\left(S c_i-\sum_j H_{ij}c_j\right).$$

The source diagonal is summed exactly and emitted once as `(1 + dt * (shift - diagonal)) * coefficient`. Raw-route sampling proposes a monomial and successive sparse local transitions with their actual probabilities. Its `p_raw` is a **route probability**, not a coalesced-target probability. Multiple routes can reach the same target. Dead paths, forbidden configurations, diagonal proposals, and padded transitions remain null attempts; conditioning on successful excitations would change the law. Sampled off-diagonal contributions use the inverse proposal probability and the requested attempt count.

Spawning is `exact`, `sampled`, or `semistochastic`. A sampled source requests `n = max(1, ceil(boost * abs(coefficient)))`; requests above `attempt_capacity` refuse, never clamp. Semistochastic spawning is exact when

$$\frac{|c|}{B}\geq\frac{\text{relative_threshold}}{\text{boost}}
\quad\text{or}\quad
|c|>\frac{\text{absolute_threshold}}{\text{boost}},$$

where `B` is the physical sparse-table-derived `raw_route_bound` (zero bound selects exact). Relative equality is inclusive; absolute equality is not. With a guide, `c` here is the represented coefficient. `B` is not `maximum_raw_routes`, group capacity, event capacity, or support capacity: enlarging storage must not change the scientific spawning decision.

Events stream through bounded chunks into exact word-key groups with seeded compensated high/correction accumulation. Every incoming contribution, including the retained source contribution, participates in **complete annihilation before one late compression**. A temporarily zero group remains retained until completion; early pruning or per-chunk compression changes cancellation and resource semantics. Threshold compression retains a nonzero coefficient below `theta` with probability `abs(coefficient)/theta`, preserving its phase and conditional expectation. This is a per-step algorithm property, not an unbiased stationary-energy certificate.

## Units and a frozen positive guide

Declare native `energy_unit` and exactly reciprocal `inverse_energy_unit`, plus one explicit native unit per requested physical observable. `dt` is scaled imaginary time in inverse-energy units; `dt * H` is dimensionless. If dimensional imaginary time is used externally, `dt` corresponds to `Delta t / hbar`, not an unexplained time in seconds. Shifts and `reference_energy` use the declared energy scale. The original Hamiltonian must certify self-adjointness; a guide is not permission to admit an arbitrary non-Hermitian physical Hamiltonian.

`QuantumGuide` freezes a callable `StrictModule` provider returning `LogAmplitude` and binds its explicit provider/mapping/domain/numerical identities. Only its positive magnitude is used. `globally_positive=True` is a provider declaration, **not a global proof** obtained from encountered configurations. Nodes use explicit `reject` or `positive-log-floor` policy. A finite positive floor can preserve the exact invertible similarity when the actual resulting guide is valid throughout the claimed domain; it is not inherently a physical bias. Invalid phases, nonfinite values, or numerically unrepresentable ratios/metrics refuse rather than being silently regularized.

With `D = diag(g)`, the represented vector is `x = D c` and the propagated operator is `H_g = D H D^-1`. Physical overlaps use the **same actual guide**, including the selected floor:

$$c_i=x_i/g_i,\qquad M=D^{-\dagger}D^{-1},\qquad M_{ii}=1/g_i^2.$$

For a constant guide `g=3`, the metric is `1/9`, not `1/3`. A physical observable `Q` uses `D^-dagger Q D^-1`, with elements `Q_ij/(g_i g_j)` for positive real guides. The observer recovers physical coefficients and applies the original physical columns, retaining complex observables. An ordinary Euclidean Rayleigh quotient of `H_g` is not a physical estimator.

## Raw histories and statistical qualification

Every accepted transition records its incoming applied shift, resulting population, projected numerator/denominator, and every ordered replica-pair observable numerator/denominator. Projected estimates are ratios of aligned means, not averages of instantaneous ratios. Independent replica controllers do not make pairs sharing a replica independent: ordered-pair numerators and denominators are aggregated **per time** before temporal covariance analysis.

[Correlated ratios](api/uq/correlated_ratios.md) retain full joint real-channel covariance, synchronous complete batch means, raw-channel and ratio-influence correlation diagnostics, and explicit unresolved-tail/insufficient-block statuses. Real denominators require bounded Fieller evidence; complex denominators require a confidence region excluding the origin. An exploratory finite value is not a qualified estimate. Observed constant stochastic records remain unresolved; deterministic source declarations must not be inferred from a quiet run. Statistical confidence uses an asymptotic approximation, not a finite-history guarantee.

`ProjectorEstimatorPolicy` selects burn-in/cadence and history depths from the complete accepted-step history. For a measured state numbered `n`, depth `h` uses the incoming shifts of exactly its preceding `h` accepted transitions, including the transition into that state. It never weights only the thinned measurement sequence. Incomplete windows are excluded, not filled. Replica-pair weights combine the histories of both replicas. Log normalization avoids unnecessary overflow; weight ESS measures concentration, not temporal independence.

Finite exponential history reweighting does **not** exactly cancel the additive finite-step Euler projector or certify absence of population-control bias. Compare population, history depth, timestep, relaxation, stationarity, and sign resolution independently of small error bars. Any applicable asymptotic bias-removal argument needs `dt -> 0`, `h -> infinity`, a horizon `h * dt` resolving physical relaxation, and its population/sign assumptions. No universal order-independent joint limit is promised. Analysis records propagation, statistical, weight, and systematic evidence separately.

## Resource refusal, checkpointing, and same-draw replay

`ProjectorMonteCarloPlan` admits replicas, support `S`, intermediate groups `G >= S`, event chunk `E`, attempts `A`, per-source work, raw history `T`, and retained/workspace bytes. All capacities are explicit positive bounded native work indices; preparation also admits packed-address, local-transition, route, and decoded-coordinate storage. Keys are `uint32`, coefficients `complex128`, shifts/history diagnostics `float64`, and accepted-step counters `int64`. `jax_enable_x64` is required; distributed/multidevice arrays and float32 execution are refused. These are capacity/precision contracts, not qualified scale claims.

Attempt/work limits, intermediate group overflow, final support overflow, extinction, history exhaustion, and numerical/guide/operator/observation failures are distinct. A refused step returns the prior committed state: coefficients, all replica controllers, raw history, logical step, and root random stream roll back atomically. Never discard events, evict groups, clip support, compress early, or retry with a new key until success.

Write a committed state with `write_projector_monte_carlo_checkpoint(path, prepared, state)`. Restore with `read_projector_monte_carlo_checkpoint(path, prepared, template)`: the caller prepares the exact scientific/storage structure and key implementation; executable providers are not deserialized. The archived root stream is retained even if the template uses another root value. Continue with the same `solve_projector_monte_carlo` entry point.

For explicit resource enlargement, prepare the same scientific problem and numerical policy with enlarged capacities, then call `transport_projector_monte_carlo_resources(source, target, state)`. It returns the enlarged state **and a native restart relation** and retains raw history/cursors/root key. It does not permit changed replicas, Hamiltonian values, domain, guide, units, timestep, or spawning/controller/compression policy. Transport includes snapshot/destination byte admission. Replay the refused logical step from this transported commit using its original physical draws. Random addresses include full physical key words, accepted logical step, replica, local attempt, and role, not storage lanes/chunks/capacities or failed wall-clock attempts.

Rich statistical results contain many independently typed arrays. The projector
result writer supplies native `ArrayArchiveLimits` derived from its retained-byte
budget and exact owned member count; it does not change the archive format or
disable admission. `lifecycle.create` accepts these explicit limits and uses them
for publication and reopening. A later untrusted `lifecycle.open` still requires
the caller's explicit member/byte budget when the archive exceeds the conservative
default member count. Use the returned validated archive directly for immediate
queries, or preserve the run's resource policy for a later bounded read.

## Candidate evidence and scope

The exact `quantum_lattice_candidate_profiles()` declarations are unreleased candidates, not release evidence. Resource, sign, raw-route probability, raw-history, guide-metric, denominator, same-draw replay, and independent locked-reference gates remain explicit. The qualification driver `tools/projector_monte_carlo_qualification.py` and phase-separated `benchmarks/projector_monte_carlo.py` produce their own observations; a driver existing or completing does not release a profile or establish accelerator superiority.

This implementation does not claim initiator approximations, excited-state targeting, distributed execution, transcorrelated Hamiltonians, sign/phase-problem removal, unrestricted size, unbiased stationary energies, differentiability through discrete stochastic choices, or transferable accuracy outside the exact checked support. Existing finite eigen, VMC, TDVP, TPQ, response, and quantum-jump workflows are unchanged.

Scientific context: [Rimu software and algorithms](https://arxiv.org/html/2601.19505) describe coefficient projector methods and their statistical/resource distinctions. This is native composition, not a Rimu backend or compatibility API.
