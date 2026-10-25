# Atomistic dynamics

PhydraX atomistic dynamics is a fixed-capacity, conservative particle runtime for
molecules and materials. It extends the existing atomistic learning and material-particle
substrates; it does not introduce another atom identity or graph system.

## Contracts

`AtomicStructure` remains a position-bearing data snapshot. `AtomisticSystemPlan` owns
position-independent particle identity, masses, force-field types, charges, regions,
cell policy, and molecular topology. `AtomisticDynamicsState` owns time, step index,
`configuration_revision`, kinematics (positions, momenta, integer image counts),
species, the runtime cell vectors, exactly one `AtomisticNeighborhoodBundleState`
(`state.neighborhoods`), the force cache (`state.force`, an `AtomisticForceState`),
constraint multipliers and residuals, the typed controller continuation
(`state.controllers`), the RNG key, the accepted energy ledger (`state.energy`), and
the read-only term states (`state.term_states`). There is no separate neighborhood
cache, thermostat state, or barostat state.

`AtomisticControllerState` holds every thermodynamic controller of the bound method:
`particle_chain` and `cell_chain` (Nose–Hoover chains), `gle` (generalized-Langevin
memory), `cell` (the MTK chart momentum and the reference lattice rows), and the
counters `move_counter`, `accepted_moves`, `cell_epoch`, and `rescale_events`.
Controllers absent from the method are `None`; their presence is fixed by the method.

The force cache is valid only under its complete key. The single rule is
`dynamics.force_cache_valid(state, controls)`. It requires a successful evaluation by
this runtime's prepared Hamiltonian and bundle, the current `configuration_revision`,
the row's controls, the bundle's source and graph epochs, the RNG key, and the
revisions of every bound term state. When `dynamics.strain_response` is set, it also
requires an available strain derivative. Every committed coordinate or cell change
advances `configuration_revision`. `force_cache_finite(force)` checks finiteness of the
fields a transition consumes; unrequested strain fields are NaN by contract and are
not checked. The cache stores `strain_derivative = dE/dstrain` and the
tension-positive `stress`. It stores no virial.

`AtomisticScaleContract` continues to define exact length and ordinary
single-simulated-system energy for learning and electronic operators.
`AtomisticUnitSystem` composes it with mass, time, charge, temperature, and a
named physical constant set. Host construction derives
`kinetic_to_energy = sM*sL^2/(sT^2*sE)`, its reciprocal force rate, Boltzmann,
Coulomb, and reduced Planck constants, plus pressure, velocity, and frequency
`UnitDefinition` values. Callers never supply those numeric constants.

The `electronvolt_angstrom_dalton_femtosecond()` preset retains the eV-angstrom-
dalton-femtosecond-K numerical convention. `reduced()` is one canonical
uncalibrated reference system and cannot be converted to SI. Learned models,
potentials, and batches compare only the length/energy scale; systems, dynamics,
and trajectories compare the complete unit-system identity.

## Material activation boundaries

`prepare_dormant_system` restricts a fully parameterized material system to the
chemically present support, rebuilding bonded routes, exceptions, constraints and
interaction-site identities. Future atoms are not merely mobile-mask exclusions.
`TopologyEpochTransition` and `activate_topology_epoch` replace the complete prepared
runtime between fixed-topology segments, refresh force/neighborhood identities, and
retain an `InsertionLedger` of mass, momentum and energy sources.

The insertion profile is nonperiodic and Cartesian with an identity coordinate map.
Existing material identity, capacity, units, masses and chemistry cannot change.
Virtual-site activation and periodic insertion are refused rather than inferred.
Failed evaluation retains the old runtime and state. Segments separated by activation
are not continuous lag-pair data, and derivatives do not pass through this host
topology transaction. See the [protein guide](guides_protein_folding.md) for the
reference-conditioned nascent-chain consumer.

## Potential programs

`AtomisticPotentialProgram` is an ordered additive scalar-energy program. Each term
declares its spatial requirements and execution capabilities. Preparation constructs pair
geometry, directed `GraphIR`, bonded routes, or a reciprocal grid only when a term needs it.
Forces are the negative position gradient of the total scalar energy.

Supplied unwrapped/fractional coordinate representations follow Cartesian
perturbations on their declared fixed image branch. They are not detached,
Cartesian-independent bonded inputs; force and curvature derivatives remain tied
to the conservative energy.

Native terms include harmonic bonds and angles, finite-extensible nonlinear-elastic
bonds, periodic proper or improper torsions, Lennard-Jones, direct Coulomb, direct
Ewald, and particle-mesh Ewald. Lennard-Jones cutoff-energy shifting is explicit and
mutually exclusive with switching; WCA additionally requires cutoff `2¹ᐟ⁶σ`.
Pair exclusions and 1–4 scales are explicit sparse stable-ID exceptions. Active
singular geometries and FENE extension at or beyond the maximum fail; they are not
repaired by clipping distances or logarithm arguments.

PaiNN, NequIP, and native MACE use `LearnedGraphPotentialTerm`, which declares a
directed image-graph neighborhood requirement. Route storage belongs to the
neighborhood bundle plan: image source search capacities and role capacity ladders.
Execution resources belong to `AtomisticGraphExecutionPlan`, never to model
architecture identity. These include the optional `StreamedRelationPlan`, which
prepares the receiver-major schedule once per topology epoch. Periodic learned-graph
execution requires `allow_periodic=True`. That is an execution capability, not evidence
that a fitted model is stable for molecular dynamics.

## Neighborhood bundles

A prepared Hamiltonian owns exactly one neighborhood bundle, and dynamics has no
second neighborhood argument. The requirements are the canonical
`program.requirements.neighborhoods`. Each `AtomisticNeighborhoodRequirement` names an
owning binding, an `AtomisticNeighborhoodRole`, a cutoff, and optional nearest-neighbor
selection, receiver group, and pair-exclusion convention. The roles are classical
minimum-image pairs, interaction-site pairs, directed image graph, coordination, and
image-pair sum. A cutoff of `None` declares the complete finite all-pairs relation of a
physical or interaction-site pair role, and is admitted only without periodic support.
Requirements carry no storage, so their identities stay stable across capacity growth.

`AtomisticNeighborhoodBundlePlan(requirements, sources, role_capacities, *,
case_count=1, frame=None)` binds every requirement explicitly:

- An `AtomisticNeighborhoodSource(name, entity, serves, search, capacities, *, skin,
  search_radius=None, receiver_group=None, receiver_rows=None)` is one certified image
  search. The `search` policy is `AtomisticCellListImageSearch`,
  `AtomisticClusterTileImageSearch`, `AtomisticTiledImageSearch`, or the bounded
  `AtomisticDenseImageSearch` reference. `capacities` is a
  `ParticleImageCapacityLadder` whose entries use the per-case segment layout
  (`maximum_case_edges`). The search radius defaults to the largest served cutoff plus
  `skin`.
- An `AtomisticCompletePairSource(name, entity, serves, *, maximum_pairs)` serves the
  `cutoff=None` pair requirements of finite systems. `maximum_pairs` is an explicit
  per-case bound that `N (N - 1) / 2` must fit.
- Every bounded requirement has exactly one `AtomisticRoleCapacityLadder(requirement,
  entries)`. Its entries are strictly growing `AtomisticRoleCapacity(maximum_routes=...,
  maximum_degree=..., maximum_case_routes=None)` values. Complete requirements have
  none.
- `frame` is the explicit nonperiodic search domain of a finite system with image
  sources.

A binding is admitted only when the source searches the requirement's entity and
covers `cutoff + skin`. There is no automatic dense fallback and no hidden default.
`program.prepare(system, neighborhood_bundle=..., graph_execution=None)` accepts the
plan or an already prepared `PreparedAtomisticNeighborhoodBundle`.
`potential.neighborhood_bundle` is the prepared bundle, and
`potential.with_neighborhood_bundle(bundle)` rebinds it without rebuilding unchanged
term preparations. `dynamics.bundle` is the same prepared bundle, and
`dynamics.bundle.slot(requirement)` and `dynamics.bundle.capacity(slot)` report a role
view's slot and its current storage. Standalone evaluation forms the bundle state from
case-batched geometry with `bundle.initialize(bundle.geometry(...))` and
`bundle.refresh(state, geometry)`. The program's `energy` and `evaluate` methods take
that one bundle state as their second positional argument.

Routes of image sources carry explicit integer translations. For row cell `H`, every
route has

`d_e = r_receiver - r_source + n_e @ H`

with an explicit integer translation `n_e`. The pair `(i == j, n == 0)` is excluded.
Graph and image-sum roles keep every nonzero self image and every repeated image of a
pair as separate routes. A classical minimum-image pair role rejects a repeated
minimum image or a self image inside its cutoff (`IMAGE_AMBIGUITY`), so classical pair
multiplicity is preserved. A program with classical terms also requires the cell's
unique-image radius to cover every classical cutoff at preparation, and runtime cells
outside that guard fail closed. Rewrapping an atom across a periodic face updates the
stored `n` by the exact image-count difference. The source certificate charges particle
displacement and cell deformation; when it expires, the source searches again. A cell
outside the prepared stencil envelope is a scientific failure, not a capacity failure.

### Rejections and capacity replay

No overflow truncates neighbors. `AtomisticStepRejectionReason` keeps separate bits:
`SOURCE_CAPACITY` (image search storage), `ROUTE_CAPACITY` (role view storage),
`DOMAIN`, `RELATION`, `IMAGE_AMBIGUITY`, `POTENTIAL`, `CONSTRAINT`, `NONFINITE`,
`STALE_FORCE`, `UNIT_SYSTEM`, `THERMOSTAT`, `THERMODYNAMIC_STATE`, `CELL`, `BAROSTAT`,
`STRAIN_UNAVAILABLE`, `FORCE_PORT`, `TERM_STATE`, and `BINDING`. `CAPACITY_REJECTIONS`
is the integer mask `SOURCE_CAPACITY | ROUTE_CAPACITY`. A failed step retains the last
accepted state.

Capacity growth is a declared host transaction.
`phx.atomistic.retry_atomistic_step_with_capacity(dynamics, state, thermodynamic,
states)` attempts one step. It replays only when the rejection carries a capacity bit
and no other cause, apart from the potential and nonfinite flags that an overflow
causes. On a replay it does the following:

1. `dynamics.bundle.grow_capacities(...)` selects the next covering declared entries.
2. `PreparedAtomisticDynamics.rebind_hamiltonian` rebinds the accepted state to
   `potential.with_neighborhood_bundle(grown)`.
3. `prime_state` re-primes the force cache.
4. The same physical attempt is retried at the same step and stochastic address.

Kinematics, cell, controllers, counters, RNG, energy ledger, step index, and revision
are retained exactly. Scientific, domain, stale, thermodynamic, and constraint failures
are returned unchanged and never retried. An exhausted declared ladder raises, so
capacity never grows beyond what the plan declares.

### Priming, rebinding, and Hamiltonian transitions

`initialize_state(positions, thermodynamic, *, velocity=None, momentum=None, key,
term_states=None, cell_vectors=None, ...)` forms and primes the initial accepted state
and raises if it is inadmissible. Exactly one of `velocity` and `momentum` is required.
`initialize_state_detailed` takes the same arguments and returns an
`AtomisticInitialStateEvaluation(state, successful, rejection_reasons)` instead. A
failed initial state carries an unsuccessful force cache, so every transition refuses
it as stale.

`prime_state(state, table)` evaluates every required energy, force, and strain resource
at an accepted state without consuming an event. The bundle is refreshed at the
accepted geometry, and the force cache is rebound to the current revision, controls,
epochs, and key. Time, step, revision, RNG, controllers, counters, and adaptive
histories are untouched, and the energy jump is booked as external work. It returns
`AtomisticPrimeEvaluation(candidate_state, accepted_state, potential, successful,
rejection_reasons)`.

`rebind_hamiltonian(potential, state, table, states)` is resource-only. The new
potential must have the same `scientific_id`. The returned state's force cache is stale
until `prime_state` refreshes it. `transition_hamiltonian(potential, state, table,
states, *, term_states=None)` moves an accepted state to a different scalar
Hamiltonian of the same prepared system, such as new weights, terms, or coefficients.
The table must keep its state IDs and intensive variables. Forces are re-evaluated at
the unchanged state, and the energy jump is booked as external work and reported as
`energy_jump`. `lineage_id` binds the source and target `scientific_id`, prepared
runtimes, and tables. Term states carry over only when the stateful slots and their
owners are unchanged; otherwise `term_states` must bind the new owners explicitly. On
failure the source runtime, table, and untouched state are returned.

## Thermodynamic state and methods

`AtomisticPhaseSpaceMeasurePlan(system, *, scaled_entity_count=None,
volume_coordinate_convention="molecular-center", momentum_restriction="none")` is the
single authority for the particle, topology, mass, constraint, cell, and unit measure.
`volume_coordinate_convention` names what a volume change scales: `"molecular-center"`
scales molecule centers, and `"atomic"` scales every active atom.
`momentum_restriction="center-of-mass"` declares total linear momentum a removed
degree of freedom, and requires every active atom to be mobile.

`AtomisticThermodynamicStatePlan` declares `nve`, `nvt` (temperature), `npt`
(temperature and pressure), or `nph` (pressure only) intensives and Hamiltonian
controls. `PreparedThermodynamicStateTable` supplies the numeric rows used during
execution. The table's `degrees_of_freedom` is

`N_f = 3 * mobile_count - independent_constraint_count - (3 if momentum_restriction == "center-of-mass" else 0)`

and the instantaneous temperature `2K / (N_f k_B)` uses it. MTK inertia and
Nose–Hoover masses use the same `N_f`.

The integrator alone fixes which ensembles a table may bind. Every refusal is raised
when the table is prepared.

| Method | Ensembles | Notes |
| --- | --- | --- |
| `VelocityVerletPlan(dt)` | nve | conserves `K + U` |
| `BAOABLangevinPlan(dt, friction, *, realization_id=0)` | nvt; npt with a scheduled Monte Carlo cell | refuses a center-of-mass restriction |
| `NoseHooverChainPlan(dt, NoseHooverChainParameters(tau, *, chain_length=3, yoshida_suzuki_order=1, substeps=1))` | nvt; npt with a scheduled Monte Carlo cell | Suzuki–Yoshida order 1, 3, or 5; chain energy enters the extended energy |
| `StochasticVelocityRescalingPlan(dt, tau, *, realization_id=0)` | nvt; npt with a scheduled Monte Carlo cell | exact CSVR kinetic propagator |
| `GeneralizedLangevinPlan(dt, transition_matrix, noise_factor, ...)` | nvt; npt with a scheduled Monte Carlo cell | refuses a center-of-mass restriction |
| `VelocityRescalingPlan(dt, *, interval=1)` | nvt | deterministic and noncanonical |
| `MTKNPTPlan(dt, barostat, particle_chain, cell_chain)` | npt | owns the cell; atomic measure; no constraints |
| `MTKNPHPlan(dt, barostat)` | nph | owns the cell; no thermostat; atomic measure; no constraints |

`VelocityVerletPlan` implements conservative kick–drift–kick integration with canonical
momenta. `BAOABLangevinPlan` owns step size and friction while the selected
thermodynamic row supplies temperature. Its Ornstein–Uhlenbeck substep is exact, but the
finite splitting is not advertised as an exact canonical sampler without separate
kernel qualification. Randomness is addressed by root key, realization, accepted step,
operator, and stable particle ID.

`NoseHooverChainPlan` uses the schedule `T(dt/2) K(dt/2) D(dt) [constraints] K(dt/2)
T(dt/2)`. The chain masses are `Q_1 = N_f kT tau^2` and `Q_k = kT tau^2` (`k > 1`),
evaluated from the bound row at every application, so a warm temperature rebase
rebinds the chain inertia explicitly. `StochasticVelocityRescalingPlan` is the
Bussi–Donadio–Parrinello method (J. Chem. Phys. 126, 014101, 2007). It applies the
exact CSVR kinetic-energy propagator over each half step of `R(dt/2) K(dt/2) D(dt)
[constraints] K(dt/2) R(dt/2)`. Each application makes two global scalar draws: a
Gaussian and a chi-square with `N_f - 1` degrees of freedom.

`VelocityRescalingPlan` scales all admitted momenta by `sqrt(T_target / T)` after every
`interval`-th committed velocity Verlet step. It is an equilibration control, never a
canonical sampler. `canonical` is false, the injected energy is booked as reservoir
heat, and it is refused wherever a canonical measure is required, such as a Monte
Carlo volume move.

`HydrodynamicBrownianPlan` is the transactional fixed-capacity Itô position
runtime. It composes a prepared matrix mobility with deterministic force drift,
matrix-square-root Brownian increments, centered random-finite-difference thermal
drift, optional affine background velocity, stable replay keys, and rollback on
any failed proposal. `OverdampedAtomisticPlan` remains the constant-isotropic
mobility convenience entry point. Free-space and periodic RPY and confined
MAC/FIB routes share the same runtime; holonomic constraints remain unsupported.

`GeneralizedLangevinPlan(step_size, transition_matrix, noise_factor, *,
covariance_tolerance=1e-10, realization_id=0)` is the single generalized-Langevin
owner. It is an in-schedule thermostat of `PreparedAtomisticDynamics`
(`O(dt/2) K(dt/2) D(dt) [constraints] K(dt/2) O(dt/2)`) whose half-step operator
is a fixed discrete transition of the momentum channel and `K` auxiliary channels
in thermal-reduced coordinates. The discrete fluctuation-dissipation identity
`T Tᵀ + S Sᵀ = I` is checked at construction within `covariance_tolerance` and
reported at execution; the target temperature comes from the bound thermodynamic
row, and the auxiliary memory is a typed dynamics continuation. Noise is
addressed by stable particle identity, committed step, half-step phase, and
channel. Supplying such a kernel does not by itself establish coarse kinetic
fidelity.

`DistanceConstraintPlan(*, maximum_iterations=32, tolerance=1e-10,
maximum_component_atoms=16, rank_tolerance=1e-8)` applies fixed-capacity SHAKE/RATTLE
position and momentum projections. Admission assumes constant rank:
`PreparedDistanceConstraints.independent_constraint_count` declares every topology
constraint independent and equals the topology constraint row count. That declaration
is certified only at explicit geometries. `rank_evidence(positions)` returns a
`ConstraintRankEvidence(component_rank, minimum_relative_singular_value, regular)`
computed per connected component of the constraint graph from the mass-weighted
Jacobian. A singular value counts toward the rank when it exceeds `rank_tolerance`
times the component's largest. `regular` holds when every component has full row rank.
`maximum_component_atoms` bounds the per-component Jacobian, and a larger component is
refused at preparation. Dynamics checks the rank at initialization and after every
position projection. A trajectory that reaches a rank-changing configuration is
rejected (`CONSTRAINT`) rather than silently changing its degrees of freedom.
Constraint residuals, iterations, multipliers, velocity tangency, and work remain
explicit.

## Native force ports and term states

`AtomisticDynamicsPlan(system, potential, integrator, *, constraints=None,
strain_response=None, force_ports=())` takes every neighborhood resource from
`potential.neighborhood_bundle`. `force_ports` are typed `AbstractAtomisticForcePort`
contributions: Cartesian forces that are not the gradient of any scalar the program
holds. Each port is evaluated at every conservative force boundary with the same
accepted geometry. Its forces are added inside the integrator's own kick operators,
before constraint projection, and are cached separately as
`state.force.nonconservative_forces`. A port has no energy and no strain derivative.
Its trajectory work is booked as the step's `external_work`.

A runtime with ports reports `conservative_force_definition=False`. It refuses every
consumer that needs the complete scalar Hamiltonian:

- integrator-owned MTK cells, refused at plan construction;
- Monte Carlo volume moves;
- multistate thermodynamic tables (cross-evaluated reduced potentials);
- controlled switching work;
- RESPA force-group splitting.

Its pressure tensor is NaN.

Stateful terms and ports read their state from `state.term_states`, an
`AtomisticTermStateBindings` mapping of named slots. The slots are each stateful term's
`name` and each port's `name`, which must be distinct. `initial_term_states(dtype)`
supplies the declared initial states. A missing, unsuccessful, or foreign-owner state
is a `TERM_STATE` rejection, and the force cache records the term-state revisions it
consumed. Biases, adaptive histories, and ABF ports are documented in the
[sampling guide](guides_atomistic_sampling.md). Adaptive-history trajectories are not
equilibrium samples.

## Periodic stress, PME, and pressure

`atomistic_cell_energy_and_stress`, and `program.evaluate(..., compute_stress=True)`,
differentiate the same scalar energy with respect to homogeneous strain, at fixed
fractional coordinates and fixed integer images. The convention is unchanged: column
deformation `F = I + strain` maps row lattice vectors to `H' = H @ F.T` and positions to
`origin + (r - origin) @ F.T`. `AtomisticStressConvention` names the result:
`stress = sym(dE/dstrain) / |det H|`, positive under tension.

- `CAUCHY_TENSION_POSITIVE` applies to fully periodic cells, where `|det H|` is the
  physical volume.
- `CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME` applies to partially periodic cells. It
  requires a full invertible 3x3 cell whose nonperiodic rows declare the embedding
  thickness.

A lower-rank lattice without an embedding volume is refused rather than given an
invented thickness. Requested stress also refuses unless every term owns a cell
derivative. It never falls back to zero or to the evaluation's `virial` field. That
field is the finite position-moment diagnostic `-sum (r - c) (x) f` about the active
center of mass. It is not a periodic virial, and dynamics neither caches it nor uses it
for pressure.

Learned directed graphs with `allow_periodic=True` own the cell derivative through the
explicit image vectors `n @ H`. Stress is therefore available for periodic PaiNN,
NequIP, and MACE programs, including stress beyond the unique-image radius. Image terms
enter the strain derivative directly. A self-image edge contributes zero net positional
force but a nonzero strain response, so a one-atom periodic cell has zero atomic force
and a nonzero strain derivative.

`ParticleMeshEwaldPotential` uses the native periodic tensor B-spline particle-grid
transfer, an FFT reciprocal solve, B-spline influence correction, real-space Ewald,
self energy, pair-exception correction, and an explicit neutral or uniform-background
policy. `EwaldReferencePotential` provides a direct reciprocal reference.

### Strain stress, pressure, and the position moment

Three tensors are kept distinct:

- The **strain stress** `sym(dE/dstrain) / V` is configurational and tension-positive.
  It is taken at fixed fractional coordinates and integer images, so stored self images
  and boundary-crossing routes contribute exactly. The force cache stores
  `strain_derivative` and `stress` when `strain_available`.
- The **instantaneous internal pressure** is computed by
  `dynamics.pressure_tensor(state)`, which returns `(P, V)`, and reported as
  `diagnostics.pressure_tensor`:
  `P = (sum_i p_i p_i^T / m_i - sym(dE/dstrain)) / V`. It is the kinetic momentum flux
  of the mobile atoms minus the configurational strain stress, and is
  compression-positive. `diagnostics.pressure = trace(P) / 3`.
- The **position-moment diagnostic** `virial` of a program evaluation is a fixed-cell,
  finite-system quantity. It is never a substitute for either tensor above.

`dynamics.strain_response` requests `dE/dstrain` with every force evaluation. It
defaults to true exactly when the system has a fully periodic 3x3 cell and the program
owns its cell derivative, and it is mandatory for MTK methods. Requesting it otherwise
raises. Without a strain response, or with force ports, pressure and volume are NaN;
there is no finite position-moment fallback. Pressure contributions of individual
terms, for example a reciprocal-space or polarization term, are not separated in this
tensor.

### MTK cell dynamics

`MTKNPTPlan` and `MTKNPHPlan` implement the Martyna–Tobias–Klein equations
(J. Chem. Phys. 101, 4177, 1994) in the Liouville factorization of Tuckerman et al.
(J. Phys. A 39, 5629, 2006). The realization is `mttk-tuckerman-2006-exponential`:
position, cell, and momentum sub-flows are integrated exactly with the native small
matrix exponential and `phi_1` actions. The source's linear cell update is not used.
NPT uses `Tc(dt/2) Tp(dt/2) B(dt/2) K(dt/2) D(dt) K(dt/2) B(dt/2) Tp(dt/2) Tc(dt/2)`,
and NPH uses `B(dt/2) K(dt/2) D(dt) K(dt/2) B(dt/2)`. With `G = p_g / W`, the equations
are

- `dr/dt = p/m + G r`,
- `dp/dt = F - (G + tr(G)/N_f I) p`,
- rows `dH/dt = H G^T`,
- `dp_g/dt = Pi_chart[sum_i p_i p_i^T / m_i - sym(dE/dstrain) - V P_ext I + (2K/N_f) I]`,

plus the chain friction terms under NPT.

- **Chart.** `MTKBarostatPlan(tau_p, *, chart="isotropic", inertia_temperature=None,
  maximum_condition=1e3)` selects an `AtomisticCellChart`, which is a
  Frobenius-orthogonal projection of the cell force. The charts are `isotropic` (scalar
  `p_eps`, `K_cell = p_eps^2 / 2W`), `semi-isotropic-xy` (`diag(a, a, c)`), `diagonal`,
  and `triclinic-symmetric` (six cell degrees of freedom). The last three use
  `K_cell = tr(p_g^2) / 2W`.
- **Gauge.** Every chart momentum is symmetric, so the gauge is rotation-free and no
  cell rotation is integrated.
- **Inertia.** `W = (N_f + 3) k T_W tau_p^2`. `T_W` is the bound row temperature under
  NPT, or the declared `inertia_temperature`. NPH has no thermostat of any kind.
  `inertia_temperature` only sets `W` and is never a physical target, and
  `MTKNPHPlan` refuses a barostat without it.
- **Pressure target.** Only a hydrostatic `P_ext` is supported. A tensor target is a
  different work and measure contract, and is refused.
- **Measure.** MTK scales every atom, so the phase-space measure must declare
  `volume_coordinate_convention="atomic"`. Constrained MTK lacks a derived constrained
  measure and projection, and is refused. MTK also requires a fully periodic 3x3 cell,
  a program that owns its cell derivative, and no force ports.
- **Condition.** `maximum_condition` bounds the runtime lattice condition number. A
  candidate cell beyond it is a `CELL` failure.
- **Conserved quantity.** `extended_energy = K + U + K_cell + P_ext V + E_chains`
  (`diagnostics.extended_energy`, `state.energy.extended_energy`). Its drift is a
  diagnostic of the realization, not a claim of stepwise equality with any external
  implementation. `state.controllers.cell` carries the chart momentum, and
  `cell_epoch` advances on every committed cell change.

Admission for learned-graph programs follows the same rules as for classical programs.
It is not evidence that a fitted model is stable under cell fluctuations.

### Monte Carlo volume moves

`IsotropicMonteCarloBarostatPlan(maximum_log_volume_change, *, realization_id=0)` is a
scheduled cell owner. Schedule it with `AtomisticRolloutPlan(..., barostat=plan,
barostat_interval=k)`, or apply one move with
`apply_isotropic_monte_carlo_barostat(dynamics, state, table, plan)`, which returns an
`AtomisticBarostatEvaluation`. A run has exactly one cell owner: an NPT rollout
requires either an MTK integrator or a scheduled move, never both. The move requires
the following:

- a fully periodic cell;
- a canonical particle method that does not own the cell;
- `npt` rows;
- a conservative force definition.

One move proposes `ln V' = ln V + u` with `u ~ U(-a, a)`. It scales the measure's
declared entities at fixed fractional coordinates, refreshes a pure candidate of the
accepted bundle, and evaluates the complete scalar Hamiltonian there, including any
embedded bias term. With the symmetric `ln V` proposal, the scaled-coordinate measure
`V^N dV` contributes the exact Jacobian:

`ln alpha = -beta (U' - U + P (V' - V)) + (N + 1) ln(V'/V)`

Here `N` is the measure's `scaled_entity_count`. Draws are addressed by the run key,
realization, move operator, `state.controllers.move_counter`, and draw index. The
counter advances once per completed Metropolis decision, whether accepted or rejected.
A capacity, scientific, resource, or stale-force failure advances nothing, so a
capacity-only replay re-attempts the identical move. With holonomic constraints, atomic
scaling is refused, and every constraint must stay within one scaled molecule.

## Rollout, replay, and persistence

`AtomisticRolloutPlan` carries trajectory buffers inside the scan. It never constructs a
hidden full trajectory before applying the sample stride. Retention is final-only or a
fixed-capacity trajectory. Full, per-step, and block rematerialization use the shared
checkpointed scan. They record route, image, and stochastic replay digests, and the
stochastic digest includes the Monte Carlo move counter.

Long runs are explicit fixed segments through `run_atomistic_segments`. Runtime
checkpoints use the canonical checksummed pickle-free array archive.
`AtomisticCheckpointPlan(dynamics, thermodynamic, *, scope_id=None)` binds the
following:

- the prepared dynamics and thermodynamic table;
- the system, the executed Hamiltonian and its program;
- the prepared neighborhood bundle, including its plan and capacity levels;
- the integrator and the scope.

`write_atomistic_checkpoint(path, plan, state)` stores the complete accepted state and
the bundle recipe with its capacity levels, but no model. There is no repair, implicit
minimization, or changed protocol on resume. A fresh process recreates the identical
prepared Hamiltonian over the same bundle plan. When the saved run grew its
capacities, it rebinds that Hamiltonian with
`read_atomistic_checkpoint_resources(path).bind(potential)` before building the
dynamics plan, and then calls `read_atomistic_checkpoint(path, plan, template)`.

For learned models, `write_atomistic_restart(path, plan, state, *, sources=None,
licenses=None)` writes a portable restart. The learned terms are derived from the
prepared Hamiltonian of `plan`; there is no model argument. For every executed learned
term, the restart archives the following:

- the term's native model artifact;
- its complete binding: name, force group, periodic admission, bound conditioning,
  typed random key, and observable request;
- the dynamics state, with the bundle recipe and capacity levels.

`sources` and `licenses` are keyed by learned term name. A key that names no executed
learned term is refused. Terms whose artifacts agree in full identity share one
archived artifact, and distinct revisions never alias.

`read_atomistic_restart_models(path)` returns a read-only mapping from term name to
`AtomisticRestartModelBinding`, in name order, so a two-model Hamiltonian restores two
bindings. Each `binding.term` is a `LearnedGraphPotentialTerm` rebuilt over exactly
`binding.artifact.model`. A fresh process prepares its Hamiltonian over the restored
terms and the recorded bundle plan, then rebinds it with
`read_atomistic_checkpoint_resources(path).bind`. It then rebuilds the dynamics and
resumes with `read_atomistic_restart(path, plan, template)`. The executed learned
terms must be exactly the bundled bindings. The following are refused before any state
is returned:

- a missing, additional, or altered term, model, conditioning, or key;
- a wrong source binding;
- an altered system, Hamiltonian, bundle plan or capacity level, integrator,
  thermodynamic table, scope, or state recipe.

The restored state is exact. Continued steps equal the uninterrupted trajectory
bitwise only under the following conditions:

- the executed graph plans select ordered streamed accumulation;
- the resuming process uses the same device type and runtime versions;
- the compiler profile is fixed (on GPUs, XLA autotuning disabled).

Compiled caches are not part of restart correctness. See the
[periodic MACE recipe](cookbook/atomistic_dynamics.md#periodic-native-mace-dynamics)
and the [atomistic API](api/atomistic.md).

Prepared-program identities embed the neighborhood bundle plan (requirements, sources,
capacity ladders, and levels) and the graph-execution identity. Dynamics states carry
one bundle state, a revision-keyed force cache, typed controllers, and term-state
bindings. States, checkpoints, and restarts written before this layout cannot be
resumed; rebuild the preparation and write a new checkpoint.

## Native relaxation

`AtomisticRelaxationPlan(program, chart, /, *, method, convergence,
target_pressure=0.0, group_count=1)` binds one prepared scalar Hamiltonian (an
`AbstractPreparedAtomisticHamiltonian`, such as the result of
`program.prepare(system, neighborhood_bundle=...)`), one coordinate or cell chart,
one native force-driven optimizer, and one convergence policy. `plan.prepare()`
returns a `PreparedAtomisticRelaxation` whose `group_count` independent
relaxations of that program run as the groups of one `OptimizerGroupLayout`. The
program owns its neighborhood bundle; there is no relaxation neighborhood argument
and no dense fallback. Systems with holonomic topology constraints are refused,
because no constrained-manifold retraction exists for them, and at least one atom
must be active and mobile.

The optimizer is `phydrax.optim.FIRE`, `FIRE2`, or `TrustScaledLBFGS`. FIRE2 and
trust-scaled L-BFGS are force-driven and never compare objective values, so they
do not claim monotone decrease. `FIRE(downhill_check=True)` compares the
objectives of accepted iterates and rolls back an increase. Method controls,
including their defaults, are configured parameters of the method. See the
[optimization API](api/optim.md#force-driven-local-relaxation).

### Charts and the minimized objective

Every chart maps optimizer parameters to *unwrapped* Cartesian positions and cell
rows. Integer image counts are formed outside differentiation, and only the
resource owner sees the wrapped point, so a boundary crossing never changes the
optimizer coordinates or its L-BFGS history. One reverse pass through
`chart.realize` gives the chart gradient as the exact pullback of the Cartesian
covectors `dE/dr` and `dE/dH`.

- `AtomisticCartesianChart(*, constraint=None)` is a fixed-cell chart over
  displacements `{"atoms": d}`, with `r = r0 + C d`. Immobile and inactive atoms
  are inactive optimizer sites.
- `AtomisticDeformationChart(kind=AtomisticCellChartKind.LOGARITHMIC, /, *,
  strain=None, constraint=None, cell_factor=None)` adds symmetric strain
  coordinates `{"strain": theta}`, with `c = sum_k theta_k B_k / cell_factor`.
  The deformation is `F = I + c` (`AFFINE`) or `F = exp(c)` (`LOGARITHMIC`, the
  native dense matrix exponential). Positions are `(r0 + C d) F^T` and cell rows
  `H0 F^T`. The generalized cell force is the pullback of the same objective
  through `F`, never `-V sigma`. `cell_factor=None` uses the number of mobile
  atoms. The chart requires a fully periodic 3D cell.
- `AtomisticSymmetryChart(layout, system_atoms, /, *, vary_cell=True)` moves an
  `AtomicSymmetryPlan` or `RigidSymmetryLayout` by its genuine retraction from a
  fixed anchor. `system_atoms` maps every active system atom exactly once, with
  matching elements, and every mapped atom must be mobile. Start it with
  `init(chart_states=...)`, stacked over groups. See
  [crystal relaxation](guides_atomistic_crystal_prediction.md#relaxation) and
  `symmetry_evidence(state, group)` for the tangent-operator residual and the
  symmetry deviation.

`AtomisticStrainSubspace` supplies Frobenius-orthonormal symmetric generators:
`symmetric()` (six Voigt strains), `hydrostatic()`, `fixed_volume()` (the five
traceless strains), and `selected(mask)`. A traceless subspace preserves volume
exactly only under the logarithmic chart, so an affine chart refuses it.
`AtomisticCoordinateConstraint(*, projectors=None, remove_translation=False)` is a
sparse per-atom tangent action, not a dense atom-by-parameter basis. Projectors
are symmetric idempotent `(N, 3, 3)` blocks, and `fixed_components(mask)` builds
them from an `(N, 3)` boolean mask. `remove_translation` removes the mean mobile
displacement. The two forms are separate contracts and cannot be combined.

A nonzero `target_pressure`, in program energy per cubed length unit, minimizes
the enthalpy `E + P |det H|` and requires a variable-cell chart. A tensor target
stress is not a scalar enthalpy and is not accepted.

### One round: accepted iterates, rollback, and capacity

`init(*, positions=None, cell_vectors=None, chart_states=None, active=None,
hamiltonian_lineage=())` starts every group at rest. `positions` are `(G, N, 3)`
unwrapped Cartesian coordinates for Cartesian and deformation charts, and
`active` `(G,)` marks lanes in use (others are `INACTIVE`). `step(state)` is pure
and device resident:

1. The pending proposal `state.optimizer.parameters` of every iterating group is
   realized by the chart.
2. Candidate resources are refreshed purely from that group's accepted bundle
   state. A trial never modifies accepted resources.
3. The objective, chart gradient, forces, and, for a variable cell, the cell
   covector and stress come from one reverse pass.
4. A capacity-only resource failure is not consumed. The group becomes
   `CAPACITY_PENDING` with trial outcome `CAPACITY`, and its optimizer state and
   pending proposal stay unchanged.
5. A scientific, nonfinite, or cell failure (a singular, nonfinite, or
   orientation-reversing cell) is consumed as a nonfinite evaluation. The
   optimizer returns to its accepted iterate, and the group terminates with
   `SCIENTIFIC_FAILURE` or `CELL_FAILURE`.
6. Candidate resources are committed only for groups whose evaluation the
   optimizer accepts. A valid trial that `FIRE(downhill_check=True)` rejects
   leaves the group iterating with outcome `ROLLED_BACK`.

Every reported position, cell, energy, objective, force, stress, and residual in
`AtomisticRelaxationState` belongs to the accepted iterate
(`state.optimizer.accepted_parameters`), never to the unevaluated proposal.
`AtomisticTrialOutcome` records the latest trial: `ACCEPTED`, `ROLLED_BACK`,
`CAPACITY`, `SCIENTIFIC`, or `CELL`. `rejected_trials` counts consumed but
rejected trials, and `capacity_replays` counts capacity-refused ones.

`grow(state)` is the host capacity transaction. For every `CAPACITY_PENDING`
group, it replays the pending proposal from that group's accepted resources at
successive declared capacity levels. It then rebinds the program with
`with_neighborhood_bundle` (a new `prepared_id`), rebases every accepted resource
state, and reprimes accepted evidence without consuming budget. Optimizer
histories are kept, and pending groups resume `ITERATING` with the identical
proposal. An exhausted declared ladder ends that group with `RESOURCE_EXHAUSTED`.
`solve(state, *, maximum_rounds, maximum_growths=8)` alternates the compiled
`run(state, *, maximum_rounds=...)` with `grow`, and returns the possibly rebound
runtime with the state. Continue only with that returned runtime.

### Residuals, convergence, and status

`AtomisticRelaxationConvergence(*, projected_tolerance, maximum_evaluations,
full_force_tolerance=None, full_stress_tolerance=None, require_full=False)`
keeps two residual families apart:

- **Projected**: the largest optimizer-site norm of the chart gradient, in chart
  units (`state.projected_residual`).
- **Full**: the largest Cartesian force norm over mobile atoms
  (`full_force_residual`) and, for a variable cell, the largest component of
  `sigma + P I` (`full_stress_residual`), where `sigma` is the tension-positive
  Cauchy stress of `E`.

A constrained or symmetry-chart minimum is not a full minimum unless the full
residual is also small. With `require_full=False` a group converges on the
projected residual alone, and `state.full_converged` reports the full residual
without terminating. `require_full=True` needs `full_force_tolerance`.
`full_stress_tolerance` applies only to variable cells. `maximum_evaluations`
counts consumed trials, not capacity replays.

`AtomisticRelaxationStatus` values are `ITERATING`, `CONVERGED`,
`EVALUATION_LIMIT`, `CAPACITY_PENDING`, `SCIENTIFIC_FAILURE`, `CELL_FAILURE`,
`RESOURCE_EXHAUSTED`, and `INACTIVE`. `CAPACITY_PENDING` is the only nonterminal
status that does not iterate. A group still `ITERATING` after `solve` reached the
round or growth bound; it has not converged.

```python
import numpy as np

from phydrax.atomistic import (
    AtomisticCartesianChart,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
    AtomisticRelaxationStatus,
)
from phydrax.optim import FIRE

# Prerequisites: `potential` is a prepared Hamiltonian that owns its bundle,
# `potential = program.prepare(system, neighborhood_bundle=bundle_plan)`, and
# `starts` holds (2, N, 3) unwrapped Cartesian start positions.
plan = AtomisticRelaxationPlan(
    potential,
    AtomisticCartesianChart(),
    method=FIRE(maximum_step=0.05),
    convergence=AtomisticRelaxationConvergence(
        projected_tolerance=1e-6,
        full_force_tolerance=1e-6,
        require_full=True,
        maximum_evaluations=500,
    ),
    group_count=2,
)
relaxation = plan.prepare()
relaxation, state = relaxation.solve(
    relaxation.init(positions=starts), maximum_rounds=500
)
status = np.asarray(state.status)
for group in np.flatnonzero(status == int(AtomisticRelaxationStatus.CONVERGED)):
    evaluation = relaxation.accepted_evaluation(state, int(group))
```

`accepted_evaluation(state, group)` is the full program evaluation at a group's
accepted iterate and resources, with term evidence and, for a variable cell, the
program's own stress. It uses the saved accepted positions and cell, not a fresh
nonlinear-chart realization: eager and compiled rounding can differ, while the
resource binding names the exact accepted geometry. It refuses a group without
an accepted iterate. A variable-cell enthalpy relaxation changes only the chart,
the target, and the residual policy:

```python
from phydrax.atomistic import (
    AtomisticCellChartKind,
    AtomisticDeformationChart,
    AtomisticStrainSubspace,
)
from phydrax.optim import TrustScaledLBFGS

plan = AtomisticRelaxationPlan(
    potential,  # a fully periodic 3D cell
    AtomisticDeformationChart(
        AtomisticCellChartKind.LOGARITHMIC,
        strain=AtomisticStrainSubspace.hydrostatic(),
    ),
    method=TrustScaledLBFGS(maximum_step=0.02),
    convergence=AtomisticRelaxationConvergence(
        projected_tolerance=1e-6,
        full_force_tolerance=1e-6,
        full_stress_tolerance=1e-6,
        require_full=True,
        maximum_evaluations=500,
    ),
    target_pressure=0.4,
)
```

`state.cell_vectors` is `None` for a finite system. `state.cell_covector`
(`d(E + P V)/dH` at fixed unwrapped positions), `state.stress`, and
`state.volume` are `None` unless the chart varies the cell.

### Identity, binding, and Hamiltonian transitions

Plan, resource-binding, and lineage digests are dynamic `IdentityWords`
(`state.plan_identity`, `binding_identity`, `lineage_identities`). The host
properties `plan_id`, `binding_id`, and `hamiltonian_lineage` only decode them.
Change them through the owner transactions (`grow`, `transition_hamiltonian`),
never by editing the state. `step` compares the words on device: an iterating
group of a state bound to another plan or resource binding of the same layout is
not evaluated, and terminates with `SCIENTIFIC_FAILURE` (outcome `SCIENTIFIC`)
without consuming budget. A state of an incompatible layout, such as one grown
to larger capacities, is refused structurally. Host transactions raise for a
foreign state. The lineage depth is an array shape, so each transition depth
compiles once.

`transition_hamiltonian(program, state)` continues the bank under another
prepared Hamiltonian of the same prepared system. It returns
`AtomisticHamiltonianTransitionResult(runtime, state, record)`.
`record.kind` is `SCIENTIFIC_MODEL` when the canonical `scientific_id` changed
(model, parameters, terms, or coefficients). It is `NUMERICAL_REALIZATION` when
only the prepared term realizations changed. A program of another prepared system
raises `ValueError`, and so does a pure resource rebinding, because that is
`grow`. The transition works as follows:

- Active `ITERATING`, `CONVERGED`, and `CAPACITY_PENDING` groups keep their
  accepted geometry. Their resources are initialized, and grown if needed, under
  the new program.
- Groups with an accepted iterate (`record.reprimed`) are re-evaluated without
  consuming budget. That evaluation is the first evaluation of a fresh optimizer
  (`record.optimizer_policy == "reset-history"`): no L-BFGS secant pair, FIRE
  velocity, time step, mixing, or positive-step count is reused. Their status is
  decided anew, so a converged group may iterate again.
- Pending proposals of the old Hamiltonian are discarded. Terminal and inactive
  groups keep geometry, evidence, and status.
- Evaluation and iteration totals continue across the transition, and the
  lineage gains `record.transition_id`.

If the plan refuses the program, the declared capacities cannot hold a revisable
group, or a reprimed iterate fails, `record.successful` is false with a `reason`.
The returned runtime and state are then the previous ones, unchanged. Always
continue with `result.runtime`; the old runtime terminates the transitioned
groups as unbound. New lanes join the bank with
`init(..., hamiltonian_lineage=state.hamiltonian_lineage)`.

### Stationary sensitivity and inertia

`CONVERGED` is not evidence of a regular minimum.
`sensitivity(state, group, /, *, program_tangent=None, pressure_tangent=0.0,
gauge=None, policy=None)` returns an `AtomisticRelaxationSensitivity`: the
implicit tangent of the minimum (chart `parameters`, `positions`,
`cell_vectors`), the envelope derivative `objective`, and its
`AtomisticStationaryEvidence`. `program_tangent` is a PyTree with the program's
inexact-array structure, and a nonzero `pressure_tangent` requires a variable
cell. Admission is checked in order:

1. The group status is `CONVERGED` (else `NOT_CONVERGED`).
2. The chart gradient on the free, non-gauge space is below
   `stationarity_tolerance` (else `NOT_STATIONARY`).
3. Every declared gauge direction is a numerical Hessian null direction (else
   `GAUGE_NOT_NULL`).
4. The native eigensolve of the gauge-deflated Hessian action succeeds (else
   `EIGEN_FAILED`), and its lowest curvature exceeds `curvature_tolerance` (else
   `WRONG_INERTIA`).
5. The native PCG tangent solve succeeds (else `SOLVE_FAILED`).

`AtomisticGaugePolicy(*, translations=False, rotations=False)` declares the
continuous gauge. Translations are admitted for Cartesian and deformation charts.
Rotations are admitted only for finite Cartesian charts, and symmetry charts
declare none. An undeclared zero mode is reported as `WRONG_INERTIA`, not removed
silently. `AtomisticStationaryPolicy` holds the tolerances and the eigen and
solve budgets. A refusal returns NaN values with its status; check
`result.admitted` before using any value. `atomistic_stationary_inertia(objective,
point, /, *, free, gauge_basis=None, policy=None, saddle=False, converged=True)`
applies the same evidence to a flat objective. A first-order saddle requires
exactly one negative curvature with the next one positive.

## Native reaction paths

Two nudged-elastic-band routes share `nudged_elastic_band_forces`. It implements
the improved energy-weighted tangent of Henkelman and Jónsson (J. Chem. Phys.
113, 9978, 2000), the tangential spring force, and the climbing-image force
`F - 2 (F.tau) tau` on the highest interior image. Both routes reuse the native
`FIRE`, `FIRE2`, and `TrustScaledLBFGS` owners and refuse
`FIRE(downhill_check=True)`, because the NEB force is not the gradient of a scalar.

| Route | Surface | Execution |
| --- | --- | --- |
| `phydrax.chemistry.NudgedElasticBandPlan(system, surface, ...)` | any force-capable `AbstractPreparedPotentialEnergySurface`, including external and electronic providers | host loop: `surface.evaluate` per interior image per round, one band |
| `phydrax.chemistry.reaction.AtomisticNudgedElasticBandPlan(surface, ...)` | a native `AtomisticPotentialEnergySurface(system, potential)` | bounded device loop over `G` whole bands |

The host route takes an `AtomisticSystemPlan` and runs one band with
`run(reactant, product, *, initial_images=None, correspondence=None,
branch=ReactionPathBranch.EXPLICIT, product_image_shifts=None)`. Every round
calls the provider surface on the host, and no provider call is hidden inside
compiled code. Endpoint, correspondence, and branch rules are the device rules
below. Climbing starts once `climbing_start` rounds have passed, and
`maximum_steps` bounds the rounds. Use it for external or electronic surfaces.
The device route evaluates only the native prepared Hamiltonian, so it calls no
provider, and it batches whole bands.

### Device NEB

`AtomisticNudgedElasticBandPlan(surface, /, *, image_count=7,
spring_constant=1.0, force_tolerance=1e-3, maximum_evaluations=500,
climbing_start=20, group_count=1, method=None)` (FIRE by default) refuses
holonomic topology constraints. `plan.prepare().init(endpoints, /, *,
correspondences=None, branch=ReactionPathBranch.EXPLICIT, image_shifts=None,
initial_images=None, active=None, hamiltonian_lineage=())` admits `G`
`(reactant, product)` pairs of `AtomicStructure`. It grows declared capacities on
the host until every image of every active lane fits, and evaluates both
endpoints once. It returns `(runtime, state, certificates)`; the runtime may be
rebound, so continue with it. An endpoint failure or an exhausted ladder raises
`ValueError`.

- **Endpoints and correspondence.** The reactant must match the system's particle
  IDs, elements, active mask, active masses, scale, and exact cell and periodic
  axes. The product must share the scale and the exact fixed cell. Without a
  correspondence, both endpoints keep the same particle order.
  `correspondences[g]` is a permutation that gives, for each reactant slot, its
  product atom. It must preserve elements and the active mask. Nonmobile active
  atoms must coincide at both endpoints, and `initial_images` `(G, M, N, 3)` must
  keep the endpoints and nonmobile atoms fixed.
- **Periodic branch.** Fixed-cell periodic bands use unwrapped Cartesian
  coordinates, so tangents and spring lengths keep the declared winding. The
  product is `product + image_shifts[g] @ H`, with integer `(N, r)` shifts that
  vanish on nonperiodic axes (`EXPLICIT`, the default). `SHORTEST_IMAGE` is a
  named policy: it picks the nearest product image per active atom and refuses
  when the next-best image is within `1e-6` (the `ambiguity_tolerance` of
  `resolve_reaction_path_branch`). It never replaces a declared branch. Each
  `ReactionPathBranchCertificate` records the shifts and per-atom margins. A
  finite system has no branch.
- **Band-atomic transitions.** One `step` evaluates every interior image of every
  pending band against its own accepted case-one resources. A capacity failure of
  any image leaves the whole band unconsumed (`CAPACITY_PENDING`). Any other
  image failure, or a zero or nonfinite tangent, fails the whole band
  (`SCIENTIFIC_FAILURE`) at its accepted band. Resources of all interior images
  are committed only when the band is accepted. `grow`, `solve(state, *,
  maximum_rounds, maximum_growths=8)`, binding checks, and
  `transition_hamiltonian` follow the relaxation contracts above, applied to
  whole bands. A transition also re-evaluates both endpoint energies and refuses
  the whole transition when any revisable band fails. It returns an
  `AtomisticPathHamiltonianTransitionResult`.
- **Climbing and convergence.** Climbing is active once a group has consumed
  `climbing_start` band evaluations (run totals). Convergence requires an active
  climbing image and every interior residual (largest per-atom NEB force norm)
  at most `force_tolerance`. `maximum_evaluations` bounds consumed band
  evaluations.

`state.images`, `energies`, `forces` (physical), `neb_forces`, `tangents`,
`image_residual`, `climbing_image`, and `climbing_active` describe one accepted
band of each group. Before the first accepted band they hold the initial band
with NaN energies. `runtime.path_result(state, group)` returns that iterate as
a host `ReactionPathResult`. Its `successful` is true only for `CONVERGED`, and
`provider_evaluations` counts interior image evaluations, including capacity
replays, plus the two endpoint evaluations.

```python
import numpy as np

from phydrax.atomistic import AtomisticRelaxationStatus
from phydrax.chemistry import AtomisticPotentialEnergySurface
from phydrax.chemistry.reaction import (
    AtomisticNudgedElasticBandPlan,
    ReactionPathBranch,
)
from phydrax.optim import FIRE

# Prerequisites: `system` is the PreparedAtomisticSystem of `potential`;
# `reactant` and `product` are AtomicStructure endpoints on the fixed cell;
# `shifts` is the (N, 3) integer product winding of a 3D cell.
surface = AtomisticPotentialEnergySurface(system, potential)
plan = AtomisticNudgedElasticBandPlan(
    surface,
    image_count=7,
    force_tolerance=1e-3,
    maximum_evaluations=2000,
    climbing_start=20,
    method=FIRE(maximum_step=0.05),
)
runtime, state, certificates = plan.prepare().init(
    [(reactant, product)],
    branch=ReactionPathBranch.EXPLICIT,
    image_shifts=[shifts],
)
runtime, state = runtime.solve(state, maximum_rounds=2000)
if int(np.asarray(state.status)[0]) == int(AtomisticRelaxationStatus.CONVERGED):
    path = runtime.path_result(state, 0)
    saddle = runtime.transition_state_inertia(state, 0)
    if saddle.admitted:
        transition_state = path.transition_state_positions
```

`transition_state_inertia(state, group, /, *, gauge=None, policy=None)` takes the
HVP of the climbing image on the free (mobile) non-gauge space at its accepted
resources. It admits a first-order saddle only for a converged band with an
active climbing image. Periodic systems may declare translation gauge only. A
converged band without that evidence is a converged NEB, not a qualified
transition state.

Nonclaims: variable-cell (solid-state) NEB is not provided; coordinate and cell
relaxation does not imply it. Image slots are never refilled individually.
Convergence evidence describes the declared band, branch, and Hamiltonian; it is
not a global minimum-energy-path search. The per-image building blocks,
`evaluate_cartesian_trial`, `trial_geometry`, `program_energy`,
`initialize_capacity`, and `replay_capacity`, are public in `phydrax.atomistic`.

### Relaxation and path stages in campaigns

`AtomisticRelaxationStageOperation(runtime, /, *, rounds_per_segment=1,
hamiltonian_lineage=())` and `AtomisticReactionPathStageOperation(runtime, /, *,
rounds_per_segment=1, hamiltonian_lineage=())` bind a prepared relaxation or
device NEB runtime to an `AtomisticStagePlan`. The runtime's `G` groups are the
bank's lanes, so `lane_capacity` must equal the group count. Convergence, the
evaluation budget, rollback, and in-flight capacity growth remain the owner's; the
stage `budget` applies only to fixed-length operations. A path lane is one whole
band, carried in the task packet's `path`. With `AtomisticRevisionPolicy.REPRIME`,
`PreparedAtomisticCampaign.rebind_operation(state, stage, family, operation)`
applies the owner's `transition_hamiltonian` atomically to the whole bank.
`DRAIN` finishes admitted lanes under their own Hamiltonian. Relaxation and path
stages move the configuration, so they end a walker's dynamics continuation; a
following dynamics stage initializes a new run from the relaxed geometry. Between
dynamics stages a walker carries its complete `AtomisticDynamicsContinuation`
(controllers, step, time, RNG key, images, term states, force cache, resources),
so the observation cadence does not change the trajectory. See
[staged campaigns](api/atomistic.md#staged-campaigns).

`examples/atomistic_relaxation_campaign.py` runs a heterogeneous FIRE relax →
BAOAB NVT → MTK NPT → label campaign of Lennard-Jones dimers and tetramers. Its
NPT stage declares `AtomisticContinuationPolicy.RESTART`, because changing the
integrator and ensemble is a new run rather than a continuation. The example
includes one scientific failure, capacity-limited tetramers rebucketed to the
next declared role capacity, and a throttled output sink. Its first process exits
after publishing a round but before committing it, and a fresh process resumes
from the last durable commit. It is not a SIGKILL test. Killed-rank and
exact-restart gates for stage × domain campaigns belong to the
[distributed qualification tools](guides_atomistic_distributed_execution.md#stage-x-domain-campaigns).
`benchmarks/atomistic_dynamics_campaigns.py` measures campaign preparation,
compilation, round throughput, lane utilization, refill, and publication backlog
for the recorded configuration only. Its perturbed dimer/trimer fixture has
nonzero initial interactions under the unchanged Lennard-Jones definition.
Records retain initial forces, actual relaxation evaluations and residuals, every
stage visit, terminal task outcomes and reason bits. Terminal-task throughput is
separate from successful final-label throughput; a completed campaign is not
evidence that every task succeeded.

`benchmarks/atomistic_neighbor_routes.py` preserves infinite certificate coverage
margins with explicit exceptional-scalar JSON tags. An unbounded margin is not a
scientific failure: the native certification, coverage-loss and nonfinite-geometry
flags remain separate and unchanged.

## Hybrid and specialized dynamics

Potential composition includes coefficients, typed controlled Hamiltonians, region
masks, subtractive regional replacement, force groups, and RESPA stepping. External electronic
providers have explicit conservative and differentiable capabilities. The
Born–Oppenheimer adapter is a host provider boundary, not an electronic-structure engine.

`NativeAtomisticProviderPlan(model, graph_execution, *, finite_resources=None,
periodic_resources=None, name="learned")` prepares one learned model as a
`NativeAtomisticProvider` through `prepare(system, *, conditioning=None, key=None)`.
Its resources are complete, explicit `AtomisticNeighborhoodBundlePlan` values that
cover exactly the learned term's requirements. At least one of them must be declared:

- `finite_resources` serves systems without a cell. Image sources need an explicit
  nonperiodic `frame`.
- `periodic_resources` serves periodic systems and is searched in each system's own
  cell, so it declares no frame. It requires a model with periodic execution.

A system whose kind has no declared resources is refused, and there is no dense or
implicit fallback. The provider serves energy, forces, and stress from one
`program.evaluate` pass. `evaluate` initializes fresh bundle resources for every call,
while `evaluate_state` takes and returns a caller-owned accepted bundle state. Stress
is available exactly when the system is a fully periodic 3D cell and every program
term owns cell derivatives. Dynamics never receives a neighborhood from a provider. A
learned model served for dynamics is a prepared program, such as `provider.program` or
a `LearnedGraphPotentialTerm` program prepared with an explicit bundle, passed as the
`potential` of `AtomisticDynamicsPlan`.

Ring-polymer dynamics uses a leading bead axis, mass-correct springs, centroid and
radius-of-gyration estimators, and PILE normal-mode thermostatting. The method is intended
for path-integral equilibrium and RPMD approximations; its fictitious bead dynamics is not
claimed to be exact quantum real-time dynamics. Variance-constrained semi-grand moves keep
stable site identity separate from dynamic species.

## Differentiability

Gradients are valid inside one fixed discrete execution program. Bundle routes, image
integers, source re-search decisions, capacity replays, constraint convergence branches,
Monte Carlo acceptance, and species transitions are not smoothed. Deterministic and stochastic
short-horizon pathwise derivatives are supported through checkpointed replay. No global
meaning is claimed for an arbitrary long chaotic trajectory gradient.

## Nanoflow observables and closure artifacts

`AtomisticRolloutPlan` accepts a fixed tuple of
`AbstractAtomisticObserverPlan` values. Observer state is carried inside the same
checkpointed scan and updates only after accepted dynamics steps. Final-only
trajectory retention therefore still returns complete observer summaries.

`PlanarWallFramePlan` defines exact static parallel-wall normal and tangential
coordinates. `PlanarWallProfileObserverPlan` accumulates fixed-group number, mass,
charge, tangential velocity, and peculiar-velocity temperature profiles.
`MultiOriginCorrelationObserverPlan` retains a bounded origin ring and reports MSD
and VACF tensors, origin counts, and covariance without storing a full trajectory.
`DrivenSlipFitPlan.evaluate` fits an explicitly selected bulk linear region and
extrapolates to both exact wall planes. `DiffusionTensorFitPlan.evaluate` requires an
explicit lag window, minimum independent origins, finite covariance, nonnegative
diagonal diffusion, and split-window stationarity. Both evaluations require the
exact `support_id`, `system_id`, `force_field_id`, and `rollout_id` that own their
observations. `WallForceCorrelationPlan` binds the same provenance identities at
construction and accepts only an explicitly identified exact tangential wall-force
channel. It reports Green--Kubo friction with sampling covariance; a total-system
force is not accepted as an implicit wall force. Degenerate shear, empty bins, or
nonfinite regression covariance make the corresponding fit ineligible.

Admitted fits can become immutable `AtomisticNanoflowClosureArtifact` values with
exact units, temperature/composition/confinement support, wall identities, force
field, rollout, observer, and uncertainty provenance. Artifacts do not extrapolate
or activate themselves in another model. Thermal conductivity, general stress
viscosity, dielectric profiles, contact angle, filling, and curved-wall local
frames remain unsupported rather than represented by placeholder estimators.
