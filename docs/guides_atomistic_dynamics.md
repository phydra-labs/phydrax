# Atomistic dynamics

PhydraX atomistic dynamics is a fixed-capacity, conservative particle runtime for
molecules and materials. It extends the existing atomistic learning and material-particle
substrates; it does not introduce another atom identity or graph system.

## Contracts

`AtomicStructure` remains a position-bearing data snapshot. `AtomisticSystemPlan` owns
position-independent particle identity, masses, force-field types, charges, regions,
cell policy, and molecular topology. `AtomisticDynamicsState` owns positions, momenta,
periodic image counts, neighborhood cache, force cache, stochastic state, and accepted
energy ledgers.

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

PaiNN, NequIP, and native MACE use `LearnedGraphPotentialTerm`. Graph resources belong
to `AtomisticGraphExecutionPlan`, not to model architecture identity. Its
`image_capacity` charges image-aware routes, and its optional `StreamedRelationPlan`
prepares the receiver-major schedule once per topology epoch. Periodic learned-graph
execution requires `allow_periodic=True`. That is an execution capability, not evidence
that a fitted model is stable for molecular dynamics.

## Fixed-capacity neighborhoods

No overflow truncates neighbors. Candidate, cell, pair, domain, image, potential,
constraint, thermostat, nonfinite, and stale-force failures remain separate rejection
bits. A failed step retains the last accepted state.

### Pair-once neighborhoods

Dense, cell-list, metric triclinic cell-list, and certificate-based Verlet backends
retain the existing fail-closed particle contract for `ParticlePairRelation`. Each
unordered distinct pair appears once, under its minimum image. A triclinic
`PeriodicCell` prepares a finite minimum-image stencil from its condition number.
Classical pair-once terms keep their unique-image guards. `AtomisticPotentialProgram`
requires the cell's unique-image radius to cover every classical cutoff. With a
pair-once neighborhood, `AtomisticDynamicsPlan` also requires it to cover the program
cutoff plus the Verlet skin. Runtime cells outside the unique-image certificate fail
closed. Verlet certificates include both particle displacement and cell deformation.

### Image-aware neighborhoods for learned graphs

A learned directed graph may use a cutoff larger than the unique-image radius. It then
consumes a directed `ParticleImageRelation`. For row cell `H`, every route has

`d_e = r_receiver - r_source + n_e @ H`

with an explicit integer translation `n_e`. The pair `(i == j, n == 0)` is excluded.
Every nonzero self image and every repeated image of a pair are separate routes. Route
identity is the stable `(source, receiver, n, case)` tuple, and reversal maps it to
`(receiver, source, -n)`. Nonperiodic axes of a partially periodic cell have zero
integer components.

`PeriodicCell.image_stencil` bounds the complete translation set from the lattice
inverse at `cutoff + skin`. It refuses ill-conditioned cells and stencils over
`maximum_image_count`. The image searches (`CellListParticleImageNeighborhoodPlan`,
and `DenseParticleImageNeighborhoodPlan` as a bounded reference) charge cell occupancy,
edges, receiver degree, and images separately through `ParticleImageCapacity`.

An image-aware neighborhood is admitted only for programs made purely of directed-graph
terms. A program that mixes classical and learned terms must use a pair-once
neighborhood, so every classical term's unique-image guard still applies. An image
neighborhood refuses such a program, and a pair-once neighborhood refuses a cutoff
beyond the unique-image radius. This preserves classical pair multiplicity.

`ImageVerletParticleNeighborhoodPlan` caches the image relation. Rewrapping an atom
across a periodic face updates the cached `n` by the exact image-count difference, so
displacements are unchanged and no rebuild or sort occurs. The cached relation is reused
while its `ImageCertificate` holds. The certificate charges particle displacement and
the cell deformation of every stencil coefficient, which covers images missing from the
cached relation. When it expires, or when the coverage margin of the current cell is
exhausted, the search re-enumerates. A cell outside the prepared stencil envelope is a
scientific failure, not a capacity failure.

Capacity growth is an explicit host transaction. `ParticleImageCapacityLadder`
declares a finite, strictly increasing capacity sequence.
`phx.atomistic.retry_atomistic_step_with_capacity(dynamics, state, thermodynamic,
states, ladder)` attempts one step. If the step is rejected only for image cell or
route capacity, it selects the next covering ladder entry and rebinds the accepted
state through `PreparedAtomisticDynamics.rebind_neighborhood`. Kinematics, RNG,
thermostat/barostat state, energy ledger, step index, and force cache are retained
exactly. It then retries the same physical attempt. Scientific, domain, and geometric
failures are returned unchanged and never retried. An exhausted ladder raises, so
capacity never grows without bound.

## Thermodynamic state, NVE, and NVT

`AtomisticPhaseSpaceMeasurePlan` binds the common particle, topology, mass, constraint,
cell, and unit measure. `AtomisticThermodynamicStatePlan` declares NVE/NVT/NPT
intensives and Hamiltonian controls; `PreparedThermodynamicStateTable` supplies the
numeric rows used during execution.

`VelocityVerletPlan` implements conservative kick–drift–kick integration with canonical
momenta. `BAOABLangevinPlan` owns step size and friction while the selected
thermodynamic row supplies temperature. Its Ornstein–Uhlenbeck substep is exact, but the
finite splitting is not advertised as an exact canonical sampler without separate
kernel qualification. Randomness is addressed by root key, realization, accepted step,
operator, and stable particle ID.

`HydrodynamicBrownianPlan` is the transactional fixed-capacity Itô position
runtime. It composes a prepared matrix mobility with deterministic force drift,
matrix-square-root Brownian increments, centered random-finite-difference thermal
drift, optional affine background velocity, stable replay keys, and rollback on
any failed proposal. `OverdampedAtomisticPlan` remains the constant-isotropic
mobility convenience entry point. Free-space and periodic RPY and confined
MAC/FIB routes share the same runtime; holonomic constraints remain unsupported.

`GeneralizedLangevinRuntimePlan` composes deterministic Velocity Verlet with a fixed
discrete memory transition. Its transition and noise factors must satisfy the
stationary covariance identity before preparation. Auxiliary memory, addressed noise,
candidate failure, and replay remain explicit; supplying such a kernel does not by
itself establish coarse kinetic fidelity.

`DistanceConstraintPlan` applies fixed-capacity SHAKE/RATTLE position and momentum
projections. Constraint residuals, iterations, multipliers, velocity tangency, and work
remain explicit. Instantaneous temperature uses the prepared unconstrained mobile degree
count.

## Periodic stress, PME, and pressure

`atomistic_cell_energy_and_stress`, and `program.evaluate(..., compute_stress=True)`,
differentiate the same scalar energy with respect to homogeneous strain, at fixed
fractional coordinates and fixed integer images. The convention is unchanged: column
deformation `F = I + strain` maps row lattice vectors to `H' = H @ F.T` and positions to
`origin + (r - origin) @ F.T`. `AtomisticStressConvention` names the result:
`stress = sym(dE/dstrain) / |det H|`, positive under tension (pressure
`-trace(stress) / 3`).

- `CAUCHY_TENSION_POSITIVE` applies to fully periodic cells, where `|det H|` is the
  physical volume.
- `CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME` applies to partially periodic cells. It
  requires a full invertible 3x3 cell whose nonperiodic rows declare the embedding
  thickness.

A lower-rank lattice without an embedding volume is refused rather than given an
invented thickness. Requested stress also refuses unless every term owns a cell
derivative. It never falls back to zero or to the position-moment diagnostics virial,
which remains available for fixed cells.

Learned directed graphs with `allow_periodic=True` now own that cell derivative, through
the explicit image vectors `n @ H`. Stress is therefore available for periodic PaiNN,
NequIP, and MACE programs, including stress beyond the unique-image radius. Image terms
enter the strain derivative directly. A self-image edge contributes zero net positional
force but a nonzero cell stress, so a one-atom periodic cell has zero force and a
nonzero strain response.

`ParticleMeshEwaldPotential` uses the native periodic tensor B-spline particle-grid
transfer, an FFT reciprocal solve, B-spline influence correction, real-space Ewald,
self energy, pair-exception correction, and an explicit neutral or uniform-background
policy. `EwaldReferencePotential` provides a direct reciprocal reference.

`IsotropicMonteCarloBarostatPlan` owns proposal width and realization policy; pressure,
temperature, and the volume-measure entity count come from the thermodynamic state and
phase-space measure. The move records proposal work and acceptance. Dynamic-cell moves
currently require a dense pair authority and reject learned graph terms.

Periodic learned graphs, MACE included, are admitted for fixed-cell NVE
(`VelocityVerletPlan`) and NVT (`BAOABLangevinPlan`) dynamics over an image-aware Verlet
neighborhood. No NPT or other dynamic-cell method is admitted for learned graph terms.
Fixed-cell dynamics and strain evaluation alone do not establish NPT support.

## Rollout, replay, and persistence

`AtomisticRolloutPlan` carries trajectory buffers inside the scan. It never constructs a
hidden full trajectory before applying the sample stride. Retention is final-only or a
fixed-capacity trajectory. Full, per-step, and block rematerialization use the shared
checkpointed scan and record route, image, and stochastic replay digests.

Long runs are explicit fixed segments through `run_atomistic_segments`. Runtime
checkpoints use the canonical checksummed pickle-free array archive and bind the exact
prepared system, Hamiltonian, neighborhood, integrator, thermodynamic table, and
parameter identity. There is no repair, implicit minimization, or changed protocol on
resume.

A runtime checkpoint still requires the caller to recreate the matching prepared
potential. For learned models, `write_atomistic_restart(path, plan, state, model=model)`
writes a portable restart instead. It bundles the pickle-free native model artifact with
the dynamics state, and every learned term executed by `plan` must run exactly that
model. A fresh process restores the model with `read_atomistic_restart_model(path)`,
rebuilds its dynamics over the restored model, and resumes with
`read_atomistic_restart(path, plan, template)`. An altered model, source binding,
system, integrator, thermodynamic table, scope, or graph preparation is refused before
any state is returned. Compiled caches are not part of restart correctness. See the
[periodic MACE recipe](cookbook/atomistic_dynamics.md#periodic-native-mace-dynamics)
and the [atomistic API](api/atomistic.md).

The `AtomisticGraphExecutionPlan` identity now includes `image_capacity`, `streamed`,
and `maximum_candidate_slots`. As a result, graph-execution identities, and the
prepared-program identities that embed them, differ from those prepared before this
change. Rebuild the preparation and write a new checkpoint rather than resuming across
that boundary.

## Hybrid and specialized dynamics

Potential composition includes coefficients, typed controlled Hamiltonians, region
masks, subtractive regional replacement, force groups, and RESPA stepping. External electronic
providers have explicit conservative and differentiable capabilities. The
Born–Oppenheimer adapter is a host provider boundary, not an electronic-structure engine.
`NativeAtomisticProviderPlan(model, graph_execution, finite_neighborhood=..., skin=...)`
prepares one learned model as a `NativeAtomisticProvider`. Periodic systems get an
image-aware Verlet cache, sized by `graph_execution.image_capacity`, over a cell-list
image search of radius `cutoff + skin`; finite systems get a Verlet cache over
`finite_neighborhood`. The provider serves energy, forces, and stress from one
`program.evaluate` pass. Stress is available exactly when the cell is fully periodic
3D. `provider.program` and `provider.neighborhood` drive the same model in
`AtomisticDynamicsPlan`. A periodic learned graph served through the provider requires
the image-aware neighborhood.

Ring-polymer dynamics uses a leading bead axis, mass-correct springs, centroid and
radius-of-gyration estimators, and PILE normal-mode thermostatting. The method is intended
for path-integral equilibrium and RPMD approximations; its fictitious bead dynamics is not
claimed to be exact quantum real-time dynamics. Variance-constrained semi-grand moves keep
stable site identity separate from dynamic species.

## Differentiability

Gradients are valid inside one fixed discrete execution program. Pair routes, image
integers, neighbor rebuild decisions, constraint convergence branches, Monte Carlo
acceptance, and species transitions are not smoothed. Deterministic and stochastic
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
