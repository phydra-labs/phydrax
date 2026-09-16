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

PaiNN and NequIP use `LearnedGraphPotentialTerm`. Dense prediction resources now belong
to `AtomisticGraphExecutionPlan`, not model architecture identity. Periodic learned-graph
execution requires `allow_periodic=True`; that is an execution capability, not evidence
that a fitted model is stable for molecular dynamics.

## Fixed-capacity neighborhoods

Dense, cell-list, metric triclinic cell-list, and certificate-based Verlet backends retain
the existing fail-closed particle contract. A triclinic `PeriodicCell` prepares a finite
minimum-image stencil from its condition number. Short-range preparation requires a
unique-image radius. Verlet certificates include both particle displacement and cell
deformation.

No overflow truncates neighbors. Candidate, cell, pair, domain, image, potential,
constraint, thermostat, nonfinite, and stale-force failures remain separate rejection bits.
A failed step retains the last accepted state.

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

`atomistic_cell_energy_and_stress` differentiates fixed-fractional energy with respect to
homogeneous strain. It is available only when every term supports dynamic cell geometry.
The ordinary diagnostics virial remains available for fixed cells.

`ParticleMeshEwaldPotential` uses the native periodic tensor B-spline particle-grid
transfer, an FFT reciprocal solve, B-spline influence correction, real-space Ewald,
self energy, pair-exception correction, and an explicit neutral or uniform-background
policy. `EwaldReferencePotential` provides a direct reciprocal reference.

`IsotropicMonteCarloBarostatPlan` owns proposal width and realization policy; pressure,
temperature, and the volume-measure entity count come from the thermodynamic state and
phase-space measure. The move records proposal work and acceptance. Dynamic-cell moves
currently require a dense pair authority and reject learned graph terms.

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

## Hybrid and specialized dynamics

Potential composition includes coefficients, typed controlled Hamiltonians, region
masks, subtractive regional replacement, force groups, and RESPA stepping. External electronic
providers have explicit conservative and differentiable capabilities. The
Born–Oppenheimer adapter is a host provider boundary, not an electronic-structure engine.

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
`DrivenSlipFitPlan` fits an explicitly selected bulk linear region and
extrapolates to both exact wall planes. Degenerate shear, empty bins, and
nonfinite regression covariance make the fit ineligible.
`WallForceCorrelationPlan` accepts only an explicitly identified exact
tangential wall-force channel and reports Green--Kubo friction with sampling
covariance. A total-system force is not accepted as an implicit wall force.
`DiffusionTensorFitPlan` requires an explicit lag window, minimum independent
origins, finite covariance, nonnegative diagonal diffusion, and split-window
stationarity.

Admitted fits can become immutable `AtomisticNanoflowClosureArtifact` values with
exact units, temperature/composition/confinement support, wall identities, force
field, rollout, observer, and uncertainty provenance. Artifacts do not extrapolate
or activate themselves in another model. Thermal conductivity, general stress
viscosity, dielectric profiles, contact angle, filling, and curved-wall local
frames remain unsupported rather than represented by placeholder estimators.
