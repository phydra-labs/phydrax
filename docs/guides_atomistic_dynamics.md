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

## Potential programs

`AtomisticPotentialProgram` is an ordered additive scalar-energy program. Each term
declares its spatial requirements and execution capabilities. Preparation constructs pair
geometry, directed `GraphIR`, bonded routes, or a reciprocal grid only when a term needs it.
Forces are the negative position gradient of the total scalar energy.

Native terms include harmonic bonds and angles, periodic proper or improper torsions,
Lennard-Jones, direct Coulomb, direct Ewald, and particle-mesh Ewald. Pair exclusions and
1–4 scales are explicit sparse stable-ID exceptions. Active singular geometries fail; they
are not repaired by clipping distances.

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

## NVE and NVT

`VelocityVerletPlan` implements conservative kick–drift–kick integration with canonical
momenta. Initializers accept exactly one of velocity or momentum. `BAOABLangevinPlan`
adds exact Ornstein–Uhlenbeck momentum transitions. Randomness is addressed by the root
key, realization, accepted step, operator, and stable particle ID, so a particle
permutation permutes the same stochastic path.

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

`IsotropicMonteCarloBarostatPlan` performs explicit configurational NPT volume moves.
It records proposal work and acceptance probability. Dynamic-cell moves currently require
a dense pair authority and reject learned graph terms.

## Rollout, replay, and persistence

`AtomisticRolloutPlan` carries trajectory buffers inside the scan. It never constructs a
hidden full trajectory before applying the sample stride. Retention is final-only or a
fixed-capacity trajectory. Full, per-step, and block rematerialization use the shared
checkpointed scan and record route, image, and stochastic replay digests.

Long runs are explicit fixed segments through `run_atomistic_segments`. Runtime
checkpoints use the canonical checksummed pickle-free array archive and bind to the exact
prepared system, potential program, neighborhood, integrator, and parameter identity.
There is no repair, implicit minimization, or changed protocol on resume.

## Hybrid and specialized dynamics

Potential composition includes coefficients, region masks, subtractive regional
replacement, alchemical scaling, force groups, and RESPA stepping. External electronic
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
