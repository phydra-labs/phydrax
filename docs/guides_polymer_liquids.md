# Polymer liquids

PhydraX separates realized particle polymers from integral-equation liquid structure. Both routes live in `phydrax.applications.polymer_liquids`; neither defines chemical construction or field-theory contour topology.

## Particle reference profile

`KremerGrestProfilePlan` validates a monodisperse, fully periodic FENE–WCA bead-spring system with optional harmonic bending and BAOAB dynamics. WCA is the existing Lennard-Jones term with `shift_energy_at_cutoff=True` and cutoff `2¹ᐟ⁶σ`; bonded beads retain WCA. `kremer_grest_evidence` checks the FENE margin, potential evaluation, and the discrete BAOAB fluctuation–dissipation identity against an explicit thermodynamic-state table.

The atomistic package also provides chain conformation, contour statistics, Debye scattering, and partial structure-factor operators. Their inputs require explicit unwrapped coordinates, chain slot maps, masks, scattering weights, and normalization conventions.

`OverdampedAtomisticPlan` is the compatibility entry point for the generic
`HydrodynamicBrownianPlan` with `ConstantIsotropicMobilityPlan`. The same
transactional Itô runtime accepts free-space equal-radius RPY, positive-split
periodic spectral RPY, or a confined MAC/FIB mobility adapter. Matrix square-root
noise and centered random-finite-difference thermal drift use replay-addressed
randomness. Holonomic constraints and entangled-melt/long-range-HI combinations
remain outside the admitted matrix.

## Entanglement and reptation

`PrimitivePathSnapshotPlan` freezes one accepted open-linear molecular topology and
reconstructs its unwrapped coordinates without mutating molecular bonds. Native
force-based PPA fixes chain ends, removes intrachain excluded volume, retains an
interchain uncrossability barrier, and reports convergence, contour, contact,
endpoint, bond-length, and minimum-separation evidence. Z1+ is an explicit
versioned export/import oracle boundary; it is never a silent runtime fallback.
Coil, kink, primitive-step, plateau-modulus, multi-chain-length, block-error, and
periodic winding estimators retain their conventions and source snapshot identity.

Particle reptation observables use explicit unwrapped time origins for monomer,
center-of-mass, internal, end-to-end, and Rouse-mode statistics. Doi–Edwards and
the named single-chain Likhtman–McLeish spectrum expose their finite mode
truncations. Reversible slip-spring birth/death uses the full
Metropolis–Hastings proposal ratio. The nonlinear GLaMM route is the declared
contour-tensor variant with transactional positivity and stability gates, not an
implicit replacement for particle dynamics.

## Hydrodynamics and driven flow

`FreeSpaceRPYMobilityPlan` implements overlap-regularized equal-radius RPY.
Periodic RPY uses a finite, surface-averaged transverse Fourier oracle with a
positive unresolved self tail; the positive split exposes separately positive
wave and local operators. `ConfinedFIBMobilityPlan` composes the existing
work-adjoint MAC marker transfer with a certified positive inverse-Stokes
operator. Marker assignment routes are frozen for one prepared mobility epoch;
a route change rejects and requires explicit re-preparation. Leading normal
hard-sphere lubrication is an explicit resistance correction.
Solvent, stresslet, ideal Brownian, and extra Brownian stresses use
one tension-positive Cauchy convention and declare missing contributions.

Dynamic flow cells advance the lattice by the homogeneous velocity-gradient
exponential and reject condition, volume, trace, or matrix-function failures.
Lees–Edwards and generalized or planar Kraynik–Reinelt remaps are unimodular
transactions that preserve unwrapped Cartesian positions. Underdamped
Kremer–Grest flow uses peculiar-momentum SLLOD; overdamped HI uses affine
advection in the Itô position equation instead. Stress work is integrated in a
separate balance ledger. Steady shear, startup/cessation windows, extension,
LAOS harmonics, normal-stress differences, overshoot, and cycle dissipation have
typed analysis contracts.

## Production admission

`polymer_liquids.production` contains the exact regime support matrix,
primitive-path/tube/slip-spring adapters, composite replay checkpoints,
qualification campaigns, and bounded smoke evidence. Unsupported tuples fail
closed. In particular, the package does not qualify entangled
Kremer–Grest melts with long-range hydrodynamic interactions.

## PRISM

`IsotropicRadialTransformPlan` owns the interior DST-I convention

- `rⱼ = (j + 1)R/(N + 1)`;
- `kₙ = (n + 1)π/R`;
- no synthetic zero endpoint.

`PRISMPlan` combines that transform with `SiteMixturePlan`, a sequence or tabulated intramolecular form factor, `SitePairPotentialPlan`, and an explicit closure. Supported closures are HNC, Percus–Yevick, mean-spherical, and Martynov–Sarkisov. Hard cores are explicit masks; invalid exponential or square-root domains fail instead of being clipped.

The Ornstein–Zernike site-matrix system uses native batched dense LU solves and independent native SVD conditioning evidence. `solve_prism` retains nonlinear and per-wave-number linear evidence. `solve_prism_implicit` differentiates only a successful smooth fixed branch. `continue_prism_density` uses native transactional continuation.

Near singular OZ matrices, closure boundaries, hard-core changes, or branch switching, derivatives are not admitted. PRISM predicts homogeneous isotropic correlations, not spatial morphologies.

## Cross-scale observation

`trajectory_intramolecular_form_factor` measures per-chain Ω(k) from unwrapped particle trajectories and returns an exact-grid `TabulatedFormFactorPlan`. PRISM, Debye, partial particle structure factors, and SCFT density scattering expose explicit `TheoryVector` adapters. Product identities include coordinates, units, contrast, normalization, and source provenance; equal array shapes are not compatibility evidence.

`PolymerStressCorrelationPlan` implements finite-system Green–Kubo shear-stress integration with origin counts, split-window stationarity, block uncertainty, and a fail-closed viscosity gate. It does not claim driven-flow rheology.
