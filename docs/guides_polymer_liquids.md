# Polymer liquids

PhydraX separates realized particle polymers from integral-equation liquid structure. Both routes live in `phydrax.applications.polymer_liquids`; neither defines chemical construction or field-theory contour topology.

## Particle reference profile

`KremerGrestProfilePlan` validates a monodisperse, fully periodic FENE–WCA bead-spring system with optional harmonic bending and BAOAB dynamics. WCA is the existing Lennard-Jones term with `shift_energy_at_cutoff=True` and cutoff `2¹ᐟ⁶σ`; bonded beads retain WCA. `kremer_grest_evidence` checks the FENE margin, potential evaluation, and the discrete BAOAB fluctuation–dissipation identity against an explicit thermodynamic-state table.

The atomistic package also provides chain conformation, contour statistics, Debye scattering, and partial structure-factor operators. Their inputs require explicit unwrapped coordinates, chain slot maps, masks, scattering weights, and normalization conventions.

`OverdampedAtomisticPlan` is a separate constant-diagonal-mobility runtime with stable-particle addressed noise. `GeneralizedLangevinRuntimePlan` composes a discrete covariance-certified memory transition with deterministic Velocity Verlet. Neither route claims hydrodynamic interactions, configuration-dependent mobility, constrained stochastic dynamics, or automatically faithful coarse kinetics.

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
