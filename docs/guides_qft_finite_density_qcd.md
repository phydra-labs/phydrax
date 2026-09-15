# Finite-density QCD production

Finite-density products remain under `phydrax.applications.lattice_field`. Generic complex-weight and Fourier methods do not acquire a QCD production claim merely because a QCD adapter calls them.

## B/Q/S semantics

`ChemicalChargeConvention` fixes baryon, electric-charge, and strangeness chemical potentials, an energy unit, CP symmetry, and `p/T^4` normalization. `GeneralizedSusceptibilityIndex` stores exact B/Q/S derivative orders. `FiniteDensityDomain` bounds temperature, each chemical-potential-to-temperature ratio, and maximum Taylor order.

`SusceptibilityEstimate` requires an increasing temperature grid, fixed index set, joint covariance, masks, source kind, and provenance. A finite-regulator result and a continuum-extrapolated lattice result are different source kinds.

## Taylor equation of state

`prepare_taylor_eos` builds one pressure polynomial. `evaluate_taylor_eos` derives pressure, B/Q/S densities, entropy, energy density, and the susceptibility matrix from that same potential. It reports the thermodynamic identity residual and refuses points outside the declared domain.

`solve_heavy_ion_path` uses a bounded native solve for `n_S = 0` and caller-declared `n_Q = r n_B`. Failure, poor source evidence, or an invalid Jacobian produces an unqualified result.

The Taylor polynomial is a local truncation-bounded construction. It is not a convergence-radius, critical-point, or unrestricted real-density claim.

## Independent finite-density methods

`MultiChargeCanonicalPlan` performs a finite B/Q/S Fourier transform with explicit node and charge support, periodicities, volume, reconstruction residual, and conjugation residual. It does not remove Fourier aliasing, finite-volume limitations, or the sign problem.

`QCDReweightingPlan` wraps the generic phase-quenched measure with distinct reference/target theory points, determinant prescription, and chain evidence. Failed average phase, effective sample size, or denominator evidence is abstention.

Complex Langevin and thimble methods remain research-only and unqualified for a QCD equation of state.

## Phenomenological providers

`HRGSpectrum` is a pinned ideal-Boltzmann hadron-resonance-gas provider. It records spectrum release, checksum, B/Q/S charges, masses, degeneracies, and interaction prescription. Its output is phenomenological, not a lattice result.

`FiniteDensityProviderGrid` admits pinned resummed or three-dimensional-Ising critical-model pressure grids. Critical grids preserve regular and singular pressure components separately. A critical point is an input model assumption, not a prediction.

## Qualified EoS tables

`EOSGridPlan` currently materializes an explicit `mu_Q = mu_S = 0` grid from a prepared Taylor potential. `qualify_eos_table` checks finite values, the thermodynamic identity, susceptibility stability, entropy, and grid-line causality. `evaluate_eos_table` requires four valid interpolation corners and never extrapolates or fills holes.

`eos_table_metadata` refuses an unqualified table. The result is a finite-density constitutive artifact, not relativistic hydrodynamics or a transport-coefficient closure.
