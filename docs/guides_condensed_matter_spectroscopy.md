# Condensed-matter spectroscopy

Phydrax keeps a spectroscopy calculation in three distinct stages:

1. a forward model creates a `SpectralResponseProduct` containing unmodified line strengths or density per spectral coordinate;
2. a `PreparedSpectralInstrument` applies the existing `SpectralProfilePlan` or a normalized stationary kernel and creates a `SpectralInstrumentResult`;
3. the result exposes an ordinary `TheoryVector`, which may be passed to `LinearObservationPlan`, covariance actions, and existing likelihoods.

A raw response is not an instrument result, and neither type contains experimental data, covariance, or a likelihood. The instrument result retains its raw parent and reports pre/post area evidence and finite-window loss. Instrument convolution uses `phydrax.signal.convolve`; line broadening uses the established spectroscopy profile implementation. `FourierSpectrumPlan` is the signal-owned deterministic complex transform and makes the exponent sign, window, padding, frequency order, and Parseval residual explicit.

## Convention

The condensed-matter response convention uses physical fields proportional to `exp(-i ω t)`, a retarded transform proportional to `∫ dt exp(+i ω t)`, and positive energy loss when the probe transfers energy to the sample. Gridded responses are densities per coordinate. Integrated line strengths are tagged separately and cannot be reinterpreted as densities.

## Bounded modality profiles

- **Optical/dielectric:** `OpticalDielectricPlan` consumes a successful positive-frequency `FiniteFrequencyKuboResponse`. It projects the regular interband conductivity and forms `ε(ω) = I + i σ(ω)/(ε₀ ω)`. The zero-frequency Drude weight remains separate and is never divided by frequency or broadened into the interband response.
- **Periodic IR/Raman:** `PeriodicVibrationalSpectroscopyPlan` consumes one Γ row from `PhononDispersionResult` plus a `PeriodicSpectroscopyTensorResult`. Born effective charges and Raman tensors carry provider identity and source hashes. The result retains charge-neutrality, mass-orthonormality, Raman-symmetry, nonnegative-strength, and Stokes/anti-Stokes balance evidence.
- **ARPES:** `ARPESPlan` requires a `PhotoemissionMatrixElementResult`; missing matrix elements are an error, never an implicit value of one. The profile is the sudden approximation on a fixed k/band/energy support and preserves forbidden matrix-element zeros and spectral-moment evidence.
- **STM/STS:** `TersoffHamannPlan` requires provider vacuum LDOS. It implements the weak-tunneling, s-wave, constant-tip-DOS Tersoff–Hamann transform. The signed current is integrated from the computed nonnegative differential conductance; lock-in broadening belongs to the instrument stage.
- **EELS:** `ElectronEnergyLossPlan` computes the macroscopic longitudinal valence loss `-Im ε⁻¹` for nonzero q and converts it to a positive-transfer dynamic structure response with the declared thermal factor. This is a single-scattering profile without local-field matrices, core loss, or plural scattering.
- **Elastic scattering:** X-ray amplitudes use provider-supplied real nonresonant form factors. Neutron amplitudes use explicit provenanced coherent nuclear lengths. Both use `F(Q) = Σⱼ fⱼ(Q) exp(+i Q·rⱼ)` with explicit Debye–Waller tensors and retain Friedel/passivity evidence.
- **Dynamic structure:** `DynamicStructureFactorPlan` evaluates all transitions of a bounded exact finite-state system, coalesces equal transition energies, retains the unbroadened transition bank, and checks operator adjoints, equal-time weight, and q/−q detailed balance. Broadening is a later instrument operation.

Provider-derived inputs remain provider-derived after native contractions. Provider IDs and source hashes are part of result identity. These transforms do not claim first-principles matrix elements, universal experimental agreement, crystallographic refinement, realistic photoelectron final states, local-field/core EELS, resonant or magnetic scattering, multiple scattering, or a universal experiment language.

## Resource boundaries

Plans declare fixed k, q, state, transition, band, channel, atom, and spectral capacities before evaluation. ARPES scales with k × band × energy; elastic scattering with q × atoms; exact dynamic structure with q × states²; and stationary direct convolution with channels × grid × kernel. `benchmarks/cm_spectroscopy.py` measures instrument and scattering work separately. `tools/cm_spectroscopy_smoke.py` demonstrates the raw-response → instrument → `TheoryVector` → observation path without introducing another inference stack.
