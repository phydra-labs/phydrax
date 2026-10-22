# Advanced and candidate computational-chemistry API

## Qualification and support

::: phydrax.chemistry.production_chemistry_support_tuples


::: phydrax.chemistry.candidate_complete_chemistry_support_tuples

::: phydrax.chemistry.periodic_chemistry_support_tuples

::: phydrax.chemistry.periodic_chemistry_qualification_campaigns

::: phydrax.chemistry.periodic.periodic_candidate_profiles

## Periodic lifecycle

::: phydrax.chemistry.periodic.write_periodic_artifact_archive

::: phydrax.chemistry.periodic.read_periodic_artifact_archive

## General Gaussian operators

::: phydrax.operators.quantum.gaussian.GaussianShellPlan

::: phydrax.operators.quantum.gaussian.GaussianBasisPlan

::: phydrax.operators.quantum.gaussian.PreparedGaussianBasis

::: phydrax.operators.quantum.gaussian.boys_values

::: phydrax.operators.quantum.gaussian.overlap_matrix

::: phydrax.operators.quantum.gaussian.kinetic_matrix

::: phydrax.operators.quantum.gaussian.nuclear_attraction_matrix

::: phydrax.operators.quantum.gaussian.electron_repulsion_tensor

::: phydrax.operators.quantum.gaussian.range_separated_electron_repulsion_tensor

::: phydrax.operators.quantum.gaussian.PreparedDirectJK

::: phydrax.operators.quantum.gaussian.FactorizedERITensor

::: phydrax.operators.quantum.gaussian.PivotedCholeskyERIPlan

::: phydrax.operators.quantum.gaussian.DensityFittingPlan

### Integral definitions and optional cuEST

These types live at the scientific owner
`phydrax.operators.quantum.gaussian`, not in the chemistry facade.
`MolecularHartreeFockPlan(..., integrals=MolecularIntegralPlan(source))`
selects native exact dense ERIs (`source=None`, the default), native screened
direct J/K, fixed factors, or explicit cuEST density fitting. The representation
and definition identity must match every downstream energy, derivative,
response, and correlation consumer.

::: phydrax.operators.quantum.gaussian.MolecularIntegralPlan

::: phydrax.operators.quantum.gaussian.PreparedMolecularIntegrals

::: phydrax.operators.quantum.gaussian.MolecularOneElectronResult

::: phydrax.operators.quantum.gaussian.MolecularJKResult

::: phydrax.operators.quantum.gaussian.MolecularDensityFactorizationEvidence

::: phydrax.operators.quantum.gaussian.CuestIntegralPlan

::: phydrax.operators.quantum.gaussian.PreparedCuestIntegrals

::: phydrax.operators.quantum.gaussian.CuestGeometry

`CuestIntegralPlan` explicitly names the prepared spherical auxiliary basis,
metric fitting cutoff, relative-conditioning flag, QR/matrix-power algorithm,
screening threshold, variable/maximum workspace bytes, cache count, and
actual-density factorization tolerance. It is full-range native-FP64 DF, not
exact four-center ERIs. The fitting cutoff does not certify an energy error or
metric rank. Budgets cover wrapper-owned resources only.

Prepare with `source.prepare(orbital_basis)`, then call
`prepared.bind_geometry(positions_bohr, charges_fp64)` at the explicit host
boundary. The resulting binding owns `one_electron()`,
`jk(density, *, factors=None)`, and
`gradient(density, energy_weighted_density, *, factors=None)`. Factors have
shape `(rank, AO)` and must reconstruct the actual, possibly damped, density
with a certified PSD lower bound. The gradient returns
`(one_electron, two_electron, pulay, successful)`; nuclear repulsion and
stationary HF admission are owned by `MolecularHartreeFockPlan`.

`bound.refresh(*, positions=None, charges=None)` returns a newly certified
binding. `bound.restore()` recertifies same-content restored dynamic arrays
against its stored key; changed data require refresh, and changed source
definitions require preparation again. A manually replaced PyTree is not
admitted until explicitly recertified. The runtime owner
`phydrax._cuda_native.release_cuda_component("cuest")` drains outstanding
effects and clears native memos at the host boundary; a valid binding
reconstructs on next use, without a
`CuestGeometry.release()` method. See [backend lifecycle](backends.md).
Private weak guards/runtime IDs are not portable scientific identity.

cuEST admits uniform Cartesian or spherical orbital shells, spherical
auxiliary shells, matched systems/centers, and contiguous angular momenta from
zero. Prepared primitive/contraction normalization and AO scaling are preserved.
The chemistry route admits all-electron, nonperiodic integer RHF only and
supplies matched energy/analytic first nuclear forces, including Pulay and
stationary checks. It does not admit arbitrary AD, basis derivatives, analytic
Hessians, CPHF/CPKS, orbital-Hessian stability, or correlation/response by
substituting exact integrals. The bound API is host-only, not a JIT-SCF promise.

Install optional runtime dependencies with
`pip install 'phydrax[cuda13,cuest]'` and follow
[installation](../installation.md) for native-extension/runtime-wheel details.
See [the guide](../guides_computational_chemistry.md) and
[typed cookbook](../cookbook/production_computational_chemistry.md).
These are candidate contracts, not GPU-performance or release qualification.


## Embedding and QM/MM

::: phydrax.chemistry.ElectrostaticEmbeddingState

::: phydrax.chemistry.PermanentMultipoleEmbeddingState

::: phydrax.chemistry.PolarizableEmbeddingState

::: phydrax.chemistry.QuantumRegionPlan

::: phydrax.chemistry.SubtractiveQMMMSurface

::: phydrax.chemistry.ElectrostaticEmbeddingQMMMSurface

::: phydrax.chemistry.MutualPolarizableQMMMSurface

::: phydrax.chemistry.AdaptivePartitionedQMMMSurface

::: phydrax.chemistry.PeriodicMultilevelQMMMSurface

## Advanced spectroscopy and nuclei

::: phydrax.chemistry.SpectralAxis

::: phydrax.chemistry.SpectralLineShape

::: phydrax.chemistry.RamanOpticalActivityResult

::: phydrax.chemistry.PeriodicSpectroscopyResult

::: phydrax.chemistry.FranckCondonResult

::: phydrax.chemistry.VibrationalPerturbationResult

::: phydrax.chemistry.VibrationalConfigurationResult

## Periodic electronic structure

::: phydrax.discretization.ReciprocalMeshPlan

::: phydrax.chemistry.PeriodicElectronicSectorPlan

::: phydrax.operators.periodic.PeriodicBlochGauge

::: phydrax.operators.periodic.PeriodicOrbitalBasisPlan

::: phydrax.operators.periodic.PeriodicOrbitalPencilPlan

::: phydrax.chemistry.PeriodicHubbardMeanFieldPlan

::: phydrax.chemistry.NativePeriodicSCFPlan

::: phydrax.chemistry.SpinPeriodicSCFPlan

::: phydrax.chemistry.SpinPeriodicSCFEvidence

::: phydrax.chemistry.PeriodicEwaldPlan

::: phydrax.chemistry.PeriodicEwaldEvidence

::: phydrax.chemistry.GTHPseudopotentialPlan

::: phydrax.chemistry.GTHLocalEvaluation

::: phydrax.chemistry.GTHNonlocalEvaluation

::: phydrax.chemistry.GammaFFTDFPlan

::: phydrax.chemistry.GammaGDFPlan

::: phydrax.chemistry.GammaSCFEvidence

::: phydrax.chemistry.PeriodicStationaryDerivativePlan

::: phydrax.chemistry.PeriodicDerivativeLedger

::: phydrax.chemistry.ElectronicPeriodicEvaluation

::: phydrax.chemistry.CallableElectronicProvider

## Periodic lattice and many-body methods

::: phydrax.chemistry.PrimitiveSupercellImageMap

::: phydrax.chemistry.FiniteDisplacementIFC2Plan

::: phydrax.chemistry.SecondOrderForceConstants

::: phydrax.chemistry.HarmonicPhononPlan

::: phydrax.chemistry.HarmonicThermodynamicsPlan

::: phydrax.chemistry.HarmonicThermodynamicsResult

::: phydrax.chemistry.QuasiHarmonicPlan

::: phydrax.chemistry.QuasiHarmonicResult

::: phydrax.chemistry.IFC3ModeVertexPlan

::: phydrax.chemistry.ThreePhononRTAPlan

::: phydrax.chemistry.ThreePhononRTAResult

::: phydrax.chemistry.DiagonalGWPlan

::: phydrax.chemistry.GWQuasiparticleEvidence

::: phydrax.chemistry.BetheSalpeterPlan

::: phydrax.chemistry.BSEPostprocessResult

::: phydrax.chemistry.BSEPostprocessEvidence
