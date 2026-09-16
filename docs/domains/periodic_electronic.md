# Periodic electronic structure

Periodic electronic calculations share the canonical periodic cell, reciprocal
mesh, translation-family, orbital-basis, and generalized H/S pencil contracts.
Every production path carries exact units, immutable identities, and numerical
evidence. A numerically successful result is not by itself a qualification or a
claim about an external material.

## Production coordinates

### Ewald electrostatics

`PeriodicEwaldPlan` binds a fully periodic three-dimensional `PeriodicCell` and
an `AtomisticUnitSystem`. The splitting parameter and distance tolerance are in
the unit system's length coordinate. `prepare()` constructs the finite real and
reciprocal route once. The plan has two explicit neutrality policies:

- `require-neutral` rejects a cell whose net charge exceeds the declared charge
  tolerance.
- `uniform-background` includes the homogeneous background term and records
  whether it was applied.

`PeriodicEwaldResult` reports energy, forces, stress, their units, a four-term
energy ledger, and neutrality, force-balance, stress-symmetry, and component
closure evidence. Shell truncation remains explicit in the plan identity and
must be refined by qualification cases.

### Governed GTH components

`GTHPseudopotentialPlan` requires energy and length units plus a
`PeriodicProvenanceManifest`. Local reciprocal coefficients and separable
nonlocal projector energies return typed evaluations whose evidence retains the
manifest and plan identities. Projector ranks, Hermitian couplings, finite
parameters, and component alignment are checked before evaluation.

### Periodic Hubbard SCF

`SpinPeriodicSCFPlan` consumes a cell-bound `ReciprocalMeshPlan`, a prepared
`PeriodicOrbitalPencil`, a `PeriodicElectronicSectorPlan`, and a separate
`PeriodicHubbardMeanFieldPlan`. `reference_kind="restricted"` requires zero
magnetization and spin-symmetric references; `reference_kind="collinear"`
allows fixed alpha and beta populations. Zero smearing requires integer
occupations and a cross-k insulating gap. Finite smearing solves weighted Fermi
occupations and reports total energy, free energy, entropy, and both chemical
potentials.

The result retains coefficients and density matrices in the positive-definite
S metric. Its evidence independently reports energy and density convergence,
generalized eigenpair residual, generalized commutator residual, electron and
spin counts, and free-energy closure. The named one-electron/Hubbard/ionic
ledger closes the reported total.

### Supplied-integral Gamma GDF-RHF

`GammaGDFPlan` is the production Gamma-point route only for governed supplied
one-electron matrices, a positive-definite overlap, and a
`FactorizedERITensor`. The factor source must match the supplied provenance
manifest, and its residual bound must pass the plan's admission threshold. The
SCF uses the generalized S metric and reports electron count, eigenpair,
commutator, energy, free-energy, and factorization evidence. Custom density
functionals and implicit integral generation are outside this coordinate.

### Stationary force and stress

`PeriodicStationaryDerivativePlan` differentiates an explicitly selected total
energy or free energy. Its ledger requires exactly one named
Hellmann--Feynman, Pulay, entropy, nonlocal, and ionic component, including
explicit zero terms when a contribution is absent. The result contains each
component's energy, force, and cell stress; independently differentiated totals;
and energy, force, and stress closure residuals. Caller-supplied stationary
residual evidence and central directional force/stress closures must both pass
the declared tolerances.

### External periodic providers

External periodic work uses `ElectronicCalculationPlan`, typed
`GroundStateTaskPlan` or `BandStructureTaskPlan`, and
`CallableElectronicProvider`. `ElectronicProviderCapabilities` must admit the
periodic rank, method/reference, task, and every requested property.
`make_electronic_evaluation` creates `ElectronicPeriodicEvaluation`, validates
forces, stress, bands, density matrices, and polarization, and records provider,
task, geometry, unit, work, convergence, and governed artifact identities in
the common header. There is no separate periodic provider hierarchy.

### Supplied GW and BSE postprocessing

`DiagonalGWPlan` solves bounded scalar quasiparticle equations from an explicitly
supplied diagonal self-energy callable. It retains provider, kernel-definition,
source-manifest, bracket, and root-residual evidence. It does not construct a
self-energy.

`BetheSalpeterPlan` diagonalizes a bounded provider-supplied transition kernel.
Ordered transition IDs are mandatory. Tamm--Dancoff and full resonant/coupling
results wrap the electronic manifold with eigenpair, transition-order, provider,
and source-manifest evidence. The plan does not generate screened interactions
or claim an ab initio BSE implementation.

## Candidate coordinate

`GammaFFTDFPlan` is explicitly classified
`candidate-local-gth-lda-x`. It builds a bounded Gamma real-space grid from
governed local GTH coefficients, Hartree electrostatics, and Dirac local-density
exchange. It refuses nonlocal GTH channels rather than silently dropping them,
requires a neutral valence cell, retains every GTH source manifest, and reports
the same count, spectral, commutator, energy, and free-energy evidence as the
production Gamma route. It is not full GTH-LDA, does not include LDA correlation
or nonlocal projectors, and does not imply grid, basis, or material convergence.
The dedicated benchmark and qualification tool expose this classification and
its refinement residuals rather than promoting it from numerical success.
