# Computational chemistry

`phydrax.chemistry` uses the atomistic system, units, stable particle IDs, linear
algebra, nonlinear response, geometric propagation, lifecycle, and qualification
substrates already provided by PhydraX. It does not define a second molecule,
cell, unit, or execution model.

## Contracts before execution

A calculation has independent physical and numerical coordinates:

- `MolecularElectronicSectorPlan` fixes charge, multiplicity, electron count, and
  alpha/beta populations.
- A concrete `AbstractElectronicMethodPlan` fixes the physical approximation.
  Concrete plans cover HF, KS-DFT, MP2, coupled cluster, CI, multiconfiguration,
  ADC, and explicitly external methods.
- `ElectronicModelChemistryPlan` binds method, basis artifact, environment,
  corrections, and relativistic definitions.
- `GroundStateTaskPlan`, `CorrelationTaskPlan`, `LinearResponseTaskPlan`,
  `ExcitedManifoldTaskPlan`, `NonadiabaticCouplingTaskPlan`, and
  `BandStructureTaskPlan` describe the requested observable rather than hiding it
  in a generic property list.
- `ElectronicNumericalPlan` separately records integral representation,
  stationary solver, and derivative route.
- `ElectronicCalculationPlan` binds system, sector, model, task, and numerics.

Providers declare nested theory, geometry, observable, embedding, and execution
capabilities. Preparation fails before execution when any requested coordinate is
unsupported. There is no provider search, silent fallback, or automatic method
substitution.

`ElectronicEvaluationContext` carries positions, optional cell vectors, fixed
charges, permanent multipoles, accepted induced dipoles, electric and magnetic
fields, frequency/gauge data, an explicitly bound initial guess, time, and a
partition/topology epoch ID. Unsupported context state is rejected.

## Native molecular electronic structure

`GaussianBasisPlan` represents contracted Cartesian or real-spherical shells
keyed by stable nuclear IDs. The working representation is Cartesian; exact
fixed transforms return requested real-spherical functions. The native integral
layer provides AO values and first/second derivatives, robust Boys functions,
one-electron multipoles, nuclear attraction, general electron repulsion,
range-separated repulsion, Schwarz screening, direct J/K, density fitting,
pivoted Cholesky, and first/second geometry derivatives. ECP data are typed; a
provider is required for semilocal ECP integrals.

`MolecularHartreeFockPlan` implements RHF, UHF, ROHF, and GHF. It supports
core, zero-density, extended-Hueckel, explicit, and externally projected guesses;
integer, explicit, maximum-overlap, and Fermi-Dirac occupations; damping, DIIS,
and level shifting. SAD/SAP and unimplemented acceleration names fail closed
rather than being approximated under the wrong label.

`MolecularKohnShamPlan` implements RKS/UKS over moving atom-centered
radial-Lebedev grids with Becke partition derivatives. Native functionals include
spin LDA exchange, PW92 correlation, PBE exchange/correlation, PBE0, a declared
long-range-HF/PBE-correlation composition, and an explicitly named regularized
meta correction. Functionals requiring unimplemented components fail at plan
construction.

Both molecular mean-field routes retain energy, density, commutator,
electron-count, spin, iteration, stability, and finite-value evidence. Restricted
HF stability uses orbital-rotation Hessians. Stationary Lagrangian gradients,
implicit CPHF/CPKS electric response, and implicit nuclear Hessians do not
differentiate through SCF iterations. Continuum GB/GK is native for fixed
charges; PCM/COSMO and relativistic decoupling use exact provider/transform
boundaries.

## Correlation and excited states

`MolecularIntegralTransformationPlan` creates a partition-bound MO tensor store
from dense or factorized AO integrals. Native bounded routes provide RMP2,
spin-component scaling, regularization, determinant FCI/CASCI, and state-averaged
CASSCF orbital optimization. Coupled-cluster providers return T, Lambda,
right/left residuals, triples corrections, and restartable
`CoupledClusterCheckpoint` objects. The optional molecular provider executes RHF,
UHF, or ROHF CCSD/CCSD(T) analytic gradients when its external engine supports
them. Selected-CI, DMRG, and FCIQMC remain explicit active-space provider
boundaries with residual and discarded-weight evidence.

All excited methods return `ElectronicManifoldResult` with a representation that
matches the eigenproblem: orthonormal TDA, symplectic/biorthogonal RPA,
biorthogonal ADC/EOM, or determinant CI. `track_excited_states` performs global
root assignment followed by polar alignment inside declared degenerate
subspaces. TDA, full TDHF, restricted adiabatic TDDFT, correlated-provider, and
CAS manifolds share this result contract.

Analytic TDA/RPA eigenvalue and derivative-coupling contractions retain
energy-weighted couplings. TDA transition-dipole and oscillator-strength
derivatives use the differentiated eigenproblem. MECI/MECP workflows retain the
gradient-difference/coupling branching plane. Fewest-switches surface hopping
uses velocity Verlet nuclei, unitary electronic propagation, NAC and spin-orbit
population transfer, energy-conserving momentum rescaling, frustrated-hop
policy, decoherence policy, RNG state, and an energy ledger.

## Spectroscopy and nuclear motion

`SpectralProfilePlan` provides normalized Gaussian, Lorentzian, and exact Voigt
profiles with finite-grid area evidence. Transition energies convert to eV,
cm^-1, nm, THz, or atomic angular frequency without changing integrated line
strength.

Harmonic IR, nonresonant Raman, dynamic Kramers-Heisenberg-Dirac Raman, and
provider-bound ROA/periodic spectra keep tensors, invariants, line strengths, and
broadening evidence distinct. Multidimensional Gauss-Hermite quadrature evaluates
Duschinsky Franck-Condon amplitudes from the initial vibrational ground state;
Condon and linear Herzberg-Teller dipoles produce atomic-unit spontaneous
emission rates.

`AnharmonicForceFieldPlan` obtains quadratic, cubic, and quartic derivatives of a
differentiable normal-coordinate energy. Bounded product spaces provide VPT2,
resonance-aware GVPT2, VSCF, and VCI. Periodic Fourier-DVR hindered rotors and
Boltzmann conformer ensembles report partition/free-energy evidence.

## Reactions and embedding

`MolecularCoordinateSystemPlan` evaluates regularized redundant bonds, angles,
and periodic dihedrals, their Jacobian/rank, and trust-bounded Cartesian
retractions. Internal BFGS and eigenvector-following optimization, dimer saddle
refinement, CI-NEB, predictor-corrector IRC, transition-state/Wigner rates, and
conservative master-equation networks retain convergence and source identities.

QM/MM supports affine link atoms, subtractive ONIOM, fixed-charge embedding,
permanent multipoles, mutually self-consistent induced dipoles, smooth adaptive
partition-of-unity blending with topology epochs, and periodic multilevel energy
ledgers. A polarizable quantum provider must return the electric field and both
region and embedding-site forces; otherwise conservative dynamics is not
admitted.

## Periodic electronic and lattice methods

Periodic contracts cover task-bound external references, Ewald electrostatics,
GTH local/nonlocal definitions, bounded Gamma FFTDF LDA exchange, Gamma Gaussian
GDF HF/hybrid SCF, spin-resolved metallic k-point AO SCF, analytic forces/stress
for differentiable energies, band paths, Wilson-loop Berry phases/Wannier
centers, and defect formation-energy ledgers.

Supercell force constants retain raw and symmetry/acoustic-sum-rule-projected
tensors. Real-space force constants produce q-point phonons and optional
nonanalytic LO-TO corrections. Harmonic lattice thermodynamics, discrete-volume
QHA, three-phonon RTA transport, diagonal quasiparticle GW, and resonant/full BSE
are bounded by explicit basis, grid, quadrature, and root limits.

## Support and failure semantics

`production_chemistry_support_tuples()` remains the released, narrowly qualified
surface. `candidate_complete_chemistry_support_tuples()` and
`candidate_chemistry_qualification_campaigns()` describe the broader candidate
surface; neither is release-gate evidence. Candidate methods must not be
advertised as released until independent references, criteria, runtime
attestations, and review produce a released capability profile.

Every iterative result separates convergence from construction and finite-value
checks. Approximation labels are exact. Unsupported ECP integrals, general
selected-CI/DMRG execution, PCM/COSMO surfaces, correlated excited properties,
ROA tensors, and arbitrary periodic reference data require declared providers.
