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

### Integral definitions and optional cuEST DF-RHF

`MolecularHartreeFockPlan(..., integrals=...)` takes the canonical
`phydrax.operators.quantum.gaussian.MolecularIntegralPlan`. Its source selects
the integral definition explicitly:

| Integral plan | Definition |
| --- | --- |
| `MolecularIntegralPlan()` | Native exact dense full-range ERIs; the default |
| `MolecularIntegralPlan(prepared_direct_jk)` | Native geometry-valid screened direct J/K |
| `MolecularIntegralPlan(factorized_tensor)` | Explicit fixed `FactorizedERITensor` in the bound AO ordering |
| `MolecularIntegralPlan(cuest_plan)` | Optional cuEST full-range density-fitting J/K |

There are no separate HF `factorized_eri` or `direct_jk` keyword arguments.
Direct and fixed-factor choices remain available through this integral owner;
they do not acquire exact moving-geometry gradients or response by substitution.
Equal AO dimensions alone do not establish scientific identity.

cuEST accelerates integrals, not an end-to-end vendor SCF implementation.
PhydraX still owns RHF iteration, damping, DIIS, convergence, orbital solves,
energy assembly, Pulay terms, stationary-state admission, and result provenance.
Install the optional dependencies with `pip install 'phydrax[cuda13,cuest]'`;
the `cuest` extra and CUDA runtime wheels do not themselves supply the PhydraX
native extension or qualify the runtime. Follow the
[installation instructions](installation.md) for wheel/native-build and runtime
requirements. Native exact execution does not require this extra.

`CuestIntegralPlan(auxiliary_basis, ...)` requires an explicitly prepared
auxiliary basis and explicit `fitting_cutoff`, `relative_conditioning`,
`fitting_algorithm` (`"qr"` or `"matrix-power"`), `screening_threshold`,
`variable_workspace_bytes`, and `maximum_workspace_bytes`. Optional
`maximum_cached_geometries` and `density_factorization_tolerance` bound caching
and actual-density PSD admission. These choices belong to the numerical
definition; a generic density-fitting label cannot replace them.

Density fitting approximates the four-center ERIs in the declared auxiliary
Coulomb metric. It is **not exact ERI evaluation**, including when screening is
zero. The metric cutoff is neither an energy-error certificate nor a measured
metric rank; cuEST does not expose that rank here. Workspace bounds cover
wrapper-owned resources, not total process, vendor-private, JAX, or JIT memory.

The admitted cuEST mean-field route is nonperiodic, unpadded, all-electron,
integer-occupation RHF in real native FP64, with `jax_enable_x64=True`.
Orbital shells must use one uniform representation (Cartesian or real
spherical); auxiliary shells must be real spherical. Both bases must belong to
the same system and actual centers, with contiguous angular momenta starting at
zero at each represented center. Basis preparation normalizes primitives and
contractions; the cuEST layout converts native AO order and Cartesian component
scales without normalizing twice. Densities, occupied factors, and
energy-weighted densities use the corresponding contravariant conversion.
Gaussian exponents use inverse bohr squared. Low-level integral and
`*_atomic_units` calls use bohr, Hartree, and Hartree/bohr; the typed calculation
provider and PES accept positions and return energies/forces in the declared
atomistic unit system.

Exchange is evaluated from the **actual density supplied to J/K**, including
the damped density, not silently from the newest orbital coefficients.
Optional rank-by-AO factors must reconstruct that density within the declared
tolerance and certify its PSD lower bound. Otherwise a native eigensolve
constructs factors; densities indefinite beyond tolerance are refused.
`MolecularJKResult.factorization` retains reconstruction residual, tolerance,
factor bytes, eigensolver-use flag, and spectral lower bound. Unmeasured
eigenvalues/negative magnitudes are NaN rather than invented evidence.

For cuEST, `analytic_gradient_atomic_units` contracts one- and two-electron
derivatives and the overlap/Pulay contribution for the **same DF definition**
as the energy, then adds nuclear repulsion. Successful forces require successful
SCF and finite integrals, reconstructed-density agreement, a small commutator,
and the restricted Fock/orbital stationary equation. Convergence alone is not
force admission. These are first analytic nuclear forces, not arbitrary
JVP/VJP, basis-parameter differentiation, analytic Hessians, or a JIT-SCF
contract. CPHF/CPKS, orbital-Hessian stability, excited response, or correlated
MO transformation must refuse an unsupported nonexact definition rather than
silently use exact dense kernels.

`MolecularRHFProvider(plan, model.model_chemistry_id)` exposes typed energy and
energy/force tasks and `ElectronicPotentialEnergySurface`; pass
`numerical=provider.numerical` to `ElectronicCalculationPlan`. This binds the
actual source definition, not a display name. It admits native exact and cuEST
RHF, not fixed-factor/direct analytic-force substitution, periodic cells,
embedding/field contexts, arbitrary initial-guess contexts, batching, AD, or
Hessian tasks. See the complete
[typed energy/force/PES recipe](cookbook/production_computational_chemistry.md).

### Rank-deficient DIIS and geometry lifecycle

DIIS minimizes the residual subject to coefficients summing to one. An
orthonormal Helmert nullspace reduces that constrained problem to native
SVD least squares with minimum-norm coefficients, including rank-deficient
histories. It does not add diagonal jitter, solve a singular bordered Gram
system, or disable DIIS to hide rank loss. `state.evidence.diis` is
`SCFDIISEvidence` when DIIS ran: it retains coefficients, affine-constraint
residual/tolerance, original error scale, and `linear_solve` status, rank,
rank cutoff, and normal residual. A failed native solve or constraint check
fails explicitly; rank deficiency alone is not failure.

For repeated low-level use, prepare once with
`prepared = cuest_plan.prepare(orbital_basis)` and bind explicitly with
`bound = prepared.bind_geometry(positions_bohr, nuclear_charges_fp64)`.
`bound.one_electron()`, `bound.jk(density, factors=...)`, and
`bound.gradient(density, energy_weighted_density, factors=...)` share that
geometry. The gradient tuple is `(one, two, pulay, successful)`; it excludes
the nuclear-repulsion derivative and is not by itself a stationary-force result.
Use the HF/provider result to enforce the stationary checks above.

`bound.refresh(positions=..., charges=...)` returns a newly certified binding;
omitted arguments retain their current values. `bound.restore()` recertifies
same-content restored dynamic arrays against the stored key and refuses changed
data with the old key. Manual PyTree replacement does not authorize execution:
use refresh for changed geometry or reprepare for a changed source definition.
Bindings are host-only ephemeral objects, never durable SCF-state identities.
Their prepared recipe, positions, charges, and key remain dynamic; private
weak-reference guards and runtime object IDs are not portable scientific IDs.
The runtime owner
`phydrax._cuda_native.release_cuda_component("cuest")` drains outstanding
effects and releases reconstructible native memos; the same valid binding
rebuilds them on next use. Release at the host orchestration boundary, not
inside a transformed numerical function. There is no
`CuestGeometry.release()` method. See
[backend resource lifecycle](api/backends.md) for the owning release interface.


`MolecularKohnShamPlan` implements RKS/UKS over moving atom-centered
radial-Lebedev grids with Becke partition derivatives. Native functionals include
spin LDA exchange, PW92 correlation, PBE exchange/correlation, PBE0, a declared
long-range-HF/PBE-correlation composition, and an explicitly named regularized
meta correction. Functionals requiring unimplemented components fail at plan
construction.

Both molecular mean-field routes retain energy, density, commutator,
electron-count, spin, iteration, stability, and finite-value evidence. With the
native exact definition, restricted HF stability uses orbital-rotation Hessians;
stationary Lagrangian gradients, implicit CPHF/CPKS electric response, and
implicit nuclear Hessians do not differentiate through SCF iterations. The
cuEST/nonexact limitations above are not overridden by those exact-native
capabilities. Continuum GB/GK is native for fixed charges; PCM/COSMO and
relativistic decoupling use exact provider/transform boundaries.

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
canonical sparse translation families, separate Hamiltonian/overlap pencils,
bounded periodic mean field, generalized eigenspectra, DOS/PDOS, velocities,
finite realizations, and mesh-bound Wilson/Zak and first-Chern evidence.
Wilson eigenphases are not localized Wannier functions.

Second- and third-order force constants retain raw and corrected tensors,
permutation/symmetry/acoustic/rotational residuals, units, source identity, and
success evidence. Harmonic lattice dynamics, discrete-volume QHA, supplied-IFC3
RTA transport, diagonal supplied-self-energy roots, and supplied-kernel BSE are
separate bounded approximations; none establishes first-principles generation
of its input.

## Atomistic free-energy protocols

Electronic and force-field providers may prepare the existing
`AtomisticForceFieldPlan`; free-energy execution then remains owned by the atomistic
phase-space, thermodynamic-state, controlled-Hamiltonian, multistate, UQ, and lifecycle
contracts. No electronic provider introduces a second molecule, cell, unit, trajectory,
or scheduler model.

Protocol value plans consume already prepared and qualified leg results. They do not
infer protonation, mapping, solvent composition, force-field parameters, partial
charges, or restraint anchors. See
[Native alchemical free energy](guides_atomistic_alchemical_free_energy.md) for exact
measure, sign, correction, charge, and support boundaries.

## Support and failure semantics

`production_chemistry_support_tuples()` and
`periodic_chemistry_support_tuples()` describe exact support coordinates, while
owner-local candidate profiles and leakage-controlled campaigns describe
unreleased prospective scope. None is release-gate evidence. The global
`ReleaseIndex` is the sole release/dependency authority, and every missing,
unreleased, stale, or mismatched exact dependency fails closed.

Every iterative result separates convergence from construction and finite-value
checks. Approximation labels are exact. Unsupported ECP integrals, general
selected-CI/DMRG execution, PCM/COSMO surfaces, correlated excited properties,
ROA tensors, and arbitrary periodic reference data require declared providers.
