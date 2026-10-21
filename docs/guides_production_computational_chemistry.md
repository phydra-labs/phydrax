# Production computational-chemistry architecture

The production chemistry layer separates three states that must not be confused:

1. implemented: code and local behavioral evidence exist;
2. candidate: bounded support coordinates and leakage-controlled campaigns exist;
3. released: independent reference evidence, runtime attestation, review, and a
   released capability profile exist.

`production_chemistry_support_tuples()` lists only released bounded profiles.
`candidate_complete_chemistry_support_tuples()` lists the broader implemented
candidate surface. A candidate tuple is deliberately not returned by the
production registry.

## Identity and lifecycle

Scientific model, task, numerical policy, prepared provider, result, checkpoint,
partition, and campaign identities are content addressed. A cuEST prepared
definition binds the actual dynamic basis/policy/layout contents, not just
cached labels; changing those contents requires preparation again and invalidates
older cuEST identity/evidence. Runtime handles, object IDs, and weak-reference
guards are ephemeral staleness machinery, never portable scientific identities.
`ChemistryResultCodec` maps result families without weakening field/unit
contracts. `ChemistryRunEnvelope` and `ProductionChemistryArchivePlan` retain
payload digests, semantic kind, scientific plan identity, environment, and
lifecycle state.

A calculation request uses a concrete method plan and one typed task. Numerical
choices are stored in `ElectronicNumericalPlan`, never in the physical method
identity. Dynamic geometry, embeddings, external fields, initial guesses, and
partition epoch belong to `ElectronicEvaluationContext`. Providers fail closed
on unsupported context or task coordinates.

## Evidence carried by numerical results

The native paths retain the evidence needed to reject plausible silent failures:

- Gaussian integral normalization, spherical-transform metric residuals,
  Schwarz bounds, factorization residual bounds, and derivative closure;
- SCF energy/density/commutator/electron/spin residuals, iterations, stability,
  finite values, and free-energy entropy;
- response residuals and condition estimates for CPHF/CPKS;
- right and Lambda residuals plus restart identity for coupled cluster;
- eigenpair, orthogonality, symplectic, and biorthogonality residuals for excited
  manifolds;
- root assignment overlap, subspace singular values, and alignment unitarity;
- branching-plane rank and crossing convergence;
- unitary norm, RNG state, hop outcome, frustration, and energy residual for
  nonadiabatic trajectories;
- spectral integrated strength and finite-grid area residual;
- force-field permutation symmetry, VPT resonance lists, VSCF variance, and VCI
  residuals;
- internal-coordinate rank/condition, force and gradient gates, path endpoint
  identity, and network population conservation;
- mutual-polarization residual, force validity, partition-of-unity weights, and
  topology epoch;
- periodic SCF population/free-energy residuals, Ewald force/stress closure,
  Wilson-link unitarity, acoustic-sum-rule residual, and quasiparticle/BSE
  residuals.

### Optional cuEST evidence is definition-specific

`MolecularHartreeFockPlan` selects its integral owner through
`integrals=MolecularIntegralPlan(source)` from
`phydrax.operators.quantum.gaussian`. Omitting the source retains native exact
dense ERIs; prepared native direct J/K and fixed factorized tensors remain
explicit alternatives. cuEST is an opt-in DF source, not automatic acceleration
and not an exact-ERI label. Its auxiliary basis, Coulomb metric cutoff,
absolute/relative conditioning, QR/matrix-power choice, screening, native FP64
mode, density-factorization tolerance, and workspace/cache budgets are explicit
numerical coordinates. No measured auxiliary metric rank is exposed.

The admitted cuEST route is nonperiodic all-electron integer RHF, with uniform
orbital shell representation and spherical auxiliary shells. Prepared
normalization, native/vendor AO ordering, Cartesian component scales, and units
are part of the contract. J and K use the actual damped density. Candidate
occupied factors must reconstruct it and certify the PSD lower bound; native
actual-density factorization reports its own evidence rather than replacing D
with a newer orbital density.

Energy and contracted analytic first nuclear forces share the same DF
definition. The force result includes one-electron, two-electron, Pulay, and
nuclear terms. Success additionally gates reconstructed-density consistency,
commutator and restricted orbital-stationarity residuals. This is not arbitrary
AD, basis differentiation, analytic Hessians, CPHF/CPKS, orbital-Hessian
stability, or correlation/response computed under an exact replacement
Hamiltonian. Unsupported cross-definition requests fail closed.

`MolecularRHFProvider` connects native exact or cuEST RHF to typed energy/force
calculations and PES consumers. Use the provider's exact `numerical` plan; a
generic DF selector does not identify its recipe. The provider is serial and
host-orchestrated, with no batching, AD, Hessian, periodic, or embedding-context
claim. Low-level bound geometries are ephemeral, with dynamic reconstructible
recipes: bind explicitly, refresh changed geometry, restore same-content
restored arrays, and use the owning runtime release operation to clear native
memos. Release/eviction permits reconstruction; it does not bless stale recipes.
Workspace budgets cover wrapper-owned resources, not total process memory.

Rank-deficient DIIS is supported by the native constrained minimum-norm SVD
solve, not diagonal jitter or a disabled acceleration phase.
`SCFDIISEvidence` retains coefficients, affine-constraint residual/tolerance,
original error scale, and native status/rank/cutoff/normal-residual evidence.
Force and DIIS admission remain numerical evidence, not release evidence.

See [the computational-chemistry guide](guides_computational_chemistry.md) for
the exact bind/refresh/restore contracts and
[the complete typed recipe](cookbook/production_computational_chemistry.md).
Optional installation is `pip install 'phydrax[cuda13,cuest]'`; consult
[installation](installation.md) for the distinction between runtime wheels and
the required PhydraX native extension. This surface is candidate only: it makes
no GPU-performance, hardware-wide qualification, or release-promotion claim.


## Fixed native boundaries

The general molecular Gaussian engine is dense or factorized and explicitly
capacity bounded. It does not silently turn an ECP definition into an all-electron
calculation. Native SAD/SAP, unsupported SCF accelerators, PCM/COSMO surface
solves, arbitrary selected-CI/DMRG algorithms, correlated transition properties,
and ROA tensors require declared providers.

Native exact molecular HF supports RHF/UHF/ROHF/GHF. Molecular KS supports
RKS/UKS and the native functional compositions documented by
`DensityFunctionalPlan`; a name whose components are unavailable is rejected.
Exact-native CPHF/CPKS and nuclear Hessians use implicit stationary-density
response rather than unrolled SCF differentiation. These capabilities do not
extend the optional cuEST first-force or other nonexact integral contracts.

Native correlated work is deliberately bounded to RMP2 and determinant/CAS
spaces that fit declared capacities. The coupled-cluster contract is provider
based because a production CC implementation needs semicanonical/open-shell,
triples, Lambda, gradient, memory, and checkpoint semantics together. The
optional molecular adapter exercises this boundary for RHF/UHF/ROHF
CCSD/CCSD(T).

Excited-state amplitudes never share one misleading tensor contract. TDA, RPA,
biorthogonal, and CI representations remain distinct. Restricted adiabatic TDDFT
uses exact real/imaginary orbital Hessians of the declared KS energy. General
open-shell and correlated excited derivatives remain provider coordinates.

Tensor-product Franck-Condon quadrature, vibrational product bases, and dense
periodic Gamma solvers have explicit maximum-size gates. They raise before an
unbounded allocation. QHA minimizes the supplied discrete volume support; it
does not extrapolate an equation of state beyond it. RTA transport consumes
caller-supplied cubic vertices and does not claim a native third-order periodic
force-constant generator.

## Multiscale conservation

Fixed-region link atoms are affine maps and use exact force pullback.
Fixed-charge embedding requires embedding-site forces. Mutual polarization uses
the variational MM polarization functional plus the quantum energy response to
induced dipoles; the quantum provider must return the electric field at every MM
site. Smooth adaptive QM/MM includes the weight-gradient force term. Partition
changes produce `TopologyEpoch` identities and remain nondifferentiable as
partition decisions even though each fixed weighted evaluation is
differentiable.

Periodic multilevel surfaces are explicit signed energy/force ledgers. They
require cell vectors and never infer an electrostatic or excited-state
correction.

## Candidate qualification campaigns

`candidate_chemistry_qualification_campaigns()` returns four fixed campaigns:

- molecular ground-state/correlation;
- excited-state/vibronic;
- reaction/multiscale;
- periodic electronic/lattice.

Each has disjoint calibration and locked-evaluation independent units,
preparations, and batches. Criteria include energy/force closure, stationary
response, electron/spin count, eigensystem normalization, state continuity,
trajectory conservation, mutual polarization, topology identity, Ewald stress,
acoustic sum rules, and GW/BSE residuals.

Campaign membership is not observation evidence. Promotion still requires
content-verified references, predeclared criteria, campaign start/observation
records, build/environment identity, runtime-distribution attestation, reviewer
identity, expiration, signing, and release-index publication.

## Current non-claims

The broader candidate implementation does not claim released chemical accuracy,
basis-set completeness, complete relativistic Hamiltonians, native semilocal ECP
integrals, arbitrary-size post-HF, general multireference dynamics, a universal
ROA convention, automatic reaction atom mapping, discontinuous adaptive-QM/MM
energy conservation, production plane-wave pseudopotential convergence, or
first-principles three-phonon vertices. Those require separate support tuples and
qualification evidence.
