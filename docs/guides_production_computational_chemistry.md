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

Every model, task, numerical policy, prepared provider, result, checkpoint,
partition, and campaign is content addressed. `ChemistryResultCodec` maps result
families without weakening their field/unit contracts. `ChemistryRunEnvelope`
and `ProductionChemistryArchivePlan` retain payload digests, semantic kind,
scientific plan identity, environment, and lifecycle state.

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

## Fixed native boundaries

The general molecular Gaussian engine is dense or factorized and explicitly
capacity bounded. It does not silently turn an ECP definition into an all-electron
calculation. Native SAD/SAP, unsupported SCF accelerators, PCM/COSMO surface
solves, arbitrary selected-CI/DMRG algorithms, correlated transition properties,
and ROA tensors require declared providers.

Molecular HF supports RHF/UHF/ROHF/GHF. Molecular KS supports RKS/UKS and the
native functional compositions documented by `DensityFunctionalPlan`; a name
whose components are unavailable is rejected. CPHF/CPKS and nuclear Hessians use
implicit stationary-density response rather than unrolled SCF differentiation.

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
