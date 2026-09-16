# Supersymmetric lattice field theory

::: phydrax.applications.supersymmetric_lattice

The existing `TwistedSYMPlan` and `BFSSPlan` remain finite bosonic reference
actions. `TwistedN2SYMPlan` adds one concrete production-oriented theory:
regulated two-dimensional twisted N=(2,2) U(N) SYM.

## State and fermion conventions

- Forward and reverse complexified links remain independent. They are never
  reunitarized or silently related by conjugation.
- `TwistedSYMCoordinateLayout` stores both complex link branches as explicit
  real/imaginary coordinates, making the RHMC phase space real and invertible.
- `TwistedKahlerDiracOperator` uses the ordered components
  `(eta, psi_0, psi_1, chi_01)`, a backward covariant divergence, and a forward
  covariant curl with explicit temporal fermion boundary phase.
- The unregulated Kähler–Dirac operator is antisymmetric under the declared
  coefficient ordering. `RegulatedTwistedDiracOperator` augments its target so
  its normal operator is exactly `M†M + mu² I`; it does not pretend the
  regulator is part of the Pfaffian matrix.
- A bounded Euclidean coordinate geometry and conservative covariant-difference
  norm provide the structural spectral interval used by rational
  approximations. Leaving the coordinate domain rejects the trajectory.

## RHMC

`prepare_twisted_n2_rhmc` composes the existing bosonic action, optional scalar
and U(1) regulators, the regulated normal operator, independently generated
action/refresh minimax rational approximations, one quarter-power
pseudofermion, nested force partitions, exact endpoint acceptance policy, and
checkpoint-safe RHMC runtime.

The sampled measure is explicitly regulated and phase quenched.
`sample_twisted_n2_rhmc` reports acceptance, Hamiltonian defect, divergence,
membership, force, and solve-bound evidence. `assess_twisted_n2_chain`
materializes only admitted tiny fermion matrices to retain Ward controls,
Pfaffian magnitude/phase, average phase, and phase-reweighting effective sample
size.

Complexified bosonic gauge invariance, fermion antisymmetry, force directional
derivatives, rational approximation, reversible integration, Ward identities,
Pfaffian phase, regulator removal, volume, lattice spacing, and continuum
restoration are independent evidence axes. A successful finite chain makes no
restored-supersymmetry, large-N, thermodynamic, or continuum claim.

`tools/supersymmetric_lattice_qualification.py` emits one tiny raw algebra,
RHMC, Ward, and phase record.
`benchmarks/supersymmetric_lattice.py` separates rational/workflow preparation,
dense tiny-volume algebra qualification, first trajectory, repeated trajectory,
logical bytes, and scientific residuals.

## Pfaffian, reweighting, campaigns, and fermionic BFSS closure

`ScalablePfaffianPlan` uses cubic skew elimination with pivot,
antisymmetry, and determinant-identity evidence. Phase chains retain unwrapped
phases and phase ESS; `phase_reweight_observable` uses a ratio jackknife and
abstains at inadequate overlap. The observable portfolio keeps Ward, bosonic,
Polyakov, scalar-spectrum, and autocorrelation axes separate.

Native shifted Krylov solves and the generated minimax certificates remain the
authoritative pseudofermion route. Regulator, spacing, inverse-volume, and
inverse-rank campaigns use prespecified linear/quadratic fits with abstention.
Independent chains can be assigned to explicit local devices.

`BFSSFermionPlan` adds a bounded Majorana operator, Clifford and
antisymmetry checks, a verified positive partial-fraction action, and exact
Metropolis HMC for tiny dense references. It is not the scalable production
route and does not resolve the Pfaffian sign problem.
