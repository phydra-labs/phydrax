# Conformal bootstrap

::: phydrax.applications.conformal_bootstrap

The package separates four scientific layers:

1. `ConformalDataPlan` declares four external scalar primaries, exchanged
   representation/spin sectors, crossing channels, coordinate and block
   normalization, and an optional finite `RepresentationCategory`.
2. `CrossingSectorPlan` binds caller-supplied crossing matrices to one explicit
   tensor-structure basis gauge. Preparation verifies nonsingularity and every
   declared involution. Fusion rules can reject impossible exchanged sectors,
   but PhydraX never infers missing recoupling coefficients from fusion
   multiplicities alone.
3. `GlobalScalarBlockPlan` evaluates finite scalar global blocks and derivatives
   at real Euclidean `(z, zbar)` points. Two and four dimensions use factorized
   hypergeometric series. Other dimensions use finite meromorphic radial
   recursion; higher even dimensions use a declared symmetric dimension-limit
   offset. Evidence retains truncation proxies, Casimir residuals, precision,
   method, spins, derivative orders, and evaluation points.
4. `ConformalPolynomialMatrixProgram` preserves objective, normalization,
   damped-rational prefactors, and every polynomial coefficient as exact decimal
   strings. `audit_pmp_samples` is explicitly a finite sampled PSD audit, never
   continuum positivity.

## Degenerate Virasoro references

`BPZVirasoroBlockPlan` compiles a caller-derived second-order BPZ block of the
form `z^p (1-z)^q 2F1(a,b;c;z)` on one named branch. The central charge,
external/internal weights, hypergeometric parameters, derivation source,
series order, final term, tail proxy, and elliptic nome remain explicit.
PhydraX does not infer BPZ parameters from weights.

`IsingSigmaVirasoroPlan` supplies an independent exact `c=1/2` four-spin
reference for the identity and energy channels. `ising_sigma_crossing_evidence`
retains both holomorphic blocks and verifies the exact channel-summed
`z <-> 1-z` crossing relation on the principal real branch. This qualifies a
degenerate minimal-model reference; it is not a generic Virasoro recursion,
Liouville correlator, mixed-correlator bootstrap, or continuum exclusion.

## External semidefinite optimization

`SDPBProvider` requires separately pinned `pmp2sdp` and `sdpb` executables from
the same release and license. `execute_sdpb` runs both through the bounded
host-only process runtime, preserves exact input/output artifact hashes, parses
the documented decimal solver summary, reconstructs the normalized functional,
and evaluates it independently at PMP sample points. Conversion failure, solver
failure, malformed output, numerical inconclusiveness, unavailable audit, and
failed audit remain distinct statuses. External solver success alone never
creates a conformal exclusion.

The existing `ScalarBlockPlan` and `CrossingConePlan` remain bounded 1D
SL(2,R), finite-grid references. They are not aliases for the global-block/PMP
path and cannot establish continuum bootstrap bounds.

## Approximation and claim axes

Results retain:

- spacetime dimension and external-dimension differences;
- representation/tensor basis gauge;
- spin roster and operator gaps;
- evaluation points and derivative multi-indices;
- radial/hypergeometric truncation;
- meromorphic pole and integer-dimension-limit policy;
- frontend precision;
- exact PMP identity;
- external executable hashes, solver precision, errors, and termination;
- independent sampled positivity evidence.

`tools/conformal_bootstrap_qualification.py` emits finite global, crossing,
exact-decimal PMP, sampled PSD, and exact Ising Virasoro controls with raw values
and residuals. `benchmarks/conformal_bootstrap.py` separates planning, block
preparation, JAX lowering/compilation, steady derivative evaluation, evidence,
PMP audit, and Virasoro crossing. Neither is a continuum CFT or released-bound
claim.
