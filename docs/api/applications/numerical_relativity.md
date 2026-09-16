# Numerical relativity

The public surface contains snapshot-bound Z4c state/equations, finite differences,
gauges, boundaries and accepted-step enforcement; fixed-grid and coupled matter
runtimes; initial data; spherical marginal-surface search; separately typed apparent,
isolated/dynamical and Hamilton-evolved offline event-horizon products; corrected
characteristic $\Psi_4$ and harmonic-exactness BMS products; fixed-epoch block-AMR
transfer/distribution/topology transactions; typed complete distributed restart from
committed shards; explicit multifidelity/learned additive corrections; and fixed-grid
Z4c production state, resolver-evidenced limits, acknowledged output receipts, verified
checkpoint receipts and exact production bindings.

## Conformal Einstein–AdS references

The AdS surface is separate from Z4c:

- `ConformalEinsteinState`, `ConformalEinsteinDerivativeData`, and
  `ConformalEinsteinSystem` evaluate the four-dimensional vacuum metric
  conformal Einstein zero quantities for one explicit mostly-plus curvature
  convention. The ledger includes the conformal Hessian, Friedrich-scalar
  gradient, Schouten curl, rescaled-Weyl divergence, scalar conformal
  constraint, Riemann decomposition, Weyl symmetries/traces, and metric inverse
  residual.
- `exact_ads_conformal_reference` supplies the constant-curvature AdS control
  with conformal factor one. This qualifies signs and algebra; it is not a
  dynamical-gravity evolution.
- `GeneralizedWaveGaugePlan` makes the source sign, initial/boundary sources,
  spacetime/radial transition, damping, and conformal scalar-curvature gauge
  explicit.
- `AdSConformalBoundaryPlan` audits a Cartesian timelike conformal boundary:
  conformal-factor zero, nonzero spacelike normal, induced Lorentzian metric,
  declared incoming-radiation data, and initial-boundary corner compatibility.
  It is not null infinity and does not reuse characteristic-scri result types.
- `ConformalAdSScalarPlan` is a fixed-background Einstein-cylinder scalar
  reference with reflecting boundaries, RK4 evolution, normal-mode and energy
  evidence. It is not nonlinear Einstein–matter evolution.
- Scalar source/response fits and holographic stress tensors require explicit
  asymptotic exponents, Fefferman–Graham coefficients, counterterms,
  normalization, divergence, and expected trace. PhydraX does not infer
  holographic renormalization.

The implementation is derived from the declared metric-conformal equations and
independent analytic controls. No source or constants are imported from the
unlicensed CFE4D research snapshot.

`tools/ads_conformal_relativity_qualification.py` retains the complete exact-AdS
zero-quantity vector, scalar normal-mode history, boundary fit, and stress
audits. `benchmarks/ads_conformal_relativity.py` separates JAX lowering and
compilation of the zero-quantity ledger from fixed-background scalar runtime.
These are classical finite references and make no quantum-gravity or dynamical
AdS/CFT claim.

Read [Numerical relativity, horizons, and radiation](../../guides_numerical_relativity.md)
before choosing a horizon product. Production and deployment claims are governed by
[Black-hole execution, qualification, and production boundaries](../../guides_black_hole_execution.md).

::: phydrax.applications.numerical_relativity
    options:
      members: true
      show_root_heading: true

## Nonlinear spherical AdS support tuple

`SphericalConformalAdSPlan` adds a distinct 3+1 spherically symmetric,
compactified Einstein–scalar evolution. Radial Hamiltonian constraints
reconstruct the metric functions at every Runge–Kutta stage, scalar stress
backreacts nonlinearly, timelike-boundary policies are explicit, and generalized
wave sources relax toward the declared boundary-compatible target. Initial
data, radial refinement, halo/core partitioning, mass balance, collapse/horizon
status, and campaign evidence are retained.

`FeffermanGrahamExtractionPlan` fits scalar source/response and the spherical
mass aspect, then constructs the declared homogeneous boundary stress tensor
with trace and conservation residuals. This is automatic only for the admitted
spherical bulk-four/boundary-three convention. It is not a general
counterterm generator.

The nonlinear runtime is the spherical Einstein–scalar reduction, not a full
3+1 metric-conformal-field-equation evolution. The pointwise
`ConformalEinsteinSystem` remains the complete metric-conformal zero-quantity
owner. Classical evolution and boundary one-point evidence make no quantum
gravity or AdS/CFT validation claim.
