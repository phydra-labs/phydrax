# Relativistic matter, plasma, and radiation

Relativistic physical systems live in `phydrax.equations`; conservative stepping and
recovery live in `phydrax.solver`; compact-object initial data and closures live in
`phydrax.applications.compact_objects`. They exchange immutable ADM geometry and
stress-energy records rather than reaching into one another's runtime state.

## Scales, units, and EOS

Every relativistic system carries a `RelativityScaleContract`. Geometric $c=1$ models
and SI models are separate scale identities. Density, energy, pressure, temperature,
magnetic field, opacity, and time are never inferred from array magnitudes.

`AbstractRelativisticEOS` returns `RelativisticEOSState` with pressure, total energy
density, specific enthalpy, sound speed, temperature, composition, pressure
partials, and `RelativisticEOSDomainEvidence`. The public implementations are:

- `GammaLawEOS` for a causal gamma-law caloric EOS;
- `PiecewisePolytropicEOS` for a continuous cold piecewise polytrope;
- `HybridColdThermalEOS` for the same cold curve plus a gamma-law thermal excess; and
- `TabulatedFiniteTemperatureEOS` for bounded trilinear density--temperature--
  composition tables with host-established table evidence.

The stable EOS status taxonomy separates nonfinite values; density, thermal, and
composition support on either side; cold-constraint mismatch; instability; acausality;
and nonconvergence. Extrapolation, clipping, and a zero uncertainty are not supplied as
fallbacks. A table is qualified only when its axes, values, pressure monotonicity, heat
capacity, mechanical stability, and causality evidence pass.

## SRHD and Valencia GRHD

`RelativisticHydrodynamicsLayout` fixes primitive components
$(\rho,\epsilon,v^i)$ and conserved components $(D,S_i,\tau)$. Velocity is a
contravariant Eulerian spatial vector and momentum a covariant spatial vector.
`SRHDSystem` is the unit-Cartesian specialization. `ValenciaGRHDSystem` uses
$\sqrt\gamma(D,S_i,\tau)$, an explicit `ADMGridGeometry`, source-only
`ValenciaGeometrySource`, and returns a `StressEnergyProjection` bound to the same
`geometry_lineage_id` and exact dynamic `snapshot_token`.

`GRHDC2PPolicy` owns the fixed recovery ladder:

```text
warm log-pressure root -> bounded pressure bracket -> explicit atmosphere -> reject
```

`GRHDC2PCandidateRecord` retains all three candidate slots, nonlinear status,
iterations, normalized residual, recomposition defect, selected branch, and derivative
evidence. `AtmosphereFloorPolicy` is not an EOS branch. Its exact conservative
increment and mass/energy budgets live in `AtmosphereCorrectionLedger`; an applied
floor is never hidden in a physical source.

`FixedGridGRHDSSPRK3Plan` is an atomic fixed-grid SSPRK(3,3) finite-volume runtime with
metric-aware exterior states, Valencia-normal HLLE or Rusanov face fluxes, explicit
geometric source, C2P at every required state, accepted flux/source/floor ledgers, and
all-or-nothing commit. Finiteness, stability, recovery, admissibility, conservation,
and derivative predicates are independent.

`GRHDC2PStatus` distinguishes primary or bracket success, explicit atmosphere,
inactive lanes, nonfinite conserved input, invalid geometry, exhausted root budget,
recomposition failure, and atmosphere-budget failure. `AtmosphereFloorStatus`
distinguishes not applied, near vacuum, recovery failure, and budget exceeded.
`GRHDFiniteVolumeRunStatus` separately distinguishes invalid initial state/geometry,
C2P failure, stability-limit failure, atmosphere-budget failure, nonfinite state, and
conservation defect.

## Ideal GRMHD and constrained transport

`IdealValenciaGRMHDSystem` uses primitives
$(\rho,v^i,p,B^i)$ and densitized conserved fields
$\sqrt\gamma(D,S_i,\tau,B^i)$. It requires the canonical mostly-plus,
$K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$ convention and a geometric $c=1$ scale.
`ValenciaPrimitiveRecovery`, `ValenciaHLLEBounds`, and `ValenciaHLLEFlux` retain
root, fast-wave, magnetization, floor, physical, and derivative evidence.

`GRMHDConstrainedTransportPlan` stores face-integrated
$\sqrt\gamma B^i$ as oriented two-cochains and updates them from edge-integrated EMFs.
Its vector-potential gauge is explicit; divergence and $B=dA$ compatibility have
separate ledgers. `GRMHDSSPRK3Plan` couples first-order finite-volume HLLE/UCT material
and magnetic updates atomically. A stage that hits a primitive floor or fails recovery,
magnetization, divergence, potential compatibility, conservation, or finiteness rejects
the complete material--CT candidate. The accepted ledger is then zero.

The current CT plan admits only periodic structured axes, and the runtime requires every
cell active. A bounded grid, excision mask, or mixed active/inactive geometry is rejected
until an explicit boundary-aware UCT/excision flux exists; it is not treated as a
zero-flux or periodic boundary.

`ValenciaRecoveryStatus` distinguishes nonfinite input, geometry, bracket,
nonconvergence, superluminality, density/pressure floors, EOS validity, and
magnetization. `GRMHDRunStatus` separately distinguishes invalid initial state,
geometry, stability, recovery, high-magnetization qualification, magnetic divergence,
vector-potential compatibility, conservation, and nonfinite-state failures.

This landed route is an explicit first-order finite-volume GRMHD runtime. It does not
claim high-order reconstruction, arbitrary staggerings, resistive time integration,
kinetic plasma evolution, or an unrestricted magnetization domain.

## Resistive and force-free closures

`ResistiveGRMHDOhmicClosure` evaluates scalar-conductivity relativistic Ohm current in
one 3+1 Eulerian frame. Electric field is a spatial covector; velocity, magnetic field,
and current are vectors. Conductivity zero gives charge advection, and
$E_i=-(v\times B)_i/c$ makes the conductive current vanish at finite conductivity.
The result retains ideal residual, Lorentz factor, nonnegative entropy production,
metric inversion evidence, and validity. Hall, pressure-anisotropy, kinetic, and
implicit stiff evolution are outside this closure.

`GRForceFreeSystem` evolves contravariant $E$, $B$, and electric/magnetic GLM scalars.
Its current evaluator requires caller-supplied covariant spatial derivatives; connection
and curvilinear discretization remain with the geometry/discretization owner.
Degeneracy $E\cdot B=0$, magnetic dominance, current, projection correction, and
`derivative_valid` are reported separately. Projection is explicit, never a silent
state repair.

## General-relativistic grey M1 radiation

`GRGreyM1RadiationSystem` evolves local $(E,F^i)$ while its metric-aware closure accepts
covariant flux $F_i$. It uses the native M1 Eddington closure and distinguishes physical
from reduced light speed. Absorption and scattering coefficients are inverse code
lengths. Matter interaction is built as a fluid-frame four-force, transformed
covariantly, with matter energy and momentum sources exactly opposite the radiation
sources.

Closure clipping is a guarded evaluation of the M1 formula, not evidence that an
arbitrary transport discretization preserves $|F|\le cE$. `GRGreyM1ClosureEvaluation`
and `GRRadiationMatterExchange` retain realizability, metric, finite, conservation,
physical, qualification, and derivative evidence. This is grey M1, not multigroup GR
radiation transport, Monte Carlo transport, variable Eddington tensors, or neutrino
microphysics.

## Spherical stellar structure

`EquationOfStateTable` is a monotone pressure-to-energy-density table in geometric
$G=c=1$ units. Construction requires positive/stable causal finite tabulation.
`TovPlan` integrates the Tolman--Oppenheimer--Volkoff equations by classical RK4 on one
fixed positive radial grid and reports the first pressure-floor or trapped-surface stop,
mass, radius, compactness, validity, and status. `solve_tov_sequence` vectorizes that
plan over central pressures and marks the sampled positive finite-difference
$dM/dp_c$ branch. It is a fixed-grid spherical structure model, not a tidal,
rotating-star, merger, or EOS inference solver.

## Compact-object fluid and plasma initial data

`MichelBondiAccretionPlan` solves the two relativistic conserved integrals for transonic
Michel--Bondi flow on Schwarzschild and can lower the solution to Valencia primitives.
`FishboneMoncriefTorusPlan` constructs a constant-angular-momentum torus in Kerr
Boyer--Lindquist data with explicit magnetization via a vector potential. These are
initial-data plans with domain and status evidence, not time-evolution solvers.

`TwoTemperatureElectronIonClosure` applies exact caloric electron--ion relaxation with
an antisymmetric energy ledger. `BoundedNonthermalParticleDistribution` carries a
fixed-bin isotropic population on bounded Lorentz-factor support and reports number,
kinetic-energy, and pressure moments. Neither plan is a Vlasov/PIC solver or a universal
collision/radiation closure.

## Derivative and acceptance boundaries

Analytic EOS and fixed-branch flux/source kernels are differentiable on their smooth
physical interiors. Table-cell changes, EOS support edges, C2P branch selection,
atmosphere activation, wave-speed branch ties, floors, magnetization limits, CT
projection/gauge choices, force-free projection, M1 cone boundaries, runtime rejection,
and active-mask/topology changes are explicit boundaries. A runtime may succeed while a
particular sensitivity remains invalid. Always consume the most specific status and the
returned `derivative_valid` predicate.