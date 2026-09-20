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
separate ledgers. `GRMHDSSPRK3Plan` couples finite-volume HLLE/UCT material and magnetic
updates atomically, supports piecewise-constant or PLM reconstruction, and uses explicit
outflow, horizon-outflow, reflective, conducting, or prescribed traces on bounded axes.
A stage that fails recovery, realizability, boundary qualification, magnetic
compatibility, conservation, or finiteness rejects the complete material--CT candidate.

`FixedGridGRRMHDIMEXPlan` composes that material/CT update with
`FixedGridGRM1SSPRK3Plan` and `GRRMHDImplicitSourcePlan`. The IMEX-SSP2(2,2,2) stages
solve the local four-force implicitly, preserve total Eulerian energy and momentum, and
commit material, magnetic, and radiation states together. `GRRMHDDefectLedger` keeps
transport, geometry, CT, and source defects separate. Fixed-background stages can be
used directly; `GRRMHDZ4cStageAdapter` supplies the same proposal contract to dynamic
Z4c coupling.

`ValenciaRecoveryStatus`, `GRMHDRunStatus`, and `GRRMHDRunStatus` retain separate
nonfinite, geometry, stability, recovery, realizability, source, magnetic, and
conservation failures. Active-cell excision still requires a caller-owned qualified
excision geometry and flux policy; a physical bounded boundary is not treated as
periodic or zero flux.

## Resistive and force-free evolution

`ResistiveGRMHDOhmicClosure` evaluates scalar-conductivity relativistic Ohm current in
one 3+1 Eulerian frame. `FixedGridResistiveGRRMHDIMEXPlan` adds an analytic
backward-Euler conductive relaxation, conservative charge continuity, explicit electric
and material energy exchange, and entropy/charge ledgers to one accepted GRRMHD step.
Zero conductivity leaves the electric field unchanged; large conductivity approaches
the ideal electric constraint without an explicit stiff time-step restriction.

`GRForceFreeSystem` evolves contravariant $E$, $B$, and electric/magnetic GLM scalars.
`GRMHDForceFreeTransitionPlan` performs hysteretic, non-blended cell transitions at
declared magnetization thresholds. Entry projects force-free constraints; restoration
requires explicitly supplied conservative material support. A transition-energy
reservoir records and exactly balances projection/restoration field-energy changes.

## General-relativistic radiation

`GRGrayM1RadiationSystem` owns transport, closure, characteristics, and stress-energy
projection only. `GRGrayRadiationInteractionPlan` separately owns opacity-dependent
absorption, emission, scattering, and Compton exchange. Matter energy and momentum
sources are exact negatives of the radiation sources.

`FixedGridGRM1SSPRK3Plan` evolves densitized moments with periodic or explicit vacuum,
outflow, reflective, and prescribed boundaries, piecewise-constant or PLM
reconstruction, a realizability limiter, and asymptotic-preserving optically thick
dissipation. `GRMultigroupM1RadiationSystem` and
`FixedGridGRMultigroupM1SSPRK3Plan` retain one bounded frequency-group state per group.
`GRNeutrinoM1System` adds named electron-neutrino, electron-antineutrino, and
heavy-lepton species; `FixedGridGRNeutrinoM1Plan` transports their groups and commits
energy, momentum, and electron-fraction exchange with exact lepton-number ledgers.

Higher-angular alternatives are explicit rather than hidden behind M1:
`VariableEddingtonTensorClosurePlan` validates a supplied tensor,
`DiscreteOrdinatesRadiationPlan` provides positive directional intensities, and
`MonteCarloRadiationClosurePlan` reports packet effective sample size and sampling
error. `GRPolarizedRadiationFeedbackPlan` advances local Stokes beams with the native
matrix-exponential action and feeds the Stokes-$I$ energy-momentum change back to
matter exactly. Every route exposes finite, physical, qualification, and derivative
evidence.

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
flow. `FishboneMoncriefTorusPlan` supplies equilibrium Kerr data;
`GRRMHDTorusInitialDataPlan` lowers it through `IngoingKerrGridPlan` into one CT- and
radiation-consistent runtime state. The plan uses the spacelike horizon-penetrating
time coordinate $t_{\rm in}=v-r$, not null constant-$v$ hypersurfaces.
`grrmhd_fast_light_snapshot` converts an accepted state into the native
chart/scale/convention-bound imaging medium.

`ThermalBremsstrahlungGrayOpacityPlan`, `ThermalSynchrotronGrayOpacityPlan`, and
`KleinNishinaScatteringPlan` produce state-dependent interaction coefficients.
`GRPhotonNumberPlan` transports and creates/absorbs photon number separately from
energy moments. `RelativisticTwoTemperaturePlan` combines adiabatic species work,
declared dissipation partition, radiation exchange, and exact Coulomb equilibration.
`NonthermalElectronEvolutionPlan` evolves a positive bounded Lorentz-factor
distribution with explicit injection, loss, thermalization, and escape ledgers.
`PairCreationAnnihilationPlan` conserves charge, two-photon stoichiometry, and total
energy; `GyrotropicPlasmaClosurePlan` reports anisotropic stress, field-aligned heat
flux, entropy production, and firehose/mirror margins. These are fluid/kinetic
closures, not Vlasov/PIC solvers.

## Derivative and acceptance boundaries

Analytic EOS and fixed-branch flux/source kernels are differentiable on their smooth
physical interiors. Table-cell changes, EOS support edges, C2P branch selection,
atmosphere activation, wave-speed branch ties, floors, magnetization limits, CT
projection/gauge choices, force-free projection, M1 cone boundaries, runtime rejection,
and active-mask/topology changes are explicit boundaries. A runtime may succeed while a
particular sensitivity remains invalid. Always consume the most specific status and the
returned `derivative_valid` predicate.