# Radiation transport and material coupling

Phydrax separates radiation transport, spectral material coefficients, closure assumptions, and matter exchange.

`RayTransferPlan` performs absorption-emission transfer along prescribed independent rays. It does not implement isotropic scattering; a physical scattering source requires angular coupling. `PolarizedRadiativeTransferPlan` uses an augmented matrix exponential and remains valid for singular or zero propagation matrices.

`MultigroupM1RadiationSystem` provides hyperbolic moment transport and checks realizability. Closure clipping is a numerical guard, not proof that an arbitrary discretization preserves the realizable cone.

`GRGrayM1RadiationSystem` is the separate 3+1 gray moment transport and closure
system. `GRGrayRadiationInteractionPlan` consumes a distinct
`AbstractGRGrayOpacityPlan` to construct the fluid-frame four-force. This separation
allows one transport discretization to use constant, composite, bremsstrahlung,
synchrotron, Klein--Nishina, or caller-defined coefficients without changing the
hyperbolic system.

`FixedGridGRM1SSPRK3Plan` evolves densitized moments with metric-aware geometric
sources, explicit periodic or physical boundaries, PLM realizability limiting, and
asymptotic-preserving thick-limit dissipation. `GRMultigroupM1RadiationSystem` composes
bounded frequency groups. `GRNeutrinoM1System` composes species and groups and adds
exact opposite material energy, momentum, and lepton-number exchange.

`GrayLinearRadiationDiffusionPlan` is constant-coefficient linear diffusion. It distinguishes transport extinction from absorption and treats its supplied equilibrium radiation energy as frozen during a step.

## Angular closure and polarization

`VariableEddingtonTensorClosurePlan`, `DiscreteOrdinatesRadiationPlan`, and
`MonteCarloRadiationClosurePlan` expose the pressure tensor provenance instead of
presenting it as M1. They respectively validate an external positive trace-one tensor,
retain positive directional intensities, or derive moments with effective-packet and
sampling-error evidence.

`GRPolarizedRadiationFeedbackPlan` is the dynamical counterpart to observer-ray
polarized transfer. It advances local weighted Stokes beams through a canonical
absorption/dichroism/Faraday matrix exponential and applies the exact opposite
Stokes-$I$ energy-momentum change to material. It validates the Stokes cone and
propagation-matrix structure before atomic commit.

## Spectral coefficients

`SpectralFrequencyGrid` records physical frequencies and quadrature weights. `RadiationCoefficientTable` assigns every table one role: absorption, scattering, or transport. Table interpolation uses the native rectilinear gather substrate and returns explicit support.

`radiation_means` computes a Planck absorption mean and Rosseland transport mean. Supplying one undifferentiated opacity for both roles is not supported.


### Diagnostic-photon coefficients

`PhotonEnergyGrid` and `DiagnosticPhotonCoefficientTable` are separate from the
thermal frequency/temperature opacity path. They retain ordered material IDs,
mass-coefficient role, explicit area-per-mass units, evaluated-data provenance,
bounded linear or log-log interpolation, and optional exact provenance pinning.
There is no extrapolation or material-axis reordering. A table alone is not a
transport solver; `RadiationCrossSectionLibrary` prepares three such source-pinned
photoelectric, Compton, and Rayleigh tables with material mass density for native
photon histories.

## Conservative matter exchange

`RadiationMatterExchangePlan` couples radiation energy to the full homogeneous material internal energy. It solves the local backward exchange equation while holding species mass densities fixed and enforces exact combined radiation-material energy conservation. Its light-speed contract distinguishes physical and reduced light speed explicitly.

## General-relativistic ray transfer

Black-hole imaging uses invariant transfer in
`phydrax.applications.astrophysics`, not `RayTransferPlan`.
`InvariantScalarTransferPlan` advances `I_nu / nu**3` through a fixed active prefix of
piecewise-constant incident-to-observer segments with explicit path and intensity
units. `PolarizedRayPath` first binds one exact null `GRRayResult` and metric and
recomputes its transported-basis evidence. `PolarizedInvariantTransferPlan` advances
`(I,Q,U,V)` by matrix exponentials on that typed path. Support, propagation convention,
Stokes cone, basis/ray residual, qualification, and derivative validity remain in the
result.

`ThermalSynchrotronModel` qualifies only the declared MNY96 Stokes-$I$ support and
validated $K_2$ approximation. Its active linear-polarization and Faraday forms are
independent reference-unqualified approximations, so a composite active polarized
prediction is not reference-qualified. `FastLightSnapshot.prepare_path_sampling`
requires exact chart identity with `PolarizedRayPath` and binds the sampling stencil to
active/valid segment midpoints from that path. `MonotoneSlowLightWorldtube` supplies
explicit fixed-capacity spacetime sampling. These are
not selected automatically by the transfer plan. See
[General-relativistic rays, transfer, and imaging](guides_black_hole_imaging.md) and
[Relativistic matter, plasma, and radiation](guides_relativistic_matter.md).

## Separation from scattering and Hawking radiation

Material scattering in transfer, real-frequency black-hole potential scattering, and
semiclassical Hawking occupation are different problems.
`BlackHoleScatteringPlan` closes an independently solved scalar Killing-energy flux
ledger and reports signed graybody factors. `HawkingSpectrumPlan` consumes independently
qualified graybody data, corotation slopes, and tail bounds; it does not solve
radiative transfer or radial scattering. See the
[perturbation and Hawking guide](guides_black_hole_perturbations.md).

## Governed ionizing interaction data

`RadiationCrossSectionLibrary` is the prepared macroscopic runtime view of three
canonical `DiagnosticPhotonCoefficientTable` inputs. It requires one exact material
basis and energy grid, checks each source manifest's requested rights, preserves each
table's linear or log-log interpolation policy, converts area-per-mass coefficients
and kg/m³ density to inverse-meter interaction rates, retains every
table/provenance identity, and never extrapolates.

`ChargedRadiationMaterialLibrary` separately owns stopping power, scattering power,
and bremsstrahlung rates. Repository licenses do not grant rights to external atomic
or material data.

## Diagnostic photon Monte Carlo

`VoxelRadiationGeometryPlan` supplies material lookup and exact global/voxel boundary distances. `PhotonTransportPlan` uses semantic history/event random addresses, Woodcock delta tracking, bounded Klein–Nishina and Rayleigh rejection sampling, local photoelectron/recoil KERMA, scatter-class tallies, uncertainty, truncation, and a per-history energy ledger. The profile transports photons only; it labels deposition KERMA and makes no absorbed-dose claim near interfaces or outside charged-particle equilibrium.

`AliasSpectrumPlan`, `DiagnosticXRaySourcePlan`, `PlanarXRayDetectorPlan`, and `DiagnosticXRayExperimentPlan` compose spectrum, cone source, transport, and detector response. Native deposited events can be materialized at the explicit host boundary with `photon_result_to_interaction_ledger`; virtual collisions and aggregate-only values are not fabricated as biophysical records.

## Discrete ordinates

`CertifiedSlabAngularQuadrature` proves the zeroth, first, and second Gauss–Legendre moments. `MultigroupSlabTransportProblem` binds one-dimensional cells, total and group-transfer scattering cross sections, sources, group sets, and vacuum/incident/reflecting boundaries. `DiscreteOrdinatesTransportPlan` performs directional sweeps, source iteration, optional diffusion synthetic acceleration, current/leakage calculation, response integration, and global balance evidence. The landed support is slab geometry with isotropic group transfer; arbitrary-mesh OpenSn-style sweeps are not claimed.

## Charged-particle condensed history

`ChargedParticleTransportPlan` advances electron or positron kinetic energy with boundary-limited continuous stopping, multiple-scattering deflection, bounded bremsstrahlung tallies, cutoff deposition, escape, and separate positron-annihilation rest-energy photons. Secondary photons are tallied rather than recursively transported. Every history retains its kinetic-energy ledger and fixed step capacity.

## Hybrid IMC/DDMC

`HybridIMCDDMCPlan` stores fixed-capacity packet census and material energy on one-dimensional multigroup cells. Fleck effective absorption couples packets to material; optically thick cells use DDMC leakage. Packet, material, and escaped energy close one transactional ledger. The current profile does not emit new thermal packets or feed radiation momentum into hydrodynamics.

## Spectral and polarized experiments

`CorrelatedKDistributionPlan` owns positive normalized k quadrature per band. `ScalarRadiativeExperimentPlan` evaluates prescribed rays for every band/ordinate and applies one `RadiativeSensorPlan`. `PolarizedRadiativeExperimentPlan` composes existing Stokes matrix-exponential transfer over a prescribed frequency set and checks the Stokes cone. Scene construction, scattering path tracing, canopy geometry, and atmospheric databases remain caller-owned.

## Qualification and nonclaims

`radiation_transport_candidate_profiles()` and `radiation_transport_candidate_campaigns()` keep photon, S_n, charged-history, IMC/DDMC, and spectral/polarized claims separate. `tools/radiation_transport_qualification.py` establishes synthetic numerical validity only. Authoritative cross sections, benchmark decks, experimental measurements, and independent locked comparisons require governed manifests before release. See [Radiation transport sources](radiation_transport_sources.md).
