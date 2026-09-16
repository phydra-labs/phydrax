# Radiation transport and material coupling

Phydrax separates radiation transport, spectral material coefficients, closure assumptions, and matter exchange.

`RayTransferPlan` performs absorption-emission transfer along prescribed independent rays. It does not implement isotropic scattering; a physical scattering source requires angular coupling. `PolarizedRadiativeTransferPlan` uses an augmented matrix exponential and remains valid for singular or zero propagation matrices.

`MultigroupM1RadiationSystem` provides hyperbolic moment transport and checks realizability. Closure clipping is a numerical guard, not proof that an arbitrary discretization preserves the realizable cone.

`GRGreyM1RadiationSystem` is the separate 3+1 grey moment transport and closure
system. `GRGreyRadiationInteractionPlan` consumes a distinct
`AbstractGRGreyOpacityPlan` to construct the fluid-frame four-force. This separation
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
There is no extrapolation or material-axis reordering. These tables support the
deterministic primary CT route; they do not implement photon histories,
secondary-electron transport, scatter, or absorbed-dose transport.
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
ledger and reports signed greybody factors. `HawkingSpectrumPlan` consumes independently
qualified greybody data, corotation slopes, and tail bounds; it does not solve
radiative transfer or radial scattering. See the
[perturbation and Hawking guide](guides_black_hole_perturbations.md).
