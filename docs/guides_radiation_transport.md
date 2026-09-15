# Radiation transport and material coupling

Phydrax separates radiation transport, spectral material coefficients, closure assumptions, and matter exchange.

`RayTransferPlan` performs absorption-emission transfer along prescribed independent rays. It does not implement isotropic scattering; a physical scattering source requires angular coupling. `PolarizedRadiativeTransferPlan` uses an augmented matrix exponential and remains valid for singular or zero propagation matrices.

`MultigroupM1RadiationSystem` provides hyperbolic moment transport and checks realizability. Closure clipping is a numerical guard, not proof that an arbitrary discretization preserves the realizable cone.

`GRGreyM1RadiationSystem` is the separate 3+1 grey moment system. It evolves local
`(E, F^i)` while its metric-aware closure accepts covariant `F_i`, uses an explicit
`ADMGridGeometry`, and constructs a fluid-frame interaction four-force whose matter
sources are exact negatives of the radiation sources. Absorption and scattering are
inverse code lengths, and reduced light speed cannot exceed the physical speed in the
bound `RelativityScaleContract`. It does not turn the nonrelativistic multigroup M1
system into multigroup GR transport.

`GrayLinearRadiationDiffusionPlan` is constant-coefficient linear diffusion. It distinguishes transport extinction from absorption and treats its supplied equilibrium radiation energy as frozen during a step.

## Spectral coefficients

`SpectralFrequencyGrid` records physical frequencies and quadrature weights. `RadiationCoefficientTable` assigns every table one role: absorption, scattering, or transport. Table interpolation uses the native rectilinear gather substrate and returns explicit support.

`radiation_means` computes a Planck absorption mean and Rosseland transport mean. Supplying one undifferentiated opacity for both roles is not supported.

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
