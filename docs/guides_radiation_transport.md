# Radiation transport and material coupling

Phydrax separates radiation transport, spectral material coefficients, closure assumptions, and matter exchange.

`RayTransferPlan` performs absorption-emission transfer along prescribed independent rays. It does not implement isotropic scattering; a physical scattering source requires angular coupling. `PolarizedRadiativeTransferPlan` uses an augmented matrix exponential and remains valid for singular or zero propagation matrices.

`MultigroupM1RadiationSystem` provides hyperbolic moment transport and checks realizability. Closure clipping is a numerical guard, not proof that an arbitrary discretization preserves the realizable cone.

`GrayLinearRadiationDiffusionPlan` is constant-coefficient linear diffusion. It distinguishes transport extinction from absorption and treats its supplied equilibrium radiation energy as frozen during a step.

## Spectral coefficients

`SpectralFrequencyGrid` records physical frequencies and quadrature weights. `RadiationCoefficientTable` assigns every table one role: absorption, scattering, or transport. Table interpolation uses the native rectilinear gather substrate and returns explicit support.

`radiation_means` computes a Planck absorption mean and Rosseland transport mean. Supplying one undifferentiated opacity for both roles is not supported.

## Conservative matter exchange

`RadiationMatterExchangePlan` couples radiation energy to the full homogeneous material internal energy. It solves the local backward exchange equation while holding species mass densities fixed and enforces exact combined radiation-material energy conservation. Its light-speed contract distinguishes physical and reduced light speed explicitly.
