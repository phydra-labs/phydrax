# Thermal dark-sector rates

Thermal corrections are immutable rights-qualified artifacts tied to one model,
renormalization/resummation prescription, species/channel set, units and finite support.

`ThermalKernelArtifact` carries thermal masses, widths, self-energies, screening,
spectral/rate/EOS tables, covariance, source rights, stop-gradient semantics and exact
species/rate identities. `ThermalDarkRatePlan` evaluates only inside its admitted domain.

## HTL and Landau support

`HTLPolarizationPlan` exposes longitudinal/transverse response, screening and Landau-cut
evidence. Ward/transversality, spectral support, free/vacuum limits and hierarchy bounds
are explicit. An HTL table cannot be extrapolated into strong coupling or an unsupported
scale hierarchy.

## LPM profile

`LPMIntegralPlan` solves a fixed-basis transverse integral equation through
`phydrax.linalg`. `LPMSolveResult` records residual, basis, collision kernel, rate
prefactor/unit, overlap subtraction and finite-support evidence. LPM and independent
Bethe--Heitler emissions are not added without overlap removal.

## EOS consistency

Thermal products must satisfy pressure/energy/entropy/charge derivatives and stability
under one convention. Rates and EOS using different quasiparticle/resummation schemes
cannot share a production profile.
