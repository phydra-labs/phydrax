# Compact objects API

Stationary Kerr and extended equilibrium thermodynamics, qualified aligned-binary
remnant mass/spin fits, separated angular/radial perturbations, radially qualified
complex-frequency QNMs, real-frequency scattering, exact-state-bound semiclassical
Hawking spectra and bounded evaporation, advanced resonance/ringdown and first-order
self-force plans, Michel--Bondi and Fishbone--Moncrief initial data, ingoing-Kerr
GRRMHD torus lowering and fast-light export, state-dependent thermal opacity and photon
number, two-temperature/nonthermal/pair/gyrotropic plasma evolution, and EOS/TOV
structure models.

The Schwarzschild radial API is a fixed-work two-sided Riccati/log-amplitude solver
with declared RK4 substeps, infinity-series order, complex matching and independently
gated Chebyshev ODE residual. Its infinity coefficients come from generic
$V/f=z^2W(z)$: Regge--Wheeler has finite $W$ coefficients, while Zerilli expands its
exact rational denominator recursively. The maintained QNM qualification uses the
65-node, $r_{\rm out}=30M$, 32-substep, order-12 axial plan plus a strict independently
gated axial/polar isospectral regression documented in the guide.

Read the [geometry and stationary-horizon guide](../guides_black_hole_geometry.md),
[perturbation/QNM/scattering/Hawking guide](../guides_black_hole_perturbations.md), and
[relativistic matter guide](../guides_relativistic_matter.md) for conventions, domains,
qualification, and nonclaims.

::: phydrax.applications.compact_objects
    options:
      members: true
      show_root_heading: true
