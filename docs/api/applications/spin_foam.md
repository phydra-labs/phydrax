# Spin-foam research references

::: phydrax.applications.spin_foam

All capabilities on this page are research-only.

`assess_su2_bf_identities` audits finite Condon–Shortley Clebsch–Gordan
orthogonality, recoupling unitarity, Wigner-6j tetrahedral symmetry, and the
pentagon/Ponzano–Regge local identity.

`EPRLVertexPlan` is semantic admission, not amplitude evaluation. It fixes the
ten lexicographically ordered boundary triangle spins, five ordered
intertwiners, coupling-tree admissibility, positive Immirzi parameter,
`delta_l` support, face/edge amplitudes, coherent-state phase, normal frame,
quadrature identity, precision, and maximum support tuples.

`ExternalSpinFoamProvider` executes only a caller-pinned process protocol. It
requires the returned plan, cutoff, and precision identities to match and
preserves exact decimal amplitude/error strings and process artifacts. No GPL
library is linked or imported.

The only native SL(2,C) numerical kernel is the independently analytic zero-spin
B4 control. It evaluates
`B4(0,0;0,0) = (4 pi)^-1 integral r^4/sinh(r)^2 dr = pi^3/120`
with bounded Gauss–Legendre quadrature and explicit cutoff-tail evidence. It
makes no nonzero-spin booster, EPRL-vertex, semiclassical, continuum, or
empirical quantum-gravity claim.

`tools/spin_quantum_geometry_qualification.py` combines all finite identity,
fixed-graph, semantic, and zero-spin controls. `benchmarks/spin_quantum_geometry.py`
records spin-network preparation and booster quadrature scaling without treating
timing as scientific evidence.
