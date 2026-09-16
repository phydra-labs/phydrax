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

The analytic zero-spin B4 control remains the independent root reference. It
evaluates
`B4(0,0;0,0) = (4 pi)^-1 integral r^4/sinh(r)^2 dr = pi^3/120`
with bounded Gauss–Legendre quadrature and explicit cutoff-tail evidence.
Nonzero-spin kernels and finite EPRL amplitudes use the separate truncation- and
quadrature-audited native route described below.

`tools/spin_quantum_geometry_qualification.py` combines all finite identity,
fixed-graph, semantic, and zero-spin controls. `benchmarks/spin_quantum_geometry.py`
records spin-network preparation and booster quadrature scaling without treating
timing as scientific evidence.

## Native nonzero-spin and finite-complex closure

`SL2CPrincipalSeriesPlan` exponentiates a finite-j boost-generator truncation
and reports generator Hermiticity, unitarity, composition, and cutoff-boundary
weight. `NativeB4BoosterPlan` contracts four native boost kernels with explicit
boundary/internal intertwiners and radial quadrature. Its cutoff and quadrature
evidence remain part of the result.

`su2_15j_symbol` contracts five oriented four-valent intertwiners.
`NativeEPRLVertexData` combines finite internal-spin support, native booster
values/errors, face convention, and SU(2) contraction into a bounded Lorentzian
vertex. Finite 2-complex tensors, face sums, cancellation evidence, SVD
coarse-graining, and large-spin Regge phase/power studies are separate
capabilities. Every result remains research-only and explicitly makes no
continuum or empirical quantum-gravity claim.
