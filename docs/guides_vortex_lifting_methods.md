# Vortex filaments and lifting methods

Phydrax uses one orientation-aware, fixed-capacity filament representation for finite
segments, bound vortex rings, and free wakes. Geometry and connectivity are prepared;
vertex coordinates, circulation, core size, wake age, and active masks are numeric
state. A change in connectivity or wake occupancy is an explicit state transition.

The executable `examples/vortex_lifting_surface.py` builds a lifting surface,
solves a steady system, initializes a fixed wake, and performs an accepted UVLM
step through the public API.

## Oriented filament topology

Each valid segment points from its declared start vertex to its declared end vertex.
Reversing the endpoints reverses its induced velocity for unchanged circulation. The
segment core regularizes the finite-segment Biot--Savart law; it does not alter the
stored circulation or imply a viscous diffusion model.

Prepared filament topology fixes:

- vertex and segment capacities;
- endpoint indices and segment orientation;
- valid/active masks and stable identities;
- an explicit core policy and evaluation provenance.

The evaluator accepts arbitrary targets. Geometric degeneracy, nonfinite inputs,
capacity violations, and requested self handling are reported rather than hidden by a
position-based equality test. A desingularized segment field is a numerical filament
model, not the same kernel as a 3-D Gaussian vortex particle.

## Prepared lifting surfaces

A lifting-surface preparation lowers declared panel vertices to fixed bound-ring and
collocation geometry. It retains control points, panel normals, areas, span/chord
metrics, and trailing-edge ownership. Bound quarter-chord and control-point conventions
are part of the prepared identity; a result from one convention must not be reused with
another.

Trailing-edge metadata is explicit. The implementation does not infer a Kutta edge
from array position or from the last panel index. This matters for multiple surfaces,
nonrectangular indexing, and wake ownership.

The aerodynamic assumptions are those of an incompressible, inviscid, thin lifting
surface with a prescribed incident velocity. A vortex-lattice solution does not by
itself claim viscous drag, stall, boundary-layer separation, or thickness effects.

## Steady vortex-lattice solve

The steady solve assembles the aerodynamic influence coefficient matrix from the same
finite-segment evaluator used for result evaluation. At each control point it imposes
normal-flow tangency,

\[
  n_i \mathbin{\cdot} \left(U_{\infty,i} +
  \sum_j A_{ij}\Gamma_j\right) = 0,
\]

with the prepared trailing-edge/Kutta semantics. The dense solve is routed through
Phydrax linear algebra and retains its diagnostics. A finite circulation vector is not
a successful solution unless the reported normal residual, linear-solve status, and
geometry evidence all pass.

Loads use the declared density and local incident/bound velocity through a
Kutta--Joukowski line-force model. The result exposes dimensional force/moment and
coefficient evidence. These are discrete lifting-surface loads; pressure drag and
viscous skin friction are outside the model.

## UVLM wake state and accepted steps

The unsteady vortex-lattice method owns a fixed wake pool. Every slot carries finite
geometry, circulation, age/core state, and activity. A step separates three operations:

1. solve bound circulation against the current, already accepted wake;
2. create and validate the trailing-edge circulation-change candidate;
3. accept the emission atomically and convect the accepted wake.

Shedding never occurs inside an ODE stage. It is an explicit accepted-step transaction.
The transaction reports wake-capacity status, circulation conservation, truncation,
Kutta residual, and finite-state evidence. Insufficient capacity fails closed: a
truncated inspectable candidate is not presented as an accepted physical wake.

A prescribed wake follows the supplied kinematics. A free wake adds induced velocity
from bound and wake filaments. Neither policy silently reconnects, coarsens, merges, or
remeshes wake segments.

## Differentiation boundary

With connectivity, active masks, self mappings, and the wake population fixed,
filament geometry, influence assembly, linear solves, circulation, and loads are JAX
programs. Gradients across a wake-shedding decision, pool overflow, connectivity
change, or panel-topology rebuild are not defined by that program. Differentiable
studies should hold those decisions fixed or segment the calculation at accepted state
transitions.

## Nonlinear polar-coupled closure

The advanced vortex-step closure couples induced velocity to sampled section-polar
loads through a prepared nonlinear root. Polar interpolation has an explicit endpoint
policy; values outside the sample interval are not silently extrapolated. The result
retains nonlinear convergence, circulation residual, interpolation admissibility, and
load evidence. Root sensitivities use the prepared implicit-root contract rather than
a hand-coded inverse.

This closure adds section data to the lifting calculation. Its validity is bounded by
the supplied polar data and model assumptions. It is not a Reynolds-averaged flow
solver and does not manufacture dynamic-stall or three-dimensional separation physics.

## Complete filament and lifting workflows

Shared-vertex edge/ring/sheet topology retains circulation incidence and avoids
duplicated wake segments. Velocity, gradient, vorticity, core diffusion,
midpoint/RK3 convection, and curvature/age/reconnection candidates remain
explicit. Adaptive changes commit transactionally between accepted steps.

`MultiLiftingSurfacePlan` retains component/body/frame/control-surface and
trailing-edge ownership across horseshoe, ring, lifting-line, and free/prescribed
UVLM solves. Loads remain independent Kutta–Joukowski, unsteady Bernoulli,
impulse, added-mass, moment, and Trefftz providers.

Section polars resolve angle, Reynolds number, Mach number, and flap state.
Dynamic stall and low-Mach corrections are opt-in. Rotor/blade-element and
actuator-line/surface workflows report circulation, wake ownership, thrust,
torque, power, and balance. Hierarchical FMM/P3M selection remains a prepared
field-backend decision rather than an implicit solver side effect.

## Primary method references

- Rosenhead, "The formation of vortices from a surface of discontinuity,"
  *Proceedings of the Royal Society A* 134 (1931),
  [doi:10.1098/rspa.1931.0196](https://doi.org/10.1098/rspa.1931.0196).
- Falkner, "The calculation of aerodynamic loading on surfaces of any shape,"
  *Aeronautical Research Council Reports and Memoranda* 1910 (1943),
  [catalog record](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3888).
- Albano and Rodden, "A doublet-lattice method for calculating lift distributions on
  oscillating surfaces in subsonic flows," *AIAA Journal* 7 (1969),
  [doi:10.2514/3.5086](https://doi.org/10.2514/3.5086).
- Maskew, "Program VSAERO theory document: A computer program for calculating
  nonlinear aerodynamic characteristics of arbitrary configurations," NASA CR-4023
  (1987), [NASA NTRS 19870002280](https://ntrs.nasa.gov/citations/19870002280).
