# Deformable contact

Phydrax separates collision topology, candidate discovery, contact physics,
continuous step safety, and accepted mechanics state. The deformable-contact
path is a physical barrier model; it is not penalty DEM and it does not reuse
hard-contact impulses.

## Supported contract

The initial qualified surface contract covers:

- piecewise-linear 2-D segment boundaries;
- triangular 3-D collision surfaces;
- self-contact and contact among several dynamic or static surfaces;
- fixed-topology mechanics-to-surface linear maps with exact transpose action;
- deterministic dense or host sweep-and-prune candidate discovery;
- fixed candidate capacities with fail-closed overflow;
- an area-weighted physical clamped-log barrier;
- conservative linear-trajectory inclusion CCD with minimum separation;
- optional continuous T3/T4 orientation bounds;
- conservative finite-element energies, static equilibrium, and implicit Newmark dynamics;
- lagged isotropic Coulomb friction;
- fixed-route implicit JVPs and VJPs.

The original finite-element barrier workflow retains its T3/T4 and host-solve
support boundary. The shared substrate now also exposes nonlinear participant
kinematics, per-vertex separation, certified swept-AABB CCD, cone impact,
mortar/Nitsche interfaces, hydroelastic/rough closure, multiphysics transport,
distributed ownership, and remeshing/state-transfer contracts. See
[Contact formulations](guides_contact_formulations.md),
[Multiphysics contact](guides_contact_multiphysics.md), and
[Contact differentiation](guides_contact_differentiation.md).

## Collision surfaces

`CollisionSurfacePlan` owns immutable vertex/edge/face topology, stable feature
IDs, body/material/patch labels, static/codimensional masks, exclusions, and
per-vertex minimum separation. `PreparedCollisionSurface` binds numeric rest
positions and an `AbstractLinearOperator` from mechanics state to collision
vertex displacement.

For state `u`, the collision positions and dual pullback are

```text
x_contact = x_reference + A u
f_state   = Aᵀ f_contact
```

`prepare_cell_mesh_collision_surface` extracts a compact boundary from a nodal
2-D polygonal mesh or 3-D tetrahedral mesh. High-order collision proxies can
instead bind a caller-prepared interpolation operator. Several surfaces sharing
one mechanics state compose through `PreparedCollisionScene`.

## Candidate epochs

`DenseContactSearchPlan` is the small-scene correctness authority.
`SweepAndPruneContactSearchPlan` builds deterministic static or swept AABB
candidates and packs homogeneous edge–vertex, face–vertex, and edge–edge
batches. Search output retains true counts, overflow counts, elapsed work,
resource evidence, and a complete/unusable decision.

A capacity, memory, or time limit never returns a usable partial collision set.
The physical step must roll back, prepare a larger epoch at the accepted host
boundary, and retry.

Candidate construction, sorting, compaction, and closest-feature selection are
discrete derivative boundaries. Numeric evaluation of the selected distance
branch remains differentiable.

## Contact potential

`ConvergentContactPotentialPlan` uses a physical clamped-log barrier on squared
distance. For minimum separation `d_min` and activation distance `d_hat`, its
scalar coordinate and activation threshold are

```text
s     = d² − d_min²
s_hat = (2 d_min + d_hat) d_hat
```

The barrier is exactly zero outside `d_min + d_hat` and diverges as distance
approaches `d_min`. Material-space vertex/edge measures weight the potential.
The node potential applies face/edge/vertex inclusion–exclusion corrections;
a mollified edge–edge term preserves finite-resolution wire contact.

`kappa` has pressure units. `d_hat`, minimum separation, topology, capacities,
and search tolerances are numerical/static data. A runtime stiffness override
can be differentiated on one fixed contact route.

The evaluation reports energy, surface/state forces, minimum gap, active count,
action–reaction and moment residuals, feature margin, and finite/successful
evidence.

## Continuous safety

A divergent barrier alone does not prevent a line search from crossing through
an intersection. `InclusionCCDPlan` encloses the distance over recursively
subdivided time intervals using a Hausdorff/Lipschitz motion bound. An interval
is discarded only when its lower distance bound exceeds minimum separation; an
unresolved interval is reported conservatively at its lower time endpoint.
Work-limit exhaustion produces a zero usable step.

`SimplexInversionStepPlan` writes the affine triangle/tetrahedron determinant as
a Bernstein polynomial. Positive Bernstein coefficients certify an interval;
uncertain intervals are subdivided. Contact and inversion limits combine as

```text
alpha_safe = min(1, alpha_contact, alpha_inversion)
```

Only float64 geometry/certification supports the conservative capability.

## Finite-element dynamics

The finite-element problem must be compiled from one
`phydrax.variational.Functional`. Arbitrary residuals, nonsymmetric fluxes, and
unlabelled nonconservative sources are not silently treated as potentials.
`CompiledFiniteElementProblem.potential` evaluates the same terms, density, and
quadrature whose discrete first variation is returned by `residual`.

For Newmark state `u`, the minimized functional contains the inertial quadratic,
FE stored energy, normal contact energy, and a properly position-scaled lagged
friction potential. A host Newton–CG/Steihaug loop evaluates JAX gradients and
Hessian actions, applies the continuous safety limit, and then performs Armijo
backtracking. Candidate search is refreshed at every accepted/trial geometry.

`ContactDynamicsState` wraps displacement, velocity, acceleration, optional
material history, the final candidate epoch, and friction lag state. A failed
search, CCD query, inversion bound, nonlinear solve, friction lag, or energy
check preserves the complete previous state.

## Friction

`LaggedCoulombFrictionPlan` accepts one scalar coefficient or a square body-pair
table and a velocity threshold `epsilon_v`. A lag state freezes route keys,
closest-point coefficients, tangent bases, and normal-force magnitudes for one
incremental-potential solve. The smooth velocity potential has a zero force at
zero slip and a continuous force Jacobian through the regularized stick/slip
transition.

Lag updates occur between complete nonlinear solves. Lag nonconvergence rejects
the physical step; it does not fall back to frictionless contact.

## Differentiation

Contact sensitivities differentiate the final fixed stationarity equation, not
the search or solver iterations. Differentiated quantities include selected
branch geometry, collision maps, material-space measures, bulk parameters,
reference shape, initial conditions, barrier stiffness, and friction
coefficient. Search, compaction, feature choice, CCD, line-search rate,
capacity growth, and lag refresh are stopped-gradient decisions.

A derivative is qualified only when the primal/root linear solves succeed and
gap, feature, route, CCD, and inversion margins remain positive. Crossing a
contact route invalidates the local certificate.

## Failure semantics

The contact result retains a rejection bitmask covering nonfinite data, search,
CCD, inversion, nonlinear convergence, line search, contact evidence, energy,
and friction lag. No failed category substitutes another physical model.
