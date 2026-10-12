# Deformable contact

Phydrax separates collision topology, candidate discovery, contact physics,
continuous step safety, and accepted mechanics state. The deformable-contact
path is a physical barrier model; it is not penalty DEM and it does not reuse
hard-contact impulses.

## Supported contract

The surface contract includes:

- piecewise-linear 2-D segment boundaries;
- triangular 3-D collision surfaces;
- self-contact and contact among several dynamic or static surfaces;
- fixed-topology mechanics-to-surface linear maps with exact transpose action;
- deterministic dense or host sweep-and-prune candidate discovery;
- fixed candidate capacities with fail-closed overflow;
- an area-weighted physical clamped-log barrier;
- conservative linear-trajectory inclusion CCD with minimum separation;
- optional continuous T3/T4 orientation bounds;
- mechanics residual/JVP participants, static equilibrium, and implicit dynamics;
- lagged isotropic Coulomb friction;
- fixed-route implicit JVPs and VJPs.

The [monolithic implicit contact guide](guides_monolithic_contact.md) describes
shared-unknown mechanics, affine bodies, conservative envelope preparation, and
atomic acceptance. The shared substrate also exposes nonlinear participant
kinematics, per-vertex separation, certified swept-AABB CCD, cone impact,
mortar/Nitsche interfaces, hydroelastic/rough closure, multiphysics transport,
distributed ownership, and remeshing/state-transfer contracts. These are distinct
formulations, not automatic extensions of one qualification profile. See
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
instead bind a caller-prepared interpolation operator. Independent geometry maps
compose through `ContactParticipantScene`, with explicit mechanics-block or fixed
wall ownership in `MonolithicContactPlan`.

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

## Monolithic mechanics

`prepare_finite_element_contact_participant` binds a compiled finite-element
problem to the mechanics protocol. A conservative `phydrax.variational.Functional`
can supply a consistent potential, but arbitrary residuals and nonsymmetric
material tangents remain physical residual/JVP problems.

The ordered coordinate/history tuple is solved by native `NewtonKrylov` and
GMRES. Contact pullback includes mixed participant blocks and geometry curvature;
no symmetric projection or independent participant solve replaces them.
`ContactSolvePolicy.merit` is `"automatic"`, `"potential"`, or `"residual"`.
Potential merit requires verified consistency of every active contribution.

Candidate discovery occurs only at the explicit preparation boundary. The
candidate epoch remains frozen through Newton and friction refresh, and a
conservative motion envelope plus participant step limits protects route
completeness. `MonolithicContactState` stores accepted coordinates and histories.
A failed search, CCD query, domain bound, root solve, or friction lag rejects
every participant together; `result.commit()` preserves the previous state on
failure and `result.rollback()` always returns it.

## Friction

`LaggedCoulombFrictionPlan` accepts one scalar coefficient or a square body-pair
table and a `velocity_threshold`. A lag state freezes route keys,
closest-point coefficients, tangent bases, and normal-force magnitudes for one
inner root solve. The smooth velocity potential has a zero force at
zero slip and a continuous force Jacobian through the regularized stick/slip
transition.

Lag updates occur between complete nonlinear solves. Lag nonconvergence rejects
the physical step; it does not fall back to frictionless contact.

## Differentiation

`monolithic_contact_solution_jvp` and `monolithic_contact_solution_vjp` differentiate
the final fixed physical root on the result's retained candidate epoch, not
the search or solver iterations. Differentiated quantities include selected
branch geometry, collision maps, material-space measures, bulk parameters,
reference shape, initial conditions, barrier stiffness, and friction
coefficient. Search, compaction, feature choice, CCD, line-search rate,
capacity growth, and lag refresh are stopped-gradient decisions.

A derivative certificate requires successful primal/root linear solves and
positive gap, feature, route, CCD, and participant-domain margins. Crossing a
contact route invalidates the local certificate.

## Failure semantics

The contact result retains rejection reasons, candidate evidence, participant
evaluations, safety evidence, and nonlinear/lag diagnostics. No failed category
substitutes another physical model or partially commits accepted history.
