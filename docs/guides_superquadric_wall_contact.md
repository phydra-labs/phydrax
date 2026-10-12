# Superquadric triangle-wall contact

`SuperquadricTriangleContactPlan` resolves convex superquadrics against finite triangle walls without substituting a bounding-sphere force.

## Feature oracle

The contact oracle evaluates seven deterministic candidates per particle/triangle route:

- face interior;
- three edge interiors;
- three vertices.

Face contact uses the analytic superquadric support point in the oriented wall-normal direction and checks barycentric inclusion. Edge and vertex candidates use bounded fixed-iteration support/KKT alignment. Every result reports witness residual, selected feature, feature-tie margin, and validity.

One-sided walls reject back-face candidates. Two-sided walls orient the normal toward the particle center.

## Stable ownership

`TriangleWallPlan` accepts stable vertex and triangle IDs. Preparation constructs canonical edge IDs, adjacency, and the lowest stable triangle owner for each shared edge or vertex. Only that owner emits the nonsmooth feature contact, preventing duplicate forces at mesh seams.

Particle-wall interaction identities contain particle ID, wall object ID, feature kind, and stable feature ID. Tangential and rotational history therefore survives slot reordering and particle-capacity growth. A genuine feature transition is an explicit contact death/birth event.

## Curvature

Planar-face Hertz contact combines superquadric curvature with zero wall curvature. A mathematically sharp edge or vertex does not claim a Hertz radius. `edge_rounding_radius` and `vertex_rounding_radius` make that curvature physical and explicit when required. Linear and plastic contact laws remain available on sharp features.

## Dynamics and observables

`DiscreteElementProblemIR(walls=(wall_plan,))` supplies triangle walls to the shared compiler; `CompliantDEMMethodPlan(..., wall_geometry=SuperquadricTriangleContactPlan())` selects the witness solver. Initialize through `compiled.initialize_state(..., orientation=orientation)`. `evaluation.boundaries` contains barriers followed by triangle-wall responses, including `particle_force`/`particle_torque`, equal-and-opposite `reaction_force`/`reaction_torque`, contact history, flat shared `geometry`, and wall power. Triangle responses retain the shape-specific `triangle_witness` used by facet traction and Finnie wear. Moments use owner arms at the common force point, including the half-gap offset rather than an uncorrected surface-witness arm.

AABB distance against the superquadric bounding radius and interaction range provides the fail-closed broad phase; dense routes remain the correctness authority.

Sharp derivatives are valid only away from feature ties and convergence boundaries.

Run `examples/superquadric_triangle_wall.py` and `tools/superquadric_wall_qualification.py`.
