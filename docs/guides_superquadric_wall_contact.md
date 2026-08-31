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

`SuperquadricDEMPlan` accepts prepared triangle walls. Wall responses contribute particle force/torque, equal-and-opposite reaction, contact history, and wall work. Existing facet traction, Finnie wear, and servo-control surfaces accept superquadric wall responses through the common rigid-contact geometry contract.

AABB distance against the superquadric bounding radius and interaction range provides the fail-closed broad phase; dense routes remain the correctness authority.

Sharp derivatives are valid only away from feature ties and convergence boundaries.

Run `examples/superquadric_triangle_wall.py` and `tools/superquadric_wall_qualification.py`.
