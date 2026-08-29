# DEM rigid bodies and shapes

`RigidBodySetPlan` composes with stable particle ownership. In 2-D, orientation is an SO(2) angle and inertia is scalar. In 3-D, orientation is a scalar-first Hamilton quaternion mapping body vectors to world vectors; inertia is a symmetric positive-definite body-frame tensor and angular velocity is world-frame.

The rigid-body step uses half translational/angular kicks, Lie-group orientation drift, a refreshed load, and a closing kick. Quaternion addition is never used. Invalid norm, non-SPD inertia, or nonfinite pose rejects.

## Sphere clumps

`SphereClumpTemplatePlan` stores centered component offsets, radii, masses, materials, total mass, inertia, and bounding radius. `RigidSphereClumpSetPlan` stores only owner template IDs and owner state. Component keys include both owner stable IDs and immutable local component indices.

Owner contact reduction preserves equal/opposite force and computes torque about the owner center. A rigid clump cannot fracture; fracture uses independent bodies and a bond graph.

## Contact geometry

`RigidContactGeometry` carries normal, signed gap, overlap, effective radius, common contact point, owner/contact lever arms, relative point velocity, stable feature key, degeneracy code, and feature margin.

Triangle walls have deterministic face/edge/vertex ownership. Convex polyhedra use face normals and all edge-cross-edge separating axes. Sphere-to-implicit contact requires declared distance-error and Lipschitz certificates. Convex and implicit paths do not claim Hertz/Mindlin curvature semantics.
