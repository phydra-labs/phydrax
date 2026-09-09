# Graded-index rays and caustics

`GradedIndexRayPlan` integrates the Hamiltonian ray system with canonical momentum `p = n t`, a fixed symplectic kick–drift–kick schedule, geometric length, optical length, and the full canonical tangent map.

Refractive media are explicit `StructuredRefractiveIndexField` or `TetrahedralRefractiveIndexField` values. Structured interpolation is piecewise trilinear. Tetrahedral P1 gradients are cellwise constant and route derivatives are conditional on unchanged cell membership.

Evidence retains field coverage, Hamiltonian drift, symplectic residual, finite state, and plan identity. `RayFanPlan` lowers tangent maps to transverse determinants, singular values, and bounded caustic-crossing evidence. A caustic invalidates single-route geometric amplitude; it is not repaired by clipping.

`CurvedSchlierenPlan` projects final-minus-initial ray direction onto the detector basis and returns a typed angular measurement. It remains distinct from straight-ray and coherent-wave Schlieren.
