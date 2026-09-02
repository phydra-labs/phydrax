# 3D Laplace capacitance

The prepared solver binds one immutable `BoundaryMeshEpoch`, one closed
triangle-DP0 Laplace Galerkin product, and a disjoint conductor partition.

```python
import jax.numpy as jnp
import phydrax as phx

region = phx.geometry.MeshRegion(vertices, faces)
galerkin = phx.operators.prepare_laplace_single_layer_dp0_3d(region)
epoch = phx.operators.BoundaryMeshEpoch(galerkin._binding.mesh)

left = phx.discretization.EntitySelection(galerkin.surface_entities, left_mask)
right = phx.discretization.EntitySelection(galerkin.surface_entities, right_mask)

prepared = phx.solver.LaplaceCapacitancePlan3D(
    epoch,
    galerkin,
    {"left": left, "right": right},
).prepare()
result = prepared.solve(permittivity=8.8541878128e-12)
assert bool(result.valid)
```

Conductor names are sorted and define both capacitance axes.
`layer_density[:, j]` is the normalized density for excitation `j`; physical
surface charge and the Maxwell capacitance matrix scale smoothly with
permittivity. The native solve policy uses mathematical implicit
differentiation. `differentiate_laplace_capacitance_coordinates_3d` consumes
the exact fixed-pair Galerkin operator and face-area coordinate tangents and
returns the implicit density/capacitance JVP. Pair classes and topology remain
discrete derivative boundaries.

`advance_laplace_capacitance_3d` accepts an explicit
`BoundaryRefinementResult`, target Galerkin product, target conductor
selections, and one-to-one conductor correspondence. It returns a candidate
prepared target plus DP0 parent transfer. The transition is atomic and
nondifferentiable; stale meshes are rejected. Boundary refinement transfers
surface tags, selections, and interfaces through exact parent lineage.

The default Jacobi-preconditioned FGMRES path remains an explicitly named
baseline. For a caller-prepared, rank-certified barycentric dual family,
`prepare_laplace_stable_dual_calderon_3d` validates the shape-regularity margin
and cross-mass rank and builds `M_d^-T (W_tilde + R) M_d^-1`. No
mesh-independent claim is made for slivers, open screens, rank loss, or a
foreign dual family.

::: phydrax.solver.LaplaceCapacitancePlan3D

::: phydrax.solver.PreparedLaplaceCapacitance3D

::: phydrax.solver.LaplaceCapacitanceResult3D

::: phydrax.solver.advance_laplace_capacitance_3d
