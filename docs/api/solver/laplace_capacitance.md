# 3D Laplace capacitance

The capacitance solver uses a fixed, closed, triangular `MeshRegion`, one discontinuous
constant density per face, and the Laplace single-layer kernel. Conductor selections
must partition complete disconnected surface components. The initial implementation
rejects nested, intersecting, inward-oriented, open, and geometry-changing inputs.

```python
import jax.numpy as jnp
import phydrax as phx

region = phx.geometry.MeshRegion(vertices, faces)
galerkin = phx.operators.prepare_laplace_single_layer_dp0_3d(region)

left = phx.discretization.EntitySelection(galerkin.surface_entities, left_mask)
right = phx.discretization.EntitySelection(galerkin.surface_entities, right_mask)

result = phx.solver.solve_laplace_capacitance_3d(
    galerkin,
    {"left": left, "right": right},
    permittivity=8.8541878128e-12,
)

assert bool(result.valid)
field, target_report = phx.operators.evaluate_laplace_layer_3d(
    result.potentials[0],
    jnp.asarray([[2.0, 0.0, 0.0]]),
    target_side="exterior",
)
```

Conductor names are sorted and define both capacitance axes. `layer_density[:, j]` is
the normalized single-layer density for unit voltage on conductor `j`;
`surface_charge_density` is permittivity times that density. `capacitance[i, j]` is the
area-integrated physical charge on conductor `i`. Existing potential evaluators consume
the normalized density, so unit-excitation potentials do not change with permittivity.

The production operator is nonmaterializable. A dense oracle is available only when
explicitly requested during Galerkin preparation. The default Jacobi-preconditioned
FGMRES solve is a bounded first-kind baseline, not a mesh-independent preconditioner.
Every column retains its complete `LinearSolveResult`; numerical failure produces
`valid=False` in status mode and never triggers a fallback.

Current scope excludes solve differentiation, FMM/H² acceleration, mixed or Neumann
conditions, transmission, open screens, topology changes, and continuum-discretization
error estimation.

::: phydrax.solver.LaplaceCapacitanceResult3D

---

::: phydrax.solver.solve_laplace_capacitance_3d
