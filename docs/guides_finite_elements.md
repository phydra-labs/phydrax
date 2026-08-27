# Finite elements

Phydrax finite elements compile immutable meshes, reference elements, field spaces,
and weak terms into native linear, nonlinear, and differential-algebraic problems.
The discretization never owns a Newton method or time integrator.

## Computational mesh

`CellMesh` is the shared computational realization used by finite elements and
unstructured finite volume. `CellBlock` retains ordered local vertices and a cell
kind; `CellComplexTopology` remains the incidence authority. Connectivity and
entity identities are static, while coordinate arrays are numeric geometry.

Supported cell blocks are triangles, quadrilaterals, and tetrahedra. Polygonal
meshes order triangle blocks before quadrilateral blocks so global cell/facet
routes remain canonical.

## Reference elements and fields

`lagrange_element(cell_kind, degree)` currently constructs:

- triangle P1 and P2;
- quadrilateral Q1;
- tetrahedron P1.

`FiniteElementSpec.tabulate(points)` returns basis values and reference gradients.
A `FiniteElementFieldSpec` assigns one compatible element to every mesh block.
Preparation builds `DiscreteFieldSpace` coordinates and a separate
`FiniteElementDofMap` for cell-local gathers. P2 triangles share edge DOFs through
the mesh edge entities.

## Geometry

Physical points, metric determinants, and physical gradients are computed from
reference geometry in pure JAX. For fixed connectivity:

```text
areas = discretization.evaluate_geometry("u", new_coordinates)[0].measure
```

This path is differentiable with respect to `new_coordinates`. Changing
connectivity or coordinate shape requires a new plan.

## Weak forms and dual residuals

`WeakForm` is an ordered graph of `DiffusionTerm`, `MassTerm`, `SourceTerm`, and
`BoundaryLoadTerm`. Constants become `ResolvedCoefficient` values. A staged
callable coefficient requires an explicit ID:

```python
import phydrax as phx

forcing = phx.equations.coefficient(
    lambda points, args: args["amplitude"] * points[..., 0],
    coefficient_id="x-forcing",
)
```

The assembled weak residual belongs to `DualSpace(test_space)`. This preserves the
difference between a test functional, a primal field vector, the field Riesz map,
and the physical mass operator.

`FiniteElementFunctional` uses the same geometry and quadrature machinery but
reduces to one scalar. Its reduction uses compensated accumulation when enabled by
`FiniteElementPrecisionPolicy`.

## Essential constraints

`dirichlet_constraint` constructs the affine map

```text
u = P z + g
```

with explicit full and reduced spaces. Raw weak residuals use the algebraic dual
pullback, while primal vectors use the pairing-aware adjoint. Every connected mesh
component must be anchored. Natural boundary data remains a weak-form term.

## Solvers

A compiled affine form provides `linear_system()`, returning native `LinearSystem`
and right-hand side values. General residuals expose `as_nonlinear_problem()` for
the existing nonlinear preparation and Jacobian policies.

`as_dae_system()` constructs the native mass-matrix residual

```text
M u_dot + r(u, t) = 0
```

without conflating `M` with the field Riesz map.

## Sparse and matrix-free execution

Cell residual evaluation is matrix-free gather/evaluate/scatter. Certified
constant affine mass and diffusion terms additionally lower to
`SparseCoordinateOperator` over current sparse relations. Constraint projection
is applied after the full-space action.

## Current limits

The current release is fitted, fixed-topology, and single-device. H(div), H(curl),
DG/HDG, curved high-order geometry, path-dependent materials, adaptivity, contact,
and distributed FE layouts are not yet exposed. Unsupported element/cell
combinations fail during planning.
