# Explicit polygon H1 elements

Phydrax provides a deterministic lowest-order H1 discretization on conforming,
straight-sided, star-shaped polygon meshes. It constructs an explicit piecewise-P1
basis on a fan from a transported star-kernel witness, then statically condenses
the private witness value by a local discrete-harmonic solve. Only polygon vertex
values remain as global degrees of freedom.

This is a finite-element-like discretization and is separate from the
[virtual-element substrate](guides_virtual_elements.md). Virtual elements retain
polynomial projections and projector-kernel stabilization without constructing an
interior basis. Explicit polygon H1 elements instead provide actual interior values
and gradients and use no virtual-element projector or stabilization.

## Supported contract

The qualified surface is deliberately narrow:

- planar two-dimensional `CellMesh` values with `PolygonalConnectivity`;
- simple convex or concave polygons accepted by `PolygonAdmissibilityPolicy`;
- degree-one H1 fields with one scalar coefficient per mesh vertex;
- arbitrary trailing field component shape, including two-component displacement;
- mixed fixed-arity cell blocks with padded local routes;
- exact linear edge traces, partition of unity, and affine reproduction;
- cell value/gradient and exterior-facet value/trace local actions;
- dense matrix-free variational execution;
- strong component-aware Dirichlet constraints;
- fixed-topology differentiable geometry refresh;
- direct value and piecewise-gradient reconstruction;
- prepared, evidenced field queries and exact prepared edge traces.

Sparse realization, higher order, three-dimensional polyhedra, Hessian jets,
interior-facet/DG terms, and mixed incompressibility are not provided. Nonmatching
T-junction interfaces are rejected: every geometric interface must have identical
edge segmentation on both incident cells. Matched collinear vertices are valid.

## Construction and evidence

For a polygon with vertices `v_i`, preparation freezes affine weights for a point
`w` strictly inside the star kernel. Runtime geometry reconstructs `w` from the
current vertices and creates triangles `(w, v_i, v_{i+1})`. Ordinary affine P1
stiffness is assembled on this fan. Partitioning the fine stiffness into boundary
and private-witness blocks gives the extension

```text
u_w = -K_ww^{-1} K_wb u_b.
```

The solve uses the native local-block factorization and fails on an invalid pivot;
no pseudoinverse or diagonal repair is inserted. Because boundary fine-node rows
remain the identity, adjacent cells have the same linear trace. Discrete harmonic
extension reproduces affine functions exactly up to the declared precision.

`ExplicitPolygonH1BasisEvidence` retains fan measure, area partition,
factorization, condensation residual, boundary identity, partition and affine
reproduction, stiffness rank/spectrum, mass positivity, conditioning, and finite
value checks. Initial preparation fails if any cell misses its qualification policy.
A geometry refresh preserves topology but recomputes every geometry-dependent
basis quantity and certificate.

## Preparation and solve

```python
import jax.numpy as jnp
import phydrax as phx

coordinates = jnp.asarray(
    ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
     (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
     (0.0, 1.0), (0.5, 1.0), (1.0, 1.0))
)
cells = (
    (0, 1, 4, 3), (1, 2, 5, 4),
    (3, 4, 7, 6), (4, 5, 8, 7),
)
mesh = phx.discretization.CellMesh.from_polygons(coordinates, cells)
field = phx.discretization.ExplicitPolygonH1FieldSpec("u")
space = phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()
constraint = phx.discretization.explicit_polygon_h1_dirichlet_constraint(space, "u")
form = phx.equations.FiniteElementForm(
    "poisson",
    "u",
    (
        phx.equations.DiffusionAction("u", 1.0),
        phx.equations.SourceAction("u", source),
    ),
)
compiled = phx.equations.compile_finite_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=boundary_value,
)
problem, right_hand_side = compiled.linear_system()
solution = phx.linalg.solve(problem, right_hand_side)
full_solution = compiled.expand(solution.value)
```

The same prepared local actions execute representation-independent functionals.
In particular, a displacement field with `component_shape=(2,)` can use
`phydrax.applications.solid_mechanics.neo_hookean_functional` without a
polygon-specific mechanics compiler.

## Reconstruction and differentiation

`prepare_explicit_polygon_h1_reconstruction` binds a state to one runtime.
`evaluate_explicit_polygon_h1_reconstruction` locates each supplied point in the
fixed witness fan and returns the actual basis value and gradient. Values agree on
internal fan edges; the gradient is piecewise constant and the lowest-index fan
triangle is selected deterministically at a tie. Point location itself is a
discrete operation and is not a geometry-differentiation surface.

Assembly and runtime refresh are differentiable while connectivity, arity, fan
routes, witness weights, and admissibility remain fixed. Crossing an inversion,
star-margin, factorization, or qualification boundary is a validity boundary rather
than a differentiable repair.

## Prepared queries and side traces

`phydrax.discretization.explicit_polygon_h1.prepare_explicit_polygon_h1_field_reconstruction`
returns a `PreparedFieldReconstruction` with `approximation="exact"`, `C^0`
regularity, first derivatives, and a cell-sided trace policy. Its support
geometry is the region covered by the witness fans. `prepare_query` locates
fixed points once (bounded BVH candidate cells, fixed point batches, no
point-by-cell tensor) and reuses the gather/scatter route for every coefficient
refresh. Points outside the mesh are `OUTSIDE_SUPPORT` and can be masked with
`coverage="masked"`.

The discrete gradient is constant on each fan triangle. A derivative query on a
shared cell edge needs a bound trace side; a derivative query on an interior
fan edge or at a polygon vertex reports `SIDE_UNRESOLVED` for every side,
because the gradient jumps inside the cell there. No fan triangle is chosen
silently, and Hessians remain unavailable.

`ExplicitPolygonH1Discretization.prepare_side_trace(field, domain, rule=...)`
prepares the exact value trace on the selected exterior or interior facets of
`integration_domain(...)`. The trace is linear in the two edge vertex values
(`trace_degree=1`) and equals `evaluate_explicit_polygon_h1_trace`. Sites follow
the owner cell's local edge parametrization and are the same physical points
for `side="owner"` and `side="neighbor"`; `normals` point out of the side cell
and `weights` are the physical edge measure. Only `quantity="value"` exists;
`side="neighbor"` on exterior facets and `side="average"` are refused.

```python
from phydrax.discretization import FacetTraceRule
from phydrax.discretization.explicit_polygon_h1 import (
    prepare_explicit_polygon_h1_field_reconstruction,
)

reconstruction = prepare_explicit_polygon_h1_field_reconstruction(space)
query = reconstruction.prepare_query(points, derivative=(1, 0))
slopes = query.apply(full_solution)
trace = space.prepare_side_trace(
    "u", space.exterior_facet_domain, rule=FacetTraceRule(points=3)
)
boundary_values = trace.apply(full_solution)
boundary_load = trace.inject_load(density)
```

## References

- S. Berrone, M. Pintore, and G. Teora, *The Neural Approximated Virtual Element Method for Elasticity Problems*, 2025.
- S. Berrone, M. Pintore, and G. Teora, *Two continuous extensions of the Neural Approximated Virtual Element Method*, 2026.
- S. Berrone et al., *The Zipped Finite Element Method: High-order Shape Functions for Polygons*, 2025.
