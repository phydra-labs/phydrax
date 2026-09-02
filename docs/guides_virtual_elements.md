# Virtual element methods

Phydrax implements the enhanced conforming scalar virtual element method on
straight-sided two-dimensional polygonal meshes. Virtual elements are a spatial
discretization: linear, DAE, and generalized-eigenvalue solution continue through
`phydrax.linalg` and the existing solver substrates.

## Supported contract

The qualified family has polynomial degree `k=1`, `k=2`, or `k=3`. On each polygon
`K`, the trace is continuous and polynomial of degree `k` on every edge. The
observable coordinates are vertex values, `k-1` interior Gauss--Lobatto edge
values, and normalized cell moments through degree `k-2`.

The enhanced local space makes both the energy projection and the full degree-`k`
L2 projection computable. No explicit interior virtual basis is constructed.
Interior values and gradients returned by the API are polynomial projections.

Current forms cover:

- constant or cellwise-constant scalar/SPD-tensor diffusion;
- constant or cellwise-constant reaction and mass coefficients;
- cell source terms;
- strong Dirichlet trace values;
- Neumann boundary loads;
- Robin boundary operators;
- native mass-matrix DAEs and generalized eigenproblems.

Callable pointwise diffusion, nonlinear forms, vector/mixed spaces, curved edges,
three-dimensional polyhedra, adaptivity, and distributed execution are explicit
non-capabilities.

## Polygon meshes

`CellMesh.from_polygons` accepts cyclic vertex-index loops, canonicalizes them to
counter-clockwise orientation, and groups cells into fixed-arity blocks. The
prepared topology remains one `CellComplexTopology` with shared oriented edges.
Triangles and quadrilaterals retain their existing cell kinds; arity-five and
larger cells use the `polygon` kind.

Preparation requires simple star-shaped cells under `PolygonAdmissibilityPolicy`.
A deterministic ear triangulation is fixed during preparation and used only for
polynomial/source cubature. Geometry refresh may move vertices but cannot change
connectivity or triangulation. Runtime evidence checks positive edges and
triangles, area partition, and a transported star-kernel witness.

## Projection algebra

For centered and measure-scaled monomials `m`, preparation builds:

- `D`: polynomial values under every local DOF functional;
- `B`: integration-by-parts energy functionals;
- `G = B D`: the augmented energy Gram matrix;
- `H`: the polynomial mass Gram matrix.

Batched native local factorizations produce the H1 coefficient projector and the
enhanced L2 coefficient projector. Evidence retains the independent `G-BD`
defect, polynomial reproduction, idempotence, factor validity, singular margins,
and condition estimates. Rank loss fails instead of selecting a pseudoinverse.

## Stabilization

The consistent polynomial action is supplemented only on the projector kernel.
`VirtualElementStabilizationPolicy` selects named `dofi_dofi` or `projected`
stabilization. The selected policy and its scale are part of compilation identity.
Evidence reports polynomial leakage, symmetry, and the stabilized-kernel spectrum.
No diagonal repair or undocumented coupled stabilization is inserted.

## Preparation and solve

The lifecycle mirrors other Phydrax discretizations:

```python
import phydrax as phx

mesh = phx.discretization.CellMesh.from_polygons(coordinates, polygons)
element = phx.discretization.conforming_h1_virtual_element(2)
field = phx.discretization.VirtualElementFieldSpec("u", element)
space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
form = phx.equations.VirtualElementForm(
    "poisson",
    "u",
    (
        phx.equations.DiffusionAction("u", 1.0),
        phx.equations.SourceAction("u", source),
    ),
)
compiled = phx.equations.compile_virtual_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=boundary_value,
)
problem, right_hand_side = compiled.linear_system()
solution = phx.linalg.solve(problem, right_hand_side)
full_solution = compiled.expand(solution.value)
```

`VirtualElementExecutionPolicy` independently selects matrix-free or sparse
realization and fast, deterministic, or compensated local accumulation. Sparse
realization materializes exact local tensors and reuses the ordinary sparse
assembly lifecycle. Matrix-free realization retains the factorized projected
consistency action.

## Boundary and nullspace semantics

A strong constraint acts only on trace point-value DOFs; cell moments are never
assigned fictitious coordinates. A selected exterior-edge domain constrains its
endpoints and high-order edge nodes. Every connected component must be anchored.

An unconstrained diffusion-only form has a declared constant nullspace. The
linear problem requires explicit compatibility and gauge behavior; an
incompatible Neumann right-hand side is not silently projected.

## Reconstruction and differentiation

`project_virtual_element_field` stores cellwise L2 and H1 polynomial coefficients.
The reconstruction API evaluates projected values, projected gradients, and exact
polynomial edge traces. It does not claim access to the virtual interior basis.

Coordinates, polynomial moments, projectors, stabilization, local actions, and
linear solves are differentiable while topology and triangulation remain fixed.
Self-intersection, triangle inversion, loss of the star witness, rank loss, and
boundary selection are validity or discrete derivative boundaries.

## Extended bounded envelope

Polynomial degree is any positive static value that fits
`VirtualElementResourceBudget`; callable scalar/tensor diffusion and mass
coefficients are evaluated at prepared cubature rather than replaced by cell
averages. `ConformingHdiv`, `ConformingHcurl`, and `DiscontinuousL2`
specifications make mixed-space intent explicit.

`CellMesh.from_polyhedra` owns oriented face incidence through root
`PolyhedralConnectivity`. `prepare_polyhedral_h1_virtual_element_3d` is the
first consumer: it prepares a matrix-free degree-one H1 consistency plus
projector-kernel stabilization action on star-visible, outward-oriented
polyhedra, retaining volume, rank-margin, and polynomial-reproduction evidence.
`prepare_polyhedral_polynomial_vem_3d` accepts the exact higher-degree
face/cell-moment projector and consistency/stabilization tensors produced by
the root-topology cubature route, and rejects rank loss or failed P_k
reproduction. `VirtualElementProductPlan` binds mixed H1/L2, H(div)/L2, or
H(curl)/L2 block actions with explicit inf-sup and commuting evidence.
`CurvedVirtualElementEdge` retains mapped chart points, tangents, arc weights,
and the minimum Jacobian. `adapt_virtual_element_p` and
`adapt_virtual_element_hp` create immutable degree/topology epochs with
constant-preserving transfers; marking and topology remain nondifferentiable.

## References

- L. Beirao da Veiga et al., *Basic Principles of Virtual Element Methods*, 2013.
- L. Beirao da Veiga et al., *The Hitchhiker's Guide to the Virtual Element Method*, 2014.
- L. Mascotto, *The role of stabilization in the virtual element method: A survey*, 2023.
- A. Dedner and A. Hodson, *A framework for implementing general virtual element spaces*, 2023.
