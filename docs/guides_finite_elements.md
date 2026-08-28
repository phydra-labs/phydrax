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

`lagrange_element(cell_kind, degree)` constructs triangle P1/P2,
quadrilateral Q1, and tetrahedron P1. Additional constructors provide
discontinuous P0, triangular RT0, and triangular first-kind Nedelec order zero.
Compatible elements carry their Piola mapping and edge orientation through the
prepared DOF map.

`FiniteElementFieldSpec` supports replicated component shapes and multiple named
fields. Preparation exposes one ordered `BlockSpace`; mixed weak forms return one
residual block per declared field. P2 vertex and edge coordinates remain
entity-stratified in `BlockDofLayout`.

`FiniteElementCoordinateSpec` assigns an independent coordinate element and
geometry DOF map to every block. This permits curved P2 geometry with a lower-
or higher-order field element.

## Geometry

Physical points, metric determinants, normals, physical gradients, and Piola-
mapped compatible bases are computed in pure JAX. `prepare_runtime` creates a
fixed-topology numeric realization:

```text
runtime = discretization.prepare_runtime(new_coordinates, numeric_version="moved")
context = phx.equations.FiniteElementExecutionContext(runtime)
residual = compiled.residual(state, context)
```

Coordinates flow through residuals, sparse refresh, functionals, DAE mass
operators, and shape derivatives. Connectivity changes require a new plan.

## Domains, coefficients, and weak forms

`EntitySelection` composes union, intersection, difference, and complement over
one exact entity set. A selected cell, exterior-facet, or interior-facet
`IntegrationDomain` owns resolved owner/neighbour and local-facet routes. Terms
bind existing `phydrax.integration` reference rules by cell block.

`WeakForm` supports built-in diffusion, mass, source, and boundary load terms,
general `CellResidualTerm`, energy-derived `CellEnergyTerm`, optimized
`CellBilinearTerm`, and two-sided `InteriorFacetTerm` numerical fluxes.

Coefficients may be point functions, cell arrays, facet arrays, or quadrature
arrays. A staged coefficient receives the execution context:

```python
import phydrax as phx

forcing = phx.equations.coefficient(
    lambda points, context: context.user_args["amplitude"] * points[..., 0],
    coefficient_id="x-forcing",
)
```

The assembled weak residual belongs to `DualSpace(test_space)`. This preserves
the distinction between a test functional, a primal field vector, the field
Riesz map, and the physical mass operator.

`FiniteElementFunctional` accepts selected domains and native reference rules;
its reduction follows `FiniteElementPrecisionPolicy`.

## Essential constraints

`dirichlet_constraint` constructs the affine map

```text
u = P z + g
```

with explicit full and reduced spaces. Raw weak residuals use the algebraic dual
pullback, while primal vectors use the pairing-aware adjoint. Every connected mesh
component must be anchored. Natural boundary data remains a weak-form term.

## Solvers and execution

A compiled affine form provides `linear_system()` and explicit nullspace policy.
General residuals expose `as_nonlinear_problem()`, matrix-free linearization,
lagged/Picard operator factories, and a scalable adjoint solve.

`as_dae_system()` includes dynamic geometry, time, lift, and lift-rate terms.
`as_second_order_system()` adds configuration, velocity, acceleration, and
lift-acceleration semantics. `as_generalized_eigenproblem()` returns native
constrained stiffness/mass operators.

`FiniteElementExecutionPolicy` selects matrix-free or sparse realization and
fast, deterministic, or compensated residual accumulation. Sparse execution
uses the existing `SparseAssemblyPlan` prepare/refresh lifecycle.

## Materials, compatible methods, and hierarchy

Pure `ConstitutiveModel` updates return response, candidate quadrature state, and
diagnostics. `FiniteElementMaterialTransaction` commits or rolls back all
material regions atomically; FE checkpoints bind field/material state to exact
prepared and compilation IDs.

The substrate exposes local elimination, HDG trace condensation, explicit
transfer roles, refinement lineage, residual/jump and DWR estimators, embedded
quadrature, enrichment/multiscale bases, partitioned DOF maps, and halo
sum/average/update semantics.

## Current limits

Implemented compatible and discontinuous families are deliberately small:
triangle RT0/Nedelec0 and discontinuous P0. General p, hexahedral compatible
families, automatic mesh refinement kernels, cut-cell classification, contact,
and real multi-process communication backends remain future family/backend
implementations over the now-explicit contracts.
