# Finite elements

Phydrax finite elements compile immutable meshes, reference elements, field spaces,
and weak terms into native linear, nonlinear, and differential-algebraic problems.
The discretization never owns a Newton method or time integrator.

## Computational mesh

`CellMesh` is the shared computational realization used by finite elements and
unstructured finite volume. `CellBlock` retains ordered local vertices and a cell
kind; `CellComplexTopology` remains the incidence authority. Connectivity and
entity identities are static, while coordinate arrays are numeric geometry.

Supported cell blocks are triangles, quadrilaterals, tetrahedra, and
hexahedra. Polygonal meshes order triangle blocks before quadrilateral blocks
so global cell/facet routes remain canonical.

Arbitrary straight-sided polygons use the separate
[virtual-element substrate](guides_virtual_elements.md); they are not represented
as reference finite elements with fabricated interior basis tabulations.

## Reference elements and fields

`lagrange_element(cell_kind, degree)` constructs nodal triangle, quadrilateral,
tetrahedron, and hexahedron elements. Arbitrary-order conforming entity
numbering is executable for polygonal and hexahedral H1 fields; discontinuous
fields remain cell-local. Coefficient representation is declared by the
element independently of H1/L2/H(div)/H(curl) conformity. Additional
constructors provide discontinuous P0/P1+, triangular RT0, and triangular
first-kind Nedelec order zero.

`FiniteElementFieldSpec` supports replicated component shapes and multiple named
fields. One `CompiledFiniteElementProblem` owns the ordered product space and
scatters every coupled term directly to its output residual block.

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

`FiniteElementForm` supports diffusion, mass, source, boundary load, general
cell residual/energy/bilinear actions, exterior and interior numerical fluxes,
SIPG facet actions, and prepared global operator actions. The compiled
`WorksetProgram` is the authoritative residual execution schedule.

Coefficients may be physical point functions or arrays explicitly bound to
cell entities, facet entities and sides, quadrature rules, or field DOFs.
Non-point coefficients retain exact support, entity-set, field-space, rule,
side, shape, and axis-layout identities; array rank is not used as scientific
metadata. A staged point coefficient receives the execution context:

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

with explicit full and reduced spaces. Raw weak residuals use the algebraic
dual pullback. Pairing adjoints and physical mass projections are separate
operators and coincide with a raw transpose only under the corresponding
identity pairings. Every connected mesh component must be anchored. Natural
boundary data remains a weak-form term.

## Solvers and execution

A compiled affine form provides `linear_system()` and explicit nullspace policy.
General residuals expose `as_nonlinear_problem()`, matrix-free linearization,
lagged/Picard operator factories, and a scalable adjoint solve.

`as_dae_system()` includes dynamic geometry, time, lift, and lift-rate terms.
`as_second_order_system()` adds configuration, velocity, acceleration, and
lift-acceleration semantics. `as_generalized_eigenproblem()` returns native
constrained stiffness/mass operators.

`FiniteElementExecutionPolicy` independently selects matrix-free versus sparse
realization, dense/partial/sum-factorized/collocated local kernels, and
fast/deterministic/compensated residual accumulation. Sparse execution uses the
existing `SparseAssemblyPlan` prepare/refresh lifecycle.

## Materials, compatible methods, and hierarchy

Pure `ConstitutiveModel` updates return response, candidate quadrature state, and
diagnostics. `FiniteElementMaterialTransaction` commits or rolls back all
material regions atomically; FE checkpoints bind field/material state to exact
prepared and compilation IDs.

The substrate exposes local elimination, HDG trace condensation, explicit
transfer roles, refinement lineage, residual/jump and DWR estimators, embedded
quadrature, enrichment/multiscale bases, partitioned DOF maps, and halo
sum/average/update semantics.

## Local-action IR and high order

`FiniteElementForm` lowers to `LocalActionIR`, `KernelTable`, and a typed
`WorksetProgram`. Cell and facet worksets own the static gathers, orientations,
domains, and rule identities used by residual execution. Matrix-free JVPs
differentiate this same program.

`SimplexNodalFamily` and `ReferenceNodalFamily` provide arbitrary-order
simplex and anisotropic tensor nodal references. `TensorProductTabulation`,
`SumFactorizationPlan`, and `PreparedFiniteElementReference` bind explicit
volume/facet rules, dense actions, trace data, and reusable two- or three-axis
tensor contractions. The authoritative compiler selects dense, partial,
sum-factorized, or collocated kernels through the same workset program.

See [Spectral elements](guides_spectral_elements.md) for mapped tensor
geometry, high-order CG/DG, GLL mass, DGSEM, multigrid, mortars, hp
transactions, and distributed ownership.

## SIPG Poisson

`sipg_poisson_form` implements cell diffusion, weighted consistency and
symmetry terms, harmonic coefficient weighting, explicit `p²/h` penalty
scaling, Nitsche Dirichlet data, natural Neumann data, Robin data, and a
verified constant nullspace for pure Neumann problems. Plus is the owner side;
the stored normal points outward from plus; both normal derivatives use that
same normal. Current executable SIPG support is scalar DG on one homogeneous
triangle or quadrilateral block.

## Local adaptation and applications

`dorfler_mark`/`maximum_mark`, `refine_triangles_local`, complete-family
coarsening, P1 primal/dual transfers, local DWR indicators, and
`FiniteElementTopologyTransaction` provide a single-device accepted topology
transaction. Failed material transfer or certification preserves the accepted
state.

Tensor hp adaptation uses `FiniteElementHPTopology` as an allocated refinement
forest and `FiniteElementHPEpoch` as the immutable prepared snapshot. Isotropic
quad/hex h-refinement composes with anisotropic p, master-trace H1 constraints,
DG mortar interfaces, curved parent-map inheritance, role-specific transfers,
degree-bucket condensation/multigrid, and atomic solver promotion. Error evidence
and hp decisions remain separate; the marking decision is a discrete derivative
boundary.

Executable application namespaces live under `phydrax.applications`:
phase-field Allen-Cahn/Cahn-Hilliard, finite-strain crystal plasticity,
frictionless persistent-pair contact, phase-field fracture, and fixed-crack
XFEM classification/enrichment.


## Smoothed finite elements

Cell-, edge-, node-, and fully smoothed axisymmetric methods use composite
smoothing patches and boundary moments rather than ordinary cell quadrature.
See [Smoothed finite elements](guides_fem_smoothing.md) for exact method scopes,
stability evidence, source-backed presets, and axisymmetric primitive moments.

## Time laws and accepted-step schedules

`TimeLaw` supplies value and time derivatives. `FiniteElementAcceptedState`,
`FiniteElementAcceptedStepSchedule`, and `FiniteElementTopologyTransaction`
separate immutable accepted data from candidate field/material/topology state.
Rejected attempts do not increment state or material versions.

## Current limits

Execution remains single-device unless a caller supplies a JAX named-axis
collective context; no MPI runtime or mesh partitioner is claimed. Compatible
elements are triangle RT0/Nedelec0; HDG is lowest-order triangular primal HDG;
the legacy SIPG convenience remains scalar on one homogeneous 2-D polygon
block. Triangle adaptation remains conforming T3 refinement; tensor h-refinement
is isotropic and 2:1 balanced. DGSEM remains stationary-mesh only.
Search, active-set selection, marking, topology changes, and hp candidate
promotion are discrete derivative boundaries.
