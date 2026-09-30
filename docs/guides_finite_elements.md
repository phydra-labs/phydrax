# Finite elements

Phydrax finite elements compile immutable meshes, reference elements, field spaces,
and weak terms into native linear, nonlinear, and differential-algebraic problems.
The discretization never owns a Newton method or time integrator.

## Computational mesh

`CellMesh` is the shared computational realization used by finite elements and
unstructured finite volume. `CellBlock` retains ordered local vertices and a cell
kind; `CellComplexTopology` remains the incidence authority. Connectivity and
entity identities are static, while coordinate arrays are numeric geometry.

`CellMesh.from_simplices(..., dimension=...)` admits intervals and canonical
full-dimensional n-D affine simplices with explicit dimension.
`ReferenceCellTopology` uses `simplex:N`/`tensor:N` identities and a generic
nominal facet descriptor. Generic tensor meshes remain dimension1–3; standalone
tensor form bases may be n-D. Polygonal meshes order triangle blocks before
quadrilateral blocks so global cell/facet routes remain canonical.

Arbitrary straight-sided polygons have two separate substrates. The
[explicit polygon H1 method](guides_explicit_polygon_h1.md) constructs a
degree-one interior basis by discrete-harmonic fan condensation. The
[virtual-element substrate](guides_virtual_elements.md) retains functional
degrees of freedom, polynomial projections, and projector-kernel stabilization.
Neither path fabricates a reference-polygon polynomial element.

Affine simplex reconstruction uses exact membership and exterior-distance queries,
not a fabricated bounding box or an integration-measure claim. Quadrilateral and
planar-face hexahedral queries use their exact native regions; a warped hexahedron
requires an explicit curved-geometry witness.


## Reference elements and fields

`lagrange_element(cell_kind, degree)` constructs nodal triangle, quadrilateral,
tetrahedron, and hexahedron elements. Arbitrary-order conforming entity
numbering is executable for polygonal and hexahedral H1 fields; discontinuous
fields remain cell-local. `FiniteElementSpec.value_spec` declares canonical form
degree, twist and proxy; conformity/mapping are derived, with continuity declared
separately. `form_element` supplies trimmed/full simplex and tensor-trimmed
families in canonical n-D references. Physical flux/density requires explicit
twist, and ambiguous degree1 2-D maps require circulation or flux explicitly.

`FiniteElementFieldSpec` supports replicated component shapes and multiple named
fields. One `CompiledFiniteElementProblem` owns the ordered product space and
scatters every coupled term directly to its output residual block.

`phydrax.discretization.CellGeometrySpec` assigns an independent coordinate element and
geometry DOF map to every block. This permits curved P2 geometry with a lower-
or higher-order field element.

`FiniteElementDeRhamComplex` provides exact sparse degree maps, metric-only Gram
Hodges, reconstruction/traces and commuting `ComplexMap` transfers. Full-family
complex order is its top polynomial order, so degree k uses P_(order+n−k)Λk;
trimmed/tensor-trimmed complexes keep the same order in each degree. Material
weights are separate coordinate forms, not part of an SPD metric Hodge.
`hodge_solve` accepts a native `LinearSolvePolicy`, never a string selector.
Hiptmair–Xu uses native vector/potential corrections; high-order preparation
composes the low-order auxiliary plan instead of another private solver.


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
`IntegrationDomain` owns resolved owner/neighbor and local-facet routes. Terms
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

`finite_element_form_from_functional` and
`compile_finite_element_functional` bind `phydrax.variational.Functional`
terms to the same worksets used for scalar value, dual first variation, and
matrix-free Hessian actions. Cell terms support value and gradient jets;
two-dimensional polygonal exterior terms support value jets and outward normals.

Prepared-local implementations, including IGA, consume the same portable term
through their interpolation, gradient, geometry, and transpose actions.
`CellEnergyAction` and `FiniteElementFunctional` remain lower-level adapters for
representation-specific callbacks.

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

Integration-site laws implement `AbstractConstitutiveModel` (`MODEL` authority).
Pure `ConstitutiveModel` updates return response, candidate quadrature state, and
diagnostics; `LearnedConstitutiveModel` evaluates a learned model bound to the slot
with unit-carrying strain/stress/history ports, a support box, and an
`AdmissibilityHeader` per site. Its consistent tangent is the exact forward-mode
derivative of the learned response, and out-of-support sites are invalid with NaN
derivatives, so native Newton solves own acceptance and implicit parameter
gradients never silently use an invalid tangent. `LearnedLocalImplicitMaterial` is
the learned counterpart of `LocalImplicitMaterial` for local constitutive roots.
`FiniteElementMaterialTransaction` commits or rolls back all
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

`dorfler_mark`/`maximum_mark`, residual/jump and local DWR indicators select
cells; `phydrax.meshing.prepare_mesh_adaptation` with the `NATIVE_BISECTION`
route refines them conformingly and coarsens complete bisection patches. The
`MeshAdaptationResult` carries the sparse P1 `FiniteElementTopologyTransfer` and
lineage that `FiniteElementTopologyTransaction.execute(accepted, mesh,
adaptation)` consumes as a single-device accepted topology transaction. Failed
material transfer or certification preserves the accepted state.

`FiniteElementTopologyTransfer` stores the primal coefficient map (target DOFs by
source DOFs) either as one `SparseLinearMap` with O(targets x stencil width)
memory or as a linear operator whose action couples every DOF. `apply` is the
primal transfer, `pullback` is its algebraic transpose for residual and load
duals, and an optional `hilbert_adjoint` carries the inner-product adjoint;
trailing payload axes pass through both. Constant, linear, positivity, and
conservation claims are certified when the transfer is constructed (positivity
only from sparse coefficients); `vertex_interpolation_transfer` builds the
fixed-width row-stencil form used by local refinement.

Tensor hp adaptation uses `FiniteElementHPTopology` as an allocated refinement
forest and `FiniteElementHPEpoch` as the immutable prepared snapshot. Isotropic
quad/hex h-refinement composes with anisotropic p, master-trace H1 constraints,
DG mortar interfaces, curved parent-map inheritance, role-specific transfers,
degree-bucket condensation/multigrid, and atomic solver promotion. Error evidence
and hp decisions remain separate; the marking decision is a discrete derivative
boundary.

Executable application namespaces live under `phydrax.applications`:
phase-field Allen-Cahn/Cahn-Hilliard, finite-strain crystal plasticity,
fixed-capacity barrier contact with conservative continuous step safety,
phase-field fracture, and fixed-crack XFEM classification/enrichment.

### L2 projection between non-matching meshes

Galerkin L2 projection of a scalar Lagrange field (continuous or discontinuous,
any degree, on affine triangles or tetrahedra) onto a second FE space on a
different mesh separates the target from the source:

- `prepare_l2_projection_target(target, field_name=...)` is the sole constructor
  of `PreparedL2ProjectionTarget`: the exact target mass `M_T`, its reverse
  Cuthill-McKee symbolic Cholesky plan and bound numeric factor (status and pivot
  diagnostics), a condition estimate, and the target DOF measures. One target
  artifact serves every source field and refinement onto that target; direct
  construction is refused so a same-shape factor from another geometry cannot
  be substituted.
- `prepare_l2_projection_transfer(source, prepared_target, refinement,
  field_name=...)` assembles only the mixed mass `B`, exactly on the overlap
  simplices of a successful `prepare_common_refinement(source.mesh,
  target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True))`. The
  primal is the `FiniteElementL2Projection` `M_T^{-1} B` and the pullback is
  `B^T M_T^{-1}`; trailing payload axes are solved as one multi-right-hand-side
  block.
- `refresh_l2_projection_target(prepared_target, moved_target)` refactors the
  numeric mass of the same field on moved geometry with an unchanged DOF
  structure (equal `dof_map_id`), reusing the symbolic plan and the compiled
  factorization and projection kernels; `structure_id` is unchanged and
  `target_id` follows the new geometry. A changed DOF structure raises
  `ValueError`.

Constants and linears are preserved when the refinement certifies every target
cell covered; the integral is conserved when every source cell is covered.
Failed or mismatched refinements (including a refinement of the geometry before
a refresh), unsupported elements, failed target factorizations, and direct
construction of prepared targets or projection operators raise `ValueError` or
`TypeError` at their owning boundary.

```python
prepared_target = prepare_l2_projection_target(target, field_name="u")
refinement = prepare_common_refinement(
    source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
)
transfer = prepare_l2_projection_transfer(
    source, prepared_target, refinement, field_name="u"
)
target_values = transfer.apply(source_values)

# `moved` is the same FE plan prepared on the target mesh moved with fixed
# topology: refactor numerically, then prepare transfers from its refinements.
refreshed = refresh_l2_projection_target(prepared_target, moved)
```


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


## CAD meshing and prepared cell maps

`phydrax.meshing.GmshProvider` owns Gmsh construction from a reopenable,
revision-checked `BRepModel` or solid `BRepSource`. Native specifications are separate from
`GmshOptions`. Audited results retain `CellMesh`, independent `CellGeometrySpec`,
source associations, compliance, and an execution trace; FEM consumes these
artifacts rather than a solver-specific meshing API. There is no query-
tessellation fallback or differentiable host remeshing. Provider preflight
rejects unsupported controls. See [Meshing](guides_meshing.md).

`prepare_finite_element_cell_map(discretization, block_index)` freezes the
coordinate element and gathers while keeping coordinates, cell indices and
reference iterates as JAX inputs. Integer indices retain block identity;
`block_index=None` prepares one homogeneous whole-support map through the same
canonical owner. Evaluation returns physical points, Jacobians, left inverses,
determinant/measure and validity margins; curved location reuses this map.

## Field views

`prepare_finite_element_field_reconstruction(discretization, field)` turns one
declared scalar or form field into `PreparedFieldReconstruction`.
Form fields use `FormFieldReconstructionKernel` with canonical physical proxy
and twist; scalar H1/L2 retains its existing value/derivative behavior.
Native tabulation and oriented DOF routes provide cell-sided values and physical
derivatives, with the declared regularity rather than an inferred global C0 claim.
The coefficient adjoint is the exact transpose scatter
(`reconstruction.transpose`, `reconstruction.duality_evidence`). A route with
any invalid query point has no transpose: both raise `ValueError` naming the
invalid points and their `FieldQueryStatus`.

Affine interval and n-D simplex queries use `PreparedSimplicialCellLocator`;
every containing cell is reported, never replaced by a nearest cell.
Quadrilateral/hexahedral/tensor cells use their native inverse-map locator.
Support geometry still follows exact region admission: warped hex needs an
explicit curved witness. Prepared quadrature/node interpolation also supports
physical derivatives with `derivative_axis=`.

A view binds coefficients to an explicit, equivalent `GeometryDomain`:

```python
reconstruction = phx.discretization.fem.prepare_finite_element_field_reconstruction(
    discretization, "u"
)
domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
view = phx.discretization.DiscreteFieldFunctionView(
    reconstruction, coefficients, domain, variable="x"
)
u = view.as_domain_function()
du = phx.operators.grad(u, var="x")
```

The support geometry of an affine simplicial mesh is derived from the mesh; an
explicit analytic `support_geometry` is accepted when the mesh evidences that
it covers it (vertices inside, equal measure). `C^0` gradients are defined in
cell interiors only: at a point shared by several cells the gradient raises
`ValueError` (`FieldQueryStatus.SIDE_REQUIRED`), and higher orders than the
tabulation provides are refused instead of being differentiated generically.
Facet gradients use a side-bound trace:

```python
owner = view.trace(facet_points, side="owner", cell_ids=owner_cells)
neighbor = view.trace(facet_points, side="neighbor", cell_ids=neighbor_cells)
flux_jump = phx.operators.grad(owner, var="x") - phx.operators.grad(neighbor, var="x")
```

`side="average"` averages every containing cell's limit. Views compose with
other domain fields through ordinary `DomainFunction` algebra; see
[Domains](guides_domain.md#discrete-field-views).

## Side traces, reaction fluxes, and imposition provenance

`discretization.prepare_side_trace(field, domain, rule=FacetTraceRule(...))`
prepares the exact trace of an identity-mapped scalar-basis H1 or L2 field (any
degree: simplex Lagrange, tensor GLL spectral elements, prisms; multi-block
meshes) on selected exterior or interior facets produced by the same
discretization. The sites are the owner cell's local-facet rule points: an
interior facet traced with `side="neighbor"` addresses the same physical points
through the dihedral map of the shared facet vertices, and the agreement is
verified on the host. Weights are the physical facet measure, and unit normals
point out of the traced side cell (`n ds = det(J) J^{-T} N_ref ds_ref`).
`quantity="normal"` and `"tangential"` contract vector fields against that
normal (`u . tau` with `tau = (-n_y, n_x)` in 2-D, `u - (u . n) n` in 3-D). The
route gathers each facet's closure DOFs (H1) or nonzero-trace cell DOFs (L2)
and never forms a coefficient-by-site matrix.

```python
rule = phx.discretization.FacetTraceRule("gauss-legendre", points=4)
trace = discretization.prepare_side_trace("u", boundary_domain, rule=rule)
values = trace.apply(coefficients)             # (facets, sites)
load_rows = trace.inject_load(density)          # sum_q w_q g_q phi_i(x_q)

flux = compiled.prepare_conormal_flux(trace)    # residual reaction
reaction = flux.evaluate(compiled.expand(solution))
impositions = compiled.boundary_impositions()
```

`CompiledFiniteElementProblem.prepare_conormal_flux(trace)` publishes the
residual reaction of the compiled physical operator: the full weak residual on
`trace.support_rows` of an H1 field of the problem (the field block of a mixed
form) on exterior facets. At a solution it equals the outward conormal flux
tested against the facet basis functions over their whole boundary support.
`boundary_impositions()` reports discrete Dirichlet constraints as strong rows,
boundary loads and SIPG Neumann data as natural facets, SIPG Robin data as
Robin facets, and SIPG Dirichlet (Nitsche) terms, exterior-facet residual
actions, and exterior functional terms as weak facets, in form-field, kind, and
source order. Plain `ConstraintMap`, periodic, and hanging-node constraints
restrict the space rather than impose a boundary law and are not reported.

`compiled.prepare_pointwise_flux(trace)` publishes the exact pointwise
conormal flux `q = n . K grad(u)` at the sites of a scalar value trace
(`representation="quadrature-values"`, `approximation="exact"`, outward from
the traced side cell, exterior or interior facets). `K` sums the form's
`DiffusionAction`/`TensorDiffusionAction` diffusivities over the actions whose
cell domain contains the side cell; the gradient is the discrete field's own
gradient on the whole side cell, gathered through a prepared route with an
exact scatter transpose. The flux is linear in the full coefficients. A form
with any other term acting on the field (besides mass, source, and boundary
terms) declares no boundary flux law and is refused, as are quadrature- or
facet-located diffusivities. The descriptor's `trace_degree` is the flux's
polynomial degree along the facets (for example `k - 1` for P_k simplices with
cell-wise constant `K`); callable diffusivities and side cells that are not
affine along a facet publish `None`.

`compiled.certify_flux_stability(flux)` returns a
`phydrax.discretization.TraceInverseEvidence` with, per facet, the sharp
constant `C_F = max ||q(v)||²_F / a_K(v, v)` over the side cell's local space.
`a_K` is the cell energy of the same diffusion terms on the owner's own cell
rules, `||.||_F` uses a Gauss–Legendre facet rule exact for the squared flux,
and each constant is the largest eigenvalue of the local pencil solved by the
native batched dense Hermitian eigensolver (the energy, which sees only the
symmetric part `S` of `K`, is deflated on the constants, which carry no flux).
For P1 it equals `|F| w.S^-1 w / |K|` with `w = K^T n` (`|F| n.K n / |K|` for
symmetric `K`). These are the flux and stability publications a Nitsche
transmission law consumes.

```python
pointwise = compiled.prepare_pointwise_flux(trace)
densities = pointwise.evaluate(compiled.expand(solution))   # (facets, sites)
stability = compiled.certify_flux_stability(pointwise)
stability.constants, stability.cell_multiplicity
```

Piola-mapped H(div)/H(curl) fields have no scalar-basis side trace and are
refused; their normal/tangential traces would be facet-moment maps, which are
not published. `side="average"` is refused: compose the owner and neighbor traces.

`phydrax.solver.coupling.VariationalComponent(name, compiled, field="u")`
publishes one scalar field of a compiled problem, with these traces, reaction
and pointwise fluxes, stability evidence, and impositions, to spatial coupled
assembly, for example a mortar transmission to a virtual-element region or a
Nitsche transmission to another finite-element region; see
[Spatial coupled problems](guides_numerical_interoperability.md#spatial-coupled-problems).

## Current limits

Execution remains single-device unless a caller supplies a JAX named-axis
collective context; no MPI runtime or mesh partitioner is claimed. Compatible
elements are triangle RT0/Nedelec0; HDG is lowest-order triangular primal HDG;
the legacy SIPG convenience remains scalar on one homogeneous 2-D polygon
block. Triangle adaptation remains conforming T3 refinement; tensor h-refinement
is isotropic and 2:1 balanced. DGSEM remains stationary-mesh only.
Search, active-set selection, marking, topology changes, and hp candidate
promotion are discrete derivative boundaries.
