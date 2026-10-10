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
separately. `form_element` supplies trimmed/full simplex, tensor-trimmed,
prism-trimmed, and rational pyramid-trimmed families. Prism and pyramid cells
select their canonical compatible family when `family="trimmed"` is requested.
Physical flux/density requires explicit twist, and ambiguous degree1 2-D maps
require circulation or flux explicitly.

Prism spaces are the triangle–interval FEEC product, not replicated scalar
nodal functions. Pyramid spaces use the trace-constrained generalized Whitney
complex of the five rational vertex coordinates: triangular traces are
P_r^-Λk, the quadrilateral trace is Q_r^-Λk, and the terminal space is the exact
exterior image of the flux space. This is a declared enriched rational
complex, not a claim of the minimal pyramid family: its degree-wise dimensions
are `(5, 8, 7, 3)` at order one and `(14, 31, 30, 12)` at order two.
It contains scalar polynomials through degree r and k-form polynomials through
degree r−1. Entity moments are independently owned; a shared entity explicitly
converts its scientific test basis and full orientation action. Equal numbers
of moments do not establish compatible coefficient identities.
Mixed prism/tetrahedron plans admit these canonical H(curl)/H(div) form
elements as well as nodal H1 and supported cell-local DG fields. Admission
checks the owning cell family, approximation order, degree/twist, declared
proxy and actual tabulator/basis source identity before shared moments are
numbered. A raw vector tabulator or merely attached form metadata is not a
replacement for that canonical contract.

Body moment tests form a descending exterior-closed complex. The top scalar
constant owns total flux; the remaining scalar tests are boundary-vanishing
bubbles. Each lower-degree test space includes their exact exterior
derivatives and is completed by zero-trace bubble forms. Stokes' identity
therefore makes canonical interpolation commute with exterior differentiation
on smooth fields, not only on represented polynomials. Body labels identify
these actual stored test-source rows; they are not guessed monomial modes.

`FormBasis.entity_kind` and `entity_basis` identify each native entity chart.
`functional_weights_at` takes cubature in that chart, including actual physical
reference coordinates for prism/pyramid bodies. `component_expressions` and
`functional_density_expressions` expose the exact owning physical-reference
polynomial/rational source for certified integration. Pyramid generator source
fractions and body-test fractions are immutable preparation; numerical
generators, moment-test coefficients, dual coefficients, rank, condition, and
native solve-error/status diagnostics remain dynamic array leaves. Prepared
hybrid duals refuse deficient rank, condition above 1e12, or the 24,000,000
coefficient-entry work preparation bound before publication. This entry-work
bound is not a claim about elapsed time or dense-factorization flop counts.
Exact source extraction combines generator polynomials with the actual dual
dyadics before applying the collapsed rational chart. Each final DOF owns one
temporary preparation lifetime; its returned numerator/denominator source is
charged as retained data. Repeated identical collapse denominators therefore
do not consume the caller's storage allowance merely because there are many
generators.

`FiniteElementFieldSpec` supports replicated component shapes and multiple named
fields. One `CompiledFiniteElementProblem` owns the ordered product space and
scatters every coupled term directly to its output residual block.

`phydrax.discretization.CellGeometrySpec` assigns an independent coordinate element and
geometry DOF map to every block. This permits curved P2 geometry with a lower-
or higher-order field element.
Shared H1 edge nodes are ordered from their actual reference-node corner
weights, not their storage positions. The equispaced coordinate lattice and
solution-node families can therefore reuse this same DOF-lowering owner.

For a `CellMesh` carrying `PeriodicMeshTopology`, high-order H1 nodes and
compatible H(div)/H(curl) moments are numbered on the authored quotient entity
orbits. Relative image shifts retain distinct winding entities even when their
representative vertices coincide. Compatible seam maps compose the owning
`FormBasis` canonical entity charts with the declared corner permutation;
oriented face loops are not themselves moment-coordinate charts. Proper
rotations and commuting axial translations therefore transform physical
vector traces rather than equating their Cartesian components. Invalid
transformation cycles, nonfinite lifted coordinates, and image shifts outside
the persistent integer representation are refused without mutating the mesh.

`FiniteElementDeRhamComplex` provides exact sparse degree maps, metric-only Gram
Hodges, reconstruction/traces and commuting `ComplexMap` transfers. Full-family
complex order is its top polynomial order, so degree k uses P_(order+n−k)Λk;
trimmed complexes keep the same order in each degree and select the compatible
family of every block, including mixed simplex/tensor/prism/pyramid meshes.
Material weights are separate coordinate forms, not part of an SPD metric Hodge.
`hodge_solve` accepts a native `LinearSolvePolicy`, never a string selector.
Hiptmair–Xu uses native vector/potential corrections; high-order preparation
composes the low-order auxiliary plan instead of another private solver.


## Geometry

`CellGeometrySpec` is the coordinate owner; a field element never supplies
geometry implicitly. Physical points, tangent metrics, measures, normals where
defined, physical gradients, and Piola-mapped compatible bases are evaluated in
pure JAX. For a square Jacobian J, the runtime factors J itself, reports
`abs(det(J))`, and derives the inverse metric without first conditioning the
squared Gram matrix. For an embedded cell it instead retains
`G = J.T @ J`, the physical density `sqrt(det(G))`, and the tangent gradient
map `J @ inverse(G)`. The native small/dense solve result owns rank,
conditioning, determinant, and success evidence. A finite coordinate array or
positive corner area cannot override a failed tangent metric.

This is the metric used by embedded interval, triangle, and quadrilateral FEM
mass, diffusion, functionals, and transfer content. It is not an ambient-volume
surrogate and does not imply a unique exterior normal for arbitrary codimension.
Side traces that require a unique normal therefore retain their separate
full-dimensional/codimension contract.

`prepare_runtime` creates a fixed-topology numeric realization:

```text
runtime = discretization.prepare_runtime(new_coordinates, numeric_version="moved")
context = phx.equations.FiniteElementExecutionContext(runtime)
residual = compiled.residual(state, context)
```

Coordinates flow through residuals, sparse refresh, functionals, DAE mass
operators, and shape derivatives. Connectivity, coordinate-element structure,
or DOF-count changes require a new plan.

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
route refines them conformingly and coarsens complete bisection patches.
`FiniteElementTopologyTransaction(certify, fields=...)` declares the
finite-element space of every accepted field and `execute(accepted, mesh,
adaptation)` moves each field by its own family, never by array width: on a
nesting adaptation (`phydrax.solver.refinement_parent_cells`) through
`prepare_nested_field_transfer`, otherwise every field through the Galerkin L2
projection on a certified common refinement
(`prepare_projection_field_transfer`), compatible fields through their Piola
maps. `MaterialTopologyTransferResult` retains the transferred
`MaterialTransaction` together with every prepared conservative remap owner; a
material array with a plausible shape is not transfer evidence.

The mesh, reprepared discretization, fields, materials, history, and solver
state are staged as one `phydrax.lifecycle.CompositionRebind`; failed transfer
evidence, material transfer, certification, or independent physical reanalysis
returns the complete accepted state with the refused receipt. A published
rebind's receipt ID becomes the promoted state's `transition_id`, which
checkpoints and restart records carry.

`prepare_nested_field_transfer(source, target, parent_cells, field_name=...)`
reconstructs the parent basis in every child with the target Piola-mapped basis
(identity for H1/L2 Lagrange, covariant for planar and tetrahedral Nedelec,
contravariant for tetrahedral and planar RT/BDM), which equals applying the target DOF functionals
(nodal values, edge circulations, face flux moments) where the target space
contains the source space. `FiniteElementTransferEvidence` certifies witness
containment, space reproduction, shared-DOF continuity (trace continuity and
orientation), and curl/divergence commutation; a failed certificate claims
nothing and its epoch transition refuses the rebind.
Tetrahedral H(curl) fields use
`form_element("tetrahedron", 1, order, family="trimmed", proxy="circulation")`.
The canonical form basis owns edge/face moments and their full orientation
transformations; meshing does not introduce a separate lowest-order space.

Mapped simplex and tensor form fields also accept `geometry_transition` or
`coarsening_witnesses` naming an exact complete reference partition. Refinement
applies the child's circulation/flux functionals to the pulled-back source form.
Coarsening integrates coarse entity functionals over their actual fine entity
partitions, retaining independent coordinate and field coefficient supports.
Its compatible projection preserves boundary moments and, when needed, solves
for cell-interior moments against the descending exterior-derivative complex.
This is neither an interpolation transpose nor an orthonormal restriction.
Full entity transformation matrices are solved
with prepared native factors; degree, twist, physical proxy, and scientific
source/quotient identities cannot be replaced by matching array dimensions or
nearby points. The certificate checks every source column of exterior-derivative
commutation and shared-entity agreement. `maximum_work` and
`maximum_storage_bytes` bound cumulative compatible preparation, with charged
work/storage estimates, local rank, condition, and solve defects in its evidence.
Nonlinear or rational root coordinate maps remain exact source restrictions.
Source-authored bilinear quad reference charts additionally retain their full
coefficient action, whole-chart Jacobian bound, source bank and scientific roots;
corner proximity does not supply a witness. Prism/pyramid compatible
source components retain their actual product/rational form identity through
the same moment and Piola owners; a rational source integral requires exact
denominator cancellation or an owning certified integration error bound.
Pyramid body integrals compose the source components and actual functional
density factors into the owning collapsed cube chart before expanding their
contraction, then apply its Jacobian once. This retains the same physical
moment while avoiding needless rational coefficient expansion within the
original preparation work and storage limits.
Common-piece moment integration keeps the source and target reference maps
independent. On an owning simplex integration entity it pulls back the actual
source form and the target complementary test form, including a genuinely
rational projective target map. Exact entity inclusion and sign-definite
Jacobian bounds establish orientation; an affine map through the same corners
is not a replacement. Rational remainder uncertainty is retained in the moment
matrix and must fit the unchanged consumer error and resource policy.
Twisted common-piece fluxes and their top-density companions also retain the
full-cell orientation bundle: the relative multiplier is the product of the
source and target full-chart Jacobian signs, each proved nonzero over the
whole simplex. Entity orientation alone does not establish this bundle; a
reversed source chart uses the physical absolute-determinant flux law.
`FiniteElementFieldTransfer.epoch_transition(...)` returns a
content-ledger `TopologyEpochTransition` for conservative transfers with positive
DOF measures and a `FieldEpochTransition` otherwise; both expose the
`composition_transport` physical-remap transport.

`FiniteElementTopologyTransfer` stores the primal coefficient map (target DOFs by
source DOFs) either as one `SparseLinearMap` with O(targets x stencil width)
memory or as a linear operator whose action couples every DOF. `apply` is the
primal transfer, `pullback` is its algebraic transpose for residual and load
duals, and an optional `hilbert_adjoint` carries the inner-product adjoint;
trailing payload axes pass through both. Constant, linear, positivity, and
conservation claims are certified when the transfer is constructed (positivity
only from sparse coefficients) and `semantics` declares its checked meaning;
`vertex_interpolation_transfer` builds the fixed-width row-stencil form used by
local refinement.

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
any degree) or a Piola-mapped H(curl)/H(div) field (Nedelec, RT, BDM) on affine
triangles or tetrahedra onto a second FE space on a different mesh separates the
target from the source:

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
  block. Compatible fields pair covariant/contravariant Piola values of both
  owning cells on every overlap simplex. `prepare_projection_field_transfer`
  certifies their `reproduction` of constants plus `x` (H(div)) or rotations
  (H(curl)) and the conserved total vector `content`, and reports the target
  `solve-residual`, the sampled cellwise `commuting` defect, and the
  `divergence-content`/`curl-content` (net boundary flux or circulation) as
  `FiniteElementTransferEvidence.estimates`.
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
