# Numerical interoperability

Phydrax composes numerical methods by binding existing scientific owners, not by
replacing them with a universal discretization, region, or solver hierarchy.
Geometry, meshing, discretizations, boundary-integral operators, native linear
and nonlinear solvers, temporal coupling, model components, measurements, and
lifecycle owners each keep their mathematical meaning. This guide describes the
prepared bindings that let those owners interoperate, and the exact scientific
limits of each binding.

## Ownership map

| Owner | Responsibility |
|---|---|
| Geometry, B-Rep, compartments, multiregion surfaces | Physical entities, incidence, orientation, lineage, sheet views and their normal convention |
| Analytic subdomain covers (`phydrax.domain.decomposition`) | Patch and pairing IDs, coordinate maps, left-to-right pairing normal, cover revision and audits |
| `phydrax.meshing` | Exact scopes, geometry associations, carriers, assemblies, adaptation, distribution, mesh interface attachments |
| `phydrax.discretization` | Supports, DOF layouts, field spaces, local actions, queries, measures, transfers |
| Equation compilers (FE, VEM, conservation, strong form) | Native approximation, stabilization, closure, boundary semantics |
| Layer-potential owners | Kernels, boundary spaces, singular quadrature, jumps, exterior conventions |
| `phydrax.linalg`, `phydrax.nonlinear` | Operators, solves, linearizations, pairings, preconditioning, evidence |
| `phydrax.solver.coupling` | Interface bindings, spatial coupled preparation, and the canonical partitioned temporal runtime |
| Native DAE and temporal owners (`phydrax.solver`) | Stage coordinates, histories, consistency, regularity, supported integration methods |
| `phydrax.system_modeling` | Acausal connector equations, lumped constitutive and DAE semantics |
| Model/component layer, training kernel | Ports, authority, array roles, derivative admission, accepted updates |
| Measurement, observation, UQ | Quantity and sampling meaning, uncertainty, likelihoods |
| ROM, control, stochastic | Residual-provider, transition, observation, linearization, and acceptance contracts |
| `phydrax.lifecycle`, execution | Durable identities, checkpoints, composition rebind transactions, worksets, sharding, resource evidence |
| External runtimes (`phydrax.interchange`) | FMI co-simulation and pinned external processes behind an explicit host boundary |

## Interface bindings

An `InterfaceBinding` names one physical interface explicitly and binds it to
the geometry revision it lives on and to every endpoint's native support. It
links existing identities; it never replaces them with a common region
ontology, and it never infers identity from matching names, shapes, or entity
dimensions.

```python
from phydrax.solver.coupling import InterfaceBinding, InterfaceEndpoint, InterfaceSource

binding = InterfaceBinding(
    "material-wall",
    InterfaceSource(model.source_id, model.source_revision, (wall_edge,)),
    "two-sided",
    (
        InterfaceEndpoint("spectral", minus_attachment, fields={"value": "u-sem"}),
        InterfaceEndpoint("virtual", plus_attachment, fields={"value": "u-vem"}),
    ),
)
binding.require_current(assembly)
```

A binding records:

- the explicit `interface_id` and the authoritative `InterfaceSource`
  (`source_id`, `source_revision`, and the owner's `entity_ids`);
- the `incidence`: `"two-sided"`, `"junction"`, `"overlap"`, or `"embedded"`;
- one `InterfaceEndpoint` per incidence, with an explicit `role`, the attached
  field identities by field role (`fields`), and one owner attachment carrying
  the support revision and its witness;
- a deterministic `binding_id` over all of the above.

| Attachment | Owner and native revision | Witness | Orientation |
|---|---|---|---|
| `phydrax.meshing.MeshInterfaceAttachment` | `MeshPart` name and `part_id`; exact part scopes and certified patch/zone/label IDs | Certified `GeometryAssociation` of the part's carrier, classifying every attached entity on the declared entities within a tolerance | Side of the B-Rep normal (3-D face normal; 2-D edge tangent rotated clockwise) from the association orientation and the attached cells |
| `PairedSupportAttachment` | `SubdomainCover.cover_id` and `SubdomainCover.revision`; pairing and patch IDs | Fresh verified `PairedSupport.audit` on supplied support points, plus a sampled sign audit of the pairing normal against the patch supports | Left patch is minus, right patch plus, when the pairing has a normal |
| `SheetViewAttachment` | `MultiRegionSheetViews.views_id` and the current sheet geometry | The owner's face selection `view_id` | View normals point out of `region_ids[0]` (minus) into `region_ids[1]` (plus) |

Incidence semantics are distinct:

- **Two-sided** endpoints are ordered `(minus, plus)`; the normal points out
  of the minus side. Both must be sided and attach the same entities. A
  declared minus endpoint that lies on the plus side is refused as reversed.
  `side(InterfaceSide)` and `side_of(role)` resolve sides.
- **Junction** endpoints (three or more) keep their declared incidence order.
  A junction never expands into pairwise laws; a coupling law must address all
  incidences explicitly. A selected pair at a junction is its own two-sided
  binding.
- **Overlap** endpoints are codimension-zero and unsided: an overlap never
  acquires a normal. They are ordered by role.
- **Embedded** (mixed-dimensional) incidence is `(host, embedded)` with the
  embedded support at higher codimension. `meaning` states whether the host
  field is traced (`"trace"`), averaged (`"average"`), or integrated
  (`"integral"`) over the embedded support; averages and integrals name their
  measure with `measure_id`. Equal entity dimensions are not required here,
  while the point-pair `MeshCoupling` overlays keep their own requirement.

`require_current(*owners)` accepts `MeshPart`, `MeshAssembly`,
`SubdomainCover`, and `MultiRegionSheetViews` values and refuses a missing
owner or any changed owner revision: moving or remeshing a part, changing a
cover, or moving a sheet invalidates the binding until its endpoint is
re-attached. The interface ID survives rebinding. Computational ownership is
not part of a binding, so redistributing a part (`MeshDistribution`) keeps the
binding current and its identity unchanged.

Every law that acts on a binding audits each of its prepared sides against the
endpoint's attachment at `prepare_coupled_problem` (`AbstractCouplingLaw.prepare(
components, interface_owners)`, through `InterfaceEndpoint.require_side`): the
side's trace sites must lie on the attached support and its outward normal must
lie on the attached side of the authoritative normal. Mesh attachments compare
facet entities exactly when the side acts on the part's own entity set and
measure the sites against the attached simplices within the attachment
tolerance; paired supports probe the patch supports `tolerance` behind and
ahead of every site along its outward normal (a sampled audit); sheet views
measure the sites against the sheet triangles within the attachment's
`tolerance`. A side on another interface, or across the declared one, is
refused with a `ValueError`.

Current limits: mesh attachments require certified cell carriers, because only
cell results carry geometry associations. Compartment meshes publish zones and
interface patches but no geometry association, so compartment-to-mesh binding
is not published. The sign of a `PairedSupportAttachment` normal is audited on
the supplied points by probing the patch supports `tolerance` (which must be
positive) behind and ahead of it; sided pairings whose patch supports overlap
at the cut cannot witness their side and are refused. The audit is sampled.
Analytic maps authored as opaque Python callables are identified by their
declared IDs only. The example `examples/numerical_interface_binding.py` binds
two independently meshed parts of one B-Rep and shows stale-revision refusal.

## Prepared field queries and side actions

Observations and interface couplings read discretized fields through prepared
products owned by `phydrax.discretization`. Preparation locates points and
tabulates facet routes once on the host; execution only gathers, contracts,
and scatters with changing coefficients.

**Point queries.** `PreparedFieldReconstruction.prepare_query(points, *,
derivative, side, cell_ids, coverage)` returns a `PreparedFieldQuery` bound to
the fixed points, coordinate derivative, trace side, and coverage policy. It
retains the owner route, the pointwise `FieldQueryEvidence` of every requested
point, and the admitted subset. `coverage="complete"` refuses any invalid
point with its status; `coverage="masked"` admits only valid points, so
`apply` never reports a value at a rejected point, and `complete` tells a
coupling law that requires full support to refuse the query. A
coefficient-linear query exposes `transpose` (the exact scatter) and
`as_linear_operator(coefficient_pairing=..., value_pairing=...)`, whose
`transpose_mv` is the coordinate transpose and whose `adjoint_mv` is the
Hilbert adjoint relative to the declared Riesz pairings. The operator's
coefficient space uses the reconstruction's native `coefficient_dtype` (complex
modal storage for spectral fields; real coefficients in the point precision
when none is declared). A real-valued query of complex coefficients, such as
real-output spectral synthesis `Re(R c)`, is only real-linear, so its operator
acts on the realified space of shape `(*coefficient_shape, 2)` holding
`(Re c, Im c)`; its transpose and adjoint are exact for real pairings on that
space. A nonlinear
reconstruction (for example WENO-Z) refuses a transpose and exposes
`linearize(coefficients)`, a `PreparedLinearization` at that coefficient
state. `require_reconstruction` refuses a reconstruction prepared on another
geometry revision: coefficient refresh reuses the route, geometry refresh
requires a new query.

```python
query = reconstruction.prepare_query(sensors, derivative=(1, 0))
for state in coefficient_history:
    slopes = query.apply(state)          # no relocation
pulled_back = query.transpose(residual)  # exact coordinate dual
```

**Curved-cell location.** Finite-element point location prunes candidate cells
with boxes of the Bernstein control net of each polynomial coordinate map, which
enclose the whole mapped cell, so points where a curved cell bulges past its
coordinate nodes are still located.

**Projection labels.** `PreparedFieldReconstruction.approximation` is
`"exact"` when the reconstruction evaluates the discrete field itself and
`"h1-projection"` or `"l2-projection"` for the labeled virtual-element interior
channels of `phydrax.equations.prepare_virtual_element_field_reconstruction(space,
channel=...)`; the label reaches every query. Finite-element
(`fem.prepare_finite_element_field_reconstruction`) and explicit polygon H1
(`prepare_explicit_polygon_h1_field_reconstruction`) reconstructions are exact. A virtual-element exact edge trace
and its projected interior polynomial remain different capabilities.

**Side traces.** A discretization publishes geometric traces on selected
exterior or interior facets through `prepare_side_trace(field_name, domain, *,
rule, quantity, side)` (the `SideTraceProvider` protocol), returning a
`PreparedTraceAction`. Its `SideActionDescriptor` records the owner and field
space, the quantity (`"value"`, `"normal"`, `"tangential"`), representation,
orientation, trace side, facets, geometry revision, and exactness evidence
(`approximation`, trace degree, facet-rule exactness). The action keeps three
distinct maps:

- `apply`: coefficients to trace values at the facet `sites`;
- `dual_pullback` / `dual_pullback_operator()`: the exact coordinate transpose
  from trace covectors to the owner's residual rows, and `inject_load(g)`, the
  pullback of a load density through the facet measure `sum_q w_q g_q v(x_q)`;
- `hilbert_adjoint(coefficient_pairing)`: the adjoint relative to an
  explicitly declared coefficient Riesz map and the facet measure.

Owner and neighbor traces of an interior facet use the same physical sites, so
jumps and averages are formed pointwise; `support_rows` names the coefficient
rows each trace depends on. `FacetTraceRule` declares the facet quadrature
(Gauss--Legendre or tensor Gauss--Lobatto--Legendre) and its exact degree.

**Conormal fluxes.** A flux comes from the physical operator, never from a
value trace. Compiled physics owners publish `PreparedFluxAction` values
through `prepare_conormal_flux(trace)` (the `ConormalFluxProvider` protocol):
the `"residual-reaction"` representation restricts the owner's full weak
residual to the trace's support rows (the variationally consistent flux,
labeled `"variational-reaction"`), and the virtual-element compiler also
publishes `prepare_projected_flux(trace)`, a quadrature-point flux of the H1
projection labeled `"h1-projection"`.

**Imposition provenance.** `boundary_impositions()` (the
`BoundaryImpositionProvider` protocol) reports every strong, natural, Robin,
and weak boundary law of a compiled owner as `BoundaryImposition` records with
their rows or facets; `BoundaryImposition.overlaps(trace)` lets a coupling
refuse a second law on an already imposed side.

The example `examples/prepared_field_observations.py` observes a refreshed
finite-element field through one prepared query, pulls a sensor residual back,
and reports a rejected sensor under masked coverage.

**Boundary-element trace spaces.** Boundary-integral owners have no volume
facets, so they publish their native boundary coefficient spaces instead of
facet side actions. `ScalarBoundarySpaces2D.cauchy_trace_capability()` and
`ScalarBoundarySpaces3D.cauchy_trace_capability()` return a
`CauchyTraceCapability`: P1 Dirichlet and DP0 Neumann
`BoundaryTraceSpaceCapability` records with the owner's native coordinate
space, its arc-length or area Gram pairing and mass, the Neumann orientation
out of the declared `interior`, the geometry revision, and the sparse
Dirichlet-to-Neumann-dual `duality`. `RWGSurfaceCurrentSpace3D.trace_capability()`
and `BuffaChristiansenDualSpace3D.trace_capability()` publish tangential
currents as distinct `H^(-1/2)(div_Γ)` records that are refused as scalar
Cauchy data. See [Boundary layer potentials](guides_boundary_layer_potentials.md#boundary-trace-space-capabilities).

**Finite-difference/SBP and global spectral traces.** Tensor-grid owners
publish exact boundary traces on the faces of bounded axes only; periodic axes
have no face or outward normal and are refused. A finite-difference trace
(`PreparedFiniteDifferenceDiscretization.prepare_side_trace(..., norm=SBPGridNorm(...))`)
is the nodal restriction on (face, boundary node) facets measured by the
tangential factors of the declared tensor SBP norm, whose `SBPClosureEvidence`
sets `quadrature_exact_degree`; its Hilbert adjoint uses
`norm.pairing(layout="rows")`. A global spectral trace
(`TensorSpectralDiscretization.prepare_side_trace`) synthesizes the field on
each face through a sum-factorized `SpectralFaceRoute`, at the native tangential
nodes with Clenshaw--Curtis or Gauss weights or at the points of a
`FacetTraceRule`. Neither owner infers a conormal flux. Interior point queries
use `prepare_finite_difference_field_reconstruction` (declared multilinear or
B-spline interpolation) and `prepare_spectral_field_reconstruction`.

**Isogeometric queries and traces.** `iga.prepare_isogeometric_field_reconstruction`
evaluates a single-patch field at physical points by inverting the runtime NURBS
map (clamped Newton from overlay-cell seeds, in JAX), with exact values and first
physical derivatives and pointwise `OUTSIDE_SUPPORT`/`LOCATION_FAILED` evidence;
curved patches bind to an explicit support region verified against the patch
boundary, interior, and measure. `PreparedIsogeometricDiscretization.prepare_side_trace`
traces patch-boundary faces and the interior knot faces between overlay cells
with rule weights times the surface Jacobian and outward normals, on the compiled
field's public control layout `(*control_shape, *component_shape)`; its route
flattens the control axes exactly as `LocalFieldBinding.flatten`, and
`support_rows` index those flattened controls (`trace.row_shape`,
`trace.flatten_rows`). Multipatch interfaces are not prepared domains. See
[Isogeometric analysis](guides_isogeometric_analysis.md#prepared-field-queries-and-side-traces).

**Finite-volume face traces.** Structured, unstructured, and triangular
finite-volume discretizations publish `integration_domain("exterior_facet" |
"interior_facet")` in their canonical face order and `prepare_side_trace(...,
reconstruction=None)`. The descriptor's `representation` states the face
semantics: `"cell-average"` repeats the side cell average at every site (the
first-order face state), `"face-state"` evaluates a linear reconstruction
(unstructured k-exact polynomials, triangle k-exact and unlimited MUSCL
planes, structured unlimited MUSCL) through per-facet stencil routes with
exact transposes. Owner and neighbor sides of an interior face share sites and
carry opposite unit normals; weights are the physical face measure. WENO-Z,
structured WENO, and limited MUSCL face states are nonlinear:
`prepare_nonlinear_face_trace` returns a `PreparedNonlinearFaceTrace` whose
derivative contract is `linearize(state)` only. Point queries of the same
reconstructions use `prepare_finite_volume_field_reconstruction`. See
[Unstructured finite volume](guides_unstructured_finite_volume.md#face-traces-and-field-views).

Finite-element form reconstruction publishes Piola-mapped H(curl)/H(div) point
queries and tangential/normal side traces through `FormValueSpec`; embedded maps
require declared coorientation. Point-cloud owners publish no side actions and
boundary-element owners publish native trace spaces instead of facet actions.
Finite-volume polyhedral/seam faces and bounded-edge-reaching structured stencils
retain their owning refusal contracts.

## Spatial coupled problems

One steady spatial problem often spans several native owners: a
finite-element region next to a virtual-element region, or a volume owner next
to a boundary-integral product. `phydrax.solver.coupling` assembles such a
problem from what each owner already publishes. It never reassembles an
owner's operator, never merges compilers, and contains no formula specific to
a pair of methods.

```text
components (AbstractSpatialComponent) + bindings (InterfaceBinding) + laws (AbstractCouplingLaw)
  -> CoupledProblemPlan -> prepare_coupled_problem -> PreparedCoupledProblem
  -> solve_coupled_problem -> CoupledSolution (native status + certificates)
```

### Components

An `AbstractSpatialComponent` publishes one native owner to the assembler:

- named **state blocks** (the columns it is solved for) and **row blocks** (the
  residual rows of its own equations), paired one-to-one by position with equal
  sizes;
- whether those blocks are the owner's `"full"` coordinates or its
  constraint-`"reduced"` solve coordinates (`ComponentSpace`);
- its **fields** (`ComponentField`): the full coefficient space of each field
  that a law may address, its constraint map `P` and lift `g` (the full field is
  `P z + g`, rows are pulled back by `P^T`), and the free rows of a plain
  row-selection chart;
- `residual(state, args)` (the original equations), `linear_operator(args)`
  (the native affine operator, or `None` for a residual-only owner),
  `lift(field, args)`, `nullspace(args)` (kernel columns per state block),
  `boundary_impositions()`, and `field_space_id(field)`.

`AbstractTraceComponent` additionally publishes `prepare_side_trace` and
`prepare_conormal_flux` (see
[side actions](#prepared-field-queries-and-side-actions)); laws that act
through facet traces require it and refuse other components with `TypeError`.

| Component | Owner | Blocks |
|---|---|---|
| `VariationalComponent(name, problem, *, field, affine=True, location_policy=None)` | One scalar field of a `CompiledFiniteElementProblem` (Lagrange and spectral elements, explicit polygon H1, single-patch isogeometric) or a `CompiledVirtualElementProblem`; mixed forms are refused | State block = the owner's solve coordinates (reduced by its Dirichlet chart when it has one); row block = its weak residual rows; kernel = the right nullspace its native `linear_system` declares; `affine=False` publishes no linear operator, so the problem is solved by Newton on the owner residual |
| `GalerkinBoundaryComponent(name, galerkin, *, far_field, far_field_tolerance)` | `ScalarLaplaceGalerkin2D` | Unknowns `"conormal"` (DP0 `q`) and `"far_field_constant"` (`c`); rows `"exterior_boundary_equation"` and `"total_conormal"` of the exterior relation; `dirichlet_operator()` is the `M/2 - K` column that a coupling law applies to the trace it supplies; `far_field` declares a `"bounded"` (default) or `"decaying"` exterior |
| `ReducedComponent(name, galerkin)` | A `phx.rom.FullResidualGalerkin` over a `ComponentResidualProvider` of one single-field trace component | Reduced trial coordinates and Galerkin rows `V^T R`; traces, fluxes, and reconstruction act on the reconstructed full field (see [Reduced-order components](#reduced-order-components)) |
| `ScalarLaplaceFEMBEMComponent(name, product)`, `ElasticityFEMBEMComponent(name, product)` | A prepared matching 3-D FEM–BEM product | The product's own named blocks, operator, and right-hand side (see [Existing 3-D FEM–BEM products](#existing-3-d-fembem-products)) |

`GalerkinBoundaryComponent` and the FEM–BEM product components publish no facet
traces; they are coupled only through laws that use their own boundary trace
spaces or named blocks. A new numerical method joins by publishing its own
component subclass; the assembler does not change.

### Contributions and endpoint spaces

A law lowers to typed contributions that state exactly which coordinates they
read and which rows they write. A `ContributionEndpoint(owner, block, *,
space)` is either

- `"full"`: `block` names a component **field**. The assembler composes the
  component's constraint map exactly once, `P z + g` on the source side and
  `P^T` on the row side, so a law never sees reduced coordinates; or
- `"reduced"`: `block` names a native block in solve coordinates, either a
  component block or a block the law owns (`LawBlock`, for example a mortar
  multiplier), used without further composition.

| Contribution | Meaning |
|---|---|
| `LinearContribution(target, source, operator, ...)` | Adds `operator(source)` to the target rows; `operator.source`/`operator.target` must equal the resolved source space and target row space (`DualSpace` of a full field, or a row block). A mismatch is refused as a row-space mismatch. Sparse and matrix-free operators are kept as given |
| `LoadContribution(target, values, ...)` | State-independent rows; the residual convention is `R = A u - b`, so a load publishes `-b` |
| `ResidualContribution(targets, sources, residual, *, affine, ...)` | Law-owned residual term; affine terms enter the linear assembly through their exact linearization, nonaffine terms make the problem nonlinear |
| `EliminationContribution(eliminated, retained, rows, columns, relation, ...)` | Explicit relation `u_e[rows] = E u_r[columns]` on free full rows of the eliminated field; those rows leave the solve coordinates and their residual rows are summed into the retained rows by `L^T` |

Every contribution carries its `law_id` and `imposition_id`, and each law also
publishes the facets and rows it claims (`LawImposition`). Preparation refuses
two laws on the same facets of one field, a law on facets where the owner
already imposes a natural, Robin, or weak boundary law, and an interface facet
whose whole trace the owner imposes strongly: one boundary law is imposed once.

### Laws and impositions

A law states physics; an imposition states how the physics enters the discrete
problem. `ScalarTransmissionLaw(law_id, binding, (minus, plus), imposition)`
states continuity of a scalar potential and balance of its conormal flux,
`u_minus = u_plus` and `q_minus + q_plus = 0`, on a two-sided
`InterfaceBinding`. Each `TransmissionSide(role, component, field, domain)`
names a binding role in the binding's `(minus, plus)` order, a trace component
field, and the component's exterior-facet domain on the interface; each
endpoint must bind `fields={"value": component.field_space_id(field)}`. Values
and loads flow only through the owners' side traces and their exact dual
pullbacks, and fluxes come from the owners' conormal-flux publications.

- **`MatchingElimination(eliminated=role)`** requires coincident facet
  partitions and an explicit basis relation `E`, computed from both owners'
  trace routes at the common quadrature (never from equal node coordinates).
  The eliminated side's trace must be unisolvent, and the relative relation
  residual `||T_e E - T_r||` must stay below `relation_tolerance`, which proves
  that the two trace spaces coincide. Only free eliminated rows are replaced;
  a strongly constrained eliminated row (for example a crosspoint carrying the
  owner's Dirichlet value) keeps its native constraint, so preparation refuses
  an elimination whose strong rows relate to free retained rows (continuity
  would never be imposed there): eliminate the other side or use a mortar or
  Nitsche imposition. `EliminationEvidence` records the
  eliminated rows, the retained columns, the relation residual, the trace rank
  and degrees, and the coverage.
- **`MortarImposition(MortarMultiplier(family, side=role, degree=None))`**
  declares the multiplier space: `"side-trace"` (the trace basis of `side` on
  its free rows) or `"discontinuous-polynomial"` (Legendre polynomials of a
  declared degree on every facet of `side`). The multiplier is the outward
  conormal flux of the minus side; the law owns the blocks `(law_id,
  "multiplier")` and `(law_id, "constraint")` and adds the rows
  `R_minus - B_minus^T l`, `R_plus + B_plus^T l`, and
  `-B_minus u_minus + B_plus u_plus`. Preparation computes the singular values
  of `[B_minus, -B_plus]` on the free trace rows and the discrete L2 inf-sup
  constant, and refuses a numerical rank below the multiplier dimension
  (`sigma_min <= rank_tolerance * sigma_max`) or an inf-sup constant below
  `minimum_inf_sup`. Dependent constraints are never dropped silently and a
  multiplier family is never selected by size. `MortarEvidence` records the
  family, side, degree, dimension, free trace rows, numerical rank, extreme
  singular values, inf-sup constant, trace degrees, quadrature degree, and
  coverage. `max_evidence_entries` bounds the dense host evidence.

### Nitsche imposition

`NitscheImposition(variant, *, penalty_factor, weights=(0.5, 0.5), quadrature,
max_evidence_entries)` imposes the same `ScalarTransmissionLaw` weakly through
the owners' exact pointwise conormal fluxes, with no law unknowns. With the
outward fluxes `q_minus`/`q_plus`, the averaged flux leaving the minus side
`{q} = w_minus q_minus - w_plus q_plus`, and the jump `[u] = u_minus - u_plus`
at the common interface quadrature, the law adds

```text
-∫ {q(u)} [v] - θ ∫ {q(v)} [u] + ∫ γ [u] [v]
```

to the owners' weak rows, with `θ = 1` for `"symmetric"` (adjoint consistent)
and `θ = -1` for `"nonsymmetric"`. The flux terms come from each owner's
`prepare_pointwise_flux(trace)` (densities at its trace sites, resampled
exactly to the common points; their transposes are exact linear transposes)
and never from a value trace.

The penalty is never a hand formula. Each owner certifies its flux with
`certify_flux_stability(flux)`, which returns a
`phydrax.discretization.TraceInverseEvidence`: for every interface facet `F`
with side cell `K`, the sharp constant of the discrete trace-inverse
inequality `||q(v)||²_F <= C_F a_K(v, v)`, solved as the largest eigenvalue of
the local pencil (facet flux Gram matrix, cell energy deflated on the
constants) by the native batched dense Hermitian eigensolver. The deflation's
premises are solved, not assumed: the cell energy must have exactly a
one-dimensional kernel seen by the cell moments, and the flux must vanish on
it to roundoff; a zero-energy state that carries flux has no finite constant
and is refused. At every common
point the penalty density is `γ = penalty_factor · Σ_s w_s² m_s C_s`, where
`C_s` belongs to the facet of side `s` containing the point and `m_s` counts
the law's interface facets sharing its side cell. The symmetric form is then
coercive with constant `1 - penalty_factor^(-1/2)` in the energy-plus-penalty
norm, so preparation refuses `penalty_factor <= 1` and reports the certified
penalty range; the nonsymmetric form is coercive for every positive factor. A
side with zero weight contributes no flux, so it needs neither a pointwise
flux nor stability evidence: a finite-element side with weight one can couple
to a virtual-element side, whose virtual interior gradient is not computable
and which therefore publishes no pointwise flux (it is refused when its weight
is positive).

`NitscheEvidence` records the variant, factor, weights, coercivity constant,
penalty range, trace and flux degrees, quadrature degree, both sides'
`TraceInverseEvidence`, and the coverage. The certificate reports
`law-residual` and `flux-conservation` (gated; the numerical flux
`{q} - γ [u]` enters both owners with opposite signs, and constants carry no
flux) and `trace-jump-l2` (evidence; a discretization error that decays with
the mesh). Every law certificate scales its reaction-based defects by the
uncancelled terms as well as by the sums themselves: the owner reaction and
the law rows linearized along the sign-scrambled state (fixed Rademacher signs,
`E ||A (s v)||² = ||A diag(v)||_F²`). An exact state whose terms cancel, such
as a constant equilibrium with zero source, is therefore measured against the
terms that cancel rather than against roundoff. The boundary-integral law does
the same for its Cauchy and logarithmic-compatibility defects with the
exterior Dirichlet-to-Neumann response to the sign-scrambled trace. Nitsche is
consistent: fields that lie in both discrete spaces are
reproduced to roundoff on nonmatching interfaces, including diffusivity
jumps, and P1/P2 errors converge at optimal order.
`examples/nitsche_transmission.py` solves a 1:4 diffusivity jump on 2:3
nonmatching P1 and P2 refinements, prints the certified constants, penalty
range, coercivity constant, and certificates, compares both variants, and
couples a virtual-element region one-sidedly.

Finite-element owners publish the pointwise flux for forms whose terms on the
traced field are `DiffusionAction`/`TensorDiffusionAction` plus mass, source,
and boundary terms. Its facet degree is polynomial for constant, cell-wise, or
DOF-field diffusivities on side cells that are affine along the facets;
callable diffusivities publish no degree, and the law refuses such a flux
because the common quadrature cannot re-evaluate it exactly. Finite-difference
SBP owners publish derivative operators, norms (`SBPGridNorm`), and nodal
traces, but no compiled physics owner with residual rows and no
`AbstractTraceComponent`, and their traces publish no polynomial facet degree,
so an FE–FD SAT coupling is not published through this assembler; SBP–SBP
interface SAT stays the formulation-specific `SATInterfacePlan`.

### Meshfree reconstruction/capacity components

`MeshfreeComponent` publishes a prepared scalar point-cloud, surface, or
conservative graph equation through reconstruction and capacity capabilities.
Its native operator source, field identity, quadrature, source program, and
numeric revision are admitted explicitly; equal shape or coincident nodes do
not establish ownership. It does not subclass `AbstractTraceComponent`.

`SurfaceExchangeLaw` lowers actual bulk-query/surface contributions using the
owner's coordinate transpose. Signed polynomial interpolation can conserve, but
positive native amount partitions require an explicitly positive reconstruction.
`LangmuirAdsorptionFlux` reuses native adsorption kinetics.
`MeshfreeBulkSurfaceMethod` supplies a native fixed-step participant; moving
queries relocate at host window boundaries with lag evidence.

Local edge-law learning uses MODEL-authorized bindings and `SolverObjective`
over native nonlinear/implicit solves. Law metadata remain static; numerical
parameters cross sparse derivative execution as filtered array leaves.
See [Meshfree solvers](guides_meshfree.md) for scalar-only scope, derivative
refusals, conservation versus positivity, and qualification.

### Flux, port, and transfer laws

- **`ConservativeFluxLaw(law_id, binding, (minus, plus), flux, *,
  quadrature)`** imposes one shared numerical flux: `flux`
  (`AbstractInterfaceFlux`) evaluates the minus side's outward conormal flux
  density `F(u_minus, u_plus, x, n)` at the common interface quadrature, and
  the one density enters both owners with consistent orientation, `-∫F v_minus`
  in the minus rows and `+∫F v_plus` in the plus rows, so what one side loses
  the other gains. `InterfaceConductance(h)` (imperfect contact,
  `F = h (u_plus - u_minus)`) is affine and lowers to linear assembly;
  `GapRadiation((e_minus, e_plus), *, stefan_boltzmann)` (gray-body exchange
  across a thin gap, quartic in `u`) makes the problem nonlinear. The law owns
  no unknowns; it gates `law-residual` and `flux-conservation` and reports
  `trace-jump-l2` as evidence, with `ConservativeFluxEvidence`.
- **`IntegralPortLaw(law_id, PortSide(connector, component, field, domain),
  system, equations, rhs, *, potential_unit, flux_unit)`** joins a field
  boundary port to a lumped `phydrax.system_modeling.AcausalSystem`. The
  connector's single across variable (identified by its declared kind, never
  by name) is the port potential `V`, and its single through variable is the
  flow `I = ∫ κ ∇u · n` entering the field. The coupling is a mortar with
  constant multipliers: field rows receive the uniform flux density
  `-(I / |port|) ∫ v`, and one law row imposes `mean_port(u) = V`. The
  network's constitutive rows and `compile_linear_acausal_system` connection
  equations stay with `system_modeling`. `port-potential`, `port-flux`, and
  `network-residual` are gated; `port-potential-deviation` measures how far
  the field is from equipotential on the port (`IntegralPortEvidence`).
- **`FieldTransferLaw(law_id, source, target, transfer, measure, *,
  exchange)`** exchanges `q = α (u_source - u_target)` between two co-located
  fields (bidomain, double-porosity, two-temperature models) through a native
  prepared `FieldTransfer` `T` whose field spaces are the components' own
  (the explicit identity binding of the two fields): target rows receive
  `-α M (T u_s - u_t)` and source rows, by work duality through the published
  dual pullback, `+α T^T M (T u_s - u_t)`, where `M` is the target measure.
  The exchange conserves the exchanged quantity exactly when `T` preserves
  constants, which preparation measures. `exchange-balance` is gated;
  `exchange-dissipation` and `transferred-mismatch-l2` are evidence
  (`FieldTransferEvidence`).

### Interface quadrature

`prepare_interface_quadrature(first, second, *, exact_degree, policy)` forms
the exact common refinement of two independently discretized sides of one 2-D
curve interface. Both traces are scalar value traces on
Gauss--Lobatto--Legendre facet sites with published polynomial degrees, so each
side is re-evaluated exactly between its sites. Every facet must be a straight
segment; every facet of one side is intersected with the collinear overlapping
facets of the other, and each common segment carries a Gauss--Legendre rule
exact to at least `exact_degree`. The common segments must tile every facet of
both sides: on each facet their union is the whole facet and no two overlap, so
a gap and an overlap on one facet are both counted rather than offsetting
(`InterfaceQuadraturePolicy.geometry_tolerance`, relative to the interface
length, bounds the uncovered plus multiply covered length of every facet). The
two outward normals must be opposite (`normal_tolerance`); otherwise
preparation refuses. `InterfaceCoverageEvidence` records both measures, the
common measure, the per-facet covered fractions, the largest per-facet tiling
defect, the largest normal defect, and the segment count.
`InterfaceSideQuadrature.values` and `pullback` are exact transposes built from
a per-point gather (`FacetResampling`), not a dense coefficient-by-point
matrix; both evaluate in the precision of the trace data, so float32 and
complex64 traces are not promoted to float64.

### Prepared problems, layouts, and gauges

`CoupledProblemPlan(plan_id, *, components, bindings, laws, gauge, parameters,
observations, resources)` orders components by name and laws and bindings by
identity; every law's binding must be declared and every declared binding
used, and component names and law IDs share one namespace.
`prepare_coupled_problem(plan, *, interface_owners, arguments, parameters)`
checks every binding against the current revision of its owners, lowers every
law through its components' publications, refuses duplicate impositions,
resolves eliminations, and validates every contribution against its endpoint
spaces.

A `PreparedCoupledProblem` has two layouts:

- **owner coordinates**: the native state and row blocks of every component
  and every law-owned block;
- **solve coordinates** (`state_space`, `row_space`): nested `BlockSpace`
  values addressed by `(owner, block)` paths, components by name and then laws
  with unknowns by law ID, with eliminated rows removed. The chart maps
  `z -> L z + h` (`owner_states`) and pulls rows back by `L^T`.

`residual(state, arguments)` and `weak_operator(arguments)` are the weak
residual and Jacobian into `row_space`. `linear_system(arguments)` returns a
native `LinearSystem` whose `BlockLinearOperator` is an endomorphism of
`state_space` (rows identified with their paired state blocks by the inverse
Riesz maps, as in the finite-element and virtual-element owners), together
with its right-hand side, so direct and Krylov policies both apply.
`nonlinear_problem()` is the native `NonlinearSystemProblem`, and
`field(component, field, state, arguments)` expands one full field.
`arguments` are always per component: a mapping from component name to that
owner's runtime arguments.

The coupled kernel is sought in a declared, bounded span: the kernels the
components publish, completed by every coordinate of the law-owned blocks
(multipliers, trace projections) and of the interface-only components
(components without side traces, such as `GalerkinBoundaryComponent`). Volume
owners enter only through their published kernels. The operator's images of an
orthonormal basis of that span are materialized once, within the plan's
`CoupledResourcePolicy.materialization`, and their null directions are the
right kernel; the left kernel is found the same way from the component kernels
read as row covectors and the rows of the same owners, and a pair of unequal
dimension is refused. A direction is null when its image, with every output
block divided by that block's gain on a fixed probe, is at most
`CoupledResourcePolicy.kernel_tolerance` (default `1e-8`) times its norm; a
stiff owner (the degree-4 virtual ring of the matching flagship variant, probe
scale 4.7e6) therefore does not make boundary-integral images of size 4e-2
look null. A pure-Neumann interior next to a bounded exterior is refused at
preparation with its kernel `u = 1, φ = 1, q = 0, c = 1`, which runs through
the law's trace projection `φ` and the exterior's constant `c`. A
coupled problem with a kernel and no `CoupledGauge(gauge, *, compatibility)`
is refused; with a gauge, the two kernels become the native `NullspacePolicy`
(`"minimum-norm"` or `"project"` gauge, `"error"` or `"project"`
compatibility). `CoupledResourcePolicy` holds the materialization bound and the
kernel detection tolerance. `ParameterBinding` records and
`AbstractObservationBinding` implementations (point, boundary, and flux
observations) are declared on the plan, which validates their component names
and keeps them in canonical order; see
[Parameters, derivatives, and observations](#parameters-derivatives-and-observations).

### Solving and acceptance

`solve_coupled_problem(prepared, *, arguments, parameters, policy, termination,
initial_state, tolerance)` runs the native solve (`phydrax.linalg.solve` for an
affine problem, `phydrax.nonlinear.root` with Newton--Krylov otherwise) and then
certifies the original equations:

- one `ComponentCertificate` per component: the norm of the owner's own
  residual plus the law terms on the rows it keeps, accepted when it is at most
  `tolerance` times the sum of the norms of its terms (eliminated rows are
  certified by the law's flux balance);
- one `InterfaceDefectReport` per law. A mortar reports `weak-continuity` and
  `flux-balance` (the owners' reaction fluxes on free interface rows against
  the shared multiplier injections), both gated, and `trace-mismatch-l2`, the
  discretization error of a nonconforming interface, as evidence only. A
  matching elimination gates `trace-continuity-l2` and `flux-balance`.

`CoupledSolution.accepted` requires native solver success and every
component certificate and gated defect; finite values or a small transformed
residual never imply acceptance. `native_successful` keeps the native status
separately, and `derivative_valid` is the native derivative admission combined
with acceptance and a non-stopped parameter route. It certifies that derivatives
are admitted at an accepted primal solve, not the later tangent or cotangent
solve: a derivative solve that fails is reported by the native linear contract,
as NaN-poisoned derivatives under `FailurePolicy("status")` (and an error under
`"error"`). `field(component, field)`, `law_state(law_id)`, `interface(law_id)`,
and `observation(binding_id)` read the result.

Without a `policy`, an affine problem is solved by `DenseLU` with
`LinearDerivativeSolvePolicy(route="primal-factors")` whenever a dense solve fits
linalg's default budgets and neither an `initial_state` nor a coupled gauge
(nullspace) is declared: coupled systems with interface multipliers are indefinite,
and the implicit tangent and adjoint solves then reuse the primal factors instead
of an independent restarted GMRES solve that can stagnate (on the boundary-integral
square it returned NaN derivatives while `derivative_valid` held). Otherwise
linalg's capability-selected default applies, and an explicit `policy` is always
used as given; linalg's own default `route="krylov"` is unchanged.

`initial_state` is either a solve-coordinate state, used as given (its
derivative follows the differentiation policy), or an
`AbstractInitialGuessProvider` such as `phx.linalg.LearnedInitialGuess`. A
provider is an accelerator: its untrusted proposal is used only when it is finite
and has a strictly smaller original residual than the native zero state, the
selected start carries no derivative, and acceptance certifies the solved state
alone. `CoupledSolution.initial_guess` reports the proposal and baseline residual
norms and whether the proposal was used; certifying the proposal itself with
`certify_coupled_state` keeps it distinct from the corrected state.

### Example: a finite-element region next to a virtual-element region

`examples/coupled_scalar_regions.py` solves `-Δu = f` for
`u = sin(πx/2) e^y` on `[0, 2] x [0, 1]`: P1 triangles on the left half and
degree-one conforming virtual elements on a perturbed brick mesh of hexagons
and quadrilaterals on the right half, whose interface vertices do not match the
triangles'. One law couples them through a side-trace mortar:

```python
import phydrax as phx
from phydrax.solver.coupling import (
    CoupledProblemPlan,
    MortarImposition,
    MortarMultiplier,
    prepare_coupled_problem,
    ScalarTransmissionLaw,
    solve_coupled_problem,
    TransmissionSide,
    VariationalComponent,
)

triangles = VariationalComponent("triangles", fe_problem, field="u")
polygons = VariationalComponent("polygons", vem_problem, field="u")
law = ScalarTransmissionLaw(
    "transmission",
    binding,  # two-sided InterfaceBinding: minus "triangles", plus "polygons"
    (
        TransmissionSide("triangles", "triangles", "u", fe_interface_facets),
        TransmissionSide("polygons", "polygons", "u", vem_interface_facets),
    ),
    MortarImposition(MortarMultiplier("side-trace", side="polygons")),
)
plan = CoupledProblemPlan(
    "fe-vem-plate", components=(triangles, polygons), bindings=(binding,), laws=(law,)
)
prepared = prepare_coupled_problem(plan, interface_owners=(cover,))
solution = solve_coupled_problem(
    prepared, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
)
if not bool(solution.accepted):
    raise RuntimeError("The coupled solution is not certified.")
report = solution.interface("transmission")  # weak continuity, flux balance, ...
```

The facet domains come from the owners:
`space.integration_domain("exterior_facet", EntitySelection(edges, mask))` on
both the finite-element and the virtual-element discretization. The example
prints, per refinement level, the nodal errors of both regions against the
exact field and their observed rates, the interface defects, the mortar
evidence, the component certificates, and the native and certified status, and
raises when a level is not accepted.

### Current limits

- Components are scalar and single-field: `VariationalComponent` publishes one
  scalar field of a compiled finite-element (including spectral-element),
  explicit-polygon H1, single-patch isogeometric, or virtual-element problem and
  refuses mixed forms.
- `ScalarTransmissionLaw` and `ConservativeFluxLaw` couple two different trace
  components across a two-sided binding on exterior facets of each side,
  through scalar value traces with published polynomial degree.
  `IntegralPortLaw` closes one scalar field port with one lumped connector
  that declares exactly one across and one through variable.
- The exact common refinement covers 2-D interfaces made of straight facets;
  curved facets and 3-D surface interfaces are refused.
- Mortar rank and inf-sup evidence and the matching relation are dense host
  computations bounded by `max_evidence_entries`.
- Nitsche is published for owners with an exact pointwise flux and certified
  trace-inverse evidence (finite elements); SAT penalties are not published
  through the spatial assembler. Nitsche certificates cover one law's terms;
  side cells shared with other weak interface laws are not accounted. Junction
  laws over three or more incidences are not published.
- The nonlinear route reports no derivative admission (`derivative_valid` is
  false).

## Two-dimensional Galerkin boundary operator

Two-dimensional exterior Laplace problems couple through a native Galerkin
boundary operator, not through the Nyström layer-potential carriers.
`prepare_scalar_laplace_galerkin_2d(ClosedPolygonalCurve2D(vertices, source_id=...))`
prepares, on one closed simple straight-panel polygon with normals pointing from
the bounded interior to the exterior and kernel `-log|x-y|/(2π)`:

| Published record | Meaning for a coupling law |
|---|---|
| `spaces.dirichlet_trace` | Continuous P1 value trace `φ` on vertices; arc-length Gram map with a prepared native inverse; `H^(1/2)`-conforming |
| `spaces.conormal_trace` | DP0 conormal `q` (outward normal derivative) on panels; diagonal arc-length Gram map; `H^(-1/2)`-conforming |
| `spaces.mixed_mass` | Physical duality `∫ q φ ds` from P1 into the DP0 dual |
| `spaces.far_field_space` | The exterior far-field constant `c` |
| `single_layer`, `double_layer` | Weak `V` (DP0 x DP0) and `K` (DP0 x P1) into the DP0 dual, with transposes and pairing adjoints |
| `exterior_relation` | Rectangular weak exterior Cauchy relation `((M/2 - K)φ + Vq - m c, m^T q)` from `(dirichlet_trace, conormal, far_field_constant)` to `(exterior_boundary_equation, total_conormal)` |
| `convention` | Jumps `γ0^± D = K ± I/2`, exterior representation `u = c + Dφ - Sq`, interior representation `u = Sq - Dφ` |
| `report` | Pair classes, quadrature errors, work, bytes, and the exact candidate `BoundarySupportEnvelope` |

The exterior relation subtracts the boundary trace; `(M/2 + K)φ - Vq = 0` is the
interior relation and is a different equation. A bounded exterior field requires
zero total conormal, so the far-field constant is a declared unknown column and
`m^T q = 0` a declared row. `prepare_exterior_laplace_dirichlet_2d` solves the
square bordered Dirichlet-to-Neumann system for `(q, c)`; a `"decaying"` far
field additionally requires `|c|` below its tolerance and reports a nonzero
constant as unsatisfied instead of accepting it. Acceptance recomputes both
original equations and scales them by their terms at the solution and at the
response to the sign-scrambled Dirichlet data (one extra solve with the same
prepared system): constant data, whose double-layer and identity terms cancel
and leave `q` at roundoff, are accepted with `q = 0` and `c` equal to the
constant rather than refused against a roundoff-sized total conormal.

Higher-order volume traces reach P1 only through
`prepare_boundary_trace_projection_2d`: its `dirichlet_projection` is an L2
projection operator from panel samples to P1 (Hilbert adjoint: P1 evaluation at
the samples), and `project_dirichlet` / `project_conormal` return the measured
defect `||f - Πf||`. A projection is never labeled an exact trace.

`examples/exterior_laplace_galerkin.py` recovers an analytic off-center dipole
field under panel refinement and prints near- and far-field errors, the solved
far-field constant, the total conormal, and quadrature evidence.

Current limits: one closed curve with straight panels; no open curves, curved or
moving geometry, Helmholtz, hypersingular operator, FMM Galerkin action,
geometry derivatives, or continuum error certificate.

## Spectral-element, virtual-element, and boundary-element transmission

`examples/sem_vem_bem_transmission.py` solves one Laplace field across three
owners in ONE coupled solve. The square annulus `0.5 <= |x|_inf <= 1.5`
surrounds a hole containing the origin; conforming virtual elements on perturbed
polygons discretize the inner ring, spectral elements (tensor Lagrange
quadrilaterals on Gauss–Lobatto nodes) the outer ring, and the 2-D Galerkin
boundary operator on the outer square the unbounded exterior. The reference
`u = x / (x^2 + y^2)` is harmonic away from the origin, decays, and has zero net
exterior flux; the hole carries its Dirichlet data and `κ = 1`. The reference is
evaluated on the host and never computed from the discrete system.

An analytic `SubdomainCover` of concentric square bands is the geometry
authority of both interfaces: pairing `inner-ring|outer-ring` binds the internal
square and `outer-ring|exterior` the outer square, each with its outward normal
(minus side inside). Its bounded window only witnesses the plus side of the
outer square; the unbounded exterior belongs to the boundary owner.

### Boundary-integral transmission law

`BoundaryIntegralTransmissionLaw(law_id, binding, volume, boundary, *,
projection_order, interface_source, quadrature)` couples any trace component
(`volume`, a `TransmissionSide` on the minus side) to a
`GalerkinBoundaryComponent` (`boundary`, a `BoundaryIntegralSide` on the plus
side). The binding attaches `{"value": volume field}` and
`{"conormal": exterior conormal field}`. With `n` pointing into the exterior,
`κ_ext = 1`, `q = κ ∂_n u` on the curve, `φ` the law's continuous P1 Dirichlet
trace, and `c` the far-field constant, the bordered Johnson–Nédélec system is

| Rows | Equation | Owner of the term |
|---|---|---|
| volume field | `R(u) - ∫ q γv ds - ∫ g γv ds` | owner residual; law injects the DP0 load (and the optional declared jump `g`) |
| law `trace-projection` (P1 dual) | `M_P1 φ - L γu` | law; `L` is the load of the declared L2 projection (`prepare_boundary_trace_projection_2d`) on per-panel Gauss samples of the exact volume trace |
| exterior `exterior_boundary_equation` (DP0 dual) | `(M/2 - K) φ + V q - m c` | law adds `(M/2 - K) φ`; the component owns `V q - m c` |
| exterior `total_conormal` | `m^T q` | component |

The signs follow from the volume weak form `∫ κ∇u·∇v - ∫ κ ∂_n u γv = ∫ f v`
and the exterior trace of `u = c + Dφ - Sq` (`γ0^+ D = K + I/2`). The projection
row is the declared L2 projection `φ = Πγu` of the boundary owner written as an
equation, so no Gram inverse appears in the coupled operator and dense direct
solves materialize it. `interface_source` prescribes the conormal jump
`κ ∂_n u_- - ∂_n u_+`, sampled at the projection points.

Exactness and geometry are checked at preparation: the default
`projection_order` is the smallest Gauss rule that integrates the P1 load of a
degree-`p` trace exactly (`(p + 3) // 2` points; a smaller declared order is
refused), every boundary panel must lie inside one straight volume facet
(panels refine facets), the two partitions must cover each other, and the
volume's outward normal must be the curve normal. The certificate gates the
exterior Cauchy residual, the logarithmic compatibility `|m^T q|`, the
projection equation, and the flux balance (the volume owner's reaction on its
boundary rows against the injected load); it reports as evidence the
projection defect `||γu - φ||`, the work-pairing defect `|⟨q, γu⟩ - ⟨q, φ⟩|`,
and `|c|`. `BoundaryIntegralEvidence` records the formulation, trace degree,
projection order and exactness, panel count, and coverage.

The far-field behavior is declared on the boundary owner, with the same modes
as `prepare_exterior_laplace_dirichlet_2d`:
`GalerkinBoundaryComponent(name, galerkin, *, far_field="bounded",
far_field_tolerance=None)`. A `"bounded"` exterior (the default) keeps `c` a
free unknown and `|c|` evidence. A `"decaying"` exterior requires
`far_field_tolerance` (a bounded mode takes none) and adds the gated
`far-field-decay` defect `max(|c| - tolerance, 0)` with scale `tolerance`, so a
solved constant beyond the declared tolerance makes the solution unaccepted
instead of being reported silently. On the P2 square (`|c| = 2.7e-3`, its
discretization error) a tolerance of `1e-2` is accepted and `1e-3` refused.

### Flagship evidence

The default flagship (spectral elements: 2 cells across the outer ring,
degree 4; virtual elements: 3 cells across the inner ring, degree 2; 2 panels
per spectral facet, 96 panels; side-trace mortar multiplier of degree 2 on the
virtual-element side, rank 96 of 96, discrete inf-sup 1.33) has 2185 unknowns
and one dense LU solve. Measured with `kappa = 1`:

| Check | Value |
|---|---|
| exterior Cauchy residual / logarithmic compatibility / projection row / boundary flux balance (gated) | 1.9e-14 / 1.1e-16 / 1.8e-15 / 4.7e-14 |
| internal weak continuity / internal flux balance (gated) | 4.3e-15 / 1.5e-13 |
| projection defect `‖γu - φ‖` / work-pairing defect / `\|c\|` (evidence) | 6.2e-4 / 8.5e-6 / 7.5e-5 |
| max nodal error: spectral ring / virtual ring | 3.15e-3 / 6.02e-3 |
| DP0 conormal L2 error / max far-field error (radii 2.5 and 10) | 3.93e-3 / 1.84e-4 |
| `sensors` observation of the inner ring (`h1-projection`) | 2.68e-3 |

The heterogeneous case (same discretization) is accepted with the same gated
defect levels (boundary flux balance 1.8e-13 against a reaction scale 0.80,
internal flux balance 5.4e-13 against 2.6) and errors 3.16e-3 / 6.03e-3 nodal,
3.72e-3 conormal, 2.63e-4 far field.

### Substitutions

Only component preparation changes:

| Substitution | Declarations | Status |
|---|---|---|
| Spectral elements → Lagrange triangles | same bindings, laws, observation | supported (`outer_method="fe"`) |
| Spectral elements → explicit polygons (P1) | same | supported (`outer_method="polygon"`) |
| Virtual elements → Lagrange triangles | same, including the `sensors` point observation of component `inner` | supported (`inner_method="fe"`); the observation reports `"exact"` instead of `"h1-projection"` |
| Spectral elements → single-patch IGA | same boundary law and exterior owner | supported on the full square (`solve_square("iga", ...)`): the side trace acts on the compiled field's public control layout; not on the annulus, because the isogeometric owner is one untrimmed tensor patch (no hole, multipatch, or periodic knots) |
| Mortar → matching elimination | same law, `MatchingElimination` | supported on coincident facets with equal trace degrees and one-axis coefficient rows (a tensor-layout side such as an IGA patch couples through a mortar) |

`CompiledFiniteElementProblem.prepare_conormal_flux` publishes the residual
reaction for every local H1 provider whose trace acts on its field coordinates
(finite elements, explicit polygons, isogeometric patches), which is what the
transmission and boundary laws consume. The same boundary law also couples
spectral-element, triangle, explicit-polygon, and single-patch isogeometric
owners of the full square (`solve_square`); there the interior carries a unit
reaction (`-Δu + u = f`), because a pure-Neumann interior next to a bounded
exterior with a free far-field constant has the kernel `u = 1, φ = 1, q = 0,
c = 1` (`neumann_kernel` in the example prints its preparation-time refusal and
the gauged kernel pair).

Measured on the default discretization: P2 triangles on the outer ring give
2.77e-3 / 6.09e-3 nodal and 4.29e-3 conormal error; explicit P1 polygons give
3.63e-3 / 1.01e-2 and 1.88e-2 (the P1 trace is the coarser boundary datum).
On the full square (2 cells per half side, 2 panels per facet) the same law
gives nodal / conormal / far-field errors 5.1e-3 / 2.6e-2 / 7.2e-4 for degree-4
spectral elements, 4.5e-2 / 8.1e-2 / 6.3e-3 for P2 triangles, and 8.8e-2 /
1.3e-1 / 9.4e-3 for explicit polygons; quadratic B-splines (4 spans per side,
36 control values, sampled through the exact reconstruction on a 13 x 13 grid)
give 1.3e-1 / 6.5e-2 / 4.3e-4. The spline's point error sits at the cap peak
(`x = -0.5`, `|u| = 1.37`), which the coarse control net does not resolve; its
boundary samples are within 9.1e-3. All are accepted with gated defects at
roundoff.

### Refinement campaigns

Every refinement axis of `Configuration` is independent (`outer_cells`,
`outer_degree`, `inner_cells`, `inner_degree`, `panels_per_facet`,
`imposition`/`multiplier_degree`, `regular_order`/`projection_order`); the
example prints one table per axis. The h study (spectral degree 2, virtual
degree 1, one panel per facet, both mesh sizes halved per level) gives:

| Level | Max nodal error, spectral / virtual | DP0 conormal L2 error | Far-field max error | `\|c\|` |
|---|---|---|---|---|
| 1 | 1.05e-2 / 1.57e-2 | 3.86e-2 | 2.65e-3 | 1.1e-3 |
| 2 | 2.76e-3 / 5.23e-3 | 1.17e-2 | 4.05e-4 | 3.2e-5 |
| 3 | 7.44e-4 / 1.50e-3 | 3.58e-3 | 1.10e-4 | 8.4e-6 |

Rates approach 2 (1.9 for the spectral ring, 1.6-1.8 for the virtual ring,
1.7 for the conormal). The other campaigns start from the default flagship
(errors: spectral ring / virtual ring nodal, DP0 conormal, far field):

| Campaign | Values | Reading |
|---|---|---|
| spectral p = 2, 4, 6 (panels fixed) | 2.81e-3 / 6.04e-3 / 4.00e-3 / 1.82e-4; 3.15e-3 / 6.02e-3 / 3.93e-3 / 1.84e-4; 3.31e-3 / 6.02e-3 / 3.92e-3 / 1.83e-4 | saturated: the spectral elements are not the floor |
| virtual degree k = 1, 2, 3 | 9.76e-3 / 7.77e-3 / 3.63e-3 / 7.8e-4; 3.15e-3 / 6.02e-3 / 3.93e-3 / 1.84e-4; 2.38e-3 / 2.42e-3 / 3.90e-3 / 3.1e-5 | the virtual ring (near the hole) is the active volume floor |
| panels per facet 1, 2, 4, 8 | conormal 1.31e-2, 3.93e-3, 1.33e-3, 5.48e-4; volumes unchanged (3.15e-3 / 6.02e-3) | P1/DP0 boundary data limit `q` only |
| multiplier: side-trace, discontinuous degree 2 and 4 on the spectral side, matching elimination (virtual k = 4 on coincident facets) | 3.15e-3 / 6.02e-3; 2.04e-3 / 6.04e-3; 2.83e-3 / 5.97e-3; 1.17e-3 / 5.0e-4 | stable multipliers do not change the floor; the matching case is more accurate only because its virtual degree is 4 |
| quadrature: panel rule 8 → 16, projection rule minimal → 8 points | identical to all printed digits | quadrature is not active |

Raising the spectral degree with fixed panels therefore saturates at the
virtual-element and P1/DP0 floors: total p-convergence is algebraic unless the
other owners and the panels are refined with `p`. The integration tests'
tolerances are set from these tables.

### Refusals

Preparation refuses, with the reason: inadequate mortar rank or inf-sup; panels
that straddle volume facets, a curve that does not coincide with the volume
boundary, or facing normals; strong Dirichlet rows or an owner natural law on
the coupled boundary; an inexact declared projection order; interface-source
data of the wrong shape; and Galerkin resident-byte budgets. A native solve
that fails (for example an iteration-capped Krylov policy) returns
`native_successful = False` and `accepted = False`.

### Projection and gauge limitations

- The spectral-element trace reaches the P1/DP0 boundary spaces only through the
  declared L2 projection; the boundary error floor is that of P1/DP0 Galerkin
  data (second order in the panel size), so total p-convergence is algebraic
  unless the panels are refined with `p`. No exponential total convergence is
  claimed.
- The exterior is the unit-coefficient Laplace field of one closed straight-panel
  polygon; its far-field constant is an unknown. The flagship declares the
  default `"bounded"` exterior and reports `|c|` (7.5e-5 at the default
  resolution, converging with the discretization) as evidence of the decaying
  reference; a `"decaying"` declaration gates `|c|` at a declared tolerance
  instead (`c` stays a solved unknown; the declaration never fixes it to zero).
- There is no gauge freedom in the flagship: the hole's Dirichlet data fixes
  the constant (the spectral ring's published constant, completed through the
  mortar multipliers, the trace projection, and the boundary unknowns, is not a
  coupled kernel). A pure-Neumann interior coupled to a bounded exterior has the
  kernel `u = 1, φ = 1, q = 0, c = 1`, detected at preparation through the law
  unknown `φ` and the boundary unknown `c` and refused without a
  `CoupledGauge`. With a gauge the prepared `NullspacePolicy` carries this
  right kernel and the different left kernel `u = 1, c = 1` (rows) of the
  nonsymmetric coupling. Dense LU cannot factor the singular operator (its
  native status reports failure); no solver route for the gauged singular
  Johnson–Nédélec system is qualified here, so remove the kernel physically
  (Dirichlet data or a reaction term) for a solve. The detection span's dense
  images count against `CoupledResourcePolicy.materialization`; the flagship
  declares its dense budget there.
- Virtual-element interior values are an `h1-projection` reconstruction; its
  edge traces are exact.

### Existing 3-D FEM–BEM products

The 3-D scalar Johnson–Nédélec and static-elasticity Costabel FEM–BEM products
remain the scientific owners of their matching P1/DP0 and Kelvin/DP0 couplings.
`ScalarLaplaceFEMBEMComponent(name, product)` and
`ElasticityFEMBEMComponent(name, product)` publish an already prepared product
to the spatial assembler with its own named blocks (`interior_field`,
`exterior_conormal` / `interior_displacement`, `boundary_traction`), operator,
right-hand side (`ScalarLaplaceFEMBEMArguments`, `ElasticityFEMBEMArguments`),
and `prepared_id`, so a coupled plan reproduces the product's solve and other
laws may address its fields. Nothing is reassembled; the products' geometry,
formulation, and derivative refusals are unchanged. The nonmatching dense and
dynamic convolution-quadrature products expose no named blocks and are not
components; no H(curl)-to-RWG construction or 3-D virtual-element compiler is
claimed.

## Parameters, derivatives, and observations

### Owner-qualified derivatives

Derivatives belong to the scientific owner that prepared the numerical map. A
prepared product publishes an `OwnerDerivativeCapability` (field
`derivative_capability`, also carried by each result). The capability names the
runtime arguments that may be differentiated, the `DerivativeSurface` each one
enters, and the route. It also names every refused quantity with a reason:
runtime arguments outside the selected route, and fixed prepared structure such
as geometry, kernels, and quadrature. A consumer checks
`capability.require("conductivity")` before it binds a parameter; the check
raises `derivative-unsupported` with the owner's reason.

The linear policy's `DifferentiationPolicy` selects the route at preparation.
Host preparation is never traced. This includes interface matching, pair
classification, singular quadrature, and symbolic solve plans. Runtime
coefficients change numerical values only. The scalar FEM–BEM conductivity
changes the diagonal of `G^T diag(κ|T|) G` and rebinds the prepared solve
through `phydrax.linalg.refresh`, so a compiled gradient runs again for a new
coefficient without replanning or host synchronization.

| Owner | Admitted runtime arguments | Route |
|---|---|---|
| Scalar 3-D FEM–BEM (`rhs-only`) | `volume_source_coefficients`, `dirichlet_jump`, `conormal_jump` | implicit solution map |
| Scalar 3-D FEM–BEM (`mathematical`) | the above and per-cell `conductivity` | implicit solution map |
| Elasticity 3-D FEM–BEM (`rhs-only`) | `interior_load`, `boundary_load` | implicit solution map |
| 2-D Galerkin operator actions | `dirichlet`, `conormal`, `far_field_constant` | direct (linear) |
| 2-D bordered exterior solve (`rhs-only`) | `dirichlet` | implicit solution map |
| Compiled FE / VEM problems (`linear_system(args)` + `phydrax.linalg.solve`) | `rhs-only`: source, boundary load, Robin value, Dirichlet lift; `mathematical`: also diffusion, tensor-diffusion, mass/reaction, and Robin coefficients | implicit solution map |

Compiled FE and VEM problems rebind runtime data on every call without
re-preparation, and the whole path compiles under `jax.jit`. FE callable
coefficients receive the `FiniteElementExecutionContext` and read
`context.user_args`; VEM callable coefficients receive `user_args` directly.
Runtime Dirichlet data enters through the execution context `lift`. The
adjoint route is `solve_adjoint` followed by the residual VJP over the runtime
arguments; `linearization_operator` and the VEM `affine_operator` carry exact
transposes. Mesh topology, DOF routes, quadrature, and constraint rows are
fixed and not differentiable.

The following rules apply at evaluation:

- A JVP, VJP, or gradient through an argument that the capability does not
  admit raises at transformation time. A stopped route never becomes a silent
  zero derivative.
- Primal acceptance (`valid` or `accepted`) and `derivative_valid` are
  separate. `derivative_valid` also requires an admitted route and a converged
  solve. Derivatives of a result that is not accepted are NaN under the status
  failure mode and raise under the error failure mode. An implicit tangent or
  adjoint solve that misses its `LinearDerivativeSolvePolicy` also returns NaN.
- Derivatives are qualified in order: right-hand-side and boundary data, then
  scalar or material coefficients, then operator parameters. Geometry, kernel,
  and topology derivatives are never inferred from the first two.
- A derivative study of a two-dimensional `decaying` exterior solve must
  perturb inside the compatibility regime (a vanishing far-field constant).

The support table and refusal reasons for each owner are listed in
[Boundary platform qualification](guides_boundary_platform.md#derivative-support).

### Parameter bindings

A `ParameterBinding(binding_id, port, *, targets, role, derivative=None,
change="refresh")` on a `CoupledProblemPlan` binds one scientific parameter to
named runtime inputs of its components. `port` is a `phx.ValuePort` (semantic
identity, event shape, representation, physical dimensions); a bound array is
inexact and has exactly the port's `event_shape`. Each
`RuntimeInput(component, name)` names the key under which the owner reads the
value from its runtime user arguments: finite-element coefficient callables
receive the `FiniteElementExecutionContext` and read `context.user_args[name]`,
and virtual-element coefficient callables receive the user-argument mapping and
read `args[name]`. One parameter may target several components; one runtime
input is bound by at most one parameter.

| `role` | Meaning | Admitted `derivative` |
|---|---|---|
| `"coefficient"` | material coefficient in the coupled operator | `PHYSICAL_PARAMETER` or `None` |
| `"source"` | right-hand-side data | `SOLVER_ARGUMENT` or `None` |
| `"boundary"`, `"control"` | boundary or control input, entering where declared | `PHYSICAL_PARAMETER`, `SOLVER_ARGUMENT`, or `None` |

`derivative=None` requests no derivative. A `"refresh"` change is a numeric
refresh: prepared topology, layouts, interface quadrature, and routes stay
fixed, and the value is a dynamic runtime argument supplied at every solve and
never retained. A `"reprepare"` change fixes the value at preparation (it enters
the `problem_id`), refuses it at solve time, admits no derivative surface, and
never takes a learned model.

`prepare_coupled_problem(plan, *, interface_owners, arguments, parameters)`
requires a reference value of every declared binding. Preparation determines
the execution kind with them and, for a linear problem, evaluates the component
kernels and verifies every `SOLVER_ARGUMENT` declaration: the traced program of
the coupled operator's action must carry no data dependence on that parameter
(a structural fact that holds for every value, not a tangent at the reference
value, which vanishes for `1 + p**2` at `p = 0`), otherwise preparation refuses
the binding and asks for a physical-parameter declaration. Only `"reprepare"`
values are retained. The `problem_id` fingerprints the law declarations and
their lowered content, the gauge, the resources, and the parameter bindings.

`prepared.bind_arguments(arguments, *, parameters)` returns `CoupledArguments`:
`arguments` is the per-component mapping consumed by `residual`,
`linear_system`, and the other per-component methods, and `parameters` pairs
every binding with its value. `solve_coupled_problem(prepared, *, arguments,
parameters, ...)` binds the same way. Bound inputs are merged into a
component's user arguments, which must then be a mapping or omitted; a runtime
input supplied both as a parameter and in `arguments` is refused. A component
whose arguments are an explicit finite-element or virtual-element execution
context (for example one carrying a runtime Dirichlet `lift`) cannot receive
bound parameters; pass its user arguments as a mapping instead.

A refresh value may also be a learned model: a `phx.ComponentBinding` with
`MODEL` or `DISCRETIZATION` authority whose `owner_ports.outputs` contain the
parameter port. The owner then receives `binding.model` under the runtime input
name, and its coefficient callable evaluates it, for example at quadrature
points. Accelerator, surrogate, and decision components are refused, because a
learned physical input changes the accepted equations.

### Differentiating coupled solves

`prepared.derivative_capability(policy)` is the coupled problem's
`OwnerDerivativeCapability` (owner ID `problem_id`); every `CoupledSolution`
carries it as `derivative_capability`. Derivatives exist only through parameter
bindings: a derivative through a raw `arguments` leaf of `solve_coupled_problem`
(a coefficient, load, or model not supplied by a binding) raises
`derivative-unsupported` instead of returning a silent value.

| Execution | `DifferentiationPolicy` | Admitted bindings | Route |
|---|---|---|---|
| linear | `"mathematical"` | `PHYSICAL_PARAMETER` and `SOLVER_ARGUMENT` | implicit |
| linear | `"rhs-only"` | `SOLVER_ARGUMENT` | implicit |
| linear | `"algorithmic"`, `"none"` | none | stopped |
| nonlinear | any | none | stopped |

Bindings without a derivative surface and `"reprepare"` bindings are refused
with their reasons. Prepared `"geometry"` (mesh coordinates, interface
matching, common refinements), `"quadrature"`, and `"topology"` (component
topology, DOF layouts, constraint rows) are always refused. An implicit
capability states its conditions: `accepted-result`, `prepared-geometry-fixed`,
and `solve-converged`.

Derivatives of an affine coupled problem are the implicit solution-map
derivatives of the native `phydrax.linalg.solve` of
`prepared.linear_system(arguments)`. Bound parameters enter through the owners'
runtime-argument assembly, so a JVP, VJP, or gradient with respect to their
values differentiates the assembled operator and right-hand side without
re-preparation. The whole path compiles under `jax.jit`, and a `CoupledSolution`
is a valid compiled result. Coupled systems with interface multipliers are
indefinite, where a restarted Krylov derivative solve can stall. The default
policy of `solve_coupled_problem` already uses `DenseLU` with
`route="primal-factors"` when a dense solve fits (see
[Solving and acceptance](#solving-and-acceptance)). With an explicit `DenseLU` or
`DenseCholesky` policy and no coupled gauge, declare
`LinearDerivativeSolvePolicy(route="primal-factors")` so the tangent and adjoint
solves reuse the primal factors and are exact to roundoff
([implicit differentiation](api/linalg.md#differentiation-semantics)); linalg's
default `route="krylov"` keeps the independent GMRES derivative solve.

```python
policy = phx.linalg.LinearSolvePolicy(
    phx.linalg.DenseLU(),
    differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
    derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
)
prepared.derivative_capability(policy).require("conductivity-left")


def misfit(theta):
    solution = solve_coupled_problem(
        prepared,
        parameters={
            "conductivity-left": theta[0],
            "conductivity-right": theta[1],
            "heat-flux": theta[2],
        },
        policy=policy,
    )
    residual = solution.observation("left-sensors").values - observed_values
    return 0.5 * jnp.sum(residual**2)


gradient = jax.jit(jax.grad(misfit))(theta)  # adjoint: one transposed solve
_, directional = jax.jvp(misfit, (theta,), (direction,))  # tangent solve
```

- A JVP, VJP, or gradient with respect to a refused parameter raises
  `ValueError("derivative-unsupported: ...")` at trace time with the owner's
  reason. For example, a conductivity gradient under `"rhs-only"` is refused,
  while the heat-flux gradient of the same solve is admitted.
- `derivative_valid` requires acceptance, the native derivative admission, and
  a non-stopped route, and it stays separate from `accepted`. Derivatives of a
  solution that is not accepted, or of a failed derivative solve, are NaN under
  failure mode `"status"` and raise under `"error"`.
- Nonlinear coupled execution publishes no parameter derivative:
  `derivative_valid` is false, and every parameter derivative is refused.
- Geometry, quadrature, and topology derivatives are never published through
  parameter bindings.

### Observation bindings

An observation binding names a component and field, never an owner object, so
substituting a component's method under the same name keeps the binding. It
also names the measurement it predicts: a `QuantitySpec`, a scalar
`ValueLayout` (default `ValueLayout.scalar()`), a `SampleSupport`, and
`SamplingSemantics`. Preparation lowers each binding once through the owner's
published capabilities. Every `CoupledSolution` then evaluates it on the
certified fields, and `solution.observation(binding_id)` returns a
`phx.measurement.PreparedQuantityField` whose identities equal those of data
prepared from the same records. A prediction is valid evidence only when
`accepted` holds. `prepared.observation(binding_id).approximation` labels what
is observed.

| Binding | Operation | Sampling | Quantity unit dimension |
|---|---|---|---|
| `FieldPointObservation(id, component, field, *, quantity, support, sampling, field_unit, derivative=None, length_unit=None, coverage="complete")` | field value, or one partial derivative (coordinate multi-index), at the active `PointSampleSupport` points | `POINT` | `field_unit`, or `field_unit / length_unit^order` |
| `FieldBoundaryObservation(..., domain, *, statistic="trace", rule, ...)` | exact side trace at the `FacetTraceRule` sites of exterior facets | `POINT`, support points exactly at the sites | `field_unit` |
| `FieldBoundaryObservation(..., statistic="average", ...)` | facet-measure mean | `SURFACE_AVERAGE`, one sample | `field_unit` |
| `FieldBoundaryObservation(..., statistic="integral", length_unit=...)` | facet-measure integral over one-dimensional facets | `PATH_INTEGRAL`, one sample | `field_unit * length_unit` |
| `FieldFluxObservation(id, component, field, domain, *, rule, quantity, support, sampling, reaction_unit)` | sum of the owner's residual reaction on the facets' trace rows | `PATH_INTEGRAL`, one sample | `reaction_unit` |

A declared sampling kind that differs from the operation is refused at
construction ("a point value, an average, and an integral are distinct
measurements"). So is a quantity unit whose dimension differs from the
operation's unit ("a density, a content, and a field value are never
identified"). Commensurable units are converted exactly. Point supports are
three-dimensional; coordinates beyond the component's physical dimension must
be zero. The coordinates of a `PointSampleSupport` are in the length unit of
its coordinate contract, so `length_unit` defaults to that unit and a differing
declaration is refused. `coverage="complete"` refuses points outside the
component and inactive support samples; `coverage="masked"` reports both as
invalid samples.

Observed data is a `phx.measurement.QuantityField` built from the same quantity,
layout, support, and sampling records and prepared once. The comparison then
refuses mismatched identities:

```python
observed = phx.measurement.QuantityField(
    "left-sensors",
    temperature,
    phx.measurement.ValueLayout.scalar(),
    left_sensors,
    point,
    measured_values,
    uncertainty=phx.measurement.IndependentStandardUncertainty(
        sigma, phx.units.KELVIN
    ),
).prepare()
comparison = phx.observation.MeasurementComparisonPlan(observed).evaluate(
    solution.observation("left-sensors")
)
```

The limits are explicit:

- Point observations require a component that publishes a pointwise
  reconstruction (`AbstractReconstructionComponent`: `VariationalComponent`
  and `ReducedComponent`). Finite-element values are exact (`"exact"`) within
  the cell that contains the point, located with the component's
  `location_policy`
  (`VariationalComponent(..., location_policy=SimplicialLocationPolicy(...))`,
  finite-element owners only); the binding itself carries no location policy.
  Virtual-element point values and derivatives are the labeled
  `"h1-projection"` of the virtual field, not the field itself.
  Boundary traces, averages, and integrals use the owner's exact side traces,
  virtual elements included.
- Flux content pairs the residual reaction with the discrete indicator
  `w = sum_i phi_i` of the selected trace rows. It equals the total outward
  conormal flux through the facets only when the one-element fringe where `w`
  decays carries no flux reaction, for example the whole boundary, or a wall
  whose adjacent sides are insulated. It is a content, never a flux density.
  Boundary integrals and flux contents are path integrals over the boundary of
  planar components.
- Boundary-element and FEM–BEM components publish no point, trace, or flux
  observation; binding one is refused at preparation.

### Inverse problems and learned components

Inference over bound parameters is an ordinary JAX objective over one prepared
problem. Each evaluation solves with new values, compares the observations
with prepared data, and differentiates with `jax.grad` (adjoint) or `jax.jvp`
(tangent). Nothing is re-prepared between iterations. Training uses the
existing frontend, never a separate loop:

- A learned physical coefficient, such as a conductivity field bound through a
  `ComponentBinding` with `MODEL` or `DISCRETIZATION` authority, trains with
  `phx.solver.SolverObjective(solve, bind, measure, ...)` and
  `phx.solver.train_components`. `bind` passes the model as the parameter
  value, `measure` returns a `SolverCaseResult` with `accepted=solution.accepted`
  and the observation residual, and non-accepted solves reject the attempt.
- An accelerator (preconditioner or initial guess) changes Krylov work, not the
  solution, so a `SolverObjective` refuses it ("no admissible training signal").
  It trains with `phx.solver.AlgorithmicWorkObjective`: `measure` runs exactly
  `work` iterations (for example FGMRES with `DifferentiationPolicy("algorithmic")`,
  zero tolerances, and `max_steps=work`) and returns an `AlgorithmicWorkResult`
  of the original residuals before and after. A learned right preconditioner
  subclasses `phx.linalg.AbstractPreconditioner` over `prepared.state_space`,
  binds its model to that slot (`phx.bind_component(model, MyPreconditioner)`),
  and enters production solves through `PreconditioningPolicy`. A learned
  initial guess enters as an `initial_state` provider (see
  [Solving and acceptance](#solving-and-acceptance)). Acceptance certifies the
  original equations, so neither changes the accepted solution; a singular
  learned preconditioner makes the solve fail, never succeed with another answer.
- A learned input that changes the accepted equation is a model, not an
  accelerator. An `ACCELERATOR`, `SURROGATE`, or `DECISION` component supplied
  as a parameter value is refused, as is a model whose owner output ports do not
  publish the parameter's port. The solution-map objective scores models on
  accepted states only, and a `SURROGATE` has no implicit solution-map signal.
- In one tree holding both kinds, each objective trains only the authorities it
  admits and stops the others: the `SolverObjective` trains the `MODEL`
  conductivity and stops the preconditioner, and the fixed-work objective does
  the reverse (`objective.admit(tree)` lists both groups).

`examples/coupled_inverse_problem.py` infers the right-region conductivity and
the right-wall heat flux of the finite-element / virtual-element plate. The
data are four temperature sensors, the right-wall mean temperature, and the
left-wall heat inflow (the conormal flux content `∫ κ ∂u/∂n ds` with the outward
normal that `FieldFluxObservation` reports; negative here because the heat
leaves through that wall), drawn from the host analytic plate with declared
independent uncertainty. The unknowns live in a `MODEL`-authority
`ComponentBinding`, and a `SolverObjective` over the whitened comparison
residuals is fitted by L-BFGS through `train_components`. The script then forms
the fixed-noise posterior with `posterior_problem_from_solver_objective` and
checks the estimate against the Gauss–Newton standard deviation plus the shift
caused by the discretization bias at the truth.

`examples/coupled_learned_interface.py` trains both learned roles on the same
plate:

- A log-linear conductivity field of the finite-element region (`MODEL`
  authority) is fitted to point temperatures of an independent analytic plate
  whose left conductivity grows exponentially along `x`. The field's relative RMS
  error against the analytic conductivity drops from 0.43 to 0.033 in 15
  accepted L-BFGS steps; the trained field fits the data better than the true
  field does on the P1 mesh, so the remaining error is discretization bias.
- A learned right preconditioner `I + W0 + W1 / kappa_left + W2 / kappa_right`
  (`ACCELERATOR`) is trained on eight FGMRES steps at sampled parameters. At
  held-out parameters FGMRES(30) needs 54 and 64 iterations instead of 447 and
  885, and both accepted states match the dense LU solution to 4e-11.
- A warm start from a neighboring accepted state is used and saves iterations,
  but it is not an accepted solution itself (worst relative residual 3e-2). A
  constant 1e3 proposal is rejected, and the solve matches the zero start
  iteration for iteration.

The script also prints each refusal.

## Measurement noise and likelihoods

A coupled prediction reaches data through `phydrax.observation.MeasurementComparisonPlan`,
whose result keeps the residual, its weighting, and its normalization separate:

- `noise_model` states the declaration: `"unquantified"`, `"reference_weighting"`,
  `"independent_uncertainty"`, or `"covariance"`. Absent uncertainty means
  unquantified data, never unit Gaussian noise.
- `whitened_residual` is factor-backed or absent. Diagonal, dense Cholesky,
  Kronecker Cholesky, and circulant covariances whiten through their own
  factors, and the squared norm equals `quadratic`. Dense precision,
  diagonal-plus-low-rank, and matrix-free precision covariances report
  `whitening="unavailable"` with the quadratic and log determinant only.
- `logdet_covariance` and `log_likelihood` exist only for a declared real-valued
  noise model; `successful` also requires positive uncertainty and, for a
  covariance, a prediction valid on every correlated active value.
- A correlated covariance is declared on the active observed values.
  `restrict_observation_covariance` forms the exact Gaussian marginal for
  diagonal, diagonal-plus-low-rank, dense Cholesky, and dense precision
  covariances and refuses structured partial restrictions.

Least-squares training may use an explicit `reference_scale` weighting; posterior
inference requires a noise model. `posterior_problem_from_solver_objective` keeps
its fixed-noise, prewhitened contract, whose omitted normalization is constant.
Noise that depends on inferred parameters uses the normalized
`FixedObservationLikelihood` or `LinearizedGaussianMeasurementLikelihood` terms,
whose values and derivatives include the covariance log determinant.

## Physical exchange and temporal coupling

Partitioned participants exchange physical amounts through the canonical runtime
in `phydrax.solver.coupling`; the
[partitioned coupling guide](guides_partitioned_coupling.md) states the complete
contract.

- **Measurement functionals.** A port's physical inventory is a prepared
  `CouplingMeasurement`: a linear functional from native coordinates to one amount
  per declared component, with a unit, a `"density"`, `"extensive"`, or declared
  `"functional"` representation, and support and provenance identities. It is
  separate from the port's `reference_scale` and vector-space pairing, which only
  define the interface residual norm. Density storage (point values, cell
  averages, basis coefficients) is integrated against a physical measure,
  extensive storage (cell integrals, flux moments, circulation moments, cochains)
  is summed exactly once, and other storage requires a declared functional.
  Conservative transfers are certified by `L_target P = L_source` per component
  through transposed actions, and ledger rows are kept per component for
  componentized inventories.
- **Temporal conversions.** Each exchange declares one `CouplingTemporalConversion`
  (`"sample-end"`, `"hold"`, `"interpolate"`, `"window-integral"`, or
  `"integrate"`) matching its ports; only instantaneous endpoint-to-endpoint
  exchanges declare none. Preparation refuses a missing or mismatched conversion,
  and an endpoint value is never multiplied by the window as an exact integral.
- **Method participants.** `FixedStepCouplingParticipant` (fixed-step methods and
  `ConservationIMEXMethod`), `DAECouplingParticipant` (adaptive, event-free
  `PreparedDAESolve` carrying its `DAEContinuation`), and
  `SteadyResponseCouplingParticipant` bind native owners. Their checkpoint holds
  the native continuation, model state, and carried key data, so implicit iterates
  and adaptive retries replay from one accepted state. A steady response is not a
  transient solver.
- **Lowering.** `PartitionedCouplingDeclaration` and `lower_partitioned_coupling`
  produce an ordinary `CouplingProblem`, deriving initial exchange values from
  method participant checkpoints; no additional window loop exists.
- **Work evidence.** Fixed-point interface iterations declare each evaluation's
  participant work to `FixedPointIteration` (`FixedPointProblem(...,
  evaluation_work=True)`), which returns the exact sum over every executed iterate
  in `NonlinearResult.evaluation_work`. Window diagnostics therefore count rejected
  iterates and adaptive rollouts sum rejected attempts; both report
  `counts_complete=True` when every participant reports complete counts. A
  successful adaptive window without a reliable finite error estimate ends with
  status `UNRELIABLE_ERROR_ESTIMATE`, never `SUCCESS`.

Mixed-method workflow: `examples/mixed_method_time_coupling.py` couples a P1
finite-element solid and a cell-centered finite-volume fluid across a nonmatching
interface. The solid takes backward Euler steps with one prepared LU factorization
reused for every step; the fluid takes SSPRK(3,3) steps with a face-heat
accumulator in its state, and the two use different step counts per window. The
solid interface-temperature waveform reaches the fluid through `"interpolate"` and
exact face averages `I`. The fluid's whole-window face heat (extensive flux moments)
reaches the solid nodal loads, measured by a counting functional, through the
certified conservative map `P = Iᵀ` and `"window-integral"`. Each window is solved
implicitly with a fluid-first Gauss--Seidel sweep. Against `scipy.linalg.expm` of
the assembled coupled semi-discrete system, both integrator pairings converge at
first order in the window size (observed rates 1.04 to 1.14). The ledger rows
balance and the total energy error is about `6e-16`. A rejected window leaves the
fluid's carried key and model state at the checkpoint, and its replay is bitwise
identical. `tests/integration/test_mixed_method_time_coupling.py` asserts the same
contracts.

Explicit nonclaims: general-root implicit policies (for example Newton or Broyden
interface solves) report exact evaluation counts but only the final evaluation's
participant work, with `counts_complete=False`, because those nonlinear methods do
not accumulate per-evaluation work. DAE participants refuse waveform inputs, and
fixed-grid or event-driven DAE solves are refused as participants. Participants
with external randomness are refused by implicit cycles and adaptive windows.

## Block coordinates, condensation, and transient systems

Named block coordinates describe existing native coordinates; they never replace a
`BlockSpace`, a native `ArraySpace`, or the DAE stage spaces.

- **Named algebra.** `phydrax.linalg.BlockSelection` selects members of a
  `BlockSpace` by path, groups them, or identifies explicit coordinate ranges of
  any space with another member space (`CoordinateBlock`). Its restriction and
  prolongation are exact reversed coordinate maps; Hilbert adjoints use the
  declared pairings. `assemble_block_operator` builds explicit grids from named
  entries, and `select_block_operator` extracts the exact named grid of an operator,
  traversing explicit grids, identity-ordered `MappedBlockLinearOperator` values,
  and Newton's canonical coordinate view without densifying. Subspace correction
  consumes named transfers as exact local block grids, so the existing 2 by 2
  `BlockFactorizationPreconditionerBuilder` and its Schur setup serve any named
  two-group split of an N-field operator.
- **Reduced DAE coordinates.** `phydrax.solver.DAECoordinateAdapter` admits only a
  `ReducedDAECompilation`, so declarations refused by structural analysis (for
  example an unreduced index-two constraint beyond the declared differentiation
  capacity) never reach the runtime. It names variables, reduced equations, and
  kinematic rows with their native ranges, roles, and scales. A physical
  named-block linearization, evaluated at each root's physical state and rate, is
  mapped exactly to the stage, consistency, and bordered event roots, to tangent
  solves, and, by coordinate transpose, to adjoint solves. The residual scale is
  applied once; the native Hilbert adjoint remains separate from the transpose.
- **Named preconditioning.** `correction_transfers` gathers named residual rows and
  scatters corrections into named root columns in the canonical Newton
  coordinates, so a single subspace-correction term with a block-factorization
  local solver preconditions the native root by its named blocks. The original
  DAE residual still certifies every accepted step.

Explicit nonclaims: named DAE coordinates require a structurally admitted
compilation and block-aligned initialization masks; semidiscrete layouts that
interleave field components are not named blocks. Correction transfers are specific
to one root kind and orientation; adjoint derivative solves apply them to the
coordinate transpose, where an exact local solver remains exact only when the
transposed local pivot is nonsingular.

### Field-split preconditioning of a coupled problem

The named solve layout of a `PreparedCoupledProblem` is an ordinary nested
`BlockSpace`, so the existing builders precondition the original coupled system
directly. For a mortar saddle point, group the component blocks against the law
multiplier and give the 2 by 2 builder an approximate pivot action:

```python
la = phx.linalg
split = la.BlockSelection(
    prepared.state_space,
    (("primal", (("left", "u"), ("right", "u"))), ("multiplier", (("gamma", "multiplier"),))),
)
term = la.SubspaceCorrectionTerm(
    la.BlockRestrictionLinearOperator(split),
    la.BlockProlongationLinearOperator(split),
    la.BlockFactorizationPreconditionerBuilder(
        la.JacobiPreconditionerBuilder(), la.DenseInversePreconditionerBuilder(), "upper"
    ),
)
policy = la.LinearSolvePolicy(
    la.FGMRES(restart=200),
    preconditioning=la.PreconditioningPolicy(la.AdditiveSubspaceCorrectionBuilder((term,))),
)
solution = cpl.solve_coupled_problem(prepared, policy=policy)
```

Subspace correction hands the builder the exact named 2 by 2 grid
(`select_block_operator`), and the Schur setup `D - C M B` uses the approximate
pivot action `M`. On a P2 finite-element mortar problem (119 unknowns) FGMRES
needs 60 instead of 101 iterations at relative tolerance `1e-11`. The approximate
pivot action only preconditions: the Krylov solve still acts on the original
operator, and `solution.accepted` still requires the original component
certificates and interface defects. With a starved iteration budget the same
preconditioner returns `native_successful = False` and `accepted = False`.

### Exact static condensation

`condense_coupled_problem(prepared, *, pivot, arguments, factorization,
schur_policy, materialization, condition_limit)` eliminates named `(owner, block)`
solve paths of an affine coupled problem exactly; the retained set is their
complement. The pivot grid from `select_block_operator` is factorized once on the
host by `phydrax.linalg.factorize` at the nominal arguments, and the preparation
refuses, with the numbers in the message, a rank-deficient pivot, a condition
estimate above `condition_limit` (which requires a kind that reports one, such as
`"svd"`), dense blocks beyond the materialization budget, empty or unknown path
sets, nonlinear execution, and problems with a declared kernel.

`solve_condensed_problem(condensed, *, arguments, parameters, tolerance)`
refreshes the pivot factorization natively for the runtime arguments (never
reusing the nominal factors), performs one batched pivot solve `A^{-1}[B | b_p]`,
solves the materialized Schur complement `S = D - C A^{-1} B` with
`schur_policy`, reconstructs `x_p = A^{-1}(b_p - B x_r)`, and certifies the
original coupled equations at the reconstructed full state with
`certify_coupled_state`. `CondensedSolution` keeps every `LinearSolveResult`
(elimination, Schur, reconstruction) with its per-right-hand-side status and the
refreshed `CondensationEvidence` (rank, condition estimate or NaN when the
factorization reports none, pivot status, full rank, condition-limit check). No
solve is wrapped as a matrix-vector product that drops its status. On the P2
mortar problem the condensed and uncondensed direct solutions agree to `1e-14`.

Derivatives follow the factorization's `DifferentiationPolicy`: `"mathematical"`
admits implicit derivatives with respect to `arguments` (operator coefficients and
loads alike; JVP, VJP, and a host central difference agree with the uncondensed
solution map to `1e-14` and `2e-9`), `"none"` stops the route and any derivative
through `arguments` raises `derivative-unsupported`, and `"rhs-only"` or
`"algorithmic"` are refused at condensation because a partial or unrolled pivot
derivative is not the eliminated solution-map derivative. The default LU pivot
factorization and dense LU `schur_policy` declare
`LinearDerivativeSolvePolicy(route="primal-factors")`, so the implicit pivot and
Schur derivative solves reuse their factors; an explicit `factorization` or
`schur_policy` carries its own declared derivative route.

### Transient coupled problems

`prepare_coupled_transient(prepared, *, fields, arguments, arguments_id,
quasistatic, scales, parameters)` turns a prepared spatial problem into one native
differential-algebraic system without a second assembler:

- Each `TransientField(component, field, capacity=...)` declares a differential
  field. The component must publish `AbstractCapacityComponent.prepare_capacity`:
  the owner's own mass operator, compiled on the host once
  (`CompiledFiniteElementProblem.prepare_mass` for finite elements, the
  virtual-element mass action for virtual elements). Every other component must be
  named in `quasistatic`; law unknowns are algebraic. Nothing is inferred.
- `arguments(time, parameters)` returns the owners' runtime arguments at `time`
  (moving lifts, sources, coefficients). The rows are
  `F(t, z, z') = L^T [ P^T M d/dt u(t, z) + R(L z + h(t)) ]`, and `d/dt u` is one
  forward derivative in `(z, t)` with tangent `(z', 1)`. Time-dependent lifts, and
  eliminated coordinates that follow a moving retained field, therefore carry
  their exact rate terms.
- The declared block incidence (owner rows on their states, capacity rows on
  their field rates, law contributions on their sources, closed under the
  chart's eliminations) is lowered through the existing structural compiler with
  no differentiation or tearing capacity. Only structurally index-one systems are
  admitted. A mortar multiplier constraint on differential traces is refused with
  its analysis status (`differentiation-capacity-exceeded`), required
  differentiations, and matching; select a `MatchingElimination` or a
  multiplier-free imposition instead. A transient field eliminated in favor of a
  quasistatic field is refused as well, because its capacity would act on an
  algebraic variable.
- `TransientScale(owner, block, state=..., rate=..., residual=...)` sets the
  native scales of one solve block and its paired rows. `PreparedCoupledTransient`
  exposes the compilation, the structural analysis, its `DAECoordinateAdapter`
  (named variables, equations, roles, and scales), and exact `state_view` /
  `native_state` maps to the nested solve layout.

`solve_coupled_transient(transient, initial_state, time_grid, *, parameters,
policy, continuation, tolerance)` prepares the native DAE on the host, integrates
it with the native BDF or theta runtime (structural consistent initialization,
adaptive steps, regularity evidence), and certifies the original transient rows at
every sample per solve owner (`TransientCertificate`). Like the steady component
certificates, an owner's scale sums the norms of its individual terms: the
state-dependent steady rows `R(t, z) - R(t, 0)`, the steady offset `R(t, 0)`
(sources and lifts), the rate-dependent capacity rows, the capacity offset (lift
rates), and the component's full-field capacity action `M d/dt u` before it is
pulled back to solve rows. That last term matters when pulled-back capacity rows
cancel, for example on the retained side of a matching elimination
(`M_ii u'_i + M_ie u'_e`) at a state at rest. A quasistatic owner has no capacity
term, so its row defect is measured against its stiffness and forcing terms,
never against the defect itself. `solution.accepted` requires native success and
every certified sample.

An adaptive run continues from `solution.continuation`: the accepted BDF history,
the step controller, and the modified-Newton reuse state (Jacobian age, last
shift, the iteration count of the last accepted stage, a pending refresh). The
continued window therefore makes the same step and Jacobian-reuse decisions, and
its accepted times and final state equal those of one uninterrupted segment
exactly.

The stage residual of a capacity row is dominated by `C / h`. At small adaptive
steps the Newton correction that removes a residual of `1e-10` is smaller than the
default state step floor of `NonlinearTermination` (`absolute_step=1e-12`), so
the solver would report a converging stage as stagnated. The examples and tests
therefore certify stages by the residual threshold alone
(`absolute_step=0.0, relative_step=0.0`) and bound the work with `maximum_steps`.

`examples/coupled_transient_fields.py` runs transient conjugate heat conduction
across a finite-element and a virtual-element region with an oscillating wall
temperature (a moving lift) and a time-dependent source, compares fixed-step BDF2
with the matrix exponential of the semi-discrete system, continues an adaptive run
across two windows, and shows the refusal of the same problem declared with a
mortar multiplier:

```text
solve blocks: conductor/u, insulator/u
roles: ('differential', 'differential') | structural index: 1
BDF2 dt=0.0400: relative error 1.123e-03, max certified row defect 7.59e-10
BDF2 dt=0.0200: relative error 2.645e-04, max certified row defect 1.44e-09
BDF2 dt=0.0100: relative error 6.397e-05, max certified row defect 7.07e-10
observed temporal orders: 2.09, 2.05
adaptive BDF: 907 accepted steps in one segment, 907 across two windows joined by the accepted history; max |window - segment| 0.00e+00; relative error 1.64e-06
final temperatures: conductor in [0.0000, 0.7174], insulator in [0.2818, 1.4347]
mortar transient refused: Coupled transient refused: structural DAE analysis status 'differentiation-capacity-exceeded' (index-one admission, no differentiation or tearing)
```

The unit tests compare a finite-element pair with a fully independent host P1
assembly on the union mesh: fixed-step BDF2 converges at second order to the
matrix exponential of the semi-discrete system, two adaptive windows reproduce the
accepted times and final state of one segment exactly, and a quasistatic region
solved with unequal scales and mixed roles matches a native array DAE to `1e-9`.

Explicit nonclaims: no index reduction of coupled multiplier systems, no
time-dependent exterior boundary-integral model (a quasistatic exterior is allowed
only when declared quasistatic), no condensation of nonlinear or gauged problems,
and no derivative through an approximate or frozen pivot action; those remain
preconditioners of the original system.

## Control, state-space inference, reduced models, and hybrid learning

Control, state-space, reduced-order, and learning consumers keep their existing
contracts. A coupled problem reaches them through its parameter and observation
bindings and an accepted coupled transition; an owner adapter exists only where
an abstract consumer cannot take those outputs directly.

### Coupled transitions

`prepare_coupled_transition(transient, *, step, policy, parameters, control=None,
substeps=1, tolerance=1e-6)` turns a `PreparedCoupledTransient` into one native
discrete transition between two physical times. The native DAE solve (a
`PreparedDAESolve` with the given `DAESolvePolicy`) is prepared once on a template
window of `substeps` uniform steps of length `step / substeps`; every call rebinds
the window's save times to its own `[source, target]`, the same pattern as the
partitioned DAE participant. `PreparedCoupledTransition.step(source, target, state,
parameters)` returns a `CoupledTransitionStep`:

- `candidate`: the native end state; `accepted`: the candidate when the step is
  accepted, otherwise the unchanged source state (never a repaired value);
- `successful`: native DAE success, a `TransientCertificate` of the original
  coupled transient rows at every window sample, and a finite state;
- `status` (`coupled_transition_status_name`: `"success"`, `"native-failure"`,
  `"certificate-failure"`, `"nonfinite"`), `native_status`, and the largest
  certified row defect relative to its term scale (`residual_ratio`).

The state is the native transient coordinate vector with Euclidean (exact)
geometry, and the step is a Markov map: each window starts the native method from
its own start state with no history carried across windows. The runtime DAE
arguments are the values of every refresh `ParameterBinding` of the prepared
problem, keyed by binding ID; the transient's `arguments(time, parameters)` binds
them with `PreparedCoupledProblem.bind_arguments`. One binding with a rank-one port
may be declared the `control`; preparation refuses a control that is not a
refresh binding or whose components do not enter the coupled transient rows (a
forward-derivative incidence check at the reference values, probed at the origin
and at fixed pseudo-random states, rates, and times within one step, so a control
entering as `u z` or `u t` is admitted; non-finite probes are ignored).

Derivatives flow through the native solve with respect to the state and the
parameter values. Derivatives with respect to the step times are refused with
`derivative-unsupported` (the native solve stops time derivatives), and
derivatives of a non-accepted step are NaN.

`transition.observation_port(binding_ids)` is a `CoupledObservationPort` over the
prepared problem's observation bindings: `measure(time, state, parameters)`
returns the predicted `PreparedQuantityField`s (with the owners' approximation
labels, e.g. `"h1-projection"` for virtual elements) on the full fields, lifts
included, and `values` concatenates them. The control-output and state-space
location ABIs have no validity channel, so the port refuses a binding with
samples that are never valid (a masked point observation with unlocated or
inactive samples). It also refuses `FieldFluxObservation`: its steady residual
reaction omits the capacity action `C u'` of the transient reaction.

### Control of a coupled transient

`transition.discrete_system(system_id=...)` is a `DiscreteSystem` whose transition
runs the coupled step on the `DiscreteStepContext` interval
`(source, target, step_index)`, reads the control from the input (an
`InputLayout` built from the control port's shape and component IDs), reads the
other refresh parameters from the system `args`, and returns a
`DiscreteTransitionResult` with the candidate, accepted state, success, and status.
Every existing consumer of `DiscreteControlDynamics` applies unchanged:

- `ControlProblem(...).rollout` records per-step candidate/accepted states and
  status in `DiscreteTransitionEvidence`; a failed step stops the trajectory
  (`CONTROL_DYNAMICS_FAILED`) and is never repaired.
- `prepare_control_linearization(dynamics, t, x, g, target_time=..., step_index=...,
  output=port)` gives matrix-free `[A B]` and `[C D]` actions; the step context
  selects the interval, so linearizing over `t -> t + 2 dt` gives the two-step
  operator. The port's `__call__(time, state, control, args)` is the control output
  ABI.
- `linearize_discrete_dynamics` and `linear_quadratic_problem_from_discrete_dynamics`
  materialize dense stage Jacobians only under an explicit `MaterializationPolicy`,
  and refuse failed stages ("failed accepted-state transitions are not repaired").
- `solve_receding_horizon_mpc`/`RecedingHorizonMPC` solve the dense LQ windows;
  `prepare_receding_horizon_mpc_sensitivity` keeps its regularity gate, so a
  weakly active (tied) control bound is refused, not differentiated.

`examples/coupled_control.py` controls the right-wall heat flux
`g = (g_lower, g_upper)` of a P1 finite-element / degree-1 virtual-element
transient (matching elimination, owners' capacities, implicit Euler) to drive
three sensor temperatures to a target under `|g| <= 0.6`. It checks the rollout
of random fluxes against the host implicit-Euler recursion of the semi-discrete
system and the sensor rows assembled in NumPy from the two meshes (P1 and
degree-1 VEM mass and stiffness weighted by each owner's `ρc`, barycentric and
projected point evaluation; only the native coordinate chart, verified against
the mesh geometry, is read from the prepared transient), the
matrix-free actions over a two-step interval against the host operator, the
port's sensor map against the host sensor rows, and the
dense MPC controls against the host bound-constrained least-squares optimum
(BVLS) of the same finite-horizon problem, replays the MPC controls through the
coupled transition, and shows the same materialization budget refusing the
refined full-order plant. `tests/integration/test_coupled_control.py` covers the
same contracts plus the failed-transition and tied-bound refusals.

### State-space inference with a coupled transition

`phydrax.stochastic.CoupledTransitionKernel(transition, *, noise_factor=None)` is
the `AbstractTransitionKernel` of an autonomous coupled transition (known forcing
values are refresh parameters in the problem `args`, read from `context.args`):
`x' = f(x) + L w`, `w ~ N(0, I_r)`, with the declared `(n, r)` process-noise
factor `L`. The consumer's key is the only randomness, so the semantic
case/step/member key addressing of `state_space_key` is preserved. A failed step
is an invalid sample with the step status. The normalized density
`N(x'; f(x), L L^T)` exists only for a nonsingular covariance: a deterministic
simulator (no factor) or a rank-deficient factor (for example uncertainty in the
wall flux only) has `has_log_density=False`, so density consumers
(`state_space_path_log_density`, density-based particle smoothing, guided
proposals) refuse it.

`port.location(state, time, context)` is the state-space observation location, and
`port.noise_covariance(plans)` derives the Gaussian observation covariance from
the declared noise of one `MeasurementComparisonPlan` per binding: independent
standard uncertainty gives `diag(std^2)`, a diagonal or dense Cholesky covariance
gives its matrix; unquantified data or a least-squares reference weighting is
refused, and each observed field's quantity, layout, support, sampling, and unit
identities must equal the predicted ones. The observation values and masks of an
`ObservationSequence` come from the same records at their physical sample times.

`examples/coupled_data_assimilation.py` runs the existing
`ensemble_transform_kalman_filter` on the same FE-VEM transient with known wall
flux, declared process noise, three sensors at non-uniform times, and one sensor
dropout. The reference is the exact linear-Gaussian Kalman filter of the host
semi-discrete system and sensor rows (the same host assembly as the control
example): the ETKF analysis means stay within the ensemble sampling error of the
Kalman means, and one analysis step equals the exact Kalman update of its own
forecast ensemble to roundoff.
`tests/integration/test_coupled_state_space.py` also reproduces each forecast
member from its semantic key on two physical cases, checks every forecast,
analysis, and incremental log-likelihood of the full ETKF run against the host
step of the previous analysis, the exact Kalman update, and the Gaussian
innovation likelihood of its forecast sample statistics, and exercises the
failed-transition (rollback to a nonzero source state), missing-density, and
undeclared-noise refusals.

The ETKF log-likelihood is an ensemble Gaussian approximation. The exact Kalman
likelihood exists in this example only because the coupled step is affine and the
noise Gaussian; no exact-likelihood claim is made for a nonlinear simulator.

### Reduced-order components

A region of a spatial coupled problem can be replaced by a prepared Galerkin
reduced-order model without touching any binding, law, parameter, or
observation of the plan. `ComponentResidualProvider(component, *, field)`
publishes a single-field component's steady residual through the ROM owner's
`AbstractResidualProvider` contract, and `ReducedComponent(name, galerkin)`
publishes the resulting `phx.rom.FullResidualGalerkin` back to the assembler.

- The provider's `state_space` is the component's state block (its solve
  coordinates), `residual_space` the dual, `support_id` the field's discrete
  space, and `geometry_id` the owner identity; `FullResidualGalerkin` refuses a
  basis fitted on another component, mesh, or field.
- Offline, the snapshots are the component's block `solution.state[i][0]` of
  full-order `solve_coupled_problem` solutions at training parameters; an
  uncentered `PhysicalPODPlan` on `provider.state_space` gives the basis, bound
  as a `ReducedBasisArtifact` and reduced with
  `trial_test_reduction_from_bases(basis)` (Galerkin).
- `ReducedComponent` keeps the full component's name, field-space identity,
  boundary impositions, side traces, conormal reaction fluxes, and pointwise
  reconstruction, all acting on the reconstructed field `P (V a + l) + g`, so
  point and boundary observations keep their identities and exact
  evaluation. Its rows are the Galerkin rows `V^T R`; a linear owner
  contributes the dense projected operator `V^T A V`, a nonaffine owner is
  solved by Newton.
- Parameter bindings still reach the full owner through its runtime arguments;
  implicit derivatives through the ROM-coupled solve are the ordinary coupled
  parameter derivatives.
- Certificates: the component certificate of the ROM is its reduced rows; a
  mortar's `weak-continuity` holds exactly at every rank, while its
  `flux-balance` compares the full owner's reaction of the reconstructed field
  with the multiplier. That balance is the interface residual of the ROM: it
  decays with the rank, and the solution is `accepted` only once the basis
  spans the region's parametric solution manifold.
- The basis, provider, and owner topology are fixed prepared structure (their
  leaves resolve as `FIXED`). A new basis is a new `ReducedComponent` with a new
  `owner_id` and a new prepared problem, never an online gradient. Refused: a
  Petrov–Galerkin reduction (`ValueError`), a model that is not a
  `FullResidualGalerkin` or a provider that is not a `ComponentResidualProvider`
  (`TypeError`), a trainable model hidden in the provider, which would be frozen
  silently (`ValueError`; bind it through a `ParameterBinding`), and an owner
  that publishes a kernel.

`examples/coupled_rom_swap.py` replaces the P1 triangles of the FE–VEM plate by
POD Galerkin models of increasing rank fitted to full-order solves at seeded
training conductivities and wall fluxes. At held-out parameters it tabulates the
relative errors of both regions' fields, the multiplier, and the observations
against the full-order coupled solution, the interface defects, and acceptance,
and compares the spanning-rank solution with the host analytic temperature.
`tests/integration/test_coupled_rom.py` covers the retained identities, the
rank decay, the interface certificate, the implicit parameter derivatives, the
reprepare contract, and the refusals. The model is full-order assisted (every
residual and operator action calls the original owner); it demonstrates the
coupling contract, not a hyperreduced speedup.

### Hybrid PINN and classical responses

A physics-informed network keeps its `SURROGATE` binding when it meets a
classical owner. Three existing routes carry the hybrid workflow; none of them is
a hybrid optimizer, and none widens what the surrogate authority admits.

1. **Direct pretraining.** `phx.solver.FunctionalSolver` trains the network on
   its region's physical residuals (direct `PHYSICAL_RESIDUAL` training, which
   `SURROGATE` admits). The network is bound once,
   `phx.ComponentBinding(network, authority=phx.ComponentAuthority.SURROGATE)`,
   and wrapped with `domain.Model(...)` for training.
2. **Accepted classical response.** The classical owner reads the network through
   its runtime user arguments (for example a `BoundaryLoadAction` whose
   coefficient evaluates the network's conormal heat flux at the owner's facet
   points). `phx.optim.StateDesignProblem` takes the owner's solve coordinates as
   state and the network's PARAMETER lane as design; `phx.combine_parameters`
   recombines the network inside the residual from the held MODEL_STATE and
   FIXED lanes passed through `args`. The owner's admission
   `StateDesignComponentAdmission(binding, *, kind, surfaces, policy)` forms the
   binding's contract for the requested derivative surfaces (for example
   `INPUT` for the flux and `MODEL_PARAMETER` for the design), requires the
   model's `DIRECT` route, and requires `authority_admits(binding.authority,
   DIRECT, kind)`. `prepare_state_design_linearization(problem,
   admission.design(binding), initial_state, *, args, linear_policy,
   component=admission)` solves and accepts the classical state;
   `state_design_response_vjp` returns the response, its design cotangent,
   `state_acceptance` (accepted primal), `adjoint_acceptance` (accepted
   transpose solve), the joint `accepted` flag, and the `component` admission.
3. **Dirichlet–Neumann coupling.** The accepted classical coefficients become a
   fixed `phx.discretization.DiscreteFieldFunctionView` whose reconstruction
   declares a typed `value_port` (for example a temperature port). Read at the
   interface sites with valid `view.query(points)` evidence, it is the target of
   the network's interface `phx.conditions.Observation`, and the same
   `FunctionalSolver` continues training. Field algebra between the view and the
   network requires the network to declare the same port: an unported network is
   refused ("requires the other field to declare a value port"), so the network
   publishes it through `model_ports()` and is bound with an explicit
   `PortMapping`. A new accepted classical response reports the coupled
   interface residual.

A port-declaring wrapper that does not re-declare its inner network's smooth
regularity keeps the conservative undeclared contract; its admission then needs
`policy=phx.RegularityPolicy(allow_undeclared=True)` and carries the
`"regularity-undeclared"` condition, the same exploratory policy under which
`FunctionalSolver` trains it.

Refusals keep the authority honest: `StateDesignComponentAdmission(binding,
kind=ObjectiveKind.SOLUTION_MAP)` of a surrogate raises `ValueError` ("cannot
supply the design of a state-design response consumed under (direct,
solution-map)"); a `phx.solver.SolverObjective` (implicit solution map) refuses
to train the same binding with "no admissible training signal"; a design that
is not the admitted PARAMETER lane (another structure, shape, or dtype) is
refused before the state solve. `authority_admits` is unchanged.

`examples/hybrid_pinn_classical.py` runs the workflow on the plate
`[0, 2] x [0, 1]` (`-div(kappa grad u) = s`, `u = 0` at `x = 0`, heat flux `g`
at `x = 2`, insulated top and bottom) with P1 triangles on `[0, 1]^2` and a
`tanh` MLP on `[1, 2] x [0, 1]`. Pretraining holds the interface at `u = 0`;
the right region's interface heat flux `kappa_right u_x(1) = s + g` does not
depend on that value, so the finite-element owner loaded with the network's
flux already reaches its discretization floor. The accepted response is
the interface-continuity residual `J = 1/2 sum_w (u_theta - u_FE)^2` at the FE
interface trace sites; its design cotangent matches central finite differences
of host dense solves (`tests/integration/test_hybrid_numerical_field.py`), and
one Dirichlet–Neumann step closes the interface.

The coupled accuracy is the classical discretization error plus the network's
interface-trace error, not the network's training accuracy. P1 is not nodally
exact for this `x`-only quadratic: the diagonal triangulation gives the
interface corners one and two triangles, so their consistent source loads
differ from the symmetric quarter cell. With the exact interface flux the FE
nodal error is 1.29e-2 / 4.13e-3 / 1.25e-3 with 4 / 8 / 16 mesh intervals per side
(`O(h^2 |log h|)`), largest at the corner `(1, 0)`. The P1 stiffness of this
mesh is an M-matrix and a unit interface flux loads the exactly representable
`x / kappa_left`, so a network flux error `dg` moves the FE nodes by at most
`max|dg| / kappa_left`. With 8 intervals per side the pretrained network's flux
error is 2.7e-5 and the FE error is 4.13e-3 (the floor). The coupling step
fits the network to that FE trace, so the network inherits the trace error
(2.6e-3 against the exact plate) and a zero-mean flux error (6.0e-3 peak) whose
FE response (1.3e-3) partly cancels the corner error: the coupled FE error is
2.85e-3, within `floor + max|dg| / kappa_left`. The integration test asserts
these bounds and the floor's convergence rather than one absolute tolerance.

### Learned interface laws

`MonotoneInterfaceConductance(response, *, baseline, quadrature_degree)` is a
learned contact conductance for `ConservativeFluxLaw` that is monotone and
dissipative by construction. With the temperature jump `d = u_minus - u_plus`,
the heat leaving the minus side is `q(d) = h d + φ'(d) - φ'(0)`, where
`h = baseline >= 0` is a declared conductance and `φ` a learned scalar
potential; the minus conormal flux density is `-q`. A convex `φ` makes `q`
nondecreasing with `q(0) = 0` for every parameter value, so heat always flows
from hot to cold and the interface power `∫ d q(d)` is nonnegative. The law
reads `φ` from the coupled runtime arguments at every solve, never from its own
fields:

```python
response = cpl.RuntimeInput("polygons", "contact-potential")
flux = cpl.MonotoneInterfaceConductance(response, baseline=0.5)
law = cpl.ConservativeFluxLaw("contact", binding, sides, flux)
contact = cpl.ParameterBinding(
    "contact-potential", potential_port, targets=(response,),
    role="coefficient", derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
)
potential = phx.bind_component(
    phx.nn.models.InputConvexNetwork(in_size="scalar", width_size=8, depth=2, key=key),
    phx.ComponentAuthority.MODEL,
    owner_ports=phx.ModelPorts(inputs=(jump_port,), outputs=(potential_port,)),
)
solution = cpl.solve_coupled_problem(
    prepared, parameters={..., "contact-potential": potential}, policy=policy
)
```

Physical validity is never inferred from the model port. The flux requires the
scalar potential's `input-convex` construction certificate (published by
`InputConvexNetwork` from its positive hidden couplings), so a plain `MLP`
potential is refused when the flux is evaluated ("requires a monotone
response"). The value reaches the law only
through the `ParameterBinding` authority check: a `SURROGATE`, `ACCELERATOR`,
or `DECISION` binding is refused like any other learned parameter value. A
negative or non-finite baseline is refused at construction.

A flux that reads runtime arguments names them in
`AbstractInterfaceFlux.runtime_inputs`; preparation then never probes it
without arguments, refuses an affine declaration (so the coupled problem is
`"nonlinear"` and solves with Newton), and refuses the plan unless every named
input is the target of a refresh `ParameterBinding`, so a model supplied as a
raw runtime argument never bypasses the authority and port admission. The law
gates `law-residual` and
`flux-conservation` as for every `ConservativeFluxLaw`. `quadrature_degree` is
the declared exactness of the common interface quadrature: the density is not
polynomial in the traces, so the interface integral is part of the law's
discretization.

Derivatives follow the fixed-structure route. `prepared.derivative_capability`
refuses every parameter of a nonlinear coupled problem (route `STOPPED`, "the
nonlinear coupled solve publishes no qualified implicit solution-map
derivative"). The response of the accepted state to the potential's parameters
is a `StateDesignProblem` instead: the state is the solve-coordinate state of
`prepared.state_space`, the design is the potential's PARAMETER lane admitted
with `StateDesignComponentAdmission(potential, kind=ObjectiveKind.DATA_FIT)`,
and the residual is `prepared.residual(state, arguments)` with the model
recombined from the design (`phx.combine_parameters(design, model_state, fixed)`)
in the bound arguments. `prepare_state_design_linearization(..., component=
admission, linear_policy=dense LU)` certifies the state and
`state_design_response_vjp` reports the primal and adjoint acceptance separately
from the design cotangent. Warm-started from the accepted Newton state, the
state owner's `LeastSquaresStateSolver` needs an
`OptimizationTermination(absolute_optimality=...)` at the scale of its residual
acceptance: the default `1e-14` gradient stop lies below the roundoff of a
converged state, the method reports stagnation, and the state is not accepted.

`tests/integration/test_coupled_learned_interface_law.py` replaces the mortar of
the finite-element / virtual-element plate by this law. The temperature depends
on `x` only and the right region's balance fixes the interface heat
`Q = -(s + g)`, so the host reference solves `q(d) = Q` for the jump with the
network's own derivative and shifts the analytic right-region temperature by it.
The tests check the accepted Newton solution against that reference; the side
injections against host Gauss quadrature of `q(d(y))` and `d q(d)` on the
piecewise-linear traces (equal and opposite heat `Q`, positive dissipation);
monotonicity for several random certified potentials; the linear limit (hidden
input weights zero, so `q = h d`) against `InterfaceConductance(h)`; the
refusals; and the state-design cotangent along a random parameter direction
against central differences of independent nonlinear coupled solves. With
`h = 0.5`, `g = 0.5`, conductivities 1 and 2, and a width-8, depth-2 network,
Newton is accepted with `law-residual` 9e-13 (scale 2.5) and
`flux-conservation` 4e-16, the jump magnitude (`trace-jump-l2`) is 0.89 instead
of the linear conductance's `|Q| / h = 5`, the dissipated power is 2.23, and the
state-design cotangent matches the central difference (step 1e-4) to 1e-9.

## Lifecycle, adaptation, and junctions

A running coupled problem is a *composition*: owner artifacts (topologies,
discretizations, interface routes, transfers, preconditioners, observations,
prepared graphs, worksets) and owner state (physical, exchange, budget, model,
optimizer, history, statistics, RNG) bound at one accepted boundary. Changing a
mesh changes PyTree structure and array shapes, so a topology change is never a
selection between old and new prepared objects. `phx.lifecycle` owns only the
shared transaction invariant; every numerical operation stays with its owner.

### Entries, bindings, and dispositions

`CompositionEntry(value, *, entry_id, role, owner_id, structure_id, revision_id,
semantics_id, dependencies)` carries three explicit identities: the static
structure (topology, partition, layout, symbolic sparsity, routes), the exact
revision, and the scientific meaning. A consumer records what it was prepared
against with `entry.binding(facet)`. `Composition(entries, boundary_id=...)`
refuses any unresolved or stale binding, so an observation prepared on a coarse
support, a factorization of an old sparsity pattern, or an interface action on an
old geometry route can never be published beside a refined mesh. Model parameters
are bound by semantics only: a mesh-independent parameter, whether retained,
refreshed, or transported, survives a rebind only when a consumer of the
candidate binds its meaning again.

`CompositionRebind(source, *, retain, refresh, reprepare, transports, invalidate)`
stages everything on the host and assigns every source entry exactly one
disposition:

| Disposition | Admitted roles | Contract |
|---|---|---|
| retain | all | same object; all bindings must still hold |
| numeric refresh | all | same structure and meaning, new revision, identical PyTree layout, the same dependencies with unchanged structure and semantics bindings (only bound revisions may advance); committed through `TransactionalCandidate` |
| topology reprepare | derived artifacts | rebuilt whole by its owner from staged dependencies |
| physical remap / ownership migration | state | an owner `CompositionTransport` from the structure it was prepared for, with evidence |
| invalidate | derived artifacts | dropped; nothing may still bind it |

State is never reprepared from nothing, invalidated, or zero-filled: model state,
optimizer moments, histories, statistics, and RNG cross only by retention (their
bindings unchanged) or by an explicit owner transport, otherwise the rebind is
refused. Ownership migration must report the content it moves; created rows are
never physical content. A `budget` entry (a conservation ledger) crosses only by
retention, refresh, or a transport with conserved-content evidence.
`commit_composition_rebind(rebind, accepted_boundary=...)`
takes one explicit host decision and publishes only when the boundary is accepted
and every transport reports owner success and, when conservative, content
agreement within its owner tolerance. Otherwise the receipt carries the original
composition object, so every old owner stays usable. The receipt lists retained,
refreshed, reprepared, remapped, migrated, consumed, and invalidated entries and
the transport evidence; `Composition.structure_id` is the topology identity a
checkpoint binds.

### Owner hooks

| Owner | Hook |
|---|---|
| Finite elements | `FiniteElementTopologyTransfer.epoch_transition(...)` binds a certified conservative transfer (for example `vertex_interpolation_transfer`) as a `TopologyEpochTransition` with its own claims; the transition measures must satisfy the transfer's conservation certificate (`action_condition` times 64 ulp of the total measure per DOF), whose bound it carries |
| Finite volumes | `UnstructuredConservativeRemapPlan.epoch_transition(...)` binds a complete common-refinement cell remap, with its exact CSR transpose and the plan's per-cell coverage limits as its measure-defect bound |
| Topology epochs | `TopologyEpochTransition(..., measure_defect_bound=...)` carries the owner-certified per-DOF bound on `abs(Pᵀ m_target - m_source)`; its `apply` admits a content residual up to `100 eps (Σ m_source abs(v) + Σ m_target abs(P v)) + Σ bound abs(v)` (`TopologyEpochTransitionResult.content_tolerance`). `composition_transport(source, target)` reports that evidence and succeeds only if the staged target is its image |
| Partitioned coupling | `coupling_composition_entries`, `coupling_state_from_composition`, `coupling_exchange_transport` split an accepted `CouplingState` into participant native, model-state, key, and window entries, exchange values, the budget ledger (bound by its physical contract), the clock, and the prepared epoch; entry revisions fingerprint the boundary and the value. `coupling_exchange_transport` refuses a target epoch lowered at another time than the source clock, from other than the staged `<subsystem>/native` target entries, or from a native other than the accepted one on an unchanged structure |
| Distribution, training, worksets | `MeshDistributionTransition.composition_transport`, `PreparedTrainingKernel.composition_entries`, `PreparedExecutionWorksets.composition_entry` |
| Runtime checkpoints | `RuntimeRestartRelation.from_composition_rebind(receipt, restorer, classification=...)` |

### Adaptive FE/FV interface

`examples/adaptive_fe_fv_rebind.py` continues the nonmatching conjugate-heat
coupling of the mixed-method workflow and, at the accepted boundary `t = 0.2`,
refines one side:

- refining the P1 solid (4×4 to nested 8×8): its temperature crosses through the
  native FE vertex-interpolation transfer, certified linear and conservative
  against the P1 basis integrals and bound as a topology-epoch transition; the
  backward-Euler factorization and the prepared point query are reprepared;
- refining the FV fluid (3×3 to 6×6 cells): cell temperatures cross through the
  first-order common-refinement remap and the face-heat accumulators split to
  their child faces through the interface-face `EntityLineage`, both reported as
  one two-component transport; the fluid probe is reprepared.

Both variants reprepare the interface common refinement (`I`, `P = Iᵀ`) and the
prepared coupling epoch, re-derive the changed side's exchange value through the
new route, retain the consumed exchange budgets and the clock, and retain every
entry of the unchanged side as the same object. The run then continues the
accepted coupling windows:

```text
continued convergence to the rebound semi-discrete reference at t = 0.4
  refined side     window   max error   rate
  solid            0.1000   1.005e-02
  solid            0.0500   4.827e-03  1.058
  solid            0.0250   2.340e-03  1.045
  fluid            0.1000   1.063e-02
  fluid            0.0500   5.133e-03  1.051
  fluid            0.0250   2.461e-03  1.060
```

Heat content is unchanged by the rebind to roundoff, total energy over the whole
run changes only through the conservative ledger, and both probes read the same
physical values before and after the rebind (the nested prolongations are exact).
Retaining the coarse solid probe, carrying a fluid model state declared
grid-dependent without a transport, preparing a non-nested remeshed solid (the FE
owner refuses to certify conservation), and a rejected boundary all leave the
original composition in use; it continues bitwise like a never-rebound run.

### Checkpoint restart

A coupled checkpoint is the accepted `CouplingState` written through
`RuntimeCheckpointEnvelope`, with the composition `structure_id` as mesh
identity, the declared coupled problem as method identity, and the prepared epoch
as topology epoch. The participant checkpoints carry key data (never a typed key),
model state, and accepted-window histories; streaming observers ride as observer
states. A restart rebuilds every prepared owner from the declaration and, within
the identity relation, resumes bitwise. A checkpoint written before a rebind
restores into the rebound topology only through
`RuntimeRestartRelation.from_composition_rebind`, whose restorer replays the
rebind's owner transports; a changed partition or topology without such a
relation is refused, and no portable storage is inferred from in-memory objects.

### Foam sheets, Plateau-border junction, and rendering

`examples/foam_junction_rebind.py` runs the same transaction on a non-manifold
multiregion surface. The three sheets of a double bubble that meet its ring of
valence-three Plateau borders are declared once as one `"junction"`
`InterfaceBinding` of three `SheetViewAttachment` endpoints; no pairwise sheet
laws are generated. Every sheet boundary cell drains through its explicit
B-on-E half-edge routes (a declared first-order suction closure) and
`PreparedPlateauBorder.step` applies equal and opposite sheet and border rates,
so per border the six half-edges of the three sheets balance its inflow and
liquid and surfactant totals close to roundoff.

At an accepted boundary one border is split at its midpoint by the E event
pass. The rebind uses three owner hooks:

| Owner | Hook |
|---|---|
| Multiregion surfaces | `multiregion_topology_epoch(topology, positions)` is the structure identity of sheet-slot and border state; `SurfaceEventPassResult.transition.composition_transport` moves sheet-slot liquid and surfactant |
| Plateau borders | `PreparedPlateauBorder.reprepare_after_events(event, surface, film_slots, state)` prepares the target network and returns a `PlateauBorderAdaptation` whose `transition` keeps retained borders' content and shares a split border's content by child length |
| Film sheets | `PreparedFilmSheetSlots` owns no content and is re-prepared by construction on the target epoch |

The rebind reprepares the surface geometry, sheet views, junction binding,
border network, and rendering observation, remaps the four content fields, and
retains the unresolved-rim ledger. A retained border keeps its content bitwise;
both children of the split border keep the parent's cross-section `V / L` and
surfactant concentration. Thin-film colors are then rendered from the published
thickness `V_slot / A_slot`, and the physics state fingerprint is unchanged by
rendering. A burst (rupture) or a T1 pop has no declared border-content rule:
staging raises `PlateauBorderTransportError`, nothing is published, and the old
composition continues exchanging bitwise like a never-staged run. An omitted,
invalidated, or stale-retained border state and a rejected boundary are refused
the same way. The route transports extensive border content only; dynamic
meniscus reshaping, border coarsening, and post-event rim motion are not
inferred.

## External execution

External runtimes keep four distinct execution tiers; none acquires another's
capability by being wrapped:

| Tier | Owner | Derivatives |
|---|---|---|
| Forward oracle (pinned command, native worker) | `phydrax.interchange.external_runtime` | none; derivative-free methods are listed, never selected |
| Host inference | `HostInferenceAdapter` | none (`derivative_support.route == "none"`) |
| FMI co-simulation | `FMICoSimulationSession`, `FMICouplingParticipant` | none |
| Staged external adjoint | `ExternalAdjointAction` | reverse only, at the replayed realization of a staged primal |

An FMU joins a coupled problem through physical ports, not through a callback
hidden inside a native trace:

- **Binding.** `FMICouplingBinding` maps actual FMI2 communication variables to
  physically typed `CouplingPort`s (with `CouplingQuantity` and, for amounts,
  `CouplingMeasurement`) through four declared realizations: `"hold"` and
  `"uniform-rate"` inputs, `"sample"` and `"increment"` outputs. The model
  description decides: Real type, communication causality and variability, and a
  `BaseUnit`-declared unit whose SI dimension and factor must realize the port
  quantity exactly. Affine, undeclared, and dimension-mismatched units are refused.
- **Host boundary.** A declaration containing an `FMICouplingParticipant` is the
  same `PartitionedCouplingDeclaration` a native problem uses, but
  `lower_partitioned_coupling`, `prepare_coupling`, and the native window runtime
  refuse it. `prepare_host_coupling` validates it with the canonical route,
  physical-exchange, temporal-conversion, and certificate rules, and
  `advance_host_coupling_window` executes the declared sweep with the native window
  status, certification, ledger, and accepted-state transition. Native participants
  keep their compiled window maps; only the host lifecycle is added.
- **Lifecycle.** Implicit or retried host windows require a real save/restore of the
  FMU state (`canGetAndSetFMUstate`): every iterate restores the checkpoint, and a
  rejected window restores it. Without it only a declared explicit non-retrying
  route is admitted; a rejected window that advanced the FMU is reported as
  unrecoverable and the coupling refuses to continue.
- **Derivatives.** Any coupled differentiation request through a host participant
  is refused with its derivative route and the derivative-free alternatives. An
  existing `ExternalAdjointAction` remains usable only through its own staged
  primal and replay-checked adjoint.
- **Acausal models.** `phydrax.system_modeling` keeps its native algebraic compiler
  and native execution; FMI contracts declared there do not make it host-only.

Workflow: `examples/fmi_host_coupling.py` compiles Phydrax's original thermal-zone
FMU (`tests/interchange/data/thermal_zone.c`) and couples it to a native SSPRK(3,3)
node. The FMU holds the node temperature as its boundary and returns its cumulative
conducted heat as the node's whole-window heat loss, certified in the ledger. Both
the explicit zone-first sweep and the implicit fixed point (restoring the FMU per
iterate) converge at first order to the analytic two-capacity solution at `t = 1`
(`C1 = 2`, `C2 = 1`, `G = 0.5`, `T1(0) = 350 K`, `T2(0) = 300 K`):

| Route | `Δw = 0.1` | `0.05` | `0.025` | Observed rates | FMU steps / restores at `0.025` |
|---|---|---|---|---|---|
| explicit, no restore needed | `3.05e-1` | `1.50e-1` | `7.44e-2` | 1.02, 1.01 | 40 / 0 |
| implicit fixed point | `2.86e-1` | `1.45e-1` | `7.32e-2` | 0.98, 0.99 | 280 / 240 |

The heat content `C1 T1 + C2 T2` drifts by at most `1e-15` relative and every ledger
row balances exactly. `tests/interchange/test_fmi_composition.py` also checks each
route against its exact discrete recursion, a `"uniform-rate"` and `"sample"`
coupling in which the node owns the conduction, an FMU discard (early return at its
temperature cutout) that restores the zone, bitwise replay after a restored
rejection, an unrecoverable rejection without restore, and every refusal above.
Without FMPy or a C compiler these checks skip as inconclusive.

## Bounded and distributed execution

### Particle-in-cell field solvers

Changing the PIC field solver is construction of a new prepared solver: the same
`PICSpeciesPlan`s, prepared `PICParticleCochainTransferPlan` transfers,
`ChargeConservingCurrentPlan`s, processes, and `ElectromagneticPICPlan` options drive the Yee
cochain solver or Cartesian PSATD unchanged. A mid-run state handoff is a separate, explicit
operation: `hand_off_pic_state(source_plan, target_plan, state, step_size)` converts between
`CochainMaxwellPICFieldSolver` and `PreparedSpectralMaxwell` only for the pair whose conversion is
exact — a periodic uniform grid shared by both, a lossless stateless homogeneous medium, no
boundaries, CPML, observers, antennas, or PML memory, the same transfers and run declaration, and
PSATD with `grid="staggered"`, the standard variant, and the order-2 finite stencil, whose
divergence is the Yee node difference. Edge circulations and face fluxes map to the Yee-position
point values, so the Gauss and `∇·B` residuals, the synchronized field energy, and the total charge
are preserved to roundoff and reported in `PICFieldHandoffEvidence` (with the leapfrog energy term
by which the PIC ledger's field energy jumps). Because preservation carries a violation over, the
evidence's `constraint_satisfied` (required by `successful`) also checks the converted state
against the target plan's absolute `constraint_tolerance`: Gauss's law including periodic
neutrality, and `∇·B`. Every other pair is refused with its reason: other
stencils or collocated grids would need a Gauss projection that changes the field, and Galilean,
PML, antenna, observer, CPML, and dispersive states have no counterpart. Distributed solvers are
not converted.

Execution wrappers never inherit protocols by name. Each prepared solver's
`pic_capabilities` declares, per optional protocol, whether it is published (checked against the
structural test the runtime performs), admitted by this configuration, and on what basis;
`pic_distribution_support` states whether `distribute_pic_field_solver` admits a base. The
distributed cochain, reduced, and PSATD solvers publish exactly the protocols whose distributed
route they execute — forwarded base routes admitted exactly when the base admits them and refused
with the base reason otherwise, plus the
fused block-window multi-deposit — and state every withheld base protocol (moving windows, which
would move particles off their owner blocks; cochain relativistic self-fields, which need a
bounded grounded boundary; and cochain Huygens sampling, whose boxes Maxwell refuses beside the
PIC current). Quasi-cylindrical PSATD, unstructured Whitney PIC, and bounded cochain
grids refuse distribution, and ownership changes only at restart. The full matrices are in
[Particle-in-cell methods](guides_particle_in_cell.md#distributed-pic-and-restart) and
[Spectral PIC](guides_spectral_pic.md#advertised-capabilities-and-distribution); each published
distributed route is exercised on four forced host devices (functional evidence only).

### Coupled lane worksets

Many identical subdomains of one spatial problem are one executable evaluated many
times. `CoupledResourcePolicy(execution=CoupledExecutionPolicy(lane_capacity=8,
max_working_set_bytes=None, execution_group=None))` declares that preparation groups
such work; without it every owner executes on its own and the coupled operator stays
one named block grid.

```python
plan = cpl.CoupledProblemPlan(
    "strips",
    components=components,
    bindings=bindings,
    laws=laws,
    resources=cpl.CoupledResourcePolicy(
        execution=cpl.CoupledExecutionPolicy(
            lane_capacity=8, max_working_set_bytes=64 * 2**20
        )
    ),
)
prepared = cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
for group in prepared.worksets.components:
    print(group.members, group.estimate.working_set_bytes)
```

- **Executable signature.** Preparation traces each component's residual, field
  expansions, and own affine operator actions (forward, transpose, and their
  Riesz-identified forms) with the component's array leaves as program arguments.
  Components whose traced programs and hoisted constants are identical form one
  group: the representative's program on a member's arrays is that member's own
  evaluation. Names, classes, or equal shapes never group anything; a component
  whose source function differs is its own executable even with identical meshes,
  while coefficient values held as owner arrays (for example a different
  diffusivity) stay per-lane data of one executable. Static runtime-argument leaves
  enter with their type, so `2`, `2.0`, and `True` are different signatures. Law
  contribution blocks of the native linear system are grouped the same way (for
  example the mortar blocks of translated interfaces). Components with eliminated
  rows, affine-residual and nonlinear contributions, and singletons execute per
  owner.
- **What the signature certifies.** A printed program shows a custom
  differentiation rule or a host callback by name only. When a program holds
  custom rules (`custom_jvp`, `custom_vjp`), the signature also contains its
  first-order JVP and VJP programs, in which the rules are traced as ordinary
  equations, so members whose rules differ (for example a derivative rule with
  another slope) never share lanes. A `pure_callback`, `io_callback`, or
  `debug_callback` enters with its host function: functions with the same code,
  globals, and equal captured values (and JAX's callback wrappers around them)
  are the same function; any other callable object compares by identity, so an
  unrecognized callable never merges two members. `LaneWorkset.derivatives`
  (`LaneDerivatives`) states what is certified. A program without custom rules is
  differentiated by its primitives' standard rules (`route="primal"`, every
  order). With custom rules, first derivatives are certified, in every inexact
  array or, when a rule fixes the member's data, in the lane inputs only
  (`lane_data=False`). Higher derivatives are certified only when no custom rule
  survives in the first-order programs (`higher_order`). Finite-element owners
  keep equinox runtime checks, which are custom rules applied to values without
  a tangent, so their lanes certify first order. Lanes raise `ValueError` for
  every uncertified derivative (a second derivative, a derivative in fixed lane
  data, or any derivative of a program whose first-order programs cannot be
  traced) instead of evaluating the representative's rules for another member;
  differentiate a per-owner execution for those. The first-derivative rule wraps
  the whole bucket loop, because JAX's partial evaluation of a loop body
  differentiates nested custom rules through their primal functions at second
  order.
- **Bounded lanes.** Each group is one `PreparedExecutionWorksets` of the
  execution-workset substrate (its members are the items): `lane_capacity` lanes per
  bucket are vectorized and buckets run in sequence through `lax.map`, so one bucket
  is the live working set; padded lanes repeat a valid lane and are masked out.
  `LaneWorksetEstimate` reports the stacked lane data the workset retains, per-lane
  inputs/outputs, the traced intermediate bytes of one lane (no buffer reuse, an
  upper estimate), and the bucket working set
  `lane_capacity * (lane data + inputs + outputs + intermediates)`. A group above
  `max_working_set_bytes` is refused at preparation, before its lanes are stacked or
  any batched program is traced.
- **Assembly.** Residuals, certification, and field expansions evaluate grouped
  components as lanes; every full-coordinate law source is expanded once per
  evaluation. `linear_system` and `weak_operator` return the named block grid of the
  remaining entries plus one exact lane operator per group (a sum of operators): the
  action and its transpose are identical to per-owner execution, but block
  consumers that select named blocks (condensation, field-split preconditioning)
  see selections of the sum; declare lane execution only where the solve applies the
  whole operator.
- **Arguments and state.** Runtime arguments of grouped components must reproduce
  the prepared structure, shapes, dtypes, and static leaves; otherwise the call is
  refused and the problem must be re-prepared. States, arguments, and prepared lane
  data are only read: nothing is donated, and repeated evaluations reuse them.
- **Devices.** `execution_group` (an `ExecutionGroupSpec`, for example
  `ExecutionRuntime.current().root_group.spec`) places the lane axis of every bucket
  on the group's mesh through the existing execution-runtime binding;
  `lane_capacity` must be a multiple of its device count. No second runtime or
  collective layer is created, and method-specific sharding owners (distributed
  linear algebra, FE/FV/LBM/PIC distributions) are unchanged.

`examples/coupled_lane_worksets.py` prepares a twelve-strip mortar chain per owner and
with lanes of capacity four, prints every workset with its buckets, padding, and
working set, and checks the residuals and certified solutions of both executions
against each other, against the native conforming owner on the union band, and
against the exact harmonic field.

`tests/unit/solver/coupling/test_execution_contracts.py` compares lane execution with
per-owner execution on a five-strip mortar chain (residual, both operator actions,
right-hand side, certified solve and certificates, the harmonic reference field),
checks that equal programs with different coefficient data share lanes while a
changed source function, a changed custom derivative rule (the lane gradient then
equals the per-owner gradient, and a lane-grouped second derivative is refused), or a
changed `pure_callback` host function does not, refuses a working set above its
bound, capacities outside `[1, 64]`, and runtime arguments outside the prepared
signature, and checks that caller states and lane data remain reusable. Under
`XLA_FLAGS=--xla_force_host_platform_device_count=4` it also checks lanes placed on a
four-device group against the single-device reference.

The benchmark rows `homogeneous-worksets` (component count 4 to 64, lane capacity,
cells/degree and multiplier size; compiled residual and operator action measured as
lowering, compilation, first synchronized and warm runs with compiler argument,
output, temporary, and code bytes; certified solves; retained bytes; lane estimates;
and the difference to the specialized conforming owner on the union band, the same
discrete problem on matching meshes), `execution-group-parity` (four forced CPU host
devices; functional parity only, no hardware performance claim),
`coupled-observation-count`, and `coupled-temporal-samples` extend the campaign;
boundary-panel scaling stays in `boundary-integral-law`.

Remaining loops, host synchronizations, and dense or sorting work in
`phydrax.solver.coupling` are preparation or explicit-boundary work with a stated
bound: host point location, quadrature refinement, `unique`/`setdiff` of row sets,
mortar rank/inf-sup evidence (bounded by `max_evidence_entries`), and kernel
detection (a multi-RHS action over the published kernel span, bounded by the
materialization policy) run once at preparation; condensation factorizes its pivot
once and refreshes it for new coefficients; host-participant orchestration
synchronizes at its declared external boundary; static trace-time loops run over
owners, laws, members, or declared substeps, never over runtime-sized axes.

## Support matrix and explicit nonclaims

The tables state what each owner publishes and which interactions are exercised.
Cell values mean:

- **yes**: published and exercised by a cited example or test;
- **not exercised**: no refusal is declared, but no example or test covers the
  combination, so it is not claimed;
- **no**: refused at preparation or not published.

### Owner publications

| Method (owner) | Point query | Side trace | Flux publication | Spatial component | Coupled observations | Transient capacity |
|---|---|---|---|---|---|---|
| Lagrange finite elements (`FiniteElementDiscretization`) | yes, exact | yes: identity-mapped H1/L2 value traces | residual reaction; exact pointwise flux of diffusion forms | `VariationalComponent` | yes: point, boundary, flux | yes (`prepare_mass`) |
| Spectral elements (tensor Gauss–Lobatto Lagrange, finite-element compiler) | yes, exact | yes | residual reaction; pointwise flux as finite elements | `VariationalComponent` | not exercised | not exercised |
| Explicit polygon H1 | yes, exact | yes | residual reaction only | `VariationalComponent` | not exercised | not exercised |
| Single-patch isogeometric | yes: exact values and first derivatives | yes: patch-boundary and interior knot faces | residual reaction only | `VariationalComponent` | not exercised | not exercised |
| Virtual elements | yes, labeled `h1-projection`/`l2-projection` | yes: exact edge traces | residual reaction; projected flux (`h1-projection`); no pointwise flux | `VariationalComponent` | yes: point values labeled `h1-projection`; exact boundary traces | yes (virtual-element mass) |
| Finite difference / SBP | yes: declared multilinear or B-spline interpolation | yes: nodal restriction on bounded faces with the tangential SBP norm | no | no | no | no |
| Global spectral | yes | yes: bounded faces (`SpectralFaceRoute`) | no | no | no | no |
| Finite volume (structured, unstructured, triangular) | yes | yes: `cell-average` and linear `face-state`; WENO and limited states linearize only | no | no | no | no |
| 2-D Galerkin boundary operator | no | P1 Dirichlet / DP0 conormal trace spaces | conormal is an unknown | `GalerkinBoundaryComponent` | no | no (quasistatic only) |
| 3-D matching FEM–BEM products | no | P1/DP0 trace spaces of the boundary owner | product-owned | `ScalarLaplaceFEMBEMComponent`, `ElasticityFEMBEMComponent` | no | no |
| Galerkin reduced-order model | through the full owner's reconstruction | the full owner's traces on the reconstructed field | the full owner's reaction | `ReducedComponent` | yes: point and boundary | no |

Finite-difference, global spectral, and finite-volume owners publish queries and
traces for observation and partitioned exchange; they are not spatial components.
A finite-volume field couples to a finite-element field through the partitioned
temporal runtime (`examples/mixed_method_time_coupling.py`).

### Spatial laws

| Method | Matching elimination | Mortar | Nitsche | Conservative flux, port, transfer | Boundary-integral law (volume side) |
|---|---|---|---|---|---|
| Lagrange finite elements | yes (`test_transmission.py`) | yes (`coupled_scalar_regions.py`) | yes (`nitsche_transmission.py`) | yes (`test_interface_laws.py`) | yes (`outer_method="fe"`) |
| Spectral elements | yes (flagship matching variant) | yes (flagship) | not exercised | not exercised | yes (flagship) |
| Explicit polygon H1 | not exercised | yes (nonmatching mortar test, flagship variant) | no: no pointwise flux | not exercised | yes (`outer_method="polygon"`) |
| Single-patch isogeometric | no: tensor-layout rows | not exercised | no: no pointwise flux | not exercised | yes (full square) |
| Virtual elements | yes | yes | one-sided only (flux weight zero) | yes: conductance, integral port | not exercised |
| Galerkin reduced-order model | not exercised | yes (`coupled_rom_swap.py`) | not exercised | not exercised | not exercised |
| Finite difference, global spectral, finite volume | no | no | no: no FD SAT through the assembler | no | no |

### Coupling families

| Family | Supported | Evidence |
|---|---|---|
| Interface bindings | Two-sided, junction, overlap, and embedded incidence over mesh parts, analytic covers, and sheet views; stale revisions refused | `numerical_interface_binding.py` |
| Prepared queries and side actions | Exact coordinate duals, labeled projections, masked coverage | `prepared_field_observations.py` |
| Exterior Laplace in 2-D | Bordered Dirichlet-to-Neumann solve with declared far field | `exterior_laplace_galerkin.py` |
| Three-owner transmission | Spectral, virtual, and boundary elements in one certified solve; method substitution | `sem_vem_bem_transmission.py` |
| Derivatives and inverse problems | Implicit solution-map derivatives of affine coupled solves under `"mathematical"` or `"rhs-only"` | `coupled_inverse_problem.py` |
| Learning | `MODEL`/`DISCRETIZATION` parameter values, accelerators under fixed-work objectives, monotone learned conductance, surrogate PINN through state-design responses | `coupled_learned_interface.py`, `hybrid_pinn_classical.py` |
| Physical exchange and temporal coupling | Measurement functionals, declared temporal conversions, native method participants, replayable windows | `mixed_method_time_coupling.py` |
| Block coordinates and DAE views | Named selections, reduced-DAE coordinates, field-split preconditioning, exact condensation | `named_block_dae.py` |
| Transient coupled problems | Index-one DAE from owner capacities, adaptive continuation | `coupled_transient_fields.py` |
| Control and state-space inference | Accepted coupled transitions for existing control, MPC, and ensemble filter consumers | `coupled_control.py`, `coupled_data_assimilation.py` |
| Reduced-order components | Galerkin replacement of one region under unchanged declarations | `coupled_rom_swap.py` |
| Lifecycle, adaptation, junctions | Atomic composition rebind, adaptive FE/FV interface, N-way Plateau-border junction, checkpoint restart | `adaptive_fe_fv_rebind.py`, `foam_junction_rebind.py` |
| External execution | FMI 2.0 Co-Simulation participant under explicit host orchestration | `fmi_host_coupling.py` |
| Bounded execution | Signature-grouped lane worksets, execution-group placement | `coupled_lane_worksets.py` |
| Particle-in-cell field solvers | Solver substitution and one exact Yee/PSATD state handoff | `pic_field_solver_substitution.py`, `pic_field_handoff.py` |

### Explicit nonclaims

Geometry, bindings, and components:

- Compartment-to-mesh binding is not published, because compartment meshes carry
  no geometry association. The geometric sign of an analytic pairing normal is
  not checked independently.
- Junction, overlap, and embedded bindings are recorded, but no spatial law
  consumes them: transmission, flux, and boundary-integral laws require a
  two-sided binding, and a junction never expands into pairwise laws. The N-way
  junction exchange is owned by `PreparedPlateauBorder`.
- Spatial coupling components remain scalar and single-field; this component
  boundary does not negate standalone form reconstruction and Piola facet traces.
- The exact common refinement covers 2-D interfaces of straight facets only;
  curved facets and 3-D surface interfaces are refused.

Impositions:

- Nitsche is limited to finite-element owners that publish an exact pointwise
  flux with a polynomial facet degree; callable diffusivities, explicit polygon,
  isogeometric, and virtual-element sides with positive flux weight are refused.
  Side cells shared with other weak interface laws are not accounted in its
  certificate.
- No finite-difference SAT is published through the spatial assembler: an FE–FD
  SAT coupling is not published, and SBP–SBP interface SAT remains the
  formulation-specific `SATInterfacePlan`.
- Matching elimination refuses tensor-layout coefficient rows (an isogeometric
  side couples through a mortar).

Boundary integrals:

- The 2-D Galerkin owner is the unit-coefficient Laplace operator of one closed
  straight-panel polygon. Helmholtz, the hypersingular operator, curved, open, or
  moving geometry, FMM Galerkin actions, geometry derivatives, and continuum error
  certificates are not published.
- The spectral-element trace reaches the P1/DP0 boundary data through a declared
  L2 projection, so no exponential total p-convergence is claimed. No solver
  route for the gauged singular Johnson–Nédélec system is qualified.
- The 3-D nonmatching dense and dynamic convolution-quadrature products are not
  generic spatial components. Standalone matching Maxwell FEM–BEM does provide
  H(curl)-to-RWG trace, BC dual conormal and actual boundary operators through
  `prepare_matching_maxwell_fem_bem_3d`; caller-built periodic bounded-image
  coupling remains a separate envelope, not automatic infinite-lattice coupling.
  No three-dimensional VEM compiler is inferred from that route.

Derivatives:

- Nonlinear coupled solves publish no parameter derivative (the fixed-structure
  route is a state-design response). Geometry, quadrature, and topology
  derivatives are never published through parameter bindings.
- Condensation admits only `"mathematical"` or `"none"`; coupled transitions
  refuse derivatives with respect to step times; host participants and FMI inputs
  publish no derivative.

Temporal and transient coupling:

- General-root implicit interface policies (Newton, Broyden) report only the final
  evaluation's participant work, with `counts_complete=False`.
- DAE participants refuse waveform inputs; fixed-grid and event-driven DAE solves
  are refused as participants; participants with external randomness are refused
  by implicit cycles and adaptive windows.
- Transient coupled problems are structurally index one: a mortar multiplier on
  differential traces is refused, there is no index reduction of coupled
  multiplier systems, and no time-dependent exterior boundary-integral model.

Measurement, learning, and reduced models:

- Unquantified data has no likelihood; whitening is unavailable for dense
  precision, diagonal-plus-low-rank, and matrix-free precision covariances.
  The ensemble filter likelihood is an ensemble Gaussian approximation; an exact
  likelihood is claimed only for an affine step with Gaussian noise.
- A `SURROGATE` has no implicit solution-map signal; an accelerator never changes
  the accepted solution; a learned conductance requires an input-convex
  certificate.
- Reduced components are Galerkin (Petrov–Galerkin is refused), full-order
  assisted, and not hyperreduced; a new basis is a new prepared problem.

Lifecycle, external, and execution:

- Burst (rupture) and T1 Plateau-border transports, non-nested remeshed
  finite-element transfers, dynamic meniscus reshaping, and restart across a
  changed topology without `RuntimeRestartRelation.from_composition_rebind` are
  refused or not inferred.
- FMU-only graphs, waveform ports on FMUs, FMI input interpolation,
  general-root or Anderson implicit host windows, FMI 3, Model Exchange, and
  serialized FMU state transport are not published.
- Lanes group only identical traced programs; named block consumers of a lane
  problem see selections of the lane sum. Execution-group placement is functional
  evidence only, with no hardware performance claim.
- A PIC state handoff exists only for the periodic, lossless, boundary-free Yee
  and staggered order-2 PSATD pair; distributed solvers are not converted, and
  quasi-cylindrical PSATD, unstructured Whitney PIC, and bounded cochain grids
  refuse distribution.

Numerical verification, derivative verification, performance, provider and
hardware support, scientific validation, and release authorization are separate
evidence dimensions. Passing the scenarios below does not release a capability;
its catalog status is set by the capability lifecycle.

## Qualification

`tools/numerical_interoperability_qualification.py` qualifies the families of this
guide as a scenario matrix. A scenario is identified by its coordinates (capability,
interaction, concrete providers, geometry, execution, and derivative route), written as
one slash-separated ID such as
`scalar-transmission/mortar-multiplier/finite-element-p2+virtual-element-k2+explicit-polygon-h1/unit-square-pair-cut-x1-nonmatching/single-device/none`.
Each scenario belongs to one family and names existing pytest nodes as positive
evidence and as negative boundaries: refusals and visible failures that must hold. A
reference `path::test` covers every parametrization; `path::test[id]` names one.

```bash
uv run --extra tests python -m tools.numerical_interoperability_qualification --list
uv run --extra tests python -m tools.numerical_interoperability_qualification \
    --family derivatives --family external --workers 8 --output report.json
uv run --extra tests python -m tools.numerical_interoperability_qualification \
    --scenario 'coupled-lane-worksets/execution-group-placement/finite-element-p1/unit-strip-chain/forced-host-devices-4/none'
```

The families are `semantic-attachments`, `queries-and-sides`, `spatial-transmission`,
`exterior-operator-2d`, `sem-vem-bem`, `method-substitution`, `derivatives`,
`inventory-transfer`, `temporal-coupling`, `block-dae-views`, `learning`,
`measurement-inverse`, `control-uq-rom`, `lifecycle`, `junctions-embedded`,
`external`, `execution-wrappers`, and `scaling`. `--scenario` and `--family` are
repeatable and select their union; without either, every scenario runs.

The runner collects the referenced test files once and runs every resolved node once
per device count. Single-device nodes run in-process (`--workers N` distributes them
with pytest-xdist). Scenarios with execution `forced-host-devices-4` (distributed PIC,
lanes on an execution group) run in a fresh interpreter started with
`XLA_FLAGS=--xla_force_host_platform_device_count=4`, so their device-count skips do
not occur. `--route-timeout SECONDS` bounds each fresh-interpreter route: a route
still running at the deadline is killed together with its pytest-xdist workers and
observes no node. Each scenario becomes one `phydrax.qualification` record chain: the support
tuple of its coordinates, a zero-unqualified-node criterion, the campaign start, the
raw observation of every resolved node, the campaign observation, and
`QualificationEvidence` checked by `validate_qualification_causality`. Records use a
logical clock and content addresses. Per-node durations and per-route wall times are
reported beside them and never enter a content address. The report also carries the
git revision and worktree state, the source/build fingerprint of
`benchmarks/_runtime.py`, a build ID over the package, the referenced tests, test
support, and the runner, and the captured runtime environment.

| Scenario outcome | When |
|---|---|
| `passed` | Every resolved positive and negative node passed. |
| `failed` | A node failed, a reference does not resolve in a collection that completed, or a scaling scenario names a row that `benchmarks/numerical_interoperability.py` does not register. |
| `inconclusive` | Collection failed for the reference (a failed module, package, or directory collector covering it, or a collection that aborted, for example on a conftest import error), a node was skipped or not observed (including a route that missed `--route-timeout`), or an optional provider is unavailable. |

The FMI scenario declares FMPy and a native C compiler as optional providers. Without
either, its nodes skip and the scenario is inconclusive, never passed. Unselected
scenarios make the report outcome `inconclusive`. The exit status is nonzero when a
selected scenario failed or was inconclusive. Scaling scenarios qualify resource
admission (bounded worksets, materialization budgets, route sizes) as operational
evidence and reference the benchmark rows that record compiler, runtime, and
retained-byte measurements separately; the runner does not execute the benchmark
campaign. A benchmark case whose correctness evidence fails (duality check, solve
acceptance or success, derivative validity) raises before its timings are reported,
so the campaign exits nonzero rather than timing a wrong answer.
