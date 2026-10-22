# Geometry substrate

`phydrax.geometry` is the representation-aware geometry layer. It separates
host-side construction from JAX execution and gives analytic primitives, simplicial
meshes, CAD B-Reps, CSG expressions, and reconstructed geometry one compiled
contract.

## Source, kernel, state, and realization

A `GeometrySource` is the authoritative construction object. Calling `compile(coordinate_unit=...)`
produces a `CompiledGeometry` containing:

- an immutable `GeometryKernel` that defines algorithms and topology;
- a dynamic `DesignState` containing all trainable numeric values;
- a `ParameterSchema` with stable, feature-scoped parameter identities; and
- a `GeometryTolerance` used by tolerance-sensitive queries.

Compilation requires an explicit `coordinate_unit` imported from `phydrax.units`.
It declares the native unit of source coordinates and spatial design parameters;
it does not rescale numeric values. BRep sources must use their model's declared
coordinate-contract length unit.

```python
import jax
import jax.numpy as jnp
import phydrax as phx
from phydrax.units import METER

source = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
)
geometry = source.compile(coordinate_unit=METER)

radius = phx.geometry.ParameterId("body", "radius")
larger = geometry.with_parameters({radius: 1.5})
volume_gradient = jax.grad(
    lambda value: geometry.kernel.measure(
        geometry.state.replace_at(geometry.schema.index(radius), value)
    )
)(jnp.asarray(1.0))
```

Compiled queries are JAX-safe and batch-preserving:

```python
points = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
inside = geometry.contains(points)
signed_field = geometry.boundary_field(points)  # negative inside
normals = geometry.boundary_normal(points)
```

Use `phx.domain.GeometryDomain(geometry)` only when the geometry must participate
in labeled domains, components, integration, or constraints. Construction and CSG
belong in `phx.geometry`; the domain layer is deliberately a thin adapter.

## Affine simplex maps

`AffineSimplexMap` prepares segments, triangles, and tetrahedra with intrinsic
dimension one through three in any ambient dimension at least as large. It owns
barycentric coordinates, physical gradients, reference reconstruction,
containment, simplex measure, Jacobian measure scale, orientation where defined,
and explicit degeneracy evidence. Full-dimensional maps use the small linear
substrate; embedded maps use the induced Gram system. Degenerate maps retain
evidence and fail containment rather than silently substituting a pseudoinverse.

::: phydrax.geometry.AffineSimplexMap

---

::: phydrax.geometry.AffineSimplexEvidence

## Capabilities and field certificates

Every kernel declares a set of `GeometryCapability` values. Consumers can require
region queries, signed fields, boundary normals, measures, sampling, boundary
atlases, or seam diagnostics without checking concrete representation classes.

`FieldCertificate` records the actual numerical contract of the boundary field:
zero-set accuracy, sign reliability, distance semantics, regularity, validity
region, safe-step information, parameter differentiability, and optional
`lipschitz_upper_bound`, `evaluation_error`, and `topology_identity` (each `None`
when undeclared; declared bounds are finite and non-negative). Approximate CAD
and reconstruction fields therefore do not masquerade as exact analytic signed
distances. Analytic and mesh exact signed distances declare a Lipschitz bound of
one and zero evaluation error; rigid transforms preserve every declared bound.

`ExactSDFEnclosureCertificate` qualifies a globally reliable exact signed-distance
field certificate whose declared evaluation error and Lipschitz bound certify sign
enclosures over cells and faces. It qualifies interval classification, not exact
curved-interface measures. `QualifiedSharpGeometry` then records absolute fluid
volumes/open measures, lower/upper bounds, source fidelity, topology and epoch
identities, GCL evidence, and fail-closed status. Exact measure fidelity requires
independent clipping evidence; an exact SDF with unresolved sub-boxes remains
certified bounded error.

## Interface observables

`phase_geometry_metrics` integrates a flattened phase fraction against explicit
physical quadrature weights. It reports phase measure and centroid; a
zero-measure phase retains `centroid_defined=False` and a NaN centroid instead
of inventing a location.

`interface_distance_metrics` compares extracted predicted and reference point
sets. It reports the symmetric directed-nearest mean, symmetric Hausdorff
distance, and a configurable percentile Hausdorff distance. Masks exclude
fixed-capacity padding. Point extraction, isovalue, spacing, and physical units
remain caller-owned and must be identical across compared interfaces.
The default source and target caps are `chunk_size=None` and
`target_chunk_size=None`: both derive from `interface_workspace_bytes`
(2 MiB), counting broadcast cases and dtype item size for the pair table.
This admits a 256-by-1024 float64 pair table. Full-fit sets execute the
dense parallel computation directly, without tile loops; larger sets retain
bounded pair tiles. Positive integer chunks override their respective automatic
extent; supplying both bypasses budget derivation. The budget is validated even
when overridden and recorded as static metadata in `InterfaceDistanceMetrics`.
All controls must be static under JIT. Every active target is considered;
tiling preserves masks, broadcasts, and metric reductions. The planning budget does not bound
input/output arrays, percentile sorting, other compiler scratch, or process RSS.

::: phydrax.geometry.phase_geometry_metrics

---

::: phydrax.geometry.interface_distance_metrics

---

::: phydrax.geometry.PhaseGeometryMetrics

---

::: phydrax.geometry.InterfaceDistanceMetrics

## Analytic geometry and CSG

Analytic sources provide closed-form fields, measures, samplers, and boundary
atlases. Sources compose before compilation:

`Ball`, `Orthotope`, and `AxisAlignedEllipsoid` are dimension-neutral radial
and axis-aligned region owners. `Circle`/`Sphere`, `Rectangle`/`Box`, and
`Ellipse`/`Ellipsoid` retain explicit low-dimensional frontends. Interior and
boundary masses retain `ExactMass`, `EstimatedMass`, or `UnknownMass`; fixed
numerical perimeter or surface-area quadrature is never labeled exact.

::: phydrax.geometry.Ball

---

::: phydrax.geometry.Orthotope

---

::: phydrax.geometry.AxisAlignedEllipsoid

```python
left = phx.geometry.Sphere((-0.4, 0.0, 0.0), 1.0, feature_id="left")
right = phx.geometry.Sphere((0.4, 0.0, 0.0), 1.0, feature_id="right")

union = (left | right).translated((0.0, 0.0, 1.0))
intersection = left & right
difference = left - right
scaled = left.scaled((1.0, 2.0, 1.0))
```

Sharp CSG preserves exact set membership but generally yields a nonsmooth level-set
field at operation seams. Blend CSG provides a smooth approximate zero set and
reports that weaker contract through its field certificate.

Blend width is a geometry approximation parameter, not an optimizer guarantee.
A fixed positive width solves a different geometric problem. If blend CSG is used
for continuation, schedule it outside the geometry and solver abstractions, finish
against the sharp geometry, and report terminal sharp-field metrics. Phydrax does
not couple a continuation policy to either abstraction.

`python -m tools.geometric_benchmarks --csg-continuation --smoke` compares
sharp, fixed-width blend, and width-annealed training while evaluating all terminal
scientific metrics on the sharp geometry.

### Superquadrics

::: phydrax.geometry.Superquadric

`Superquadric` exposes analytic volume, principal inertia moments, support points, normals, and contact curvature for smooth convex three-dimensional shapes. The DEM-specific prepared set and pair oracle are documented in the particle API.

## Simplicial geometry

`TriangleMesh` and `SegmentMesh` own canonical validated arrays and topology.
`TriangleTopology` provides half-edge twins, boundary loops, manifold checks, and
connected components. `TriangleBVH` and `TriangleMeshQueryIndex` answer exact
closest-point and k-nearest-face queries by branch-and-bound traversal with a
per-query stack of `max_depth + 2` entries; `TriangleBVH(mesh, policy=...)` takes a
`BVHBuildPolicy` (median, Morton radix tree, or binned SAH) and `refit(vertices)`
moves the fixed hierarchy to differentiable vertex positions with one update per
tree level. `winding_number(points)` is exact: leaves whose box contains the query
sum triangle solid angles and every other subtree is replaced by the closing fan of
its boundary (Jacobson et al. 2013). `fast_winding_number(points, opening_angle=beta)`
is the approximate first-order dipole route of Barill et al. (2018); both return a
`WindingNumberResult` whose `route` and `approximate` fields name the evaluation.
`MeshRegion` and `PlanarMeshRegion` lower watertight 3D meshes and planar
triangulations to the common geometry kernel; `MeshRegion` refits its prepared
`TriangleBVH` to the current design vertices for distance, closest-point, and
inside queries instead of forming query-by-face arrays.

`SegmentBVH(vertices, edges, policy=...)` prepares the corresponding exact
planar hierarchy from finite `(V, 2)` vertices and nonempty `(E, 2)` edge indices,
canonicalized to float64 and int32. Zero-length edges are refused. The default
build policy is `BVHBuildPolicy(leaf_size=8)`. `nearest_segments(points, k=1)`
accepts `(..., 2)` points, supports `k=1` and `k=2`, and returns
`BVHNearestResult.items` and `distance_squared` with shape `(..., k)`. Equal
distances are ordered by original segment index; unavailable ranks are padded
with index `-1` and squared distance `inf`. `query(points)` returns
`SegmentQueryResult.closest_point`, Euclidean `distance`, original
`segment_index`, and unit right-hand `normal`, preserving leading point axes.
The normal for segment direction `(dx, dy)` is `(dy, -dx)` after normalization;
no enclosing-region orientation is inferred. `refit(vertices)` preserves
topology, segment order, and policy and validates shape only; callers own finite
vertices and nonzero edge lengths in the new embedding. Selected projections
and squared distances remain locally differentiable away from branch changes.
`SegmentMesh`, compiled `Polygon`, and compiled `PlanarMeshRegion` reuse this
prepared hierarchy instead of retaining point-by-edge projections. Ray parity
scans fixed-width edge and point tiles.
The closed `execution` selector is `"auto"` (default), `"bvh"`, or `"dense"`.
Automatic execution evaluates small batches densely only when the conservative
`64 * query_count * segment_count` logical-byte admission fits
`dense_workspace_bytes` (default 512 KiB); larger batches traverse the hierarchy.
Explicit dense execution refuses workloads above that budget instead of
silently switching routes. Explicit BVH execution always bounds candidates to
the leaf capacity and stacks to tree depth. This logical admission excludes
prepared arrays and required outputs and is not a process-RSS bound. Both routes
use the same original segment-index tie order; dense selection uses two linear
minima, not a full sort.

Polygon validity scans nonadjacent pairs in lexicographic tiles; the compiled
polygon no longer retains an `intersection_pairs` leaf. Prepared segment
hierarchies add numerical PyTree leaves to `SegmentMesh` and compiled planar and
polygon kernels. These are intentional persistence representation changes;
feature and physical source identities remain unchanged. Leaf serialization
requires a template with the current hierarchy structure and static policy;
old leaf layouts are not migrated. The public `segment_query_evidence(points,
closest_points, selected_coordinate, selected_length, second_distance_squared)`
takes selected `(..., 2)` points/projections and scalar leading-axis arrays.
The coordinate is the clipped segment fraction; the second squared distance is
to the nearest other segment, or `inf` if absent. It returns `(unique, regular,
margin)` without inferring enclosing-loop topology. Endpoints and unresolved
ties have false flags and zero margin, not a regular projection certificate.

::: phydrax.geometry.SegmentBVH

::: phydrax.geometry.SegmentQueryExecution

::: phydrax.geometry.segment_query_evidence


`discrete_operators(...)` constructs matrix-free DDG incidence, mass, Laplacian,
and gradient operators from the same topology. Mesh adapters accept native
triangle arrays, `TriangleMesh`, Meshio data, or Meshio-supported paths through
the canonical import functions.

## Boundary atlases and measure partitions

`BoundaryAtlas` is the common boundary-integration structure. A chart maps a
reference coordinate to a physical boundary point and supplies its physical
Jacobian, outward frame, trim domain, source entity identity, physical tags, and
seam ownership. Atlas metadata survives rigid transforms, scaling, selection, and
fixed-topology CAD reevaluation.

```python
atlas = geometry.boundary_atlas
selected = atlas.select(entity_ids=(0,))
partition = phx.geometry.BoundaryAtlasPartition(selected)
```

`BoundaryAtlasPartition` estimates one physical measure per chart and supports
fixed-size stratified sampling. `GeometryMeasurePartition` is the explicit simplex
partition for boundary segments or planar/surface triangles. Sampling APIs return
`SamplingResult`; bounded rejection exposes completion, acceptance, and proposal
counts rather than silently returning too few points.

## Native cubature atlases

`CubatureAtlas` maps certified canonical cubature rules directly to a physical
interior or boundary measure. Unlike `BoundaryAtlas`, it owns no sampling,
frames, or general trim semantics: it supplies only a closed reference identity,
physical point map, Jacobian, active mask, source entities, and tags.

Analytic circles and spheres expose disk/circle and ball/sphere atlases.
Watertight `MeshRegion` boundaries expose direct unit-triangle charts. Rigid
transforms, translations, and uniform scaling preserve this capability.
Nonuniform scaling and CSG do not advertise it until their physical Jacobian
contracts can be represented without approximation.

## CAD B-Reps

Native CAD construction and file decoding produce `BRepModel`, whose exact
carriers, oriented topology, p-curves, occurrences and coordinate contract own
geometry identity. Derived query tessellations do not replace that authority.
`phx.interchange.read_cad(path, policy, trusted_root=...)` selects the native
STEP, IGES or external OCCT BRep text codec. The format-specific `read_step`,
`read_iges` and `read_brep_text` preserve their own coverage and refusal reports.
STEP and IGES own embedded units; BRep text requires `source_length_unit`.
The native decoders implement these file formats without invoking OCP.

Native carriers remain authoritative through construction, bounded intersections,
sewing, Booleans, and tessellation. `BRepTessellationPolicy(realize=False)`
publishes source geometry without a query tessellation. Overlapping-sphere
partitions retain exact rational chart coordinates for collapsed-pole radial
splits; binary64 vertices are execution representatives bound to that authority.
Curved-void trim ribbons are enclosed per source-curve interval and subdivided
before publication when their bound exceeds the declared deviation. Native
BRep text refuses implicit curved intersection branches rather than fitting
them. Supported entity families and typed refusals are reported by each codec;
native decoding is not a claim of complete STEP or IGES coverage.

Fixed-topology B-Rep nearest queries prepare one triangle hierarchy, refit it
to current realized vertices, and select one or two faces without a full
face-distance sort. The second selected distance supplies closest-point
uniqueness evidence, and projection derivatives remain local to those faces.
This adds prepared hierarchy leaves but preserves model/revision source IDs.
The dense solid-angle containment route remains separate from native
carrier-based certified membership; its numerical winding value is not an
outward whole-source certificate.

Rational spline evaluation uses the shared span-local B-spline kernel. Each
curve query gathers `degree + 1` controls; each surface query gathers only the
tensor product of the active controls in its two parameter axes. Expanded
nonuniform and repeated knot vectors are preserved. At an exact chart
endpoint the final polynomial span supplies the one-sided differential, so
surface Jacobians and boundary frames remain finite instead of collapsing to a
constant endpoint branch.

::: phydrax.geometry.brep.ParabolaCurve

---

::: phydrax.geometry.brep.HyperbolaCurve

---

::: phydrax.geometry.brep.OffsetCurve

`OffsetSurface(base, distance)` retains a signed normal-offset operation tree.
It is available from `phydrax.geometry` and `phydrax.geometry.brep`; the base
definition and offset remain the source authority. An analytic equivalent may
be used only when the original source expression is proved equivalent, not
because a rounded radius or sampled surface looks close.

::: phydrax.geometry.OffsetSurface

`AffinePCurve` applies an exact UV matrix and offset to the original p-curve,
without refitting its conic or rational coefficients. `PeriodicPCurve` retains
the original curve and an explicit native surface patch plus two integer period
shifts. Its floating-point UV coordinates are representatives; interval bounds
enclose the patch's mathematical periods. Distinct period shifts identify
distinct chart sheets even when their physical images coincide. Neither carrier
changes source edge parameters or root-valued endpoint identity.

`phydrax.geometry.brep.NativePeriodEndpoint` authors a source parameter as
`rational + turns × 2π`, bound to the original period-owning curve or an explicit
surface-period axis. Turn zero and turn one identify the exact endpoints of an
authored closed circle. Its binary64 parameter and outward enclosure are only
numerical representatives; this does not reinterpret an independently authored
floating-point endpoint or infer closure from nearby coordinates.

::: phydrax.geometry.brep.NativePeriodEndpoint

These source chart operations do not by themselves declare a quotient mesh or
prove a physical field-transfer correspondence across poles and seams. Those
contracts require their own explicit topology and atlas witnesses.

::: phydrax.geometry.AffinePCurve

::: phydrax.geometry.PeriodicPCurve

```python
coordinate_contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
model = phx.geometry.brep_box(
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    coordinate_contract=coordinate_contract,
)
source = phx.geometry.BRepSource(model)
geometry = source.compile(coordinate_unit=model.coordinate_contract.length_unit)
print(source.model.report)
```

`FixedTopologyBRepSource` reevaluates an imported tessellation from trainable
patch parameters while preserving face topology, boundary entity identity, knot
vectors, degrees, and knot-span topology. Rational B-spline control points and
weights remain differentiable; moving knots across a query is intentionally not
part of this fixed-topology contract. A realization exposes the current
vertices, faces, atlas, and a differentiable seam residual. The validity region
requires unchanged topology, positive surface Jacobians, and compatible seams;
`BRepSeamCompatibility` makes the last condition an explicit design constraint.

`prepare_brep_projection(model, policy=..., query_policy=...)` binds native
closest-point queries to the exact revision of a native `BRepModel`. It returns
`NativeBRepProjection`; callers provide no live external shape or fallback
tessellation. Projection results retain their entity parameters, residuals,
frames and `BRepProjectionStatus`. `prepare_brep_query` owns native containment,
surface queries and measure evidence under its explicit query budget.

Trimmed area and volume rules partition inner Green integration legs at the
owning source's native-u knot and chart walls, through placement, offset, sweep,
ruled and spline source trees. Splitting only the outer trim curve is
insufficient for a C0 profile. The original quadrature order and subdivisions
remain unchanged; source-wall preparation and generated points consume the
same query allowances. Reported fine/coarse agreement is a numerical error
estimate, not an outward integral certificate.

Native spline-profile extrusion interpolation covers the complete footprint
against every original knot stratum. A footprint touching or crossing a C0 knot
uses closed one-sided span jets and a complete first-jet hull where a global
second derivative is unavailable; it does not replace the source profile or
declare that derivative finite. Surface generation retains the original source,
remaining resource scope and default fidelity gates.

`phydrax.geometry.brep.BRepQueryBudget` shares caller-authored operation, point
and scratch allowances across prepared `contains` and `closest` calls. Complete
candidate covers are admitted before work; only executed operations, points and
subdivisions are consumed. Negative counts and noninteger inputs refuse without
crediting or changing the ledger. `BRepQueryResourceError` retains the refused
resource, requested quantity and remaining allowance.

Containment reports include distance lower/upper bounds, `query_operations` and
`resource_exhausted`; surface queries also retain `query_operations`. An
unfinished interval proof remains failed or unresolved, never successful merely
because its proposed point is finite.

For complete native-gauge sphere charts, containment prepares the exact affine
coefficients of retained placement operations and the provable rational radius
of the original normal-offset tree. Approximately orthogonal placement matrices
are not treated as exactly orthogonal. Native seam and pole walls establish
whole-source coverage even when root enclosures pad the numerical parameter box.
The authored source tree, pose, and root identities remain intact through
archive round-trips; boundary ambiguity remains explicit. Other source or trim
compositions retain their generic certified-query path.

::: phydrax.geometry.brep.BRepQueryBudget

::: phydrax.geometry.brep.BRepQueryResourceError

Intentional OCCT comparisons use the separate optional
`phx.interchange.model_from_occt_shape`, `import_occt_brep`, `persist_occt_shape`
and `read_occt_shape` boundaries, installed with `phydrax[cad-occt-interop]`.
These APIs call an external geometry engine; a `.brep` extension denotes the
external OCCT text format, not a native Phydrax persistence archive.
`save_brep_archive`/`load_brep_archive` provide the canonical native lifecycle
persistence instead.

`AbstractBRepProjection` is the engine-neutral query contract that B-Rep
association and high-order curving accept. It owns the closure relation,
tolerance classification and coordinate frames; a concrete projection supplies
`project` and `locate_solids`. Native queries and optional external comparisons
are distinct implementations of this contract, not aliases.

### Public coordinate request resources

Certified host geometry queries take a
`phydrax.discretization.CoordinateEnclosureBudget` borrowing an admitted
`phydrax.BatchTicket`. For standalone host preparation, construct
`phydrax.HostBatchTicket(work_upper, bytes_upper=...)`, enter its context,
construct `CoordinateEnclosureBudget(ticket, maximum_work_units=...,
maximum_memory_bytes=...)`, and enter `budget.activate()` before evaluating.
The ticket bounds parent work and simultaneous managed storage; coordinate
limits bound exact coefficient work and retained/temporary enclosure storage.
Reservations must fit both layers, and child batches cannot overcommit their
parent. These are explicit work/storage limits, not requested accuracy, retry
settings, or geometric tolerances.

`CoordinateEnclosureResourceError` and `RationalEnclosureError` are public at
`phydrax.discretization`. A resource refusal retains `.resource`, `.limit`,
`.requested`, `.completed`, and available `.phase_evidence`; a rational
enclosure refusal reports an unresolved denominator or chart limit. Neither
means an approximate result is certified. Scope ticket/budget lifetimes to the
request; do not store them in a numerical model or process-global cache.
See the [executable request helper](../guides_differentiable_geometry.md#host-certification-and-coordinate-tickets).


### Certified mass properties and signed shell volumes

`brep_mass_properties(model, geometry, subdivisions=...,
coordinate_budget=...)` returns `BRepMassProperties` bound to `model_id` and
the model's exact coordinate unit. It publishes geometric volume, first and
second moments, centroid, and centroidal inertia, independently of a material
density. Face area/centroid rows follow canonical face order; solid rows follow
canonical solid order. The lower/upper axis comes immediately after the entity
axis: `volume_enclosures` is `(S, 2)`, `centroid_enclosures` is `(S, 2, 3)`,
and `inertia_enclosures` is `(S, 2, 3, 3)`. The corresponding `*_error_bound`
properties bound error about the reported midpoint outward.

Complete exactly orthonormal sphere and torus charts use enclosed closed forms.
Other native analytic and NURBS trims use signed Green-map interval cell
integration, including certified endpoint parameter intervals. `subdivisions`
declares work, not a promised accuracy. A volume interval containing zero
refuses centroid/inertia publication. JAX x64 must be enabled before evaluating
these binary64 certificates.

Mass-weighted fields (`mass`, `mass_enclosures`, `mass_inertia`, and
`mass_inertia_enclosures`) require all of `density`, `density_unit`, and
`coordinate_unit`; the density dimension must be mass per length cubed and the
coordinate unit must equal the source contract. No conversion is implicit.
The older prepared-query fine/coarse measure estimate is not this certificate.

`brep_signed_shell_volume_enclosures` uses the same integration with only the
signed volume channel. Its `(H, 2)` rows are keyed by canonical source `shell_ids`;
correctly oriented cavities are negative. Negative or zero-containing intervals
are retained, not rejected by the solid mass-property positivity gate.

### Certified edge classification

`brep_edge_classification(model, geometry, maximum_cells=...,
coordinate_budget=...)` returns one `BRepEdgeClassification` per canonical edge.
Each row retains `edge_id`, `model_id`, `sign_enclosure`, `dihedral_range`,
`length_enclosure`, and enclosed `change_parameters`. `dihedral_range` is the
angle between material-side normals in radians, not the interior solid angle.
The closed status is `convex`, `concave`, `sign_change`, or `unresolved`.
Classification uses whole-edge derivative intervals and coedge sense, not a
point sample. A sign change requires regular continuous carriers; singular,
nonmanifold, or conflicting material-side incidence remains unresolved.
Lengths use line/circle closed forms where admitted and interval speed
integration otherwise. The caller supplies a cell limit and coordinate ticket;
binary32 certificate publication refuses. Consumers choose their own feature
angle rather than receiving an implicit "smooth" classification.

### Analytic fillets, chamfers, and planar offsets

`brep_fillet(model, edges, radius, ...)` and
`brep_chamfer(model, edges, (first_setback, second_setback), ...)` take exact
source edge identities. `BRepBlendResult` retains those identities, a reason,
`certified`/`unsupported`/`unresolved` status, the represented `.model`, and
per-boundary `g1_angle_bounds`. Inspect status before using the optional model.
Supported local analytic constructions include plane pairs, admitted circular
plane/cylinder/cone/sphere contacts, and parallel-cylinder intersections.
Several selected edges sharing a vertex refuse corner-patch blending;
variable-radius, general NURBS rolling-ball, and G2/G3 blends are not supported.

An available fillet `.source` is a correlated radius parameter map consumed by
`FixedTopologyBRepSource`: surfaces, curves, p-curves, vertices, bounds, and
seam evidence move together. `sensitivity_status` is independently
`available`, `unsupported`, or `not_applicable`. Oblique intersection-cap
fillets can certify the primal while reporting unsupported radius sensitivity
and `source=None`; chamfers have no radius source. A certified primal does not
imply a supported derivative.

`brep_offset` returns **`BRepOffsetResult`**, not `BRepModel`; consume `.model`
explicitly. Planar solids, including affine degree-one extrusion charts and
admitted concave prisms, use rational incident-plane solves, local edge-sign
checks, and continuous vertex-face and edge-edge contact proofs over the whole
requested displacement. These proofs include adjacent faces and holes and
account for trim vertices moving farther than the normal plane offset.
Vanishing/reversed edges, collisions, irrational unit normals, and sharp curved
adjacencies refuse instead of approximating a displacement. Rational work and
temporary storage are admitted before each proof stage.

Planar `.evidence` retains canonical `face_rounding_bounds`,
`vertex_rounding_bounds`, and a `volume_enclosure` enclosing rational intent and
rounded publication. Intrinsic endpoint/face gaps belong to the generic model.
Smooth normal-operation trees may still construct exactly, but publish these
planar evidence fields as `None` with unresolved `volume_status`; absence is
not a zero-error proof. Use `brep_mass_properties` for their independent volume
certificate.

### Sewing with declared gaps

`BRepSewingPolicy.maximum_gap` defaults to zero. Positive-gap sewing admits
unique full-range, equal-length parallel line lamina pairs whose exact
coefficient displacement is outward-bounded within that fixed declaration.
`source_uncertainty` may supply a recorded STEP bound only up to the policy
maximum. Curved gaps, partial overlaps, root-authored gap endpoints, and
ambiguous pairings refuse. No carrier is moved and no tolerance grows.

`BRepSewingLineage.pair_gap_bounds` records retained/other source edges and their
pair-carrier bound. `edge_face_gap_coedges` identifies affected face uses;
the bounds themselves live once in `BRepGeometry.edge_face_gap_bounds`, in
canonical coedge order. `edge_vertex_gap_bounds` separately records endpoint
displacement. Publication and replay recompute admitted bounds and reject
underclaims. Both optional fields are absent for exact gap-free models.
Native archives retain present evidence, and downstream source-fidelity bounds
include it; a positive gap cannot claim zero source equality.

### Sections, contact junctions, and clearance

`brep_section(first, second, maximum_boxes=..., coordinate_budget=...)` accepts
models or `SurfaceRegion`s, with a `PlanePatch` also allowed as the second
operand. `BRepSectionResult.intersections` retains native certified curve atlases
and unresolved parameter regions, paired with canonical `.face_pairs`. Infinite
planes are clipped conservatively to source bounds. Authored rectangular model
face domains are admitted; nonrectangular model trims are explicitly unsupported.
The result does not mutate either operand.

`ParametricIntersectionPoint.kind` distinguishes `crossing`, `tangent_point`,
`tangent_curve`, and `singular_point`. `.branch_count` is a certified outgoing
half-branch count, or `-1` if unresolved. Interval common-graph second forms
certify zero branches for a definite difference and four for an indefinite one;
zero-containing determinants and degenerate contacts remain unresolved.
The count must agree with the native branch atlas, including periodic seams.

`brep_clearance(first, second, maximum_boxes=256, coordinate_budget=...)`
encloses the minimum **boundary** distance, not solid penetration depth.
`distance_lower`/`distance_upper` are outward bounds;
`witness_lower`/`witness_upper` enclose the two chart witness points, one row per
operand. Full exact-orthonormal spheres and parallel-axis cylinder walls use
closed forms, including nested/concentric and touching cases. General admitted
native charts use a face BVH and best-first subdivision of both parameter
domains; certified trim membership, not chord masks, admits point candidates.
The upper bound comes from witnesses and the lower bound from all surviving
pair cells. Budget-limited results retain `unresolved_pairs` (zero-based face
indices) and `boxes_processed`, with conservative bounds. Unsupported geometry
has no witnesses and distance `[0, infinity]`; inspect status and reason rather
than treating an unavailable bound as a clash decision.

### Certified polynomial deformation

`PolynomialDeformation` represents a Bernstein lattice with fixed identity
boundary controls, or a named `.taper(...)`/`.shear(...)`. `identity_lattice`
constructs identity controls. `certify_deformation(..., maximum_cells=...,
coordinate_budget=...)` encloses the physical determinant with rational
Bernstein subdivision; cells retain dyadic boxes and
`certified`/`negative`/`unresolved` evidence. Sampled positive Jacobians are
insufficient. Identity on the box boundary together with strictly positive
determinant certifies lattice injectivity; named inverses require their own
positive-scale certificate.

`deform_brep` returns a `BRepDeformationResult.parameter_map` for
`FixedTopologyBRepSource(model, parameter_map=...)`. It transports definition-space
patches, curves, and vertices while retaining p-curves, UV bounds, incidence,
and declared occurrence placements. Interior lattice positions are dynamic
length-unit leaves. Full-span rational Bernstein and plane carriers compose
exactly with coefficient enclosures; aligned cylinder taper retains an exact
cone carrier. Patches crossing the proof box, restricted/multi-span splines,
implicit intersection edges, unaligned cylinders, twist, and bend refuse.
Outside lattice patches remain identity. This is shared-definition motion,
not per-occurrence world-space deformation.

### Certified defeaturing

`brep_defeature(model, face, operation=..., budget=...)` selects one source face
by identity for `fillet`, `chamfer`, or `hole` removal. Its result distinguishes
`certified`, `unsupported`, `unresolved`, and `collision`. Admitted families
include local plane-pair cylindrical fillets/planar chamfers with planar ends,
normal plane/cylinder quarter-torus blends, and single cylindrical/conical
through-hole walls between planar hosts. Retained faces and pockets keep their
source topology; the operation does not rebuild an unrelated bounding prism.
Sphere corner blends, cylinder/cylinder quartics, non-normal circular blends,
nonanalytic neighbors, and unsupported intersection/clipping families refuse.
An overlapping unsupported curved face remains unresolved, not collision-free.

The record publishes symbolic volume terms where supported or enclosed
differences of certified mass properties otherwise. `volume_gap_error_bound`
accounts for closing-gap flux; `hausdorff_enclosure` includes intersection
rounding, intrinsic gaps, and phase enclosures. `intersection_kind` interprets
`carrier_intent_points` as none, line endpoints, or a circle center, with
`intersection_rounding_bounds`. Canonical source `g1_edges` key
`g1_angle_bounds`; `face_images`/`edge_images` retain surviving lineage.
Unaffected face recipe keys survive. Edge naming refuses when a new sharp edge
prevents exhaustive source ancestry.

### Feasible parameter ranges

`CertifiedParameterRange` retains the parameter, an **open** `.interval`,
per-end `.end_status`, binding `.constraints` and `.entity_ids`, and `.evidence`.
An `attained_exactly` end is an exact limiting value, not an admissible boundary.
`certified_inner_bound` is conservative; `unknown` has endpoint `None`.
Signed infinity means proved unbounded, never "no limit found." Finite feasible
endpoints round inward.

Fillets, chamfers, offsets, tubes, and deformations accept
`include_parameter_range=True` to retain success evidence. Construction parameter
refusals retain it without opt-in, either on their result or through
`refusal_parameter_range(error)` for existing `ValueError` boundaries.
Chamfer ranges describe dimensionless `setback_scale` at the fixed authored
setback ratio; tube ranges are `(0, reach)`. Planar offsets distinguish
edge-vanishing roots from the first verified contact in either direction.
Representable endpoints retain the attained rational root; other finite
endpoints round inward. Unsupported whole-source proofs report unknown rather
than an estimate. Resource errors propagate.
`deformation_feasibility_scale` certifies space and the whole scale interval,
not only endpoint samples; determinant singular suprema are not admitted.

### Occurrence placement evidence

`BRepOccurrence.placement_evidence` is optional `BRepPlacementEvidence`.
It encloses source-frame/polar-factor rotation deviation, translation deviation,
native model extent, point deviation, and optional `(3, 4)` entrywise affine
coefficient errors. Positive evidence enters geometry identity and native
archives; exact occurrences omit it. Placement queries remain available.
Materialization, sewing, and Booleans refuse nonzero displacement with
`placement-deviation`, and external CAD writers refuse unsupported projection.
The [interchange reference](interchange.md#step-assemblies-and-placement-evidence)
describes explicit source-frame admission; these bounds are not incidence gaps.


## Physical CAD identity and selection

CAD construction, import, and persistence are bound to an explicit
`SpatialCoordinateContract`. `CADRevision` inventories the exact occurrences in
one revision; `CADSelectionSet` contains exact occurrence selectors from that
inventory; and `AssociationGraph` records explicit correspondence between two
revisions. Selection and association never use proximity or array position as a
fallback.

`cad_revision_from_brep_model`, `all_brep_solids`, and `all_brep_faces` expose a
model's revision inventory and complete topology selections. A B-Rep partition
or process result carries its resulting `CADRevision` and `AssociationGraph`, so
downstream scopes remain bound to the created physical entities.

### Geometric predicates and normal ranges

`BRepPredicate` expresses analytic family, facing/parallel/perpendicular,
edge convexity, extremal position, adjacency, and combinations. `a & b`,
`a | b`, `~a`, and `a.neighbors()` preserve three-valued logic.
`select_brep_entities(model, predicate, kind="face", maximum_cells=...,
coordinate_budget=...)` returns disjoint selected/rejected/unresolved identities
exhausting the canonical face or edge family. Unsupported normal-on-edge and
convexity-on-face predicates remain unresolved. Extremal selection requires an
entity wholly on the supporting extreme; touching at a corner is insufficient.

`brep_normal_ranges(model, direction, draft_angle=..., maximum_cells=...,
coordinate_budget=..., angle_thresholds=...)` encloses whole-face angles in
radians against outward material-side normals. Rows retain canonical `face_id`,
`angle_enclosure`, `unresolved_cells`, and enclosed `singular_points`. The status
is `positive`, `insufficient`, `vertical`, `undercut`, or `unresolved`.
Exact threshold contact is unresolved; singular normals are reported, not
nudged. Native derivative families and certified trims qualify each decision;
budget exhaustion never becomes a sampled classification. The same direction
test supports draft or overhang analysis, not a draft modeling operation.

`selection.bind_region(name, names, domain, adjacent_region_names,
include_unresolved=...)` returns the existing meshing `PatchControl`. Unresolved
membership requires an explicit inclusion/exclusion choice; empty membership
refuses. Its `recipe_binding` retains stable selected keys and the exhaustive
family inventory. `control.rebind(names, domain)` resolves those keys after a
topology-preserving edit and refuses splits/merges, missing names, and revision
mismatches. Definitions expand only through declared domain occurrence rows,
never spatial proximity or occurrence-path prefixes.


## B-Rep and planar partitions

`partition_brep` accepts a `BRepPartitionPlan` with an explicit coordinate
contract, ordered `BRepPartitionOperand` values, roles, void targets, and
`BRepPartitionPolicy` precedence. Its persisted `BRepPartitionResult` exposes
named solid regions and face patches as exact `BRepEntityId` selections,
revision-to-revision association evidence, and deleted/created occurrence
history. It does not offer general Boolean repair.
Region ownership is decided before voiding: the highest-precedence present
region wins. A present void targeting that owner leaves a hole; lower-priority
regions never refill removed material. Void targets are explicit region
identities, not inferred from material names or geometric overlap.

`partition_planar` applies the same selection and history discipline to
`PlanarMeshRegion` operands embedded by `PlanarEmbedding`. Its result has
topological dimension two: regions select B-Rep faces and patches select B-Rep
edges. The embedding, coordinate contract, and precedence are caller-owned.

## Explicit layout process stacks

`geometry.process` lowers decoded planar regions only after the caller supplies
every physical stack fact. `StackRegion` requires a planar footprint,
`ZInterval`, and precedence; `StackVoid` additionally names its target regions.
`ProcessStack` carries the coordinate contract, and `lower_process_stack`
persists the resulting B-Rep partition with exact region, patch, revision, and
association identities. It does not infer a fabrication process, layers, or
void targets from layout content.

Design parameters also declare exact native units:
`ParameterSpec(..., role=..., unit=...)` and
`ParameterCollector(coordinate_unit).bind(..., role=..., unit=...)` refuse
incompatible dimensions or repeated declarations with differing units.
Schema identity includes unit IDs. Compilation requires an explicit length
coordinate unit; angular and dimensionless kernel roles use `RADIAN` and `ONE`.
`ZInterval(lower, upper, unit)` must match the process coordinate unit exactly;
conversion is explicitly owned by `phydrax.units.convert_value`.

`geometry.design.RecipeEntityName(kind, region, source_entity, lineage_path)`
is a stable value key, not a revision-scoped entity ID. `RecipeEntityNames`
binds canonical faces/edges to these keys with `.resolve(name)` and
`.name(entity_id)`. `name_partition_entities` requires exhaustive authored
source occurrence keys and bijective lineage; sewing transport appends its
operation path. Splits/merges and ambiguous or missing ancestry refuse.
Planar partitions support both families; current 3D partitions expose
`edge_names=None`, explicitly unsupported rather than an empty supported set.
See [stable recipe names](../guides_planar_process_layout.md#bind-stable-recipe-names).


## Sketches and geometric constraints

`Sketch` solves fixed-connectivity 2D line/arc/circle systems with declarative
constraints such as `Coincident`, `Horizontal`, `EqualLength`, `Radius`, and
tangency. `Sketch.solve` and `DesignConstraintSystem.solve` use native
`phydrax.optim.least_squares` with Levenberg–Marquardt and `phydrax.linalg`
Jacobian solves. Only admissible, residual-decreasing trials commit; nonpositive
radii refuse rather than being clamped.

Both results expose the native `OptimizationStatus` as `status` and preserve
diagnostics and provenance in `solver_result`. `converged` additionally requires
the weighted residual norm to satisfy the requested tolerance. A stationary
least-squares point with a nonzero, out-of-bound residual is not a solved sketch.
`SketchSolution.points` and `circle_radii` carry implicit sensitivities to the
constraint targets and numerical constraint data. The native nonlinear output
attachment differentiates the converged weighted least-squares stationarity
equations, \(J_r^\mathsf{T}r=0\), without resolving the primal or differentiating
the LM iterations. Initial seeds remain conditioned inputs, not algorithmic
solution sensitivities.

`derivative_evidence` is native `phydrax.nonlinear.SensitivityEvidence`;
`derivative_rank` is the native numerical rank of the endpoint stationarity
Jacobian. Rank and conditioning preparation uses a bounded native SVD:
at most 64 coordinates, 4,096 materialized entries, and one MiB workspace.
Tangent and adjoint actions use native matrix-free linear solves. Nonconverged,
rank-deficient, ill-conditioned, or uncertified endpoints retain their primal
solution and optimizer evidence but return NaN derivative actions, with the
reason recorded in `derivative_evidence.status`. A sketch larger than the
declared derivative preparation budget reports `UNSUPPORTED` sensitivity;
its matrix-free primal solve remains available. For example, an independently
targeted radius has unit target sensitivity and zero initial-radius sensitivity,
while an underdetermined sketch does not claim a unique implicit derivative.

`DesignConstraintSystem.solve` continues to provide values and optimizer
evidence; this sketch endpoint attachment does not extend its sensitivity scope.

`to_source(solution)`, `to_profile(solution)`, and `to_model(solution, ...)`
require a converged `SketchSolution` bound to the sketch's stored geometry and
constraints. Missing, unconverged, or differently bound solutions refuse
lowering. A line loop or circle lowers through
`to_source`; mixed line/arc/circle loops use `to_profile` or `to_model`. The
profile's `residual_bound` retains the weighted solve-residual bound. It is not
a spatial error enclosure or a tolerance for changing geometry; profile topology
is certified independently from the solution snapshot.

### Root certification and priority tiers

`geometry.design.certify_sketch_solution(solution, budget=...)` is a separate
host certification step. It publishes component rank/nullity, native rank-gap
and backward-error evidence, authored under/over/redundant constraint IDs,
and, for admitted regular square systems, a strict Krawczyk unique-root box.
Its status is `certified`, `rank_unresolved`, `not_fully_constrained`,
`unsupported`, or `enclosure_unresolved`; residual convergence alone is not
root certification. Redundant rectangular systems retain DOF evidence but
refuse the square-root enclosure. Custom constraints lack admitted interval
work bounds and report unsupported rank/nullity `-1`.

`certificate.to_profile()` returns `SketchCertifiedProfile` coupling the native
profile with point/radius source-fidelity enclosures, not merely a weighted
residual. `sketch_dimension_derivatives(certificate, budget=...)` admits only
fully constrained, certified-rank, converged regular endpoints. Rows follow
flattened points then radii; columns follow authored distance/radius/angle
dimensions. The derivative uses native implicit-root attachment at the endpoint,
not LM iteration differentiation or finite differences.

`Sketch.constraint_ids` names every authored constraint;
`constraint_priorities` declares zero mandatory and ascending optional tiers.
`solve_priority_sketch(sketch, budget=...)` solves cumulative tiers and advances
`lowest_satisfied_tier` only after strict root inclusion. Underconstrained tiers
remain provisional. `tier_certificates`, unsatisfied/rejected IDs, and authored
residual norms retain every attempted decision. A failed tier rejects itself
and all lower-priority tiers; no certified tier yields `-1`.


`DesignConstraintSystem` solves geometry-state constraints without coupling to a
particular representation. Available constraints include parameter targets and
equalities, interior/exterior clearance, measure and boundary-measure targets,
boundary-point conditions, and B-Rep seam compatibility. Physical parameter
bounds and `CompiledGeometry.validity(state)` reject inadmissible trials without
projecting them. Invalid or unresolved initial geometry validity refuses before
iteration. Native trial refusal or stationary nonconvergence remains observable
through solver status and diagnostics.

Ball primitives, including circles and spheres, orthotopes, ellipses, and
ellipsoid families require finite centers and strictly positive finite radius
or size components after design-state updates as well as at source construction.
Their geometry-validity evidence reports dimensionless positive-extent margins
normalized by the parameter's declared physical scale. Restricted field
certificates remain inconclusive rather than bypassing their representation
contract.

## Bounded global design search

`phydrax.optim.DifferentialEvolutionSearch` is intended for low-dimensional geometry
problems whose residual objective is nonsmooth, multimodal, or poorly served by a
single local initialization. It searches the squared residual from
`DesignConstraintSystem` over an explicit finite box. Differential evolution is a
stochastic global heuristic: convergence reports population-fitness dispersion, not
a proof of global optimality or coverage of every basin.

Search bounds and physical schema bounds have different roles. `ParameterSpec.bounds`
describe physical admissibility and may be one-sided or absent. `search(..., bounds=...)`
defines the finite algorithmic box. Every trainable degree of freedom must have finite
lower and upper search limits, and those limits must remain inside any finite physical
schema bounds. Scalar limits broadcast across a parameter; array limits must match its
declared shape.

The root PRNG key is required. The initial population uses the typed
`phydrax.sampling` design substrate; Latin hypercube is the default, while scrambled
Sobol and the other supported reference designs may be selected explicitly. The
current state is inserted into the first population member. Generated candidates are
reflected into the box before both evaluation and storage.

```python
import jax.random as jr
import phydrax as phx
from phydrax.units import METER

geometry = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
).compile(coordinate_unit=METER)
center = phx.geometry.ParameterId("body", "center")
radius = phx.geometry.ParameterId("body", "radius")

system = phx.geometry.DesignConstraintSystem(
    geometry,
    (phx.geometry.ParameterTarget(radius, 1.5),),
)
search = phx.optim.DifferentialEvolutionSearch(
    32,
    100,
    design=phx.sampling.SobolDesign(scrambled=True),
)
global_result = system.search(
    search,
    key=jr.key(0),
    bounds={
        center: ((-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)),
        radius: (0.25, 2.5),
    },
)

# Local refinement is a separate, explicit phase.
local_result = system.solve(initial_state=global_result.state)
if not bool(local_result.converged):
    raise ValueError(
        f"design.local_result.converged must be true; got status={local_result.status}"
    )
optimized_geometry = geometry.with_state(local_result.state)
domain = phx.domain.GeometryDomain(optimized_geometry)
```

`DesignSearchResult` preserves the final population and objectives, best-objective
history, exact generation and objective-evaluation counts, invalid-evaluation count,
resolved bounds, root key, design signature, and termination reason. Its global
`converged` flag is independent of `ConstraintSolveResult.converged` from the optional
local phase. Keeping the phases separate makes extra evaluations and failure modes
observable.

Global search evaluates `CompiledGeometry.validity()` for every candidate. Parameter
finiteness and declared `ParameterSpec.bounds` are always checked. Restricted
representations require a `GeometryValidityProvider`; without one their disposition
is `INCONCLUSIVE` and search fails before objective evaluation. Invalid provider-backed
candidates become explicit invalid objective evaluations.

## Exact sweeps and fixed-topology realization

`Extrusion` lifts any full-dimensional region in R-d into a centered region in
R-(d+1). `Revolution` remains the explicitly axisymmetric 2D-to-3D operation.
`CompiledGeometry.validity()` exposes parameter and representation validity as
`GeometryValidityEvidence`.

`Tube(path, radius, feature_id=..., include_parameter_range=...)` is the path
distance minus radius, with a JAX field and normals. Admitted paths are finite
line `CurveRange`s (round caps, infinite reach) and full `CircleCurve`s with
exactly orthonormal binary axes (reach equal to circle radius). Radius is a
length-unit design parameter and must be strictly between zero and reach.
Other paths refuse rather than estimating reach from samples.
`distance_bounds(points)` outward-encloses the **path** distance with rational
arithmetic and directed square roots under the coordinate request budget.
See [tube and range workflow](../guides_differentiable_geometry.md#certified-tubes-and-construction-ranges).


`ImplicitPointProjectionPlan` supplies fixed-shape normal-gauge boundary motion.
`discover_implicit_curve` discovers an oriented closed planar segment topology.
`discover_implicit_surface` creates a host-side `ImplicitSurfacePlan` whose JAX
runtime preserves triangle connectivity and reports sign, root, QEF, orientation,
and intersection evidence.
The plan retains the full accepted initial realization evidence as dynamic
numerical leaves, including projection, QEF, finiteness and status checks.
The proof identity binds every leaf to the exact initial design state, source,
topology and realized coordinates, and participates in the plan identity.
The original `base_vertices` fallback and `faces` remain unchanged: because
projection/QEF coordinates can differ from that fallback, initial execution
still recomputes and meters them rather than retaining a second coordinate bank.
It reuses only the bound proof; changed or traced states execute the full
differentiable realization. Stored-plan reconstruction rederives and checks the
complete proof against the restored source state; edited proof leaves or
incompatible bindings refuse rather than receiving a historical identity.

`discover_adaptive_implicit_surface(geometry, domain=..., policy=...)` refines a
2:1-balanced octree over `domain` and never treats a leaf as empty unless its
value enclosure excludes zero. `AdaptiveImplicitSurfacePolicy.enclosure` selects
the bound source: `"interval"` evaluates the field program in outward-rounded
interval arithmetic with independent directional-derivative enclosures,
`"lipschitz"` uses the certificate's established Lipschitz bound for values only,
and `"sampled"` makes no enclosure claim. The returned `AdaptiveImplicitSurface`
carries the dual-contoured `TriangleMesh` (one vertex per leaf boundary cycle,
polygons on shared minimal edges, so level transitions are crack free),
`AdaptiveImplicitSurfaceEvidence` (`accuracy` `"certified"`, `"enclosed"` or
`"sampled"`, status flags, unresolved boxes with their issues, residuals and
evaluation counts), the `CertifiedImplicitCover`, a `CertifiedImplicitTopology`
when certified, and an `ImplicitVolumeQuery` that classifies points and boxes
as inside, outside or unknown. Budget exhaustion stops refinement and reports
the remaining boxes as unresolved. Tangential zeros, near-zero gradients and
sharp creases whose wedge an axis direction crosses remain unresolved rather
than certified; the adaptive product is not differentiable.
`AdaptiveImplicitSurfacePolicy.flatness_tolerance` bounds each surface leaf's
diagonal times its gradient-direction spread. For an exact three-dimensional
ball kernel or a full torus kernel, that spread is the smaller of the enclosure
spread and an analytic cone over the leaf box. The cone comes from the closed-form
distance gradients, with outward-rounded box and meridional-rectangle corners.
Leaves that reach the ball center, the torus axis or the core circle keep the
unit spread. Other kernels, transformed sources and partial tori use the
enclosure spread alone. With a rigorous value enclosure, the same exact kernels
refine only their support box, `c +- r` or `c +- (R + r, R + r, r)`, padded by a
quarter of the reach and clipped to `domain`. The rest of the authored domain is
tiled exactly by at most six slabs, and each slab joins the cover only after its
value enclosure excludes zero. If any slab fails, discovery refines the whole
domain. `cover.domain` and `volume.domain` remain the authored domain. Volume
queries evaluate points outside the refined box with the bound source.

`NeuralImplicitRegion` is the region `{x in B : phi(x; w) <= 0}` of a scalar
network over declared axis-aligned bounds `B`. The network's PARAMETER arrays are
design parameters of the compiled `DesignState`; FIXED arrays (`fixed_field`
data) stay fixed kernel data, and networks with model state are refused.
Construction refuses without a Lipschitz bound (constructed for a plain `MLP` or
declared), an evaluation-error bound, sign margins on explicit sample points, and
a topology resolved from the field signs on a `discovery_resolution` lattice. The
evidence is sampled, not a covering proof: a zero-set component between samples
can go undetected, so the field certificate reports `SignReliability.LOCAL`,
`ZeroSetAccuracy.APPROXIMATE`, and no `topology_identity`.
`NeuralImplicitRegion.recertify(state)` rejects trained weights whose sampled
margins fail or whose sampled `ImplicitRegionTopology` changed; other states
report inconclusive validity.

`FiniteElementMeshMotionPlan` is owned by `phydrax.discretization`; it consumes any
structural fixed-route boundary provider, extends it to the interior along an explicit
`FiniteElementMeshMotionRoute`, and returns a safe `FiniteElementRuntimeData` plus
signed corner-Jacobian evidence.

See [Differentiable fixed-topology geometry](../guides_differentiable_geometry.md)
for contracts, nonclaims, and a complete workflow.

## Reconstruction with provenance

Point-cloud, planar, terrain, and LiDAR reconstruction are explicit pipelines:
`reconstruct_planar_region`, `reconstruct_surface_region`,
`reconstruct_dem_region`, and `reconstruct_lidar_region`. They return a
`ReconstructedGeometrySource` carrying an immutable `ReconstructionReport` with
input/output counts, algorithm parameters, watertightness, winding consistency,
recentering, warnings, and an input digest. Invalid reconstruction raises
`ReconstructionFailure` with the same report; approximation is never hidden behind
a primitive constructor.

Planar and terrain reconstruction triangulate with the exact native Delaunay
kernel. Point-cloud surfaces use native screened Poisson reconstruction:
`estimate_point_normals` supplies PCA normals over exact BVH neighborhoods,
oriented along a minimum spanning forest (`NormalOrientation`) with
`NormalEstimationEvidence`; the indicator is solved with continuous trilinear
elements either on a 2:1-balanced adaptive octree refined only in sample cells
(`PoissonDiscretization` `"octree"`, the default, with hanging-node
constraints) or on the full regular grid (`"regular"`), by
Jacobi-preconditioned conjugate gradients (`PoissonSolveEvidence`, including
unknown, leaf, and hanging-node counts), and extracted by crack-free marching
tetrahedra. Reports add the Euler characteristic and
`SampledSurfaceDeviation`, a sampled two-sided deviation rather than a
continuous Hausdorff certificate. Poisson smoothing does not preserve features
below the finest cell width.

`ReconstructionRobustness` declares how imperfect samples are treated and each
policy reports its evidence: statistical outlier removal
(`OutlierRemovalEvidence`), surface farther than a coverage distance from every
sample (`SamplingCoverageEvidence`, closed and reported or refused by
`IncompleteSamplingPolicy`), stacked sheets thinner than the resolution
(`ThinFeatureEvidence`), and sample components whose sampled indicator is not
fit by the level set, which are refused (`ComponentFitEvidence`).
`reconstruct_trimmed_surface` instead trims unsupported triangles and returns
the open `TriangleSurface` in a `TrimmedSurfaceReconstruction`.

Reconstructed `NativePlcSource` revisions retain the actual point-cloud digest,
screened-Poisson report, and extracted triangle coordinates. Native constrained
volume fill certifies that represented PLC and independently checks signed
boundary volume while retaining region identity. Reconstruction's sampled
distances remain separate from exact extracted-domain coverage; an analytic
reference surface never replaces the reconstructed triangles.
`FiniteElementTopologyTransaction` and `CompositionRebind` preserve the accepted
mesh and state when a continuation proposal is rejected. Repaired-envelope
transitions retain raw/repaired identities, explicit feature/topology
permissions, directed bounds, and inclusion margin.
Generated interior vertices inherit PLC authority only from one explicit
incident-cell/source-parent owner and an exact mapped interior-support proof;
boundary, interface, and feature vertices still require their lower-stratum
witnesses. Strict volume audits remain active after adaptation. Independent
volume and scalar-inventory gates cover every retained/refined simplex block,
including global vertex routing, rather than assuming a single carrier block.
The target PLC support proof is handed directly to final acceptance through its
owning prepared-evidence record. Reuse requires the same actual mesh, coordinate
map, represented domain, region rows, complete certification request, original
certificate limits, and scoped source/facet/tolerance tuples. The embedding and
coverage work ledger is retained once, not recomputed or replaced with zero
cost. The strict target audit remains independent, and any required whole-source
or scoped fidelity checks still run; a volume proof is not a fidelity verdict.

Native circumcenter failure is reported as structured meshing failure before
publication, rather than replacing nonfinite quality evidence with a finite
value. Envelope carrier initialization enforces the request's actual native
scratch-allocation allowance.
Ordinary single-device organization-ID banks use immutable NumPy host preparation
instead of repeated dynamic-size JAX selections. Named/collective placements,
nonaddressable arrays, and tracers retain their existing array route. This does
not merge source facets or change target patch, zone, label, scope, inventory, or
lineage identity; collapsed sources retain their existing deciding-source
precedence and ambiguous membership still refuses.

Restricted tetrahedral roots reuse the linear embedding theorem only after
proving every complete source expression affine, every exact root corner
binary64-representable without rounding, and all shared scientific corner
identities consistent. This proves equality of whole affine maps, not a corner
surrogate. Curved, rational, and nonrepresentable roots retain their generic
theorem. Candidate/ray/subdivision ledgers retain the original remaining caps;
a failed complete-root premise names its actual source-cell IDs without
claiming an overlap of the retained children.

## Exact predicates, triangulations, and diagrams

`orient2d`, `orient3d`, `incircle`, and `insphere` return a `PredicateResult`
(int8 `signs` in `PredicateSign`, boolean `certain`). `PredicateMode.FILTERED`
(host NumPy) and `PredicateMode.FILTERED_DEVICE` (pure JAX, jittable) certify
signs with Shewchuk's static error bounds for the input dtype and report
`UNCERTAIN` otherwise; a certified sign is never wrong. `PredicateMode.EXACT`
resolves uncertain host entries with the native `phydrax-meshcore` adaptive
expansion library when the optional `phydrax[meshcore]` extra (or
`PHYDRAX_MESHCORE_LIBRARY`) is available. Otherwise it evaluates the binary64
coordinates as exact dyadic rationals, so the host route remains exact without
an optional provider. `GeometryPrecisionPolicy(predicate_mode=...)` selects the
route of host geometry decisions; nonfinite inputs and resource exhaustion
remain fail-closed.

`segment_intersections_2d(a, b, c, d, mode=...)` classifies closed segments as
`SegmentIntersectionStatus` `DISJOINT`, `PROPER_CROSSING`, `ENDPOINT_CONTACT`
(one common point that is an endpoint), `COLLINEAR_OVERLAP` (a common piece of
positive length), or `UNCERTAIN`.
`polygon_simplicity_2d(vertices, mode=..., maximum_candidate_pairs=...)`
certifies `(..., n, 2)` vertex loops as `PolygonSimplicityStatus` `SIMPLE`,
`SELF_INTERSECTING`, or `UNCERTAIN`: non-adjacent edges must be disjoint and
adjacent edges may meet only at their shared vertex, so repeated vertices,
touching vertices, and doubled-back edges are self-intersections. Candidate edge
pairs stream from a BVH broad phase over exact edge boxes. The result carries the
certified orientation, offending and unresolved pair counts, the processed
candidate count, and whether the capacity was exhausted; exhaustion leaves every
unproved loop uncertain. Both are host algorithms (`FILTERED` or `EXACT`) whose
classes are exact wherever every contributing sign is certified.

`phydrax-meshcore` is released in lockstep with Phydrax: `phydrax[meshcore]`
pins the identical version. Before binding other entry points, Phydrax verifies
`phx_mc_abi_contract`, the SHA-256 of the public C header with LF line endings.
Missing or different contracts, another release, missing bound symbols, and
null/malformed release or build identities produce `MeshcoreUnavailableError`
with the reason. Editing `native/meshcore/include/phydrax_meshcore.h` requires
updating `phydrax/_meshcore.py:_ABI_CONTRACT` and rebuilding the native library;
an old binary is never treated as compatible merely because it exports the
same function names. No C++ exception crosses the C ABI: a refused allocation
is the call status `CAPACITY_EXCEEDED`.
`phx_mc_build_hash` is the configured digest of the canonical meshcore source
and header map, while `MeshcoreLibrary.binary_hash` is SHA-256 of the exact
shared-library bytes selected by the loader. `phx_mc_build_configuration`
records compiler, platform, floating-point flags, sanitizer selection, and build
configuration. Qualification runtime records retain those three identities
separately; an installed wheel and an isolated local build may share source and
ABI identity while having different binary digests. None of these identities is
a numerical qualification or benchmark result.

The root package does not bundle or rename an external geometry engine.
`phydrax[meshcore]` installs the separately packaged, version-identical
`phydrax-meshcore`; `PHYDRAX_MESHCORE_LIBRARY` is an explicit override. A
missing native library is not repaired by a SciPy/Qhull or provider fallback,
and optional OCCT, Gmsh, Mmg, fTetWild, Manifold, OpenVDB, Poisson, VoroCrust,
Omega_h, TIOGA, and METIS results remain separate provider evidence.
To build and test the library from source, and to repeat the tests under
AddressSanitizer and UndefinedBehaviorSanitizer:

```console
cmake -S native/meshcore -B build/meshcore
cmake --build build/meshcore && ctest --test-dir build/meshcore
cmake -S native/meshcore -B build/meshcore-sanitize -DPHX_MC_SANITIZE=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/meshcore-sanitize && ctest --test-dir build/meshcore-sanitize
```

`DelaunayTriangulation` (2D/3D) and `ConstrainedDelaunayTriangulation`
(segment recovery, hole carving, Ruppert/Chew refinement bounded by
`max_steiner`) are exact, deterministic, and canonically ordered; ties are
resolved by index-ordered symbolic perturbation. `VoronoiDiagram` and
`PowerDiagram` clip each generator's bisector halfspaces to a box and optional
convex domain with exactly classified clipping and store the cells as
`DiagramCells` CSR with measures and centroids. Each result carries
`TriangulationEvidence` (route, provider, status, provider identity, counts).
`DelaunayTriangulation(points, provider="qhull")` selects SciPy's Qhull with
recorded options instead (predicate mode `filtered`: deterministic, not
certified, no optional provider required) for auxiliary constructions such as
meshfree auxiliary-space preconditioning.

`SimplexQualitySubcomplex(points, simplices, minimum_quality=q)` screens a
triangulation by normalized volume-length ratio `|T| / v_d(l_rms)` (1 for the
regular simplex): flat simplices are never kept, simplices below `q` (3-D
slivers) are excluded, and excluded simplices are restored best-first only
where a vertex would be uncovered or the facet-connected component count would
grow. `SimplexQualityEvidence` records the threshold, degenerate, excluded and
restored counts, the excluded measure fraction, and the minimum retained
quality.

## Common refinement (supermesh)

`prepare_common_refinement(source, target, policy=CommonRefinementPolicy())`
certifies the overlap of two `CellMesh` instances of one dimension (2D or 3D,
any mix of triangle, quadrilateral, polygon, tetrahedron, hexahedron, prism,
pyramid, and polyhedron blocks) and requires meshcore. Triangles, tetrahedra,
and convex polygons are single convex pieces; every other cell is the cone of
its canonically triangulated boundary (faces fanned from their smallest vertex,
so neighbors agree on shared faces) from one of its vertices, accepted only when
all nondegenerate cone simplices share one exact orientation. Float64 BVHs over
the cell boxes enumerate candidate pairs; meshcore clips every piece pair with
exactly classified vertices in batches of bounded working set.

`PreparedCommonRefinement` stores CSR rows grouped by target cell
(`target_offsets`, `source_cells` ascending per row, `volumes`,
`first_moments` = integral of x, optional `second_moments` = integral of x x^T,
optional `simplex_offsets`/`simplices`: an exact signed simplex partition of
every overlap for quadrature), the certified cell measures and first moments of
both meshes, cell global IDs, and mesh identities. Nothing is repaired:
`CommonRefinementStatus` reports uncertified cells (`INVALID_GEOMETRY`),
unresolved filtered predicates or coordinates outside the exact domain
(`PREDICATE_UNCERTAIN`), native clip failures, `DOUBLE_COVERAGE`,
`COVERAGE_GAP` against the `CommonRefinementCoverage` requirement (`COMPLETE`,
`TARGET`, `SOURCE`, `PARTIAL`), and `RESOURCE_LIMIT` refusals of the candidate,
accepted-pair, and memory limits. Cell `i` is covered when its covered measure
differs from its measure `m_i` by at most
`coverage_tolerance * m_i + 256 eps h_i^d` (`h_i` the cell box diagonal).
`CommonRefinementEvidence` carries per-cell defects and tolerances, gap and
double-coverage counts, candidate/accepted/piece-pair counts, and retained and
working bytes. Finite-volume remap, block-AMR cut-cell transitions, and
finite-element L2 projection transfers consume this one artifact.

Host storage is admitted before constructing decomposition, BVH/traversal,
candidate, clipping/reduction, and publication banks, and released when its
last temporary consumer expires. Retained output aliases survive call-scratch
cleanup. A memory refusal after both decompositions may return `RESOURCE_LIMIT`
with empty overlaps and the **real** per-cell measures and first moments.
If even the canonical negative artifact cannot fit, preparation instead raises
`CoordinateEnclosureResourceError` **before allocation**, carrying the resource,
original limit, requested bytes, and completed bytes; it never substitutes zero
scientific columns. Static mesh-kind and dimension errors retain precedence.

Passing a live native `CoordinateEnclosureBudget` uses that same ticket and
native pool for host admissions and actual native clipping, with the policy
tightening (not renewing) its native allowance. Native predicates preserve the
uncached decomposition query schedule. Standalone live `HostBatchTicket`
budgets remain supported, including their existing zero host-source-query
allowance, but provide **PARTIAL native resource coverage**: they account host
preparation/storage, while native execution is bounded by the existing finite
`CommonRefinementPolicy` and any supplied native scope. They are not a
same-ledger complete-request qualification, and preparation does not mint an
extra host/native work, query, or storage allowance.

## Triangle-surface arrangements and Booleans

`arrange_triangle_surfaces(first_vertices, first_triangles, second_vertices,
second_triangles, operands=((vertices, triangles), ...),
limits=SurfaceArrangementLimits())` splits two or more embedded triangle surfaces
(open sheets admitted) in one simultaneous exact arrangement. Float64 BVHs
enumerate candidate pairs; the native arrangement represents every constructed
point implicitly by the original input planes defining it (line-plane, coplanar
edge-edge and triple-plane points) and decides every orientation, order and
equality on those implicit points with filtered predicates and exact dyadic
fallback, so a point where three or more operands meet is one exact vertex. Each
touched triangle is split by exact point insertion and constraint recovery, and
points are welded across operands by exact equality, never by proximity.
Coordinates are rounded only on publication, with rigorous max-norm bounds.
`SurfaceArrangement` publishes welded vertices, bounds, construction families,
per-operand feature simplices and every exact source-face incidence, fragments
with source operand/face, per-operand coplanar coverage and orientation, and the
contact edges. Self-intersecting or degenerate inputs, coordinates outside the
exact contact-predicate domain, budget exhaustion, and exact arrangements whose
binary64 publication would invert or collide fragments
(`unrepresentable_publication`) raise `SurfaceArrangementError` with a
`SurfaceArrangementStatus`.

`surface_boolean(first, second, SurfaceBooleanOperation.UNION | INTERSECTION |
DIFFERENCE, operands=(...))` computes the n-ary Boolean of closed, edge-manifold,
outward oriented `SurfaceModel` solids (`DIFFERENCE` removes every later operand
from the first). A `SurfaceBooleanResult` operand declares its region expression
over its original models, so nested calls such as
`surface_boolean(surface_boolean(a, b, UNION), c, DIFFERENCE)` evaluate the CSG
tree on one exact arrangement of the original operands instead of re-cutting
rounded intermediates. Fragment components bounded by contact curves are
classified against every other operand exactly by the native arrangement: the
winding number is counted along a symbolically perturbed axis ray from the
centroid of a representative's implicit corners with filtered and exact dyadic
predicates, with no floating-point trust margin. The classification work
(winding-matrix entries plus representative/triangle scans, reported as
`classification_matrix_entries` and `classification_pair_scans`) is admitted
against `maximum_winding_evaluations` and any active native execution budget
before allocation. Coplanar fragments follow exact per-operand coverage
orientation and are emitted once, from the lowest contributing operand whose
subexpression actually bounds the region. Open or non-solid operands raise
`SurfaceBooleanError`. `SurfaceBooleanResult` carries the oriented triangles
with per-triangle original operand, source cell global ID, source vertices and
barycentric weights (`source_corner_values` transfers operand vertex data),
vertex construction bounds, `operand_ids`, `SurfaceClosureEvidence` (unpaired and
non-manifold edges, non-manifold vertices, components, signed volume), the
arrangement evidence and a `SurfaceModel` for nonempty edge-manifold results.
Empty and disconnected results are legitimate; tangent contacts may yield
closed results with non-manifold vertices or edges, which the evidence reports.

## Core API

::: phydrax.geometry.CompiledGeometry

---

::: phydrax.geometry.GeometrySource

---

::: phydrax.geometry.FieldCertificate

---

::: phydrax.geometry.ExactSDFEnclosureCertificate

---

::: phydrax.geometry.QualifiedSharpGeometry

---

::: phydrax.geometry.SharpGeometryEvidence

---
---

::: phydrax.geometry.GeometryValidityEvidence

---

::: phydrax.geometry.Extrusion

---

::: phydrax.geometry.Revolution

---

::: phydrax.geometry.discover_implicit_curve

---

::: phydrax.geometry.discover_adaptive_implicit_surface

---

::: phydrax.geometry.AdaptiveImplicitSurfacePolicy

---

::: phydrax.geometry.AdaptiveImplicitSurface

---

::: phydrax.geometry.AdaptiveImplicitSurfaceEvidence

---

::: phydrax.geometry.ImplicitVolumeQuery

---

::: phydrax.geometry.ImplicitCurvePlan

---

::: phydrax.geometry.ImplicitPointProjectionPlan

---

::: phydrax.geometry.ImplicitSurfacePlan

---

::: phydrax.geometry.NeuralImplicitRegion

---

::: phydrax.geometry.NeuralImplicitCertificate

---

::: phydrax.geometry.ImplicitRegionTopology

---

::: phydrax.geometry.BoundaryAtlas

---

::: phydrax.geometry.CubatureAtlas

---

::: phydrax.geometry.TriangleMesh

---

::: phydrax.geometry.BRepModel

::: phydrax.geometry.CADRevision

---

::: phydrax.geometry.CADSelectionSet

---

::: phydrax.geometry.AssociationGraph

---

::: phydrax.geometry.BRepPartitionPlan

---

::: phydrax.geometry.BRepPartitionResult

---

::: phydrax.geometry.PlanarEmbedding

::: phydrax.geometry.AbstractBRepProjection

---

::: phydrax.geometry.PlanarPartitionPlan

---

::: phydrax.geometry.ProcessStack

---

::: phydrax.geometry.ProcessStackResult

`ProcessStackResult.operand_artifact_paths` records `(operand_name, Path)` pairs
in stack operand order for the lossless native `.phx` operand archives.
These locations are distinct from model source identities; returned operand
models are restored from the archives.

---

::: phydrax.geometry.StackRegion

---

::: phydrax.geometry.StackVoid

---

::: phydrax.geometry.ZInterval

---

::: phydrax.geometry.FixedTopologyBRepSource

---

::: phydrax.geometry.Sketch

---

::: phydrax.geometry.DesignConstraintSystem

---

::: phydrax.geometry.DesignSearchResult

---

::: phydrax.geometry.ReconstructionReport

---

::: phydrax.geometry.estimate_point_normals

---

::: phydrax.geometry.PointNormals

---

::: phydrax.geometry.NormalEstimationEvidence

---

::: phydrax.geometry.PoissonSolveEvidence

---

::: phydrax.geometry.SampledSurfaceDeviation

---

::: phydrax.geometry.ReconstructionRobustness

---

::: phydrax.geometry.reconstruct_trimmed_surface

---

::: phydrax.geometry.TrimmedSurfaceReconstruction

---

::: phydrax.geometry.prepare_common_refinement

---

::: phydrax.geometry.PreparedCommonRefinement

---

::: phydrax.geometry.CommonRefinementPolicy

---

::: phydrax.geometry.CommonRefinementCoverage

---

::: phydrax.geometry.CommonRefinementStatus

---

::: phydrax.geometry.CommonRefinementEvidence

---

::: phydrax.geometry.arrange_triangle_surfaces

---

::: phydrax.geometry.SurfaceArrangement

---

::: phydrax.geometry.SurfaceArrangementEvidence

---

::: phydrax.geometry.SurfaceArrangementLimits

---

::: phydrax.geometry.SurfaceArrangementError

---

::: phydrax.geometry.surface_boolean

---

::: phydrax.geometry.SurfaceBooleanOperation

---

::: phydrax.geometry.SurfaceBooleanResult

---

::: phydrax.geometry.SurfaceClosureEvidence

---

::: phydrax.geometry.SurfaceBooleanError

## Native source, CAD, and certification records

The following public records expose the exact owners used by native meshing.
They do not turn a sampled bound into a global certificate or an optional
external shape into native source authority.

::: phydrax.geometry
    options:
      members:
        - SphereMaterialCellAtlas
        - SphereMaterialInverse
        - SphereProjectiveReferenceMap
        - SphereProjectiveTriangleBounds
        - prepare_sphere_material_atlas
        - sphere_material_atlas_from_result
        - AnalyticBoundaryCoverCapacityError
        - AnalyticImplicitFamily
        - AnalyticImplicitProfile
        - BRepAssemblyContainer
        - BRepQualifiedIncidence
        - ParametricCurveBoundarySource
        - CompartmentImageInterpretation
        - CompartmentMeshingSource
        - AffineChainResourceEvidence
        - AffineChainResourceMetric
        - AffineChainResourceRole
        - SourceBoundaryChartCover
        - SourceBoundaryChartQuery
        - boolean_brep
        - BRepBooleanFailure
        - BRepBooleanOperation
        - BRepBooleanPolicy
        - BRepBooleanResult
        - BRepSewingContact
        - BRepSewingContactRelation
        - BRepSewingEdgeImage
        - BRepSewingFailure
        - BRepSewingLineage
        - BRepSewingPolicy
        - BRepSewingResult
        - sew_brep
        - AbstractTrimCurve
        - AdaptiveImplicitBoxIssue
        - AdaptiveImplicitSurfaceStatus
        - CoincidentParameterRegion
        - CoordinateMapScope
        - CurveIntersectionResult
        - CurveSurfaceIntersectionRoot
        - CurveRange
        - CurveTrimLoop
        - CurveTrimSegment
        - ApproximateSourceCoverageCertificate
        - CadSourceDomain
        - DomainCoverageCertificate
        - FieldBoundOrigin
        - GlobalEmbeddingCertificate
        - PeriodicAffineQuotientEvidence
        - ImplicitBoundOrigin
        - ImplicitBoundarySource
        - EmbeddedPlanarBoundarySource
        - ImplicitDiscoveryAccuracy
        - ImplicitDiscoveryEnclosure
        - ImplicitTopologyPremise
        - ImplicitVolumeClass
        - ImplicitVolumeClassification
        - IntersectionCurve
        - IntersectionCurvePoint
        - IntersectionCurveSide
        - IntersectionEndpointKind
        - IntersectionPCurve
        - MappedBoundaryDegreeEvidence
        - MappedBoundaryDegreeStatus
        - MappedReferenceDomain
        - MeshCertificateBinding
        - MeshCertificateEntityKind
        - MeshCertificateFinding
        - MeshCertificateLimits
        - MeshCertificateStatus
        - MeshFindingStatus
        - ParametricIntersectionCertificate
        - ParametricIntersectionKind
        - ParametricIntersectionPoint
        - ParametricIntersectionPolicy
        - ParametricIntersectionWork
        - PiecewiseLinearDomain
        - PeriodicDelaunayTriangulation
        - PeriodicImageBudgetError
        - PeriodicImageLimit
        - PeriodicTriangulationEvidence
        - PolygonTrimLoop
        - RestrictedPowerDiagram
        - SourceBoundSemantics
        - SourceBoundaryDistance
        - SourceBoundaryQuery
        - SourceBoundarySamples
        - SourceFidelityCertificate
        - SurfaceIntersectionResult
        - SurfaceRegion
        - TrimRootEndpoint
        - TrimClassification
        - UnresolvedIntersectionReason
        - UnresolvedParameterRegion
        - certify_approximate_source_coverage
        - certify_domain_coverage
        - certify_global_embedding
        - certify_source_fidelity
        - establish_implicit_cover
        - implicit_state_id
        - intersect_curve_ranges
        - intersect_curve_region
        - intersect_surface_regions
        - AbstractCurve
        - BRepContainmentResult
        - BRepGeometry
        - BRepMeasureResult
        - BRepOccurrence
        - BRepQueryPolicy
        - BRepSurfaceQueryResult
        - BRepTessellationPolicy
        - CircleCurve
        - EllipseCurve
        - ExtrusionSurface
        - LineCurve
        - PlanarProfile
        - PreparedBRepQuery
        - ProfileArc
        - ProfileLine
        - ProfileLoop
        - ProfilePlane
        - RationalBezierPiece
        - RevolutionSurface
        - RuledSurface
        - brep_box
        - brep_cone
        - brep_cylinder
        - brep_extrusion
        - brep_offset
        - brep_planar_face
        - brep_revolution
        - brep_sphere
        - brep_torus
