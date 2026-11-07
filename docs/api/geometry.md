# Geometry substrate

`phydrax.geometry` is the representation-aware geometry layer. It separates
host-side construction from JAX execution and gives analytic primitives, simplicial
meshes, CAD B-Reps, CSG expressions, and reconstructed geometry one compiled
contract.

## Source, kernel, state, and realization

A `GeometrySource` is the authoritative construction object. Calling `compile()`
produces a `CompiledGeometry` containing:

- an immutable `GeometryKernel` that defines algorithms and topology;
- a dynamic `DesignState` containing all trainable numeric values;
- a `ParameterSchema` with stable, feature-scoped parameter identities; and
- a `GeometryTolerance` used by tolerance-sensitive queries.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

source = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
)
geometry = source.compile()

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

## Cell embedding and source-domain coverage

Local coordinate-map validity, global embedding, and source coverage answer
different questions. A positive Jacobian does not exclude overlapping cells,
and equal total volume does not establish the requested material interfaces.

`certify_global_embedding(mesh, geometry, validity)` checks the actual coordinate
maps and their declared shared entities. Exact affine contacts include planar
cells, standard volume cells, and convex polyhedra. Periodic contacts include the
zero image and translated images; vertex identity includes its declared winding,
and quotient incidence is oriented as a physical cell chain. A periodic subdomain
may have a physical boundary without being a full fundamental-domain cover.
Nonlinear contacts without an applicable continuous proof remain `unresolved`.

Polyhedral cells may share a straight segment represented by several declared
edges. Contact is accepted on that segment only after proving that the edges
form a complete, nonoverlapping collinear chain connected by the declared
vertex IDs. Coincident but distinct vertices or disconnected edge pieces do not
authorize contact on their convex hull.

For polynomial maps on convex reference cells, exact center-Jacobian
preconditioning and positive-definite Bernstein controls can prove mapped-cell
injectivity. Shared curved edges and quadrilateral facets use exact restriction
traces and opposite-side polynomial signs; curved facets may use an exact
implicit relation. A missing exact sign or trace proof remains unresolved.
Facet-sharing polynomial hexahedra can also use the glued-facet monotonicity
proof. Exact shared traces and an exact affine reference glue are required; the
two glued reference cells must form a convex union. A single preconditioner must
make the symmetric part of every enclosed Jacobian positive definite, with all
signs checked exactly. The resulting strict monotonicity proves that their
physical images meet only on the shared facet, which need not be planar. The
geometry embedding owner checks these premises independently of source ancestry;
similar cell shapes or matching corners do not establish a common physical map.
The extension adds hexahedra to the existing per-pair facet proof: it retains the
same bounded reference-facet enumeration and Jacobian-control work per pair,
not an additional all-cells scan. Inconclusive premises remain unresolved.

The contact owner may also represent a polynomial volume cell by a
planar-faced solid, but only after proving injectivity of the complete physical
map and proving that every complete face trace has exactly its planar corner
polygon as image. The exact closed-face/star decomposition must succeed, and
shared traces are still compared on the original maps. This is not a
corner-only planarity theorem or a claim of common source parenthood. The new
preparation performs a per-cell, per-face exact planarity screen followed by
the existing map/trace and star proofs for eligible cells. Before this route,
all mapped volume pairs used curved-map separation; after it, proved
planar-faced cells reuse exact tetrahedral contact filtering. The existing BVH
still enumerates every candidate; filtering now expands piece pairs in bounded
cell-pair windows rather than retaining a full BVH block's expanded work.
Nonplanar or undecided cells retain the curved-map route.

For a nonperiodic three-dimensional complex, the exact global-degree route may
close unresolved pairwise contacts only when every exterior facet trace is
proved affine and nondegenerate, exact triangle contacts establish an embedded
complete boundary, interior facet loops cancel with exact matching mapped
traces, and exact cell, edge, and vertex links certify one connected oriented
volume manifold with one spherical boundary. With positive local Jacobians,
the boundary has degree one inside; every regular preimage contributes
positively, so the global map is injective. Curved exterior facets and any
failed boundary, trace, or topology premise remain unresolved or violated.
Optional manifold and shell-degree premises decline only this theorem; complete
exact pairwise contact proofs remain available for a valid embedded complex.

Mapped targets may also use a forward nonaffine reference-chain proof when
retained `ReferenceCellMapWitnesses` provide exact target-to-parent chart
expressions, the target physical map equals the parent map composed with those
charts, every whole-cell chart Jacobian is strictly positive, and exact shared
traces plus the planar oriented boundary prove a degree-one partition of each
parent cell. The current parent global-embedding certificate is required before
composition is accepted. Cold reconstruction rechecks the parent, chart bank,
partition, and target map; chart IDs or volume agreement alone are not proof.


Pyramid cell maps retain their canonical vertex and facet strata. Bernstein
proofs use collapsed-box expressions only when exact constant apex continuation
is established for the represented map and required derivatives.


`PiecewiseLinearDomain` retains oriented source facets and explicit region
identities. `MappedReferenceDomain` retains an authoritative reference mesh and
its coordinate maps. `certify_domain_coverage` consumes dense indices into the
domain's `region_ids`, verifies the oriented exterior/interface partition, and
retains exact region measures as additional consistency evidence:
MappedReferenceDomain requires an explicit Cartesian SpatialCoordinateContract for the physical output of source_geometry; its spatial_id is bound into the domain identity and cold validation. reference_domain remains a mathematical chart, never an inferred physical frame, and no default contract is supplied from the target mesh.

Exact piecewise-linear coverage also accepts 3-D PolyhedralConnectivity targets with a retained CellVertexGeometryElement (and affine standard-cell blocks): rational physical vertices, simple planar face loops, and signed cell-face incidence are replayed exactly, with nonconvex loops ear-triangulated by exact predicates. After complete oriented per-facet coverage and certified global embedding, the certificate retains one material-degree identity per region, bound to its source facets, target assignment, and embedding. Nonplanar, non-rational, singular, or untriangulable cases remain unresolved; volume agreement alone never supplies this proof.


The mapped-reference ancestry owner admits whole degree-one tetrahedron charts
(including the canonical simplex:3 alias), prism, pyramid, and hexahedron target
charts when exact Bernstein bounds prove the complete target image lies in one
unique affine reference-root chart. It retains exact per-cell
target-local-to-root-reference expressions, root global IDs, source occurrences,
and regions. Pyramid roots are accepted only when the represented chart proves
an exact affine inverse; non-affine collapsed roots are refused. Outside, crossing,
and shared-boundary-only
ownership is refused. Localization uses reference-domain coordinates and never
inverts the potentially curved or rational physical source map; consumers
compose the original physical coordinate expressions with this retained chart.

Rationally represented degree-one charts are accepted only when exact expression
equality proves they are the canonical reference-cell interpolation.
Per-cell ReferenceCellMapWitnesses retain rational-polynomial expressions
without corner-stencil reduction; PerCellRestrictedCellGeometryElement carries
the maps through exact source-basis composition and tabulation. Its constructor
proves rational denominator nonvanishing over each target cell but does not
establish source-cell inclusion or ancestry. Restriction/coverage consumers need
a separately validated ReferenceCellMapWitnesses bank; cold validation rechecks
exact corner agreement and whole-cell inclusion using Bernstein controls.

Pyramid chart normalization uses the canonical collapsed-cube parameterization.
It cancels only exact common powers of 1-w and returns the collapsed-apex trace;
a single apex value is supplied only when that trace is independent of the cube's
transverse coordinates. Callers must separately prove every remaining rational
denominator nonzero over the cube. Normalization prepares a proof expression and
does not rewrite persisted source data.

Exact source bases for `HybridReferenceFamily` pyramid elements are derived
from that family's rational modal space and exact nodal conversion. Admission
requires the bound family identity, reference-node bank, and modal ordering; a
matching node count alone is not treated as a basis definition.


For pyramid charts, H1 source-map validity proves a unique coordinate apex value
and positivity of the Jacobian determinant over every collapsed-cube approach;
it does not assert a unique apex derivative. `CellGeometrySpec.evaluate` and
`evaluate_exact` evaluate the unique value without requesting chart derivatives.
`tabulate` and exact JVP/gradient consumers refuse a pyramid apex whose derivative
continuation is nonunique. H1 validity therefore does not certify exact JVP
availability; native pyramid element apex conventions remain a separate contract.


```python
validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
coverage = phx.geometry.certify_domain_coverage(
    mesh, geometry, domain, cell_regions, embedding=embedding
)
```

`MeshingDomain` is admitted directly as the source authority for CAD volumes.
When its authored planar patch maps and trims have an exact affine realization,
coverage is checked against a chart-derived oriented source chain retaining the
original B-Rep domain identity and patch ancestry; a generated PLC is never
substituted for the B-Rep.
Curved exact-represented CAD routes require the revision-bound source query and
a certified whole-boundary fidelity report: the producer additionally checks
the complete source-chart partition, material sides, orientation, and boundary
homotopy. Distance alone is not a coverage proof. Toleranced CAD source
representations remain `unresolved` until a source-owned authenticated
incidence-ribbon proof to a closed representative is available; a scalar gap
allowance does not supply one.

Continuous source fidelity is separate from field residual. Its retained chart or
point witness binds the authored source IDs and occurrence paths to actual target
mesh entities and `CellGeometrySpec` maps, then is revalidated from those inputs
when a certificate is restored. Curve witnesses cover complete authored intervals;
surface and volume-boundary witnesses retain source chart ancestry and target edge
or facet maps; reconstructed and adaptive-implicit witnesses retain their original
boundary premises. A zero-dimensional feature uses a source-point-to-target-vertex
witness evaluated through every incident cell map, not an edge or face residual.

Piecewise-linear curve charts compose the exact retained source segment with the
actual target parent-reference edge map and bound the complete parameter
interval; nonzero deviation requires the caller's explicit tolerance.

For a source-authorized shared trim root, a chart junction may use a
deterministic representable reference point even when the two endpoint shadows
are different. Root identity and periodic-sheet agreement, not proximity,
authorize that join. Exact polygon area checks apply to the constructed reference
partition; they do not turn the representative into an exact source root.
Continuous coverage and physical fidelity include the root and endpoint
enclosures, so nonzero junction uncertainty cannot certify zero deviation.
For a conic revolution pole, the latitude is carried by its exact conic root;
the binary64 latitude is a reference realization, not proof of the pole.
Pole-cap bounds include both the latitude enclosure and the source-corner
realization error.


On standard volume targets, target-reference chart corners retain all three
parent-cell coordinates and `local_entity_indices` names the actual local face;
source UV charts remain two-dimensional. Validation maps the full parent-reference
triangle through that exact face basis rather than dropping a coordinate.
Variable-topology polyhedral charts retain `target_face_vertex_global_ids` for
actual parent facet triangles. `target_reference_vertices` may partition each
triangle virtually, allowing source-chart seams inside a face without adding
mesh vertices. Revalidation reproduces the actual `PolyhedralConnectivity` face
triangulation and checks exact `CellGeometrySpec` source-map coordinates without
inventing a standard volume-reference simplex for the polyhedron.

SourceBoundaryChartWitness retains canonical numerator/denominator banks for
the exact target-reference and source-parameter charts. Its float64 chart arrays
are checked rounded realizations only; exact partition, containment, and ancestry
checks use the rational banks, and witness identity plus cold restore retain them.
Float-only producers canonically bind the represented binary64 rationals.
The identity-bearing fields are target_reference_rational_tokens and
source_parameter_rational_tokens; cold archive replay retains both banks rather
than rebuilding them from rounded arrays.

`SourceBoundaryFaceImageWitness` instead represents exact physical overlap
pieces of a proved target-face image. `SourceBoundaryImagePullback` denotes the
inverse image of one physical polygon or segment through the complete bound
target map; it is not an affine reference triangle. Independent replay proves
the whole face image, source and target coverage, occurrence-qualified source
identity, and oriented source sides. Distinct authored strata remain separate
even when coplanar. Target material assignments require the current domain and
cell-region evidence in addition to the image witness.

::: phydrax.geometry.SourceBoundaryFaceImageWitness

::: phydrax.geometry.SourceBoundaryImagePullback

SurfaceChartWitness.root_mesh_cell_bindings stores one optional typed root binding per chart row, with root_geometry_id, root_topology_id, and the actual root_cell_global_id. A None entry is valid only for a chart with no root CellMesh lineage (for example, a direct analytic BRep chart without a root map); root restriction consumers require a non-None binding and validate it against the actual root owner. source_root_cell_ids and source_piece_ids remain source-atlas and extraction-piece identities, never substitutes for root mesh cell IDs.

Adaptive implicit P1–P3 boundary charts are certified only when source-authored smoothness, interval Hessian/gradient bounds, unique normal-line IFT roots, positive whole-chart Jacobian and injectivity, orientation, complete scaffold-triangle ancestry, and two-directed physical distance bounds all revalidate. Missing continuous root or distance premises remain unresolved; boundary samples alone do not imply coverage.

Each adaptive-implicit chart row uses the canonical source entity key
`source_entity_id(source_id, source_revision, "implicit-surface", "zero-set")`
for its complete zero-set scope. The witness keeps source ID and revision as
separate binding fields and uses empty occurrence paths; a raw source ID is not
a substitute for the entity key.

Implicit witness admission scans the complete retained root and triangle banks
in bounded worksets while retaining lookup entries only for requested chart
ancestry. Worksets do not limit the source or skip late duplicate keys.
Scaffold-carrier witnesses still require complete one-to-one triangle coverage;
remeshed carriers retain the separate source-coverage and boundary-degree proof.

An implicit wall or interface witness may cover every source scaffold
triangle while proving fidelity only on its actual target entities. The
target extent is rederived from retained entity IDs and cell/facet incidence;
it does not certify unrelated output-exterior facets. Whole-output-domain
coverage remains an independent premise.


MappedReferenceBoundarySource consumes a current native mapped-source authority,
including NativeMappedHexSource and NativePlcSource with its exact PLC-to-root
relation; root charts admit homogeneous tetrahedral, hexahedral, and prism
meshes, retaining triangular and quadrilateral prism faces in their actual
canonical face loops.
MappedReferenceDomain requires the physical output
`SpatialCoordinateContract`; mapped queries expose that exact domain-owned
object and bind its `spatial_id` into query identity. Reference-domain
coordinates remain mathematical chart coordinates, not the physical frame.
The query restricts the original `CellGeometrySpec` maps to declared root face
triangles and edge segments, with source IDs, occurrence paths, local root
entities, and orientation. Continuous source/target distance bounds use those
exact expressions and the target map; curved chart bounds refine until the explicit
tolerance is proved, violated, or binary64 parameter resolution is exhausted.
Mapped surface charts use interval Jacobian/cross bounds for curved oriented
normals, refining until the explicit limit is proved, violated, or numerical
parameter resolution is reached. A singular or unbounded range remains unresolved.

Its target extent is the retained set of actual target entity IDs and root
cell/local-face maps; even a complete source query may certify only a scoped
layer wall. Whole-output-boundary coverage remains a separate domain proof.

PiecewiseLinearBoundarySource retains a derived PWL owner and supports exact
vertex, edge, and facet source queries. Its certificates concern only the derived
PWL boundary; they do not infer original-to-envelope topological equivalence.
SurfaceEnvelopeDistanceWitness remains separate scoped metric evidence for the
original reconstructed source.

Profile-bound and envelope-bound distance witnesses instead bind the complete target
exterior-facet extent to a certified source cover at an explicit requested radius or
the kernel envelope operator. They report physical set-distance bounds only; they do
not imply source-domain coverage, boundary homotopy, or original-solid equivalence.
Source-cover requests use physical radii; realized sample counts are replay evidence,
not capacity limits that coarsen the continuous proof.
Distance bounds are two-directed and physical. The requested tolerance is explicit;
exact mode requires zero bound, and unsupported continuous bounds or normal proofs
remain `unresolved` rather than being inferred from samples or a field value.

Bounded domain certification retains both operands explicitly; it does not infer
coverage from a carrier source or a tolerance alone:

```python
coverage = phx.geometry.certify_domain_coverage(
    mesh,
    geometry,
    source_domain,
    cell_regions,
    embedding=embedding,
    source=source_boundary_query,
    fidelity=source_fidelity,
)
```

Bounded CAD volume coverage also proves chart continuity, not merely per-face
area and source IDs. Every target edge must map through its two incident face
charts to the same authored curve occurrence and parameter interval with opposite
material traversal; the complete per-region target/source vertex and edge maps,
closed oriented chains, and matching components/Euler characteristics establish
a recomputed oriented degree of +1 on each material-boundary component.
Discontinuous, ambiguous, or unsupported coedge maps remain unresolved rather than
receiving a degree fingerprint from local chart matches.

`AdaptiveImplicitSurface` volume coverage is bounded and source-specific. It
requires a rigorous interval cover and gradients, unique simple roots on retained
grid edges, regular cell/face strata, a complete root-triangle ancestry, and an
oriented match from the volume boundary to the original scaffold. Singular,
tangent, unresolved, or unmatched root cases remain `unresolved`; a valid carrier
or complete flag cannot create source coverage. For bounded coverage,
`source_distance_upper` records the revalidated two-directed source bound, while
unavailable exact source measures remain `None`; per-cell material assignments
are retained and checked independently.
AnalyticImplicitProfile volume coverage supports the established exact
sphere/ring-torus region through a RestrictedDelaunayBoundaryWitness that binds
the target mesh, analytic profile, source/domain, and separate source-distance
witness identities. The validator independently recomputes normal-gradient
transversality over the complete target boundary, verifies its closed embedded
surface structure, and checks projection degree one onto the source zero set.
The SourceBoundaryDistanceWitness remains a separate fidelity premise; closeness
below reach and a valid carrier cannot establish topology. Singular or tangent
faces, invalid boundary topology, or incomplete projection degree remain
explicitly unresolved or violated.

For this analytic route, the exact public arguments are:

```python
analytic_coverage = phx.geometry.certify_domain_coverage(
    mesh,
    geometry,
    profile,
    cell_regions,
    embedding=embedding,
    source=profile,
    fidelity=distance_fidelity,
    source_coverage_witness=restricted_witness,
)
```
The distance-only fidelity witness and restricted topology witness are retained
as separate premises and independently revalidated on cold restoration.

For a bounded homotopy certificate, `source_region_degree_ids` records the
recomputed oriented degree witness for each declared material region.

CoverageDomain is the canonical type union of PiecewiseLinearDomain,
MappedReferenceDomain, MeshingDomain, AdaptiveImplicitSurface, and
AnalyticImplicitProfile. The original exact piecewise-linear and mapped-reference
paths keep their exact boundary/chart proofs. coverage_method and
source_coverage_id distinguish those exact proofs from bounded source homotopies
in retained certificates.

Embedding and coverage constructors compute their findings from retained inputs;
they do not accept a caller's success flag as proof. Normal composition checks
input bindings without repeating the proof. Cold restoration recomputes and
checks the mathematical witness. Status is `certified`, `violated`, or
`unresolved`; unresolved evidence is not acceptance.

Composed coordinate tabulation keeps query points and coordinate arrays as
dynamic JAX values. Rational charts retain exact numerator/denominator
coefficient tuples as immutable preparation, rather than unhashable expression
objects. Their canonical storage and fingerprints are part of the current
representation; restoration must rebuild that representation rather than reuse
an obsolete template.

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
This admits a 256-by-1024 float64 pair table, the measured moderate-size
crossover that exceeded the earlier 512 KiB default. Full-fit sets execute the
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

`read_step`, `read_iges`, and `read_brep_text` decode native CAD carriers into
`BRepModel` using an explicit `CadImportPolicy`; native import does not require
OCP. The policy is source-only by default. Supplying a `BRepTessellationPolicy`
explicitly requests a derived triangle realization with physical deviation
bounds. A positive `maximum_placement_deviation` admits a rounded STEP pose only
when its per-occurrence rotation and translation errors, scaled by that local
solid's extent, fit the requested target-length bound. The resulting
`BRepAssemblyOccurrence.placement_evidence` retains the certificate. BRep-text
input also requires its declared source length unit.

In-memory `TopoDS_Shape` conversion is a separate optional operation through
`phydrax.interchange.import_occt_shape`; OCP is loaded only when
that adapter is called. The adapter serializes the exact shape as BRep text and
uses the native decoder, never an OCCT mesh as source geometry.

`BRepModel` keeps stable vertex/edge/wire/face/solid incidence, one parametric
surface patch per face, trim loops, optional tessellation-to-face identities, and
an import report. Supported analytic source surfaces remain analytic patches;
supported freeforms use rational tensor-product B-splines or validated local
T-spline patches.
`BSplineSurfacePatch` also accepts exact `exact_control_points`, `exact_weights`,
`exact_u_knots`, and `exact_v_knots` source banks. Control and weight leaves retain
`Fraction`/`RadicalElement` values and must match the numeric bank's binary64
realization; omitted exact banks lift authored leaves to exact dyadic values. Surface
parameter knot banks are `Fraction`-valued. Persistent `control_point_ids` bind
coefficient identity; source-root atlas construction refuses missing IDs rather than
using coefficient row positions as ancestry.
Source assemblies are stored once as a shared `BRepAssemblyGraph` of definitions,
uses, exact affine placements, and occurrence-bound placement evidence where a
source pose was rounded. `BRepGeometry.iter_occurrences()` yields leaf instances
lazily; source identity and queries do not cache an expanded instance model.
Occurrence-qualified CAD revisions enumerate their requested complete
inventories from that graph.
Native STEP, IGES, and BRep-text writers preserve supported exact carriers and
assembly sharing without loading OCP. Their reports declare external renumbering
of native assembly identifiers. A one-piece full-domain T-spline has an exact
rational B-spline realization; export proceeds only when source topology
tolerances are also certified. The report declares the omitted T-mesh refinement
and transfer structure. A native intersection branch remains exact by default;
an explicit `CadExportPolicy(intersection_approximation=CadCurveFitPolicy(...))`
may substitute a fitted rational B-spline only after continuous physical,
chart-parameter, surface-lift, branch-topology, and trim-separation bounds pass.
The report records that geometry-changing approximation. If exact T-spline
structure or fit evidence is unavailable, export refuses rather than assigning
zero tolerances or silently splitting a face.


Fixed-topology B-Rep nearest queries prepare one triangle hierarchy, refit it
to current realized vertices, and select one or two faces without a full
face-distance sort. The second selected distance supplies closest-point
uniqueness evidence, and projection derivatives remain local to those faces.
This adds prepared hierarchy leaves but preserves model/revision source IDs.
The differentiable `FixedTopologyBRepSource.contains` remains a numerical
solid-angle calculation on the current realized triangle surface. Source-exact
solid membership and projection use `prepare_brep_projection(model).locate_solids`
and its explicit inside, boundary, or unresolved outcomes.

`brep_normal_ranges(model, direction, draft_angle=..., angular_tolerance=...)`
returns a whole-trim angle enclosure in radians and an explicit face status.
Analytic and interval derivative enclosures are subdivided until the requested
angular enclosure is resolved or parameter resolution prevents refinement.
Meridian line/circle revolution faces use a normal that stays regular on the
axis. Cells with an undefined normal are refined until they are regular or
certified to contain a declared degenerate-edge pole or an exactly verified
isolated zero of `S_u x S_v` (`singular_cells`, `singularities`,
`singular_points`); cells undecided at binary64 resolution are returned as
`unresolved_cells` with typed `unresolved_reasons`.

Rational spline evaluation uses the shared span-local B-spline kernel. Each
curve query gathers `degree + 1` controls; each surface query gathers only the
tensor product of the active controls in its two parameter axes. Expanded
nonuniform and repeated OCCT knot vectors are preserved. At an exact chart
endpoint the final polynomial span supplies the one-sided differential, so
surface Jacobians and boundary frames remain finite instead of collapsing to a
constant endpoint branch.
The shared interpolation substrate also prepares `rational_bspline_curve_spans`
from the current control, weight and expanded-knot bank. Optional exact source banks
(`exact_control_points`, `exact_weights`, `exact_knots`) retain `Fraction` or
`RadicalElement` coefficients; each must agree with the corresponding binary64
runtime bank, while omitted exact banks lift the represented binary64 leaves to
their exact dyadic values. Source identities bind the retained exact banks. Each
actual nonempty knot span retains exact homogeneous Bernstein coefficients, a
positive denominator lower bound, and exact one-sided value/first/second-derivative
enclosures on any subinterval contained in that span. `rational_bspline_curve_continuity`
compares those endpoint jets exactly at adjacent spans and, when requested, the closed
profile seam; repeated-knot sides are never merged or inferred from sampling.

The same substrate exposes `rational_bspline_surface_spans` over tensor control,
weight, and expanded u/v knot banks. Optional exact
`exact_control_points`, `exact_weights`, `exact_u_knots`, and `exact_v_knots`
retain their source leaves and must match the supplied binary64 banks; omitted
exact banks lift the numeric leaves to dyadic values. Each span's
`enclose(((u0, u1), (v0, v1)))` restricts its exact homogeneous Bernstein net
to one parameter sub-box and returns rational outward `value_bounds`, `du_bounds`,
`dv_bounds`, and positive `denominator_bounds`. The derivative bounds are the two
ambient-coordinate columns of the surface Jacobian; they are exact Bernstein
enclosures, not sampled estimates or budget-limited subdivision results.
`BSplineSurfacePatch.derivative_bounds(parameter_box, /, *, order=1)` is the CAD
facade: it intersects the query box with each knot span and outwardly unions the
per-span Jacobian intervals. Only first-derivative bounds are admitted.

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from phydrax.geometry.brep import BRepTessellationPolicy
from phydrax.interchange import import_occt_shape

coordinate_contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
imported = import_occt_shape(
    BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(),
    coordinate_contract=coordinate_contract,
    tessellation=BRepTessellationPolicy(
        linear_deflection=0.1,
        angular_deflection=0.1,
    ),
)
print(imported.report.losses)
source = phx.geometry.BRepSource(imported.model)
geometry = source.compile()
print(source.model.report)
```

`import_occt_shape` is an explicit optional OCCT adapter. It routes source
carriers through the native BRep-text codec and returns the codec's
`CadImportResult`: the model together with its adapter report, so dropped
derived records, synthesized natural boundaries, chart-closure repairs and
source tolerances reach the caller. `persist_occt_shape` returns the same
result for the published carrier. OCP is loaded only when the adapter is
called. Native STEP, IGES, and BRep-text file readers do not require OCP.

`FixedTopologyBRepSource` reevaluates an imported tessellation from trainable
patch parameters while preserving face topology, boundary entity identity, knot
vectors, degrees, and knot-span topology. Rational B-spline control points and
weights remain differentiable; moving knots across a query is intentionally not
part of this fixed-topology contract. A realization exposes the current
vertices, faces, atlas, and a differentiable seam residual. The validity region
requires unchanged topology, positive surface Jacobians, and compatible seams;
`BRepSeamCompatibility` makes the last condition an explicit design constraint.

`prepare_brep_projection(model)` prepares source-native queries against the
model's physical source revision. It does not require a tessellation, an OCCT
shape, or a live external engine. The returned `NativeBRepProjection` implements
`AbstractBRepProjection`; consumers use that engine-neutral contract.

`project` accepts explicit entity dimensions, indices, and optional occurrence
paths. Results retain source entity indices, source parameters, distance bounds,
normals, tangent frames, and occurrence identity. `classify` selects the
lowest-dimensional source entity within the requested tolerance;
`locate_solids` reports containing solids. Planar faces and affine extrusions
with line trims include closest points on their trim boundaries, including when
the untrimmed plane projection lies outside the parameter bounds.

Analytic cone, cylinder, sphere, and torus frames use bounded query batches;
occurrence transforms and singular/non-isolated statuses remain per query.
For a torus whose trim classification is undecided at a periodic seam, an
actual coedge point can establish a `SEAM` result: its exact curve-to-surface
lift must be proved, and its distance upper bound must lie within the declared
`ambiguity_tolerance` of the whole-carrier minimum lower bound. The result
retains both bounds. This does not snap a UV parameter, infer a complete period,
or repair an imported source gap; missing lift or distance evidence remains
unresolved.

`BRepProjectionStatus` distinguishes unique points, seams, ambiguous choices,
non-isolated minima, failed queries, and unresolved source mathematics.
Unsupported trim or carrier queries remain `UNRESOLVED`; they do not substitute
a tessellation or invoke an optional CAD engine. `PlanarEmbedding` explicitly
maps supported planar source queries to two-dimensional coordinates.

### Native source operations

`boolean_brep` computes regularized union, intersection, and difference from
native source carriers in memory. It retains exact source correspondence and
oriented lineage, publishes genuinely empty results for empty regularized
regions, and reports unresolved branch or topology proofs explicitly. Export
and archive writing are separate operations; exact output does not fall back to
fitted splines, triangle meshes, or an optional CAD engine.

`OffsetSurface` represents an exact analytic normal-offset operation tree;
`brep_offset` publishes supported source families without requiring a tessellation.
The admitted offset, blend (`brep_fillet`/`brep_chamfer`), and defeature
(`brep_defeature`) families are explicit and bounded by their native
certificates. They do not claim general rolling-ball blends or arbitrary
corner-network coverage. `sew_brep` consumes explicit source topology and
oriented lineage rather than fuzzy coordinate-based healing.

Source fidelity retains exact affine OffsetCurve coefficients, including
non-binary distances, and composes exact BRepAffinePlacement transforms for
affine PlacedCurve rails. RuledSurface values with affine rails yield exact
bilinear maps; affine OffsetSurface stacks are admitted when the exact normal
has a rational squared norm and its RadicalElement root is supported. These maps
are bounded continuously against target coordinate maps; unsupported exact-backed
carriers remain unresolved rather than using rounded dynamic values as source
authority. Approximate acceptance still needs an explicit tolerance.

`FixedTopologyBRepSource` reevaluates source patches and any fixed tessellation
parameters while preserving face topology, boundary entity identity, knot
vectors, degrees, and knot-span topology. Rational B-spline control points and
weights remain differentiable; moving knots across a query is intentionally not
part of this fixed-topology contract. A `BRepParameterGeometry` realization
keeps `source_geometry` and `source_patches` as the exact source snapshot while
exposing current numeric vertices, curves, p-curves, trims, parameter bounds,
and optional triangle parameters. `source_trim_domains` remains the exact
trim/root snapshot; runtime trims are rebuilt from current coedge p-curves and
edge intervals. A numeric realization never replaces exact roots or becomes a
new host CAD revision. It exposes the current
vertices, faces, atlas, and a differentiable seam residual. The validity region
requires unchanged topology, positive surface Jacobians, and compatible seams;
`BRepSeamCompatibility` makes the last condition an explicit design constraint.


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
For an authored recipe name, `RecipeEntityNames.select_occurrence` requires an
explicitly qualified B-Rep entity and the exact CAD occurrence inventory.
`AssociationGraph.resolve_target` preserves single-entity ambiguity; its
`resolve_target_selection` transports a declared descendant set and refuses
deletion, unresolved history, partial merges, and incompatible orientations.
Partitions expose operand-to-composite graphs before the operation graph, so
selections can be composed without identifying entities by their shape. A
same-kind descendant set does not name individual symmetric pieces; a unique
piece still requires independently proved construction or incidence roles.

## B-Rep and planar partitions

`partition_brep` accepts a `BRepPartitionPlan` with an explicit coordinate
contract, ordered `BRepPartitionOperand` values, roles, void targets, and
`BRepPartitionPolicy` precedence. Its in-memory `BRepPartitionResult` exposes
named solid regions and face patches as exact `BRepEntityId` selections,
revision-to-revision association evidence, and deleted/created occurrence
history. It does not heal or rewrite the source topology by coordinate proximity.

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

## Sketches and geometric constraints

`Sketch` solves fixed-connectivity 2D line/circle systems with declarative
constraints such as `Coincident`, `Horizontal`, `EqualLength`, `Radius`, and
tangency. A solved sketch lowers to `PlanarMeshRegion`.

`DesignConstraintSystem` solves geometry-state constraints without coupling to a
particular representation. Available constraints include parameter targets and
equalities, point distances, interior/exterior clearance, measure and boundary
measure targets, boundary-point conditions, and B-Rep seam compatibility.

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

geometry = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
).compile()
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

`ImplicitPointProjectionPlan` supplies fixed-shape normal-gauge boundary motion.
`discover_implicit_curve` discovers an oriented closed planar segment topology.
`discover_implicit_surface` creates a host-side `ImplicitSurfacePlan` whose JAX
runtime preserves triangle connectivity and reports sign, root, QEF, orientation,
and intersection evidence.

`discover_implicit_surface` remains the fixed-grid, fixed-topology realization.
The separate `discover_adaptive_implicit_surface` traverses a three-dimensional
domain by field enclosures, refining every possible-zero box to the policy's
physical cell-diameter resolution. It has no leaf, query, or depth budget.
`AdaptiveImplicitSurface.cover` retains the exhaustive enclosure; root-witness
boxes support `AdaptiveImplicitBoundarySource` distance and sample queries.
`AdaptiveImplicitBoundarySource.project_boundary` requires the exact
`SpatialCoordinateContract.spatial_id`. Its global result remains `UNRESOLVED`
unless a unique global closest point is proved: nearest retained root witnesses
are only candidate upper bounds, not closest-point certificates. Native P1–P3
source charts use a separate interval-isolated local normal-line root map tied
to the exact scaffold cell/root-edge ancestry; this local map is not a global
projection. Fidelity and coverage replay the full target coordinate maps,
interval field/gradient/Hessian bounds, shared-edge map identities, oriented
Jacobian/injectivity premises, and the complete source cover.

Unresolved singular, tangential, domain-boundary, or representability cases
remain explicit in `evidence`; `"sampled"` discovery never promotes samples to
an enclosure claim.

`AnalyticImplicitProfile` is a distinct restricted-source premise for nominal
three-dimensional spheres and full nonsingular ring tori. It establishes their
reach, topology, and certified boundary-distance/cover facts; those facts are
not inferred for arbitrary implicit fields. Native restricted-Delaunay and
general adaptive implicit tetrahedral routes therefore remain separate.


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
pins the identical version, and a library that is another release, lacks any
bound C ABI symbol, or returns a null/malformed release or build identity is
reported as `MeshcoreUnavailableError` with the reason. No C++ exception crosses
its C ABI: a refused allocation is the call status `CAPACITY_EXCEEDED`.
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

`EulerianVolumeOverlapMap` and `EulerianVolumeOverlapChain` retain the
label-gated **physical-overlap** route through
`prepare_eulerian_volume_overlap`. For curved Eulerian intersections, a
`CommonRootVolumeWitness` requires each endpoint map to be the witnessed
restriction of the same root physical geometry; exact source and target
coverage, root-map positivity, and global injectivity are revalidated. The
overlap is measured with that one physical root Jacobian. Rational integration
retains outward Bernstein error enclosures and returns `PREDICATE_UNCERTAIN`
with the exact unresolved reference tetrahedron when binary64 subdivision can
no longer make progress. This is not material-following transfer: equal labels
do not make differently placed endpoint volumes overlap.

Lagrangian transfer is a separate owner, `VolumeMaterialMap`, created by
`prepare_volume_material_map` from a `VolumeMaterialDeformation`. Its common
reference is an explicit `MaterialReferenceDeclaration`, authored by
`declare_material_reference(..., policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION, ...)`
from a named accepted configuration and frame. The declaration retains the
root mesh/geometry, source identity, cell entity/region/occurrence rows, and
accepted root coverage. It is never inferred from endpoint similarity, labels,
corner coincidence, or Eulerian overlap history. Each deformation binds exact
source- and target-cell reference charts into that declared root and a current
`DomainCoverageCertificate` for each endpoint. The distinct
`MaterialReferenceVolumeWitness` certifies the root and both independent
endpoint coordinate maps, including positive Jacobians and embedding; nonlinear
chart images are intersected only after their whole images are certified.
Curved non-polyhedral material images retain exact matching-image and certified
vertical-partition fast paths. Other certified injective charts use paired
reference-cell subdivision: Bernstein image enclosures exclude disjoint cells,
and a parametric Krawczyk inclusion proves whole-cell containment through the
target inverse. Unresolved boundary leaves carry a rigorous integral enclosure;
the route returns `PREDICATE_UNCERTAIN` when the requested tolerance is not
decidable at finite binary64 reference resolution. Image coverage diagnostics
retain exact zero defects for exact topological proofs and an
`interval-certified` marker when coverage is established by integration bounds.
No curved interior is replaced by a corner tetrahedron.

`VolumeMaterialMap.material_overlap` retains target-major sparse common-reference
entries with a root-reference measure and independently integrated source and
target physical measures/Jacobians plus error evidence. Source and target
physical domains may differ under motion; each must nevertheless be fully
covered on its own, and the material reference charts must correspond exactly.
Material-side/entity/region/occurrence and frame identities are bound in the
deformation and map identities. Gaps, positive-volume double coverage,
inverted or singular maps, stale coverage, unmatched material sides, and
foreign roots refuse; no missing measure is normalized away.

Coverage ancestry and the authored material reference are distinct. When a
mapped endpoint's coordinate bank does not retain its intermediate parent,
pass the actual `(parent_mesh, parent_geometry)` as `chart_parent` to
`declare_material_reference`, `MaterialReferenceDeclaration.revalidate`, or
`material_source_map_from_cell_association`. Deformation and overlap preparation
and revalidation accept `source_chart_parent` and `target_chart_parent`; absent
explicit inputs, they use their existing material-reference or common-root
owners. Those owners must be the immediate parent required by the coverage
proof, not merely a compatible material reference. `VolumeMaterialMap.revalidate`
also accepts `root_chart_parent` for revalidating the reference declaration.
These are operation-local proof inputs, not retained history graphs. The
coverage owner independently proves both chart partitions and the complete
physical maps; missing or foreign parents refuse.

The map's field actions deliberately keep their measures distinct:
`apply_material_extensive` partitions cell-integrated content by common
reference-volume fraction; `apply_material_density` integrates the source
density with the source physical Jacobian and divides by the target physical
cell measure; `apply_material_scalar` composes a cellwise material scalar using
reference-volume averaging. Each has a sparse value-JVP and Euclidean
transpose for fixed geometry and overlap. `VolumeMaterialChain` composes those
actions chronologically with no gap renormalization. These cell-content/P0
actions are not finite-element pullbacks: declared H1/DG/H(curl)/H(div) spaces
continue through their exact reference-map, de Rham, and Piola transfer owners.

Label-free Eulerian callers can use `prepare_common_root_volume_refinement`
directly and wrap its returned CSR in `PhysicalOverlapTransfer`. Its pure
actions are `apply_physical_overlap_extensive` and
`apply_physical_overlap_intensive`, with matching Euclidean transpose functions;
`apply_physical_overlap_extensive_with_evidence` also returns actual content,
conservation defect, and per-cell coverage defects. They use physical source and
target measures as their respective denominators and do not require material
labels.

`EulerianVolumeOverlapMap.apply_extensive` distributes content by physical
overlap/source measure; `apply_intensive` averages by physical overlap/target
measure. `EulerianVolumeOverlapChain` composes those Eulerian maps. Neither
operation substitutes for material-following transfer under moving geometry.

`phydrax.meshing._association_transfer.prepare_eulerian_material_source_association_transfer`
consumes an `EulerianVolumeOverlapMap` and exposes its physical overlap as a
relation. Each source side retains its global cell ID, source entity, material
region, and occurrence path with its unnormalized physical overlap measure; each
target side retains its own identity and physical-measure denominator.

Before building a new material map after adaptation,
`propagate_material_association_rows` binds target cell rows by global ID through
explicit three-dimensional `MeshLineage`. Every contributor must agree on the
source entity, material region, and occurrence path, and the result must match
the target's certified region rows; created, unknown, or conflicting cells are
refused rather than assigned by geometric proximity or vertex stencils.


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

---

::: phydrax.geometry.PlanarPartitionPlan

---

::: phydrax.geometry.ProcessStack

---

::: phydrax.geometry.ProcessStackResult

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
