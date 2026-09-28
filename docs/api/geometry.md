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

`BRep(path, coordinate_contract=...)` imports STEP, IGES, and BREP files through
OCCT. `import_brep` and `persist_occt_shape` likewise require an explicit
coordinate contract.
`BRepModel` keeps stable vertex/edge/wire/face/solid incidence, one parametric
surface patch per face, trim loops, tessellation-to-face identities, and an import
report. Supported analytic OCCT surfaces remain analytic patches; other faces are
represented by rational tensor-product B-splines.

Rational spline evaluation uses the shared span-local B-spline kernel. Each
curve query gathers `degree + 1` controls; each surface query gathers only the
tensor product of the active controls in its two parameter axes. Expanded
nonuniform and repeated OCCT knot vectors are preserved. At an exact chart
endpoint the final polynomial span supplies the one-sided differential, so
surface Jacobians and boundary frames remain finite instead of collapsing to a
constant endpoint branch.

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

coordinate_contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
model = phx.geometry.model_from_occt_shape(
    BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(),
    coordinate_contract=coordinate_contract,
    linear_deflection=0.1,
)
source = phx.geometry.BRepSource(model)
geometry = source.compile()
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

`prepare_brep_projection(model, source)` binds OCCT closest-point queries to the
exact revision of a `BRepModel`: `source` is the in-memory shape the model was
extracted from or its persisted CAD file, and its digest must equal
`model.source_digest` (the in-memory digest excludes cached tessellations). The
`PreparedBRepProjection` projects batches onto explicit vertices, edges, and
faces (`project`), reports `(u, v)` or `t` parameters, residuals, oriented face
normals and tangent frames, and a `BRepProjectionStatus`: `SEAM` for periodic
seams and singular poles, `AMBIGUOUS` for distinct tied minima or continua such
as a sphere center, `FAILED` when no closest point exists. Trimmed faces are
honored through the OCCT face classifier. `classify` returns the
lowest-dimensional entity within a tolerance (BVH candidates, optional admissible
entity groups), `locate_solids` the containing solid, and `contains`,
`containers`, and `members` expose the closure relation. A `PlanarEmbedding`
binds a face-only revision to two-dimensional coordinates. Each query is one host
call into OCCT extrema (the external-provider boundary); calls are grouped per
entity so projectors and classifiers are prepared once.

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

## B-Rep and planar partitions

`partition_brep` accepts a `BRepPartitionPlan` with an explicit coordinate
contract, ordered `BRepPartitionOperand` values, roles, void targets, and
`BRepPartitionPolicy` precedence. Its persisted `BRepPartitionResult` exposes
named solid regions and face patches as exact `BRepEntityId` selections,
revision-to-revision association evidence, and deleted/created occurrence
history. It does not offer general Boolean repair.

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
`TriangulationEvidence` (route, native status, library identity, counts).

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
