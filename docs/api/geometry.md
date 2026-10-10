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

Native CAD construction and file decoding produce `BRepModel`, whose exact
carriers, oriented topology, p-curves, occurrences and coordinate contract own
geometry identity. Derived query tessellations do not replace that authority.
`phx.interchange.read_cad(path, policy, trusted_root=...)` selects the native
STEP, IGES or external OCCT BRep text codec. The format-specific `read_step`,
`read_iges` and `read_brep_text` preserve their own coverage and refusal reports.
STEP and IGES own embedded units; BRep text requires `source_length_unit`.
The native decoders implement these file formats without invoking OCP.

These native carriers, codecs, queries, bounded intersections, sewing, Boolean,
and tessellation routes are implemented surfaces. Source-only
`BRepTessellationPolicy(realize=False)`, spline-preserving loft, topology/volume,
cavity/shared-shell, unattached-edge, periodic-rational, surface-array, and
scaled-oblique checks passed their targeted matrix. Overlapping-sphere
partitions retain the exact rational chart coordinate of every collapsed-pole
radial split; the rounded binary64 vertex is only an execution representative
bound to that authority. Curved-void trim ribbons are bounded per source-curve
interval, and an interval whose ribbon exceeds the requested deviation is
bisected locally before publication. Native BRep text refuses implicit curved
intersection branches instead of approximating them. This is not complete
native CAD qualification: complete STEP/IGES entity coverage, every
intersection curve, singular arrangements, the independent writer campaign,
W15, and release remain separate gates.

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

The `native-reconstruction-lifecycle` qualification scenario retains the actual
noisy point-cloud digest, native screened-Poisson report, and extracted triangle
coordinates in the revision of its `NativePlcSource`. Native constrained volume
fill must certify that exact represented PLC, independently close its signed
boundary volume, and retain the original region identity. Reconstruction's
sampled distances remain separate from exact extracted-domain coverage; an
analytic reference surface never replaces the reconstructed triangles.
The scalar diffusion continuation uses `FiniteElementTopologyTransaction` and
`CompositionRebind`, including a rejected physical-error gate that must preserve
the accepted mesh and state. The repaired-envelope scenario uses the same atomic
solver-state transition while retaining its raw/repaired identities, explicit
feature/topology permissions, two directed bounds, and inclusion margin.
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

The fixed lifecycle controls are `--resolution 4 --capacity 20000 --timeout 120
--repeats 3 --target-error 0.05 --adaptation-rounds 1`. These are acceptance gates,
not a claim that either workflow has qualified. Native circumcenter failure is
reported as structured meshing failure before publication, rather than replacing
nonfinite quality evidence with a finite value. Envelope carrier initialization
also enforces the request's actual native scratch-allocation allowance.
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
        - DomainCoverageCertificate
        - FieldBoundOrigin
        - GlobalEmbeddingCertificate
        - ImplicitBoundOrigin
        - ImplicitBoundarySource
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
