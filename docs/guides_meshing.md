# Meshing

`phydrax.meshing` owns mesh construction, adaptation, certification, provider
execution, and topology transitions. Solvers consume the resulting native
carriers; they do not own external meshing sessions.

## Workflows

Each script is a complete, bounded public-API workflow that checks its own
evidence and prints a JSON summary:

- [`examples/meshing_native.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/meshing_native.py):
  certification, bisection, exact linear field transfer, and a finite-element
  solve on the refined carrier.
- [`examples/adaptive_bisection_heat.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/adaptive_bisection_heat.py):
  adaptive 2D and 3D heat solves with Zienkiewicz-Zhu recovery, Dörfler marking,
  and `NATIVE_BISECTION` refinement.
- [`examples/adaptive_device_simplex.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/adaptive_device_simplex.py):
  solve, mark, refine, and coarsen on the device inside one capacity bucket,
  then commit one certified mesh.
- [`examples/anisotropic_metric_adaptation.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/anisotropic_metric_adaptation.py):
  recovered Hessian, `L^p` metric, `NATIVE_METRIC_2D` adaptation, and re-solve.
- [`examples/ale_conservative_remesh.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/ale_conservative_remesh.py):
  mesh motion assessed by `MeshMotionMonitor`, remeshing on its decision, and a
  conservative second-order remap through the common refinement.
- [`examples/cad_high_order_curving.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/cad_high_order_curving.py):
  OCCT cylinder and sphere, Gmsh tetrahedra, B-Rep association, P2/P3 curving,
  and the Bernstein validity certificate.
- [`examples/boundary_layer_core_mesh.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/boundary_layer_core_mesh.py):
  native advancing boundary layers, a Gmsh core fill, and the merged-mesh audit.
- [`examples/delaunay_voronoi.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/delaunay_voronoi.py):
  exact meshcore Delaunay, constrained Delaunay, Voronoi, and power diagrams.
- [`examples/meshing_omega_h.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/meshing_omega_h.py):
  serial or MPI metric adaptation through the persistent Omega_h worker.

The code blocks of this guide form one sequence: later blocks reuse the
`result` certified in the first.

## Certify an existing mesh

```python
import numpy as np
import phydrax as phx

mesh = phx.discretization.CellMesh.from_triangles(
    np.asarray(((0., 0.), (1., 0.), (0., 1.))),
    np.asarray(((0, 1, 2),), dtype=np.int32),
)
result = phx.meshing.certify_cell_mesh(
    mesh, phx.SpatialCoordinateContract.si(),
)
assert result.audit.passed
```

A `CellMeshingResult` contains the canonical `CellMesh`, independent
`CellGeometrySpec`, physical coordinate contract, quality and audit reports,
compliance, staged trace, provider/runtime identity, and derivative mode.
A failed audit or compliance check is not a successful result.
An optional `surface_source` retains the authoritative boundary chart source,
not a replacement tessellation. `region_boundary_evidence` binds oriented source
region sides to the result's actual labels and patches. An explicit
`planar_embedding` preserves the map from intrinsic two-dimensional coordinates
to the source's world frame. These scientific fields contribute to `result_id`;
runtime provider identity and trace timing do not. Retaining a source chart is
not, by itself, proof of target-to-source fidelity.


`CellGeometrySpec` belongs to `phydrax.discretization`, not FEM. Its coordinate
nodes and element routes can represent geometry of a different polynomial order
from a solver's unknown field. Corner topology remains separate from curved
geometry nodes.

`CellGeometryTransition.reconstruct_surface` handles a nonnested 2D surface edit
when every target cell has an explicit exact reference chart into the source
coordinate geometry. It samples that map at the requested target coordinate
element nodes, then retains endpoint-bound outward displacement and shared-edge
continuity bounds. Corner-only chart inference is unresolved; a reconstruction
whose certified bounds exceed either request is refused. Exact nested transitions
continue to use their existing coordinate-map equality witnesses.

`evaluate_cell_quality` samples differentiable corner quality per cell family:
signed measure, mean ratio, radius ratio, edge aspect ratio, scaled Jacobian,
corner or dihedral angles, condition number, tetrahedral sliver measure, face
warpage, and metric-space quality when a `MeshMetricField` is supplied.
`sampled_valid` is corner evidence, not a validity proof. Audit
`quality_scope` states whether those samples describe the full geometry
(`vertex_geometry`) or only the straight corner cell (`corner_cells`).

Geometric validity is certified by
`phydrax.discretization.certify_cell_geometry_validity`. Polynomial and rational
coordinate maps use whole-cell Bernstein bounds and adaptive subdivision;
variable-topology cells use their native polygon or polyhedral proofs. A map is
`CERTIFIED_VALID` only when its required strict margins are proved, `INVALID`
when an inadmissible value is proved, and otherwise `UNRESOLVED` when the
represented numerical resolution or a missing mathematical premise prevents
proof. There is no caller work/depth budget that turns an unfinished proof into
success. Positive corner samples cannot certify an inverted interior. A
polyhedron that is not star-shaped about its selected center remains
`UNRESOLVED`; that is not a validity disproof.

Native interval/simplex field views compile complete support worksets from the
source boundary without caller-selected face-entry or byte allowances. Structural
and index representability and real runtime allocation limits still apply.

For source-backed coordinates, `CellGeometrySpec` retains a canonical immutable
exact coefficient snapshot and validates its binary64 execution shadows.
Validity certifies the exact scientific map and the execution coefficient map
independently. The combined `status` is invalid if either proof is invalid,
unresolved if either remains unresolved, and valid only when both are valid.
`source_status`, `source_determinant_lower`, `source_determinant_upper`, and
`source_depth` retain the exact-source result. The reported
`maximum_coefficient_rounding_error` is scoped to the coordinate bank and has
physical-coordinate units. `maximum_element_coefficient_rounding_error` reports
dimensionless element-basis coefficient rounding separately; neither is a
whole-map or Jacobian error bound. Source snapshots, both proofs, and both
rounding measures participate in identity and are regenerated by the owning
cold-restoration methods.

An exact `CellGeometryCoordinateMap.source_coordinates()` bank is aligned with
`CellGeometrySpec.coordinates` and its element routes, not implicitly with the
target mesh's vertex table. Restricted maps may retain parent coefficients and
bind exact target-vertex images separately; those images are checked through
the retained maps rather than inferred from rounded coordinate equality.

Polygon cells are certified without assuming star-shapedness. A planar polygon
is `CERTIFIED_VALID` iff its vertices are finite and distinct, every edge clears
`relative_determinant_floor` times the polygon diameter, its boundary is simple
(`polygon_simplicity_2d`: non-adjacent edges are disjoint and adjacent edges meet
only at their shared vertex), it is counterclockwise, and twice its signed area
exceeds the policy floor. An embedded polygon must instead lie within
`CellValidityPolicy.relative_planarity_tolerance` of its Newell plane (relative
to the polygon diameter), be simple in its dominant-axis projection, and have a
squared vector area above the floor. Repeated vertices, self-intersecting
boundaries, clockwise planar loops, and nonplanar embedded loops are `INVALID`;
an undecided simplicity, orientation, planarity, or area decision leaves the cell
`UNRESOLVED`, never valid. The segment and polygon predicates are documented
with the [exact predicates](api/geometry.md#exact-predicates-triangulations-and-diagrams).

`CellMeshAuditPolicy` assigns each audit check a `CellMeshAuditDisposition`
(`REJECT`, `RECORD`, or `SKIP`). Topology checks run on the welded complex in
which vertices within `coincident_vertex_tolerance` of each other (relative to
the bounding-box diagonal, found through a Morton-sorted grid) are merged:
coincident vertices, collapsed and duplicate cells, non-manifold facets, edges,
and vertices (disconnected cell stars), inconsistent facet orientation,
watertight boundary, and self-intersection of the cells of planar and surface
meshes or the welded boundary of volume meshes, through BVH candidate pairs and
exact-or-filtered orientation predicates. By default every check is `REJECT`
except `watertight_boundary`, which is `SKIP`.

Each report partitions its dispositioned checks into `evaluated_checks` and
`skipped_checks`, by finding name in check order (`unused_vertices`,
`unused_geometry_nodes`, `coincident_vertices`, `collapsed_cells`,
`duplicate_cells`, `nonmanifold_facets`, `nonmanifold_edges`,
`nonmanifold_vertices`, `inconsistent_orientation`, `open_boundary`,
`self_intersection`, `invalid_geometry`); a default report skips only
`open_boundary`. `check_counts` gives the finding count of every evaluated check
and every `unresolved_*` entry. `audit.passed` is `True` iff `issues` is empty:
no evaluated `REJECT` check found anything, the geometry, quality, and evidence
bindings held, the quality thresholds and association requirement were met, and
no rejected check was unresolved. It certifies nothing about a skipped check,
and `RECORD` findings appear in `recorded` without failing the audit.

Uncertain predicates are never reported as a pass: self-intersection pairs whose
predicates stay uncertain name `self_intersection_predicates` in `unresolved`,
as do undecided coordinate-map validity certificates (`geometry_validity`).
`CellMeshAuditPolicy.unresolved` rejects or records these findings; it cannot
skip them. Native candidate traversal is complete and streamed, without
caller-supplied connectivity or intersection-count ceilings. Actual allocation
failure and numerical inability to decide a predicate remain distinct failures.
Supplying a `CellGeometrySpec` before a mesh reorder is refused; canonicalize
the mesh before constructing row-bound geometry. Periodic quotient topology is
rebuilt by stable lifted entity IDs while retaining source vertex representative
IDs, periodic lattice, and winding across canonical block/cell ordering.

Local audit success does not prove coverage of an authored source domain.
`certify_cell_mesh(..., certification_inputs=...)` additionally computes a
`MeshCertificationReport` from the retained source, coordinate map, material
assignments, and source-bound witnesses. The result retains it as
`result.certification`; this field is `None` for audit-only certification.
The report composes global embedding, source coverage, continuous fidelity,
and applicable feature obligations. Unsupported proofs remain unresolved, not
successful. `report.revalidate(mesh, geometry)` independently recomputes the
proofs; stored IDs or a passed flag are not substitutes for source operands.
The global-embedding theorem is computed once per mesh, coordinate map and
validity policy and handed on explicitly: a route that already proved it
passes the certificate in its certification request (or
`certify_cell_mesh(..., embedding=...)`), and `revalidate(..., embedding=...)`
accepts an in-hand certificate. Each consumer, including restricted-Delaunay
coverage replay, reuses it only after checking that it binds the same mesh,
coordinate map, local-validity proof and junction set; there is no process
cache.
A restore of a stored report first calls `report.require_premises(inputs,
audit_policy=..., organization=...)` with the rebuilt premises. It compares only
claims fixed by content identity (request, mesh and coordinate-map bindings,
junctions, source revisions, coverage region IDs and occurrence paths, material
assignment, organization IDs, report identity), so a tampered archive is refused
before any proof replay. It accepts nothing: the embedding, coverage, fidelity
and audit proofs are then recomputed once from the same premises, because a
stored claim is not determined by the identities it binds.
Native model archives store per-entity static rows of geometry associations,
embedding witnesses and surface cell tags as array columns (a value table with
int64 row indices, or UTF-8 text with offsets), so manifests stay within the
canonical `MODEL_ARCHIVE_LIMITS`; the decoder rebuilds each record through its
constructor and requires its stored identity.
Missing continuous fidelity witnesses produce an unresolved certificate with
unbounded directed deviation evidence. Those bounds remain infinity; they are
not replaced with finite estimates or accepted merely because mesh samples agree.

Distance-only analytic or envelope witnesses do not establish source topology.
An explicitly changed envelope has its own source domain; its original-source
deviation and its derived-domain coverage are separate obligations.

`evaluate_finite_volume_quality` reports skewness and non-orthogonality of every
interior face of a full-dimensional mesh.

## Scopes and organization

A `MeshingScope` names a source, its exact revision, entity kind and degree,
authoritative entity set, and persistent IDs. Boolean scope operations require
matching identity domains. Stale revisions are errors, not empty selections.

`MeshingScope.whole(source, dimension)` includes all declared source occurrences.
For CAD, `from_geometry` resolves a `BRepEntityId` through the actual source
owner, revision, definition, and occurrence. A selector without an occurrence
path is accepted only when its definition has one physical use; reused
definitions require an explicit path. Numeric scope IDs are canonical source
bank IDs, not B-Rep definition indices or provider tags. Accepted label-field
extraction results use the same source projection for scope creation and native
material meshing.

Native BRep-to-`MeshingDomain` conversion retains periodic coedges' native
integer-period gauge and continuous UV loop and pole data. The supplied BRep
authority is rechecked against exact corner incidence, occurrence, patch, and
p-curve-lift evidence. Floating trigonometric endpoint roundoff is not a
geometric gap, and a tolerance cannot authorize replacing source BRep corners
or trims. A placed-domain `authority_id` binds `model.model_id`.
The placed binary64 query linear map must also remain nonsingular and
orientation-preserving; an exact nonsingular placement is refused when its
query realization is singular or reverses orientation.

Exact `MeshingDomain` BRep authority requires a retained `BRepModel`;
`MeshingDomain.from_brep` retains that model. Bare
`MeshingDomain.from_brep_geometry` input without its owning model is an unbound,
generic conversion that checks endpoints only and confers no exact source authority.

`MeshingDomain.stratum_entity_ids(dimension)` returns every canonical source key
of one stratum in slot order, the same keys `entity_id(dimension, slot)` returns
one at a time, and `resolve_entity_id` bisects a per-stratum key index instead of
fingerprinting every slot. Per-entity loops over a domain with one patch per source
face therefore stay linear. Scope-bank numeric IDs and the `domain_id` are unchanged.
Method access on a domain or any other source module costs the same whatever its
size: every `StrictModule` binds methods as Equinox bound-method pytrees without
flattening the receiver.

BRep material regions bind to their source solid occurrences, and each region's
full oriented face/sign boundary must match the supplied topology. Missing
solids or faces and reversed boundary signs are refused.

- `MeshPatch`: geometric boundary support.
- `MeshZone`: exclusive solver/material organization.
- `MeshLabel`: overlapping semantic grouping.
- `MeshAttribute`: entity-associated numerical or categorical data.

When a `MeshPatch` records source-side regions, `source_adjacent_region_ids`
and `source_adjacent_region_occurrence_paths` are parallel oriented pairs. An
exterior side is `None` in both fields; an authored side carries its raw source
region ID and exact occurrence path. Repeated raw region IDs are distinct when
their paths differ. `RegionBoundaryEvidence` binds one such occurrence through
its `MeshingScope`, records the same raw ID and path, and selects current target
boundary entities with explicit patch-side orientations. Its `control_id` is
`None` when no explicit region control was supplied.

Occurrence paths are part of `MeshPatch.patch_id` and
`RegionBoundaryEvidence.evidence_id`; publishing them intentionally changes
those fingerprints and the static result organization structure. Recomputed
result IDs and persisted organization records therefore use the occurrence-aware
representation.

`PiecewiseLinearDomain.region_occurrence_paths` is likewise aligned with its
raw `region_ids`: an empty path denotes an uninstanced region, while repeated
IDs require distinct explicit paths. The paths contribute to the domain source
revision and identity.


Geometry associations record source entities, target IDs, residuals, and
resolution/ambiguity evidence. Persistent IDs are not positional array indices.

## Controls and providers

`SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, and `VolumeMeshingSpec` separate
physical requests from backend options. Cell-family policies, fill strategies,
size controls, regions, patches, layers, and periodic constraints are explicit.
A control's presence in the native contract does not imply every provider can
honor it. Provider preflight rejects unsupported combinations before work.

Family acceptance uses the reference-cell owner's declared family classification.
A native `simplex:3` block satisfies a `tetrahedron` request, and `tensor:3`
satisfies `hexahedron`; the block kind, reference topology ID, and source identity
are not relabeled. Equal dimension or vertex count alone never establishes a
family match.
Cell-quality and finite-volume face diagnostics use the same classification and
the canonical simplex incidence banks, including dimension-qualified meshes.
Scalar form coordinate elements also use the canonical exact affine-map validity
proof. Dimension-qualified simplex and tensor cells retain their reference
identities while positive, inverted, and collapsed maps receive the same
determinant checks as other affine representations.

When no `NativeSurfaceSchedule` is supplied, native surface meshing uses the
canonical `spacing_fraction=0.9` default. The fraction scales resolved physical
size controls; it is not a source-fidelity tolerance. Explicit caller schedules,
size controls, and deviation bounds remain unchanged.

### Native trimmed-chart surfaces

The native `parametric_surface` route accepts affine as well as rational and
spline p-curves when their chart intervals provide continuous enclosures. Each
source curve is discretized in its canonical parameter once and that chain is
shared by every selected trim use. Mapped interval boxes bound both physical
chord deviation and edge length; those bounds, rather than a finite witness
sample, drive trim refinement and remain in the construction evidence.

Sphere and cone pole trims use their authored collapsed-edge corner as one mesh
vertex. The provider places a regular chart ring from continuous surface-box
bounds, removes chart triangles that collapse onto the pole edge, and retains
the cap deviation bound while constructing positive fan triangles. Periodic
chart seams identify their equivalent endpoint vertices, so a singular chart
does not create zero-area output cells or duplicate seam nodes.


An empty `size_controls` tuple declares no physical size obligation. Routes
whose discretization is already defined by authored sites, blocks, stations,
or source cells do not require a fabricated sizing field. A route that needs
additional physical resolution must reject the missing input during preflight,
not invent a tolerance or silently ignore supplied controls. In particular,
periodic power cells use their authored sites and signed weights.



### Sweep source identity and chart maps

For continuous piecewise-linear boundary fidelity on a native sweep, bind each
source-domain vertex to its profile vertex and station with
`NativeSweepSource.domain_vertex_map`. This integer table has one row per
`domain.vertices` row and stores `(profile vertex global ID, station index)`.
It is an explicit topological correspondence: coordinate equality does not
infer profile ancestry, and the map contributes to the source binding identity.
The PWL boundary query must be full-scope and bound to that same domain.

A graded `LayerSchedule` sweeps a whole slab natively without a core fill: the
stations are the cumulative layer thicknesses, so the first-cell height is the
first scheduled thickness exactly. Each PWL wall facet must map to one swept
target face, so the domain carries a vertex at every station on its walls
rather than one tall facet spanning several layers.

The composed boundary witness keeps source-chart and target-reference
coordinates distinct. A standard triangular boundary face of a 3D swept cell
retains all three coordinates in the parent cell's 3D reference domain even
though its source chart is 2D. A polyhedral face-local chart instead uses 2D
target-face coordinates and explicit target vertex IDs. Chart dimensions are
declared independently; matching dimensions do not establish identity or
correspondence. Certification checks the witness against the retained source,
mesh, and geometry maps; matching sampled coordinates alone is not proof.


### Typing and structural contracts

Meshing conversion boundaries accept JAX/NumPy array-like values and convert once;
compiled kernels consume canonical JAX arrays. `MaskedSimplexMesh`,
`MeshMetricField`, and `MeshMetricSamples` opt into `phydrax.typing` structural
contracts. Their nominal dimensions enforce aligned vertex/cell slots, coordinate
components, simplex widths, and square metric tensors after transformations such as
`equinox.tree_at`, without adding device operations. Scientific admissibility remains
with the owning constructors and evidence: simplex orientation/capacity, metric SPD
properties, hard size/anisotropy bounds, gradation, and provider compliance are not
inferred from shapes.

`RegionControl` assigns a named material-neutral region to an explicit solid
scope. `PatchControl` assigns a named exterior or two-region interface to an
explicit face or edge scope. Create scopes with
`MeshingScope.from_geometry(model, model.face_ids[face])` or
`MeshingScope.whole(model, dimension)`. B-Rep scopes use
revision-bound numeric projections of canonical source entity keys and occurrence
paths. Gmsh binds source definitions through native CAD exporter provenance,
source-directed entity imports, and its exact OCC input/output correspondence;
it verifies geometry and topology incidence after exporter identity is bound. It
never casts scope values to rows or infers identity from display labels,
unassociated provider tags, or shape coincidence. Gmsh source evidence requires
one physical occurrence per definition; repeated definitions are rejected rather
than merged. Scoped plans also refuse nonidentity assembly placements, which this
source-directed import path cannot preserve.
This keeps every patch adjacency tied to one exact B-Rep revision.
For a volume request scoped by its boundary faces, region controls still select
three-dimensional solids from their own entity set. Face IDs and solid IDs are
not compared as though they belonged to one set. Controls must retain the same
source identity and revision, and patches must name declared adjacent regions.

Uniform, curvature, and proximity controls compile into a resolved size field.
Proximity gaps are exact BVH nearest distances between samples of the source and
target scopes; opposite-normal controls require sample normals. Hard growth
limits enforce the edge-length-aware bound `h_i <= h_j + (rate - 1) |x_i - x_j|`
exactly. `size_field_metric` compiles a resolved field into isotropic metric
constraints bound to the same entity IDs.

`MeshMetricField` represents an SPD anisotropic metric whose eigenvalues are
inverse squared target lengths. Construction certifies every row against the
declared hard bounds, `1 / maximum_size**2 <= lambda <= 1 / minimum_size**2` and
`sqrt(lambda_max / lambda_min) <= maximum_anisotropy`, on the
`phydrax.linalg.verify_dense_properties` spectrum; only eigenvalue roundoff
(dtype epsilon times the row spectrum, hence condition-scaled at `lambda_min`) is
admitted, and a violating tensor raises `ValueError` instead of being repaired.
A field carries no gradation bound: requested gradation belongs to a
`MetricGradationPolicy` (certified by `MetricGradationEvidence`) or to provider
options such as `MmgOptions.gradation` and `OmegaHOptions.gradation_rate`.
`normalize_mesh_metric` applies an explicit `MetricNormalizationPolicy`: size and
anisotropy bounds, a target complexity `sum_i V_i sqrt(det M_i)` over vertex
volumes solved by the native bracketed root, and optional gradation. Untrusted
tensors enter as `MeshMetricSamples`, are symmetrized or projected only when the
policy requests it, and `MetricNormalizationEvidence` counts every repair and
clamp. `grade_mesh_metric` bounds growth with the physical law
`h_q <= h_p + (beta - 1) |pq|` or the metric-space law
`h_q <= h_p beta**l_p(pq)`: scalar gradation is an exact minimum-first
relaxation and anisotropic gradation uses Alauzet grow-and-intersect sweeps.
`MetricGradationStatus` distinguishes `CONVERGED`, `BOUNDS_CONFLICT`, and
`SWEEP_LIMIT`; convergence requires the independently measured
`maximum_violation` to meet `relative_tolerance`. `grade_mesh_metric` and
`normalize_mesh_metric` raise `MetricGradationError` carrying that evidence
instead of returning an executable field when hard bounds or the sweep budget
prevent the requested gradation.
`combine_mesh_metrics` intersects metrics by canonical simultaneous reduction
(order independent) and returns a `MetricCombinationResult`: the combined field
exists exactly when `successful`, while `MetricCombinationEvidence` reports an
empty intersection of the declared size intervals or every tensor row that
conflicts with the most restrictive declared bounds, so a contradictory metric
can never reach an adaptation route. Adaptation requests,
`BackgroundMetricControl`, and the Mmg and Omega_h plans accept only a certified
`MeshMetricField`. `interpolate_mesh_metric` is log-Euclidean,
`metric_edge_lengths` returns Riemannian edge lengths, and `lp_metric_from_hessian`
builds the Loseille-Alauzet `L^p` metric from a recovered Hessian
(`phydrax.discretization.fem.recover_hessian`). These operations prepare
requests; they do not certify that an external generator met them.

`LayerSchedule` declares physical layer thicknesses, ordered from the wall
outward. `BoundaryLayerControl` binds a schedule to a wall scope and one explicit
`BoundaryLayerRoute`:

- `EXACT_SWEEP` meshes a pre-partitioned straight slab cap to cap (wall, cap,
  and volume scopes);
- `CAD_EXTRUSION` is realized before meshing by
  `prepare_boundary_layer_extrusion`, which extrudes planar walls along their
  inward normals, splits the source by the exact prisms, publishes the
  partition, and returns the `EXACT_SWEEP` control that meshes it; it rejects
  prisms that leave their solid, overlapping slabs, and slab sides that would
  meet the core;
- `ADVANCING` grows native layers (`prepare_boundary_layers`) from an oriented
  triangle/quadrilateral wall mesh, or from closed BRep walls inside
  `GmshProvider`;
- `PROVIDER` lowers the schedule to Gmsh: a `BoundaryLayer` field on planar wall
  curves (geometric schedules), or boundary-layer extrusion of closed 3D walls.

Native advancing layers classify wall edges by dihedral angle, split columns at
convex ridges into fan columns and corner patches (`BoundaryLayerCornerPolicy.FAN`),
and point every column along the visibility-optimal direction, the unit `d`
maximizing `min_i n_i·d` over its face normals: the normalized minimum-norm point
of their convex hull, found exactly by enumerating hull faces of at most three
normals and certified by a closed weak-duality gap. Columns whose normals enclose
the origin have no visible direction and are rejected with their wall vertices.
Concave corners are accepted only while the height stretch stays within
`maximum_corner_stretch`; otherwise they are rejected with the offending wall
vertices. Heights are limited by the
smooth concave radius and by BVH-nearest medial-crossing probes; every layer is
certified by Bernstein validity and exact-predicate intersection against earlier
layers, the wall, adjacent surfaces, and obstacles. Collisions resolve only by
`BoundaryLayerCollisionPolicy`: `FAIL`, `TERMINATE_LOCALLY` (pyramid/tetrahedron
terminations, never below the first layer), `REDUCE_THICKNESS` (never below
`minimum_thickness_fraction`), or `MERGE` (mutually paired, face-to-face opposing
fronts share their midsurface). Failures carry the wall vertices and locations.
`BoundaryLayerMesh` holds the certified layer cells, the exact cap (quadrilateral
cap faces are closed by transition pyramids), and `BoundaryLayerEvidence` with
thicknesses measured as exact distances to the wall. Per-layer statistics cover
the surviving columns that carry each layer; `layer_active[k]` is `True` iff at
least one column carries requested layer `k`. A layer carried by no column (for
example after `TERMINATE_LOCALLY` stops every column early) has
`layer_active[k] == False`, NaN achieved, minimum, and maximum thickness, and NaN
for every growth rate touching it (`achieved_growth_rates[k]` is the ratio of
achieved thicknesses `k + 1` and `k`). Read layer support from `layer_active`,
not from NaNs; `terminated_vertex_count`, `reduced_vertex_count`, and
`merged_vertex_count` state why columns stopped short of or shrank the schedule.
`GmshProvider.fill_boundary_layer_core` tetrahedralizes between a closed cap and
the remaining boundary while keeping both fixed: fixed nodes are verified
bitwise and core faces must match the fixed triangles exactly; the merged mesh
carries `boundary-layer` and `core` zones and `wall`, `layer-core-interface`, and
`outer` patches. `PlanarBandControl` instead declares explicit side schedules over
a planar partition patch; `prepare_planar_bands` returns the per-layer partition
evidence.

`NativeLayerCoreSource` can instead declare a `core_power_source` and request
`VolumeFillStrategy.POLYHEDRAL` with the polyhedron family. The native layer/core
route keeps the accepted layer cells and cap fixed, fills only the remaining
core with authored restricted-power sites, and composes the result as a mixed
standard/polyhedral `CellMesh`; each cap facet must have one layer and one core
incident cell. Missing authored core sites refuse rather than synthesizing a
power source. This route currently refuses mapped or high-order polyhedral cores
that cannot retain their exact geometry through the mixed assembly.
Restricted-power generation joins every site's carrier pieces across
unconstrained carrier faces and splits a site into per-tetrahedron pieces only
when an authored segment or protected vertex lies off every facet polygon.
Periodic PLC power quotients require the transformed source facets to have the
exact transformed output face subdivision, not only matching vertices. Two
polyhedra whose common interface is several coplanar faces declare contact only
on their exact convex face union.
`CAD_EXTRUSION` also admits a whole-solid sweep: if the authored slabs consume
the controlled solid, the partition has no core solid, and the swept cap is a
one-sided exterior patch. Source-solid volume must equal the sum of its slab
volumes; an empty core is not inferred from an absent region alone. Mesh this
case with a layer-only region control and no core fill.

`NativeMeshingProvider` grows one `ADVANCING` `BoundaryLayerControl` straight
from a BRep model (`cad_tetrahedral` route). The wall scope names source faces and
the volume scope names the whole solid. The accepted boundary's exact
per-cell charts resolve those faces to wall cells. Layers grow inward from them.
The fixed core boundary is the cap plus every unselected surface cell; each rim
vertex of an unselected cell moves to the top of its column, so the core
conforms to the cap. The core is filled with tetrahedra. The target must allow
mixed `prism`/`tetrahedron` cells (transition pyramids optional) and use the
`SIMPLEX` fill. Rim columns slide along the adjacent surfaces. A schedule
whose fronts meet those surfaces before the collision policy can resolve them is
refused with the colliding wall vertices and their locations, so choose
thicknesses that are small relative to the surface size. Source fidelity covers
every exterior facet: the grown cells keep their charts, and fixed cells and
layer side strips take the closest-point charts of their source face. A column
shared by two faces lies exactly on their common trim. The `wall` patch is only
the exterior layer faces that lie on the wall. A side strip rising from rim edge
`(a, b)` joins the `facet:` patch of the fixed core polygon holding the tops of
the columns of `a` and `b`. Strips without such a unique polygon form the
`layer-side` patch. Refinement that cannot reach its targets because of the fixed
boundary is reported as evidence; improvement that leaves unmet quality criteria
fails compliance.

`prepare_periodic_layer_core_source(source, core_vertex_representatives,
core_vertex_shifts, core_seam_polygon_pairs)` binds source-authored PLC orbits
and seam facets to the accepted layer mesh's periodic topology identity. The
precursor must retain source-bound `SurfaceSourceCharts` and their fidelity
query. Its result carries that same periodic topology identity; generated
vertices are never paired by coordinate proximity.

Gmsh is optional (`pip install 'phydrax[meshing-gmsh]'`). Use
`GmshProvider` with a `BRepModel` or solid `BRepSource`; a reopenable source path
is not required. On cache miss Gmsh exports the current model to native BRep text,
re-decodes it in the declared coordinate contract, and requires an exact
`source_revision` match before any provider import input is created. Unsupported
native exports refuse this optional route without changing the in-memory CAD
operation. Cache keys bind source owner, revision, and coordinates. `GmshOptions`
does not own coordinates. Sessions are host-side resources and must not enter JAX
state. Use a context manager for an explicit `GmshSession`; its cleanup does not
suppress the original exception. Open CAD faces use `BRepModel`; `BRepSource`
retains its closed-solid invariant. Required mixed families must actually occur in
the generated result; permitting mixed recombination does not guarantee that
triangles will remain.

### Exact semantic B-Rep and planar routes

`BRepImportReport.source_digest` identifies the original persisted CAD bytes;
`source_revision` binds the current native geometry, coordinate contract, and
physical source accuracy. Gmsh does not treat that artifact digest as authority: it
validates a native BRep-text round-trip against the supplied model's exact revision.
Replacing the original file therefore cannot silently replace the immutable model
bound to a meshing plan. `BRepPartitionPlan` and `PlanarPartitionPlan` are geometry-owned
Boolean operations: they publish persisted B-Rep output, exact
`CADRevision`/`AssociationGraph` evidence, and named solid/face regions with
face/edge patches.
Gmsh uses native exporter declarations and its explicit OCC fragment map to bind
provider tags. It accepts only one-to-one source-definition maps; a fragment that
splits a solid or collapses distinct definitions is refused, never used to infer
material meaning from labels or proximity.

`RegionControl` and `PatchControl` are dimension-generic. A volume request binds
regions to solid scopes and patches to face scopes; an ambient-two surface request
binds regions to planar B-Rep faces and patches to exact B-Rep edges through one
`PlanarEmbedding`. The resulting planar `CellMeshingResult` has ordinary
top-cell zones and main-mesh edge patches, no synthetic `SurfaceModel` boundary.

`LayerSchedule` is explicit physical thickness data. `EXACT_SWEEP` boundary layers
admit only complete, straight prism sweeps that join unswept tetrahedra through
triangular caps. `PlanarBandControl` creates exact straight CAD strip partitions;
it rejects corners, junctions, collisions, insufficient clearance, and any
topology change. Pure-Q4 band requests require an entirely rectangular
transfinite closure; otherwise request an allowed triangular remainder.

Size bounds and growth become hard only when explicitly supplied. `target_size`
is a preference, not an unstated exact bound. Provider compliance records
per-control observed edges, hard-bound tolerances, and growth evidence.

Native initial meshing uses one source-bound plan/execute API:

```python
from phydrax.meshing import NativeMeshingProvider

plan = NativeMeshingProvider().plan(
    source,
    specification,
    coordinate_contract=coordinates,
)
result = plan.execute()
```

`NativeMeshingOptions.route` may select a route explicitly. When omitted,
planning deterministically resolves a route from source capabilities and the
request; unsupported combinations fail rather than falling back to another
algorithm. The result is published only after canonical mathematical
acceptance. Sources that own a coordinate contract bind it directly; otherwise
the caller must supply the actual units and reference frame.
PLC-backed sources scope dimension-1 entities by stable source identity: explicit
segments and original polygon-loop edges, including shared seams between polygons
of one facet. Edges introduced only by recovery triangulation are not source curves;
they retain their incident source-facet and region-side ancestry and are associated
to the facet rather than assigned a new curve identity.

Native PLC boundary recovery splits a missing segment or subface only at exactly
witnessed source points (`S = A + t(B - A)` or `A + l1(B - A) + l2(C - A)` with
dyadic parameters); every predicate evaluates `S` itself, so split points stay
exactly collinear/coplanar with their sources even when `S` is not a binary64
point. After recovery such points are suppressed again: under `FIXED` every
boundary split point, under `CONFORMING` every split point that is not a binary64
point. Suppression retriangulates the subfaces through the point within their own
planes and refills each wedge of its star by contraction onto a neighbour, else by
the first of: the exact constrained Delaunay gift-wrapped fill, cones from one
cavity vertex, the constrained Delaunay fill with interior volume points, cones
from one binary64 interior point, and finally an exact exhaustive Steiner-free
search. Deciding whether a Steiner-free fill exists is NP-complete, so the
polynomial fills come first and the search, whose use is reported in
`PlcRecovery3D.suppression_search`, runs only when they all fail. Every fill is
accepted only after exact positivity and cavity-boundary checks. Published boundary vertices
therefore lie exactly on their sources. A point that cannot be removed is a typed
refusal naming its source entity: `STEINER_SUPPRESSION` under `CONFORMING`,
`FIXED_SEGMENT`/`FIXED_FACET` under `FIXED`. A requested curve feature on an
internal diagonal of a coplanar facet group is recovered like any other segment.

An analytic implicit profile with established reach uses restricted Delaunay
for volume meshing by default. Adaptive implicit volume discovery is distinct
and must be selected explicitly as `implicit_adaptive_tetrahedral`; it is not a
fallback for restricted-Delaunay failure. A direct `BRepModel` volume request
builds and accepts its parametric boundary mesh internally before CAD
tetrahedral filling. It requires a whole-source face scope and an explicit
`maximum_deviation`; no tolerance is inferred from source tessellation or
import accuracy. Initial topology decisions are nondifferentiable.

The adaptive implicit carrier is the cell-band extraction of the discovery
scaffold, whose triangles can be slivers. Because a tetrahedron on a `FIXED`
face has a radius-edge ratio at least the face's, a carrier with any face above
`NativeVolumeSchedule.radius_edge_bound` is first remeshed natively toward the
requested uniform target size (else its mean edge length) and a 30° minimum
angle. Every created triangle is admitted by the zero-set band predicate: from
native interval enclosures of the field and its gradient, its distance to the
zero set is certified at most the scaffold's own bound (residual over a
certified gradient-norm lower bound on a widened box) and its normal is
certified transversal to the gradient with the carrier's orientation. Vertex
proposals come from interval Newton along the gradient line. The published
fidelity bound is the certified maximum over the remeshed triangles; each face's
chart witness names the nearest scaffold triangle, and coverage proves degree
one without a one-to-one scaffold correspondence: the certified charts make the
lift a covering, and one lifted point per boundary component has exactly one
preimage, so folded or duplicated sheets are refused. A remeshed face still
above the bound is a typed `QUALITY_REJECTED` refusal; carriers already within
the bound keep the scaffold triangles and their one-to-one certificate. The
compliance report records the carrier ratio and minimum angle before and after.

Native curve meshing binds each `SegmentMesh` edge to a `SegmentChartCurve`: the
exact affine map of the edge onto its declared chart parameter range. Its rational
coefficients make chart fidelity an exact polynomial bound per target edge, and its
float realization reproduces both authored endpoints bit for bit. Point placement
inverts the chart's adaptive size-density table panel by panel rather than
re-integrating the chart for every bisection step. A hard `UniformSizeControl`
target is compared with measured arc lengths at the `SizeCompliancePolicy`
tolerances, which default to zero; a target such as 0.2 m that floating-point
placement cannot reproduce exactly needs an explicit tolerance in the request.

Native tetrahedral refinement and quality improvement (`TetMesh3D.refine` and
`TetMesh3D.improve`, run in that order by the native volume routes) have these
contracts:

- **Size field.** `sizes` holds one circumradius target per current vertex (0:
  unconstrained). A cell's target is the mean of its corners' targets, and is
  unconstrained if any corner is. The field is retained on the mesh: refinement
  installs it, every vertex either pass inserts receives the target interpolated
  at its position, and a removed vertex's entry is erased. `improve()` without
  `sizes` uses the retained field, so a route's refinement size field (for
  example a `UniformSizeControl`) constrains improvement too.
- **Refinement.** Circumcenters of cells violating the radius-edge bound or
  their size target are inserted through a checked cavity; an encroached
  splittable boundary face is split instead. No point is inserted closer than
  min(lfs(x), h(x))/2 to an existing vertex, h(x) the target interpolated at
  the point and lfs(x) the local feature size: each vertex entering the pass
  measures its distance to the nearest constrained vertex, segment, facet edge
  or facet not incident to it, and lfs grows from the point's neighbors by the
  distance to them. Next to a thin fixed face the spacing floor is that face's
  own scale, not the shortest edge anywhere in the mesh. Since refinement
  removes no vertex, the inserted points are packed at a positive spacing in a
  bounded domain, so finitely many are inserted, and densely only near small
  features. A cell the guard blocks is reported as `REFUSED`; one whose
  circumcenter lies beyond a fixed face as `FIXED_BOUNDARY`.
- **Conforming boundary refinement.** Under `CONFORMING`, an encroached
  subsegment is split at its midpoint and an encroached subfacet at its
  circumcenter, each inserted with the Bowyer-Watson cavity of its subfacet
  disk (Shewchuk's order: a subsegment the point encroaches is split
  instead). Size-driven boundary splits come first; a facet edge whose
  midpoint would encroach a subsegment splits that subsegment first. Every
  split point is an exact source point: the mean of the edge endpoints' exact
  source parameters when one source row carries both, otherwise the point is
  located in whichever source triangle (or segment) of the face group holds
  it. The carrier is that point's correctly rounded realization and the
  insertion cavity is checked with the commit's exact predicate on exact
  source positions. A face group is one declared PLC facet; without a
  declared exact plane, the authored triangles of a rotated (not exactly
  coplanar) facet stay the geometry and their shared edges stay segments.
  Next to an acute input feature (two segments, a segment and a facet, or two
  facets meeting below 90 degrees through the domain) a quality split point
  within the feature's protected radius is not inserted, and a cell left for
  that reason is reported as `INPUT_ANGLE` with the feature's apex vertex and
  angle (`TetMeshUnmet.apex_vertex_ids`, `apex_angles`).
- **Declared exact planes.** A PLC facet may declare an exact plane
  (`facet_planes`: origin, first axis, second axis in binary64, with
  `facet_plane_tolerances`); the TetMesh source carries it per face row
  (`TetMeshSourceComplex.face_planes`, `face_plane_tolerances`). Its
  triangles then form one face group whatever their rounding, and the shared
  edges between them are not segments. A boundary vertex of the group is
  admitted on the plane if and only if the plane meets the closed box of reals
  that round to its carrier, or the carrier lies within the declared
  tolerance of the plane; both tests are exact. Creation refuses the
  declaration otherwise, raising `TetMeshPlaneRefusal` with the first refused
  vertex, the plane's face row, the rounded parameters of the carrier's
  projection onto the plane, its outward distance bound and the tolerance;
  nothing is created. `TetMesh3D.plane_records()` lists every admitted
  (vertex, plane) incidence with the same record. Boundary split points of a
  declared group are placed on the plane itself (witness stratum 3: the point
  `origin + s * first_axis + t * second_axis` of the face row's plane, carrier
  its correctly rounded realization), a subfacet is valid when every corner
  is admitted and its projection onto the plane keeps the facet's orientation
  inside the group's triangles, and acute input angles use the plane normals.
  Downstream, a stratum-3 vertex is a facet point of that face row's facet.
  Without a declaration a rotated facet keeps its authored triangles, as above.
- **Coverage of boundary splits.** A split point's exact source point S (the
  segment, triangle or declared-plane point named by its witness parameters)
  is authoritative and its binary64 carrier is only S rounded, so a split
  boundary is not exactly on the rounded source triangles. When refinement
  inserted boundary points, the native PLC route therefore retains a
  `PlcSourcePointWitness` (each vertex's source kind, row and parameters,
  with the source segments, triangles and declared planes) as the
  certificate's `source_coverage_witness`. Coverage recomputes every S in
  rationals, refuses a parameter that leaves its segment or triangle
  (`source_point_parameter`) and a carrier that is not the correctly rounded
  S (`source_point_rounding`), and then matches the oriented facet chain and
  integrates the region measures on S. Without declared planes the proof
  stays an exact chain equality (`exact_oriented_facets`) whose
  `source_distance_upper` is the carriers' rounding bound. A declared facet
  whose rounded triangles leave its plane is compared after exact orthogonal
  projection onto that plane; every corner must be admitted to the plane as
  above, and the certificate is a `source_boundary_homotopy` bounded by twice
  the largest corner distance to the plane plus the rounding bound. Unsplit
  boundaries (every `FIXED` route) retain no witness and certify exactly as
  before.
- **Improvement acceptance.** Each cell scores the worse of its dihedral-angle
  and radius-edge target fractions (1 when both hold), floored to multiples of
  `TetMeshRun.score_quantum` (2^-20); a cell below the determinant floor scores
  0 whatever its angles and ratio (a flat cell on a cocircular boundary quad
  has a finite circumsphere and could otherwise score 1). Every face flip,
  edge-ring reconnection, interior vertex insertion, collapse, and relocation
  is accepted only if the cells it creates pass the hard gates (relative
  determinant above its floor, source constraints preserved) and the size
  gate, and the affected cells' worst-first sorted scores strictly improve. An
  interior insertion either cones a circumsphere cavity to a candidate point
  or, for a cell whose edits are otherwise exhausted, cones the closed ring of
  one of its unconstrained edges to a point in the ring's kernel. The latter
  removes a flat cell whose four vertices lie on a fixed boundary (two
  constrained faces splitting a planar or slightly folded quad) without
  moving the boundary: no flip can remove such a cell and no circumsphere
  cavity contains it. The ring point is searched over the kernel in the
  edit's own order (a cone passing the gates beats one that does not, then
  the larger sorted score vector wins), since a point merely clear of the
  ring faces can cone a nearly flat ring into worse or oversized cells. A
  relocation must also make sufficient progress: at the first position where
  its star's sorted scores change, the new score meets the target or closes at
  least an eighth of that score's gap to the target (at least one quantum). A
  cell's gap therefore shrinks geometrically rather than one quantum per move.
- **Precedence.** The determinant floor binds every edit. The size gate is
  relative to the cells an edit removes: no created cell may exceed its target
  by more than the largest excess (circumradius over target) among them, or at
  all where they were all within target. Edits therefore never create a size
  violation where none existed and never grow the mesh's largest excess, while
  cells refinement left beyond size (at a fixed boundary, or refused by the
  separation guard) can still be repaired. An edit that removes a degenerate
  cell (below the determinant floor or scoring 0, a ratio beyond 2^20 times
  its target) is not size-bound: repairing degeneracy takes precedence. The
  whole-mesh fixed-boundary retessellation must keep every cell within target.
- **Termination.** Every accepted edit strictly improves the whole mesh's
  sorted score vector, and over the finite score alphabet that order admits no
  infinite improving sequence. The gates above only discard edits, so they do
  not affect this argument. Improvement therefore ends without pass,
  candidate, or time limits, although insertion and collapse change the vertex
  set; no complexity bound is claimed.
- **Unmet evidence.** Targets the edits cannot reach end valid but unmet: the
  run status is unresolved and `TetMesh3D.unmet()` reports each remaining cell
  per criterion with its reason. Refinement reports the radius-edge ratio and
  size (circumradius over target); improvement reports the radius-edge ratio,
  minimum dihedral, and determinant validity, since the size field constrains
  improvement but is not one of its targets. Improvement never edits a
  constrained face, and a tetrahedron's radius-edge ratio is at least that of
  each of its faces (its circumsphere contains the face's circumcircle and its
  shortest edge is at most the face's). A radius-edge residual on a
  constrained face whose own triangle already exceeds the bound is therefore
  `FIXED_BOUNDARY`: no tetrahedron on that face can meet it, and the boundary
  producer must supply better-shaped faces. The record names that facet and
  its triangle ratio (`TetMeshUnmet.facet_ids`, `facet_ratios`; the largest
  when several force the cell). Other improvement residuals are
  `NO_IMPROVEMENT`. Volume routes refuse a mesh with improvement residuals
  rather than publish it.

`ImplicitSurfacePlan`, created by `geometry.discover_implicit_surface`, is the
geometry-owned fixed-lattice, fixed-topology operation. Its `realize(state)`
keeps the discovered route and triangle connectivity fixed while JAX
differentiates the realized surface coordinates. This does not differentiate
through adaptive extraction, topology discovery, or canonical meshing
acceptance; initial native meshing uses the separate nondifferentiable
`NativeMeshingProvider` workflow.

`ManifoldProvider` performs n-ary union, difference (first operand minus all
others), and intersection of closed, oriented triangular `SurfaceModel`
operands in the same physical frame. It preserves source-face ancestry and cell
tags through the backend's face relations. Named `vertex_properties` are carried
through the backend and published per face corner, each checked against
barycentric interpolation on its reported source face. Semantic
selections/interfaces require explicit transfer and are rejected rather than
silently dropped. Empty output is reported as a failure because
`CellMeshingResult` requires a nonempty carrier. Boolean topology changes are
nondifferentiable.

The separate `geometry.surface.arrange_triangle_surfaces` and
`geometry.surface.surface_boolean` APIs operate on triangle-mesh geometry and
retain source-face ancestry only. They do not perform or certify exact CAD
source operations; use `BRepPartitionPlan` or `PlanarPartitionPlan` when CAD
entities and their associations must remain authoritative.

`OpenVDBProvider` ingests sparse voxel bricks in bulk and may re-distance the
isosurface with `OpenVDBLevelSetRebuild` before extraction. `PoissonProvider`
publishes native per-vertex sample density, optional density-quantile trimming,
an explicit thread count, and the declared (Neumann) boundary condition.

### Native quadrilateral and hexahedral family routes

`NativeMeshingOptions.route` selects distinct native family constructions; an
unsupported family or source capability is refused rather than replaced by a
different mesh family or a simpler route. `planar_dual_quad` solves a
source-feature-guided cross field and uses it to drive constrained triangle
recombination, including field alignment, interior valence, boundary parity,
and quad quality. This field-guided layout changes connectivity; it is distinct
from explicit triangle-dual conversion. When the scaffold has no complete
positive pairing (for example a four-triangle star), the route keeps the
largest positive pairing. Unpaired triangles split into three quads at their
edge midpoints and centroid. Each split edge forces the opposite edge of the
adjacent paired quad to split too, so the splits run along whole quad strips
and the mesh stays conforming. Every new vertex keeps its exact source-vertex
support. Calling `extract_surface_quads` directly keeps the refusal unless it
passes `unpaired_triangles="subdivide"`.

`plc_dual_hex` is the explicit tetrahedral dual conversion. It retains parent
cell, source-vertex, material-block, and exact coordinate-map ancestry, but is
not a general all-hexahedron PLC mesher. `plc_hex_dominant` reports its actual
hexahedron, pyramid, and tetrahedron transition blocks; it does not satisfy an
all-hexahedron request by silently retaining those transitions.
`plc_hex_dominant` composes exact source charts through its mixed blocks. Curved
PLC input also requires the independent mapped-root authority described below.

`plc_balanced_grid_hex` constructs cut cells from exact PLC/tetrahedral
intersections, then closes their conforming face arrangements with an
all-hexahedron transition. It supports nonsimple shared interfaces and cavity
domains while retaining source-cell regions and exact reference coverage.
The lattice knots stay a fixed fraction of the spacing away from every source
vertex. They also stay that far from every crossing of a source edge with a
grid plane and every crossing of a source face with a grid line. Cut cells
therefore never collapse to a numerically zero volume because of binary64
coincidences.
The public PLC cut route never infers curved geometry from vertex positions.
Curved sources require both an authored `complex.geometry` and an independent
`MappedReferenceDomain` binding the root geometry, reference carrier, frame,
and region relation. Tetrahedral recovery must retain exact target-to-root
charts before this route creates new cut-cell topology; this is not the
pre-authored mapped-grid route. Internal cut axes must match the affine carrier
exactly, and uncut curved axes must fit their outer interval by Bernstein
bounds; other curved plane intersections are refused.
The source-aware relocation fixes source corners and keeps each moving cut node
within its authored root vertex/edge/face/cell support; artificial grid planes
are not treated as fixed source features. Every proposal retains exact chart and
region ancestry and must pass reference coverage, whole-cell validity, global
embedding, and the unchanged quality schedule before publication.

`plc_frame_grid_hex` is separate from that balanced carrier-grid route. It
solves spatially varying per-tetrahedron frames and exact block/integer charts,
then checks face-axis transitions, offsets, boundary parity, and cycle
holonomy. A nonintegrable field or incompatible parity produces a
source-bound obstruction; it does not fall back to one global frame or to
tetrahedral dualization. `mapped_balanced_grid_hex` and `mapped_frame_grid_hex`
are separate again: they subdivide an already mapped hexahedral root mesh and
retain its exact root-face transitions.
Their target geometry is the exact per-cell restriction of the root map, so a
request without `maximum_deviation` is certified at zero source deviation; an
explicit bound is honored as requested.

On a curved PLC source, `plc_hex_dominant` and `plc_balanced_grid_hex` both
restrict one curved tetrahedral fill. Every output cell keeps an exact chart
into its fill tetrahedron, and that tetrahedron keeps its affine chart into the
curved root.
Global embedding reuses the fill's own proof through these child-in-fill maps
(a forward reference chain) rather than proving curved cell pairs apart one by
one. Source fidelity carries the fill's boundary chart witness onto the output
faces. A quadrilateral face that is a twisted bilinear patch of its fill face
is split into straight pieces until the correspondence error stays within half
the tolerance; that error is bounded with the fill map's exact Bernstein
Lipschitz constant. Parallelogram faces keep two pieces. Curved transition
pyramids are per-cell restrictions with rational charts. Such a chart is
accepted only when its `1 - z` pole cancels in collapsed-cube form, the
remaining denominator has a strict sign, and the apex has one image. Evaluating
the gradient exactly at the apex is still refused when it depends on the
direction of approach.

Every accepted family still passes the common continuous whole-cell validity,
embedding, coverage, and source-fidelity checks. Quality samples are
observations only; they cannot prove those claims or authorize a different
cell family.

### Periodic point-site and power routes

`PeriodicPointOrbits` binds authored site IDs to a fully periodic lattice;
`NativePeriodicSource` adds the source revision, optional fixed power weights,
and, for owner-local execution, one execution-group owner slot per local site.
`NativeMeshingProvider.plan(source, specification).execute()` is the public
route. Periodic surface and simplex-volume requests use periodic Delaunay.
Three-dimensional `POLYHEDRAL` volume requests that require the `polyhedron`
family select periodic power cells. Power sites and weights remain scientific
source identity: the route computes exact source-bound carrier/power
intersections and does not move or reseed authored sites to meet a size target.
Orbit keys and windings are proven from those exact intersections and deck
actions. Each binary64 vertex is validated independently as the exact-coordinate
shadow; transforming an already-rounded representative need not reproduce a
separately rounded lifted vertex bit for bit.

For owner-local execution, each rank supplies its owned and resident ghost site
rows and calls `plan.execute(execution_group=group)`. Generation closes native
support by exact periodic predicates and returns a `MeshAssembly` containing a
compact local `OwnerLocalCellGeometry`; no rank gathers a global site table or
mesh. The local view retains stable cell owners, exact quotient keys, absolute
deck windings, source/map context, and the collectively proved quotient-facet
coverage.

Periodic Delaunay views exchange every face-neighbor cell of an owned cell that
is not already resident before publication. The owner-pair transfer retains
exact lifted coordinates, quotient IDs and windings, and cumulative
interpolation ancestry; it follows certified face adjacency rather than a fixed
local cell ring.

Owner-local uniform size controls bind the common complete fundamental-domain
scope, not rank-local point subsets; each process validates the same source,
revision, entity set, and domain ID.

Periodic triangle metric adaptation decides on the quotient, not on the lifted
fundamental-domain cut: boundary and material-interface edges are those whose
quotient edge orbit meets one cell orbit or cell orbits of different classes,
so seam edges stay unclassified. A collapse removes every lifted member of the
vertex orbit onto exact deck images of the kept vertex; a lifted member row the
collapse empties is retired onto another member of its orbit, keeping the
quotient vertex and the feature/lineage records of its quotient edges. Seam
edge orbits flip across the seam. When the route stalls it refuses with
`COMPLIANCE_FAILED`: the message reports `status=stalled`, and the
`MeshingFailureEvidence` holds each remaining edge's metric length and refusal
reasons.

Exact periodic proofs use the quotient chart: a lifted vertex's exact position
is its root's exact position translated exactly by its deck winding, and the
binary64 carrier holds the canonical rounding of that translate. Orbit-locked
bisection on a non-dyadic lattice can create midpoints that are not binary64;
its target is then published on the lifted vertex-indexed P1 carrier with an
`ExactPlcCellGeometrySource` bank of the exact midpoints, which the target's
`PeriodicMeshTopology` accepts as its exact coordinate witness. Certify or
distribute the adaptation result itself: re-certifying the bare target mesh
drops that bank.

Every owner-local periodic view retains the exact `specification_id`, including
nonadaptive periodic-power views. Cold reattachment requires the archived typed
surface/volume request to match; a map/context fingerprint cannot substitute for
the request identity.

For owner-local periodic Delaunay, passing `graph_partition_plan=plan` with the
execution group applies canonical graph labels before metric adaptation. Exact
cell, vertex, and cumulative interpolation-ancestry rows move through sparse
owner-pair exchanges; quotient-face, volume, geometry, and directory authority
are rebuilt before the assembly is accepted. The same graph evidence can label
an already accepted `OwnerLocalCellGeometry`; the provider then repartitions
the current local view and binds the renewed part/directory without changing
its global topology or geometry. Periodic power repartition moves ownership of
the exact authored site/weight bank, including inactive sites, and rebuilds the
same exact source-bound map rather than moving or reseeding sites.

The exact periodic-power context retains whether weights were authored; an
absent weight bank and an explicit all-zero bank are distinct source identities.

Periodic graph partitioning uses positive int64 work units per owned cell, not
physical cell measures; exact measures remain separate geometry evidence.

Cold replay keeps authored `source_value` distinct from the support-expanded
native source or exact power map. It validates authored site rows against the
current owner bank, then recomputes the sharded source-content root used by part
and candidate identities; lifecycle entry record IDs are not carrier identities.

Cell- and composite-entity field/history rows can be migrated with
`OwnerLocalGraphMigrationEvidence.transfer_keyed_values(...)`, supplying the
typed source entry, fixed-width integer source keys and owner ranks, source
values, target keys and owner ranks, execution group, target structure ID, and
key identity. Source and target keys must form an exact bijection; only
owner-local rows are routed. The resulting `OwnerLocalGraphKeyedStateTransfer`
preserves target row order and exact value bits. Its
`gather_target_local_values(...)` fetches local/ghost rows in request order,
including repeated keys and typed PRNG keys. Empty owner requests retain their
empty row shape. Its `composition_transport` carries partition-independent
fixed-bucket source/target content roots, not floating-point hash words or
numeric tolerances, and must be committed through the existing
`CompositionRebind` transaction. These routes do not gather a global mesh,
owner table, or field bank.
`value_fiber_shape` can declare that logical row tail from its accepted layout;
the empty tuple denotes one scalar fiber, and zero-row shards preserve the
declared shape during migration.

`OwnerLocalCellGeometry.reconstruct_local_geometry()` returns the canonical
local `(CellMesh, CellGeometrySpec)` pair without duplicating periodic lifted
vertices per cell. An empty owner shard returns `(None, None)`; a nonempty
power shard retains its exact source map rather than falling back to affine
geometry. `source_mesh_id` remains the authored source/lattice scope identity,
while global topology/geometry IDs describe the current accepted target and
`local_mesh_id`, `local_topology_id`, and `local_geometry_id` identify that
rank's reconstructed endpoint.

Nonempty periodic Delaunay views retain a cumulative exact
`VertexInterpolationStencil` from authored site IDs plus exact lattice offsets
for each target vertex. The materializer and cold replay evaluate that stencil
against the current exact root bank; rounded coordinates are only checked
execution shadows, never the ancestry carrier.

Periodic orbit-coordinate verification uses exact numeric equality: signed-zero
bits remain in the stored arrays/fingerprints, while every nonzero one-ULP
mismatch is refused rather than pruned by a geometric tolerance.

Global periodic topology/geometry identities are independent of process
placement: ranks route stable entity-key and record hashes to deterministic
logical buckets rather than gathering a global mesh or owner table. Fresh
owner-local restore rebuilds the exact point/image context and native support,
then re-proves quotient faces/CSR and recomputes those identities before
publishing the local view.
Periodic topology identity records quotient cell/source-key incidence and the
declared lattice/generator, not current coordinates or measures. The current
geometry/part authority separately binds exact vertex coordinates, site and
weight content, source-root/winding correspondence, and quotient-face coverage.
The authored source entry retains site IDs and declared source/lattice semantics;
coordinate-only motion with the same authored revision keeps that entry identity
while advancing current geometry. Cold restoration recomputes current source
content from the exact banks and stored execution-bit patterns before rebinding.

In serial periodic Delaunay, size controls are uniform source-vertex scopes;
metric edits use exact interpolation ancestry and are committed only after
whole-orbit closure. `periodic_orbit_closure` and
`stage_periodic_topology_edit` are the explicit APIs for orbit-complete
periodic topology transitions.

### Periodic quotient embedding proofs

`certify_global_embedding` proves exact degree one for closed affine 2D
periodic triangle quotients, including skew lattices; a folded map is refused
as invalid. For curved 3D periodic tetrahedra, the mapped route replays exact
reference-face maps under periodic actions, checks the closed oriented quotient
and local injectivity, and compares the mapped pullback volume with the lattice
covolume. A periodic restriction premise retains the source proof and exact
refinement transition, then certifies parent-volume partitions and boundary
subdivision; the certificate, including its premise, round-trips through the
model archive. Mapped restriction separately proves an exact common subcomplex.

Affine periodic quotients may be partially periodic or rotational: a
translational lattice of any rank up to the ambient dimension (for example, a
z-periodic channel in 3D or an x-periodic strip in 2D) or a commuting proper
`PeriodicIsometryGroup`, such as a quarter-turn wedge, including vertices on a
fixed rotation axis. The exact image frame enumerates group actions only along
periodic directions, every lifted triangle or tetrahedron is checked against
each overlapping image, and the nonperiodic directions are closed by the
quotient's oriented physical boundary. A mesh that wraps the quotient twice
overlaps its own image and is refused as violated.

Quotient cells may be convex vertex-defined polyhedra, as produced by the
restricted-power polyhedral fill, and a single cell may span a whole period or
touch a fixed rotation axis. The quotient is then a cell complex rather than a
simplicial one: an edge may be a loop whose two ends lie in one vertex orbit,
and a face may traverse an edge twice (the folded cap at a rotation axis). The
boundary check therefore reads each boundary face's corners from its lifted
representative and requires every boundary vertex to join all of its
boundary edge ends (an edge plus its start or finish) into one cycle. Pinched
quotients, such as two columns touching along one edge, are left unresolved.

Reusable periodic image frames retain lifted-source roots, shifts, and group
actions for translational lattices and isometry groups. `PeriodicIsometryGroup`
admits exact screw generators whose finite linear-order power is a nonzero
translation; orientation-reversing generators such as glide reflections are
refused because their quotient cannot carry consistently oriented cells.
Generic periodic power preparation uses exact representability/refusal
semantics rather than an arbitrary image-count or work cap.

### Gmsh sizing, protection, generation, and remeshing

`ProximitySizeControl` lowers to Gmsh distance fields between its two exact
B-Rep scopes: the requested size is the local gap (sum of distances to both
walls) divided by `elements_per_gap`, clamped to the control's bounds and
combined with every other scalar request by minimum. With
`opposite_normals_only`, the closest chord must be normal to both scopes.
Compliance measures every edge against the exact CAD gap at its midpoint.

`BackgroundMetricControl` binds a `MeshMetricField` to every vertex of an
explicit affine simplex background mesh in the source coordinate contract and
lowers it to a Gmsh post-processing view. `ISOTROPIC` uses the smallest
directional size; `ANISOTROPIC` passes the full tensor and is the sole size
field for the BAMG surface algorithm (anisotropic volumes belong to
`MmgProvider`). Compliance reports every edge's metric length as Gmsh
interpolates the view.

`ProtectedFeature` scopes on B-Rep corners and curves (and bounded faces in
volumes) are certified present in the canonical mesh with measured CAD
deviation. Free B-Rep curves and points are embedded into the unique host face
or solid containing them; unprotected free entities are discarded.

`GmshOptions` selects `GmshSurfaceAlgorithm`/`GmshVolumeAlgorithm` (including
`HXT`), sets the positive provider-level `num_threads` for `General.NumThreads`
(default one), selects Netgen optimization, and sets
`GmshHighOrderOptimization` for geometry orders two through four. Multi-threaded
runs report nondeterministic runtime; deterministic specifications preflight-
require one thread, while nondeterministic specifications may request more.
Order-three and order-four simplex maps are
resampled exactly at the canonical Lagrange nodes with shared entity nodes.
Compliance records Gmsh's native `minSICN`/`minSIGE` element quality.

`SurfaceRemeshingSpec` on a `SurfaceModel` (or a `CellMesh` with an explicit
coordinate contract) classifies sharp features and boundary curves per
`SurfaceReconstructionControl`, reparametrizes the discrete patches, and
remeshes them. Compliance bounds the two-sided vertex deviation and requires the
Euler characteristic, component count, and boundary loops to be preserved.

GmshSession caches CAD imports by source owner, revision, and coordinate contract.
Cache misses export the supplied model to native BRep text and require an exact
`source_revision` round trip before any Gmsh import. Unscoped plans import the
private snapshot directly; scoped plans use per-definition exporter provenance
and Gmsh's explicit OCC fragment correspondence. Unresolved or many-to-one maps
refuse. Cache hits verify the snapshot digest; replacing the original path cannot
mutate the immutable model bound to a meshing plan.

## Interchange

`MeshArrayArtifact` is the neutral host interchange representation.
`export_mesh_array_artifact` and `import_cell_mesh(artifact)` preserve native
block names, per-degree persistent IDs, geometry-node/vertex ordering,
attributes and units, zones/labels, coordinate contracts, and source revisions.
Pass explicit geometry-node IDs to preserve non-vertex node identities.

External `export_cell_mesh` uses meshio formats, which cannot encode the full
native contract. It requires `allow_lossy=True` and enumerates missing metadata
and any codec-level data losses; numerical fields are checked by reading the
written representation. A successful file write alone is not a lossless round
trip.

Shared reference-node ordering lives in the discretization layer. External
connectivity order must be converted there rather than duplicated in solvers.

## Adaptation and optimization

```python
refined = phx.meshing.execute_mesh_adaptation(
    phx.meshing.prepare_mesh_adaptation(
        result,
        phx.meshing.MarkedMeshAdaptation(np.asarray(result.mesh.blocks[0].global_ids)),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
        ),
    )
)
values = refined.transfer.apply(result.mesh.coordinates[:, 0])
coarsened = phx.meshing.execute_mesh_adaptation(
    phx.meshing.prepare_mesh_adaptation(
        refined.target,
        phx.meshing.MarkedMeshAdaptation(
            (), refined.target.mesh.blocks[0].global_ids, hierarchy=refined.hierarchy
        ),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
        ),
    )
)
```

`prepare_mesh_adaptation(source, request, policy=...)` binds one typed request
to one certified `CellMeshingResult` under one explicit `MeshAdaptationRoute`;
`execute_mesh_adaptation` runs exactly that route and never falls back to
another. Requests are `MarkedMeshAdaptation` (cells to refine and to coarsen by
global ID, plus the `hierarchy` returned by the previous adaptation),
`MetricMeshAdaptation`, and `RelocationMeshAdaptation` (a `MeshMetricField`
bound to the source vertices). Routes:

- `NATIVE_BISECTION`: Maubach/Stevenson tagged-simplex bisection of triangle
  (newest-vertex bisection) and tetrahedron meshes. The initial refinement edge
  is the longest edge (ties by global IDs); an incompatible initial labelling is
  rejected unless `BisectionCompatibility.UNIFORM_REFINEMENT` explicitly requests
  one barycentric compatibility refinement. That subdivision is recorded in the
  `BisectionHierarchy` as a refinement level (each source cell, its exact block
  row, its (d+1)! children, and its created edge, face, and center vertices). The
  conformity closure is vectorized per iteration; coarsening removes complete
  bisection patches (Chen–Zhang vertex removal) and then every subdivision whose
  children are all marked and whose created vertices' stars are undone together
  (whole edge-connected components), restoring the recorded cell, edge, and face
  IDs, so a refine/coarsen round trip reproduces the source topology exactly,
  including an incompatible source. Restored incompatible cells carry tag 0;
  refining such a hierarchy subdivides all of them again under
  `UNIFORM_REFINEMENT` (device bisection does so at preparation) and rejects
  otherwise.
- `NATIVE_METRIC_2D`: planar triangle metric adaptation by edge split
  (metric length above sqrt(2)), collapse (below 1/sqrt(2); a collapse that
  removes or keeps a vertex born in the run must not create an edge above
  sqrt(2)), flip, and relocation, with deterministic independent-cavity
  selection. A flip is accepted when it raises the minimum quality of its two
  cells, or when it strictly lowers the quantized `|ln L|` deviation of its
  diagonal without lowering their quality pair; a relocation likewise when it
  raises its star's minimum quality or keeps it and strictly improves the
  worst-first quantized deviations of its edges. Neither rule creates vertices,
  so each phase terminates. Orientation decisions use the policy's predicate
  mode; unresolved filtered signs reject the operation. The status is COMPLETE
  when the unit-mesh criterion holds, STALLED or NUMERICAL_STOP otherwise; a
  stalled `LocalMetricEvidence.stalled_edges` lists every remaining edge (one per
  quotient orbit) with its metric length and refusal reasons, for example
  `collapse=hysteresis+inadmissible;flip=invalid`. `RelocationMeshAdaptation`
  moves vertices only.
- `DEVICE_BISECTION`: the marked bisection request executed as compiled
  fixed-capacity device passes (below); it commits the meshes, IDs, lineage,
  and hierarchy of `NATIVE_BISECTION` byte for byte. The device passes coarsen
  bisection families; the commit then removes the recorded barycentric
  subdivisions they expose by the host rule.
- `DEVICE_METRIC_2D`: the planar metric and relocation requests executed as
  compiled device passes with FILTERED_DEVICE predicates (below).
- `MMG`, `OMEGA_H`: the configured provider's metric adaptation; the lineage
  records every entity as created/deleted and no vertex stencil is claimed. With
  an `association_transfer` the providers re-derive the B-Rep associations on the
  adapted target (see below); without one Mmg records the dropped associations as
  an adapter loss and Omega_h rejects the source.
- `HP`: tensor-product h-refinement or coarsening of a `FiniteElementHPEpoch`,
  with lineage projected by `project_hp_lineage`.

### Embedded-surface metric adaptation

`NATIVE_METRIC_SURFACE` adapts triangular two-dimensional meshes embedded in
three-space using the retained source-chart UV coordinates. It pulls the full
ambient SPD tensor back as `J.T @ M @ J`; anisotropy is never reduced to a scalar
size. The route reuses the native local split, collapse, flip, and relocation
operations, evaluating proposal vertices on the fixed source chart. Regular
periodic UV seams use an exact quotient so every lifted edit is staged as one
orbit; a singular periodic direction, such as the polar axis of a sphere chart,
is instead a fixed chart cut whose seam edges are protected. Every target vertex
keeps an exact chart point on the source face shared by its parent cells, so a
split of a source-cell interface stays on that exact edge.

The source must retain either `SurfaceSourceCharts` for a `MeshingDomain` or an
analytic implicit `SourceBoundaryChartWitness` together with its authoring
profile and `ImplicitAssociationTransfer`. Target cells retain exact chart
coordinates and source-cell reference maps. A cell that crosses a source chart
or implicit root cell without a witnessed transition is refused. Target geometry
is reconstructed under the retained source-fidelity displacement bound and an
endpoint continuity check; target charts and source associations are rebound
before certification.
`COMPLETE`, `STALLED`, and `NUMERICAL_STOP` remain distinct; an unresolved
predicate or an unmet unit-mesh criterion is not reported as success.

Protected scopes (including periodic boundaries) are never split, moved, or
removed; marks whose closure would split them are rejected and reported, with
status PARTIAL. Patches, zones, and labels are inherited through the lineage.
`MeshAdaptationResult` carries the certified target, `CellMeshTransition`,
complete `MeshLineage`, `VertexInterpolationStencil`, sparse P1
`FiniteElementTopologyTransfer`, route evidence (`BisectionEvidence`,
`LocalMetricEvidence`, or the provider result), compliance, and an optional
`MeshDistributionTransition`. `VertexInterpolationStencil.apply` resolves source
IDs once and applies one sparse gather; `as_transfer` certifies linear
reproduction and conservation claims. A `FiniteElementTopologyTransaction`
consumes the `MeshAdaptationResult`; rejection preserves the accepted state.
Unknown parentage remains unknown, never guessed from new positions; B-Rep
associations are re-derived by classification, not inherited.

When the input retains a computed source `certification`, native bisection and planar metric transitions rebuild and revalidate that acceptance on the target. Exact cell maps bind inherited geometry, regions are assigned from actual lineage, and chart/feature witnesses are carried only where source ancestry is known; a missing or ambiguous source map refuses the transition instead of retaining a stale witness or inventing a parent. Source domain, revision, and organization objects remain attached to an accepted target. Audit-only inputs keep the audit-only contract.

Layer-curving evidence is also endpoint-bound. A transition without the layer
owner’s exact source-chart rebinding refuses rather than dropping the evidence
or copying stale projection and layer-column records.

A material chain starts only from an explicitly authored accepted configuration
using `declare_material_reference` and
`MaterialReferencePolicy.ACCEPTED_CONFIGURATION`. Pass that declaration as
`material_reference=` to `prepare_mesh_adaptation` to seed a new chain. Preparation
revalidates its mesh, geometry, coverage, source map, material rows, and frame;
the declaration participates in the prepared identity. Do not supply a new seed
alongside a prior transition that already retains a reference or chain.

`material_source_map_from_cell_association` constructs the source-map capability
from a complete, uniquely resolved volume-cell `REGION` association and certified
domain coverage. CAD, PLC, mapped-root, and compartment identities remain distinct
from the coverage's material-region identifiers. The helper does not declare a
material reference or infer one from proximity.

An Eulerian overlap, vertex stencil, or new physical proximity does not establish
material identity; without an explicit reference the adaptation leaves the
material chain absent. Eulerian common-root overlap requires both endpoint maps
to be exact restrictions of the same physical root map. Material-reference
overlap instead retains separate reference, source-physical, and target-physical
Jacobian measures; these are not interchangeable with physical intersections.

For continuation, pass the exact accepted `CellMeshTransition` as
`prior_transition=` to `prepare_mesh_adaptation`. Its target must be the current
source result and geometry. Renewal uses the last deformation's exact material
charts (or the declared reference's identity charts for a seed), fresh endpoint
coverage, and target-bound region/occurrence rows to build a new paired-Jacobian
map. Exact nested coarsening composes the full chart expressions; stale or
incomplete ancestry refuses rather than reusing old overlap evidence.

`FiniteElementFieldTransferSemantics` selects the transfer independently for
each accepted field. Material-following extensive content uses a reusable
`MaterialChainSchedule` and exponent-scaled `material_chain_apply` action;
Eulerian averages use physical common-refinement overlap; compatible nested
forms keep their exact reference-map and Piola transfer; and explicitly
nonconservative fields use a vertex stencil or their declared interpolation
action. Configure `material_field_transfer`, `eulerian_field_transfer`,
`nested_field_transfer`, and `nonconservative_field_transfer` independently on
`FiniteElementTopologyTransaction`. If one required action or its coverage is
missing, the accepted state remains unchanged for the whole transition; other
fields are not zero-filled or silently switched to another route.

`material_csr_rows` is an explicit, policy-bounded material-action view; ordinary
Eulerian remap keeps its own physical-overlap CSR. `MappedNestedRemapEvidence`
binds a physical coverage report to an exact geometry-transition witness, while
`MappedSurfaceChartRemapEvidence` retains chart deformation, carried contents,
target measures, and coverage. `rational_geometry_lift` builds a
`CellGeometrySpec` from exact parent-relative rational CSR coordinate actions;
the binary64 coordinates are only the checked execution representatives.

Periodic bisection can be restricted to a selected orbit by binding a
`PeriodicRefinementEdgeLock` with `prepare_periodic_refinement_edge_lock` and
passing it on `MarkedMeshAdaptation(periodic_edge_lock=...)`. The selected edge
and its mate are split as one operation; orbit, winding, and lifted endpoint
identity are checked before the target is certified.

Surface-material charts coarsen only when exact ancestry preserves their source
root and piece. Source-boundary chart pieces compose through exact fine-to-parent
reference maps. When a restored parent entity is covered by its children's
pieces of one source chart, the pieces must agree on one exact affine parent
chart and are published as that single row; missing or ambiguous lineage refuses
rather than merging source chart, entity, or occurrence identities.

Source-boundary chart witnesses retain exact rational reference/source-parameter
tokens; their float arrays are checked realizations, not ancestry authority.
Polyhedral volume faces bind actual target triangle vertex IDs in the standard
two-dimensional face chart. Identity-lineage renewal carries those rows forward
and revalidates the canonical target triangulation; changed IDs without an exact
face/vertex map refuse.

`TargetMatrixOptimizationPlan` and `optimize_cell_mesh` optimize fixed-topology
triangle, tetrahedron, quadrilateral, hexahedron, prism, pyramid, and polygon corner
Jacobians, plus the face-centroid star simplices of polyhedra, against
target matrices with `phydrax.optim.minimize` (`ProjectedLBFGS` by default,
`NewtonTrustRegion` optionally). `MeshQualityObjective` selects the generalized
Knupp shape measure, shape plus size, metric alignment with a `MeshMetricField`,
or the Frobenius Gram-determinant energy; every objective rejects inverted
corners. Fixed vertices are eliminated from the parameters and stay
bit-identical; coordinate boxes are enforced by projection. Inverted input runs
the explicit Escobar untangling stages of `MeshUntanglingPolicy`; a stage is
accepted only when the inversion count reaches zero, the audit passes, and its
minimization converged. The optimized iterate is checked for inversions and
audited independently of the native termination status, and non-convergence is
never reported as optimized: a valid iterate is `OPTIMIZED` only when the
minimization converged, `VALID_NONCONVERGED` (accepted, certified) when it did
not and the plan sets `accept_valid_nonconverged=True` (which also admits
non-converged untangling stages), and `NONCONVERGED` (not accepted) otherwise.
`MeshOptimizationResult.result` is present exactly when `accepted`; every other
status returns the unmodified coordinates, and
`MeshOptimizationResult.minimization` carries the native termination evidence,
including the rejected iterate. `optimize_cell_geometry_coordinates` runs the
same route for user-defined high-order objectives; the objective must encode the
required curved-element validity, and `CellGeometryOptimizationResult` reports
`optimizer_status`/`converged` for the caller's explicit acceptance decision.
Derivatives of numerical objectives do not differentiate topology changes or
host-side acceptance.

`MeshMotionMonitor` assesses a moved fixed-topology coordinate proposal against its
reference mesh: the certified Bernstein lower bound of every cell Jacobian relative to
the reference certificate, per-cell mean-ratio degradation, displacement in units of
the smallest reference cell, and the caller's boundary residual.
`MeshMotionMonitorPolicy` thresholds yield a deterministic `MeshMotionDecision`
(`ACCEPT_MOTION`, `RELOCATE`, `REMESH`, or `REJECT`); uncertified cells always request
relocation first because an inverted mesh cannot seed a remesh.
`advance_mesh_motion` certifies accepted motion, relocates or untangles the free
vertices through `optimize_cell_mesh` toward the reference shapes on any crossing
(bounded by `MeshMotionMonitorPolicy.relocation_termination`; a relocation that
does not converge fails unless `accept_valid_nonconverged_relocation=True` admits
its valid iterate), and
escalates a still-failing proposal to a `MetricMeshAdaptation` (default metric: the
isotropic reference cell size) through the explicit `MeshAdaptationPolicy` route;
without a policy the advance returns the unexecuted request (candidate mesh and
`remesh_metric`, `accepted` false). The returned `MeshMotionAdvance` carries every
assessment and, for executed remeshes, the `CellMeshTransition` and adaptation
transfer consumed by the solver transaction.

Each `MeshMotionAdvance` binds its exact `source_mesh_id` and `source_result_id`
and retains the source-to-motion `CellGeometryTransition`; a remesh additionally
retains its motion-candidate-to-target `CellMeshTransition`. The vertex-coordinate
proposal cannot define new high-order control points, so non-P1 source maps are
rejected rather than silently replaced with affine geometry.
Passing `prior_transition=` to `advance_mesh_motion` requires its target to be the
exact source result. When it carries a material reference, the advance retains a
source-to-motion `material_transition`; an executed remesh keeps the distinct
motion-candidate-to-target transition and continues the same reference. Neither
stage infers material identity from the moved coordinates.

When the input retains a computed source certification, an accepted fixed-topology
move requires a fresh target-bound report: its original domain and revision,
regions, and any retained geometry-fidelity, source-coverage, and global-embedding
claims are checked against the moved geometry. A boundary move that no longer
satisfies those claims is rejected with its failed report; the immutable accepted
result remains available for retry. Audit-only results keep the monitor/audit
contract without acquiring source claims.

An adapted mesh reaches a running coupled problem only through one accepted
cross-owner rebind (`phx.lifecycle.CompositionRebind`). A certified conservative
`FiniteElementTopologyTransfer` (the adaptation transfer, or
`vertex_interpolation_transfer` with P1 coordinates and basis integrals) becomes a
topology-epoch transition with `transfer.epoch_transition(source_field,
target_field, source_epoch, target_epoch, source_measures, target_measures)`; its
claims come from the transfer's own certification. The transition's
`composition_transport(source_entry, target_entry)` reports the conserved content
and succeeds only if the staged field is the transfer image, so observations,
factorizations, and interface routes bound to the old topology must be
reprepared or the rebind refuses them (see the lifecycle section of the
numerical interoperability guide and `examples/adaptive_fe_fv_rebind.py`).

## Device adaptation epochs

```python
import jax.numpy as jnp


def near(mesh, center, radius):
    centroids = jnp.mean(mesh.coordinates[mesh.cells], axis=1)
    return jnp.linalg.norm(centroids - jnp.asarray(center), axis=1) < radius


policy = phx.meshing.MeshAdaptationPolicy(
    phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
    device_policy=phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=1 << 12, cell_capacity=1 << 13
    ),
)
prepared = phx.meshing.prepare_adaptive_simplex(result, policy=policy)
layout, state = prepared.layout, prepared.state
for center in ((0.2, 0.2), (0.25, 0.25), (0.3, 0.3), (0.35, 0.35)):
    marks = near(state.mesh, center, 0.3) & state.mesh.cell_active
    state = phx.discretization.refine_adaptive_simplex(layout, state, marks).state
    coarse = ~near(state.mesh, center, 0.45) & state.mesh.cell_active
    state = phx.discretization.coarsen_adaptive_simplex(layout, state, coarse).state
adapted = phx.meshing.commit_adaptive_simplex(prepared, state)
assert adapted.target.audit.passed
```

`prepare_adaptive_simplex(source, policy=..., hierarchy=None)` validates one
certified triangle or tetrahedron source on the host, computes the same
Maubach labels as `NATIVE_BISECTION` (or binds the supplied
`BisectionHierarchy`), resolves protection and organization classes, and pads
the epoch into an `AdaptiveSimplexState` of the policy's capacity bucket.
`AdaptiveSimplexLayout` is the static compile identity (dimension, ambient
dimension, vertex/cell/protected-edge capacities, closure and coarsening
bounds, precision); the source topology never enters it, so every epoch and
cycle in one bucket reuses one executable per entry point.

The state's `MaskedSimplexMesh` is the solver-visible layout: vertex and cell
slots with activity masks, positively oriented cell rows, and packed sibling
half-facets. Slot order equals global-ID order and slots are never reused inside
an epoch. The compiled closure repeatedly bisects selected cells and cells
forced by split edges in a `lax.while_loop` until conformity is reached or a
real refusal occurs; `closure_iterations` records actual work and is not a
limit. Marks whose own closure would split a protected edge are found by one
reverse-reachability fixed point over the union closure and rejected, as on the
host. Coarsening likewise removes every unprotected bisection vertex whose star
is exactly the marked, class-uniform children of its bisections, restores the
parents under their original IDs, and continues to its natural fixed point.
Every call records its `AdaptiveSimplexStatus` flags in
`AdaptiveSimplexState.status_flags`, which accumulate over the prepared epoch.

Capacity overflow is a transient execution request: the host wrapper grows the
private layout to the reported vertex and cell requirements, resizes the
unchanged accepted state, and retries the identical update. Capacity requests
are not recorded as epoch status flags. Protected conflicts and certified
invalid geometry are terminal scientific failures: the call rolls back every
mesh, topology, and numerical array but records its terminal flags, and later
refine or coarsen calls on that state are refused with zero operation counts.
Children are checked with FILTERED_DEVICE orientation predicates; unresolved
signs set `NEEDS_HOST_RESOLUTION`, which remains applied for exact host
orientation certification at commit.

`commit_adaptive_simplex(prepared, state)` performs one device-to-host transfer
and decides acceptance from the cumulative scientific flags: protected conflict
raises `MeshingFailure` (INVALID_SPECIFICATION), invalid geometry is rejected
(QUALITY_REJECTED), and `NEEDS_HOST_RESOLUTION` requires every committed cell
to be certified POSITIVE by exact host orientation predicates. Capacity
overflow and closure/coarsening iteration counts are not terminal limits.
Coarsening that leaves marked families ineligible is reported as a valid
partial result, not a numerical stop. An accepted epoch runs the host edit
assembly: canonical `CellMesh`, organization inherited by exact IDs,
certification, `CellMeshTransition`, complete `MeshLineage`, sparse P1
transfer, `BisectionEvidence`, and the next `BisectionHierarchy`, returned as
the `MeshAdaptationResult` the solver transaction consumes. Cell lineage
follows bisection-tree containment, so cells coarsened and re-created inside one
epoch still relate to their source cells. The next epoch is prepared from
`adapted.target` and `adapted.hierarchy`; with explicit capacities it reuses
the compiled entry points. `execute_mesh_adaptation` with the
`DEVICE_BISECTION` route runs one refine-then-coarsen epoch of a
`MarkedMeshAdaptation` and commits it.

Multi-device epochs: `partition_adaptive_simplex(prepared)` splits the epoch by
the policy's `MeshDistribution` (space-filling-curve, graph, or provider
ownership) into stacked per-part states on an `AdaptiveSimplexParts` device
mesh. `refine_adaptive_simplex_parts` refines each part's owned cells inside
one `shard_map`; every closure round all-gathers the candidate refinement edges
and selected cell IDs, so shared-boundary splits select the neighbor part's
cells until the global fixed point and new IDs are global ranks, independent of
`commit_partitioned_adaptive_simplex` returns a `PartitionedAdaptiveSimplexResult`
of owner-local source/target meshes, exact lineage/stencils, cell measures,
owner ranks, and restart state; it does not gather a global candidate mesh or
produce a global `MeshAdaptationResult`. The collective commit validates every
part before publishing the result. Periodic source meshes are refused until an
exact quotient and whole-orbit closure witness is available.

The `DEVICE_METRIC_2D` route runs split, collapse, flip, and relocation passes of
the planar metric adaptation as compiled fixed-capacity device passes: fixed-width
cavities, a deterministic independent set of non-overlapping cavities per round,
and FILTERED_DEVICE orientation predicates. An operation whose predicate is
unresolved is never applied and the pass reports `NEEDS_HOST_RESOLUTION`; there
is no host callback inside the passes. The commit reuses the host metric edit
assembly, so the result carries the same lineage kinds and vertex stencil as
`NATIVE_METRIC_2D`.

`prepare_device_metric_adaptation(source, request, policy=...)` binds a
`MetricMeshAdaptation` or `RelocationMeshAdaptation` to a `DeviceMetricState`
of the policy's capacity bucket; `adapt_device_metric(layout, state)` runs up to
`maximum_passes` passes in one compiled executable per `DeviceMetricLayout` and
returns a `DeviceMetricReport` (`AdaptiveSimplexStatus` flags, operation and
rejection counts, unit-mesh measurements); capacity overflow restores the
exact accepted arrays, grows the required static workspace at a host barrier,
and retries the same operation. Invalid geometry is terminal and refuses
later calls on that state.
`commit_device_metric_adaptation` applies the same acceptance as
`commit_adaptive_simplex` (terminal flags raise, `NEEDS_HOST_RESOLUTION` is
resolved by exact host orientation of the committed cells), performs one
transfer, and returns the `MeshAdaptationResult` with `DeviceMetricEvidence`; a
run that applied nothing is UNCHANGED, with convergence or stall in the
evidence (as for `NATIVE_METRIC_2D`).

`NATIVE_METRIC_3D` is the independent serial reference for SPD-metric tetrahedral
split, collapse, flip, ring-reconnection, and relocation edits. `DEVICE_METRIC_3D`
is a hybrid host/device route: compiled evaluation uses the full tensor metric on
live edges, then stable-ID priority selects independent split cavities. The
owning host resolves every proposed child-cell orientation exactly as one batch
barrier before the native edit owner mutates its provisional state. An invalid
batch discards provisional edits and returns the unchanged source with
`NUMERICAL_STOP`; accepted edits retain the serial route's metric and lineage
semantics. `DeviceTetraMetricEvidence` exposes candidate ordering, exact-predicate
and host-barrier counts, the native metric evidence, and whether edits committed.


### Stable-ID forest migration and restart

`extract_simplex_forest` converts the allocated cell and vertex slots in an
`AdaptiveSimplexState` to stable scientific IDs, including parent/child links,
split vertices, cell and vertex history, and protected edges. Slot numbers remain
local storage details. `validate_simplex_forest` rechecks the full link closure;
`reconstruct_adaptive_simplex_state` rebuilds a capacity-bucketed state from
those ID-keyed records while retaining the supplied epoch clocks.

For owner-local execution, wrap the forest and any cell- or vertex-aligned field
rows in `OwnerLocalMigrationState`. `prepare_ownerlocal_migration` stages cell,
vertex, protected-edge, and field packets for current target owner ranks. It
validates the exact local target ID inventories and collective content identity
before returning a `PreparedOwnerLocalMigration`; the source remains unchanged
until every owner calls `accept_ownerlocal_migration`.

`repack_simplex_restart` restores a saved owner-local forest on a changed
execution group. It requires the complete set of archived
`ShardedOwnerDirectoryData` shards plus the new current-owner row banks; stored
shard digests alone cannot reconstruct the saved global identity set. The
repacker reassigns stable-ID records, rebuilds hash-sharded owner directories,
and verifies that forest, field, and history content is unchanged before it
returns a `SimplexRestartRepack`.

`LogicalMigrationParts(execution_group, saved_part_count)` keeps the historical
rank axis separate from current physical devices. `partition_map` executes the
same rank-local kernel and its logical routes on the current group, even when
the saved rank count differs from the device count. Device handles and physical
placement are not persisted. The real-process regression in
`tests/unit/meshing/test_distribution_migration.py` exercises refinement on
three owners, migration and restart repacking onto two and four owners, logical
rank replay, and coarsening back to the original roots; run it with `pytest -n 0`.


## CAD association and high-order curving

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere

from phydrax.interchange import model_from_occt_shape

shape = BRepPrimAPI_MakeSphere(1.0).Shape()
model = model_from_occt_shape(
    shape, coordinate_contract=phx.SpatialCoordinateContract.si()
)
# An octahedron inscribed in the unit sphere, turned off its seam and poles.
c, s = np.cos(0.3), np.sin(0.3)
turn = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
tilt = np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
mesh = phx.discretization.CellMesh.from_triangles(
    np.concatenate((np.eye(3), -np.eye(3))) @ (tilt @ turn).T,
    np.asarray(
        ((0, 1, 2), (1, 3, 2), (3, 4, 2), (4, 0, 2),
         (1, 0, 5), (3, 1, 5), (4, 3, 5), (0, 4, 5)),
        dtype=np.int32,
    ),
)
projection = phx.geometry.prepare_brep_projection(model)
policy = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=policy)
curved = phx.meshing.curve_cell_mesh(
    mesh,
    association,
    projection,
    policy=phx.meshing.HighOrderCurvingPolicy(degree=2),
)
assert curved.status is phx.meshing.HighOrderCurvingStatus.CURVED
```

A B-Rep `GeometryAssociation` classifies mesh entities against one source owner
and exact revision. Each row retains the definition dimension and index
(`source_dimensions`, `source_indices`), exact `source_occurrence_paths`,
closest-point `parameters`, projection residual, ambiguity, and orientation
relative to the source tangent or oriented normal. Source entity keys bind the
owner, revision, definition, and occurrence; they are opaque identities, not
parseable revision/index strings. Two placed uses of the same definition remain
distinct. Unclassified rows also retain their owner and any known occurrence,
but do not claim a resolved source stratum. Parent provenance is recorded in
`parent_dimensions`, `parent_ids`, `parent_association_id`, and
`GeometryAssociationProvenance`.
`associate_mesh_vertices` classifies each vertex geometrically: mesh-boundary
vertices on the lowest-dimensional B-Rep entity within the classification
tolerance, interior vertices of volume meshes in their solid.
`associate_mesh_entities` derives edge, face, and cell classes from the vertex
classes: a cell lies in the unique region containing its vertex classes, an
entity interior to one region inherits it, and boundary or interface entities lie
on the lowest B-Rep entity containing their vertex classes within the closure of
every adjacent class (ties broken by the centroid distance, otherwise ambiguous).

`propagate_association` carries a vertex association through an exact
`MeshLineage`: preserved vertices (B-Rep corners included) keep their class,
relocated vertices are re-projected, children inherit the class of the source
entity they were created on (edge children inherit the edge, face-interior
children the face) and are projected, swaps change no vertex class, and a collapse
is legal only when the kept vertex's class lies in the closure of the removed
vertex's class; an illegal collapse raises `AssociationPropagationError`.
Coordinates are not snapped: residuals (e.g. the sagitta of a split chord) are the
evidence. After provider remeshing with unknown lineage, `rederive_association`
classifies target boundary vertices only on the B-Rep classes (with closures) of
source boundary facets carrying the same facet organization (the patches, zones,
and labels Mmg references and Omega_h class IDs preserve), then projects them.
`BRepAssociationTransfer` binds a projection and policy for
`MeshAdaptationPolicy.association_transfer` and the Mmg/Omega_h providers: native
routes fix B-Rep corners, protect ambiguously classified edges (ambiguity rejects
the operation), separate cell, facet, and planar feature classes by B-Rep entity,
and certify the target with the propagated associations.

`curve_cell_mesh` creates Pk simplex (P2, P3) or Qk tensor geometry nodes in the
reference ordering of the discretization's Lagrange elements, shared between cells
by owning entity and position (independent of local orientation), classifies each
node through its owning mesh entity, and projects nodes on B-Rep entities of lower
dimension than the ambient space. Interior nodes relax through
`optimize_cell_geometry_coordinates`; the objective combines the Escobar-regularized
inverse mean-ratio distortion of the Jacobian relative to the straight-sided
element, sampled at the Bernstein control-point lattice of the determinant, a
displacement term, and the distance of constrained nodes from their tangent
spaces. Constrained nodes are re-projected after every round. A candidate is
accepted only when `certify_cell_geometry_validity` certifies every cell, global
embedding is certified, and every constrained residual is within
`residual_tolerance`. Supplying `source_result=` to `curve_cell_mesh` (or
`verify_curved_geometry`) additionally renews and requires any retained source
certification on the curved target; sampled node residuals cannot substitute for
source-fidelity or coverage evidence. A source-chart transition is accepted only
when its exact chart ancestry can be rebound. The initial CAD projection is
accepted on its own when valid; a relaxation whose minimization did not converge
cannot replace the accepted geometry unless
`HighOrderCurvingPolicy(accept_valid_nonconverged_relaxation=True)` permits it.

Passing `prior_transition=` to `curve_cell_mesh` or `verify_curved_geometry`
binds the exact source result. A retained material reference produces an
edge-bound `CurvedGeometryEvidence.material_transition` with exact same-cell
reference charts and fresh endpoint coverage; a bare result never authors a
material reference.

`HighOrderCurvingResult.relaxation_statuses` records every relaxation's native
termination status and `accepted_round` names the accepted candidate (0 for the
projection). Without an accepted candidate the result rolls back to the straight
geometry with the rejected candidate's evidence (`HighOrderCurvingStatus`,
including `ROLLED_BACK_NONCONVERGED`). `PeriodicCoupling` vertex pairs make target-side
high-order nodes the declared isometry of their source-side nodes
(`PeriodicCoupling.match_points` pairs the nodes). `verify_curved_geometry`
certifies an existing high-order geometry, such as a Gmsh high-order output,
without moving it.

`curve_layer_core_construction(construction, layers, source_surface_charts, policy=...)`
builds a source-aware high-order mixed-cell map for a boundary-layer/core mesh.
Parametric sources retain exact chart ancestry and projection traces; adaptive
implicit sources require a source-bound fidelity query and root-chart witness.
Both routes certify the complete coordinate map, cell validity, global embedding,
and source fidelity. Missing or unresolved source evidence rolls back to the
straight construction; `require_layer_core_construction(result)` returns only a
fully accepted curved result and otherwise raises with the retained failure evidence.

## Polyhedral storage

Polyhedral topology uses packed incidence offsets/values. Exact-width
`PolyhedralBlock` groups coexist with standard cell blocks. Dense local worksets
are prepared from complete packed connectivity; padding and validity masks are
execution layout, not a caller-selected entry or byte ceiling. Structural,
index-representability, and real allocation failures remain explicit.

### Mixed and polyhedral adaptation

`NATIVE_MIXED` adapts three-dimensional tetrahedron, prism, pyramid, and
hexahedron blocks with marked refinement and sibling coarsening. Its closure
matches shared face signatures across family interfaces while retaining
region and layer classes. The route carries exact cell-reference ancestry into
the finalized target geometry, including high-order coordinate maps, and
exposes `MixedCellContentTransfer` for conservative transfer of extensive
per-cell quantities.

Marked cells keep their preferred family refinement; face closure then chooses
neighbor templates that match the refined shared faces, so a marked cell is
never downgraded to a partial split to spare its neighbors. Every
family-preserving template is an affine sub-cell of its parent, and target
charts are the exact affine child-to-parent maps (pyramids included), composed
with the source coordinate map. Coarsening restores a parent only when every
sibling chart reconstructs one exact parent map. A source without acceptance
inputs yields a target without a certification; whole-cell validity of the
restricted map is still checked.

Simplex coarsening of a piecewise-linear mesh keeps the restored parents affine
on their exact corners; refine/coarsen round trips through the bisection
hierarchy restore the original topology, coordinates, and global IDs exactly.
Adapting an already restricted (non-P1) map composes every target chart through
the source's retained root restriction, so restriction ancestry always names the
original root cells rather than an intermediate mesh. Non-P1 coarsening rebuilds
each restored parent's exact root chart from its children's root charts and the
inverse bisection maps; it refuses when the children name different root cells,
coordinate dofs, or charts.

`NATIVE_POLYHEDRAL` selects one explicit operation: `PLANE_SPLIT`,
`AGGLOMERATE`, or `REGENERATE`. Plane splits and agglomerations retain exact
cell lineage; site regeneration uses its authored successor sites and does not
infer parents from site IDs. Every accepted operation retains a physical common
refinement and `UnstructuredConservativeRemapPlan`, so nonnested regeneration
uses actual source/target overlaps rather than invented ancestry. Material and
region assignments must remain unambiguous through those overlaps.


## Assemblies, distribution, and coupling

`MeshPart` retains a certified cell carrier, prepared tensor grid, point cloud,
or spline carrier. `MeshAssembly` organizes named parts without tessellating
compact carriers. Part identities include numerical geometry: moving a mesh
invalidates old distribution and coupling evidence even when topology is fixed.

`CellPartition` is the solver-neutral cell ownership contract. `MeshDistribution`
adds exact global IDs, owned rows, ghost residence, dependency routes, and
`MeshPartitionEvidence` (part weights, imbalance, edge cut, ghost replicas).
Its default ghosts are the `halo_width`-layer face-adjacency reach of each part,
found by breadth-first search over CSR cell adjacency and ordered by global ID.
`lower_finite_element` produces native FE phase/workset plans whose worksets
are padded to the largest part, never a parts-by-cells table;
`lower_finite_volume` produces a grid-revision-bound Cartesian FV decomposition.
The old FEM-specific partition class is removed, without an alias.

`prepare_mesh_distribution(part, policy=MeshPartitionPolicy(kind, parts))`
selects the ownership route with `MeshPartitionKind`:

- `MORTON` and `HILBERT` quantize cell centroids isotropically, order cells by
  curve code then global ID, and split the order into weighted contiguous
  ranges; every part is non-empty and, when attainable, part weights stay
  within one cell weight of the mean, so
  `imbalance <= 1 + parts * max_weight / total_weight`.
- `GRAPH` runs METIS k-way partitioning of the cell graph through its C ABI,
  loaded from `PHYDRAX_METIS_LIBRARY` or the platform loader path; absence
  raises `MetisUnavailableError` rather than substituting another partitioner.
- `PROVIDER` takes ownership computed by a provider such as Omega_h or ParMmg.

`prepare_distribution_transition(source, target_part, lineage, policy=...)`
moves a distribution onto an adapted revision of the same named part. Preserved
and refined cells keep their parent's owner, merged cells the majority owner,
and created cells the majority owner of owned face neighbors. When that
ownership exceeds `maximum_imbalance`, the target is repartitioned by the policy
route and parts are renamed to the source ranks they overlap most, minimizing
migration. `MeshDistributionTransition` holds the target distribution with
rebuilt ghosts, `send`/`receive` sparse relations over rank-ordered message
slots, migration counts and volume, and `transfer`, which moves cell data
exactly as the migration would. Global IDs are never renumbered.

Conformal and periodic overlays validate node bijections. Periodic vector
traces include the isometry's rotation. `ContactCoupling` is frozen node-to-node
normal contact; `ContactCoupling.search` pairs each target node with its exact
BVH nearest source node inside a capture radius. `OversetCoupling` remains an
explicit interpolation overlay with its exact algebraic transpose; interpolation
is not a conservative remap.

`prepare_overset_connectivity` performs source-bound hole cutting and complete
donor search over affine, mapped, and authored polyhedral cells. `PreparedFieldQuery`
routes retain every containing-cell candidate and fail closed on incomplete
inverse maps, ambiguity, or orphan receptors. `PreparedOversetFieldTransfer`
applies the selected FE/FV reconstruction and its exact transpose. For owner-local
fields, all registered part-owner processes call
`transfer.prepare_owner_local_exchange(coefficient_ids, execution_group)`
collectively. Each donor owner re-searches all receptor points in its own full
mapped/polyhedral cells; candidate failures are synchronized before sparse field
coefficients and transpose contributions follow the execution-group halo plan.
Owners with no field rows still participate in search with an empty, valid halo
partition, and each process supplies only its owned donor/receptor arrays.

Moving connectivity consumes accepted, source-bound `MeshMotionAdvance` results
through `connectivity.moved_from_advances(...)`; it does not recertify arbitrary
vertex arrays or infer a curved coordinate-map update. The candidate rebuilds
blanking, donor search and field routes before `prepare_overset_motion_rebind`
stages its `CompositionRebind`. An incomplete/orphan candidate cannot publish,
and a rejected boundary leaves the accepted composition unchanged. Topology
changes retain their exact transition and caller-supplied field transports.

Conservative cell remapping is a separate physical-overlap operation:
`prepare_overset_conservative_remap(source, target)` requires complete exact
common-refinement coverage and exposes extensive/intensive actions and their
transposes. Curved endpoint maps require an explicit certified common-root
witness; no corner-linearization fallback is used. Failed coverage cannot be
applied or committed as accepted state.
`phydrax.solver.prepare_moving_overset_poisson_update(composition, previous,
advances, plans, "u")` performs serial full-dimensional P1-simplex FE reanalysis
for every overset part. It interpolates candidate receptor values, transports
the unit Poisson source through complete physical-overlap refinements, solves
`-Δu = 1`, and retains each part's overlap-content, linear-residual, physical
boundary-reaction, overset-reaction, and flux-balance evidence. Motion must keep
the FE topology and DOF identity; an orphan or incomplete overlap fails before
staging. `PreparedMovingOversetPoissonUpdate.commit(accepted_boundary=...)`
publishes all candidate connectivity, FE operators, and field states in one
`CompositionRebind`; a rejected boundary returns the original composition.
For owner-local assemblies, also pass `execution_group` and the accepted
`successor_assembly`. Each process holds only its owned cells plus their exact
face-neighbor halo for every part, and this completeness is checked
collectively. Each owner assembles P1 terms on its owned cells only. Owner-pair
routes refresh ghost rows and add contributions back into owner rows, and a
distributed PCG solves the owned free rows. Donor coefficients and receptor
values use the sparse owner-local exchange. Overlap evidence comes from
`prepare_owner_local_conservative_remap`. Commit synchronizes acceptance, so any
local rejection rolls the complete composition back. For topology-changing
local FE state, transfer the accepted state through
`FiniteElementTopologyTransaction.adapt_owner_local` before reanalysis; this
Poisson route requires unchanged P1 topology and DOF identity.

### Interface attachments

`MeshInterfaceAttachment(part, scope, association, geometry_entity_ids,
tolerance=..., region=...)` attaches exact entities of one named `MeshPart` to
declared authoritative geometry entities, for example the `BRepEntityId`
strings of one B-Rep edge or face. The scope may be a part-bound `MeshingScope`
or a `MeshPatch`, `MeshZone`, or `MeshLabel` certified on the part's carrier;
organization evidence is recorded in `organization_ids`. The witness is a
`GeometryAssociation` certified on the same carrier: every attached entity must
be resolved, unambiguous, classified on a declared entity within `tolerance`,
and every declared entity must be attached. An association of another carrier,
another entity set, or another geometry revision is refused; equal shapes or
names never substitute for it. The attachment does not classify anything
itself.

A sided attachment records which side of the authoritative oriented normal it
lies on. With `region=` (the cells of the attached side of a volume carrier),
every attached facet must bound exactly one region cell, and
`orientation = +1` means the region's outward normal equals the B-Rep face
normal (three dimensions) or the B-Rep edge tangent rotated clockwise,
`(t_y, -t_x)` (two dimensions); `-1` means it is opposed. A codimension-one
carrier (a surface mesh) declares instead with `carrier_side=InterfaceSide`
which side of its own cell orientation it represents. Without either, the
attachment is unsided. `require_current(part)` and
`MeshAssembly.require_attachment(attachment)` refuse a part whose revision
changed; moving or remeshing a part therefore requires a new attachment, while
redistributing its ownership does not. Attachments currently require a
certified cell carrier, since only cell results carry geometry associations.
See [Numerical interoperability](guides_numerical_interoperability.md#interface-bindings)
for the solver-level bindings that consume them.

## Learned proposals remain untrusted

Marking, size, metric, and coordinate proposals bind an exact certified source.
`project_mesh_proposal` applies deterministic bounds, protected-entity,
gradation, SPD, and displacement constraints. `prepare_mesh_proposal` invokes
trusted native refinement or optimization and returns a
`MeshProposalTransaction`. Safety audit and compliance are separate evidence;
rejection or rollback retains the accepted source.

The native size/metric proposal route performs one triangle-refinement step.
It does not claim anisotropic adaptation or that requested final sizes were
achieved. Use an appropriate external metric provider for those requests.
Mesh promotion does not implicitly transfer PDE fields: consume the exposed
transition/transfer through the solver topology transaction.

`MeshAdaptationObjective` supplies the physical stopping rule for h, p, geometry-
order, metric, and relocation alternatives. Each candidate binds its source and
target revisions, compiled layout, geometry/topology certificates, and accepted
state dispositions; the canonical `solve_checked` result is a separate numerical
admissibility gate. `PhysicalErrorEvidence` sums field, geometry, algebraic, and
transfer contributions without cancellation, and QoI targets require an explicit
well-posedness certificate. The objective chooses the lowest measured
preparation, compilation, adaptation, transfer, and reanalysis time among
candidates that improve the baseline and meet tolerance; time is a ranking cost,
not a wall-time, memory, or DOF admission ceiling. Before publication,
`decision.require_reanalysis` requires independent target-bound error evidence,
a successful checked solve, and matching `CompositionRebind` dispositions.
`FixedEpochDerivativeEvidence` binds geometry/PDE/transfer derivatives to stable
`ParameterId` and event identities; geometry-realization, topology, acceptance,
and branch-boundary events invalidate that fixed-route derivative.

`AbstractMeshProposer` is the neutral `DECISION` component slot
(`slot_semantic_id="phydrax.meshing.mesh-proposer"`). A proposer decides where
and how a mesh should adapt; it never produces a mesh. `LearnedMeshProposer`
maps one row per scoped entity to a marking score, vertex size, metric tensor,
or coordinate target. `propose(source, features)` requires a
`MeshProposalFeatures` value binding the exact source result and entity scope,
the feature owner, and ordered feature-column IDs. Raw feature arrays are
accepted only by differentiable `evaluate(features)` for training; same-shaped
rows cannot be replayed on another result revision. Proposal evaluation IDs bind
the feature record and model arrays. Native projection, safety audit, and
compliance only prepare a candidate; the adaptation objective still requires
the field transfer and fresh physical reanalysis before publication.
A model declaring ports binds through the proposer's `ports` and an
explicit `port_mapping`.

Topology proposals preserve revision-bound zones, patches, labels, and cell
regions through canonical lineage; the P1 vertex field transfer remains explicit
on the returned stencil. Implicit associations and boundary models transfer only
when `ImplicitAssociationTransfer` binds the exact retained profile or a
`NativeImplicitSource` with its authoring profile or adaptive discovery under
the matching source identity. Unsupported association families and source
attributes are refused rather than discarded.
Embedded-surface metric proposals use `NATIVE_METRIC_SURFACE` and require
source-bound charts or an exact implicit chart witness validated against its
original source query.

Supervised marking targets come from native estimators, reordered to the
proposal scope's sorted global IDs: `FiniteElementDWRIndicators.absolute` from
`phydrax.discretization.fem.local_dual_weighted_residual` for goal-oriented
finite-element refinement, and the `refine_mask` (or the per-channel
`indicators`) of the high-enthalpy AMR indicator evidence for
aerothermodynamic refinement. Size or metric targets are likewise native
projected fields; a learned proposer imitates them but is always re-certified.

## Native point-cloud reconstruction

`phydrax.geometry.reconstruct_surface_region` and
`phydrax.geometry.reconstruct_point_region` use Phydrax's native screened
Poisson finite-element solve, with an adaptive octree as the default
discretization. Missing normals are estimated by bounded-neighborhood PCA and
oriented by a minimum-spanning forest; callers may instead provide normals and
choose their orientation policy. `sample_spacing` sets the finest solve width.
This is a smoothed approximation and does not preserve features below that
resolution. The native solver uses this requested finest width without a caller-set
node quota or silent coarsening. Numerical nonconvergence, index-representation
failure, and actual allocator failure are reported with details in ReconstructionFailure.

The immutable `ReconstructionReport` records normal-orientation and solver
evidence, topology (including Euler characteristic), sampled two-sided
deviation, and sample-coverage evidence. A closed Poisson level set can extend
beyond the measured cloud: `repair_interpretation == "poisson_inferred_closure"`
marks unsupported output patches as model-inferred closure, not observed
surface. The `ReconstructionRobustness` policy (imported from
`phydrax.geometry.reconstruction`) with `incomplete_sampling="refuse"` rejects
a result while retaining the coverage evidence on `ReconstructionFailure`.
Coverage and deviation are sampled evidence, not continuous Hausdorff
certificates. `reconstruct_point_region` preserves the input digest and reports
ROI/voxel filtering; `reconstruct_lidar_region` also retains acquisition
lineage. Planar and DEM source construction uses native MeshCore predicates.
These native source routes do not import PyVista, Open3D, or Qhull.
The surface source of a reconstructed triangle complex is an exact
`MeshingDomain`: each triangle chart evaluates in barycentric form, so every chart
corner reproduces its authored vertex bit for bit, and shared corners are compared
with zero tolerance.

## Native unsigned-distance surface envelopes

`SurfaceEnvelopeSource` binds a retained `ReconstructedGeometrySource` or an
authored `AuthoredTriangleSheet` to the
native surface-envelope operator inputs `offset`, `spacing`, and
`certificate_spacing`, plus an explicit original-source `maximum_deviation`.
These distances are physical lengths in the declared Cartesian coordinate
contract. Source construction executes the native operator and retains its
output, persistent entity lineage, and replayable bounds; if the requested
original-source deviation cannot be certified at those controls, construction
refuses instead of silently relaxing the bound.

The envelope owns a distinct derived source identity/revision and an exact
`PiecewiseLinearDomain`. The `surface_envelope_tetrahedral` route fills that
derived domain using an ordinary volume specification. Its exact PWL chart
witness certifies the derived envelope boundary only. The separate
`SurfaceEnvelopeDistanceWitness` bounds the original reconstructed surface at
`maximum_deviation`; unsigned distance does not establish the original
surface's topology or solid interior, and the route does not claim exact-volume
equivalence to the original reconstruction.

An `AuthoredTriangleSheet` accepts an imperfect, possibly open triangle sheet:
boundary edges and holes, unwelded cracks (bitwise-coincident duplicate
vertices), inconsistent winding, exact duplicate faces, and several
edge-connected components. Its topology repair is geometry-preserving: it welds
only bitwise-coincident vertices, removes only faces over an identical welded
vertex set, and orients each component by propagation from its first face, so
the repaired triangles are exactly the authored point set. Boundaries stay open.
The repair is reported as `report.repair_interpretation ==
"authored_sheet_topology_repair"` with exact counts on
`TriangleSheetRepairEvidence` (`welded_vertices`,
`removed_duplicate_triangles`, `flipped_triangles`, `boundary_edges`,
`components`, ...), and the envelope route republishes every count as a
`source_topology_repair:<name>` achieved compliance row. Degenerate triangles,
edges shared by more than two distinct triangles, and non-orientable components
refuse with a typed `MeshingFailure` at stage `topology_repair` naming the
authored triangle rows. The closed volume is the explicit envelope wrap, not a
repair claim: compliance reports the native deviation proof as
`source_envelope_input_to_envelope_upper` (every authored point lies within this
distance of the envelope surface) and `source_envelope_envelope_to_input_lower`
/ `_upper` (every envelope-surface point lies in this distance band from the
authored sheet). Original-source witnesses query the authored sheet itself, not
the envelope.

## Additional optional backends

| Provider | Dependency | Supported execution boundary |
| --- | --- | --- |
| Mmg | Mmg 5.8 libraries plus the packaged persistent worker | Simplex metric, level-set, and Lagrangian adaptation through Mmg2D/MMGS/MMG3D with region/boundary references and field interpolation |
| fTetWild | `phydrax[meshing-ftetwild]` | Robust surface-to-tetrahedron generation; sampled boundary-envelope evidence |
| Poisson | `phydrax[meshing-poisson]` | Optional external Open3D screened-Poisson provider; native point-cloud source reconstruction does not depend on it |
| OpenVDB | Native `openvdb` Python binding; conda-forge provides it | Existing sparse voxel field to isosurface, with explicit background semantics |
| Omega_h | Omega_h, built either serially or with MPI, plus the packaged persistent worker | Simplex metric adaptation with class preservation, field transfer, ownership, and ghost residence |
| VoroCrust | Source-built mesher plus the packaged persistent extraction worker | Surface sampling and explicit face-defined Voronoi cells |
| TIOGA | MPI-enabled TIOGA plus the packaged persistent collective worker | Overset hole cutting, moving-part updates, and donor/receptor interpolation between affine cell parts |

fTetWild accepts one soft `UniformSizeControl` whose scope exactly matches the
complete requested source boundary. It rejects patch controls, region controls,
seeds, layers, periodic constraints, and local sizing rather than weakening or
inventing their semantics.

External topology-changing providers are nondifferentiable. Mmg and Omega_h
output IDs identify the new revision; unknown ancestry is not replaced with a
nearest-neighbor transfer. Only explicitly declared fields are transferred, each
with method evidence: Mmg locates every output vertex in the source simplices
and interpolates P1 barycentrically (projection outside the source is counted
and bounded), and Omega_h applies its own linear or conservative transfer with
integrals reported before and after. Omega_h output metrics are checked against
the input field's hard size and anisotropy bounds after provider gradation; a
violation rejects the provider result instead of widening those declarations.
fTetWild's boundary check samples vertices and centroids and is not a continuous
Hausdorff certificate. Poisson reconstruction does not invent CAD associations.
OpenVDB must know how inactive and out-of-domain voxels are extended.

Native provider worker sources ship with every distribution: under
`native/providers` in the source tree and source distribution, and inside the
installed package in a wheel. `native_provider_source_path(provider)` (from
`phydrax.meshing.providers`, for `"mmg"`, `"omega_h"`, `"tioga"`, or
`"vorocrust"`) returns the provider's standalone CMake project in either layout;
its sibling `common` directory holds the shared header-only protocol. Each
project builds one persistent worker executable against a separately installed
upstream library; importing `phydrax` neither compiles nor launches them. Build
and link each worker with the same C++ compiler and, where applicable, the same
MPI implementation as its upstream library. Install the executables into one
prefix and either put its `bin` directory on `PATH`, set the provider's worker
variable (`PHYDRAX_MMG_WORKER`, `PHYDRAX_OMEGA_H_WORKER`,
`PHYDRAX_TIOGA_WORKER`, `PHYDRAX_VOROCRUST_WORKER`), or pass the executable
explicitly. The build commands below resolve the source directory with:

```console
source_of() {
  python -c "import sys; from phydrax.meshing.providers import native_provider_source_path; print(native_provider_source_path(sys.argv[1]))" "$1"
}
```

A provider launches its worker lazily and reuses that process across calls
until `close()` (providers are context managers), a failure, or the worker
policy's call bound. The worker reports its exact runtime identity (library
release, commit, build options, MPI ranks) once at startup; no version process
is spawned per call. Arrays travel through a binary exchange directory: one
little-endian NPY file per array plus a canonical `manifest.json` listing name,
dtype, shape, and payload SHA-256, verified and memory-mapped on receipt.
Optional external workers use `NativeWorkerPolicy` for process isolation:
`operation_timeout_seconds`, `maximum_input_exchange_bytes`, and
`maximum_output_exchange_bytes` govern the external call, not native CAD or
meshing acceptance. Scientific requests carry no `MeshingLimits` quota.
Timeouts, exits, protocol violations, and native rejections surface as
`MeshingFailure` with the worker log tail; the raw worker evidence is kept on
`__cause__`.

For an installed Mmg 5.8 CMake package (shared libraries):

```console
cmake -S "$(source_of mmg)" -B build/phydrax-mmg \
  -DCMAKE_PREFIX_PATH=/path/to/mmg-install
cmake --build build/phydrax-mmg
cmake --install build/phydrax-mmg --prefix /path/to/phydrax-native
```

`MmgProvider` adapts a certified `CellMeshingResult`. Blocks and cell zones
become Mmg region references, facet patches and zones become boundary
references, and all of them are rebuilt by name on the adapted mesh; required
vertices/edges/faces and ridges are `MeshingScope` values on the source. An
exactly isotropic `MeshMetricField` is sent as scalar sizes, any other as a
tensor, and the evidence names the representation. `MmgLevelSet` discretizes a
vertex level set into interior/exterior labels per region and a contour patch.
`MmgLagrangianMotion` requires Mmg built with `-DUSE_ELAS=ON` against the ISCD
LinearElasticity library; without it the worker refuses with
`UNSUPPORTED_CAPABILITY`. Mmg's install records no runtime path to
LinearElasticity, so configure such an Mmg with
`-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`. Mmg 5.8.0's `MMG2D_Set_vectorSols`
stores 2D vector solutions one vertex slot away from where its Lagrangian code
reads them; the worker writes the layout the motion code reads. Configure with
`-DPHYDRAX_MMG_WITH_PARMMG=ON` and ParMmg (plus its own Mmg install) on
`CMAKE_PREFIX_PATH` to build the collective `phydrax-parmmg-worker` for
tetrahedral remeshing under an MPI launcher. The worker imports ParMmg's CMake
package: the ParMmg 1.5.0 release installs none, so build ParMmg at upstream
commit `d2eddc5` (1.5.0 plus its package export) or later.

For an installed Omega_h CMake package:

```console
cmake -S "$(source_of omega_h)" -B build/phydrax-omega-h \
  -DCMAKE_CXX_COMPILER=/path/to/mpicxx \
  -DOmega_h_DIR=/path/to/omega-h/lib/cmake/Omega_h
cmake --build build/phydrax-omega-h
cmake --install build/phydrax-omega-h --prefix /path/to/phydrax-native
```

Use the MPI compiler wrapper as `CMAKE_CXX_COMPILER` when the installed Omega_h
was built with MPI. The Omega_h worker stays alive across calls, one MPI world
per rank count. Blocks, zones, patches, and labels on cell and facet dimensions
become Omega_h class IDs and are rebuilt by name after adaptation; other
boundaries are classified by `OmegaHOptions.feature_angle`. `OmegaHOptions`
passes AdaptOpts quality and length targets verbatim, and evidence reports the
achieved quality and metric edge-length extremes. `OmegaHField` declares LINEAR
vertex or CONSERVE cell-density transfer. Distributed runs return per-rank
`OmegaHPartition` ownership, ghost, and global-ID arrays validated without a
per-entity merge; `gather=True` also certifies the global carrier and a
`MeshDistribution`.

TIOGA must first be installed with its CMake package metadata and global node-ID
support. Its package publishes neither a release version nor its unique-ID
option, so the worker requires both explicitly:

```console
cmake -S /path/to/tioga -B /path/to/tioga-build \
  -DCMAKE_CXX_COMPILER=/path/to/mpicxx \
  -DCMAKE_INSTALL_PREFIX=/path/to/tioga-install \
  -DTIOGA_HAS_NODEGID=ON -DBUILD_SHARED_LIBS=ON
cmake --build /path/to/tioga-build
cmake --install /path/to/tioga-build
cmake -S "$(source_of tioga)" -B build/phydrax-tioga \
  -DCMAKE_PREFIX_PATH=/path/to/tioga-install \
  -DPHYDRAX_TIOGA_REVISION=<exact-release-or-git-commit> \
  -DPHYDRAX_TIOGA_ENABLE_UNIQUEID=<ON|OFF, matching TIOGA_ENABLE_UNIQUEID>
cmake --build build/phydrax-tioga
cmake --install build/phydrax-tioga --prefix /path/to/phydrax-native
```

The worker uses the MPI compiler wrapper recorded by the TIOGA package and
refuses a different one. TIOGA is BSD-3-Clause (relicensed upstream from
LGPL-2.1; see its COPYRIGHT). The worker links the caller-installed library and
reports its caller-supplied exact revision; Phydrax does not redistribute TIOGA.

VoroCrust does not install CMake package metadata. Build a serial CPU checkout,
then point the worker at that exact source and build tree:

```console
cmake -S /path/to/vorocrust -B /path/to/vorocrust-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DVOROCRUST_ENABLE_OPENMP=OFF \
  -DVOROCRUST_ENABLE_MPI=OFF -DVOROCRUST_ENABLE_EXODUS=OFF \
  -DVOROCRUST_TPL_ENABLE_KOKKOS=OFF \
  -DVOROCRUST_TPL_ENABLE_KOKKOSKERNELS=OFF \
  -DVOROCRUST_TPL_USE_LAPACK=OFF \
  -DVOROCRUST_TPL_BUILD_OPENBLAS=OFF
cmake --build /path/to/vorocrust-build --target vc_mesh libVCMesh
cmake -S "$(source_of vorocrust)" -B build/phydrax-vorocrust \
  -DVOROCRUST_SOURCE_DIR=/path/to/vorocrust \
  -DVOROCRUST_BUILD_DIR=/path/to/vorocrust-build \
  -DPHYDRAX_VOROCRUST_REVISION=<exact-release-or-git-commit>
cmake --build build/phydrax-vorocrust
cmake --install build/phydrax-vorocrust --prefix /path/to/phydrax-native
```

The worker refuses VoroCrust builds with optional MPI, Kokkos, LAPACK, or
Exodus linkage whose transitive build-tree dependencies cannot be reconstructed
safely. It detects and links OpenMP when the upstream build enabled it. Pass the
`vc_mesh` executable and optionally the `phydrax-vorocrust-worker` path to
`VoroCrustProvider`; `vc_mesh` runs once per call as a bounded subprocess and
extraction reuses the persistent worker. Its radius control is the backend
sphere-sizing bound, not a guaranteed edge length. No material identities are
inferred from seed colors. Before extraction, output is bounded per emitted
cell: each of the m interior cells is a convex polytope with at most n - 1
facets for n seeds, so Euler's formula bounds it by 2(n - 1) - 4 vertices and
6(n - 1) - 12 face-loop entries; exceeding the limits refuses the run.
VoroCrust packed vertex indices are treated as provider-declared identity and
are retained through polyhedron assembly. Distinct indices are never welded by
coordinate proximity; if their topology leaves an open cell, conversion refuses
instead of guessing aliases. Collapsed provider faces are reported in compliance.
Upstream VoroCrust exposes no bounded native output
allocator or callback. Runtime evidence therefore records native output
entity/connectivity preallocation as unenforced. Set
`VoroCrustOptions(require_native_output_preallocation=True)` only as a
fail-closed capability requirement; execution returns `UNSUPPORTED_CAPABILITY`
instead of pretending that preallocation was enforced.

TIOGA distributes complete named parts among MPI ranks and preserves part-local
ID namespaces. It does not partition one part or support curved/polyhedral donor
cells. `TiogaProvider.move` updates the coordinates of named moving parts in the
resident registration and reruns connectivity without restarting the worker; a
lost session or a superseded registration fails explicitly instead of
registering again. Runtime-specific MPI launcher flags belong in deployment
configuration, not mesh semantics.

Run native benchmarks with
`python -m tools.meshing_benchmarks --resolution 8 16`. Add
`--gmsh-semantic` to run real semantic Gmsh scaling at 2, 8, and 32 regions;
each result contains raw partition, plan, and execution samples plus ranges.
Use `--case <name>` for one scaling case: `cad-partition`, `planar-semantic`,
`layout-decode`, `planar-band`, `hybrid-slab`, `gmsh-semantic`,
`implicit-discovery`, `lbvh`, `overlap-pairs`, `incidence`, `supermesh`,
`remap`, `device-bisection`, `host-bisection`, `local-metric`, `optimization`,
`predicates`, `boundary-layer`, `high-order-certification`, `repartition`, or
`provider-worker`. The `device-bisection`, `host-bisection`, `local-metric`,
`optimization`, `predicates`, `boundary-layer`, `high-order-certification`, and
`repartition` cases report host preparation, lowering, compilation, cold and
warm execution, compiler temporary and output bytes, and retained bytes; a
host-only route is timed as host stages while its compiled consumer (transfer
application, quality or geometry evaluation) carries the compiler measurements.
Layout decode is explicit because its parser dependency is optional, and
`provider-worker` (cold launch versus persistent session) needs a real provider
worker; without one it returns a `missing-dependency` record.

For persistent native PLC refinement, use the actual authored B-Rep workflow:

```console
python tools/meshing_benchmarks.py --case native-brep-plc-refinement \
  --resolution 1 2 --repeats 1 --plc-target-size 0.35
```

Resolution controls the initial accepted surface capacity; the requested volume
size remains explicit. The case separates authoring/plan preparation, boundary
execution and acceptance, CAD-to-PLC scope translation, recovery, persistent
state preparation, refinement, snapshots, and evidence validation. Compare
matching environments and source fingerprints with no competing work.
`UNRESOLVED` refinement and its per-criterion records remain visible: passing
benchmark postconditions does not certify that the requested quality was attained.

Source-region validation uses an oriented-face index rather than a Cartesian
facet-by-tetrahedron scan. Non-relocation edits reuse the source geometry of
unchanged retained facets and segments; replacement geometry is validated before
commit, and relocation revalidates after moving vertices. Staged copies,
candidate incidence assembly, and whole-state audits remain host-topology work
needed by the current atomic edit boundary. The temporary index is sorted from
the current mutable topology; it introduces no scientific capacity limit, dense
numerical matrix, or device synchronization.

For exact global-contact certification, run:

```console
python tools/meshing_benchmarks.py --case global-embedding \
  --resolution 2 3 --repeats 1
```

Resolution is the Kuhn lattice's cells per axis. Mesh preparation, coordinate-map
preparation, local validity, and cold/warm global embedding are separate stages.
The contact proof is host-exact; compiler measurements are not applicable.
Exact separating-axis intervals either prove disjointness or reduce touching
tetrahedra to their complete supporting point, edge, or triangle intersections.
Positive-volume overlap retains the full exact clipping path.

An uncontended Darwin arm64/CPU run on CPython 3.13.11, NumPy 2.5.4, JAX 0.12.0.1,
and Equinox 0.13.8 used x64 and disabled persistent JAX caching. One sample per
capacity gave the following warmed certification times; these are not
distributional performance estimates.

| Tetrahedra | Before (s) | Support-plane route (s) | Ratio |
| ---: | ---: | ---: | ---: |
| 48 | 4.9594 | 0.1962 | 25.27× |
| 162 | 33.7030 | 1.1440 | 29.46× |

Mesh, coordinate, validity, and global-embedding identities and retained-array
bytes were unchanged. The exact contact oracle matrix and an ordered-output
comparison against the previous implementation cover lower-dimensional contact,
volumetric overlap, degenerate refusal, and exact algebraic affine images.

The native qualification inventory is the canonical `NATIVE_SCENARIOS` tuple in
`tools.meshing_qualification.cli`, derived from the single
`NATIVE_CASE_REGISTRY`. `python -m tools.meshing_qualification --scenario
native-corpus` executes every registered native workflow and every mandatory
profile unconditionally. `tools/meshing_qualification/native_case_migration.json`
maps the recovered 13 source families and 114 profile identities to native
scenarios with source provenance, required features, independent oracles,
tolerances, and explicit failing gates; a target mapping is not a pass claim.
Each native record carries an explicit status and canonical scenario identity.
Missing meshcore, unsupported routes, or missing/extra/failed/skipped records
fail rather than disappearing into optional-provider comparisons.

Optional comparisons are selected explicitly: `manifold`, `mmg`, `ftetwild`,
`gmsh`, `vorocrust`, `poisson`, `omega-h`, `gmsh-metric`,
`gmsh-boundary-layer`, `planar-bands`, or `cad-curving`. They report missing
provider dependencies as `missing-dependency` and never act as native fallbacks.
`python tools/check_native_dependency_isolation.py --python 3.13 --all-native`
installs matching wheels, verifies missing-meshcore refusal separately, and runs
every entry in the canonical native scenario registry outside the checkout with
optional geometry-engine imports rejected. Scenario result names must exactly
match that registry; absent, extra, failed, missing-dependency, and skipped
results fail the gate. See the [native CAD guide](guides_native_cad.md) for the
installed gate and source-fidelity contract.

The local Homebrew MPI launcher required the explicit deployment setting
`HWLOC_SYNTHETIC='pack:1 core:10 pu:1'` and launcher arguments
`--bind-to none --map-by slot` to avoid an upstream hardware-topology startup
crash. This is not injected by any provider.

The QA extra installs SciPy and CAD typing packages. Native interfaces missing
from upstream declarations use narrow, runtime-checked protocols: the CAD edge
downcast and OpenVDB module, grid, accessor, and NumPy polygon outputs are
explicitly typed. OpenVDB remains lazily loaded. These boundaries require no
type ignores or unchecked casts and are exercised against the real libraries.

## Provider prerequisites and feasibility

VoroCrust source is available at
[sandialabs/vorocrust-meshing](https://github.com/sandialabs/vorocrust-meshing).
Its upstream build defaults to OpenMP, while the one-thread command above
disables it. Leave OpenMP enabled only when the upstream library and worker use
the same OpenMP-capable compiler. On Apple Silicon, the shown route avoids
assuming that Apple Clang supplies an OpenMP runtime.

Prime is an extraction feasibility decision, not an implemented provider.
The documented [Part API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.Part.html)
exposes zone/topology queries, not a direct cell-connectivity array route.
The [FileIO API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.FileIO.html)
can export Fluent meshes, LS-DYNA, and MAPDL CDB, so file-mediated extraction is
feasible in principle but requires format-specific semantic auditing.
[Prime Server](https://prime.docs.pyansys.com/version/stable/getting_started/index.html)
requires a licensed matching Ansys installation on supported Windows/Linux
systems; installing the Python client alone does not supply the server.

## Source association transfer

Topology adaptation carries existing geometry associations only through an
explicit source-kind transfer. `BRepAssociationTransfer`,
`ImplicitAssociationTransfer`, `SurfaceAssociationTransfer`, and
`MappedReferenceAssociationTransfer` each validate their own source authority;
`ComposedAssociationTransfer` combines disjoint source namespaces and assigns
joint class codes without merging source identities. A transfer never invents
a correspondence for an unclassified target.

`ImplicitAssociationTransfer` retains an exact `AnalyticImplicitProfile` or the
`NativeImplicitSource` bound to its authoring profile or typed adaptive
discovery. It keeps the analytic zero-set and interior-region strata distinct
across splits and collapses. Profile projection or the retained source-bound
boundary query verifies target zero-set classes; a carrier's exterior alone
does not establish implicit boundary identity. Native implicit binding retains
the source identity and discovery data rather than recovering semantics from
equal compiled geometry.

`SurfaceAssociationTransfer` carries occurrence-qualified source patches and
trim curves for a current `MeshingDomain` chart. Refined sheet cells retain
their source patch, and trim-edge descendants keep their source curve interval
and orientation from the exact vertex interpolation stencil. Boundary
selections, interfaces, and orientation repairs require their own explicit
remap and are refused rather than discarded.

`mapped_reference_associations` binds target cells to mapped root-cell IDs only
when the target geometry retains exact root restriction ancestry.
`MappedReferenceAssociationTransfer` propagates those identities through
lineage; physical coordinate proximity is not a substitute for root-chart
ancestry. For a mesh carrying several independent source authorities,
`ComposedAssociationTransfer` preserves each namespace in separate
`GeometryAssociation` rows. Field values remain the responsibility of the
accepted mesh transition and the field-specific transfer action.

### Surface-chart field transfer

`prepare_surface_chart_deformation` binds exact source/target material charts,
constructs rational overlap pieces, and retains the endpoint validity, global
embedding, source-fidelity, and physical-measure evidence. Supply the source and
target `SurfaceChartWitness` values and complete boundary chart covers for the
same `MeshingDomain`; a chart transfer does not infer a correspondence from
coordinate proximity.

`prepare_surface_chart_compatible_transfer` builds a full degreewise map between
two `FiniteElementDeRhamComplex` realizations. It integrates the canonical
reference form functionals over the exact material pieces and applies the
source-to-target exterior pullback, including cell orientation and twisted-form
actions. The returned `ComplexMap` carries the primal H1, H(curl), twisted
surface H(div), and top-form actions. Its `commuting_defects` are observed
coefficientwise residuals; the preparation refuses a map whose residual exceeds
the requested numerical tolerance.

`prepare_surface_chart_field_transfer(..., semantics="h1")` selects the scalar
degree-zero action, while `semantics="dg"` selects the conservative material
top-form action. These are not the same as a raw coefficient transpose:
`complex_map.adjoint()` is the Hilbert adjoint under the endpoint Hodge
pairings, and `complex_map.map(k).transpose_mv(...)` is the algebraic dual
pullback.

For cell-average finite-volume densities,
`prepare_surface_chart_finite_volume_contents` integrates physical source area
over each material piece, divides transferred content by target cell measures,
and retains the volume-weighted Hilbert adjoint separately from the raw
transpose. Its endpoint finite-volume coordinates must bind the same cell IDs
and physical vertices as the prepared surface geometry.

Their unstructured finite-volume plans accept consistently oriented triangular
2-manifolds embedded in 3D and retain intrinsic cell areas, edge lengths, and
owner-outward tangent-plane conormals.

`prepare_sphere_chart_compatible_transfer` and
`prepare_sphere_chart_field_transfer` use the retained homogeneous
projective-reference matrices and their nonconstant Jacobians. They do not fit
an affine map to sphere-cell corner images.


## Structured meshing failure evidence

`MeshingFailure` retains one `MeshingFailureEvidence` record alongside its
existing category, provider, stage, entity, and location data. The record
contains canonical requested and achieved quantity pairs, the failure
checkpoint, logical findings, and a deterministic evidence ID. Construct a
`MeshingDiagnostic` with `MeshingDiagnostic.from_failure(failure)` to preserve
those quantities and the same evidence ID in a meshing trace.

## Owner-local initial generation

The private generation owners in `phydrax.meshing._distributed_planar_generation`
and `phydrax.meshing._distributed_generation` produce local construction rows
before collective acceptance. Planar tile and seam identities are determined by
the source, request, and tile schedule; graph partitioning assigns the tiles to
the current `ExecutionGroup`. Each owner triangulates only its tiles and
exchanges seam coordinates until the shared seam sets stop changing. Surface
patches are assigned through the native graph partitioner. Each round, incident
owners first union their shared source-curve samples through the curve steward
and then regenerate their local charts from that common set, which is the serial
chart fixed point; physical refinement runs once after the seams close, so the
owner-local charts equal the serial charts patch by patch. A refinement that
would split a shared source curve is refused with `UNSUPPORTED_COMBINATION`
rather than published without incident-owner agreement. These exchanges carry
seam data and ownership metadata, not a global
mesh or candidate table. Source and request identities must agree on every
group member.

`phydrax.meshing._distribution.prepare_owner_local_cell_shard(part_name,
source_id, source_revision, contract, block_name, cell_kind, rows, group)`
publishes a sharded owner-local view of an ordinary triangle or tetrahedron mesh.
Each rank passes only its owned rows as `OwnerLocalCellShardRows`: owned cell
IDs, their global vertex IDs and exact coordinates, and the global IDs of
intermediate entities in the owned closure.

- Face neighbors are paired through hash-sharded facet records, and their rows
  are requested from their owners.
- Vertex owners are elected as the minimum incident-cell owner.
- The returned rank-local `MeshAssembly` carries a non-serial
  `OwnerLocalCellGeometry`. It holds the owned cells plus their exact face halo,
  the owned face-neighbor graph, a bound owner directory, and a coordinate-map
  authority.
- Empty owners participate collectively and publish empty views.

`MeshDistribution.owner_local_cell_shard_rows(rank)` selects one rank's rows
from an authored serial distribution, for example in test preparation. The
serial adapter `owner_local_geometry_record` remains for single-process
consumers; multi-rank overset exchange refuses its serial views.

`phydrax.meshing._device_generation` prepares an oriented simplex seed,
evaluates a complete device candidate batch, resolves uncertain source
predicates together on the host, and publishes a new seed only after exact
orientation checks of every generated simplex. Layout capacities are initial
allocation hints and grow as required up to native index addressability. Surface
candidate coordinates and normals require one host source-geometry batch, so
that route is reported as hybrid host/device execution.

These generation records retain owner-local arrays and canonical IDs; they are
not globally assembled or certified `CellMeshingResult` values. The three-rank
CPU/Gloo acceptance cases in
`tests/integration/test_distributed_meshing_generation.py` exercise planar
holes, shared curves on a multi-patch sphere, and a tetrahedral cube. Run that
coordinator test with `pytest -n0` so pytest does not nest another worker pool.
It is same-host process evidence, not accelerator or multi-host qualification.

