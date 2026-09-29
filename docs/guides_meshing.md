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

`CellGeometrySpec` belongs to `phydrax.discretization`, not FEM. Its coordinate
nodes and element routes can represent geometry of a different polynomial order
from a solver's unknown field. Corner topology remains separate from curved
geometry nodes.

`evaluate_cell_quality` samples differentiable corner quality per cell family:
signed measure, mean ratio, radius ratio, edge aspect ratio, scaled Jacobian,
corner or dihedral angles, condition number, tetrahedral sliver measure, face
warpage, and metric-space quality when a `MeshMetricField` is supplied.
`sampled_valid` is corner evidence, not a validity proof. Audit
`quality_scope` states whether those samples describe the full geometry
(`vertex_geometry`) or only the straight corner cell (`corner_cells`).

Geometric validity is certified by
`phydrax.discretization.certify_cell_geometry_validity`: the Jacobian
determinant of every mapped cell (Pk simplices, Qk quadrilaterals and
hexahedra, prisms, rational pyramids through collapsed coordinates, and
star-decomposed polyhedra) is converted to Bernstein form and
adaptively subdivided until it is proven above the policy floor
(`CERTIFIED_VALID`), proven below it (`INVALID`), or the `CellValidityPolicy`
budget is exhausted (`UNRESOLVED`). A twisted trilinear hexahedron with positive
corners but an inverted interior is therefore invalid, and curved geometry is
certified rather than rejected. A polyhedron that is not star-shaped about its
center stays `UNRESOLVED`; that is not a validity disproof.

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
predicates stay uncertain are not counted as clean but name
`self_intersection_predicates` in `unresolved`, as do unresolved validity
certificates (`geometry_validity`), an exceeded weld candidate capacity
(`coincident_vertex_capacity`), and an exhausted intersection candidate budget
(`self_intersection_capacity`). Each becomes an `unresolved_<name>` finding under
`CellMeshAuditPolicy.unresolved` (`REJECT` by default; `SKIP` is refused). Under
`RECORD` the audit may pass, but the check stays named in `unresolved` and
`recorded`. Connectivity above `maximum_connectivity_entries` is rejected with
`connectivity_capacity_exceeded`; `maximum_coincidence_candidates` bounds
welding and `maximum_intersection_candidates` bounds the streamed polygon-edge
and triangle-pair broad phases. Exhaustion fails closed without materializing an
unbounded pair table. Supplied geometry or semantic evidence that cannot survive
canonical reordering is rejected, not silently rebound. Association residual
tolerances remain provider-owned.

`evaluate_finite_volume_quality` reports skewness and non-orthogonality of every
interior face of a full-dimensional mesh.

## Scopes and organization

A `MeshingScope` names a source, its exact revision, entity kind and degree,
authoritative entity set, and persistent IDs. Boolean scope operations require
matching identity domains. Stale revisions are errors, not empty selections.

- `MeshPatch`: geometric boundary support.
- `MeshZone`: exclusive solver/material organization.
- `MeshLabel`: overlapping semantic grouping.
- `MeshAttribute`: entity-associated numerical or categorical data.

Geometry associations record source entities, target IDs, residuals, and
resolution/ambiguity evidence. Persistent IDs are not positional array indices.

## Controls and providers

`SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, and `VolumeMeshingSpec` separate
physical requests from backend options. Cell-family policies, fill strategies,
size controls, regions, patches, layers, and periodic constraints are explicit.
A control's presence in the native contract does not imply every provider can
honor it. Provider preflight rejects unsupported combinations before work.

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
explicit face or edge scope. Derive those scopes from `BRepModel.solid_ids`,
`face_ids`, `edge_ids`, and `topology` through `GmshProvider.entity_scope`;
never reconstruct them from coordinates, import tags, or positions. This keeps
every patch adjacency tied to one exact B-Rep revision.

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
and point every column along the visibility-optimal direction (the minimum-norm
point of the convex hull of its face normals, a simplex QP solved by
`phydrax.optim`). Concave corners are accepted only while a visible direction
exists and the height stretch stays within `maximum_corner_stretch`; otherwise
they are rejected with the offending wall vertices. Heights are limited by the
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

Gmsh is optional (`pip install 'phydrax[meshing-gmsh]'`). Use
`GmshProvider` with a revision-checked reopenable `BRepModel` or solid
`BRepSource`. The B-Rep import or persistence call owns the required
`SpatialCoordinateContract`; `GmshOptions` does not own coordinates. Sessions
are host-side resources and must not enter JAX state. Use a context manager for
an explicit `GmshSession`; its cleanup does not suppress the original exception.
Open CAD faces use `BRepModel`; `BRepSource` retains its closed-solid invariant.
Required mixed families must actually occur in the generated result; permitting
mixed recombination does not guarantee that triangles will remain.

### Exact semantic B-Rep and planar routes

`BRepImportReport.source_digest` identifies the persisted CAD bytes;
`source_revision` additionally binds the coordinate contract and import policy.
Replacing source bytes invalidates an existing meshing plan even when the path is
unchanged. `BRepPartitionPlan` and `PlanarPartitionPlan` are geometry-owned
Boolean operations: they publish persisted B-Rep output, exact
`CADRevision`/`AssociationGraph` evidence, and named solid/face regions with
face/edge patches. Gmsh consumes that evidence; it does not repair, fragment, or
recover material meaning from names or proximity.

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

`NativeImplicitProvider` separates surface discovery from fixed-route
realization. Discovery is vectorized host preparation (batched lattice field,
one-program ITP root isolation, table-driven dual cells, batched QEF solves,
BVH-bounded self-intersection candidates). `ImplicitMeshingPlan.execute(state)`
runs in process: audits consume the primal realization, and
`result.geometry.coordinates` carries the JAX realization, so observables of the
result differentiate with respect to design parameters while the topology and
route remain valid. Discovery and audit decisions are not differentiated.

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

`OpenVDBProvider` ingests sparse voxel bricks in bulk and may re-distance the
isosurface with `OpenVDBLevelSetRebuild` before extraction. `PoissonProvider`
publishes native per-vertex sample density, optional density-quantile trimming,
an explicit thread count, and the declared (Neumann) boundary condition.

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
`HXT`), `num_threads` for `General.NumThreads`, Netgen optimization, and
`GmshHighOrderOptimization` for geometry orders two through four. Sessions refuse
plans needing more threads than their `MeshingExecutionPolicy.parallelism`;
multi-threaded runs report nondeterministic runtimes, so deterministic
specifications require one thread. Order-three and order-four simplex maps are
resampled exactly at the canonical Lagrange nodes with shared entity nodes.
Compliance records Gmsh's native `minSICN`/`minSIGE` element quality.

`SurfaceRemeshingSpec` on a `SurfaceModel` (or a `CellMesh` with an explicit
coordinate contract) classifies sharp features and boundary curves per
`SurfaceReconstructionControl`, reparametrizes the discrete patches, and
remeshes them. Compliance bounds the two-sided vertex deviation and requires the
Euler characteristic, component count, and boundary loops to be preserved.

A `GmshSession` caches CAD imports by source revision and coordinate contract.
Every execution re-digests the persisted bytes; replaced bytes evict the entry
and fail as an invalid source. Gmsh imports a private verified snapshot.

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
  one barycentric compatibility refinement. The conformity closure is vectorized
  per iteration; coarsening removes complete bisection patches (Chen–Zhang
  vertex removal) and restores the recorded cell, edge, and face IDs, so a
  refine/coarsen round trip reproduces the source topology exactly.
- `NATIVE_METRIC_2D`: planar triangle metric adaptation by edge split
  (metric length above sqrt(2)), collapse (below 1/sqrt(2)), metric-quality flip,
  and relocation, with deterministic independent-cavity selection. Orientation
  decisions use the policy's predicate mode; unresolved filtered signs reject the
  operation. The status is COMPLETE when the unit-mesh criterion holds, STALLED or
  PASS_LIMIT otherwise; `RelocationMeshAdaptation` moves vertices only.
- `DEVICE_BISECTION`: the marked bisection request executed as compiled
  fixed-capacity device passes (below); it commits the meshes, IDs, lineage,
  and hierarchy of `NATIVE_BISECTION` byte for byte.
- `DEVICE_METRIC_2D`: the planar metric and relocation requests executed as
  compiled device passes with FILTERED_DEVICE predicates (below).
- `MMG`, `OMEGA_H`: the configured provider's metric adaptation; the lineage
  records every entity as created/deleted and no vertex stencil is claimed. With
  an `association_transfer` the providers re-derive the B-Rep associations on the
  adapted target (see below); without one Mmg records the dropped associations as
  an adapter loss and Omega_h rejects the source.
- `HP`: tensor-product h-refinement or coarsening of a `FiniteElementHPEpoch`,
  with lineage projected by `project_hp_lineage`.

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
an epoch, so the compiled closure issues vertex and cell IDs by prefix sums in
exactly the host order: each closure round sorts the new refinement edges,
bisects the selected cells with the static Maubach templates, and selects every
cell holding a split edge until the fixed point (a `lax.while_loop` bounded by
`maximum_closure_iterations`). Marks whose own closure would split a protected
edge are found by one reverse-reachability fixed point over the union closure
and rejected, as on the host. Coarsening passes remove every unprotected
bisection vertex whose star is exactly the marked, class-uniform children of
its bisections, restore the parents under their original IDs, and retire the
children. Every call records its `AdaptiveSimplexStatus` flags in
`AdaptiveSimplexState.status_flags`, which accumulate over the prepared epoch.
Capacity overflow, the closure bound, a protected conflict, and a
certified-invalid child are terminal failures: the call rolls back every mesh,
topology, and numerical array but still records its flags, and every later
refine or coarsen call on that state is refused on device (state unchanged,
`AdaptiveSimplexReport.status` holding the terminal flags, zero operation
counts); recovery needs a new prepared epoch, for resources with larger
capacities. Children are checked with FILTERED_DEVICE orientation predicates;
unresolved signs set `NEEDS_HOST_RESOLUTION`, which stays applied.

`commit_adaptive_simplex(prepared, state)` performs one device-to-host transfer
and decides acceptance from the cumulative flags: a terminal flag raises
`MeshingFailure` (capacity or closure bound: RESOURCE_EXHAUSTED; protected
conflict: INVALID_SPECIFICATION; invalid geometry: QUALITY_REJECTED),
`NEEDS_HOST_RESOLUTION` requires exact host orientation predicates to certify
every committed cell positive (otherwise, or when meshcore is unavailable to
resolve a sign, QUALITY_REJECTED at stage `device-commit`), and a coarsening
stopped at its pass bound commits the valid partial topology with status
PASS_LIMIT (not converged). An accepted epoch runs the host edit assembly:
canonical `CellMesh`, organization inherited by
exact IDs, certification, `CellMeshTransition`, complete `MeshLineage`, sparse
P1 transfer, `BisectionEvidence`, and the next `BisectionHierarchy`, returned
as the `MeshAdaptationResult` the solver transaction consumes. Cell lineage
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
the ownership. `commit_partitioned_adaptive_simplex` merges the parts by global
ID and commits exactly the single-device result; ownership moves to the target
only through the adaptation's `MeshDistributionTransition`. Part epochs refine
only; a round that would exceed any part's capacity or split a protected edge
fails for all parts, the refusal of later calls is collective, and a terminal
flag recorded on any part rejects the whole epoch at commit.

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
rejection counts, unit-mesh measurements); capacity overflow and invalid
geometry roll the arrays back, record the terminal flag in
`DeviceMetricState.status_flags`, and refuse every later call on that state.
`commit_device_metric_adaptation` applies the same acceptance as
`commit_adaptive_simplex` (terminal flags raise, `NEEDS_HOST_RESOLUTION` is
resolved by exact host orientation of the committed cells), performs one
transfer, and returns the `MeshAdaptationResult` with `DeviceMetricEvidence`; a
run that applied nothing is UNCHANGED, with convergence or stall in the
evidence (as for `NATIVE_METRIC_2D`).

## CAD association and high-order curving

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere

shape = BRepPrimAPI_MakeSphere(1.0).Shape()
model = phx.geometry.model_from_occt_shape(
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
projection = phx.geometry.prepare_brep_projection(model, shape)
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

A B-Rep `GeometryAssociation` classifies mesh entities on B-Rep entities of one
exact revision: every row names the entity dimension and index
(`source_dimensions`, `source_indices`, matching the `<revision>:<kind>:<index>`
identity), closest-point `parameters`, the projection residual, ambiguity, the
orientation relation of the entity's canonical vertex order to the B-Rep tangent
or oriented normal, and parent provenance (`parent_dimensions`, `parent_ids`,
`parent_association_id`, `GeometryAssociationProvenance`).
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
accepted only when `certify_cell_geometry_validity` certifies every cell and every
constrained residual is within `residual_tolerance`. The initial CAD projection
is accepted on its own when valid; a relaxation whose minimization did not
converge cannot replace the accepted geometry unless
`HighOrderCurvingPolicy(accept_valid_nonconverged_relaxation=True)` permits it.
`HighOrderCurvingResult.relaxation_statuses` records every relaxation's native
termination status and `accepted_round` names the accepted candidate (0 for the
projection). Without an accepted candidate the result rolls back to the straight
geometry with the rejected candidate's evidence (`HighOrderCurvingStatus`,
including `ROLLED_BACK_NONCONVERGED`). `PeriodicCoupling` vertex pairs make target-side
high-order nodes the declared isometry of their source-side nodes
(`PeriodicCoupling.match_points` pairs the nodes). `verify_curved_geometry`
certifies an existing high-order geometry, such as a Gmsh high-order output,
without moving it.

## Polyhedral storage

Polyhedral topology uses packed incidence offsets/values. Exact-width
`PolyhedralBlock` groups coexist with standard cell blocks. Padding belongs only
in explicitly bounded execution worksets, not authoritative connectivity.
`prepare_polyhedral_worksets` rejects allocations beyond its entry budget.

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
BVH nearest source node inside a capture radius. `OversetCoupling` provides
donor interpolation and its transpose with explicit holes and partition-of-unity
weights; `OversetCoupling.search` locates receptors in a simplex donor mesh and
uses clipped barycentric weights. Overset coupling is interpolation, never a
conservative overlap remap (`conservative` is false). Searches that leave a
receptor without an admissible donor raise `CouplingSearchError`, whose
`CouplingSearchEvidence` reports every receptor's `CouplingSearchStatus`.

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

`AbstractMeshProposer` is the neutral `DECISION` component slot
(`slot_semantic_id="phydrax.meshing.mesh-proposer"`). A proposer decides where
and how a mesh should adapt; it never produces a mesh. `LearnedMeshProposer`
maps one feature row per scope entity (sorted global-ID order) to a marking
score per cell, a size per vertex, or a metric tensor per vertex, and
`propose(source, features)` wraps the values in the typed proposal of its
`kind`. The typed proposal rejects non-finite values and stale scopes, and
identical values from any proposer project identically: protected entities,
size bounds, gradation, capacity limits, native refinement, safety audit, and
compliance all apply unchanged. The model is a dynamic child whose arrays stay
PARAMETER; `evaluate(features)` is the differentiable per-entity map for
supervised training. A model declaring ports binds only through the proposer's
declared `ports` (one feature-row port of shape `(in_size,)` and the proposal
value port) and an explicit `port_mapping`.

Supervised marking targets come from native estimators, reordered to the
proposal scope's sorted global IDs: `FiniteElementDWRIndicators.absolute` from
`phydrax.discretization.fem.local_dual_weighted_residual` for goal-oriented
finite-element refinement, and the `refine_mask` (or the per-channel
`indicators`) of the high-enthalpy AMR indicator evidence for
aerothermodynamic refinement. Size or metric targets are likewise native
projected fields; a learned proposer imitates them but is always re-certified.

## Additional optional backends

| Provider | Dependency | Supported execution boundary |
| --- | --- | --- |
| Mmg | Mmg 5.8 libraries plus the packaged persistent worker | Simplex metric, level-set, and Lagrangian adaptation through Mmg2D/MMGS/MMG3D with region/boundary references and field interpolation |
| fTetWild | `phydrax[meshing-ftetwild]` | Robust surface-to-tetrahedron generation; sampled boundary-envelope evidence |
| Poisson | `phydrax[meshing-poisson]` | Open3D screened Poisson reconstruction from oriented points |
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
dtype, shape, and payload SHA-256, verified and memory-mapped on receipt. Every
call is bounded by `MeshingLimits.maximum_wall_seconds` and
`maximum_data_bytes`; worker memory is limited by an address-space limit where
the host enforces one (Linux) and otherwise audited against the peak resident
size the worker reports, and runtime evidence names which applies. Timeouts,
exits, protocol violations, and native rejections surface as `MeshingFailure`
with the worker log tail; the raw worker evidence is kept on `__cause__`.

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
Backend vertices that differ only within `relative_merge_tolerance` (default
1e-9) times the output bounding-box diagonal are normalized to the lowest
source vertex ID; coordinates are not averaged. Real VoroCrust output leaves
some shared corners unwelded up to a few 1e-11 relative. Collapsed alias faces
and merged vertex counts are reported in compliance. Set the tolerance to zero
for exact aliases only. Upstream VoroCrust exposes no bounded native output
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

`python -m tools.meshing_qualification --scenario gmsh` proves native-BRep
replay identity, stale-source rejection, exact multi-region zones/patches, and
solid-scoped sizing evidence. The qualification command also names
cad-partition, composite-heat-3d, planar-semantic, composite-heat-2d,
layout-stack, layout-observable, planar-bands, hybrid-layer, and
hybrid-diffusion scenarios, as well as Manifold, Mmg, fTetWild, Poisson, and
VoroCrust. VoroCrust accepts explicit `--executable` and `--worker` paths.
The substrate scenarios `bisection`, `device-adaptation`, `metric-adaptation`,
`gmsh-metric`, `supermesh`, `remap-p1`, `cad-curving`, `boundary-layer`,
`gmsh-boundary-layer`, `predicates`, `triangulation`, `distribution`,
`forest-amr`, `omega-h`, and `omega-h-distributed` each record source and target
identities, the runtime identity (build, environment, backend, topology,
precision, and meshcore library), quality before and after, conservation and
transfer properties, resource use, and status; a scenario whose external
dependency is absent returns an explicit `missing-dependency` record instead of
passing. `examples/meshing_omega_h.py` exercises serial or MPI adaptation.

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
