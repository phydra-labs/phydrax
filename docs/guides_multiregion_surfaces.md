# Multiregion surfaces

`phydrax.geometry.multiregion_surface` owns labeled, possibly non-manifold
triangle complexes: soap-film clusters, dry foams and films spanning wire
frames. Every face separates two distinct regions and edges may carry any number
of faces within a declared capacity, so Plateau borders (three films on an edge)
and tetrahedral vertices (four regions) are represented directly. Manifold
triangle meshes (`TriangleTopology`, `TriangleMesh`) keep their manifold
contract; this package is a separate owner that exposes manifold views of its
sheets.

`multiregion_surface_candidate_profiles()` is the sole candidate-profile
provider for geometry, hard-label extraction, remeshing and topology events in
this package. These profiles require topology evidence; foam KKT evidence
belongs to the consuming foam application rather than the geometry owner.

## Labels and orientation

A face stores the ordered label pair `(left, right)`; its right-handed normal
`(x1 - x0) x (x2 - x0)` points out of `left` into `right`. Labels index a region
table of stable string identifiers declared with a kind:

- `"finite"` regions are closed 3-cells. Their oriented boundary must be a
  watertight 2-cycle and their signed volume is defined;
- `"boundary"` labels are the unbounded ambient or open regions cut by wire
  frames. They are never 3-cells and share the reference pressure of the
  surrounding medium.

Region pairs are canonical `(min, max)` label pairs. The `(vertex, region-pair)`
sheet slots of `vertex_pair_slots` carry sheet fields (film liquid volume,
surfactant amount); a vertex on a manifold sheet has one slot, a Plateau-border
vertex three and a tetrahedral vertex six.

## Capacities, topology and state

`MultiRegionSurfaceCapacityPlan` declares vertex, edge, face, region,
region-pair and event capacities, the maximum edge valence, the maximum number
of region pairs per vertex, and coordinate/index dtypes. These are parameters,
not hard-coded limits: a dry foam needs valence three and six slots, a general
complex may declare more. A surface that does not fit is refused with
`MultiRegionSurfaceCapacityError`, whose evidence names the exceeded capacity;
`MultiRegionSurfaceCapacityPlan.capacity_evidence` performs the same host
admission ahead of construction; event passes use it before committing a
candidate. `event_capacity` bounds the events accepted per pass (zero refuses
every event).

`MultiRegionSurfaceTopology` pads active entities into fixed capacities and
builds the lexicographic edge table, edge-to-face incidence with traversal
signs, region pairs and sheet slots. Stable vertex and face global ids and the
topology epoch give deterministic `topology_id` (incidence) and `lineage_id`
(identity) fingerprints. Periodic domains are refused at construction: signed
volumes require unwrapped periodic coordinates, which are not represented.

`MultiRegionSurfaceState` holds capacity-shaped positions, velocities, sheet
fields `(vertex, slot, field)` and region fields `(region, field)`. Fields that
must survive topology changes are extensive.

## Validation

`validate_multiregion_surface` is a host certificate that never raises for a
scientifically invalid surface; the status names the first failing check:

1. **Region cycles.** For every edge and finite label the signed incidence of
   the label's boundary faces must vanish. An odd count means an open region
   (`REGION_NOT_WATERTIGHT`); an even count with a nonzero sum means
   inconsistent labels or orientation.
2. **Wedge labels.** Faces around each edge are ordered counterclockwise about
   `v1 - v0` using exact `orient3d` signs. The wedge after face `a` is `right(a)`
   when `a` traverses `v0 -> v1` and `left(a)` otherwise; the wedge before face
   `b` is `left(b)` or `right(b)` correspondingly. Consecutive faces must agree.
   One mismatched wedge between two boundary labels marks a wire border; any
   other mismatch is `LABEL_ORIENTATION_INCONSISTENT`.
3. **Valence profile.** `profile="dry_foam"` enforces Plateau's topological laws
   (Taylor, 1976): interior edges carry two or three films and interior vertices
   touch at most four regions. `profile="manifold_two_region"` instead requires
   exactly one finite region and one boundary region separated by one closed
   valence-two sheet, with both regions at every vertex and no wire borders
   (`NONPHYSICAL_VALENCE` on mismatch).
4. **Vertex stars and region connectivity.** Every sheet must meet a vertex in
   one fan (`SINGULAR_VERTEX` otherwise: the unresolved stage of a pinch). Under
   the dry-foam profile every pair of regions meeting at an interior vertex must
   share a film there (`REGION_GRAPH_INCOMPLETE`: two cells touching at a point,
   the unresolved stage of a T1 process). Face sides joined across consistent
   wedges give the connected components of every label
   (`region_component_counts`); a finite region with more than one component of
   positive enclosed volume is `DISCONNECTED_FINITE_REGION` (a cavity boundary
   has negative enclosed volume and is part of its region).
5. **Embedding.** Exact face degeneracy, positive finite-region volumes and a BVH
   broad phase with an exact triangle-pair narrow phase that ignores contact at
   shared vertices and edges (`SELF_INTERSECTION`). The candidate budget is a
   resource bound; exceeding it is a refusal.

Host predicates use the native meshcore adaptive expansion provider when
`phydrax[meshcore]` is installed. Without it, filtered decisions that are not
already certified are resolved by exact dyadic-rational evaluation of the
binary64 input coordinates. Thus collinear vertices of subdivided flat
polygons and coplanar symmetric quads are certified rather than guessed.
`UNCERTAIN_PREDICATE` remains a fail-closed outcome for nonfinite inputs or
other decisions that no exact route certified.

## Prepared geometry

`PreparedMultiRegionSurface` validates one epoch and binds sparse
`phydrax.sparse` relations (face corners, face-to-finite-region signs,
face-to-pair, corner-to-slot) and a packed face BVH. All evaluations are pure
JAX over capacity-shaped arrays and differentiable at fixed topology:

- region volumes `V_r = sum_f s_rf (x0 - c) . ((x1 - c) x (x2 - c)) / 6` with
  `s_rf = +1` for `left`, `-1` for `right` and a fixed reference point `c`;
- surface energy `E = sum_f gamma_f A_f` with effective pair tensions
  (`2 sigma` for a soap film, the single interfacial tension otherwise);
- forces `-dE/dx`, so junction balance emerges without junction curvature;
- per-sheet areas, barycentric sheet-slot areas and counterclockwise junction
  wedge angles.

## Views

`multiregion_cell_complex` returns the signed `vertex -> edge -> face -> region`
`CellComplexTopology` over finite regions only; its boundary-of-boundary check
certifies closed regions. `multiregion_sheet_views` returns one
`TriangleMesh` per region pair whose faces form an oriented manifold with
regular borders (normals out of the pair's first label) together with vertex,
face and sheet-slot maps; non-manifold pairs are listed instead of forced into
a manifold type.

## Transfers

Topology changes move data between entity layouts through two sparse
contracts. `ConservativeFieldTransfer` moves extensive content with
nonnegative weights that sum to one over each active source, so totals are
conserved exactly and positivity is preserved; construction refuses
nonconservative weights and unsupported targets, and `evidence` reports totals,
defects, positivity and support. `BoundedFieldReconstruction` rebuilds intensive
fields as convex combinations and reports local-bound excess.

## Topology events

Every topology mutation is one host transaction per pass,
`apply_surface_events(topology, state, proposals, policy=...)`, over the closed
event set `SurfaceEventKind` (`SPLIT`, `COLLAPSE`, `FLIP`, `T1_POP`, `PINCH`,
`MERGE`, `REGION_SPLIT`, `BURST`) with fail-closed typed dispatch:

1. proposals are ordered canonically (physical transitions, then collapses,
   splits and flips, each by priority and stable ids) independently of the
   input order;
2. a proposal whose vertex 2-ring meets an accepted one is `CONFLICT`; at most
   `event_capacity` events of the capacity plan are accepted per pass;
3. the proposal's own builder drafts the local edit (removed and new faces with
   parent faces, new vertices with parent vertices) with stable global ids;
4. finite-region volumes changed by the edit are restored by a minimum-norm
   displacement of the event's free vertices (widened to their free one-ring
   when the event's own vertices cannot span the constraints);
5. exact guards run on the complete stars of the touched vertices: duplicate
   and degenerate faces, label/orientation wedges, valence and slot capacity,
   singular fans, region-graph completeness (dry foams), region extinction,
   face folding (`maximum_normal_rotation`) and welded triangle-pair
   intersection against the neighbourhood;
6. every motion leg of the event and the restoration motion are certified by
   conservative inclusion CCD (`InclusionCCDPlan`) against the static
   neighbourhood; a flip is certified as the motion of a virtual split vertex
   from the old to the new diagonal midpoint, which sweeps the flip
   tetrahedron;
7. the candidate topology is assembled, admitted against the capacity plan and
   validated exactly; the sheet-slot, face, region and velocity transfers are
   prepared, and state-owned sheet, region and velocity fields are applied with
   their certificates;
8. the capacity-shaped dynamic payload commits through
   `phydrax.lifecycle.commit_candidate` only when validation and every transfer
   certificate pass. Otherwise the source topology and state objects are
   returned unchanged.

`SurfaceEventPassEvidence` lists every proposal with its reason code
(`SurfaceEventStatus`), CCD time of impact and residual volume change, the
`MultiRegionSurfaceLineage` (vertex, face and region parents and removals),
the candidate capacity and validation evidence, the conservation residuals of
the extensive sheet and region transfers and the source/target
`TopologyEpoch`s. A committed pass exposes the prepared sparse sheet-slot and
face transfers and their nondifferentiable `TopologyEpochTransition`s, so an
application can reuse each route for all of its field components.
`derivative_available` is always `False`.

`multiregion_topology_epoch(topology, positions)` returns the epoch a pass
certifies for the current geometry (incidence, stable lineage and exact active
positions); it equals the pass's `source_epoch`. A cross-owner
`phx.lifecycle` rebind uses its `epoch_id` as the structure identity of
epoch-owned state, so `transition.composition_transport(source, target)` binds
sheet-slot content across the pass, and consumers such as the Plateau-border
network re-prepare on `target_epoch` (see the
[foam guide](guides_soap_films_and_foams.md#border-content-across-topology-events)).

Whole-sheet `BURST` uses `apply_surface_burst`, because vanished sheet content
cannot satisfy the ordinary surviving-sheet transfer. Stable face IDs must
name the complete separating sheet. The transaction deletes it, merges the two
region labels, conservatively transfers surviving sheet and region fields, and
returns vanished per-field content explicitly in `SurfaceBurstResult`; the
foam owner must place liquid in its rim ledger before exposing the result.

### Field transfers across events

Sheet-slot content moves by corner pooling: a slot's content is attributed to
its face corners in proportion to face area; corners of untouched faces keep
their content at the same `(vertex, region pair)`, and the corners of the faces
an event replaces are pooled per region pair and redistributed over the
replacement faces' corners by barycentric area. Pairs without replacement (a
vanished film) and new pairs (a merged wall) are served by the event's
remaining pool. The totals of every sheet field are conserved exactly, nonnegative
content stays nonnegative, and a uniform thickness is reproduced exactly by
area-preserving events (midpoint splits). Region fields keep their identity;
split regions share their parent's content by component volume (finite regions,
the equal-pressure ideal-gas share) or bounding area (boundary labels).
Velocities are reconstructed as bounded averages of parent velocities.

Face-integrated application content uses the same event groups: stable faces
retain identity, while every replaced face distributes to its local replacement
faces in proportion to target area and within the same parent region pair. The
route is sparse, exactly conservative, and shared by every per-face field.

### Quality remeshing

`remesh_edge_flags` computes device flags of every edge for a
`MultiRegionRemeshPlan`: split edges longer than `maximum_edge_length`,
collapse edges shorter than `minimum_edge_length` or the shortest edge of a
face with a smaller angle than `minimum_angle`, and flip two-face sheet edges
that violate the Delaunay criterion while their faces deviate from coplanarity
by at most `maximum_flip_dihedral`. `propose_remesh` reads the flags once at
the host boundary and returns `EdgeSplitProposal`, `EdgeCollapseProposal` and
`EdgeFlipProposal` in canonical order.

- A split inserts the edge midpoint into every incident face, so a Plateau
  border splits all three films and the geometry is unchanged. The complete
  candidate still undergoes exact self-intersection validation: meshcore
  resolves filtered-uncertain signs when installed and the exact dyadic host
  fallback handles collinear welded vertices without it. Event policies refuse
  `check_self_intersection=False`; no split-only bypass exists.
- A collapse keeps the higher-ranked endpoint (sheet interior, feature curve,
  feature corner, fixed wire); equal-ranked curve vertices collapse only along
  their curve and corners or fixed vertices are never removed
  (`FEATURE_NOT_PRESERVED`). The generalized link condition (common neighbours
  equal the opposite vertices of the collapsing faces) refuses necks and
  duplicate edges (`LINK_CONDITION_VIOLATED`).
- A flip never crosses a junction, border or label change.

### Physical transitions

- `T1PopProposal` (Weaire and Rivier, 1984; Da, Batty and Grinspun, 2014): a
  vanishing film between regions `D` and `E` collapses to one vertex whose region
  graph then misses exactly the pair `(D, E)` (region-graph incompleteness
  guard); the vertex is pulled apart along the `E -> D` axis into two vertices
  joined by a new Plateau border, every film fan meeting both sides gaining one
  triangle at its most equatorial spoke. Both new vertices have complete region
  graphs; volumes are restored and gas amounts carried exactly.
  `propose_t1_pops` keeps films bounded only by Plateau borders whose bounding
  box is smaller than the threshold (a conservative filter that never misses a
  qualifying film) and certifies each candidate's diameter by a
  branch-and-bound over box-node pairs with a declared node capacity; diagonal
  films whose box passes but whose diameter does not are rejected, and
  overflowing certificates are refused with evidence. The builder re-certifies
  before executing.
- `PinchProposal` (neck criterion): a three-edge sheet loop bounding no face is
  collapsed; the collapsed vertex must have exactly two disconnected incident
  fans, which are separated along the neck axis. Remeshing collapses shrinking
  neck rings down to three edges, where the link condition stops it.
  `propose_pinches` lists such loops shorter than the declared perimeter.
- `MergeProposal` with a declared `SurfaceMergePolicy`: facing films `X|g` and
  `g|Y` across a gap label `g` whose certified triangle-triangle distance is
  below `merge_distance` are zipped at one face pair into a new `X|Y` film
  bounded by three Plateau borders; the merged faces' content feeds the new
  film. `propose_merges` uses the prepared face BVH with proximity-inflated
  bounds and a bounded candidate capacity (overflow refuses the search) and the
  contact owner's point-triangle and edge-edge distance kernels.
- `RegionSplitProposal`: a label whose face sides form several components is
  relabeled into children `f"{id}/{k}"` ordered by their smallest face id.
  Finite regions touched by a physical event split automatically when they
  consist of several positive shells; boundary labels split only on request.

Detection searches return `SurfaceEventSearch` with candidate counts,
capacities, rejected and uncertified candidates and certificate work.

## Seeds

`seed_from_vertex_tissue` converts an oriented 3D polyhedral
`VertexTissuePlan` (fan-triangulated polygon faces, owner cells from
`cell_face_orientations`) into a seed without modifying the tissue.
`seed_sphere`, `seed_double_bubble` (the closed-form equal-tension double
bubble) and `seed_catenoid` build reference geometries, and
`MultiRegionSurfaceSeed.subdivided` refines any seed with shared edge midpoints
so junction edges remain junction edges. Named vertex sets (`"junction"`,
`"ring-lower"`, ...) carry wire and junction identities through subdivision.

### Hard-label extraction

`LabelFieldSurfaceExtractionPlan` converts a three-dimensional uniform-grid or
sparse-coordinate `threshold_dynamics.LabelFieldState` into an explicit seed.
`plan.prepare(grid_shape)` binds a dense row-major grid;
`plan.prepare(grid_shape, site_coordinates=...)` binds the sparse state's
lexicographic integer coordinates and materializes only their occupied
coordinate bounding box, not the full declared grid. Grid points are physical
cell centers: `origin + (i, j, k) * spacing`. Missing sparse sites inside that
box and a one-site outer collar carry `boundary_region_id`. The boundary ID may
name a source label or a new synthetic ambient label.

Every cube receives the same globally conforming Freudenthal six-tetrahedron
split. Junction-conforming multi-label marching tetrahedra place interface
vertices at label-transition edge midpoints, at transition centroids on primal
faces and at the corresponding pair/triple/quadruple material point inside each
tetrahedron. Every flag `edge < face < tetrahedron` whose primal edge endpoints
have different labels gives one oriented triangle. A three-label primal face
therefore gives one shared valence-three junction edge; four labels in a
tetrahedron give six pair sheets and the complete four-region graph. The
tetrahedral split resolves multi-label and checkerboard cube ambiguity
deterministically. Evidence reports the number of ambiguous cubes and
multi-label tetrahedra, and `maximum_ambiguous_cells` is a fail-closed ambiguity
admission bound.

Source label IDs must exactly match the state's stable ordered label identity
and map unchanged to explicit region IDs. Inactive, empty labels are omitted;
the region-source map, source epoch/time, and source binding, route,
preparation, and site identities are recorded in `LabelFieldSurfaceLineage`.
The complete source state is fingerprinted but is not retained or synchronized.
Threshold dynamics remains label authority before extraction.
After an accepted extraction, the returned topology/state/prepared surface is
the independent geometry authority. A rejected result may expose only a
diagnostic `candidate_seed`, never an accepted E state.

Before authority is returned, the candidate must fit
`MultiRegionSurfaceCapacityPlan`, pass the selected validation profile and pass
the exact collision certificate. Refusals distinguish ambiguity capacity,
surface capacity, unsupported dry-foam valence and other validation failures.
Evidence compares finite-region volume and pair area against the independent
voxel estimator, names every region pair, and carries the complete E validation
and capacity certificates. Periodic wrapping is never inferred: sparse periodic
C states are clipped into a free-space seed, so a region crossing the periodic
seam can be refused as disconnected. A future unwrapped-periodic E
representation is required for a true periodic conversion.

`tools/threshold_surface_seeding_qualification.py` measures two-phase sphere
volume/area convergence, certifies a three-cell junction and foam-equilibrium
preparation, repeats extraction to prove deterministic lineage, and exercises
ambiguity and topology-capacity refusals.

For the radius-0.3 sphere at 5, 7 and 9 sites per axis, analytic volume errors
are 20.3%, 21.4% and 5.4%, and area errors are 50.5%, 13.7% and 17.4%.
Hard-label digitization is not adjacent-level monotone; the coarse-to-fine
endpoint orders are positive (2.25 for volume, 1.82 for area), and the finest
errors satisfy the declared 10%/25% candidate bounds. No asymptotic order is
claimed.

## Nonclaims

- No periodic surfaces until unwrapped coordinates are represented.
- Hard-label extraction places transitions at lattice-edge midpoints; it does
  not infer a signed distance, subcell volume fraction or curvature.
- No differentiability across any event pass.
- No label extinction (a region losing its last face refuses the event), no
  film rupture and no E-to-T (border-to-film) T1 direction: a T1 pop always
  starts from a vanishing film. A pop of a film with more than three borders
  yields a border of that valence, which the dry-foam profile refuses.
- Merges zip one face pair per event; the merged-wall growth is left to the
  motion law. The merge distance is a declared threshold, not a drainage law.
- CCD certifies local event motions only; fixed-topology motion between passes
  is not collision-certified here.
- No dynamic evolution; motion laws belong to the foam application.

## References

- J. E. Taylor, The structure of singularities in soap-bubble-like and
  soap-film-like minimal surfaces, Ann. Math. 103, 489-539 (1976).
- K. A. Brakke, The Surface Evolver, Experimental Mathematics 1, 141-165 (1992).
- J. R. Shewchuk, Adaptive precision floating-point arithmetic and fast robust
  geometric predicates, Discrete Comput. Geom. 18, 305-363 (1997).
- D. Weaire, N. Rivier, Soap, cells and statistics, Contemp. Phys. 25, 59-99
  (1984).
- A. Doi, A. Koide, An efficient method of triangulating equivalued surfaces by
  using tetrahedral cells, IEICE Trans. E74, 214-224 (1991).
- F. Da, C. Batty, E. Grinspun, Multimaterial mesh-based surface tracking, ACM
  Trans. Graph. 33, 112 (2014), doi:10.1145/2601097.2601146.
- T. Brochu, R. Bridson, Robust topological operations for dynamic explicit
  surfaces, SIAM J. Sci. Comput. 31, 2472-2493 (2009).
- R. E. Goldstein, A. I. Pesci, C. Raufaste, J. D. Shemilt, Geometry of
  catenoidal soap film collapse induced by boundary deformation, Phys. Rev. E
  104, 035105 (2021).
