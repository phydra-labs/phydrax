# Native meshing kernels

Goal: Phydrax's own meshing is the fastest of the measured tools on every relevant row, natively, with
certification of the produced mesh as a separate after-the-fact step. Design, evidence and baselines:
the redesign document and its ledgers (`.tmp/NATIVE_MESHING_DESIGN.md`, `.tmp/ledger/`, `.tmp/studies/`
in the `native-cad-meshing` worktree; published to the board as `artifacts/MAC/native-redesign/`).

User decisions (Nov 19): one native mesher for both modes; certify after construction and repair locally;
certified results and identities change, guarantees stay; external libraries are baselines and test
references only, never product dependencies; first rows after the foundations: closed surface to
tetrahedra, then implicit volumes, then the CAD solid.

## Rules for every slice

- Acceptance = the slice's ledger row is beaten at equal or better quality, measured by the in-repository
  benchmark (S0-D) on one thread, plus the slice's correctness tests. A slice that does not beat its row
  is not done.
- Construction kernels contain no proof that steers construction and no Python in an inner loop. Exact
  arithmetic only inside filtered predicates.
- Certification is a separate verifier over (mesh, source, tolerances) and reports pass, fail with
  findings, or unresolved.
- A certifier counts only after an independent read-only review of its proof. Every certifier reviewed
  so far had a soundness defect on first read, all fixed and re-read: `mesh_certify`,
  `zero_set_certify` (two-way, tube plus possible-zero cover), `verify_volume`, the native surface
  verifiers `verify_surface_embedding` and `verify_chart_partition`, the quadratic validity screen
  (correct on first read), the mapped-locator root discharge; `verify_cad_fidelity` (V3) is under
  re-read. Rows certified by an unreviewed certifier are construction rows with sampled quality.
- References are re-measured on the macOS machine with each reference and our row interleaved
  (reference, ours, repeated; minimum each), because machine state alone moves identical code by
  about 15 % between minutes. Gmsh HXT runs with its default optimizer (`Mesh.Optimize 1`): with 0 its
  meshes keep dihedral angles down to 0.1 degrees, far below our bar. Stored ceilings are a guide, not
  the acceptance.
- Every replaced route is a clean cutover: callers, tests, docs, changelog, generated data in the same
  change; no compatibility path.
- Determinism: the output depends on the input only (not on thread count or timing).
- Work runs on three machines (user decision Nov 19): macOS (S3, S0-B, S1, S7, integration), Linux
  (S2), Windows (S4); acceptance timings are taken on the macOS machine only, quiet, back to back with
  the reference. Native changes pass `ctest` with Apple clang and with each machine's compiler.
  Cross-platform byte identity is not required.

## S0 Foundations

### S0-A Geometry executors
- `native/meshcore/src/interval_program.cpp`, `interval_program.hpp`: split the validated bytecode from
  its executors; add a binary64 executor (batched, structure-of-arrays over rows, no per-row allocation)
  and a forward-jet executor (value, gradient, Hessian of up to 3 inputs) for the same instruction set;
  cheap certified path for sine and cosine of small arguments in the interval executor.
- `native/meshcore/include/phydrax_meshcore.h`, `src/core_capi.cpp`: entry points
  `phx_mc_program_evaluate` and `phx_mc_program_jet` beside the interval batch.
- `phydrax/geometry/_interval_native.py`, `phydrax/_meshcore.py`: bindings; the lowering is shared.
- Tests: `native/meshcore/tests/test_program_executors.cpp` (float equals the interval midpoint within
  rounding, interval contains float and jet at random points, jet equals finite differences of float to
  stated tolerance); `tests/unit/geometry/test_interval_native.py` additions.
- Acceptance: float executor at least 300 M rows/s on the ledger polynomial; trigonometric interval at
  least 10 M rows/s; enclosure test on 24 M samples.

### S0-B Delaunay kernels
- `native/meshcore/src/triangulation3d.hpp`, `cavity.hpp`: no per-insertion boundary-vertex sort and no
  removed-vertex check for unweighted insertion; neighbor patching over the cavity boundary instead of
  the hashed star link.
- `native/meshcore/src/mesh.hpp`, `delaunay2d.cpp`, `delaunay3d.cpp`: output canonicalization as an
  explicit option of the C entry point (callers that rely on it pass it).
- `native/meshcore/src/spatial_sort.hpp`: BRIO and Hilbert keys by radix sort of 64-bit keys.
- `native/meshcore/src/predicates.cpp`: stage-C expansion paths out of line (stack frame).
- Tests: existing Delaunay tests unchanged in meaning; cell sets equal to the previous kernel on random
  inputs (canonicalized).
- Acceptance: 3D Delaunay at 1e6 points at most 1.8x CGAL (ledger: CGAL 3.26 s, ours 9.08 s); 2D at
  most 2x CGAL.
- Earlier same-distribution measurements used different point and weight bytes on the two sides.
  Those timing ratios are provisional. Compare CGAL and Phydrax on the same ordered binary64
  points and weights, retain their SHA-256 digests, and require matching cell counts before
  comparing wall times. Output canonicalization stays mandatory for every caller.

### S0-C Flat tetrahedral state
- `native/meshcore/src/tet_mesh.hpp`, `tet_mesh_state.cpp`, `tet_mesh_capi.cpp`: cells, neighbors,
  vertices as flat arrays with the slot as identity; adjacency built by sorting packed face keys; no
  ordered maps on the creation or snapshot path; ancestry optional and off by default for construction.
- Tests: snapshot equality (as sets) against the previous state on the existing fixtures.
- Acceptance: creation of 1.16 M tets under 0.3 s (ledger: 4.5 s).

### S0-D In-repository benchmark
- `benchmarks/native_meshing.py` times fixed inputs and reports size, quality, and stage costs;
  `native_meshing_ceilings.json` records reference ceilings and provenance. For CGAL point-set
  comparisons, `--triangulation-inputs` reads the ledger's raw `p2_N.bin`, `p3_N.bin`, and `w3_N.bin`
  and reports hashes of the consumed arrays. An interleaved reference comparison uses those same
  files, rather than reproducible but different random seeds.
- Acceptance: reproduces the ledger's numbers for our kernels within 10%.

## S3 Closed surface to tetrahedra (first row after the foundations)

### S3-A Boundary recovery
- `native/meshcore/src/plc3d.cpp`: replace `convex_boundary()` (O(F N) exact loop, lines 1549-1567) by a
  hull or local dihedral test; Steiner-point suppression off by default for construction (interior
  Steiner points allowed; boundary Steiner points only where unavoidable and reported); seed location by
  walk.
- Acceptance: recovery of the ledger shapes (sphere 320 to 1.3 M triangles, torus 10k to 1 M, two
  cylinders) in linear time, at least as fast as CGAL CDT_3 (sphere 81,920: 1.36 s; torus 90,000: 2.74 s).

### S3-B Volume refinement and improvement
- `native/meshcore/src/refine3d.cpp`: local feature size precomputed per boundary vertex with a spatial
  grid instead of a scan per insertion; batch point generation for oversized cells with a proximity
  filter, Hilbert-ordered insertion; index heap instead of ordered sets; meets radius-edge 2 at a fixed
  boundary (protecting balls at small input angles).
- `native/meshcore/src/improve3d.cpp`: flat worklists; only bad cells.
- Acceptance: refinement plus improvement at or above Gmsh HXT single thread (0.34 / 0.63 / 0.77 M tets/s
  on sphere / torus / two cylinders at about 0.8 M tets) with minimum dihedral at least 15 degrees and
  radius-edge at most 2.
- Status (Nov 20, quiet acceptance 3 on `04ebe132d`, HXT with its default optimizer, torus inputs
  exported from the measured tree): sphere 0.46 / 0.56x, restricted torus 25 0.87x, grid torus 0.39 /
  0.32x; recovery 0.21-0.37x CGAL CDT_3. Restricted torus 72 (14k triangles) 1.031x block-timed and
  1.004x interleaved: open, decided in the next quiet interleaved window. The torus inputs are the
  restricted-Delaunay surfaces of facet edge 0.072 and 0.025 (stuffing removed).

### S3-C One-call pipeline and certification
- Pipeline (decided Nov 19, superseding a separate `surface_to_tets.cpp` / `phx_mc_surface_to_tets`
  entry): recovery (`phx_mc_plc3d_recover`), adoption of the recovery's native mesh
  (`TetMesh3D.from_recovery`, no array round trip), refinement and improvement on that native state,
  and one export. The state never leaves native memory between stages and the Python glue is a fixed
  handful of calls, so a single C entry would add an ABI without measurable gain; the benchmark's
  `surface-to-tets-*` rows time exactly this composition end to end.
- `native/meshcore/src/verify_volume.cpp` (new) and C entry `phx_mc_verify_volume`: exact orientation of
  every cell, combinatorial conformity, boundary equals the input surface, region consistency.
- Python: the PLC volume route (`phydrax/meshing/providers/_native_volume.py`,
  `_volume_generation.py`) cut over to that native-state pipeline; `prove=True` runs the verifier and the existing
  source-fidelity statement for the given surface; `prove=False` returns the candidate.
- Tests: route tests updated; the certificate rejects a mesh with a flipped cell, a missing boundary
  face, and a cell outside the surface.
- Acceptance: the S3 rows of the benchmark beaten; quick CAD profile passes with the new volume stage.
- Status (Nov 19): `phx_mc_verify_tetrahedra` / `phx_mc_verify_volume` / `phx_mc_tetrahedra_validity`
  decide embedding, region coverage and validity natively (`tetrahedral_degree`); `prove=False` returns
  `CellMeshingCandidate`, `candidate.prove()` the same result. Rejection tests (flipped cell, missing
  boundary face, cell outside the surface, wrong region label) and a fresh-process archive round trip
  pass. Certified icosphere execute under cProfile: L4 1.3 s (13,462 cells), L5 5.1 s (89,740), L6
  32.8 s (636,257; was 77.8 s). L6 split: native recovery + refinement + improvement about 2 s;
  `CellMesh` topology 5.9 s, source tables and size field 1.7 s, source associations 8.3 s (Equinox's
  static-field array walk over per-row identity tuples about 4.6 s of it), audit 11.0 s (welded
  topology 5.5, quality 3.6, evidence 1.0). NOT met: L6 "in seconds"; the remainder is generic numpy
  topology/quality work and per-row association tuples, not per-row Python loops.
- Update (Nov 19, certify2): the route adopts the native recovery (`TetMesh3D.from_recovery`); the
  acceptance audit takes the topology checks and default quality thresholds the degree theorem implies
  from the embedding certificate (`CellMeshAuditReport.tetrahedral_degree`); simplex `CellMesh`
  topology is built once with value-sorted packed row keys. Certified icosphere execute under
  cProfile: L4 0.99 s, L5 3.9 s, L6 23.6 s (19.3 s without the profiler). L6 split: native about
  1.9 s; `CellMesh` topology 3.45 s; source tables and size field 1.7 s; source associations 8.6 s
  (Equinox's static-field walk about 4.6 s of it); audit 4.4 s (sampled quality evaluation 2.7,
  coincident-vertex search 0.5, edge coverage 0.4, evidence 0.3). NOT met: L6 "in seconds".
- Update (Nov 19, certify3): bitwise-identical host quality (cell chunks, component-wise corner
  measures and dihedral angles, component `binary64_matmul`), cheaper topology helpers, simplex
  top incidence from the cell routes, verified connectivity routes in PLC associations, vectorized
  size-control selection, sweep-proposed coincident pairs, `StaticRows` for
  `MeshingScope.unqualified_entity_ids`; identities unchanged. Certified icosphere execute
  without the profiler (one-facet driver, load 2-10): L4 0.55 s, L5 1.9 s, L6 10.3-10.5 s (was
  17.1). NOT met: L6 <= 8 s; the remainder is native about 1.9 s, simplex topology about 2.1 s
  (identity sorts and searches), association construction about 1.8 s (per-row identity JSON
  and SHA-256 of about 200 MB), host quality about 0.7 s.
- Suppression coverage (Nov 19, hygiene; S3-C item 7): the four Steiner-suppression fixtures that
  flips now recover without a split are kept as such; the exhaustive search, the `PlcSearchBudget`
  accounting (exact budget, one state short, fewer bytes than one chunk: exceeded/inconclusive)
  and relocation by peeling (`PEEL_AFTER_INCONCLUSIVE`) are exercised again end to end under FIXED
  by the sliver sheet with vertex 18 raised by 2^-40 (native
  `test_perturbed_sliver_split_is_searched_and_peeled_within_its_budget`; Python
  `recover_plc_3d` evidence and the `generate_plc_volume` route with a declared budget in
  `test_native_volume_generation.py`). Kept, not deleted: these paths are reachable. NOT
  covered end to end: wedge merging (`merges > 0`); about 15k FIXED variants (twisted n-prisms
  and towers of them, jittered sliver/cone/lobed patches, single and paired coordinate
  perturbations of the sliver sheet, each under default and starved budgets) found none, and no
  unreachability proof exists, so the path and its evidence fields stay. The same sweep found two
  FIXED recoveries (a 6-vertex twisted prism and a 20-point tower) taking 427 s and over 600 s; the
  cause was subface legalization rescanning every subface after each flip, fixed by an edge-map
  worklist (4.65 s and 12.4 s, identical output; pinned as native regressions).

## Later slices (planned when S3 is accepted; same rules)
- S1 implicit field to tetrahedra by restricted Delaunay refinement, surface first
  (`restricted_delaunay.cpp`, `box_tree.cpp`; Boissonnat and Oudot 2005, Rineau and Yvinec 2007),
  the analytic profile route `implicit_restricted_delaunay`. Seeds: the possible-zero box tree of the
  interval program (`possible_zero_boxes`, the one subdivision owner) down to the facet edge bound,
  one root per sign-change edge of a kept box, so every component is seeded by enclosure inclusion.
  One S0-B Delaunay triangulation: restricted facets refined at their surface ball centers for edge
  (the certified facet bound d^2 / (6 (R - d)) at 0.95 of the tolerance), 30 degree angle and ball
  distance; inside cells at their circumcenters for radius-edge 1.98, redirected to the restricted
  facet whose surface ball they encroach. Termination by packing: no insertion within a separation
  bound (a quarter of min(e / 2, d, size)) of a vertex; a refused facet still violating the criteria is
  PHX_MC_UNRESOLVED, a construction failure. Surface repair by failure kind (box roots of uncovered
  boxes, ball centers at singular vertices, a finer edge bound), boundary vertices projected onto the
  zero set, slivers removed by edge and face removal and relocation to 15.5 degrees, leftover
  circumcenters inserted (three rounds), the S3 fixed-boundary improvement as last fallback.
  `execute()` re-seeds facets above the certified deviation and certifies. Isosurface stuffing was
  measured on every implicit model and test (restricted 2-10x faster with 4-9x fewer cells, all
  certified) and deleted. Status (Nov 19, loaded): sphere 1.4 / 2.1 ms with 442 / 827 cells, torus
  21 / 34 ms with 3,990 / 6,695 cells, minimum dihedral 15.5 degrees or better, all four certify;
  quiet back-to-back run against CGAL opt=0 (sphere) and Gmsh (torus) pending.
- S2 CAD solid to surface and volume, then P2 (design: `.tmp/S2_DESIGN.md` in `native-cad-meshing`):
  one native call `phx_mc_cad_mesh_surface` or `phx_mc_cad_mesh_volume` on a flat geometry bundle;
  curves split once by integrating the size field and shared by every use; faces are meshed in their
  native parameter chart by metric Delaunay
  refinement that never splits a curve, then physical flips and smoothing; seams as two chart copies
  of one curve, poles by float ring caps; volume by S3 with the boundary fixed; P2 nodes from stored
  parameters with Bernstein validity checks; certification after construction (existing certifiers
  first, then native verifiers) with local repair. Slices S2-A geometry bundle, S2-B curves, S2-C
  metric refinement kernel, S2-D faces and surface assembly, S2-E solid route cutover, S2-F native
  verifiers, S2-G P2. Decisions (Nov 19): every face kind now, including B-spline/NURBS and T-spline
  (one path, no staged split); construction-ancestry fields leave the CAD witness; closed-form
  quadric evaluators only with benchmark evidence; P2 through the target's geometry order. Targets:
  Gmsh HXT per size (0.0215 / 0.0653 / 0.303 / 2.00 s); curving at or below Netgen's 0.0125 / 0.075 s
  with certified validity.
  Quadric benchmarks use the per-face interpolation budget κmax·h²/4 from the constant cylinder
  curvature and a construction margin of θ = 0.8. Compare at the same requested h and report
  element counts, both directed sampled distances, and minimum boundary angle. Construction times
  do not include the separate proof; sampling alone does not certify fidelity.
- S4 surface remesh and metric adaptation (above MMGS and MMG3D, inside tolerance).
- S5 STEP import and remaining exchange formats. S7 threads and devices at scale.
- S6 layers, hex-dominant, overset, Voronoi (MAC, planned Nov 20; ledger `cad.md` F8/F9 and
  `providers.md` F13/F14, all run on this Mac; references installed: TIOGA `tioga_drv`, VoroCrust
  `vc_mesh`, OpenFOAM v2606 with snappyHexMesh and cfMesh, Gmsh, Netgen, NekMesh). Today none of the
  four has a meshcore kernel for its main algorithm and `native_meshing.py` has no rows. Each sub-slice
  is one native call on flat arrays, construction in floating point with exact predicates,
  certification after construction with local repair, `prove=True/False` through the candidate, and a
  clean cutover of the existing Python route. Order:
  - S6-A overset assembly (F13; TIOGA one rank: hex 0.265 / 0.430 / 1.377 s, tets 0.053 / 0.496 /
    0.871 s at 1e4 / 1e5 / 1e6 cells; Phydrax today 72.9 s at 1e4 tets with 116 orphans and no hex).
    Hole cutting against the wall by a classified hole map (exact winding/orientation), fringe layers by
    adjacency, donor search on the packed BVH with exact barycentric containment for tets and a
    Krawczyk-certified inverse for trilinear hexes, the finest valid non-fringe donor, orphans reported
    as evidence (TIOGA's tet runs leave 14 and 5 silently).
    Design (Nov 20). One call, `phx_mc_overset_assemble`, on flat blocks: world-frame vertices and
    cells (rows of 8, tets use 4) per part, each part's exterior faces with a role (other, wall,
    receptor boundary), excluded cells, cell IDs and priorities. Stages: (1) hole map: each part's
    vertices against every other part's wall by +x ray parity with exact orient2d/orient3d and
    symbolic tie-breaking (the rule of the Python owner); wall quads count as their two triangles
    and a vertex in the closed corner tetrahedron of a nonplanar quad (which holds the bilinear patch
    and both triangulations) is wall contact, hence hole; (2) cut cells: a cell is cut unless an
    exact separating plane (a planar cell face or a face of the wall element's corner hull) or
    disjoint bounds prove it misses every wall element (conservative); excluded cells are cut;
    (3) fringe layers by cell adjacency and receptor-boundary vertices, wall vertices protected;
    (4) donor search through a transient per-part cell-bin index, float barycentric (tets) or
    Newton (trilinear hexes) location, candidates ordered by the policy key (finest valid ACTIVE
    donor, priority, name, ID), non-ACTIVE containing cells recorded as exclusions; (5)
    certification after construction walks each receptor's ordered candidates and accepts the
    first exactly contained one (orient3d signs for tets; for hexes a Krawczyk interval enclosure
    of the trilinear inverse inside the closed unit cube), so a float candidate that fails the
    exact check is repaired by the next; receptors without a certified donor are orphans with
    their status; (6) a cut cell with no hole vertex is ambiguous unless coverage is proved: a
    vertex of it lies (certified) in the wall part and its bounds miss all non-wall exterior faces
    of that part. `prove=False` stops after construction (donor coverage then reads unproved).
    `prepare_overset_connectivity` uses the call for serial 3D affine-tet / trilinear-hex parts
    without a source solid query when meshcore loads; the Python owner stays the documented path
    for 2D, polyhedral, curved and solid-query parts and for a missing library. The packed BVH is
    Python-built; building one per call costs more than the reference's whole call, so the kernel
    keeps its own transient bin index.
    The `overset-hex` / `overset-tet` rows time the public `prepare_overset_connectivity(prove=False)`
    on certified parts built before timing (as TIOGA registers its grids before its timed call);
    the certified (`prove=True`) call and the kernel's own clock are reported untimed. On the
    native path the call builds no mapped-cell locators or finite-element measures: cell rows and
    corner boxes come from the blocks in bulk and donor measures from the kernel.
  - S6-C prism layers on a tetrahedral mesh (F9; 5 layers, first 0.01, growth 1.2; Gmsh 0.309 s on the
    sphere at h 0.1, cfMesh 1.77 s at h 0.05; on the quadric Gmsh fails inward, Netgen thins to 17 % at
    edges, cfMesh 0.456 / 1.09 s). Native growth directions and heights with certified non-intersection,
    thinning only where the certificate requires it, the core by S3 with the top layer fixed; full depth
    measured by true distance.
    Timed scopes. Gmsh's 0.309 s is classify + extrudeBoundaryLayer + generate(3) from its surface
    mesh (layers and core); cfMesh's 0.545 / 1.77 / 0.456 / 1.09 s are generateBoundaryLayers alone
    on a tetMesh made first (3.2-5.8 s, untimed); Netgen's is the whole GenerateMesh. Our rows time
    the larger scope, Gmsh's: the Gmsh surface mesh in, certified layers, the S3 core with the top
    level fixed, one conforming prism + tet mesh out; the layer step is reported separately.
    Design and result (Nov 20). One call, `phx_mc_prism_layers` (`_meshcore.prism_layers`), on a
    closed oriented triangle wall and cumulative heights: one straight column per wall vertex (the
    SMOOTH topology). Construction in floating point: feature edges by dihedral angle split vertex
    fans into smooth sectors; per vertex and sector, the normal and second fundamental form come from
    the source surface where the wall carries per-corner source frames (a CAD wall: the patch's
    inward normal and shape operator at each corner's chart point, so levels are measured from the
    source surface itself, 1e-5 of each level on the coarse CAD cylinder, rims included; the frames
    are retained, fingerprinted and archived, and a cold restore reruns the kernel on them), else
    from a weighted quadratic jet through the wall vertices (no one-sided bias at a ridge); smooth
    columns grow along the normal by D, ridge and corner columns along the minimum-norm medial w
    with w . g_i = a_i, a_i the
    curvature-corrected height of sector i (the inward offset of a convex ridge is exact); a
    screened Laplacian (decay four times the largest ridge offset, Gauss-Seidel worklist to 1e-4 D)
    carries ridge offsets into neighboring smooth columns, each corrected for curvature along its
    offset; float Bernstein checks thin failing columns (x 0.8). Certification (`prove`): the nine
    Bernstein controls of every prism Jacobian (six corners by orient3d, three mixed by a filtered
    expansion) and the top level embedded and disjoint from the wall by exact triangle contacts
    (uniform grid), failures thinned and recertified down to `minimum_scale` (else UNRESOLVED).
    Interleaved against the best reference (minimum of three each; ours certified): sphere h 0.1
    0.052 s vs Gmsh 0.315 s; sphere h 0.05 0.232 s vs cfMesh 1.85 s; quadric h 0.1 0.041 s vs
    cfMesh 0.463 s; quadric h 0.05 0.187 s vs cfMesh 1.14 s (layer step 0.024 / 0.085 / 0.017 /
    0.062 s). 0 inverted prisms and tets, 0 generated nodes outside, no thinning; top depth by true
    distance 0.074416 +- 1e-9 on the sphere (Netgen 0.0740-0.0744, Gmsh 0.0744-0.0767) and
    0.07438-0.07442 on the quadric (Netgen thins to 0.0124 / 0.0081); minimum corner ratio 0.957
    (sphere), 0.768 / 0.736 (quadric, all at the sharp edges; Netgen 0.31 / 0.22, NekMesh 0.45).
    `prepare_boundary_layers` stays the owner of the public ADVANCING contract: it publishes
    rational per-cell validity certificates (digests the native cell-validity bridge does not decide
    for prisms), warped-wall contact proofs, column and front-row ancestry, FAN ridge splitting with
    corner patches, rim columns, obstacles, periodic and quadrilateral walls (today 126 s on the F9
    sphere and 90 s on the quadric at h 0.1, layers only; its cell certification is 174 of 206 s
    under the profiler). Cutting it over needs native emission of those certificates.
  - S6-B conforming Voronoi (F14; VoroCrust 5.2-40.5 s, 95 % surface sampling; its explicit-mesh step
    0.04-0.67 s, our restricted power cells 26-56x slower). Native conforming seeds (surface ball
    pairs, interior seeds by the size field) and restricted cells taken from the Delaunay dual, clipping
    only boundary cells. Oblique planar facets (Nov 21, decided): snapped mirror reflections across a
    non-dyadic plane are never exactly cospherical, so merging their snap-level Voronoi vertex
    clusters (fidelity 2.5e-3-8e-3 h to about 1e-11 h) makes faces through the merged vertex
    non-planar and breaks exact cell convexity (rotated cube h 0.25: 158 of 170 cells; F14 cube
    and Steinmetz rows lose convex cells). Exact convexity is kept: clusters are resolved by the
    existing rigid and lone seed moves, with the fidelity bound charged as today.
  - S6-D hex-dominant with snapping (F8; cfMesh 5.77-8.59 s at ~1e5 cells, hex fraction 0.91-0.96,
    Hausdorff 2.0-2.5e-4). An octree core snapped to the surface with feature edges, transition cells,
    optional layers by S6-C; hex fraction, Hausdorff, non-orthogonality and skew at or better than
    cfMesh's. The tet-to-hex split route stays only if a row needs it.
    Design (Nov 20). One call, `phx_mc_hex_dominant` (`_meshcore.hex_dominant`), on a closed
    oriented triangle surface with its facet rows, the core size h, the boundary size h / 2^k, the
    feature angle and an optional exact layer thickness schedule. The measured reference is cfMesh
    `cartesianMesh`: its 77,896-cell sphere has every non-hexahedron in the h / (h/2) transition shell
    (9-, 12- and 15-face cells) and reaches 18.7 degrees there only because it smooths; an unsmoothed
    2:1 octree face has 25.2 degrees. Construction (floating point, exact predicates):
    (1) surface topology (closed oriented manifold), feature edges by dihedral angle and facet
    change, smooth patches between them, feature curves and corners; a transient triangle bin
    index for closest points; (2) inside/outside of lattice vertices by +x ray parity per lattice
    line with exact orient2d/orient3d and symbolic perturbation; (3) a balanced octree from size h:
    a cell whose L-infinity distance to a triangle box is below its size is split down to the
    boundary size (this alone gives 2:1 balance over all 26 neighbors); (4) core cells: inside and,
    at the finest level, every corner at least half a boundary cell (plus the layer stack) from the
    surface and no triangle in the box; manifold repair of the staircase by removing finest cells at
    non-manifold edges and vertices; (5) the face-based mesh with hanging nodes: same-level faces
    carry the midpoints of finer neighbors, so 2:1 coarse cells become polyhedra (OpenFOAM's
    convention, as cfMesh's); (6) one projection layer of hexahedra between the staircase and its
    closest-point image: staircase faces are assigned to patches, the partition is cleaned
    (faces with three or four foreign edge neighbors flip, non-manifold chain vertices merge) and
    chain vertices snap to the feature curve between their two patches, corner vertices to feature
    corners; (7) smoothing: interior vertices of the fine band move to the mean of their cells'
    centroids (this tilts the hanging faces towards the fine cells, the mechanism that lowers
    cfMesh's transition non-orthogonality), surface vertices to the mean of their faces' centroids
    projected back onto their patch or feature curve; a move is kept only when every incident cell
    keeps positive float corner and decomposition volumes; (8) layers: L thin hexahedra per
    projection column at the exact cumulative thicknesses from the wall (hex layers: the column is
    split, so the hex fraction rises). Publication: every cell is a polyhedron with OpenFOAM face
    semantics (a non-planar face is the fan from its exact vertex average; the certifier's
    `polytope_face_triangles`), target `CellFamilyPolicy(required=("polyhedron",))`, hex fraction
    by shape from the `cell_shape` marker (0 six quadrilaterals, 1 step prism, 2 other).
    Certification (prove): every cell by exact orient3d of its fan decomposition tetrahedra about
    both its centroid (OpenFOAM's decomposition) and its exact distinct-vertex average (the
    certifier's local map); smoothing and repair keep both stars, so prove=False publishes only
    cells prove=True can certify; plus a closed genus-0 boundary, hence star-shaped and embedded;
    anything failing is UNRESOLVED. The boundary distance (vertices on the surface,
    sampled face points) is reported; the route requests `volume_plc_fidelity` (the Voronoi
    route's contract, over the exact fan wall) against `maximum_deviation` or the boundary size.
    Route: `plc_octree_hex_dominant` for `HEX_DOMINANT_POLYHEDRAL_TRANSITION` (sizes from a volume
    and a boundary uniform size control, layers from a PROVIDER `BoundaryLayerControl`). The
    tet-to-hex split (`plc_hex_dominant`, `HEX_DOMINANT_SIMPLEX_TRANSITION`) stays: it is the
    route for consumers needing trilinear hexahedra, and the only one whose boundary conforms
    exactly to the PLC with region interfaces and simplex transitions, which a snapped octree
    cannot offer by construction.
    Feature curves (Nov 21, accepted). Closed: curve-local vertex moves, staircase-piece and
    four-way splits, wall-cap splits and mixed octree levels below b at the curve (the last
    certified on quadric L0 but cost +13 % cells and left uncertified collapsed wall pairs on
    L2). Adopted, after cfMesh's `boundaryLayers` (studied stage by stage): at a convex feature
    curve the wall layers do not wrap the curve; each side's layer points slide along the other
    patch (t at most 0.9 of the distance to the far edge of the neighboring wall faces), each
    side's columns reattach to its slid points, and one corner hexahedron column per curve
    segment carries a wall strip on each patch. With layers, staircase steps split along their
    corner's own diagonal so the curve keeps every staircase vertex; a curve segment whose chord
    lies farther from the curve than any wall edge from its patch gains a midpoint on the exact
    curve (scale-free: no tolerance input; an x8 scaled fixture gives identical connectivity);
    patch vertices next to the curve slide along their patch only when local skewness and wall
    deviation do not increase. Quadric L2: Hausdorff 2.83e-4 to 1.905e-4, skewness 0.992 to
    0.745. The benchmark's surface-to-mesh distance is exact (16 nearest centroids give a bound,
    every triangle within centroid distance minus its largest vertex distance from the centroid
    is checked), equal to brute force
    bitwise; the F8 cfMesh values did not move.
    Accepted rows (prove=False, three interleaved rounds against cfMesh; seconds ours / cfMesh):
    sphere L0 5.05-5.36 / 8.96-9.17, sphere L2 5.35-5.68 / 8.45-8.76, quadric L0 4.55-4.67 /
    6.46-6.58, quadric L2 5.32-5.47 / 5.99-6.06, every quality field at least cfMesh's. Public
    prove=True certifies each row's mesh at row size on kernels `dbbd0182c` (native contacts;
    digests equal to the rational contact pass): quadric L0 77,470 cells 223 s, sphere L0 89,219
    cells 284 s, quadric L2 83,160 cells 230 s, sphere L2 98,038 cells 270 s (process peaks
    13-22 GB, mostly the dense certificate). Open: per-point denominators in the native star
    kernels (cell-common denominators reach 200-3,250 bits on Voronoi cells).
  - Debt (narrowed): `test_native_layer_core_mixed_round_trip_preserves_source_strata` runs
    again (fixture at a 0.12 core offset: 8x8 cap, 1,280 cells, same topology classes; 447 s,
    3.3 GB peak). Not a BVH rebuild (column certification already prepares one wall and one
    query tree per bank). Open: the exact refine / restore / replay / acceptance pipeline costs
    about 75 ms and retains about 0.5 MB per cell, linearly (1,280 → 5,120 cells: refinement
    96 → 372 s, +0.8 → +2.5 GB), led by the refined-mesh global embedding, source-association
    propagation and boundary-image witness renewal, so the archived 0.03-offset ridge (20,480
    cells) still needs over 16 GB.

## Row ownership (every ledger row; `.tmp/ledger/`)

| Ledger rows | Reference to beat (single thread) | Slice | Machine |
|---|---|---|---|
| F1 point-set Delaunay 2D/3D, 1e3-1e6 | CGAL DT2 / DT3 | S0-B | macOS |
| Regular triangulation 3D, periodic Delaunay (native-kernels §2-3) | CGAL Regular_3 (2.6-2.9x today), Periodic_3 (5x) | S0-B | macOS |
| F1 planar CDT; q20 unit square and ring-slit; q30 square with polygonal hole and 100-spike star | Triangle `pcQ`, `pq20a`, and `pq30a` on their respective fixtures | S2-C (`cdt2d`) | Linux |
| F3 surface to tetrahedra; recovery; refinement; export | Gmsh HXT on the same surface; CGAL CDT_3 | S3 | macOS |
| Improvement passes on a given mesh (native-kernels §3) | MMG3D `-optim` at equal quality | S3-B | macOS |
| F4 implicit to tetrahedra | CGAL make_mesh_3, Gmsh | S1 | macOS |
| F2 CAD solid, F5 curved order 2/3 | Gmsh HXT per size; Netgen `Curve`, Gmsh `setOrder` | S2 | Linux |
| F7 surface remesh and decimation (1e6 to 1e3-1e5, dense and noisy torus; sphere ladder) | MMGS, CGAL isotropic remeshing | S4-Surface | Windows |
| F6 / F6b metric adaptation | MMG3D, Omega_h | S4-3D | Windows |
| F10 queries (closest point, signed distance, winding number, point location) | libigl AABB / FastWindingNumberBVH | S0 queries | macOS after S3 |
| F8 hex-dominant, F9 prism layers, F13 overset, F14 Voronoi | cfMesh / snappyHexMesh, Gmsh / Netgen, TIOGA, VoroCrust | S6 (A overset, C layers, B Voronoi, D hex-dominant) | macOS |
| STEP import and exchange formats (spline faces are S2, decision D1) | the S2 rows on the STEP models (washer, NIST) | S5 | Linux after S2 |
| Threads and devices | each row above at 1-8 threads | S7 | macOS |

The F1 q30 size ladder uses h = 0.0431, 0.01362, 0.00431, and 0.001362, with the
same points, segments, hole seeds, and equilateral area limit on both sides.
Measure covered area, positive orientation, and maximum area as well as the
minimum angle. The star's input tips force angles smaller than 30 degrees;
report the actual minimum and the number of triangles below the requested target.

### Curved CAD qualification

The `cad-quadric-p2`, `cad-quadric-p3`, `cad-sphere-p2`, and `cad-sphere-p3`
rows use the original F5 sizes. They time native curving only, against
Netgen's archived `Curve(p)` times. The quality check runs outside the clock:
it certifies determinant positivity on the exported map and measures scaled
Jacobian, determinant ratio, and analytic boundary residual on the same
sampling lattice as the Netgen/Gmsh comparison. The sampled surface residual
does not replace the separate bidirectional CAD fidelity or embedding proof.

The macOS results across all 12 rows beat every curving clock and certified
every cell's determinant, but no row matched all reference quality fields. The
quadric rows met the boundary residual thresholds and missed minimum scaled
Jacobian. Blending in the spherical chart's unit frame brought all six sphere
rows within their boundary residual limits without losing validity or the
curving time advantage. Minimum scaled Jacobian and determinant ratio still
miss on all six sphere rows. Keep F5 open until the combined gates pass.

### Matched fixed-wall volume qualification

The original torus25 case uses the same archived 122,926 fixed boundary
triangles and numeric size request for both tools. Phydrax times site
preparation, recovery, state adoption, refinement, improvement, and export.
Gmsh 4.15.2 times HXT `generate(3)` with its optimizer enabled. These are
the best samples of three interleaved rounds on macOS:

| Workers | Phydrax (s) | HXT (s) |
|---:|---:|---:|
| 1 | 2.010 | 4.020 |
| 2 | 1.890 | 2.621 |
| 4 | 1.612 | 1.950 |
| 8 | 1.475 | 1.754 |

Both preserved the literal input wall. Phydrax published 1,522,556 cells
with identical full output banks at every worker count. The independent exact
affine embedding check certified the mesh, and common numerical measurements
found zero nonpositive tetrahedra on both tools. Phydrax met the 15° minimum
dihedral, radius-edge 2, and circumradius h targets. HXT's measured minima
were 12.92–13.39°; its size setting is a characteristic length, not the same
circumradius bound. Output counts differ; this is a same-input construction
comparison, not equal-cell throughput or a proof-time comparison.

All 24 original comparisons for the two spheres, the two restricted tori,
and the two grid tori passed at 1, 2, 4, and 8 workers. Every native case
kept its fixed wall, met all requested quality targets, and reproduced its
full output banks across worker counts. The smaller restricted torus
still uses sequential point insertion, with parallel feature-size queries;
its 8-worker construction took 0.115 s versus 0.121 s from HXT. The smaller
grid torus still takes the ordinary fixed-wall route. With the new size
dispatch, its measured 8-worker time rose from 0.243 s to 0.292 s, still
below HXT's 0.630 s.

### Matched affine point-location qualification

All eight F10 affine location comparisons passed against libigl 2.6.3 on the
original query and tetrahedron banks. At eight workers, 100,000 queries took
18.23 ms versus 19.59 ms; one million took 128.49 ms versus 179.77 ms.
Three interleaved samples per row, with the acceleration structures prepared
before timing. Both providers found the same points with zero wrong cells
in the independent check, and Phydrax cell IDs were bit-identical across
worker counts. Preparing the native preorder and taking private snapshots of
unsealed host buffers took about 9.4 ms on this fixture. The native index
and retained snapshots occupy about 76 MiB. A separate cold run measured
29.62 s for the whole finite-element locator setup. These comparisons time
queries only.

### Qualified sequential routes in the thread matrix

- F7 implicit and input-mesh remeshing remain sequential: each edit reads
  the surface topology produced by earlier edits. The measured MMGS and
  CGAL isotropic-remeshing drivers are also serial. The original 13 serial
  construction comparisons passed with separate two-direction certification.
  On the integrated locator revision, all 13 cases returned byte-identical
  native result banks and zero unmet constraints at worker requests of
  1, 2, 4, and 8. This is worker invariance, not a parallel speedup claim.
- F13 overset remains sequential. On the same ordered binary input, it beats
  TIOGA at 1, 2, 4, and 8 MPI ranks for all three hex and all three tet sizes
  (three interleaved calls per row). At the largest sizes and eight ranks, the
  prepared candidate call takes 0.274 s vs 0.649 s for hexes and 0.182 s vs
  0.517 s for tets. All native connectivity IDs are invariant under worker
  requests; certified donors pass the independent check. These are assembly
  clocks, not cold process or proof times.
- F14 conforming Voronoi remains sequential. On the shared source geometry
  and archived size controls, all 32 native rows met the timing and archived
  quality ceilings. Of the 32 comparisons against VoroCrust at 1/2/4/8
  threads, 31 pass using the archived quality ceilings and 29 pass when
  scored against the fastest retained reference meshes using the same
  ledger metrics. On the small cube, Phydrax's maximum is 27 faces per cell
  versus 25 at one reference thread and 26 at four. The fastest small
  Steinmetz 4-thread reference had a sampled convexity finding. Native mesh
  and evidence IDs are invariant across worker requests. Output counts
  differ, and the large sphere retains the archived 5,000-cell reference
  target. These times compare candidate construction, not equal-cell
  throughput or proof.
- Periodic Delaunay and fixed-boundary recovery remain sequential in the
  matched comparisons against the sequential CGAL providers. All Phydrax
  output banks are invariant under worker requests. This does not qualify
  fixed-boundary volume refinement against threaded HXT.
