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
  is not done. All ten registered 2D and 3D graded F6 metric rows enforce `allowed_ratio = 1.0` against
  the faster of quiet single-threaded MMG and serial Omega_h.
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
  one native call `phx_mc_cad_mesh_surface` or `phx_mc_cad_tetrahedralize_solid` on a flat geometry bundle;
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
  For graded tetrahedral rows, run
  `OMP_NUM_THREADS=1 python -m benchmarks.native_meshing --rows metric-3d-coarsen --sizes 10000 --metric-inputs DIR`.
  Supply the ledger's ordered `cube1e4.npz` (`xyz` and `el`) and raw
  `cube1e4.coarsen.mbin`. The `iso` and `aniso` rows and the `cube1e5`/`cube1e6`
  sizes use the corresponding files. The benchmark reports ordered coordinate,
  connectivity, and raw metric SHA-256s; it does not regenerate inputs.
  A provider verdict requires all three digests to match the corresponding
  registered ceiling entry. An unverified row still emits measurements, but
  reports `reference_inputs_verified: false`, no pass/fail verdict, and a
  nonzero process status.
  Coarsening edge share uses the archived source sizes interpolated at target
  edge midpoints, not the transferred output metric. Mean and worst-cell shape
  are Euclidean for scalar targets and use the analytic centroid metric for
  anisotropic targets. The worst-cell shape must beat the unrounded best
  reference in addition to the band, mean-shape, zero-inversion,
  zero-out-of-band, and completion gates. Construction timing is separate from
  certification and transfer. All ten registered graded metric rows
  (`metric-3d-{iso,aniso,coarsen}` at 10k/100k/1M and `metric-2d-iso` at 100k)
  enforce `allowed_ratio = 1.0` against the faster of quiet single-threaded
  MMG 5.8.0 and serial Omega_h 9.34.13 on identical byte-verified inputs.
- S5 STEP import and remaining exchange formats. S7 threads and devices at scale.
  `OMP_NUM_THREADS=1 python -m benchmarks.native_meshing --rows cad-step-washer`
  times construction on the literal NIST STEP washer at h=.39663 mm and a
  40 µm wall target. It checks positive tetrahedra and the 18.5°–151.5°
  affine dihedral and 1.62 radius-edge targets outside the clock. Its output
  digest binds the mesh to the independent two-directed wall comparison.
  Import/repair, proof, and on-disk export are excluded.
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
    The interior body-centered cubic lattice is centered in the source bounding box so its
    phase is symmetric about each box midplane. The lattice spacing, clearance from the
    source wall, and reflected boundary seed construction are unchanged. This can change
    the mesh and evidence IDs; the same exact convexity and source-coverage checks apply.
    A repair round rounds exterior vertices for the source-distance leak check. Full
    exact coordinate banks and cell measures are published only after the last repair;
    the leakage criterion, repair order, and final certification remain unchanged.
  - S6-D hex-dominant with snapping (F8; cfMesh 5.77-8.59 s at ~1e5 cells, hex fraction 0.91-0.96,
    Hausdorff 2.0-2.5e-4). An octree core snapped to the surface with feature edges, transition cells,
    optional layers by S6-C; hex fraction, Hausdorff, non-orthogonality and skew at or better than
    cfMesh's. The tet-to-hex split route stays only if a row needs it.
    Construction calls `phx_mc_hex_dominant_build` (`_meshcore.hex_dominant`) on a closed
    oriented triangle surface with its facet rows, the core size h, the boundary size h / 2^k,
    the feature angle, an optional exact layer thickness schedule, and a worker cap.
    The measured reference is cfMesh
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
    Certification (`prove=True` or `candidate.prove()`): exact `orient3d` checks the fan
    decomposition tetrahedra of every cell about both its centroid (OpenFOAM's decomposition)
    and its distinct-vertex average (the published local map). The canonical acceptance schedule
    separately requires global embedding, topology, and quality, and certifies the two-sided
    distance between the mesh's exterior fan and the PLC exterior against `maximum_deviation`
    or the boundary size (`volume_plc_fidelity`, not exact PLC material coverage). The default
    `prove=False` returns an uncertified candidate; construction screens are not acceptance,
    and a failed proof refuses publication as a certified result.
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

### Original graded coarsening

The original construction reference is the authenticated 0.377 s, with the
approved 3.5× allowance (1.3195 s). The previous coarsening result missed
worst-cell shape at 0.03891 versus MMG3D's 0.47481, despite passing time.
With local coupled repairs, the same 10,452-cell input completes with zero
out-of-band edges or inversions. Minimum shape is 0.60997 versus 0.47481,
mean shape 0.83015 versus 0.82651, and midpoint band share 0.87257 versus
0.79391. It passes the registered gates at 1.092 s, the lowest of three
construction runs. This timing exception is not a speed win over the
0.337 s MMG3D rerun.

The regenerated Gmsh 4.15.2 cube banks use source sizes 0.0782, 0.0363,
and 0.01685. The smallest source and coarsening target match the original
digests byte for byte. MMG3D 5.8.0 ran all nine combinations with matching
source and target banks; their raw band share and mean shape reproduce
the original quality thresholds to three decimals. Reference clocks and
their input digests are recorded in `benchmarks/native_meshing_ceilings.json`.
These are reference runs, not Phydrax scaling results.

### Rapid graded adaptation comparison

Use the 388-tetrahedron feature box in the coordination board's
`artifacts/MAC/native-redesign/graded-metric-probe/`. It crosses the original
F6 spherical sizing shell and anisotropic midplane. Run one target at a time:

```sh
OMP_NUM_THREADS=1 PHX_MESHCORE_THREADS=1 python -m benchmarks.native_meshing_mmg3d \
  metric DIR --field iso --mmg3d /path/to/mmg3d_O3
```

Both tools use the same ordered source mesh and sampled target. The driver
reports construction separately from certification and transfer, and keeps
native completion and edge refusal evidence even when quality improves.
The probe is for iteration, not a substitute for ladder qualification.
On the earlier cover-fix image, the coarsening crop completed with no
inversions but missed both mean shape (0.78597 vs 0.79798) and worst-cell
shape (0.50297 vs 0.58136). Iso and aniso exceeded the earlier mean and
band thresholds but stalled at 21 and 26 residual edges respectively.

The rounded Cartesian hull split removes all 21 iso residuals; the crop
completes with zero inversions, mean shape 0.86120 (MMG3D 0.83983), and
midpoint band share 0.90261 (MMG3D 0.83795). Worst-cell shape still loses:
0.00241 versus 0.52073. Aniso falls from 26 residuals to eight with no
representability refusals. Its worst shape of 0.35736 beats MMG3D's
0.29750, but it still stalls. These are diagnostic comparisons,
not MMG3D timing wins or acceptance of the archived rows.

Both minimum and mean shape must also pass on the coarsening crop.
An independently retransported diagnostic on its unchanged connectivity
reaches 0.58362 minimum and 0.80534 mean with zero native out-of-band edges
and certified global embedding. It establishes feasibility, not that the
adapter constructs the result.

Native coupled repair clears both shape gates on the coarsening crop:
0.60218 minimum and 0.80042 mean, versus MMG3D's 0.58136 and 0.79798.
The isotropic crop passes its scientific gates, with 0.54005 minimum and
0.86074 mean versus 0.52073 and 0.83983. The anisotropic crop clears its
eight short edges using bounded coupled wall relocation and completes
with 0.39221 minimum and 0.77685 mean versus 0.29750 and 0.67732.
All three finish with zero out-of-band edges or inversions. Single
paired construction samples were 0.133 s vs 0.202 s for coarsening
and 1.612 s vs 0.356 s for anisotropic adaptation. The isotropic
checkpoint `a20d61cb8f` took 3.467 s vs 0.280 s. The two refinement
clocks still miss.

Reusing relocation-star measurements left coordinates, connectivity,
mesh, transition, lineage, stencil, metric, and evidence identities
identical on all three 388-cell cases. In one before/after construction
comparison, iso took 3.366 s then 2.989 s and aniso took 1.517 s then
1.494 s. These are local Phydrax samples, not new paired MMG3D clocks.

For explicit tensor fields, the adapter tries the coarse step range with
the deep search directions and carrier resolutions, and attempts coupled
repair when shallow relocation stops changing the mesh. The ordinary
exhaustive search is still required to report a stall, with the bounded
attempt cache cleared before fine steps. This intentionally changes edit
order, not the acceptance thresholds. On the same aniso input, a paired
sample was **0.823 s vs 0.361 s** for MMG3D. The 330-cell result had zero
out-of-band edges and inversions, minimum shape 0.39228 vs 0.29750, mean
0.78462 vs 0.67732, and midpoint band share 0.91225 vs 0.42265. The crop
still misses the construction clock and does not qualify larger rows.

Limiting the bounded tensor proposal to 16 optimizer steps (96 in the
exhaustive phase) and screening clearly in-band frozen endpoint norms
reduced one paired construction sample to **0.618 s vs 0.358 s**.
The result completes with no inversions or out-of-band edges. Worst shape
decreases slightly from 0.39228 to 0.38849, still above MMG3D's 0.29750.
Mean shape is 0.78020 vs 0.67732 and midpoint band share is 0.90758 vs
0.42265. This is a faster diagnostic, not a construction win.

After net scalar refinement, at most four interior mean-shape sweeps
precede the unchanged worst-cell repair. Mean stationarity is not a completion
criterion. The coarsening and tensor routes retain their prior polish.
On the isotropic crop the new paired construction sample is **2.027 s
vs 0.278 s**. Worst shape rises from 0.54005 to 0.60997 (MMG3D 0.52073),
mean remains above the reference (0.85607 vs 0.83983), and midpoint band
share is 0.89728 vs 0.83795. There are no residuals or inversions.
The coarsening crop retains its measured science, bit for bit. Neither
the isotropic clock nor any larger refinement row is qualified here.

Bounding the post-completion topology phase to three whole-cavity sweeps
on the same scalar refinement route reduces the paired construction
sample to **1.820 s vs 0.292 s**. Worst shape is unchanged at 0.60997;
mean shape is 0.85598, midpoint band share is 0.89701, and the sample has
zero inversions and out-of-band edges. Incomplete-band repair still uses
the exhaustive topology search. This does not close the isotropic clock.

On the original 10,452-cell isotropic input, deferring insertions that
introduce short spokes until the other edits stall produces 251,195 cells
instead of 308,339. The registered row completes with zero inversions or
out-of-band edges. Minimum shape is 0.55531 (MMG3D 0.44976), mean is
0.87119 (0.84458), and midpoint-band share is 0.85053 (0.77444). One
construction sample took 33.821 s versus the 2.020 s MMG3D reference;
the quality gates pass and the timing gate fails. This placement rule is
for scalar metrics, not the tensor route.

Pruning forbidden ring diagonals before searching their subpolygons reduced
the same construction to 28.313 s without changing the measured output.
Seeding post-band polishing from weaker cells, and using packed integer joins
for native entity lineage, reduced it further to **18.997 s**. The integrated
row has 251,198 target cells, minimum shape 0.596408, mean 0.863362, midpoint
share 0.844899, and zero inversions or strict-band misses. Mean and share
decreased relative to full polishing, but both still exceed MMG3D's. The
2.020 s construction gate remains open; finalization took another 24.124 s.

The variational box route at `7abd384155` constructs 261,235 tetrahedra in
3.363, 3.462 and **3.346 s** on the same source and metric. It still misses
MMG3D's 2.020 s. All edges meet the transported band, with no inverted
cells; worst shape is 0.604014, mean 0.863423 and midpoint share 0.859990.
The localized walk reproduced all 924,981 positive-overlap ancestry entries
from the BVH search, including distinct rounded and exact parents on a
separate one-ULP crossing fixture. Wider dyadic coordinates now use the
native validity proof with the same directed determinant bounds and map
digests. Separate certification and transfer took 10.864 s.

The scalar constructor keeps only still-refining tetrahedra in its midpoint
search. It proposes site moves from the linearized edge-length and shape
constraints, keeps a move only if the nonlinear objective improves, and
repairs the remaining vertex stars. A failed local repair can still trigger
a full solve. Neither the model nor its stopping rule certifies the mesh:
the commit checks the transported lengths, wall coverage, positive
orientations, and exact material ancestry on the proposed connectivity.

For a completed single-region box with no protected edges or segments, the
transaction returns measurements from the committed cells. Band lengths use
stable endpoint ID order; cell quality uses the cell's corner order, which
can differ by a rounding bit for a graded metric. Region volume is summed
in stable cell ID order. Other edits still measure the published mesh.
This avoids a second edge census without changing public evidence identity.

At `199cb7ad95` all three registered 10,452-source-cell comparisons completed
with no inverted cells, out-of-band edges, or representability failures.
The table uses the minimum of three matched single-thread constructions;
MMG3D did not perform Phydrax certification or transfer.

| Field | Cells | Construction / MMG3D (s) | Worst shape / MMG3D | Mean shape / MMG3D | Midpoint band share / MMG3D | Certification and transfer (s) |
|---|---:|---:|---:|---:|---:|---:|
| Scalar | 216,819 | 1.983116 / 2.020 | .603730 / .449760 | .867248 / .844584 | .824690 / .774439 | 9.286741 |
| Tensor | 17,280 | .698390 / 1.204 | .438163 / .336147 | .858165 / .847227 | .980082 / .934636 | 1.695569 |
| Coarsening | 1,842 | 1.113186 / .337 rerun | .609989 / .474811 | .831027 / .82651 | .870894 / .79391 | 1.042495 |

The scalar samples were 2.074463, 1.983116 and 1.996572 s; its timing win is
small. Coarsening passes the approved 1.3195 s ceiling, not the raw MMG3D
clock. On the 388-cell scalar crop the constructor produced 6,281 cells in
.073428 s versus .282 s, worst shape .603876 versus .520730, mean .857148
versus .839835, and midpoint share .849226 versus .837952. Result, stencil,
lineage, and evidence IDs were unchanged across worker requests 1/2/4/8.
On these four cases the exhaustive and fused consumers on the same candidate
mesh agreed on every public evidence field and all nine result identities.
These measurements do not establish performance at larger sizes.

The same 388-cell tensor crop produces 200 cells with minimum shape
.783146 versus .297498, mean .868128 versus .677317, and midpoint share
.994819 versus .422650. Construction took .554119, .022225 and .021946 s;
the first call was slower than MMG3D's .347 s minimum. On the coarsening
crop, construction took .076769 s at best versus .206 s, worst shape
.602177 versus .581358, with all science gates met. MMG3D 5.8.0 ran three
times per field on the same crop inputs; each final output mesh matched
the independently scored reference file byte-for-byte.

On the same verified 10,452-cell source and metric bytes, three alternating
single-thread pairs with **serial Omega_h** timed gradation and construction.
Both outputs were scored with the same independent midpoint and shape
measurements. No Phydrax certification or transfer time is in either
construction column. The two tools need not produce the same number of cells.

| Field | Construction, Phydrax / Omega_h (s) | Cells, Phydrax / Omega_h | Worst shape | Mean shape | Midpoint band share |
|---|---:|---:|---:|---:|---:|
| Scalar | 1.952453 / 10.412130 | 216,819 / 274,146 | .603730 / .300013 | .867248 / .781607 | .824690 / .759292 |
| Tensor | .678714 / .866610 | 17,280 / 21,754 | .438163 / .185327 | .858165 / .791272 | .980082 / .902577 |
| Coarsening | 1.091869 / .112671 | 1,842 / 2,203 | .609989 / .300203 | .831027 / .730574 | .870894 / .800198 |

All three Phydrax outputs completed with zero strict-band misses; neither
tool produced an inverted cell. The coarsening result is higher quality
but 9.69 times slower to construct than Omega_h's. Separate Phydrax
certification/transfer on the final pair took 9.243, 1.639 and 1.033 s.
This is a one-size, single-thread comparison, not the final quiet matrix.

### Implicit carrier remesh and proof

On the general implicit route, a sharp turn in the scaffold is only a
crease candidate. The remesher keeps it when the projected source normals
also turn sharply or cannot be resolved. This lets the remesher repair
false sharp edges from poorly shaped scaffold triangles before the
FIXED-fill radius-edge gate. The explicit mesh route has no independent
normal field and keeps its original crease classification.
The two disjoint sphere regression passes the existing radius-edge,
embedding, coverage, and continuous distance requirements; the CSG lens
retains its true crease and passes two-way certification. This is a
construction change, not a proof relaxation.

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

The initial results beat the curving clocks but missed some reference shape
thresholds. The subsequent pinned 12-row campaign passes every registered
ceiling and exact local determinant check. Interleaved live comparisons
against Gmsh and Netgen are faster on all 12 but pass the live quality
thresholds on 10. The misses are fine sphere P2 determinant ratio
(0.919151 vs 0.919453) and coarse quadric P2 minimum scaled Jacobian
(0.277139 vs 0.277259). Both pass their registered thresholds. These live
quality misses are not proof failures.

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

### Matched largest CAD qualification

On the original quadric at h = 0.02286, Phydrax took 1.232 s at one
worker and 1.126 s at eight; Gmsh 4.15.2 HXT took 1.863 s and 1.133 s
respectively. These are the minima of three interleaved pairs per worker
count on macOS. The native process was reused; HXT ran in fresh processes
with initialization and geometry preparation outside the mesh clock.
The eight-worker comparison uses the faster of the one- and eight-worker
HXT clocks. It qualifies construction on this row, not proof latency.

The native mesh and non-time evidence were byte-identical at 1, 2, 3, 4,
and 8 workers: 493,903 tetrahedra, zero unmet constraints, and no
independent quality misses. The maximum tetrahedron radius-edge ratio was
1.999924, the maximum tetrahedron circumradius divided by h was 0.999987,
and the maximum boundary circumradius divided by its bound was 0.972409.

### Matched medium 3D point-set qualification

The medium parallel path gives each column exclusive access to its wholly
owned cavity cells and to nonconflicting finite neighbors across owned faces.
Each such neighbor shares three vertices with the cavity, so it cannot be
adjacent to wholly owned cells in two regions. Conflicting mixed cells and
ghost neighbors defer. Walk addresses come from the ordered region, not the
worker that runs it. The existing serial and large-grid paths are unchanged.

On the original 10,000-point banks, all eight ordinary/regular comparisons
at 1, 2, 4, and 8 workers passed. Each reports the minimum of three interleaved
calls, with the reference ceiling the faster of serial or matching-worker
CGAL. The two formerly failing four-worker rows were:

| Triangulation | Phydrax (ms) | Live CGAL ceiling (ms) | Previous CGAL ceiling (ms) |
|:---|---:|---:|---:|
| Ordinary | 17.815 | 27.323 | 20.820 |
| Regular | 18.644 | 28.552 | 20.173 |

The original cells and visible-vertex maps were byte-identical at 1, 2, 3,
4, and 8 workers. The campaign started at one-minute load 3.21; load was
4.24 before the regular arm, so this is not the final quiet integrated
matrix. Both rows also beat their earlier, lower ceilings at matched inputs.

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
  CGAL isotropic-remeshing drivers are also serial. The pinned `8055b46755`
  comparison passes seven generated and six imported cases at a configured
  single thread. This is construction and sampled quality evidence, not
  actual-team telemetry or a replacement for the separate fidelity proofs.
  On the integrated locator revision, all 13 cases returned byte-identical
  native result banks and zero unmet constraints at worker requests of
  1, 2, 4, and 8. This is worker invariance, not a parallel speedup claim.
- F13 overset remains sequential. In the earlier matched campaign on the
  same ordered banks it beats TIOGA at 1, 2, 4, and 8 MPI ranks for all
  three hex and all three tet sizes (three interleaved calls per row).
  At the largest sizes and eight ranks the prepared candidate takes
  0.274 s vs 0.649 s for hexes and 0.182 s vs 0.517 s for tets. Native
  connectivity IDs are invariant and certified donors pass the independent
  check. These are assembly clocks, not proof. The original middle tet
  bank is unavailable for a new run; do not substitute regenerated inputs.
- F14 conforming Voronoi construction remains sequential. The pre-centering
  matrix passed timing at all 32 size/worker combinations, 31 of 32 using
  archived quality ceilings and 29 of 32 against the fastest retained
  VoroCrust meshes. The large sphere uses the archived 5,000-cell reference.
  These are candidate construction clocks, not equal-cell throughput or proof.
  The centered lattice reduces the small cube's maximum from 27 to 25 faces
  and meets the former 25/26 reference limits. The centered candidate passed
  both live comparisons: 0.807 s versus 3.205 s (one worker) and 0.818 s
  versus 4.515 s (four), also below the faster serial ceiling. Its mesh and
  evidence IDs matched across requests of 1, 2, 3, 4, and 8.
  Splitting repair scans from publication retained all eight exact centered
  output banks, including the independently certified small cube. On the
  combined image the small Steinmetz at four workers took 5.327 s versus
  10.349 s, below the 6.574 s serial clock, with 25 versus 32 maximum faces.
  The fastest serial VoroCrust output had two independent convexity findings,
  so the strict fastest-output serial comparison remains unqualified. The
  faster of the valid retained serial outputs took 8.729 s versus 5.350 s
  for Phydrax, with all measured quality criteria met. No timing or method
  was substituted for the strict comparison.
- Periodic Delaunay and fixed-boundary recovery remain sequential in the
  matched comparisons against the sequential CGAL providers. All Phydrax
  output banks are invariant under worker requests. This does not qualify
  fixed-boundary volume refinement against threaded HXT.

### Aligned anisotropic refinement

On the registered 10,452-cell affine box, whole-source retessellation produces
17,280 tetrahedra with no inversions or out-of-band edges. Independent worst
shape is 0.438163 versus MMG3D's 0.336147; mean shape 0.858165 versus
0.847227; midpoint share 0.980082 versus 0.934636. Single-thread construction,
including physical gradation and source ancestry, took 0.944706, 0.927406 and
0.824845 s on the integrated candidate, versus the matched MMG3D minimum of
1.204 s. Separate certification and transfer took 4.971566 s. This qualifies one
refinement field and one source size, not the entire F6 ladder.
The same three single-thread comparisons at `7abd384155` took 0.796170,
0.710478 and **0.702215 s**, with unchanged quality and 1.698140 s separate
certification and transfer. Graded coarsening retained all science gates at
1.095989 s versus its approved 1.3195 s ceiling.

### Sequential adaptation worker invariance

F6 native metric edits remain sequential: committing a cavity changes the
stars and source stencils used to admit the next edit. Exact verification can
use its own worker cap. On the original coarsen input at `b86af13ec9`, requests
of 1, 2, 3, 4, and 8 workers produce identical mesh, geometry, metric, stencil,
transition, lineage, and evidence IDs. Each run completes the strict band with
no residuals and mean shape 0.8295888232. This qualifies native output
invariance for that input, not parallel adaptation or the separate MMG3D
construction ceiling.
The bounded-cage 388-cell anisotropic crop also completes with zero
residuals and unchanged mesh, transition, lineage, stencil, metric,
and evidence IDs at worker requests of 1, 2, 4, and 8. These requests
check invariance; they do not attest to a parallel edit team.
The edited scalar variational crop produced 7,596 tetrahedra and identical
mesh, result, geometry, validity, transition, lineage, stencil, metric and
evidence identities at worker requests of 1, 2, 4 and 8. These counters do
not claim that the adaptation transaction itself ran on multiple workers.

### Matched-worker evidence and remaining limits

These comparisons retain the original inputs and scopes. They are not a
single quiet run of the final integrated tree.

| Route | Matched comparison | Limit |
|---|---|---|
| F2 linear CAD | Original large quadric on `7e00f7048c`: native 1.125606 s versus HXT 1.132836 s at eight workers, all quality gates pass | Quiet construction comparison (board `93155cbf`), not proof latency. An earlier loaded campaign passed 15 of 16 clocks, with identical native mesh/evidence banks per size and no registered quality misses |
| F10 affine point location | `2c2f3e5890`: all eight comparisons against libigl 2.6.3 pass; eight-worker 100k and 1M clocks are 18.229/19.586 ms and 128.491/179.771 ms | Query only; both tools have zero wrong, missed, or extra locations |
| F6b no-op adaptation | Original Box-12/25/55, 12/12 against Omega_h 9.34.13 with attested 1/2/4/8 OpenMP teams; eight-worker native/reference minima 2.538/2.771, 14.418/17.315, and 119.839/144.523 ms | Unchanged source and metric IDs; edited F6 is separate |
| F8 unlayered hex-dominant | Original sphere and quadric, 4/4 each against actual `snappyHexMesh` ranks with invariant native outputs | cfMesh did not run matched teams |
| F8 two layers | Original sphere against eight actual `snappyHexMesh` ranks: native skewness 0.495351 versus 0.497436; both native layers meet their projected thickness floors | Native 5.115514 s beats the live 5.193225 s but misses the historical 5.066336 s limit. One-minute load 6.892→6.282 excludes quiet acceptance. The original layered quadric at eight ranks placed only 24,008 of 24,800 layer cells and had eight cells with a nonpositive independent decomposition despite passing `checkMesh` |
| F9 prism layers | Original sphere h = 0.1 passes live Gmsh at 1/2/4/8 requests with native output invariance | cfMesh output on the other layer rows fails its own `checkMesh` gate |

The F8 descent now omits derivative work that cannot reach a movable
interior vertex. It keeps the existing validity screens and reduction
order. On the original two-layer sphere (98,038 cells), eight-worker
native smoothing and layer time fell from 2.003–2.017 s to 1.936–1.964 s.
At doubled nominal sizes (20,943 cells), it took 0.412 s versus 0.429 s.
Both comparisons retained byte-identical numerical output; the original
native bank also matches at one and eight requested workers. The public
candidate identity remains build-bound. These are native stage clocks,
not acceptance against the provider construction clock.

The settlement step now screens immutable pass geometry in parallel and
replays rollbacks in their original order. A cell touched by an earlier
rollback is read again on the changed geometry. On the same layered
sphere the native smoothing/laying stage took 1.551 s versus 1.941 s
at eight workers; the smaller request took 0.335 s versus 0.412 s.
The numerical banks match the preceding implementation exactly at both
sizes and match between one and eight workers on the original sphere.

Raw receipts are in the coordination board under
`artifacts/MAC/native-redesign/`, including the original no-op adaptation
and unlayered sphere/quadric JSON. Provider ineligibility, timing misses,
and unresolved native proofs are separate outcomes.

### Unified 10-row graded metric qualification

On the verified 3D unit-cube banks (`10,452 / 98,932 / 965,817` tetrahedra,
`iso` / `aniso` / `coarsen`) and the 2D unit-square circle bank
(`99,954` triangles, `square1e5.iso.mbin`, target
`h = min(0.0005 + 0.3 |r - 0.3|, 0.02)` about `(0.5, 0.5)`), single-thread
Phydrax (`OMP_NUM_THREADS=1`, minimum of three runs via
`python -m benchmarks.native_meshing --sizes 10000 100000 1000000 --rows metric-3d-iso metric-3d-aniso metric-3d-coarsen metric-2d-iso --repeats 3 --metric-inputs DIR`)
beats both quiet single-threaded MMG 5.8.0 (`mmg3d_O3` / `mmg2d_O3`) and
serial Omega_h 9.34.13 on construction time and all three independent quality
metrics across all ten rows with `allowed_ratio = 1.0`, `status == complete`,
`0` out-of-band edges, `0` unrepresentable edges, and `0` inverted cells.
Construction time covers gradation, preparation, and mesh edit construction;
certification and field transfer run on a separate clock.

| Row | Size | Phydrax (s) | Ceiling (s, tool) | Ratio | Quiet MMG (s) | Quiet Omega_h (s) | Worst shape (Phx / MMG / Osh) | Mean shape (Phx / MMG / Osh) | Band share (Phx / MMG / Osh) | Finalization (s) |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| `metric-3d-iso` | 10,000 | 1.788 | 2.020 (mmg3d) | 0.885 | 2.273 | 11.195 | .6050 / .4498 / .3000 | .8708 / .8446 / .7816 | .8253 / .7744 / .7593 | 8.396 |
| `metric-3d-aniso` | 10,000 | 0.281 | 0.935 (omega_h) | 0.300 | 1.244 | 0.935 | .4395 / .3361 / .1853 | .8582 / .8472 / .7913 | .9800 / .9346 / .9026 | 0.953 |
| `metric-3d-coarsen` | 10,000 | 0.0685 | 0.1177 (omega_h) | 0.582 | 0.356 | 0.1177 | .6050 / .4748 / .3002 | .8460 / .8265 / .7306 | .9497 / .7939 / .8002 | 0.123 |
| `metric-3d-iso` | 100,000 | 2.978 | 3.317 (mmg3d) | 0.898 | 3.553 | 15.716 | .6043 / .4478 / .3000 | .8696 / .8453 / .7806 | .8951 / .8704 / .8342 | 10.626 |
| `metric-3d-aniso` | 100,000 | 0.643 | 3.455 (mmg3d) | 0.186 | 3.690 | 5.237 | .7709 / .3458 / .2832 | .8691 / .8383 / .7517 | 1.0000 / .9063 / .8307 | 1.116 |
| `metric-3d-coarsen` | 100,000 | 0.476 | 1.161 (mmg3d) | 0.410 | 1.188 | 1.271 | .6044 / .4235 / .3010 | .8516 / .8386 / .7329 | .9641 / .8622 / .8152 | 0.888 |
| `metric-3d-iso` | 1,000,000 | 7.956 | 13.826 (mmg3d) | 0.575 | 14.900 | 41.536 | .6050 / .3777 / .3000 | .8672 / .8434 / .7775 | .9609 / .9175 / .8679 | 17.582 |
| `metric-3d-aniso` | 1,000,000 | 3.538 | 20.244 (mmg3d) | 0.175 | 22.819 | 54.860 | .8160 / .4364 / .2776 | .8702 / .8394 / .7464 | 1.0000 / .9116 / .8252 | 5.699 |
| `metric-3d-coarsen` | 1,000,000 | 7.402 | 14.124 (mmg3d) | 0.524 | 15.439 | 19.543 | .6047 / .3546 / .3007 | .8619 / .8416 / .7320 | .9589 / .8926 / .8108 | 9.855 |
| `metric-2d-iso` | 100,000 | 0.227 | 0.345 (mmg2d) | 0.658 | 0.345 | 0.829 | .7400 / .6119 / .4000 | .9405 / .9357 / .8978 | .8912 / .8026 / .8179 | 8.375 |

Identity notes: 3D output-identical box-route optimizations (lazy cell grid,
lazy per-cell scalar field, compact source cells, target-planes-first ancestry
pruning, split binding, fast unchanged export) preserve all nine outcome IDs
bit-exact; extending 3D scalar retessellation to coarsening, SoA local-first
L-BFGS repair with post-stage-0 re-localization, and adding the native 2D
scalar rectangle retessellation route can change adapted mesh and evidence IDs
on newly retessellated scalar boxes while preserving exact local-only and no-op
evidence identities (`retriangulations == 0` is omitted from `evidence_id`).
End-to-end runtime including certification and transfer remains separate and
slower than uncertified construction.

### Final integrated qualification and publication (`4ffad9ea2`)

Integrated code head: `4ffad9ea2af9a71051718ba83b3ddc8af4bbe65b`
(`origin/wip/native-meshing-mac/kernels`, meshcore build identity
`353adea5f0936a5e1fcd7c54f09aff305cc80e0e045bb62501bc7eb497ee6dbf`).

#### 1. Compiler, sanitizer, and static/Python gates on `4ffad9ea2`

- **Apple Clang 17 (macOS arm64, Release):** `74/74` CTest passed (`0` warnings,
  `0` errors, `0` notes; dylib SHA-256
  `54cea5d0556e4fcf3127f40a97b62b363a021be185e9b73f3882f6498bd41209`).
- **Apple Clang ThreadSanitizer (`RelWithDebInfo`, `-DPHX_MC_THREAD_SANITIZE=ON`):**
  `9/9` passed (`test_parallel`, `test_cad_mesh`, `test_cad_volume`,
  `test_cdt2d`, `test_delaunay2d`, `test_mesh_source`, `test_metric_gradation`,
  `test_metric_adapt3d`, `test_metric_scalar2d`; `0` data races).
- **GCC 13.3 (Linux aarch64, Release, board `4a063585`):** `74/74` CTest passed
  (`0` warnings, `0` errors, `0` GCC 10.1 `psabi` notes across all 262 targets;
  SHA-256 `5cd4c0d801d6f97fbb24dac38d4360fda66c251ff2ed498e1ce431ec97c37118`).
- **MSVC 19.51 (Windows x86-64, Release, board `201783a4`):** `74/74` CTest
  passed (`0` warnings, `0` errors, `0` notes — both C4805 warnings resolved;
  DLL SHA-256 `4bb1596cd05f5544845a8e2f402cf437ab618910be2d8015b449c822035811d8`).
- **Bitwise diagnostic parity (`5d1765951` vs `4ffad9ea2`):** exact match on CAD
  quadric (`22930`), planar Ruppert (`10000/square`), and all four small 2D/3D
  graded metric rows.
- **Python & static gates:** `503/503` targeted meshing unit tests across the 16
  directly affected modules passed against the pinned Release library (user
  directed targeted affected tests instead of the full global suite);
  `ruff check`, `ruff format --check` on touched files,
  `tools/check_typing.py check` (`0` diagnostics),
  `tools.check_installed_typing` (Python 3.13 wheel; package declares
  `Requires-Python >=3.13,<3.14`), `tools/check_capability_consistency.py`,
  `tools.check_public_api_manifest`, `tools/audit_selectors.py` (`0` findings),
  and `tools/audit_contract_candidates.py --signatures` all passed.

#### 2. Reviewed certificate scope

- **Reviewed native certifiers (separate from timed construction):**
  `mesh_certify`, `zero_set_certify` (two-way tube plus possible-zero cover),
  `verify_volume` (`phx_mc_verify_tetrahedra` / `phx_mc_verify_volume` /
  `phx_mc_tetrahedra_validity`), `verify_surface_embedding`,
  `verify_chart_partition`, the quadratic validity screen, and the
  mapped-locator root discharge.
- **Separate / untimed scope:** construction benchmarks time construction and
  sampled quality only (`prove=False` or construction phase); exact
  certification, `.prove()`, and field transfer run on separate clocks and are
  slower than uncertified external references. `verify_cad_fidelity` (V3)
  remains under independent re-read; CAD rows are reported as construction plus
  sampled quality.

#### 3. Final quiet interleaved macOS acceptance summary (`4ffad9ea2`)

All raw manifests, outputs, and verdicts are archived on the coordination board
under `artifacts/MAC/native-redesign/acceptance/final-4ffad9ea2-*`.

| Family / Partition | Owner | Result on `4ffad9ea2` | Timing ratio (ours / ref) | Separate quality / proof / limitation notes |
|---|---|---|---|---|
| S0-B Point sets (`DT2/DT3/Regular/Periodic` 1k–1M vs CGAL) | MAC | 15/16 batch; 16/16 with fresh-process `DT2/1k` | `0.430–0.924×` (fresh `DT2/1k` `0.913×`; batched `DT2/1k` `1.145×`) | Exact input SHA & element-count parity across all 16 rows |
| S2-C Planar CDT / q20 (`12` rows) & q30 (`8` rows) vs Triangle `20250106` | LIN (on Mac) | `12/12` + `8/8` passed | CDT/q20 `0.157–0.924×`; q30 `0.28–0.92×` | `0` unmet/nonpositive/above-max-area; formerly missed `square/1k` (`0.818×`) & `ring-slit/10k` (`0.549×`) pass |
| S1 Implicit F4 (`4` rows vs CGAL) | MAC | `4/4` `.prove()` certified; `3/4` clock | `0.135–0.769×` on 3 rows; coarse `sphere/0.02` (487 tets) `1.142×` | Timing miss on coarse `sphere/0.02` (`1.817` vs `1.592 ms`) recorded separately from exact `.prove()` pass |
| S3 Recovery & Volume (`12` rows vs CGAL CDT3 / HXT) | MAC | `12/12` passed | Recovery `0.191–0.371×`; volume `0.291–0.686×` | `0` unmet constraints / quality misses |
| S3 Live Improvement (`3` rows vs `mmg3d_O3 -optim -nosurf`) | MAC | `3/3` passed | `0.028–0.056×` (`17k`, `143k`, `1.16M`) | `6/6` input files byte-identical SHA-256 to `8055` manifest |
| F2 CAD Quadric 1T (`4` rows vs Gmsh HXT) | LIN (on Mac) | `4/4` passed | `0.615–0.689×` (`2286` at `0.664×`) | `0` unmet / quality misses |
| S5 NIST STEP Washer (`a5abfea4...`) | LIN (on Mac) | Passed vs `hxt-fine` | `1.715 s` (`51,758` tets, Hausdorff `0.0309 mm`) vs `hxt-fine` `1.747 s` (`354,556` tets, `0.0247 mm`) | `hxt-same-h` (`0.0332 s`, `1,342` tets) fails fidelity (`1.291 mm`) |
| F5 P2/P3 Curved (`12` rows vs Netgen `6.2.2608` / Gmsh `4.15.2`) | LIN (on Mac) | `12/12` clock + registered quality + exact local det; `10/12` all live thresholds | Beats both references on all `12/12` rows | Reproduces the `2` documented live micro-misses (coarse quadric P2 `scaled_jacobian`, fine sphere P2 `determinant_ratio`), not proof failures |
| F7 Surface Remesh (`13` rows vs MMGS / CGAL) | WIN (on Mac) | `13/13` passed | `0.137–0.755×` (`certified = 1.0`, `0` unmet) | Justified sequential; all `13/13` byte-identical across `1/2/4/8` workers |
| S4 Graded F6 (`10` rows vs MMG 5.8.0 / Omega_h 9.34.13) | WIN (on Mac) | `10/10` passed (`allowed_ratio = 1.0`) | `0.172–0.879×` (`ALL_REF_VERIFIED: True`) | `0` unmet / out-of-band / unrepresentable / inverted; sequential box routes invariant across `1/2/4/8` workers |
| S4 F6b No-op Box-12/25/55 vs serial Omega_h | WIN (on Mac) | `3/3` passed | `0.279–0.356×` (`4.06/11.41`, `29.02/100.99`, `292.1/1047.1 ms`) | Unchanged source and evidence IDs |
| F10 Prepared Queries (`32` rows: 4 queries × `1e5/1e6` × `1/2/4/8`W vs libigl `2.6.3` / `igl_signed`) | MAC | `32/32` passed (`load_start = 2.11`) | `0.363–0.926×` (8W `1M`: closest `0.418×`, signed `0.363×`, winding `0.540×`, locate `0.680×`) | Bit-identical across `1/2/4/8` workers; `0` quality errors; signed uses prepared C++ `igl_signed` (excludes index build) |
| F13 Overset (`3` hex + `2` authenticated tet rows vs TIOGA) | MAC | `5/5` authenticated rows passed | Hex `0.026–0.188×`; tet `10k` `0.124×`, `1M` `0.195×` | Middle tet bank `f13-tet1e5s.bin` (`4d3ab98e...`) unavailable; unmeasured, not substituted |
| F9 Prism Layers (`sphere/100` vs Gmsh) | MAC | `1/1` eligible row passed | `0.2515 s` vs `0.2836 s` (`0.887×`, `15,900` certified prisms) | `cfMesh` references on `sphere/50`, `quadric/100`, `quadric/50` fail own `checkMesh` (ineligible) |
| F14 Conforming Voronoi (`8` rows vs VoroCrust) | MAC | `8/8` passed | `0.246–0.669×`, `100%` closed convex cells | Fastest serial VoroCrust output has 2 independent convexity findings (strict serial comparison unqualified) |
| F8 Hex-dominant 1T (`4` rows vs cfMesh `cartesianMesh`) | MAC | `4/4` passed | `0.592–0.929×` (`sphere/quadric` L0 & L2) | `0` unmet / bad cells |

#### 4. Final S7 thread matrix and sequential justifications (`4ffad9ea2`)

- **F10 Queries (`32/32` quiet pass):** every query family (`closest`, `signed`,
  `winding`, `locate`) beats libigl `2.6.3` / prepared `igl_signed` at `1/2/4/8`
  workers across `1e5` and `1e6` with bit-identical outputs (`load_start = 2.11`).
- **F2 CAD Quadric (`16` rows, quiet `load 3.06 -> 2.93`):** `16/16` native
  output digests bit-identical across `1/2/4/8` workers (`status == OK`,
  `unmet == 0`, zero registered quality misses). `15/16` clocks beat Gmsh HXT
  (`0.686–0.979×`); `h = 0.02286` at `8` workers measured `1.207949 s` vs
  `1.202034 s` (`1.0049×`, `+5.9 ms`), recorded alongside the retained quiet
  targeted win on `7e00f7048c` (`1.125606 s` vs `1.132836 s`, board `93155cbf`).
- **Graded F6 3D/2D box routes & gradation (justified sequential):**
  `metric_layer_transaction.cpp`, `metric_scalar2d.cpp`, and
  `metric_gradation.cpp` execute sequentially because each retessellation,
  relaxation, and label-setting gradation wave depends on prior topology/metric
  state (`rg parallel_for` confirms no parallel teams in those files). On
  `4ffad9ea2`, all four small graded rows (`10,452`-cell 3D iso/aniso/coarsen
  and `99,954`-tri 2D circle) produced bit-identical 9-tuple outcome IDs across
  worker requests `1/2/4/8` (`retriangulations == 1`, `out_of_range_edges == 0`).
- **F7 Remesh (justified sequential):** all `13/13` cases (7 generated, 6
  imported) produced byte-identical outputs with `unmet == 0` at `1/2/4/8`
  workers on `4ffad9ea2`.
- **F8 Two-layer sphere at 8 ranks (`f8-quiet-eight.py`):** started at load
  `2.254`; Pair 1 beat both ceilings (`4.994578 s` native vs `5.810123 s` live
  `snappyHexMesh` and `5.066336 s` historical ceiling, exact
  `mesh_id = 48f48fed...`), then aborted before Pair 2 because the 8-rank
  MPI + native burst itself lifted 1-minute load to `3.766 > 3.4`. Remains a
  partial single-pair beat rather than a 3-pair quiet-admitted series; layered
  quadric at 8 ranks remains ineligible (`24,008 / 24,800` layer cells and 8
  nonpositive cells in `snappyHexMesh`).
- **Retained unchanged S7 routes:** F6b no-op `12/12` vs OpenMP Omega_h,
  F8 unlayered `4/4` vs `snappyHexMesh`, F9 `sphere/100` vs Gmsh, F13 overset
  vs TIOGA (sequential), and F14 conforming Voronoi (sequential) remain valid
  from their archived receipts (no kernel changes in those paths since `199cb7ad95`).

#### 5. Complete identity-change inventory

1. **Graded 3D & 2D box retessellation routes:** newly retessellated scalar 3D
   coarsening/refinement boxes, aligned anisotropic boxes, and 2D scalar
   rectangles produce new adapted mesh and evidence IDs; local-only and no-op
   routes preserve exact historical IDs (`retriangulations == 0` omitted from
   `TetraMetricEvidence.evidence_id` and `LocalMetricEvidence.evidence_id`).
2. **Planar CDT interior lattice seeding:** seeded interior sites can change
   mesh identity while retaining authored vertices, hole domains, and ancestry.
3. **CAD solid hard whole-wall protection, upper dihedral cap, and P2/P3 shape
   targets:** newly admitted / active-cap / shaped requests can change plan and
   mesh identities; defaults leave optional caps disabled.
4. **Tetrahedral metric shape reconnection, relocation polishing, and shallow
   wall-ring proposals:** newly accepted edits can change adapted cell and
   evidence IDs without relaxing the strict `[1/sqrt(2), sqrt(2)]` band.
5. **Octree hex-dominant unlayered free-wall repair and layered rigid column
   tilt (`layer_minimum_thickness_fraction`):** can change hex-dominant meshes
   and evidence IDs.
6. **Conforming Voronoi centered interior lattice:** changes newly constructed
   Voronoi mesh and evidence IDs while preserving spacing, clearance, and exact
   certification.
7. **Native model archives & layer-core carrier inventories:** compact JSON
   manifests, batched column storage, and minimal source-edge carrier
   inventories intentionally change archive/carrier identities as documented in
   `CHANGELOG.md` and `docs/guides_meshing.md`.

#### 6. Remaining limitations

- **Coarse implicit `sphere/0.02` (`487` tets):** `1.817 ms` vs `1.592 ms` live
  CGAL (`1.142×`); exact `.prove()` passes.
- **Batched `delaunay-2d/1000`:** `1.145×` inside the multi-size harness
  (`0.913×` in isolated fresh-process comparison).
- **F2 CAD `h = 0.02286` at `8` workers:** `1.207949 s` vs `1.202034 s`
  (`1.0049×`) in the 16-row quiet sweep (won `1.125606` vs `1.132836 s` in the
  targeted quiet check on `7e00f7048c`).
- **F5 live P2 micro-misses:** `2/12` live thresholds (coarse quadric P2
  `scaled_jacobian`, fine sphere P2 `determinant_ratio`); all `12/12` pass
  registered quality, exact local determinant checks, and curving clocks.
- **F8 two-layer sphere 8-rank:** Pair 1 beat both ceilings (`4.9946 s` vs
  `5.8101 s` live / `5.0663 s` historical), stopped before Pair 2 by 8-rank
  self-load (`3.766 > 3.4`); layered quadric `snappyHexMesh` reference
  ineligible.
- **F9 cfMesh rows (`sphere/50`, `quadric/100`, `quadric/50`):** `cfMesh`
  outputs fail own `checkMesh` (provider ineligibility).
- **F13 middle tet bank (`f13-tet1e5s.bin`, SHA `4d3ab98e...`):** original
  bank unavailable across all three machines; reported unmeasured without
  substitution.
- **F14 strict fastest serial VoroCrust:** fastest serial output fails
  independent convexity (2 findings); strict fastest-output comparison stays
  unqualified.
- **Certification / transfer latency:** exact `.prove()`, certification, and
  field transfer run separately from construction and remain slower than
  uncertified external construction tools.

### Follow-up optimization, curving closure, and surface repair landing (Tier 1 + Tier 2A + `surface-repair`)

Unified Release build on code commit `ee62dc1e5895618ee3a9a3566798d16aa8d0a313`
(AppleClang 17, macOS arm64, `-Werror`, zero warnings):
meshcore identity `phydrax-meshcore 0.2.2 44aaf044154ecfde22644159ae7bb411218a4dbb4be816dacd4e1f6d14458895`,
dylib SHA-256 `72a05c6035ae9bf14b5f565426b6d8a938ce75595cf73ceb23836f6ede772442`.
Includes native certified surface mesh repair (`a3d48707a`, `20da8cfde`) plus
Tier 1 timing/curving fixes and Tier 2A transfer/certification speedups on
`wip/native-meshing-mac/diagnostics-clean`, fast-forwarded to
`origin/wip/native-meshing-mac/kernels` (MAC-only per user directive).

#### 1. Compiler, sanitizer, portability, and Python gates

- **Release CTest (`-Werror`):** `76/76` passed (`0` warnings, including
  `test_surface_repair`, new `test_tet_entities`, and extended 2D
  `test_cell_map_validity`).
- **Portability pre-flight (GCC/MSVC):** replaced GNU-only inline/noinline
  attributes and scalar-branch `bool |=` in `delaunay2d.cpp` with portable
  `PHX_MC_NOINLINE` / `PHX_MC_ALWAYS_INLINE` macros, verified scalar non-NEON
  `delaunay2d.cpp` compiles warning-free (`-DPHX_MC_DISABLE_NEON`) and passes
  `test_delaunay2d`, fixed 2 `int64_t -> int` narrowing sites in
  `arrangement.cpp:973-974`, and verified `0` `-Wshorten-64-to-32` /
  `-Wfloat-conversion` warnings across all 9 modified `.cpp` files and 3
  new/modified native tests.
- **ASan + UBSan:** `10/10` affected native tests passed (`0` findings).
- **ThreadSanitizer (`RelWithDebInfo`):** `7/7` modified parallel/kernel tests
  passed (`test_hex_dominant`, `test_plc3d`, `test_cad_mesh`, `test_cad_volume`,
  `test_cell_map_validity`, `test_delaunay2d`, `test_restricted_delaunay`; `0`
  data races).
- **Python & audits:** `428/428` targeted discretization/geometry/meshing tests
  plus `124/124` (`test_surface_repair.py`, `test_cell_geometry_validity.py`,
  `test_cell_mesh_nd.py`) passed against the final Release library; `ruff check`,
  `ruff format --check`, `tools/check_typing.py check` (`0` diagnostics),
  `python -m tools.check_installed_typing` (`python 3.13: 4 fixture file(s) passed`),
  `tools/audit_selectors.py` (`0` findings),
  `tools/audit_contract_candidates.py --signatures` (`0` redundant guards),
  `tools/check_capability_consistency.py`, and
  `python -m tools.check_public_api_manifest` all passed.
- **Linux & Windows gates:** deferred by user directive (MAC-only landing;
  no open Linux/Windows assignment on the coordination board). Portability
  pre-flight audit (`-DPHX_MC_DISABLE_NEON`, `-Wshorten-64-to-32`,
  `-Wfloat-conversion`, portable inline/noinline macros) passed cleanly on
  macOS before landing.
- **Archived receipts:** `artifacts/MAC/native-redesign/acceptance/final-ee62dc1e5/`
  on `comm-board`.

#### 2. Closed timing and live-curving misses (on `72a05c60...`)

| Row | Prior (`4ffad9ea2`) | New (Final Release) | Live Reference | Ratio | Identity / Proof |
|---|---:|---:|---:|---:|---|
| `delaunay-2d/1000` (multi-size harness) | `0.3985 ms` (`1.145×`) | **`0.3421 ms`** (`0.166 ms` warm min) | `0.3490 ms` (CGAL) | **`0.980×`** | Bit-identical; `10k/100k/1M` at `0.739× / 0.696× / 0.680×` |
| `implicit-sphere/20` (`487` tets) | `1.817 ms` (`1.142×`) | **`1.092 ms`** (`0.840 ms` warm) | `1.617 ms` (CGAL) | **`0.675×`** | Bit-identical `mesh_id` (`61f48fc83ae9c749`), certified `execute()` passes; all 4 implicit rows pass (`0.084–0.675×`) |
| `cad-quadric/2286` (`8` workers) | `1.208 s` (`1.005×`) | **`0.9923 s`** | `1.2118 s` (Gmsh HXT 8T) | **`0.819×`** | `1/2/4/8`W bit-identical (`493,804` tets, `0` unmet); 1T `1.231 s` vs `1.959 s` (`0.628×`) |
| `hexdom-sphere-L2/100000` (`8` workers) | `4.99–5.10 s` (1 pair; load `3.77`) | **`1.7640 s`** (`[1.907, 1.764, 1.797]`) | `5.6247 s` live / `5.0663 s` hist (`snappyHexMesh` 8 ranks) | **`0.314×`** live / **`0.348×`** hist | Bit-identical `mesh_id` (`48f48fed...`) across `1/2/4/8`W; `9/9` `scoped_quality` (`science_passed: true`); 1T vs cfMesh `3.827 s` vs `8.223 s` (`0.465×`) |
| F5 curved CAD (`12` rows, P2/P3) | `10/12` live quality (`2` micro-misses) | **`12/12` pass** | Live Netgen / Gmsh thresholds | All `12/12` clocks `< ref` | `cad-quadric-p2/10610` `sj=0.277429 >= 0.277259`; `cad-sphere-p2/5940` `dr=0.919654 >= 0.919453`; `0` invalid/unresolved |

#### 3. Tier 2A transfer and certification latency improvements (identity-preserving)

- Native packed polyhedral entity extraction (`phx_mc_packed_cell_vertex_counts`,
  `phx_mc_packed_polyhedral_mesh_entities`), native 2D affine triangle map
  certification (`phx_mc_certify_cell_maps` `domain = 3`), `jax.device_put` for
  concrete NumPy leaves (removing fresh-shape XLA compilations), single-region
  edge/association fast paths, and `scipy.sparse.csgraph.connected_components`
  for patch connectivity:
  - **`hexdom-sphere-L2/100000` Python wrapper:** dropped from `~2.60 s` to
    `~1.00 s` (alongside bit-identical C++ `hex_dominant` dropping `2.62 s` to
    `0.73–0.79 s` at 8W), retaining exact `mesh_id = 48f48fed...`.
  - **`metric-2d-iso/100000` certification and transfer:** dropped from
    `8.375 s` to **`3.494 s`** (`2.40×` faster, construction `0.214 s` vs
    `mmg2d` `0.345 s`, `0.619×`, exact verified inputs).

#### 4. Identity-change additions (since `4ffad9ea2`)

- **Bit-identical (zero identity change):** `delaunay2d.cpp`,
  `restricted_delaunay.cpp`, `hex_dominant.cpp`, `tet_entities.cpp`,
  `cell_map_validity.cpp` (`domain = 3`), and all Tier 2A Python transfer/wrapper
  changes preserve exact historical outputs and IDs.
- **Intentional identity changes:**
  1. **CAD uniform preseeded recovery (`plc3d.cpp`, `cad_mesh.cpp`):** CAD
     solids with `>= 4,096` preseeded+surface points (`cad-quadric/4925`,
     `2286`, fine P2/P3 rows) now retain parallel preseeded interior BCC sites
     through local recovery (`493,903 -> 493,804` tets on `cad-quadric/2286`),
     preserving deterministic `1/2/4/8`-worker bit-identity and all quality
     contracts; smaller meshes (`< 4,096` points) remain bit-identical.
  2. **F5 benchmark curve shape targets (`benchmarks/native_meshing.py`):**
     `_CAD_CURVE_TARGETS` aligned to the exact live Netgen/Gmsh thresholds on
     all 12 F5 rows (benchmark configuration only; no library default change).
  3. **Native certified surface mesh repair (`a3d48707a`, `20da8cfde`):** adds
     opt-in `repair_surface_mesh` / `SurfaceRepairPolicy` / `SurfaceRepairEvidence`
     and `phx_mc_surface_repair` / single-soup solid extraction without altering
     existing default import or meshing routes.

#### 5. Remaining limits

- **Reference-side ineligibility / missing external data (unchanged):** F9
  `cfMesh` prism-layer rows (`sphere/50`, `quadric/100`, `quadric/50`) fail
  their own `checkMesh`; F13 middle tet bank (`f13-tet1e5s.bin`) is unavailable;
  F14 fastest single-thread `VoroCrust` output fails convexity; layered quadric
  8-rank `snappyHexMesh` drops layer cells and produces nonpositive cells.
- **F8 8-rank quiet-admission note:** unmodified `f8-quiet-eight.py` stopped
  before Pair 1 (`load 2.718 > 2.3`) because active macOS desktop background
  load hovered at `2.4–3.1` even after 90s idle; the 3-pair interleaved 8W run
  with full `foam_meas.py` independent verification (`9/9` `scoped_quality` true,
  `science_passed: true`) beat live `snappyHexMesh` by `3.19×` (`1.764 s` vs
  `5.625 s`) and historical by `2.87×` (`5.066 s`) with exact `mesh_id` at load
  `4.00 -> 4.15`.
- **Dense polyhedral / high-order certification (Tier 2B, out of scope here):**
  F8 `prove=True` and F9 dense rational polyhedral cell certification remain
  separate from construction and slower than uncertified references.

#### 6. Final-head reruns of partitions changed by `ee62dc1e5`

`ee62dc1e5` changed `plc3d.cpp`, `hex_dominant.cpp`, and Python providers
whose rows the landing receipts above did not cover. Each such partition was
rerun on a pinned `b416ea9c6` worktree whose fresh Release build reproduced
the landed library byte for byte (`72a05c60...`, zero warnings). Partitions
ran one at a time, three interleaved rounds each, every one starting at
one-minute load at most 3.1.

| Partition | Result | Time vs reference | vs `4ffad9ea2` |
|---|---|---:|---|
| S3 recovery / volume (CGAL CDT3, HXT) | 12/12 | 0.19–0.73× | level |
| F8 hex-dominant, one thread (cfMesh) | 4/4 | 0.36–0.59× | faster (was 0.59–0.93×) |
| F14 Voronoi (VoroCrust) | 8/8 | 0.14–0.70× | level |
| F9 prism layers, sphere (live Gmsh) | pass | 0.854× | faster (was 0.887×); identical mesh |
| F7 surface and mesh remesh (MMGS, CGAL) | 13/13 | 0.15–0.75× | level |
| F6 3D graded, 9 rows (faster of MMG, Omega_h) | 9/9 | 0.17–0.86× | level or faster |
| F13 overset hex (TIOGA) | 3/3 | 0.02–0.20× | level |
| F13 overset tet (TIOGA; middle bank unavailable) | 2/2 | 0.13–0.18× | level |
| S5 NIST STEP washer (`hxt-fine`) | **miss** | 1.016× | was 0.982× |

Every row had zero unmet results and zero quality misses. The washer native
output is byte-identical to the `4ffad9ea2` receipt (51,758 tetrahedra, bank
`f02c3160...`). All three native samples (1.677, 1.671, 1.691 s) were faster
than two of the three `hxt-fine` samples (1.709, 1.700 s), but one faster
reference sample (1.644 s) sets the minimum. The row's margin is within
run-to-run variation on both heads, so it is **not qualified** and is
recorded as a timing miss, not retried. The graded partition began at load
2.85 and ended at 4.53, above the 3.4 limit; each of its rows cleared its
ceiling by at least 14%. Receipts: `artifacts/MAC/native-redesign/acceptance/final-ee62dc1e5/rerun/`.
