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
- A certifier counts only after an independent read-only review of its proof (Nov 20: the first three
  reviewed, `mesh_certify`, `zero_set_certify` and `verify_volume`, each had a soundness defect, all
  fixed). Rows certified by an unreviewed certifier are construction rows with sampled quality.
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
- Status (Nov 20, quiet macOS acceptance, references re-measured back to back): ACCEPTED. 2D 0.87-0.92x
  CGAL, 3D 0.88-0.93x, regular 0.95-0.99x CGAL Regular_3, periodic 0.43-0.63x CGAL Periodic_3 (direct
  covering-space engine; declared margin change), 1e3-1e6. Output canonicalization stayed mandatory for
  every caller (linear-time instead of an option).

### S0-C Flat tetrahedral state
- `native/meshcore/src/tet_mesh.hpp`, `tet_mesh_state.cpp`, `tet_mesh_capi.cpp`: cells, neighbors,
  vertices as flat arrays with the slot as identity; adjacency built by sorting packed face keys; no
  ordered maps on the creation or snapshot path; ancestry optional and off by default for construction.
- Tests: snapshot equality (as sets) against the previous state on the existing fixtures.
- Acceptance: creation of 1.16 M tets under 0.3 s (ledger: 4.5 s).

### S0-D In-repository benchmark
- `benchmarks/native_meshing_ledger.py` with its JSON record: the ledger rows (point-set Delaunay,
  closed surface to tets, refinement, improvement, implicit volume, CAD solid, curving, remesh,
  adaptation) on fixed generated inputs, phase-separated timing, our kernels through the C ABI, reference
  tools when installed (optional; skipped at the provider boundary), independent quality and distance
  measurement. Ceilings per row = the ledger's best reference.
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
- Status (Nov 20, quiet macOS acceptance against Gmsh HXT on the same surfaces): ACCEPTED. Sphere
  0.54 / 0.65x, stuffed torus 0.95 / 0.91x, grid torus 0.64 / 0.62x; recovery 0.35-0.61x CGAL CDT_3.
  The stuffed-torus rows reopen when S1 replaces stuffing (their inputs are regenerated from the
  restricted-Delaunay surface) and are re-measured in the next quiet window.

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
  unreachability proof exists, so the path and its evidence fields stay. The same sweep found
  FIXED recoveries of small twisted prisms and towers (16-30 points) running past 60 s.

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
  one native call `phx_mc_cad_mesh` on a flat geometry bundle; curves split once by integrating the
  size field and shared by every use; faces meshed in their native parameter chart by metric Delaunay
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
- S4 surface remesh and metric adaptation (above MMGS and MMG3D, inside tolerance).
- S5 STEP import and remaining exchange formats. S6 layers, hex-dominant, overset, Voronoi. S7 threads
  and devices at scale.

## Row ownership (every ledger row; `.tmp/ledger/`)

| Ledger rows | Reference to beat (single thread) | Slice | Machine |
|---|---|---|---|
| F1 point-set Delaunay 2D/3D, 1e3-1e6 | CGAL DT2 / DT3 | S0-B | macOS |
| Regular triangulation 3D, periodic Delaunay (native-kernels §2-3) | CGAL Regular_3 (2.6-2.9x today), Periodic_3 (5x) | S0-B | macOS |
| F1 planar CDT and quality refinement (square, square + ring + slit) | Triangle `pcQ`, `pq20a` / `pq30a` | S2-C (`cdt2d`) | Linux |
| F3 surface to tetrahedra; recovery; refinement; export | Gmsh HXT on the same surface; CGAL CDT_3 | S3 | macOS |
| Improvement passes on a given mesh (native-kernels §3) | MMG3D `-optim` at equal quality | S3-B | macOS |
| F4 implicit to tetrahedra | CGAL make_mesh_3, Gmsh | S1 | macOS |
| F2 CAD solid, F5 curved order 2/3 | Gmsh HXT per size; Netgen `Curve`, Gmsh `setOrder` | S2 | Linux |
| F7 surface remesh and decimation (1e6 to 1e3-1e5, dense and noisy torus; sphere ladder) | MMGS, CGAL isotropic remeshing | S4-Surface | Windows |
| F6 / F6b metric adaptation | MMG3D, Omega_h | S4-3D | Windows |
| F10 queries (closest point, signed distance, winding number, point location) | libigl AABB / FastWindingNumberBVH | S0 queries | macOS after S3 |
| F8 hex-dominant, F9 prism layers, F13 overset, F14 Voronoi | cfMesh / snappyHexMesh, Gmsh / Netgen, TIOGA, VoroCrust | S6 | assigned after S3 |
| STEP import and exchange formats (spline faces are S2, decision D1) | the S2 rows on the STEP models (washer, NIST) | S5 | Linux after S2 |
| Threads and devices | each row above at 1-8 threads | S7 | macOS |
