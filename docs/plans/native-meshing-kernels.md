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
- Every replaced route is a clean cutover: callers, tests, docs, changelog, generated data in the same
  change; no compatibility path.
- Determinism: the output depends on the input only (not on thread count or timing).
- All work runs on the macOS machine only (user decision); native changes are validated by `ctest`
  with Apple clang. Cross-platform byte identity is not required.

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

### S3-C One-call pipeline and certification
- `native/meshcore/src/surface_to_tets.cpp` (new) and C entry `phx_mc_surface_to_tets`: recovery,
  refinement, improvement in one call on flat arrays, returning cells, region labels, boundary
  classification.
- `native/meshcore/src/verify_volume.cpp` (new) and C entry `phx_mc_verify_volume`: exact orientation of
  every cell, combinatorial conformity, boundary equals the input surface, region consistency.
- Python: the PLC volume route (`phydrax/meshing/providers/_native_volume.py`,
  `_volume_generation.py`) cut over to the one call; `prove=True` runs the verifier and the existing
  source-fidelity statement for the given surface; `prove=False` returns the candidate.
- Tests: route tests updated; the certificate rejects a mesh with a flipped cell, a missing boundary
  face, and a cell outside the surface.
- Acceptance: the S3 rows of the benchmark beaten; quick CAD profile passes with the new volume stage.

## Later slices (planned when S3 is accepted; same rules)
- S1 implicit field to tetrahedra by isosurface stuffing on a body-centered cubic lattice
  (`implicit_stuffing.cpp`; Labelle and Shewchuk 2007, with quadrilateral splits through the smallest
  identity instead of the parity rule): lattice spacing from the chord rule, cut points by batched
  regula falsi, warping, conforming stencils. The analytic profile route certifies the result with the
  existing tubular-projection degree proof and certified facet deviation; a rejected lattice is
  repaired by a finer one. Targets: torus 0.02 / 0.01 in 10 / 20 ms, inside tolerance. Status
  (Nov 19, NOT accepted): the stuffing kernel alone takes 1.05 / 2.59 ms at a chord-rule spacing, but
  the route's construction (`PreparedStuffedImplicitVolume.construct`: stuffing at the route's
  spacing, lattice acceptance, fixed-boundary quality pass, `CellMesh`) on the ledger solids (sphere
  r 0.371; torus major 0.742, minor 0.371) takes sphere 0.026 / 0.033 s (CGAL opt=0 0.0014 / 0.0025,
  quality 7 degrees), torus 0.207 / 0.540 s (Gmsh 0.090 / 0.224) with 1,512 / 2,520 and 15,824 / 43,952
  cells (2.7-3x Gmsh's). All four certify through acceptance (fidelity sphere 0.0123 / 0.0069, torus
  0.0109 / 0.0063); acceptance with certification takes 7.6 / 27.5 s (sphere) and 60 / 180 s (torus).
  To finish: a graded lattice (coarse interior), a first lattice that passes acceptance, no quality
  pass when the stuffed cells already meet the schedule, faster `CellMesh` topology, and native
  certifiers (global embedding, validity, distance evidence once per acceptance). There is no
  `prove=False` path on this branch yet; the 10 / 20 ms targets are construction targets. General
  implicit fields (the adaptive route) follow with the graded lattice.
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
