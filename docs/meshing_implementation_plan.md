# End-to-end native meshing implementation plan

## 1. Deliverable and completion contract

This document was written as a static plan and is now the implementation plan of the `feature/native-meshing-20260929` worktree, rebased onto `dev` at `62abd2db0`. Proposed files and symbols below do not exist until their workstream lands; implementation revalidates the source baseline before modifying it.

The deliverable is a Phydrax-owned geometry-to-simulation meshing system, not a facade over other meshers. It includes native initial generation, improvement, adaptation, geometry preservation, state transfer, distribution, persistence, and solver integration. It also includes native CAD construction/query/Boolean/interchange capability, rather than treating OCCT independence as an unspecified future project.

Ordinary numerical/runtime infrastructure such as NumPy, JAX, compilers, and communication libraries is permitted. Gmsh, CGAL/pygalmesh, TetGen, fTetWild, Mmg/ParMmg, Omega_h, VoroCrust, Manifold, OpenVDB, Open3D, PyVista/VTK reconstruction, Qhull, OCCT, and METIS must not perform a geometric/meshing/partitioning operation on a route advertised as native. Explicit optional comparison/interoperability providers may remain; there is no hidden fallback to them. File codecs are distinguished from mesh algorithms, and any retained external codec is declared at interchange rather than described as an in-house codec.

Completion requires all mandatory positive workflows in this plan to execute with those engines absent, not merely to refuse their inputs safely. A refusal is a correctness success only for a deliberate negative case; it is a functionality failure for an admitted positive case. Quality, geometry, semantic, and resource requests cannot be relaxed to improve benchmark success rates.

General automatic high-quality all-hex construction, robust singular CAD arrangements, and reconstruction of underspecified defective surfaces have research-dependent boundaries. Those workstreams remain in scope. Their acceptance domains and evidence must be explicit; an unresolved research gate does not become a completed capability by changing its label or silently narrowing the user's scope. Narrowing the final goal requires a separate user decision.

### 1.1 Scope traceability

| Requirement from the assessment | Owning workstreams | Completion evidence |
| --- | --- | --- |
| Robust predicates, constructions, local topology and resource-bounded execution | W00, W01 | Exact/filtered agreement, adversarial degeneracies, transactional cavity validity, bounded allocation |
| Native curve, planar and curved-surface generation | W02, W07 | Feature/trim-conforming meshes with continuous fidelity bounds |
| Constrained, quality-controlled tetrahedral generation | W03 | Native solids with cavities, embedded features and interfaces; sliver-sensitive quality and complete domain coverage |
| Implicit, image and multimaterial generation | W04 | No missed certified components, interface conformity, explicit interpretation of image samples |
| Imperfect surfaces, wrapping, reconstruction and surface Booleans | W05 | Explicit repair/envelope semantics, source ancestry, native reconstruction and volume fill |
| Periodic generation and adaptation | W06 | Quotient-complex validity, paired features/geometry/fields, periodic quality and conservation |
| End-to-end CAD independence | W07 | Native construction, queries, intersections, partition/Boolean operations and declared STEP/IGES coverage without OCP |
| Native 3D and curved-surface remeshing | W08 | Metric compliance, legal local edits, preserved geometry and regions |
| Geometry and field/state transfer | W00, W08, W12, W14 | No curved-to-affine loss, appropriate conservation/commuting/positivity/history guarantees, atomic rollback |
| Complete boundary-layer and high-order workflows | W09 | Native core fill, hybrid conformity, continuous fidelity, global embedding, preserved layer schedule |
| Structured, swept, multiblock, quad, hex-dominant and all-hex meshes | W10 | Actual required cell families, valid block/gluing topology, declared all-hex research gate |
| Arbitrary-domain Voronoi/power/polyhedral meshes | W11 | Conforming domain/feature boundaries, reciprocal faces, coverage, admissible polyhedral geometry |
| CPU, device and distributed initial generation/adaptation | W01, W12 | Capacity scaling, local communication, distributed commit, native graph partitioning |
| Lifecycle, checkpoint/restart and deterministic identity | W12 | Reproducible accepted epochs, restart and repartition without semantic loss |
| Native overset connectivity and moving assemblies | W13 | Hole cutting, donor/receptor coverage, motion updates and declared transfer semantics |
| Solver-aware adaptation, differentiable design and learned proposals | W14 | Error/cost improvements, valid fixed-epoch derivatives, trusted proposal acceptance |
| Independent certification, qualification, benchmarks and complete migration | W00, W15 | Contract/corpus matrix, isolated native workflows, baseline comparisons, current docs/API data |

### 1.2 Baseline findings that drive the design

- `native/meshcore/include/phydrax_meshcore.h` exposes 2D/3D Delaunay and regular triangulation, but constrained Delaunay/refinement only in 2D. `delaunay3d.cpp` is a convex-hull point-set engine, not a constrained solid mesher.
- `_adaptation.py` has native simplex bisection and planar 2D metric routes. Dimension-generic SPD metric preparation is not native 3D remeshing.
- `_finalize_native` omits successor `CellGeometrySpec`; `_canonical.py` constructs affine geometry when it is omitted. Admission does not explicitly exclude every non-affine source. This source-level geometry-loss risk needs a failing-before regression during implementation.
- `_audit.py` uses corner coordinates for topology intersection checks; `_cell_geometry_validity.py` owns per-cell determinant certification. Neither alone establishes global injectivity and complete source fidelity of a curved mesh.
- `_curving.py` measures constrained-node CAD residuals, not a continuous two-sided boundary deviation.
- `_compartments.py` currently calls wildmeshing and assigns compartment labels from rounded segmentation samples at cell centroids. Native material conformity must be established by the meshed interface complex, not inferred from centroids.
- Native advancing layers exist in `_boundary_layer.py`; `GmshProvider.fill_boundary_layer_core` still owns the demonstrated core fill.
- `geometry/_triangulation.py` owns bounded box/convex-domain Voronoi and power diagrams. It does not provide arbitrary-domain conforming polyhedral generation.
- Native part-sharded bisection is refinement-only and globally exchanges candidate information, then commits through a host gather. It is not distributed native initial generation or complete distributed remeshing.
- `pyproject.toml` currently requires `cadquery-ocp-novtk`. `geometry/reconstruction/_core.py` uses SciPy/Qhull Delaunay and PyVista reconstruction. These are additional dependency-cutover edges, not just meshing-provider changes.
- The quadratic default 3D simplex-count expression in `_meshcore.py` is an admissible output limit, not a quadratic initial allocation. Native tetrahedral storage currently reserves approximately seven slots per input point. Do not base the performance plan on a false preallocation diagnosis.

### 1.3 Revisions required by `dev` at `62abd2db0`

The merges after the plan was drafted add owners that the implementation must reuse rather than duplicate:

- `phydrax.lifecycle.CompositionRebind` (`lifecycle/_composition_rebind.py`) is now the one accepted cross-owner rebind of a running composition: topology, discretization, routes, transfers, physical/history/RNG state and their dependencies. Every mesh-epoch change that reaches a running solver (W08 consumer migrations, W12 repartition/migration, W13 motion, W14 decisions) stages its topology reprepare and state transports as `CompositionEntry` dispositions and commits through that rebind. `FiniteElementTopologyTransfer.epoch_transition(...)` and `MeshDistributionTransition.composition_transport(...)` are the existing physical-remap and ownership-migration transports; new geometry/field transfers expose the same transport kind instead of a meshing-local commit path.
- `MeshInterfaceAttachment` (`meshing/_interface_binding.py`) attaches exact part entities to authoritative geometry entities through a certified `GeometryAssociation`, with sided orientation. Native generators must publish complete `GeometryAssociation` evidence so attachments work on native results, and `RegionMeshingEvidence` interface records bind to the same authoritative geometry entity identities. Transitions revalidate attachments against the successor part revision (`MeshAssembly.require_attachment`); a stale attachment is refused, never copied.
- `PreparedFieldQuery`/`PreparedFieldReconstruction` (`discretization/_field_query.py`, `_views.py`) and the expanded FE `_point_interpolation.py` own located point evaluation with pointwise evidence, exact transpose and masked/complete coverage. W13 donor interpolation builds its interpolative route on them instead of a new donor stencil owner; W08 point-based transfers reuse them where a located evaluation is the transfer.
- Boundary trace spaces and side actions (`_boundary_trace_space.py`, `_side_actions.py`) own one-sided traces; W09/W13 interface and fringe coupling use them rather than recomputing side restrictions.
- `jax.lax.custom_root` is banned by the linter; implicit differentiation of projections/realizations uses `phydrax._custom_root.custom_root`.
- The core dependency is now OCP 8 (`geometry/brep/_projection.py` reads `Bnd_Box` corners). W07 still removes OCP from native paths; the optional comparison adapter targets OCP 8.

## 2. Architecture and nonnegotiable invariants

### 2.1 Existing owners remain canonical

| Existing owner | Responsibility retained or extended |
| --- | --- |
| `phydrax.geometry`, its B-Rep/implicit/surface/multiregion owners | Authoritative represented geometry, source identity, strata, charts, region adjacency, queries and approximation evidence |
| `native/meshcore` | Robust geometric decisions and constructions, local connectivity, bounded cavity operations and host-native generation kernels |
| `phydrax.meshing` | Physical requests, route admission, generation/adaptation schedules, organization, association, lineage, acceptance and publication |
| `phydrax.discretization` | `CellMesh`, coordinate elements/`CellGeometrySpec`, incidence/orientation, cell validity and discretization-specific transfers |
| `phydrax.sparse`, existing BVH/spatial owners | Relations, topology-aware gathers/reductions, bounded neighborhoods, candidate search and adjoints |
| `phydrax.linalg`, `phydrax.nonlinear`, `phydrax.optim` | Reusable solves/factorizations, local roots, optimization and their failure evidence |
| Existing lifecycle, execution, sharding and qualification owners | Transactions/checkpoints, distributed execution, runtime identity and evidence artifacts |

No second public mesh carrier, CAD identity system, spline evaluator, periodic lattice, metric algebra, generic solver, or execution runtime is introduced. Mutable native construction state is a private prepared execution artifact; accepted snapshots remain canonical immutable carriers.

### 2.2 Public API decision

- Add **proposed new** `NativeMeshingProvider` and `NativeMeshingPlan` in `phydrax/meshing/providers/_native.py`.
- Reuse `SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, `VolumeMeshingSpec`, `CellMeshingTarget`, `MeshingScope`, `CellFamilyPolicy`, sizing/feature/region/layer/periodic controls and `MeshingLimits`.
- Add **proposed new** `CurveMeshingSpec` to the existing `_contracts.py` owner for standalone 1D interval meshes and declared curve networks; extend `MeshingSpecification`, operation admission, facades and support reporting. Curve generation must not require pretending that a 1D request is a surface request.
- Add **proposed new** `NativeMeshingRoute`/`NativeMeshingOptions` in `providers/_native_options.py` for algorithm selection and numerical scheduling, separate from physical requests. A prepared plan records the chosen route and does not switch it after failure. Add typed transfinite/sweep/block-interface controls to `_controls.py` for the information required by W10; no undocumented `options` dictionaries.
- `NativeMeshingProvider.plan(source, specification, ...)` performs typed source admission and prepares one declared route. Source-specific bindings retain their owning geometry/physical contracts; no catch-all dictionary of geometry parameters is admitted.
- `NativeMeshingPlan.execute(...)` returns `CellMeshingResult` only after required audit and compliance succeed. Invalid input kinds/values use the existing contract errors. Failed scientific execution raises `MeshingFailure` with structured stage evidence and, when valid, a lifecycle checkpoint. An intermediate valid mesh that misses a hard request is not a successful result.
- Existing `prepare_mesh_adaptation` / `execute_mesh_adaptation` remain the one adaptation API. Extend their route selectors instead of adding competing convenience APIs.
- Migrate `NativeImplicitProvider` consumers to the native provider and remove the obsolete facade; keep `geometry.implicit` discovery and fixed-route realization as the algorithm owner. The existing standalone low-level triangulation/diagram APIs remain distinct useful primitives, not aliases to generation.
- Move compartment generation onto this provider with a clean source/request/result cutover: replace the engine-coupled `CompartmentMeshingSpec` with geometry-owned `CompartmentMeshingSource` plus ordinary `VolumeMeshingSpec` and `NativeMeshingOptions`; return `CellMeshingResult` with canonical region evidence. Remove `FTetWildCompartmentProvider` and `CompartmentMeshingResult` after migrating their actual consumers. Preserve all compartment identity/adjacency semantics; fTetWild tuning remains only on its explicit comparison path.
- Keep external provider options outside native physical requests. Native execution never reads `GmshOptions`, `MmgOptions`, or another engine's tuning vocabulary.

Observed LSP references for the provider/specification/result cutovers include both meshing facades, `tests/unit/meshing/test_implicit_provider.py`, `tests/unit/meshing/test_compartment_meshing.py`, and the actual `NeurofluidCase`/transport consumer in `phydrax/applications/neurofluid/_model.py`. LSP also identifies the current neutral `PlanarEmbedding` declaration in `geometry/brep/_planar.py`. During implementation, rerun references before changing exported symbols and migrate documentation, examples, generated records and dynamically exposed surfaces in the same cutover.

### 2.3 Geometry and constraint compilation

Add **proposed new** `phydrax/geometry/_meshing_domain.py` for one prepared, revision-bound meshing-domain view. It owns a real invariant: all region, feature and boundary queries refer to one authoritative geometry revision and physical frame. It is not a second geometry representation.

The prepared view must expose:

- Explicit corner/curve/surface/region identities and incidence, including oriented region pairs at interfaces and higher-valence material junctions.
- Geometry-owned classification, projection/intersection and bounded-distance/normal queries with pointwise status; capability absence is explicit.
- Source-specific accuracy: exact represented PLC, bounded CAD approximation, certified implicit enclosure, sampled image interpretation, or an explicitly repaired envelope source.
- Read-only source data and a geometry/design epoch; changing the source invalidates prepared constraints, spatial indices and certificates.
- Batched geometry queries. A native construction queue may pause for a bounded batch of geometry or numerical requests; it must not invoke Python/JAX or synchronize a device once per inserted vertex.

`meshing/_domain.py` (**proposed new**) compiles this view plus existing controls into protected constraints, region seeds, size/metric worksets and periodic orbits. It does not reconstruct identity from coordinates, display names, shapes, or external entity tags.

### 2.4 Acceptance and numerical invariants

1. Separate topology legality, per-cell map validity, global embedding, source fidelity, semantic conformity, quality compliance and transfer evidence.
2. Exact predicates certify decisions for their declared represented-coordinate domain. Inexact constructions must be bounded and revalidated; no epsilon tie-breaking, unreported jitter, or automatic repair of scientific geometry.
3. A determinant certificate is not a global injectivity theorem. A set of projected nodes is not a continuous Hausdorff bound. A material label at one point is not a conformity certificate.
4. Requested hard constraints, required cell families and feature/material identities must survive every operation. Impossible quality bounds at acute fixed features are reported as conflicts or explicit unmet requests, never excluded from the quality statistic without a declared policy.
5. Budget every allocation before it occurs: entity arrays, temporary cavities, queues, spatial candidates, exact-arithmetic work, geometry queries, certification subdivisions, lineage, transfers and per-rank state. Report work exhaustion separately from invalid geometry.
6. A rejected operation/epoch preserves the accepted geometry, topology, state, IDs and ownership. Failure evidence remains available. No partially applied cavity or distributed half-commit is visible.
7. Stable scientific IDs are distinct from reusable local slots. Native local indices remain bounded; global identities use the existing wide ID contract. Repartitioning does not renumber scientific entities.
8. Declare determinism precisely. Reproducible routes use stable-ID tie ordering and specified reductions; faster nondeterministic schedules, if offered, require explicit selection and distinct runtime evidence. Do not claim cross-platform bitwise equality from input sorting alone.
9. Dynamic topology, CAD Boolean branches, active constraints and remeshing are nondifferentiable events. Coordinate/field derivatives are valid only inside an accepted fixed-route epoch, with margins and invalidation evidence.
10. New strict classes inherit `StrictModule`, use truthful `phydrax.typing` contracts and constructor validation, and keep static metadata off device leaves. Canonical records have deterministic ordering and no internal schema versions or generation suffixes.

## 3. Dependency order and integration ownership

The workstream labels below are planning identifiers, not runtime versions. Each workstream has one integration owner. Independent work is parallelizable only after shared contracts are fixed.

| Wave | Work that can proceed | Prerequisite/output boundary |
| --- | --- | --- |
| A | W00 contract closure; W01 kernel extraction/robustness; W07 native CAD representation; W15 corpus/baseline preparation | Freeze acceptance scopes, identity contracts and admitted input families before adding algorithms |
| B | W02 domain/curve/planar/surface preparation; W04 enclosure/image semantics; W05 reconstruction; W06 periodic topology; W07 intersection/query work | Reuse W01 predicates and agreed domain-query/stratum contracts; do not wait for every CAD importer |
| C | W03 constrained volume/refinement; W04 implicit/image volume routes; W05 envelope volume route; W11 polyhedral construction | All volume routes share robust local topology and preserve their distinct source-conformity semantics |
| D | W08 surface/tetra remeshing and transfers; W09 layers/high order; W10 structured/quad/hex routes; W13 overset | Native constrained fill and geometry queries provide common inputs; family-specific algorithms remain separate |
| E | W12 scalable/distributed execution and restart; W14 solver/design integration; W10 general all-hex research integration | Parallel kernel design starts in A, but full distributed/family acceptance requires the serial algorithms and transfer contracts |
| F | W15 complete qualification, clean public/documentation cutover and release | Every mandatory positive workflow succeeds with external engines absent; all research/coverage gates are accounted for |

Geometry-preservation and transfer contracts are designed in A, implemented alongside W08/W09, and checked together at integration. Do not make high-order generation and adaptation mutually dependent implementation projects. The coordinate-map/transfer interface is their shared prerequisite.

General all-hex research and singular-CAD investigations start early with bounded prototypes and independent validation, not after every engineering workstream. Their mature integration remains gated by the same correctness and resource contracts.

## 4. W00 — Close existing contract holes and establish independent acceptance

**Prerequisites:** none. **Purpose:** prevent expansion from multiplying existing semantic gaps.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/meshing/_adaptation.py` | Reproduce the curved-to-affine issue before editing. Immediately refuse unsupported non-affine adaptation rather than dropping coordinate geometry. Final W08 implementation supplies a geometry-aware successor and removes only the refusal made obsolete by supported transfer. Admission also checks field/material transfer obligations before numerical work. |
| Existing `phydrax/meshing/_canonical.py` | Keep intentional affine construction for callers supplying only an affine carrier. Require adaptation/generation callers claiming preserved geometry to provide its coordinate specification; do not guess from node counts. |
| Existing `phydrax/meshing/_contracts.py` | Extend existing physical requests and limits for explicit source fidelity, quality targets, topology/repair permissions, work/cavity/query budgets and native route controls. Add typed failure evidence/checkpoint attachment without duplicating status conventions. Required combinations belong to admission, not capability flags alone. |
| Existing `phydrax/meshing/_audit.py`, `_audit_topology.py`, `_quality.py` | Separate straight-corner evidence from mapped-cell/global/source evidence. Retain REJECT/RECORD/SKIP semantics but native generation requires the checks relevant to its declared output. Surface meshes may be intentionally open; volume domains and material closures may not skip their applicable closure checks. |
| Existing `phydrax/discretization/_cell_geometry_validity.py` | Strengthen and document polynomial/rational degree coverage, rounding-error enclosure derivation, subdivision budgets and mapped-cell binding. Audit supported high orders and denominators rather than assuming a numerically computed Bernstein transform is exact. Reuse this owner for W09/W10/W11. |
| Proposed new `phydrax/geometry/_mesh_certificates.py` | Establish geometry-owned source-fidelity, global-embedding and domain/interface-coverage records and affine algorithms before W03 needs them. Use independently constructed meshes for initial qualification; W08/W09 extend curved-map algorithms later. This removes a generation-to-adaptation certification dependency cycle. |
| Proposed new `phydrax/meshing/_certification.py` | Compose separately owned global-embedding, source-fidelity, domain-coverage and semantic-conformity evidence into one route-specific acceptance schedule. It orchestrates certificates; it does not duplicate their geometric algorithms. |
| Existing `phydrax/meshing/_result.py`, `_trace.py` | Bind evidence to source revision, topology, actual coordinate arrays, coordinate-element layout, policies and runtime. Expose exact failing/unresolved entities and achieved versus requested quantities. Successful result construction remains impossible when a mandatory check is unresolved. |
| Existing `phydrax/geometry/_certified_implicit.py`, `_certificate.py` | Distinguish user-supplied bounds from independently established bounds, bind covers to source/state/domain, validate cover completeness and topology premises. A theorem-name string plus regular sampled boxes is not sufficient evidence for an executable global-topology claim. |
| Existing `phydrax/meshing/_scope.py`, `_association.py`, `_organization.py` | Reuse source/entity identity and CAD/material incidence across all new routes; reject stale scope, ambiguous collapse class and contradictory region ownership before work. |
| Existing `phydrax/geometry/brep/_planar.py`, `_projection.py`, `_model.py`, B-Rep/geometry/meshing/provider facades; proposed new `phydrax/geometry/_planar_embedding.py`, `phydrax/geometry/brep/_projection_contracts.py` | Extract engine-neutral `PlanarEmbedding` and projection status/evidence/query protocols to canonical owners, migrate direct imports without aliases, and make optional implementation imports lazy. `_contracts.py` importing a neutral planar frame must not execute the old OCP-backed `_planar.py`; association/curving annotations must not load OCP projectors. Isolate existing engine-backed implementations without pretending that W07 native CAD algorithms already exist. |

### Tests, smoke and gate

- Extend `tests/unit/meshing/test_bisection.py`, `test_device_adaptation.py`, `test_cad_curving.py`, `test_audit_quality.py`, `test_contracts.py` and `test_trace.py` for consumer-visible geometry retention/refusal, scoped certificates, typed failures and rollback.
- Add **proposed new** `tests/unit/meshing/test_global_certification.py` for positive-Jacobian but overlapping disconnected volumes, curved cells with intersecting interiors/boundaries, contained components, omitted material interfaces, gaps and double coverage. Include correct valid controls, not negative tests alone.
- Extend `tests/unit/geometry/test_implicit_surface.py` and `test_geometry_validity.py` for stale/incomplete/unbound enclosures and explicit unresolved topology.
- Establish an early engine-free import and native affine/PLC smoke environment after neutral-contract/lazy-facade extraction. Until W07's dependency metadata cutover, install the built wheel without resolving optional legacy CAD dependencies and supply its declared numerical/native-kernel prerequisites explicitly; this proves execution isolation, not yet ordinary-install independence. W07/W15 later require a normal resolver-driven engine-free installation. Do not defer the import boundary until after checkpoint 2.
- Smoke a curved mesh through accepted/refused adaptation and inspect actual coordinate-map evaluations, not only stored degree metadata.
- Gate: no claimed property is inferred from a weaker check; unsupported curved transfer fails without modifying accepted state; tests establish the issue before and its closure after the change.

## 5. W01 — Extend the native robust kernel and shared local topology

**Prerequisites:** W00 contracts. **Purpose:** reuse the existing core instead of introducing an unrelated tetrahedralizer.

### Algorithm decisions

- Extract the existing 3D incremental triangulation state into a reusable internal owner. Preserve ghost-cell conventions, exact conflict predicates, canonical tie order and existing point-set semantics.
- A cavity edit owns its oriented boundary, reciprocal adjacency, constrained strata, region labels, local vertex/cell slots, proposed constructions and work budget. Build a candidate separately; commit only after its boundary and semantic invariants agree with the removed cavity.
- Implement exact segment/triangle, triangle/triangle and point/facet classification needed for recovery, including coplanar overlap, endpoint contacts and coincident constraints. Shared-boundary adjacency is distinguished from illegal intersections.
- Exact predicates do not make circumcenters or intersection coordinates exact. Use source-faithful bounded constructions, exact postclassification and explicit unresolved/refinement-limit status where representability prevents a legal insertion.
- Keep the existing numeric admissibility domain explicit. Any internal normalization must retain original identity and a proved relationship to the original-coordinate decisions; do not silently round away degeneracy through recentering.
- Reuse native linalg/nonlinear for reusable small solves and roots. Batch requests from the construction schedule. If a performance-critical C++ inner kernel needs a lower-level solve implementation, extend the owning linalg lowering with measured justification rather than add an independent mesh-local Gaussian-elimination routine.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `native/meshcore/src/delaunay3d.cpp` | Move reusable topology/state into the new internal header; keep point-set C entry points and their observable deterministic behavior. Add prepared incremental execution needed by volume routes without rebuilding the convex hull per point. |
| Proposed new `native/meshcore/src/triangulation3d.hpp` | Own tetrahedral adjacency, location, ghost hull, reusable slots, constraint references and transactional insertion/removal. No CAD provider handles or Python objects. |
| Existing `native/meshcore/src/triangulation2d.hpp`, `cdt2d.cpp` | Reuse planar recovery/refinement; share meaningful cavity and protection invariants without forcing 2D and 3D into one opaque dispatcher. Preserve hole/constraint/Steiner-limit semantics. |
| Proposed new `native/meshcore/src/cavity.hpp` | Own bounded tentative-edit buffers, oriented boundary comparison, reciprocal-link checks, declared preserved facets and commit/rollback. Family-specific reconstruction remains in its algorithm owner. |
| Existing `native/meshcore/src/predicates.cpp`, `predicates.hpp`, `expansion.hpp`, `filtered.hpp` | Extend robust classifications using the same exact arithmetic and filter uncertainty semantics. Expose weighted predicates only where a real construction consumer needs them. |
| Proposed new `native/meshcore/src/intersections.cpp` | Robust intersection/contact classifications and bounded construction records reused by PLC recovery and surface arrangements. Avoid a second triangle-intersection convention in Python. |
| Existing `native/meshcore/src/mesh.hpp`, `core_capi.cpp`, `capi_guard.hpp` | Add owned generation-result metadata and diagnostics for constraint/subfacet lineage, region IDs and operation status. Account for live/work/peak allocations and guard every ABI exit. Keep local indices and global scientific IDs distinct. |
| Existing `native/meshcore/src/spatial_sort.hpp` | Reuse deterministic BRIO/Hilbert ordering; add stable feature/region-aware scheduling only where necessary. Do not repeat global sorting inside every local edit. |
| Existing `native/meshcore/include/phydrax_meshcore.h` | Add bounded prepare/advance/finalize generation calls, explicit diagnostics and metadata accessors. No ABI-generation suffixes. Published release/build identities remain authoritative external lifecycle data. |
| Existing `phydrax/_meshcore.py`, `native/meshcore/python/phydrax_meshcore/__init__.py` | Bind new calls, retain ownership safely, validate output schemas and statuses, convert arrays once and avoid redundant large copies. Partial native candidates cannot bypass meshing acceptance. |
| Existing `phydrax/linalg/_small_batched.py`, `_local_blocks.py`; `phydrax/nonlinear/_local_root.py` | Reuse prepared batched construction solves/root isolation; extend their owning contracts only for genuinely missing capabilities with retained rank/conditioning/convergence evidence. |
| Existing `native/meshcore/CMakeLists.txt`, `native/meshcore/pyproject.toml`, `NOTICE` | Register sources/tests/build-hash inputs, preserve strict floating-point flags, package native sources and document independently implemented algorithm provenance/licensing. |

### Tests, smoke, benchmark and gate

- Extend native `test_predicates.cpp`, `test_capi_guard.cpp`, `test_delaunay2d.cpp`, `test_delaunay3d.cpp`, `test_cdt2d.cpp` and `test_clip.cpp` only for changed contracts.
- Add **proposed new** `native/meshcore/tests/test_cavity.cpp` and `test_intersections.cpp`: orientation, reciprocal adjacency, exact degeneracies, constrained-face preservation, rollback, slot reuse, capacity/ID overflow and allocation failure.
- Extend `tests/unit/geometry/test_predicates.py`, `test_meshcore.py`, `test_meshcore_loader.py`, `test_convex_intersections.py` for public behavior and clean refusal, not ABI spelling/source text.
- Smoke native point-set construction before/after extraction and a sequence of committed/rejected cavity edits. Independently verify coverage/orientation and original-source immutability.
- Benchmark location, conflict discovery, construction, canonical publication, predicate uncertainty/exact resolution and retained/peak bytes separately over increasing point count, local valence and degeneracy rate. Preserve the existing point-set benchmark baseline.
- Gate: reusable native topology is not a regression in numerical semantics, deterministic ordering, status or resource behavior; no second numerical substrate is introduced.

## 6. W02 — Compile geometry/controls and generate curves, planar domains and surfaces

**Prerequisites:** W00/W01; native CAD query implementation from W07 for CAD-specific gates. PLC/analytic paths do not wait for all CAD interchange work.

### Algorithm decisions

- Discretize authoritative feature curves once under chord/normal/metric bounds. Share the resulting vertices and constrained chains across adjacent surface patches; do not independently remesh seams and weld them by proximity afterward.
- Expose standalone curve discretization through `CurveMeshingSpec`, including open/closed curves, embedded curves and explicitly declared network junctions. Source incidence determines whether a junction is legal; default surface/volume manifold assumptions cannot silently reject or merge a valid curve network.
- Planar domains reuse constrained Delaunay and add spatially varying size/metric requests, region classification and protected embedded points/curves. Preserve narrow features and holes under explicit work bounds.
- Curved surfaces use chart-aware constrained triangulation or restricted-surface refinement according to the source capability. Handle seam charts, poles and overlapping charts through existing atlas/trim ownership. UV quality alone is not physical-space quality.
- Surface refinement balances approximation error, metric quality and feature protection. Reproject only under the owning source query and validate orientation/topology after the construction.
- Constraint admission detects incompatible size/gradation/feature/layer/periodic requests before growth. A hard bound is not converted to a preference.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Proposed new `phydrax/geometry/_meshing_domain.py` | Prepared source/stratum/query contract described in section 2.3; compose existing `GeometrySource`/`CompiledGeometry`, B-Rep, surface, multiregion and image representations. |
| Existing `phydrax/geometry/_contracts.py`, `_capabilities.py`, `_atlas.py`, `_validity.py` | Add only required query/bound capabilities and their status/identity bindings. Preserve represented-versus-physical geometry distinction and unique/regular projection evidence. |
| Proposed new `phydrax/meshing/_domain.py` | Resolve `MeshingScope`/region/patch/feature controls against authoritative strata, compile cavities/voids and material adjacency, produce geometry-query batches and the protected constraint complex. |
| Existing `phydrax/meshing/_contracts.py`, `_controls.py`; proposed new `phydrax/meshing/providers/_native_options.py` | Add standalone curve specification/admission and explicit native algorithm selection; extend the specification union/support reports and external-provider unsupported-operation checks. Add typed transfinite/sweep/block-interface data without mixing physical requirements with algorithm tuning. |
| Existing `phydrax/meshing/_controls.py`, `_sizing.py`, `_metric.py` | Compile scalar and tensor requests through existing bound/intersection/gradation machinery; carry resulting conflicts and achieved constraints into generation. Add explicit curve discretization and source-fidelity policy where the current specifications cannot express them. |
| Proposed new `phydrax/meshing/_surface_generation.py` | Own curve-to-patch scheduling, planar region generation, physical-space surface refinement and patch assembly; use current `CellMesh`, association and organization contracts. |
| Proposed new `native/meshcore/src/surface_mesh.cpp` | Bounded triangle cavities/reconnection and curve/patch constrained connectivity; source geometry is supplied through prepared/batched query results rather than per-vertex Python callbacks. |
| Existing `phydrax/geometry/_triangulation.py` | Expose necessary prepared constraints/refinement evidence from native 2D construction without turning low-level triangulation into the public high-level mesher. |
| Existing `phydrax/geometry/surface/_model.py`, `_contracts.py`, `_high_order.py` | Consume certified generated surface topology and source associations. Preserve a valid open-surface result separately from a closed-solid boundary requirement. |
| Proposed new `phydrax/meshing/providers/_native.py` | Implement native provider admission, prepared source dispatch and staged successful publication. Central exhaustive route selection; substantial algorithms stay in their owners. |
| Existing `phydrax/meshing/providers/_implicit.py`, both meshing facades | Migrate native implicit entry to the new provider; retain reusable implementation under the owning implicit route, remove obsolete public provider/plan aliases after all callers migrate. |

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/meshing/test_native_surface_generation.py` and `test_meshing_domain.py`; extend `test_scopes.py`, `test_sizing.py`, `test_metric.py`, `test_implicit_provider.py` (rename to its native surface contract owner when appropriate).
- Add **proposed new** `tests/unit/meshing/test_native_curve_generation.py`: interval ordering/orientation, closed loops, junction identity, periodic curves, sharp features, size/fidelity bounds and capacity refusal. Smoke a native curve/network mesh through the existing one-dimensional/metric-network consumer.
- Required cases: polygon holes and embedded cracks, sharp/near-collinear corners, multiple regions sharing exact chains, periodic seam preparation, open curved patches, spherical poles, trimmed rational patches, nonuniform curvature, mismatched units, stale revisions and contradictory bounds.
- Add **proposed new** `examples/native_surface_meshing.py`. Smoke geometry → feature curves → shared surface mesh → source-fidelity/global-embedding certificate → scalar surface solve.
- Benchmark source-query work, curve sampling, surface generation, projection, topology assembly and certification independently; vary curvature, trim complexity, feature proximity and requested edge count.
- Gate: no cracks at shared source entities; no inferred semantic welding; source-fidelity and physical metric requests are independently checked.

## 7. W03 — Native constrained and quality-controlled tetrahedral generation

**Prerequisites:** W01/W02. **Shared interfaces:** region/constraint IDs from W02, periodic orbit hooks from W06, geometry-query worksets from W07/W04, independent acceptance from W00.

### Algorithm decisions and sequence

1. Validate the oriented piecewise-linear constraint complex: legal facet intersections, per-region closed boundaries, permitted embedded sheets/curves, cavities and junctions. A globally nonmanifold material interface network is not rejected merely because each sheet is not a standalone closed surface; region incidence/link conditions decide legality.
2. Protect corners/curves and small-angle features with explicit admissible neighborhoods. Establish whether boundary subdivision is allowed or the boundary must remain fixed.
3. Build the initial triangulation with the native engine, recover segments and facets by local cavity operations and permitted Steiner insertion, and retain original constraint/subfacet ancestry.
4. Classify connected tetrahedral regions by constrained-facet adjacency and authoritative seeds/oracles. Validate cavity/region coverage; do not select arbitrary nonconvex solids by tetrahedron centroid alone.
5. Refine by geometric error, size/metric and shape criteria using bounded priority queues, encroachment protection and candidate construction. Stop with unmet-criteria evidence when budgets or geometric restrictions prevent completion.
6. Improve worst elements using topology changes and coordinate optimization; evaluate slivers explicitly. Weighted exudation is a declared stage with its own protected-feature and size/fidelity rechecks, not a synonym for regular triangulation.
7. Audit topology, positive geometry, nonoverlap, domain coverage, interfaces, requested families and quality. Publish once through the canonical result path.

Preserving a supplied triangular boundary exactly is distinct from conforming Delaunay refinement that may require boundary Steiner points. The plan must expose those policies and use a constrained-cavity fill for immutable boundaries. A route may not split a boundary-layer cap to make its Delaunay algorithm succeed and then claim the original interface was preserved.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Proposed new `native/meshcore/src/plc3d.hpp`, `plc3d.cpp` | Bounded PLC validation/recovery, protected segments/facets, region flood classification and exact constraint ancestry. Reuse W01 topology and intersections. |
| Proposed new `native/meshcore/src/feature_protection.cpp` | Shared corner/curve protection and encroachment decisions for surface/volume/periodic refinement; numeric kernel only, source identity remains Python geometry-owned. |
| Proposed new `native/meshcore/src/refine3d.cpp` | Stateful size/shape/error-driven refinement, deterministic priority queues, legal insertions and exact resource/termination evidence. |
| Proposed new `native/meshcore/src/improve3d.cpp` | Own reusable tetrahedral edge removal, bounded cavity reconnection, relocation candidate acceptance and protected weighted sliver treatment over W01 cavity transactions. W03 generation and later W08 adaptation both consume these kernels; W03 has no reverse dependency on W08. |
| Existing native ABI/build files and `phydrax/_meshcore.py` | Expose prepared constraint/region metadata, batched geometry/construction query exchange, achieved quality, failed entity IDs and resumable bounded state. |
| Proposed new `phydrax/meshing/_volume_generation.py` | Own phase ordering, quality/geometry constraints, source-query batches, region/patch assembly and publication. No external volume provider is invoked. |
| Existing `phydrax/meshing/_quality.py`, `_optimization.py`, `_metric.py` | Reuse cell/metric quality, fixed-topology optimization and native solve evidence. Add consumer-visible dihedral/sliver constraints where missing; use separate geometric and metric quality thresholds. |
| Existing `phydrax/meshing/_association.py`, `_lineage.py`, `_canonical.py`, `_result.py` | Publish generated-on-geometry parentage, region/interface labels, canonical IDs, complete evidence and genuine unknown ancestry where the source is not a mesh. Do not fabricate interpolation stencils for newly generated interior points. |
| Existing `phydrax/meshing/providers/_gmsh_boundary_layer.py` | Retain the explicit comparison path only; move native core-fill consumption to W09 using `_volume_generation.py`, with the same or stronger fixed-node/facet checks. |

### Tests, smoke, benchmark and gate

- Add **proposed new** native `test_plc3d.cpp`, `test_refine3d.cpp`, `test_improve3d.cpp` and Python `tests/unit/meshing/test_native_volume_generation.py`, `test_native_volume_quality.py`.
- Positive corpus: convex and nonconvex solids, reflex polyhedra needing interior Steiner points, nested cavities, thin channels, disconnected solids, internal sheets/curves, two-material interfaces and triple junctions, cospherical seeds, small-angle protected features, sliver-rich inputs and fixed layer caps.
- Negative corpus: intersecting incompatible constraints, inconsistent orientation/region seeds, duplicate conflicting facets, nonrepresentable constructions, unattainable hard targets, depleted exact-work/queue/cell/byte budgets and cancellation during an edit.
- Independent checks include constrained-face coverage, oriented chain closure, volume coverage without overlap, region volume totals, boundary distance, quality distribution and source immutability after failure. Reusing the generator's cavity flags alone is not an oracle.
- Add **proposed new** `examples/native_tetrahedral_meshing.py`: construct a cavity/interface domain natively, generate, certify, solve a manufactured diffusion problem and report independent error/evidence.
- Extend meshing qualification/benchmarks with separate preparation, boundary recovery, classification, refinement, improvement, publication and certification stages; compare like-for-like Gmsh/HXT, CGAL and fTetWild only on common admitted source/quality contracts.
- Gate: all mandatory PLC cases complete natively, output meets hard requests, fixed boundaries remain fixed, and no sliver/coverage failure is hidden by a successful Delaunay status.

## 8. W04 — Native implicit, image and multimaterial volume routes

**Prerequisites:** W01/W02 and W03 volume topology; use W06 for periodic domains. Native CAD is not required for native analytic/implicit/image inputs.

### Algorithm decisions

- Preserve existing fixed-grid discovery as an explicit bounded route, but add adaptive error-controlled discovery using existing spatial/octree infrastructure.
- A cell with no sampled sign change is not certified empty. Use source-provided interval/Lipschitz bounds, regularity and root-isolation evidence; split unresolved boxes within budget. Generic black-box/neural fields lacking valid bounds remain explicitly sampled/uncertified rather than receiving a global topology guarantee.
- Support a restricted-Delaunay domain-oracle route for implicit volumes in addition to PLC-based extraction/fill. Both reuse the same tetrahedral topology/quality machinery but retain distinct conformity evidence.
- Interpret labeled images explicitly: categorical samples, occupied voxel cells, or a declared reconstructed interface model. Do not switch interpretations to obtain smoother meshes. Use source physical affine/units and stable label ontology.
- Build one shared material-interface complex including triple/quadruple junctions, then recover it in the volume. Region classification follows that complex; centroid labels are only diagnostics.
- Include scalar-field isosurface/level-set discretization on existing meshes and native Lagrangian-motion/remesh workflows, closing the Mmg-type workflows rather than only its metric mode.

### Compartment API and consumer cutover

- Add `CompartmentMeshingSource` at the existing `phydrax/geometry/_compartments.py` owner: bind the label volume, `CompartmentComplex`, outer surface and extracted interface source to one revision/coordinate contract. It contains source facts, not target sizes or provider options.
- Remove `CompartmentMeshingSpec.options: FTetWildOptions` and the obsolete specification class after callers migrate. Express target size, physical envelope/fidelity, required regions/interfaces, quality and limits in `VolumeMeshingSpec`/canonical controls; native iteration/thread scheduling belongs to native options/execution policy. Do not reinterpret fTetWild's AMIPS `stop_quality` as a numerically interchangeable native quality bound.
- Add `RegionMeshingEvidence` to the existing organization/result owners. It binds authoritative source-region IDs and source-complex revision to target cell global IDs, exclusive cell-to-region assignments, mesh-zone IDs, oriented interface facet/patch IDs, adjacency pairs and coverage certificates. Display names are not scientific identity.
- `CellMeshingResult.region_evidence` is required for a compartment/material generation request and is part of result identity. Existing `zones`/`patches` remain the canonical mesh organization; the evidence binds them to source compartments rather than creating another zone store.
- Region evidence is a transition obligation, not generation-only metadata. Result-to-result adaptation, motion, relocation/optimization, curving publication and proposal acceptance must remap assignments/interfaces through exact lineage or an explicitly certified geometric reclassification, rebuild target zone/facet bindings and adjacency, and refresh every geometry-bound coverage certificate. Fixed-topology motion may preserve entity assignments but cannot blindly copy old geometry certificates. Unsupported or ambiguous propagation rejects the whole transition; no material result is silently republished as an unlabeled result.
- Remove the `CompartmentMeshingResult` wrapper. `NeurofluidCase.bulk_mesh` becomes `CellMeshingResult`; its admission validates `region_evidence` against the exact `CompartmentComplex` and segmentation revision, and transport uses `bulk_mesh.mesh`/`coordinate_contract` directly. Cell compartment IDs and adjacency are read from the validated region evidence, not reconstructed from zone names.


### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/geometry/implicit/_discovery.py`, `_curve_discovery.py`, `_policy.py`, `_realization.py`, `_projection.py` | Reuse batched roots, QEF and fixed-route realization; separate adaptive discovery from differentiable realization. Add complete query/work budgets, feature/intersection-curve handling and periodic boundary integration. |
| Proposed new `phydrax/geometry/implicit/_adaptive_discovery.py` | Adaptive certified box worklists, root/feature isolation and conforming surface extraction; compose the existing spatial hierarchy rather than implement a second octree. |
| Existing `phydrax/discretization/spatial/_level_octree.py`, `_block_sparse.py`, `_voxel.py`; `phydrax/geometry/_certified_implicit.py` | Prepare bounded balanced topology and bind enclosure/topology evidence. Unknown/budget-exhausted boxes remain explicit. |
| Proposed new `phydrax/meshing/_implicit_volume.py` | Restricted-domain query/refinement schedule with feature protection and material classification, using W03 mutable tetrahedral execution. |
| Existing `phydrax/geometry/multiregion_surface/_label_extraction.py`, `_topology.py`, `_validation.py`, `_remesh.py`, `_transfers.py` | Export authoritative interface/junction constraints with region-pair identity and extraction approximation evidence. Reuse existing junction-aware surface edits, not a manifold-only replacement. |
| Existing `phydrax/meshing/_compartments.py`, `phydrax/geometry/_compartments.py`, meshing `_organization.py`, `_result.py` | Implement the compartment source/request/result cutover above, native domain preparation and conforming fill. Preserve cell assignments, adjacency and interfaces in `RegionMeshingEvidence`; remove engine-coupled specification, provider and result wrapper after reference migration. |
| Proposed new `phydrax/meshing/_level_set.py` | Native zero-level interface insertion on an existing simplex mesh, explicit interior/exterior region splitting and field transfer through W08. Use native local root and protected cavity machinery. |
| Existing `phydrax/meshing/_motion.py` | Route 3D native remesh escalation to W08, preserve physical boundary/material motion and geometric conservation obligations; no external metric engine is needed for the native path. |
| Existing `phydrax/meshing/providers/_native.py`, `_contracts.py` | Admit image/implicit/compartment/level-set sources with their exact capabilities and explicit accuracy class; reject incompatible global-certification requests at preparation. |
| Existing `phydrax/applications/neurofluid/_model.py`, meshing/geometry facades, `examples/neurofluid_transport.py`, `tools/neurofluid_qualification.py`, `tools/neurofluid_benchmarks.py` | Migrate `NeurofluidCase` and `NeurofluidTransportPlan.prepare` off `.bulk_mesh.result` and wrapper-type checks; validate exact source-region evidence and preserve mixed-dimensional transport/physical units. Update all exports, constructor calls and qualification records in the same cutover. |

### Tests, smoke, benchmark and gate

- Extend `tests/unit/geometry/test_implicit_surface.py`, `test_implicit_projection.py`, multiregion extraction/remesh tests and `tests/unit/meshing/test_compartment_meshing.py`, `test_mesh_motion_monitor.py`.
- Extend `tests/unit/applications/test_neurofluid.py` and execute the existing neurofluid transport/qualification smoke with a genuinely native generated mesh. Assert exact compartment identity, interface adjacency, bulk/network transfer and flux/inventory balance; losing the wrapper cannot lose its scientific data.
- Add **proposed new** `tests/unit/meshing/test_implicit_volume.py`, `test_image_volume.py`, `test_native_level_set.py`.
- Edge cases: closed feature entirely inside one coarse grid cell, tangential zero without sign change, disconnected tiny components, near-zero gradient, exact-zero grid samples, finite image extent/background, anisotropic voxels, oblique physical affine, tiny but required labeled regions, material junctions and moving zero sets that change topology.
- Smoke analytic solid → implicit volume mesh → 3D anisotropic adaptation → solve; labeled image → shared interfaces → conforming tetrahedra → multicomponent transport with interface flux balance; moving level set → explicit accepted topology event and conservative transfer.
- Benchmark source evaluation/root isolation, adaptive spatial preparation, surface/interface complexity, volume construction and certification separately. Vary geometric frequency, interface area, voxel resolution, material count and unresolved-box fraction.
- Gate: bounded native volume workflows cover these sources without missing claimed features, dropping small materials or relabeling intersected cells by a single sample.

## 9. W05 — Imperfect surfaces, wrapping, native reconstruction and surface Booleans

**Prerequisites:** W01 intersections, W02 source semantics and W03 volume fill. W07 shares robust arrangement/Boolean invariants; surface and CAD geometry retain different representation owners.

### Algorithm decisions

- Separate exact represented-surface conformity from tolerance-envelope repair. A defective triangle soup does not uniquely determine an intended solid; require an explicit inside/outside/repair policy and expose topology changes.
- Use arrangement-based triangle insertion and locally valid tetrahedral refinement/improvement for the envelope volume route. Missing features or uninserted input facets remain failures/unmet evidence; a valid background tetrahedral mesh is not sufficient success.
- Implement native wrapping through bounded offset/enclosure sampling and feature-aware surface extraction, with envelope inclusion and topology-change reporting.
- Native surface Booleans split exact classified intersection arrangements, classify oriented fragments under declared region semantics, and produce ancestry/property transfer. No proximity-based patch identity reconstruction.
- Replace reconstruction engines rather than rename adapters. Oriented-point screened Poisson uses native spatial preparation, sparse assembly, linalg/preconditioning and native isosurface extraction. Unoriented/noisy point clouds need explicit normal-estimation/orientation and ambiguity evidence. Poisson smoothing does not claim feature-preserving exact reconstruction.
- Preserve existing planar/terrain reconstruction behavior through native Delaunay and declared filtering/extrusion; no hidden Qhull dependency remains.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Proposed new `phydrax/geometry/surface/_arrangement.py`, `_boolean.py` | Bounded triangle arrangements, oriented fragment classification, union/intersection/difference, exact source-face ancestry and empty/disconnected output semantics. Use native intersections; do not move general CAD Booleans into triangle code. |
| Proposed new `phydrax/meshing/_surface_envelope.py` | Explicit repair/wrapping/envelope policy, two-sided deviation/inclusion evidence, feature permissions and native envelope tetrahedralization schedule. |
| Existing `phydrax/geometry/surface/_contracts.py`, `_model.py` | Distinguish raw defective surface input from certified `SurfaceModel`; never weaken the authoritative surface invariant just to admit a soup. Publish validated repaired geometry with a new revision and explicit changes. |
| Existing `phydrax/geometry/reconstruction/_core.py` | Replace SciPy/Qhull planar triangulation with `geometry.DelaunayTriangulation`; replace PyVista execution with native reconstruction owners. Preserve source-product identity, units, recentering declarations and filtering evidence. |
| Proposed new `phydrax/geometry/reconstruction/_poisson.py`, `_normals.py` | Native bounded neighborhood/normal orientation and screened-Poisson reconstruction. Reuse spatial/BVH, sparse, linalg and discretization operators; do not add private linear solvers. |
| Existing spatial voxel/block-sparse owners and implicit extraction files | Native sparse-field isosurface extraction and explicit background/bounds/re-distance semantics, replacing the native workflow's OpenVDB requirement. Reuse existing field/differential operations rather than add another level-set runtime. |
| Existing `phydrax/meshing/providers/_manifold.py`, `_openvdb.py`, `_poisson.py`, `_ftetwild.py` | Keep explicit optional comparisons, migrate generic source/specification types out of provider modules to their geometry owners, and remove provider imports from native workflows. No compatibility re-export of moved owner types. |
| Existing `phydrax/meshing/providers/_native.py`, `geometry/reconstruction/__init__.py`, surface/meshing facades | Expose the canonical reconstruction/Boolean operations at geometry owners and consume their prepared sources through native meshing. Avoid duplicate convenience APIs. |

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/geometry/surface/test_native_boolean.py`, `test_surface_arrangement.py`, `tests/unit/geometry/test_native_reconstruction.py`, and `tests/unit/meshing/test_surface_envelope.py`.
- Retain optional provider tests as comparisons; migrate neutral property/Boolean/source tests away from provider-owned types. Positive cases include coincident faces, tangent contact, coplanar overlaps, nested/disconnected components, open sheets under a declared wrapping policy and a legitimately empty Boolean result. An empty geometry result is not forced into a nonempty `CellMeshingResult`.
- Reconstruction cases include inconsistent/missing normals, density variation, noise, outliers, thin features, incomplete sampling, nonuniform units, sparse field boundaries and a declared unresolvable component. Test error/status evidence as well as output geometry.
- Smoke defective surface → explicit repair → native tetrahedra → accepted solve; points → native reconstruction → volume mesh; native surface Boolean → source-face property transfer → remesh.
- Benchmark tolerance/feature complexity, point density/noise, spatial capacity and insertion/retry work; compare against fTetWild/Manifold/Open3D/OpenVDB/PyVista on matching semantics, not merely output counts.
- Gate: original-versus-repaired meaning remains visible, requested envelope/feature semantics are met, and no upstream geometry/reconstruction engine is loaded by the native path.

## 10. W06 — Periodic topology as a first-class construction constraint

**Prerequisites:** W01/W02 shared topology and source identity. Design before W03/W08 mature; integrate into each route before claiming that combination.

### Algorithm decisions

- Reuse `PeriodicCell` for translational lattices. Distinguish a periodic triangulation on a quotient domain from a mesh with paired boundary nodes.
- Represent quotient entities by stable representatives and relative image/lattice shifts with canonical incidence/orientation. Temporary ghost storage is execution metadata; persistent relative shifts and winding classes are scientific topology and must survive publication.
- Use bounded periodic image neighborhoods whose sufficiency is certified for the requested geometric/metric work. A fixed arbitrary tiling is not a completeness proof; image-budget failure is explicit.
- Generate/refine/coarsen feature orbits together. A seam edit must preserve all equivalent boundary and interface copies, including CAD classes, material labels, layers and high-order nodes.
- Treat general boundary isometries separately from translational torus topology. Validate composed corner/edge transformations and orientation; unsupported/nonproper identifications are not accepted by fitting nearest nodes.
- Extend quality, global-embedding, coverage and transfer checks to the quotient complex. Apparent cuts of one displayed fundamental cell are not automatically holes in the periodic domain.

### Canonical periodic publication

Use a finite lifted `CellMesh` plus a persistent typed `PeriodicMeshTopology` descriptor in the existing discretization carrier. Lifted vertices have distinct row/global IDs even when they refer to the same quotient representative, so ordinary cell distinct-vertex validation remains meaningful. Publish one representative of each top-cell orbit with a continuous local geometric lift; keep the relative shifts of every corner/edge/face incidence and the orientation of every quotient identification.

The descriptor stores the bound `PeriodicCell`/isometry identity, entity-orbit representatives by dimension, lifted-to-quotient maps, relative image shifts and incidence/orientation witnesses. An edge key is not merely its endpoint representative IDs: normalize its relative shifts by a common image translation and retain winding. Multiple quotient edges/faces with the same representative vertices remain distinct. The descriptor is part of topology identity, not an optional coupling annotation.

Canonicalization, geometry evaluation, integration, compatible DOF numbering and transfers must preserve the descriptor. Evaluate each cell in its local lift and integrate each top-cell orbit exactly once; lower quotient identifications into existing oriented incidence/gather/constraint routes. Ordinary boundary-paired meshes remain a distinct admitted case and are not automatically promoted to a torus. Native persistence is lossless; external codecs unable to carry quotient data require explicit loss reporting rather than dropping it.


### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/discretization/_periodic_cell.py` | Retain canonical lattice/condition/image-bound owner; expose any required certified neighborhood preparation without duplicating wrap/minimum-image semantics. |
| Proposed new `phydrax/discretization/_periodic_topology.py`; existing `_cell_mesh.py`, `_cell_complex.py`, `_topology.py`, `_cell_geometry.py` | Implement and validate `PeriodicMeshTopology` on the canonical lifted carrier, including repeated quotient representatives, relative-shift entity keys, orbit incidence and local geometric lifts. Keep scientific IDs separate from image execution slots. |
| Existing `phydrax/meshing/_canonical.py`, `_result.py`, `_lineage.py`, `_interop.py`; discretization FE/cochain/partition/transfer and lifecycle owners | Preserve quotient keys through ordering, audit, geometry/DOF lowering, compatible transfer, ownership and reload. `_entity_vertex_keys` cannot collapse distinct winding entities to the same sorted representative set. Bind quotient certificates into accepted results and restart artifacts. |
| Existing `phydrax/meshing/_controls.py`, `_coupling.py` | Extend `PeriodicConstraint`/`PeriodicCoupling` to bind explicit entity orbits and transformation consistency; keep vector/tensor trace transformations correct. |
| Proposed new `phydrax/meshing/_periodic.py` | Compile source strata into periodic construction orbits, canonical representative IDs, synchronized edit constraints and quotient acceptance evidence. |
| Proposed new `native/meshcore/src/periodic.hpp`, `periodic.cpp` | Periodic Delaunay/regular cavity locality, image bookkeeping, quotient constraints and bounded image expansion using W01 topology. |
| Existing `phydrax/geometry/_triangulation.py` | Add explicit periodic diagram/triangulation support through the same canonical periodic contract, not a separate box-tiling convenience convention. |
| Existing implicit/multiregion source owners and native provider | Admit compatible periodic sources, check seam values/region incidence/features, and remove only the now-obsolete blanket native periodic refusals. |
| Existing adaptation, curving, boundary-layer and distribution owners | Make orbit-preserving operations an executable path, not permanent protection that refuses all refinement near periodic boundaries. Extend lineage and ownership to orbit representatives. |

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/meshing/test_native_periodic.py` and native `test_periodic.cpp`; extend `test_coupling.py`, `test_cad_curving.py`, device/distribution and diagram tests.
- Cases: fully/partially periodic cells, skew lattices, features crossing multiple cell boundaries, triply periodic implicit surfaces, multimaterial junctions on seams, rotational wedge pairings, conflicting transformation cycles, periodic layers, refinement/coarsening across partitions, and exhausted image bounds.
- Mandatory positive publication case: a minimal periodic triangulation/tetrahedralization with repeated quotient vertex representatives and distinct winding edges sharing representative endpoints. Publish and reload the accepted carrier, refine/coarsen it, check quotient incidence composition and measure, then solve a periodic PDE/compatible-field case. Verifying only private construction images does not close this gate.
- Smoke periodic geometry → native mesh → orbit-preserving adaptation → vector/flux transfer → periodic manufactured PDE; verify quotient topology and conservation rather than only nearest-node distances.
- Benchmark image count, lattice conditioning, boundary-orbit density and distributed seam traffic separately.
- Gate: required periodic combinations actually generate/adapt conformingly; post-hoc node matching is not accepted as periodic construction.

## 11. W07 — Native CAD representation, queries, Booleans and interchange

**Prerequisites:** W00 identity/accuracy contracts and W01 geometric predicates. Begin representation and interchange work in wave A; CAD-specific meshing acceptance waits for native queries/intersections, not an OCCT conversion.

### Representation and algorithm decisions

1. Make exact represented curves, surfaces, oriented coedges, wires, trimmed faces, shells, solids and assembly occurrences authoritative in `BRepModel`. Derived tessellation gets its own identity and error evidence; changing tessellation resolution must not change physical CAD identity.
2. Extend the existing analytic/rational patch evaluator, not a second NURBS implementation. Curves/surfaces carry legal knot multiplicities, parameter domains, periodic seam representatives, denominator bounds, orientation and singularity metadata. Use the canonical interpolation jets for values/derivatives/transposes.
3. Native query preparation uses conservative curve/patch bounds and BVH candidate search. Closest-point and point-location decisions include trim membership, possible multiple minima, seams/poles, rank and residual bounds. No claim of globally exact signed distance follows from a successful local projection.
4. Curve–curve, curve–surface and surface–surface intersections use bounded interval/Bernstein subdivision, native local numerical correction and certified existence/uniqueness or explicit singular/coincident components. A small residual alone does not prove complete discovery. Tangent, coincident and zero-derivative cases have explicit outcomes and mandatory positive fixtures.
5. Sewing and Booleans use those intersection events to fragment curves/faces, classify arrangement cells, assemble oriented shells and retain exhaustive source/target lineage. Tolerance changes and repair are explicit policies; neither sampled implicit CSG nor triangle tessellation is an undeclared CAD Boolean fallback.
6. Native CAD construction includes current analytic primitives, sketch line/conic loops, extrusion/revolution and supported exact operation trees. Implicit blends/neural fields remain distinct approximate representations unless a caller explicitly requests conversion.
7. Interchange is bounded and format-aware. STEP/IGES coverage is an explicit entity/representation matrix, including the currently used geometry, topology, units and occurrence workflows. Unsupported referenced geometry fails with its dependency chain; it is not silently linearized.

### Representation of general intersection curves and trims

Analytic/rational curve families are not closed under surface intersection. A generic smooth intersection of two quadrics can be a genus-one algebraic curve and cannot be represented exactly by a finite rational B-spline. Native Boolean output must not label an approximate fitted spline as an exact intersection.

Add a geometry-owned `IntersectionCurve` representation whose authoritative definition is its generating surfaces, branch identity and certified continuation atlas. Store supporting surface definitions/revisions, coupled parameter boxes on both surfaces, oriented chart transitions, endpoint/loop connectivity and existence/regularity/completeness witnesses. Native point/jet and both p-curve evaluations come from the same chart solution with explicit numerical bounds; the represented implicit intersection and its bounded numerical evaluation are distinct claims.

Extend `TrimDomain`/`BoundaryAtlas` to consume oriented native curve loops, including these intersection branches, instead of assuming every authoritative trim is a polygon. Polygon loops remain a valid explicit affine representation/acceleration with their own approximation evidence. Classification, sewing, tessellation, projection and fixed-epoch derivatives use the coupled 3D/p-curve definition and cannot fit the two sides independently.

Native persistence retains the generating surfaces and branch atlas losslessly; it must not require live source handles or missing ancestor artifacts. STEP/IGES/external BRep export writes an exact entity only when that format/profile can represent it. Otherwise it requires an explicit bounded approximation policy with coupled 3D/p-curve consistency, topology-preserving tube/trim evidence and a declared new approximate representation, or refuses exact export. Native exact-model round-trip and externally bounded-approximation round-trip are separate positive gates.


### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/geometry/brep/_model.py` | Extend `BRepTopology`/`BRepModel` with authoritative curves, oriented coedges/wires, exact p-curves/trims, shell/solid incidence and occurrence paths. Keep `BRepEntityId`, source revisions and explicit physical coordinate contracts. Separate exact model identity from derived tessellation identity. |
| Existing `phydrax/geometry/brep/_patches.py` | Extend native curve/patch families, parameter-domain validation, jets, homogeneous bounds and seam/pole metadata. Reuse analytic Plane/Cylinder/Cone/Sphere/Torus and `BSplineCurve`/`BSplineSurfacePatch`; add line/conic/Bezier, ruled, extrusion/revolution and required offset representations with honest bounded evaluation. |
| Existing `phydrax/_interpolation/_bspline.py`, `_bspline_grid.py`, `_tensor_bspline.py`, `_rational_spline.py` | Extend canonical span, denominator and interval/Bernstein preparation only where CAD needs missing capability. Preserve knot convention, reduction order and adjoints for all existing interpolation/IGA consumers. |
| Proposed new `phydrax/geometry/brep/_constructors.py` | Explicit analytic/sketch/sweep-to-B-Rep construction with exact topology, orientation, parameter binding and source association. No implicit conversion of arbitrary level sets to CAD. |
| Proposed new `phydrax/geometry/brep/_query.py` | Prepared native point location, trim-aware closest point, bounds, closure membership and bounded geometric integration. Reuse BVH/predicates and owning numerical solvers. |
| Proposed new `phydrax/geometry/brep/_intersection.py` | Typed bounded curve/curve, curve/surface and surface/surface intersection policies/events/results; retain parameter enclosures, component completeness, tangency/coincidence/singularity witnesses and unresolved work. |
| Proposed new `phydrax/geometry/brep/_intersection_curve.py`; existing `_patches.py`, `_model.py`, `phydrax/geometry/_atlas.py` | Add authoritative `IntersectionCurve`/coupled p-curve chart representation and oriented general trim loops. Retain analytic/rational fast paths; migrate `TrimDomain`/`BoundaryAtlas.reference_mask` and CAD curve consumers to capability-based curve evaluation with certified bounds. |
| Proposed new `phydrax/geometry/brep/_sewing.py`, `_boolean.py` | Native topology assembly and Boolean/partition arrangements, explicit repair, region precedence/void semantics and exhaustive revision-qualified correspondence. Reuse surface-arrangement numerical predicates without conflating CAD and triangle representations. |
| Proposed new `phydrax/geometry/brep/_tessellation.py` | Derived adaptive trimmed-patch tessellation using W02 curve/surface generation. Preserve vertex/edge/face/UV lineage and continuous error bounds. This is not a second independently scheduled surface mesher. |
| Existing `phydrax/geometry/brep/_source.py`, `_projection.py`, `_differentiable.py` | Replace OCCT and triangle-query authority with the native query/patch owners. Preserve projection/classification/seam/ambiguity contracts and fixed-epoch differentiation. Explicitly bounded derived acceleration cannot change geometric truth. |
| Existing `phydrax/geometry/brep/_partition.py`, `_planar.py`; `phydrax/geometry/_cad_revision.py` | Reimplement current `BRepPartitionPlan`/`PlanarPartitionPlan` execution with native arrangements, sewing, persistence and `AssociationGraph`. Preserve explicit region precedence, void roles, atomic publication and exhaustive lineage. |
| Existing `phydrax/geometry/design/_schema.py`, `_constraints.py`, `_sketch.py` | Bind CAD parameters through existing `ParameterId`/`DesignState`; exact sketch lowering and native seam/clearance constraints. No new CAD-specific parameter/state system. |
| Existing `phydrax/geometry/process/_lower.py`, `phydrax/meshing/_layer_extrusion.py`, `_planar_bands.py` | Replace native-workflow OCP construction, Boolean, measure and persistence calls with native construction/query/partition. Preserve process-layer and wall/cap/volume identity. |
| Proposed new `phydrax/interchange/_cad.py` | CAD-specific import/export policies and coverage/refusal reports using existing bounded-resource, publication and adapter-report utilities; byte/entity/reference/depth/query budgets and explicit unit/placement handling. |
| Proposed new `phydrax/interchange/_step.py`, `_iges.py` | Native bounded readers/writers. STEP Part 21 reference graph, units/placements, analytic/rational geometry, p-curves/trims, oriented topology, manifold solids/voids and assembly occurrences; IGES analytic/spline/trim/topology equivalents with explicit ambiguous-graph refusal. Preserve external entity identifiers only as source provenance. |
| Proposed new `phydrax/interchange/_cad_brep_text.py` | Native reader/writer for the externally defined OCCT BRep text profiles exercised by existing `.brep`/`.brp` workflows, with explicit representation/release coverage. This implements a file format without using OCCT algorithms. Preserve the external format identity rather than repurpose its extensions. |
| Proposed new `phydrax/interchange/_cad_archive.py` | Native B-Rep persistence through the existing lifecycle container: exact geometry/topology/trims/units/lineage, bounded decode, deterministic payloads and atomic publication. One canonical representation; no internal schema-generation fields. |
| Native query/intersection/sewing/tessellation files and CAD interchange files above | Carry intersection-curve support definitions/branch witnesses through all consumers; exact native persistence, explicitly bounded external fitting where requested, and no false exact-NURBS claim. |
| Existing `phydrax/interchange/__init__.py`, `_catalog.py` | Canonical native CAD entry points and accurate format/entity profiles. External format revisions remain at the interchange boundary; generated capability data must distinguish exact represented geometry from approximations. |
| Existing `phydrax/geometry/surface/_interop.py`, `_model.py`, `_high_order.py` | Move CAD file decoding to interchange. Keep discrete surface import/export loss-aware; expose an explicit B-Rep-to-surface conversion using derived tessellation and native charts. Never export a triangle mesh as exact CAD. |
| Existing `phydrax/discretization/iga/_geometry.py`, `_basis.py`, `_actions.py`, `_manifold.py`, `_cut.py`, `_interfaces.py` | Reuse shared rational evaluation and chart/query evidence. Keep IGA field/span topology and trim/cut qualifications with their current owners; native CAD support does not silently broaden an IGA space's mathematical contract. |
| Existing `phydrax/geometry/brep/_occt.py`; proposed new `phydrax/interchange/_cad_occt.py` | Remove OCCT implementation from native dispatch. Retain only an explicit lazy interoperability/comparison adapter if needed; migrate intentional external-shape callers to it and remove obsolete native exports/aliases. Native decoding/query/meshing never invokes it. |
| Existing `phydrax/meshing/providers/_gmsh_import.py`, `_gmsh_layers.py`, `_gmsh_evidence.py` and CAD-dependent provider code | Make external comparison consume a verified native STEP/BRep export or explicit bridge; preserve entity correspondence/loss evidence. Remove assumptions that a native B-Rep requires an OCP object or an OCCT-generated query mesh. |
| Existing `pyproject.toml`, `uv.lock`, geometry/interchange facades and typing configuration | Remove mandatory OCP from core install and native imports once all native callers are cut over; retain external OCP/build123d/stubs only in clearly named optional comparison dependencies. Do not change unrelated package dependencies. |

### Required interchange coverage and migration gate

- Freeze an entity-coverage ledger against current CAD fixtures and all source types already admitted through OCCT conversion before the cutover. Final native acceptance must cover those fixtures or identify a user-approved scope change; “unsupported entity” alone cannot erase an existing supported workflow.
- Include manifold boundary representations, trimmed rational surfaces, holes, disconnected solids, assembly placements/occurrences, explicit units, p-curves, periodic seams, analytic and exact swept/revolved forms, and the external BRep files consumed by current workflows.
- More general procedural/offset/nonmanifold representations need a checked native lowering or an explicit coverage gap. Put those gaps in the final completion ledger rather than treating the initial parser subset as full CAD independence.
- Independent format-conformance fixtures and external-reader comparisons are qualification tools only. Native decoding/writing must also execute in the engine-free environment.

### Tests, smoke, benchmark and gate

- Extend `tests/unit/geometry/test_brep_projection.py`, `test_cad_partition.py`, `test_planar_partition.py`, `test_process_stack.py`, native patch/design/IGA tests and meshing association/curving tests.
- Add **proposed new** `tests/unit/geometry/test_native_brep.py`, `test_brep_intersection.py`, `test_brep_sewing.py`, `tests/unit/interchange/test_cad_native.py` and format-specific STEP/IGES/BRep test modules under the existing interchange test owner.
- Replace default OCP/build123d fixture construction with native constructors or licensed deterministic fixtures. Preserve optional real-engine comparison cases under existing test architecture; do not let `importorskip` remove the native contract tests.
- Cases: multiple closest points, sphere center, pole/seam projection, tangent and coincident intersections, rational denominators, self-intersecting trims, oriented holes, tiny trim loops, contradictory units, cyclic/dangling references, over-limit files, malformed graphs, Boolean empty/disconnected results, partition precedence and atomic failure.
- Add a mandatory positive Boolean and native persistence round-trip for two regular quadric surfaces whose intersection is non-rational, with independent branch/loop and 3D/p-curve consistency checks. Exercise exact-export refusal and an explicitly requested bounded STEP/IGES approximation; this is not merely a singular-CAD negative case.
- Smoke native CAD construction → serialize/read → Boolean partition → native surface/volume mesh → association → high-order curve → solve. Repeat with external engines absent; vary derived tessellation without changing source identity.
- Benchmark parsing, exact model preparation, candidate search, intersections, Boolean classification, projection, tessellation and source-fidelity certification separately; vary trim/occurrence complexity and near-degenerate configurations.
- Gate: native CAD is the authority all the way through generation and design; no query/tessellation/Boolean/import path requires OCP, and supported source semantics are not weakened during migration.

## 12. W08 — Native surface/tetrahedral adaptation and complete geometry/state transfer

**Prerequisites:** W01 cavity invariants, W02 domain queries, W03 tetrahedral improvement, W06 periodic contracts; W07 for CAD-specific motion. Geometry-map transfer is designed with W00 and does not depend on the new high-order generator being finished.

### Native adaptation algorithms

- Keep conforming bisection/coarsening as its own reliable h-adaptation route. Add native 3D metric adaptation and embedded curved-surface adaptation rather than forcing either through planar `_local_metric.py`.
- Use metric edge splitting, legal directional collapse, tetrahedral/surface cavity reconnection, relocation and sliver improvement. Share substantial tetrahedral improvement kernels with W03.
- Enforce topological link conditions, region/feature classes, source projection legality, cavity orientation, collision/embedding evidence and periodic orbit consistency before commit.
- A metric request is complete only when its declared unit-mesh/quality/geometry criteria pass. Preserve COMPLETE/STALLED/PASS_LIMIT and rejected-operation evidence; valid-but-unmet output is not described as converged.
- Protected features can be refined along their geometry under an explicit invariant, not merely frozen forever. Illegal class-crossing collapse or interface removal fails locally with diagnostics.
- Implement explicit `NATIVE_MIXED` and `NATIVE_POLYHEDRAL` adaptation routes with the algorithms below, not a delegation to unspecified future generation code. Pure simplex routes remain unchanged; the final family/degree/operation matrix identifies exact-map restriction versus bounded reconstruction.

### Mixed-family and polyhedral adaptation algorithms

1. Layered mixed meshes refine wall cells tangentially and propagate the same subdivision through column ancestry. Prisms refine their triangular base and permitted axial intervals; hexes use compatible tensor subdivisions; tetrahedra use the existing simplex refinement; pyramids and prism/pyramid/tet transitions use an independently validated finite template set indexed by oriented shared-face subdivision.
2. Propagate face-refinement signatures across the entire local closure, including periodic orbits and neighboring blocks. If templates cannot close a cavity, use an explicitly permitted native local reconstruction constrained by the existing family/feature/geometry policy. Never silently replace a required prism/hex region with tetrahedra.
3. Coarsen only complete compatible template sibling patches with no external hanging interface, matching region/feature classes and accepted geometry/field error. Restore recorded parent identities when their exact identity relation holds; current source geometry and layer controls still require validation.
4. Preserve requested physical layer intervals and report child ancestry. A hard first-cell-thickness request forbids axial splitting of that first cell; tangential refinement is still admissible. An explicitly permitted schedule change creates a revised schedule and measured evidence rather than claiming the original first-cell height was retained.
5. Polyhedral refinement uses plane/site insertion or certified subdivision with shared-face closure. Coarsening uses connected same-region agglomeration or explicit site removal/regeneration. Validate closure, nonoverlap, admissible cell decomposition and consumer conditioning. Site ancestry does not imply nested cell ancestry: regenerated cells use certified common-refinement transfer rather than a fabricated P1 parent stencil.
6. Replace the private simplex-only edit payload with one canonical `CellTopologyEdit` at `_topology_edit.py`: typed target fixed-family blocks or existing packed polyhedral blocks, oriented shared-face witnesses, source/target family metadata, operation lineage and geometry witnesses. Migrate bisection/local/device callers in one cutover; no `SimplexTopologyEdit` compatibility alias remains.

Exact coordinate-map restriction is required for admitted simplex-to-simplex, tensor-to-tensor and prism-to-prism affine reference submaps. For pyramid or cross-family subdivisions whose rational map is not closed in a standard target element family, implement a composed/restricted rational coordinate element at the existing `_cell_geometry.py` owner, with denominator/collapsed-apex certification, or an explicitly selected bounded reconstruction. Do not substitute nodal interpolation and call it exact. General nonnested coarsening/regeneration uses the bounded reconstruction branch with coverage and field-transfer error.


### Geometry transition decision

Add a discretization-owned geometry transition with exact source-cell/reference-coordinate witnesses where available.

- Nested refinement: compose the source coordinate map with the child reference map; reuse shared entity nodes and prove mapped-domain coverage. This can preserve a polynomial/rational map exactly when the admitted element family is closed under that restriction.
- Relocation and nonnested remeshing: preserve authoritative source geometry by constrained evaluation/projection and bounded reconstruction. Do not claim exact map equality when a target cell spans multiple source maps.
- Coarsening: use an explicit approximation/projection policy, error bound and geometry/quality acceptance; refusal retains the fine mesh. Coarsening is not lossless merely because parent IDs are restored.
- Build target corner coordinates from the accepted geometry transition before constructing `CellMesh`, then pass the complete successor `CellGeometrySpec` to certification. A P1 vertex stencil is not a curved coordinate-map transfer.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/meshing/_adaptation.py` | Add explicit native 3D, surface metric, mixed and polyhedral routes, geometry-transfer admission, per-field obligations and final geometry-aware publication. Remove W00's temporary non-affine refusal only for implemented transfers. Retain explicit external routes and unknown-lineage semantics. |
| Existing `phydrax/meshing/_bisection.py`, `_local_metric.py`, `_topology_edit.py` and device edit consumers | Retain parent-cell/reference witnesses, geometry class and ancestry; replace `SimplexTopologyEdit` with canonical `CellTopologyEdit` supporting typed mixed/polyhedral blocks and oriented face closure. Preserve pure-simplex observable behavior and remove the obsolete private payload after all callers migrate. |
| Proposed new `phydrax/meshing/_tetra_metric.py`, `_surface_metric.py` | Family-specific preparation/schedules for native metric operations, using W01/W03 kernels and existing metric algebra, quality, association and commit owners. |
| Proposed new `phydrax/meshing/_mixed_adaptation.py`, `_polyhedral_adaptation.py`; proposed new `native/meshcore/src/mixed_refinement.cpp` | Own layer-column/template closure, compatible sibling coarsening, allowed mixed-cavity reconstruction and polyhedral splitting/agglomeration/site-regeneration schedules. Reuse W03/W10/W11 construction kernels, not a second generator; publish through the shared edit/geometry/transfer transaction. |
| Existing `phydrax/discretization/_cell_geometry.py`, `_cell_geometry_validity.py` | Add validated composed/restricted coordinate maps for exact mixed-family refinement where standard target basis closure is absent; carry rational denominator/apex bounds and distinguish bounded reconstruction. |
| Existing `phydrax/meshing/_device_adaptation.py`, `_device_metric.py`; proposed new `phydrax/meshing/_device_tetra_metric.py` | Preserve geometry witnesses and deterministic lineage through device epochs. Implement a separately qualified tetrahedral capacity/cavity route; shared device helpers own real status/workset invariants, not forwarding aliases. |
| Proposed new `phydrax/discretization/_cell_geometry_transfer.py` | `CellGeometryTransition` and its policy/evidence: source/target map identities, reference witnesses, node ownership, projection/approximation residuals, coverage and resource refusal. Reuse existing coordinate elements and interpolation. |
| Existing `phydrax/meshing/_lineage.py`, `_association.py`, `_motion.py` | Bind geometry transitions and exact/unknown lineage to the existing topology transition; propagate lawful feature classes and route native 3D motion/remesh. State transfer is consumed by solver transactions, not silently performed by mesh promotion. |
| Existing `phydrax/meshing/_organization.py`, `_canonical.py`, `_adaptation.py` (`_inherited_organization`/`_finalize_native`), `_motion.py`, `_optimization.py`, `_proposals.py`, `_result.py`, `_interop.py` | Implement one canonical region-evidence remap/revalidation operation and require source semantic obligations at every result-renewal boundary. Extend certification to accept and validate the target region evidence; standalone geometric certification must not claim inherited compartment semantics. Rebind cell/facet/zone IDs, recompute coverage after coordinate changes, preserve source-complex revision or require its explicit replacement, and retain evidence in native interchange/restart. Plain coordinate-optimization primitives may remain geometry-only, but their result-renewal callers must supply and validate the source obligations. |
| Existing `phydrax/discretization/_transfer.py`, `_state_transfer.py`, `_topology_epoch.py` | Extend `FieldTransfer`/`TransferProperties` and epoch admission with checked semantics and geometry binding. Keep primal, dual pullback and Hilbert adjoint distinct. |
| Existing `phydrax/discretization/fem/_topology_transfer.py`, `_generic.py` | Dispatch by actual field family/DOF functionals: nested H1, nonnested projection, DG and compatible edge/face transfers. Preserve orientation and coordinate-map evidence; never infer a field space from array width. |
| Existing `phydrax/discretization/fem/_nedelec_tetrahedron.py`, `_simplex_hdiv.py`, `_spectral_hp_completion.py` | Build H(curl)/H(div) transfer through covariant/contravariant Piola maps, edge/face moments and existing tensor de Rham transfer. Certify commuting defects for each supported order; extend missing supported-consumer cases at this owner instead of misusing P1 interpolation. |
| Existing `phydrax/geometry/_supermesh.py`, `_convex_intersections.py`, `_tetra_intersections.py` | Extend common refinement to mapped source/target geometry using certified subdivision/intersection bounds. Retain the affine exact path; no claim of exact curved overlap when only approximate quadrature is available. |
| Proposed `phydrax/geometry/_mesh_certificates.py` from W00 | Extend the early affine source-fidelity/global-embedding/coverage owner with curved-intersection and mapped-domain algorithms. Reuse native BVH/intersection and polynomial bounds; `_certification.py` only orchestrates them. |
| Existing `phydrax/discretization/finite_volume/_automatic_remap.py`, `_unstructured_remap.py` | Reuse conservative content and bounded second-order remap; bind curved coverage/error and explicit component semantics. No new meshing-local remapper. |
| Existing `phydrax/discretization/amr/_topology_transfer.py`, `_forest_transfer.py`, `_cut_transition.py`, `_cut_cochain_transfer.py` | Preserve nested-cell, compatible cochain, Hodge-adjoint and reflux semantics through new geometry epochs. Keep active/cut coverage and positivity evidence mandatory. |
| Existing `phydrax/discretization/iga/_transfer.py`, `_tspline.py`, `_certificate.py` | Retain exact knot/refinement coefficient transfer and qualified projection; bind native CAD geometry revisions and denominator/Jacobian certificates without flattening spline carriers. |

### Field/state transfer matrix

| Consumer state | Canonical operation | Required evidence |
| --- | --- | --- |
| Continuous H1 P1/Pk | Exact nested restriction/interpolation where proved; otherwise native L2 projection/common refinement | Reproduction order, continuity, conditioning, source coverage, curved quadrature error, adjoint distinction |
| DG and FV averages/content | Existing geometric overlap remap and field-family projection | Extensive-content conservation, source/target measure consistency, positivity/bounds only under a checked limiter/positive route |
| H(curl) edge/face moments | Oriented covariant-Piola compatible transfer | Circulation, curl commutation, periodic orientation and trace continuity |
| H(div) flux moments | Oriented contravariant-Piola compatible transfer | Normal-flux continuity, divergence commutation and component conservation |
| IGA/THB/T-spline coefficients | Existing exact refinement or qualified projection | Parameter/geometry identity, knot/space closure, rational denominator and quadrature/conditioning evidence |
| AMR/cut/cochain fields | Existing nested/refinement/reflux/compatible transfer | Coverage, commuting relations, positive metrics, conservation ledger and halo consistency |
| Materials, history and integrator state | Consumer-owned transfer/reinitialization policy inside the solver transaction | Inventory/positivity, irreversibility, history causality, algebraic reclosure and accepted-step rollback |

Conservation of a discrete remap using consistent overlap measures is distinct from an exact continuum integral over curved domains. State the geometric integration error and its contribution to the conservation claim. Positivity of nodal/modal coefficients is not universally meaningful; each field family owns that admissibility decision.

`FieldTransfer` remains a linear operator contract. Nonlinear limiters, positivity repairs requested by policy, constitutive-history updates and algebraic reinitialization stay in the existing state-transfer/consumer transaction owners. Their derivatives require an explicit fixed-active-set or algorithmic contract; they cannot be disguised as an `AbstractLinearOperator` with an unconditional transpose.

### Consumer migrations

| Existing file | Required cutover |
| --- | --- |
| `phydrax/solver/_finite_element_adaptivity.py` | Replace automatic vertex-P1 assumptions with explicit field-family transfer plans and geometry/material/history admission; stage the mesh, fields and prepared artifacts as one `CompositionRebind` and commit it atomically. |
| `phydrax/solver/_finite_element_schedule.py`, `_finite_element_checkpoint.py` | Bind accepted state and restart records to the same geometry/field transition and accepted-step journal. |
| `phydrax/solver/_finite_volume_topology_events.py`, `_finite_volume_runtime.py` | Require complete remap/coverage/positivity evidence before prepared runtime refresh and accepted state publication. |
| `phydrax/applications/phase_field/_adaptivity.py` | Preserve mass/energy obligations through the correct H1/DG transfer rather than raw `adaptation.transfer` assumptions. |
| `phydrax/applications/semiconductor/_transfer.py` | Use shared positive/conservative algebra while retaining material identity, carrier/trap inventory and electrostatic reclosure. Do not invent energy conservation absent from its scientific contract. |
| `phydrax/applications/cavity_quantum.py` | Replace the adaptive H(curl) refusal only when the new compatible edge-field transfer actually passes; preserve honest unsupported-order status elsewhere. |
| `phydrax/operators/integral/layer_potential/_adaptive_boundary.py` | Retain boundary face lineage and DP0 transpose; add curved source-fidelity/interface coverage when admitting curved boundaries. |
| `phydrax/applications/solid_mechanics/_topology_reanalysis.py` | Preserve density/material transfer, mandatory independent reference reanalysis, primal/adjoint and volume/bounds acceptance. |

### Tests, smoke, benchmark and gate

- Extend current bisection/local metric/device/lineage/curving/motion tests and their solver consumers. Add **proposed new** `test_tetra_metric.py`, `test_surface_metric.py`, `test_cell_geometry_transfer.py`, `test_compatible_mesh_transfer.py` under the owning meshing/discretization test directories.
- Add **proposed new** `tests/unit/meshing/test_mixed_adaptation.py`, `test_polyhedral_adaptation.py` and native `test_mixed_refinement.cpp`. Positive cases refine and coarsen actual curved prism/pyramid/tet and hex/transition interfaces, preserve permitted layer schedules and source coverage, and transfer H1 plus conservative DG/FV state. Include rational-map restriction/reconstruction and nonnested polyhedral regeneration; freezing every nonsimplex cell cannot pass.
- Retain regression cases for the original affine downgrade, curved refinement/coarsening, feature orbits, high-valence cavities, boundary class conflicts, incomplete ancestry, unknown external lineage, changed material regions and stale geometry.
- Mandatory compartment lifecycle gate: native compartment generation → refine/coarsen and legal fixed-topology motion/relocation → region-evidence revalidation → `NeurofluidCase` re-admission → mixed-dimensional transport. Check cell assignments, interface orientations/adjacency, source revision, current coverage and inventory balance. Also test rejected stale/copied evidence, a motion requiring an explicitly updated compartment source, and atomic rollback on ambiguous reclassification.
- Independent transfer cases: constants and known polynomial fields, divergence-free/curl-free compatible fields, manufactured flux/circulation, positive inventories, irreversible damage/history, curved overlap with known bounds, uncovered/double-covered regions and deliberately exhausted projection budgets.
- Smoke solve → mark/metric → native remesh → geometry transition → all field/material transfer → reprepare → solve; rejected transfer must retain every accepted state/solver epoch component.
- Benchmark local-edit batches, geometry queries, lineage/publication, common refinement, factor preparation, multi-field transfer and downstream solve separately. Compare prepared versus cold transfer and repeated epochs.
- Gate: native surface/tetrahedral remeshing closes geometry and solver state, not merely connectivity; every currently supported consuming field family has either its required implemented transfer or an explicitly uncompleted gate.

## 13. W09 — Complete native boundary layers and high-order hybrid geometry

**Prerequisites:** W02/W03 native surface/core generation, W06 periodic constraints, W07 native CAD queries, W08 geometry-transition interface.

### Decisions

- Reuse the existing advancing column/front implementation and physical schedules. Extend corner/rim/termination/merge treatment through explicit geometry and collision policies; do not discard achieved-thickness/layer-active evidence.
- Fill the core against immutable cap vertices/facets. Match oriented interfaces by authoritative identities, not tolerance welding. A closed cap is insufficient unless all remaining external and internal boundaries are included.
- Preserve regions, patches, protected features and periodic orbits through layers and core. Layer growth is not allowed to erase a narrow material region or overwrite an interface.
- Standard high-order geometry includes triangle, quad, tet, prism, pyramid and hex families. Implement family-correct Lagrange/rational maps and shared entity-node orientation; pyramids use their rational collapsed-coordinate contract.
- Lift the current degree-two/three construction restriction through an explicit supported-degree/resource matrix. Qualify representative degrees 2, 3, 4, 6 and 10 for the standard families; unsupported degree/resource requests fail before node allocation. Geometry degree remains independent from solution-field degree.
- Accept curved meshes only after local determinant, global embedding, source-fidelity and interface-coverage checks. If optimization converges but a certificate fails, retain the previous accepted geometry.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/meshing/_boundary_layer.py` | Native front/core handoff, source/region/periodic constraints, improved concave/narrow-gap treatment and bounded adaptive termination templates; preserve current evidence and measured physical schedule. |
| Existing `phydrax/meshing/_layer_extrusion.py`, `_planar_bands.py` | Native exact sweep/partition lowering via W07 and W10; no OCP/Gmsh dependency. |
| Proposed new `phydrax/meshing/_layer_core.py` | Compose existing layer output with W03 fill and canonical assembly; own exact cap/interface matching and combined domain coverage, not a second tetrahedralizer. |
| Existing `phydrax/meshing/_curving.py`, `_optimization.py`, `_association.py` | Mixed-family/high-degree entity-node construction, native source projection, periodic node constraints, geometry-aware relaxation and independent candidate acceptance. Distinguish intentionally fixed coordinates from CAD-constrained nodes that may move tangentially and be reprojected. |
| Existing `phydrax/discretization/_cell_geometry.py`, `_cell_geometry_validity.py`, `_cell_ordering.py`, `_reference_cell.py`, `phydrax/discretization/fem/_reference.py`, `_reference_operator.py` | Extend supported coordinate families/degrees and node ordering at the canonical owner. Do not copy provider-specific element ordering into the mesher. |
| Proposed `phydrax/geometry/_mesh_certificates.py` from W00, extended in W08; existing `phydrax/geometry/surface/_high_order.py` | Continuous curved boundary/interior intersection, rational denominator bounds and source/interface error certification with explicit budgets. |
| Existing `phydrax/meshing/_quality.py`, `_result.py`, `_trace.py` | Separate metric layer quality from isotropic corner metrics, report per-layer support/thickness/growth and curved-map quality, bind combined certificates. |
| Existing external Gmsh layer/element/execute modules | Remain explicit reference adapters; migrate general examples and native workflows off them without deleting independent comparison coverage. |

### Tests, smoke, benchmark and gate

- Extend `tests/unit/meshing/test_boundary_layer.py`, `test_cad_curving.py`, `test_planar_bands.py`, `tests/unit/discretization/test_cell_geometry_validity.py`, `test_fem_high_order_reference.py` and curved-transfer tests. Add **proposed new** `tests/unit/meshing/test_native_layer_core.py`, `test_high_order_hybrid.py`.
- Cases: convex ridges, reentrant corners, adjacent walls/rims, opposing fronts, asymmetric merging, local termination, vanishing layers, first-layer preservation, thin curved gaps, material interfaces, periodic layers and mixed prism/pyramid/tet/hex interfaces.
- Verify schedule error and growth on active columns, fixed cap identity, no gaps/duplicates, orientation of shared high-order faces, interior inversion despite positive corners, denominator failure and rollback on nonconverged/uncertified relaxation.
- Migrate `examples/boundary_layer_core_mesh.py` and `examples/cad_high_order_curving.py` to native geometry/generation; retain separate explicitly named comparison scenarios in qualification tooling.
- Smoke native wall geometry → layers → native core → high-order hybrid curve → geometry-preserving adaptation → diffusion/flow-compatible solve and accepted field transfer.
- Benchmark every stage and high-order coefficient/subdivision memory over wall count, layer count, thickness ratio, curvature and geometry degree.
- Gate: an end-to-end hybrid mesh needs no Gmsh/OCCT, all required controls coexist on the mandatory corpus, and curved quality is never certified from corners alone.

## 14. W10 — Quadrilateral, structured, swept, multiblock, hex-dominant and all-hex generation

**Prerequisites:** W02 source curves/surfaces, W07 maps/queries, W01/W00 topology and certification; W03/W09 supply explicitly allowed transition/core routes.

### Concrete algorithm portfolio

1. **Mapped/transfinite blocks:** compatible boundary curves/surfaces, exact logical edge counts and orientation maps, native transfinite interpolation plus elliptic/quality optimization where selected. Require positive mapped Jacobians and source fidelity.
2. **Sweep/extrusion/revolution:** carry a source surface through a declared map/frame schedule, with prism/hex layers, twist/collision checks and exact source/cap correspondence. Analytic extrusion as a field is not automatically a valid swept mesh.
3. **Multiblock:** explicit block adjacency, face permutations and common edge/face discretization; use `MeshAssembly` for independently coupled parts and one `CellMesh` for a genuinely conforming combined complex.
4. **General surface quads:** feature-aligned cross fields, singularity/valence planning, constrained patch decomposition and integer-grid extraction or validated recombination. Pure-quad output must close its topology; a triangle remainder is allowed only by the declared family policy.
5. **Hex-dominant:** feature-aligned block/octree/field-guided hex placement followed by certified prism/pyramid/tet/polyhedral transition closure. Report actual family counts and interface quality; no all-hex claim.
6. **General all-hex:** develop both (a) feature-aware balanced-grid subdivision/projection with conforming transition templates and (b) frame-field/block decomposition with integer-grid-map extraction. The first supplies a robustness-oriented route; the second targets alignment/element efficiency. Neither is assumed universally superior. Both require bounded closure, globally embedded positive cells, preserved topology/features and independent certification.

The general all-hex workstream is not fulfilled by shipping only boxes/extrusions or refusing every non-block input. Mandatory nontrivial cases include multiply connected domains, cavities, curved feature intersections, assemblies/material interfaces and nonsweepable mechanical parts. Fix the qualification corpus and quality/fidelity requirements before optimization. If boundary parity, singularity compatibility or a positive embedding cannot be established, report the exact obstruction; the relevant functionality/research gate remains open. Do not silently switch a required all-hex request to mixed cells.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Proposed new `phydrax/meshing/_structured.py`, `_sweep.py`, `_multiblock.py` | Family-specific prepared construction from current tensor grids, source maps and explicit interface topology. Consume existing carrier/assembly contracts; no new structured mesh representation. |
| Proposed new `phydrax/meshing/_quad_generation.py` | Native cross-field/patch/quad extraction and quality schedule using geometry queries and existing linalg/optim; preserve constrained curve chains. |
| Proposed new `phydrax/meshing/_hex_generation.py`, `_hex_dominant.py` | Native frame/grid/block construction and declared pure/mixed closure. Keep algorithm selection explicit and substantial route implementations decomposed by topology, placement and acceptance. |
| Proposed new `native/meshcore/src/quad_topology.cpp`, `hex_topology.cpp` | Bounded combinatorial template/patch extraction, orientation and face-matching kernels; validate template invariants independently. Numerical fields/optimization use their owning substrates. |
| Existing `phydrax/discretization/_tensor.py`, `_tensor_entities.py`, `_tensor_index.py`, `_hexahedral.py`, `_cell_mesh.py`, `_cell_complex.py` | Reuse logical layout and canonical family incidence/global IDs. Add only missing construction/validation operations, with packed authoritative connectivity and bounded execution padding. |
| Existing `phydrax/discretization/multiblock/_core.py`, `_interpolation.py`, `_constraint_extension.py` | Reuse explicit interface/coupling and interpolation semantics. Nonconforming blocks need the existing explicit transfer/mortar route, not proximity merging. |
| Existing `phydrax/discretization/spatial/_level_octree.py`; AMR `_canonical.py`, `_core.py`, `_variable.py`, `_entity_runtime.py`, `_entity_transfer.py` | Reuse balanced spatial topology, capacity buckets and compatible structured refinement/transfer. Hex generation consumes these owners rather than inventing another octree or AMR runtime. |
| Existing `phydrax/geometry/analytic/_sweeps.py`, `surface/_g1_multipatch.py`, B-Rep maps/query owners | Supply exact maps and seam topology; mesh blocks do not infer geometry identity from equal dimensions or coordinate coincidence. |
| Existing `phydrax/meshing/_assembly.py`, `_coupling.py`, `_controls.py`, `_contracts.py` | Admit explicit blocks/sweeps, required pure versus mixed families and exact gluing. Record unresolved decomposition/closure and quality conflicts. |
| Existing geometry validity, optimization, transfer and lifecycle owners | Certify mapped hex/quad interiors, preserve accepted source geometry and solver state, and archive block/singularity/transition evidence. |

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/meshing/test_structured_generation.py`, `test_sweep_generation.py`, `test_multiblock_generation.py`, `test_quad_generation.py`, `test_hex_generation.py`; extend native topology and tensor/AMR/multiblock tests.
- Cases: opposing edge-count mismatch, reversed face maps, periodic block cycles, twisted sweep, fold inside a trilinear cell, extraordinary surface vertices, cavity closure, incompatible boundary parity, trimmed curved blocks, material boundaries and mixed transition interfaces.
- Independent positive all-hex corpus must include nonsweepable domains; pure family and quality requirements are checked on actual output, not requested options. Deliberate template faults must fail independent topology/validity checks.
- Add **proposed new** `examples/native_multiblock_meshing.py`, `native_quad_hex_meshing.py`; smoke generated blocks through existing FEM/FV/AMR consumers with conservative/compatible transfers.
- Benchmark decomposition/field solve, integer/topology extraction, placement, optimization and certification separately; vary controlling block/feature/singularity count and requested cells, not only box resolution.
- Gate: structured/swept/multiblock functionality and general quad/hex research gates have separate results. Full scope is not declared complete while the mandatory general all-hex positive corpus remains unmeshed or uncertified.

## 15. W11 — Native domain-conforming Voronoi, power and polyhedral generation

**Prerequisites:** W01 exact clipping/regular triangulation, W02 domain strata, W03 certified domain decomposition where used, W06 periodicity and W07 for CAD fidelity.

### Algorithm decisions

- Extend the existing bounded diagrams rather than introducing another diagram API. Preserve site/weight identity, duplicate/redundant-site evidence, reciprocal face ownership and canonical packed cells.
- Convex domains use regular-triangulation neighborhoods and robust halfspace clipping. Nonconvex domains use a certified conforming decomposition and restricted-cell intersection; each connected component is explicit. A disconnected restricted power cell is not mislabeled as one ordinary connected cell.
- Merge pieces only when closure, orientation, embedding and consumer geometry requirements remain certified. A non-star-shaped cell cannot be accepted solely because the current star-based validity routine fails to disprove it.
- For genuinely conforming Voronoi quality/feature routes, add feature-protection surface sampling and interior site placement/refinement with bounded local neighborhoods. Clipping an arbitrary diagram is not by itself a VoroCrust-class boundary/quality algorithm.
- Use explicit site/weight optimization under native linalg/optim and retain convergence, empty-cell, feature/fidelity and quality evidence. An empty cell from a dominated weight is not automatically a failure unless the request requires every site to survive.
- Curved boundaries either produce a certified piecewise-planar approximation or use an explicitly supported curved-polyhedral representation; never attach affine volume claims to arbitrary curved faces.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/geometry/_triangulation.py` | Extend bounded diagram contracts with explicit domain decomposition/periodic preparation and face/site ancestry; preserve existing low-level convex diagram semantics. |
| Proposed new `phydrax/meshing/_polyhedral_generation.py` | Domain-conforming site placement/refinement, restricted-cell assembly, feature protection and canonical mesh publication. The geometry diagram owner still owns diagram construction. |
| Proposed new `native/meshcore/src/power_diagram.cpp` | Bounded regular-dual/halfspace execution, canonical face/vertex reconciliation, connected restricted-cell components and reusable moment evidence. Reuse existing clipping rather than duplicate it. |
| Existing `native/meshcore/src/clip2d.cpp`, `clip3d.cpp`, `clip_common.hpp`; `phydrax/geometry/_convex_intersections.py`, `_supermesh.py` | Extend robust domain-piece intersection and certify decomposition/coverage with explicit construction/measure error. Retain fail-closed capacity behavior. |
| Existing `phydrax/discretization/_cell_mesh.py`, `_cell_complex.py`, `_cell_geometry_validity.py` | Reuse face-defined packed polyhedra. Add verified decomposition witnesses for admitted non-star-shaped cells, without reinterpreting UNRESOLVED as valid. |
| Existing `phydrax/discretization/vem/_polyhedral.py`, `_space.py`, `_projection.py`, `_stabilization.py`; FV geometry owners | Admit only cells whose numerical space/integration/stabilization contracts are supported. Preserve rank/conditioning and FV nonorthogonality/skewness evidence. |
| Existing meshing `_quality.py`, `_metric.py`, `_optimization.py`, `_lineage.py`, `_distribution.py` | Polyhedral quality/site metrics, lawful optimization, domain/component lineage and partitioning; distinguish generator ancestry from a nodal field interpolation stencil. |
| Existing `phydrax/meshing/providers/_vorocrust.py`, native VoroCrust worker | Retain explicitly optional comparison only. Native generation cannot call the upstream sampler or extraction library. |

### Tests, smoke, benchmark and gate

- Extend geometry diagram/clipping/supermesh and polyhedral carrier/validity/VEM tests; add **proposed new** `tests/unit/meshing/test_native_polyhedral.py`, native `test_power_diagram.cpp`.
- Cases: weighted hidden/duplicate sites, coplanar/collinear sets beyond the current bounded all-pairs fallback, nonconvex domains, disconnected restricted cells, holes, sharp features, thin materials, periodic sites, coincident face fragments, non-star-shaped admissible cells and exhausted clipping budgets.
- Check reciprocal face loops, incidence signs, domain measure and first-moment closure, no missing/doubly covered pieces, material-interface conformity, feature fidelity and consumer conditioning. Distinguish mathematical site degeneracy from malformed topology.
- Add **proposed new** `examples/native_polyhedral_meshing.py`; smoke native nonconvex material domain → polyhedral mesh → VEM/FV solve → conservative remap.
- Benchmark neighbor construction, clipping, piece reconciliation, site optimization, moments, publication and downstream solve over site count, domain complexity, weight contrast and per-cell face count.
- Gate: arbitrary admitted domains are genuinely meshed, not only stored; positive consumer behavior and independent domain coverage accompany every polyhedral result.

## 16. W12 — Scalable native execution, distribution, partitioning and restart

**Prerequisites:** W01 execution-state invariants and W00 limits; integrate each mature family route from W03–W11. Design owner-local data and parallel-safe edits from the start rather than parallelizing a serial global rebuild afterward.

### Execution and distribution decisions

- CPU construction uses reusable native adjacency and bounded local queues, with conflict-free spatial/cavity batches. Keep the sequential reproducible route as a reference; add threaded execution only with an explicit conflict/ordering contract.
- Device execution uses existing fixed-capacity layouts and stable module-level compiled entry points. Independent candidate evaluations use bounded vectorization; recurrences use native JAX control flow. Candidate buffers and compiler temporary memory are part of the resource model.
- Native surface/tetrahedral generation can use device candidate/quality/geometry batches and capacity-bounded topology stages, but the runtime report must expose host-native exact decisions and CAD query work. A hybrid route is not advertised as an all-device algorithm.
- Exact resolution of uncertain device predicates occurs in bounded batches at an explicit transaction barrier on the owning process. Do not apply an unresolved edit and hope final orientation checks repair a lost constraint. No callback/device-to-host scalar synchronization occurs per candidate.
- Partitioned generation/adaptation maintains owner cells plus certified search/closure neighborhoods. Neighborhoods expand when a cavity, long edge, periodic image or geometric search extends beyond the current halo; a one-ring halo is not presumed sufficient.
- Reconcile shared constraints/cavities by stable semantic priorities, not first-arrival wins. Use neighbor exchanges for topology work, global scalar reductions for completion/resource consensus, and bounded distributed ordering when assigning canonical new IDs. No per-round all-gather of the global mesh/candidate set.
- Commit owner-local topology, geometry, lineage and transferred state only after collective acceptance. Failure of any mandatory local certificate/transfer rejects the global epoch. No mandatory host-global gather is hidden in commit or checkpoint.
- Native graph partitioning is a reusable graph operation, not a second mesh-specific graph algorithm. Use deterministic multilevel matching/coarsening, weighted initial allocation and bounded gain refinement. Report cut, imbalance, ghosts and migration; do not claim optimal cut.

### Owner-local accepted publication contract

Define publication before removing the current global commit. Extend the existing `CellMesh`/`CellMeshingResult`/`MeshPart`/`MeshDistribution` owners to accept an explicit owner-local storage layout through their validated construction boundaries; do not pass partial local arrays to a constructor that claims to hold complete global topology.

`CellMesh` remains the canonical carrier. Its storage descriptor distinguishes dense serial storage from globally logical, locally addressable topology/geometry arrays, binds global entity IDs/counts and local owned/ghost routing, and includes quotient metadata where applicable. Logical global incidences are not reinterpreted as local int32 slots; prepared execution views own that lowering. Existing dense constructors retain their serial semantics. A dense export/materialization is an explicit bounded operation, never an implicit prerequisite for an accepted distributed result.

Extend `CellMeshAuditReport`/`CellMeshingResult` with partition-indexed certificate/evidence coverage and collectively established global verdicts. Local positive Jacobians do not establish global validity: reconcile cross-owner facets/constraints, distributed intersection candidates, unique ownership and domain/interface coverage before global success. Per-entity findings and certificates remain sharded; reductions publish bounded summaries and a complete evidence binding.

Use existing logical-array/lifecycle identity machinery, extended at its canonical fingerprint owner where needed, to compute partition-independent content identity from canonically ordered logical chunks and verified distributed coverage. Do not hash rank order, local slots, only the request, or an arbitrary set of partial mesh hashes as if it were the global mesh. Serial and distributed publication use the same canonical identity recipe; any required recipe change is a clean cutover of mesh manifests/fixtures and explicit artifact migration/refusal, not parallel compatibility generations.

`MeshPart` fingerprints the accepted logical carrier identity, not `np.asarray` of global arrays. `MeshDistribution` consumes local ownership/halo routes and global coverage evidence instead of materializing all parts' ownership on every host. FE/FV preparation and topology transactions consume these owner-local canonical views; unsupported serial-only consumers refuse or require an explicit bounded export rather than secretly gathering.


### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing W01 native topology/cavity files; proposed new `native/meshcore/src/parallel_schedule.hpp` | Conflict-free local work batches, stable tie ordering, bounded native workspaces and thread-safe commit. The helper owns scheduling invariants, not another mesh representation. |
| Existing `phydrax/discretization/_adaptive_simplex.py`, `phydrax/meshing/_device_adaptation.py`, `_device_metric.py`; W08 device tetra owner | Distributed refinement and complete-family coarsening, geometry witnesses, owner-local exact-resolution queues and collective rollback. Replace global-candidate all-gather/host-global commit on the native distributed path; preserve current single-device behavior and capacity-bucket compilation. |
| Proposed new `phydrax/meshing/_device_generation.py` | Bounded surface/tetra generation candidate/topology worksets over native prepared domain data, explicit host/device barriers and per-phase status. Use current masked simplex/execution worksets rather than a new accelerator mesh carrier. |
| Existing `phydrax/meshing/_distribution.py`, `phydrax/discretization/_partition.py` | Extend ownership/halo preparation, constraint/cavity neighborhoods, deterministic migration and native graph dispatch; retain `MeshDistribution`, `CellPartition`, adjacency and sparse transfer as canonical. |
| Existing `phydrax/discretization/_cell_mesh.py`, `_cell_complex.py`, `_cell_geometry.py`, `_topology.py`; meshing `_canonical.py`, `_audit.py`, `_result.py`, `_assembly.py`, `_distribution.py` | Implement owner-local canonical construction/publication, global-versus-local identity, collective certificate coverage and explicit export boundaries described above. Remove mandatory complete-array NumPy conversion/fingerprinting on the distributed path, without weakening ordinary constructor validation. |
| Existing `phydrax/_fingerprint.py`, lifecycle logical-array/archive owners and data-plane distributed index owners | Establish one canonical logical-content digest and exact distributed coverage/uniqueness validation shared by serial and distributed publication. Update dependent mesh identity consumers/manifests in one cutover. |
| Existing FE `_generic.py`, `_hp_solver.py`, FV unstructured/runtime owners and solver topology transactions | Prepare and execute directly from the accepted owner-local `CellMesh`/result/distribution; bind local worksets and global spaces/certificates. Native distributed commit is not complete until a solver consumes the result without a host-global gather. |
| Proposed new `phydrax/graph/_partition.py`; existing `phydrax/graph/__init__.py` | Canonical weighted CSR partition plan/result/evidence, reusable by mesh and other graph consumers. Own validation, nonempty part/balance policy, deterministic matching/refinement and resource refusal. |
| Proposed new `native/meshcore/src/graph_partition.cpp`; existing ABI/binding/build files | Low-level compiled CSR partition backend packaged through the existing native extension. Public semantics stay with `phydrax.graph`; no duplicate algorithm in meshing. Add a separately named native backend source/library boundary only if packaging/ownership evidence requires it, not by default. |
| Existing `phydrax/meshing/_metis.py` | Remove METIS from canonical `MeshPartitionKind.GRAPH`. Retain only an explicit comparison provider if useful; migrate old provider-availability expectations to the comparison route. |
| Existing `phydrax/_execution_plan.py`, `_execution_resources.py`, `_execution_runtime.py`, `_execution_array.py`, `_execution_workset.py` | Lower native worksets to existing execution groups/sharding/resource requests; expose addressable state and collective failure barriers without another runtime. |
| Existing `phydrax/_data_plane/_distributed.py`, `_epoch.py`, `_ordering.py` | Reuse process-local/global-array construction, distributed index epochs and deterministic ordering for neighbor messages, IDs, migration and checkpoint shards. |
| Existing AMR `_cut_distributed.py`, `_cut_complex.py`, `_variable_distributed.py`; FV `_block_amr.py`, `_amr.py`, `_dyadic.py` | Preserve structured/cut coverage, local face reconciliation, FillPatch/reflux and compatible transfer across generated/adapted native epochs. No gather of every fine cell for ordinary commit. |
| Existing `phydrax/lifecycle/_models.py`, `_archive.py`, `_array_artifact.py`, `_repository.py`, `_transaction.py` | Register canonical mesh-transition/field-transfer records and logical arrays, content digests and atomic publication. Store complete scientific state, never compiled executable caches or live provider handles. |
| Existing `phydrax/lifecycle/_distributed_checkpoint.py`, `_restart_topology.py` | Addressable shard publication, exact coverage verification, topology relation and same/different-placement restart. Recover ownership/local slots from stable IDs, not former rank numbers. |
| Existing `phydrax/meshing/_interop.py`, `_lineage.py`, `_result.py` | Persist certified mesh/geometry plus hierarchy/lineage/constraint/transfer/epoch references through lifecycle. Mesh-array interchange alone is not a complete adaptive restart. |
| Existing `phydrax/discretization/finite_volume/_unstructured_archive.py`, solver FE checkpoint owner | Bind accepted PDE fields, geometry measures, remap evidence and material history to the same committed transition and checkpoint. |

### Checkpoint payload and deterministic identity

The canonical checkpoint must retain source/target geometry/topology/result IDs, authoritative entity IDs, source revision, coordinate contracts, active slot/mask state, bisection history and retired identities, constraint/region/periodic classes, metric/quality requests, pending declared work if resumable, allocator cursors, exact/filtered decision status, accepted transfer claims, ownership/halo descriptors and solver material/history state.

Execution placement and local slot assignment are reconstruction metadata. Same-placement replay may be bitwise where the declared route supports it; changed-placement replay must use the existing restart relation and certify the requested tolerance/identity invariants. Stable IDs across partitions do not imply identical floating-point optimization trajectories. A stricter partition-independent deterministic route must actually implement canonical decision/reduction order and be benchmarked separately.

### Tests, smoke, benchmark and gate

- Extend current device/distribution/AMR/data-plane/lifecycle tests. Add **proposed new** `tests/unit/graph/test_partition.py`, `tests/unit/meshing/test_device_generation.py`, `test_native_distributed.py`, `test_meshing_restart.py`, and native `test_graph_partition.cpp`.
- Cases: unequal partition weights, disconnected graphs, indivisible heavy cells, empty-part conflicts, long cavities crossing halos, cross-owner sibling coarsening, periodic seams, uncertain predicates, one-rank overflow/failure, migration while preserving IDs, repeated refine/coarsen capacity consumption and restart with a different placement.
- Add a positive owner-local publication → `MeshPart`/`MeshDistribution` → FE/FV preparation → checkpoint/reload → solve case with genuinely non-fully-addressable arrays. Verify the global logical identity and collective evidence, not only private kernel correctness. The full constructor/publication path must remain gather-free.
- Use runtime evidence/instrumentation of actual addressable transfers and peak memory to establish no mandatory global gather; do not add tests that grep for `device_get` or assert source spelling.
- Smoke generation → distributed adaptation/coarsening → field transfer → checkpoint → different-placement restart → solve with a global conservation/identity check. Execute real multi-device/process qualification where available; simulated topology is not distributed performance evidence.
- Benchmark strong/weak scaling, neighborhood traffic, exact-resolution volume, synchronization, per-rank peak/retained bytes, compilation reuse, migration and checkpoint/restart separately. Vary cavity/halo size and capacity slack as well as entity count.
- Gate: native distributed functionality includes initial generation, refine and coarsen, not just partitioned storage; it has no external meshing/partitioning engine and no mandatory global host materialization.

## 17. W13 — Native overset connectivity and moving assemblies

**Prerequisites:** W02 authoritative geometry, W08 field/epoch semantics, W12 ownership/local communication and canonical geometry-map/locator support.

### Algorithm decisions

1. Register existing certified `MeshPart`s in `MeshAssembly`; bind topology, coordinate revision, source boundaries, excluded/protected scopes and execution ownership. Do not infer walls or holes from names.
2. Use bounded BVH candidates and authoritative geometry/region queries for hole cutting and active/hole/fringe/receptor classification. Preserve protected wall entities and report ambiguous cut cells explicitly.
3. Locate donor cells with family-correct containment/inverse-map evidence, exclude invalid/blanked donors, and select deterministically under an explicit donor-resolution policy. Ambiguous overlapping valid donors need a declared priority, not accidental traversal order.
4. Build interpolation through the donor's actual field/geometry space. Affine simplex weights can be nonnegative and reproduce affine fields; high-order Lagrange weights are not generally nonnegative. A bounded/monotone route must declare its reconstruction/order tradeoff instead of claiming both arbitrary high order and positive weights.
5. Keep interpolative overset explicitly nonconservative. A conservative overset/remap request uses certified overlap/control-volume transfer through W08 and the existing common-refinement/remap owners; interpolation's transpose is not a conservation proof.
6. For motion, refit spatial structures, refresh blanking/donors and commit a new registration epoch only if geometry, coverage and state transfer pass. Fixed topology does not make geometry-bound donor evidence reusable after arbitrary motion.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Proposed new `phydrax/meshing/_overset.py` | Canonical prepared overset assembly policy/plan/result over existing `MeshAssembly` and `OversetCoupling`; own hole/donor/motion scheduling and status, not a new mesh carrier. |
| Existing `phydrax/meshing/_coupling.py`, `_assembly.py` | Extend explicit coupling evidence for donor cells, field family, hole/fringe masks, coverage/residual and optional conservative route. Preserve vector/isometry behavior and transpose semantics. |
| Existing `phydrax/discretization/_simplicial_locator.py`, `_field_query.py`, `_views.py`, FE `_point_interpolation.py`, cell geometry/reference owners | Build donor interpolation as `PreparedFieldQuery` routes over the donor field reconstruction, reusing their pointwise evidence and exact transpose; extend the owning reconstruction/inverse-map plans for curved/mixed/polyhedral donors. No flattening to TIOGA's vertex-linear contract. |
| Existing `phydrax/geometry/_bvh_overlap.py`, `phydrax/_bvh.py`, native domain query/certificate owners | Bounded candidate search, region classification and donor admissibility with exhaustive unresolved/capacity evidence. |
| Existing `phydrax/geometry/_frame_timeline.py`, meshing `_motion.py`, `_distribution.py`, lifecycle owners | Moving registration, local donor packet exchange, rollback and immutable restart descriptors; live sessions/worker registrations are not serialized. |
| Existing common-refinement and FV remap owners | Implement conservative overlap mode under its own coverage/field contract. Never relabel the existing interpolative `OversetCoupling` as conservative. |
| Existing `phydrax/meshing/providers/_tioga.py`, `native/providers/tioga/*` | Retain only explicit optional comparison/interoperability execution and migrate generic assembly examples to native overset. Missing TIOGA must not affect native execution. |

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/meshing/test_native_overset.py`; extend `test_coupling.py`, `test_assembly.py`, distribution/motion/restart tests. Keep `test_tioga_provider.py` as an optional independent comparison.
- Cases: protected walls, disconnected holes, missing/ambiguous donors, excluded regions, moving overlaps, sliver donors, curved inverse-map failure, periodic vector transforms, high-order signed weights, positive low-order interpolation and capacity refusal.
- Verify constant/affine reproduction where declared, transpose/adjoint duality and donor evidence. Separately verify conservative overlap transfer; include a case where interpolation preserves constants but not the integral.
- Add **proposed new** `examples/native_moving_overset.py`; smoke moving assembly → hole/donor refresh → accepted transfer → PDE update → restart.
- Benchmark broad phase, classification, inverse-map work, stencil assembly, motion refresh and inter-rank donor traffic; vary part count and overlap/receptor fraction.
- Gate: native hole cutting and moving connectivity are complete without TIOGA, field-order semantics are truthful and conservation is never inferred from partition of unity.

## 18. W14 — Solver-aware adaptation, differentiable design and trusted learned proposals

**Prerequisites:** W08 complete transition/transfer, W09 coordinate-order support, W12 lifecycle/distribution and route-specific certificates. Begin estimator/cost contracts early; complete end-to-end decisions only over qualified executable routes.

### Decision design

- Optimize time/memory to a declared physical-error or quantity-of-interest target, not element quality alone. Combine residual/DWR indicators, recovered Hessians, smoothness, geometry error, conditioning and measured preparation/solve costs.
- Keep h-refinement, p-order, geometry order, anisotropic metric and relocation as distinct decisions with different admissibility and transfer obligations. A cheap predicted action is not executable until native admission passes.
- Use existing `finite_element_hp_decision`/closure/transactions, IGA adaptive design epochs and meshing requests. Add a common decision-evidence record, not another hp/refinement engine.
- Geometry error can dominate field error: refine/curve geometry only when the source-fidelity contribution requires it, and certify the independent coordinate map. An increased field order must not imply increased geometry fidelity.
- Fixed-epoch derivatives include source-to-geometry realization, PDE solve and qualified transfer pullbacks. Retopology, Boolean branches, feature classification, donor changes and acceptance remain explicit stopped-gradient events. Record margins and invalidation triggers.
- Learned proposers may supply marking, size, metric, coordinate or candidate-order scores. Project and execute through the same deterministic constraints/certificates/transfers as analytic proposers. A learned model never emits trusted connectivity or proof of validity.

### File-by-file changes

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/meshing/_proposals.py` | Reuse revision-bound proposal types, `LearnedMeshProposer`, safety projection and `MeshProposalTransaction`; extend admission to geometry/field transfer and native 3D/family routes. Remove the old single triangle-refinement limitation only where the selected route actually meets the request. |
| Proposed new `phydrax/meshing/_decision.py` | Solver-neutral budget/cost/decision evidence and route-feasibility composition. Consume estimator products from their owners; do not compute PDE residuals here. |
| Existing `phydrax/discretization/fem/_hp_runtime.py`, `_hp_solver.py`, `_spectral_hp_completion.py` | Extend existing hp decisions/closure/transactions with QoI, geometry-order, conditioning and cost evidence; preserve reusable symbolic structures where layout is unchanged. |
| Existing `phydrax/discretization/iga/_adaptive.py`, `_transfer.py`, `_certificate.py` | Reuse DWR/THB/IGA epoch evidence and transfer; keep exact/refined spline spaces distinct from approximate remeshing. |
| Existing `phydrax/solver/_finite_element_adaptivity.py`, FE schedule/checkpoint and FV topology-event owners | Consume one admitted decision and atomically commit mesh, all fields/materials, solver refresh and lifecycle evidence through `CompositionRebind`. Failed reanalysis retains the accepted epoch. |
| Existing geometry `design/_schema.py`, `_constraints.py`, `_reduced.py`; B-Rep/implicit fixed-route realization owners | Preserve physical parameter identity, geometry feasibility and derivative margins across native mesh epochs. No differentiable facade over discontinuous topology. |
| Existing `phydrax/applications/solid_mechanics/_topology_reanalysis.py` and relevant phase-field/semiconductor/cavity consumers | Keep their scientific error, history, volume, positivity and reclosure criteria. Learned/native meshing proposals cannot replace mandatory independent physical reanalysis. |
| Existing `phydrax/qualification/_builtin_catalog.py` and W15 tools | Declare qualified versus research derivative/decision profiles, no unsupported superiority claims from estimator output alone. |

### Tests, smoke, benchmark and gate

- Extend proposal, hp transaction/runtime, IGA adaptive and solver topology-event tests; add **proposed new** `tests/unit/meshing/test_solver_aware_decisions.py`.
- Cases: geometry-error-dominated versus field-error-dominated problems, incompatible p/geometry/family requests, misleading cheap cost estimates, stale learned features, infeasible metrics, event-boundary derivative invalidation, missing state transfer and rejected physical reanalysis.
- Verify fixed-epoch JVP/VJP and adjoint duality against independent finite differences/analytic cases away from events. At an event, test explicit invalidation rather than an invented smooth gradient.
- Add **proposed new** `examples/curved_native_transfer.py`, `hp_metric_order_adaptation.py` and a bounded learned-proposal example. Smoke a complete design/solve/adapt/reanalyse loop and inspect accepted physical error, conservation and derivative evidence.
- Benchmark total cost to target error, estimator/decision overhead, topology/transfer/compile reuse, linear iterations/conditioning and peak memory. Compare against uniform refinement and established analytic adaptation before claiming a learned improvement.
- Gate: decisions improve the preregistered physical objective on mandatory workflows without weakening geometry/state constraints; a learned proposer is only an accelerator of the trusted system.

## 19. W15 — Qualification, benchmark evidence, migration and documentation

**Prerequisites:** corpus and reference contracts begin in wave A. Final qualification depends on every workstream; feature presence, safe refusal and candidate disposition are not release evidence.

### 19.1 Canonical capability matrix

Extend the existing qualification catalog with explicit source × operation × cell family × geometry order × feature/material/periodic/layer controls × execution placement × derivative/transfer contract profiles. Do not claim the Cartesian product of independent capability flags.

Keep four separate facts for every profile: implemented/admitted, mandatory-positive-corpus completion, independent scientific certification, and release/leadership evidence. A failed or missing native run cannot be replaced by an external provider result. Missing optional comparison engines remain `missing-dependency`, not pass or product failure.

| Existing or proposed file | Planned change |
| --- | --- |
| Existing `phydrax/qualification/_builtin_catalog.py` | Add native generation/CAD/family/transfer/distribution/design profiles, exact nonclaims, required evidence and source references using current catalog types. Research gates remain visibly open until passed. |
| Existing `tools/meshing_qualification.py` | Extend native end-to-end and adversarial scenarios; separate native and optional comparison execution. Reuse runtime identities and policy-bound evidence; decompose changed oversized helpers by real scenario/stage invariants. |
| Existing `tools/meshing_benchmarks.py` | Add comparative generation/adaptation/family/distribution/PDE campaigns with phase-separated timing and memory, repeated samples and frozen input/request identities. Do not create another benchmark framework. |
| Proposed new `tools/_meshing_cases.py` | Shared bounded corpus descriptors/fixture preparation used by the two tools; own source/request/provenance consistency only. Numerical references and acceptance algorithms remain independently owned. |
| Proposed new `tests/data/meshing/` and `tests/data/cad/` corpus artifacts | Licensed/source-digested geometry and format fixtures with coordinate contracts, required labels/features, tolerances, resource budgets and expected admission. Keep large optional corpora fetched through pinned bounded interchange, not unreviewed repository blobs. |
| Existing `tests/_support/` | Add only reusable meshing invariants/data strategies with independent fault adequacy; do not copy generation/refinement algorithms into test helpers. |
| Existing `tools/generate_public_api_manifest.py`, `check_public_api_manifest.py` | Regenerate/check explicit exports after each public cutover, including removed native-provider aliases and new canonical geometry/transfer owners. |
| Existing `tools/generate_capability_inventory.py`, `check_capability_consistency.py` | Regenerate/check capability, portfolio, closure and source-ledger data from the owning catalog. No hand-edited generated data or internal schema versions. |
| Existing `docs/data/public_api.json`, `capabilities.json`, `application_portfolios.json`, `capability_closure.json` and generator-owned source ledgers | Regenerate only after implemented exports/profiles exist. This planning document does not advertise proposed symbols as current capabilities. |

### 19.2 Mandatory positive end-to-end workflows

| Workflow | Required native path and independent outcome |
| --- | --- |
| Planar/curved features | Native curves/trimmed surfaces → conforming triangle/quad mesh → fidelity/global-embedding evidence → surface/planar PDE |
| Constrained solid | Nonconvex solid with cavity, embedded feature and material interface → quality tetrahedra → verified region/volume/interface coverage → manufactured solve |
| Implicit domain | Certified adaptive discovery/domain queries → native tetrahedra → metric adaptation → source-fidelity and physical-error convergence |
| Labeled image | Declared image interpretation → shared junction complex → conforming material tetrahedra → interface flux/inventory balance |
| Imperfect surface | Explicit repair/wrap/envelope policy → native surface/volume result → measured geometry/topology changes and successful consumer solve |
| Native CAD | Construct/import STEP/IGES/external BRep → native Boolean/partition/query → mesh/curve → solve, without OCP/Gmsh |
| Periodic system | Translational quotient or declared isometry domain → feature/material/layer-conforming mesh → orbit-preserving adaptation and compatible vector/flux transfer |
| Hybrid layers | Native wall → advancing/swept layers → immutable cap/core fill → curved mixed mesh → geometry-preserving adaptation and solve |
| General quads/hexes | Feature-conforming nontrivial quad and nonsweepable all-hex cases → positive globally embedded maps and exact family policy → FEM/FV consumer |
| Polyhedral system | Nonconvex/material domain → restricted Voronoi/power/polyhedra → independent coverage/conditioning evidence → VEM/FV and conservative remap |
| Distributed lifecycle | Native generation → refine/coarsen/repartition → all-field transfer → process-local checkpoint → changed-placement restart → continued solve |
| Moving overset | Native hole cutting/donor search → motion refresh → interpolative or separately certified conservative transfer → accepted PDE update |
| Design/learned loop | Fixed-epoch geometry/PDE derivatives → trusted h/p/metric/order proposal → native event/transfer/reanalysis → accepted physical objective |

Each workflow needs small deterministic routine cases and larger qualification cases. At least one case must combine several demanding controls (for example periodic + multimaterial + curved, or layers + narrow gap + adaptation) so isolated feature tests cannot conceal combination gaps.

### 19.3 Adversarial corpus and independent oracles

- Geometry: near-degenerate/cospherical/coplanar points, zero/near-zero features, extreme admissible scale, trim poles/seams, tangent/coincident surfaces, containment without boundary crossing, thin channels, holes and disconnected components.
- Semantics: contradictory scopes, stale source revisions, interface multiplicity, missing small materials, region precedence, protected feature birth/death, periodic transform cycles and required pure-family requests.
- Quality: sliver populations, extreme intended anisotropy, inverted curved interiors with positive corners, rational denominator uncertainty, high-valence cavities and negative/ill-conditioned target metrics.
- Transactions/resources: every named work/byte/entity bound, failure between prepare and commit, collective rank failure, ID overflow/collision detection, checkpoint tampering/partial shards and caller-owned state reuse.
- Transfers/physics: coverage gaps/double coverage, incompatible field families, nonpositive inventories, irreversible history, H(curl)/H(div) commuting defects, curved remap uncertainty and event-boundary gradients.
- Oracles: analytic geometry/volume/moments, independently implemented exact-dyadic predicate references, high-precision/interval references, manufactured PDE solutions, topological degree/incidence invariants and independent source/target coverage. External mesher agreement is useful but not proof.
- Deliberate-fault checks should show that a lost constrained face, flipped orientation, wrong region, invalid curved interior, skipped field component, stale witness or corrupted shard is detected by the public acceptance path.

### 19.4 Comparative performance and leadership gates

Freeze input geometry, physical units, tolerances, required features/materials/families, quality criteria, resource limits and hardware/runtime identities before tuning. Baselines are Gmsh/HXT for CAD/tetra throughput, CGAL/pygalmesh for implicit/image/periodic refinement, fTetWild for declared envelope semantics, Mmg/Omega_h for metric adaptation, VoroCrust for conforming Voronoi, and suitable structured/hex references for matching admitted domains. Manifold/Open3D/OpenVDB/TIOGA/METIS remain specialist comparisons for their corresponding operations.

Measure separately:

1. Source decode/construction and native geometry preparation.
2. Spatial/constraint/metric preparation and reusable-plan retention.
3. Surface/volume generation, classification, refinement and improvement.
4. Geometry projection/curving and independent certification.
5. Canonical publication, host/device transfer and downstream field/state transfer.
6. Lowering, compilation, first execution, warmed execution and recompilation count.
7. Compiler temporary/output/code bytes; peak host/device/process bytes and logical retained bytes.
8. Partitioning, neighbor communication, migration, checkpoint/restart and failure/cancellation work.
9. Solver preparation, iterations and end-to-end time/memory to a fixed physical error or QoI.

Vary the actual controlling capacities: input/mesh entities, local valence, feature/trim/junction complexity, anisotropy, layer/geometry order, cavity/image/subdivision budgets, partition count and workset slack. Record all failures/timeouts and worst-quality tails, not just averages over successful easy cases.

For equal requested guarantees, report Pareto comparisons of success rate, fidelity/quality, time, memory and downstream error. Both native and external candidates pass the same independent acceptance checks; account for their cost rather than timing generation alone for one side.

**Proposed headline performance target, not an observed result:** on each preregistered headline workflow family, no lower mandatory-case success or weaker achieved scientific constraints; at least a 20% reduction in median end-to-end time-to-target-error versus the best eligible baseline, without worse peak memory, or an explicitly reported Pareto improvement where the family has a different declared objective. Preserve per-case tails and disclose every regression. This target is a project acceptance objective, not a promise that one algorithm dominates every possible input.

Functionality completion and a leadership claim are separate gates. If the latter fails, continue the relevant performance/algorithm work or report the unmet goal; do not rename capability breadth as measured superiority.

### 19.5 Minimal safe verification policy for implementation

No commands in this section have been run as part of planning.

- Before each implementation slice, identify changes since the last merge into `dev` in the actual fresh worktree and compute the conservative affected-path/consumer closure. Do not freeze stale branch/commit identities in this plan.
- Reproduce each reported/source-identified behavioral defect with the smallest actual workflow before the fix; retain failing-before/passing-after regression coverage when practical.
- Run affected Python tests with `-n auto`, except native/provider/device-topology isolation that requires a dedicated invocation. Do not run unaffected numerical/application suites.
- Native kernel edits require the affected CMake/CTest targets plus sanitizer qualification for memory/undefined behavior; new geometric predicates need independent exact/adversarial checks.
- Use runtime smoke workflows from each workstream. Tests alone do not establish a working geometry → mesh → transfer → solver path.
- Public annotation changes require pinned ty and Ruff annotation checks through `python tools/check_typing.py check`, with zero diagnostics. Exercise installed-wheel typing through the existing installed-typing tool where the changed public annotations apply; do not introduce another checker.
- Add public annotation fixtures in `tests/typing/cases` for native specifications, geometry transitions, field-family transfer and distributed results, using truthful `assert_type` and narrow negative diagnostics. Mark dtype/rank-promotion/warning-sensitive native JAX surfaces `strict_jax`; preserve optional providers' native runtime policies in comparison tests.
- Add **proposed new** `tests/typing/installed/meshing.py` (and geometry/transfer fixtures where separation is clearer) with imports and `assert_type`/narrow negative cases for the actual new public specifications, canonical results, geometry/region/periodic evidence, transfers and owner-local publication. `tools/check_installed_typing.py` checks `tests/typing/installed/*.py` outside the checkout; source-only `tests/typing/cases` cannot establish installed-wheel exports or annotation preservation.
- Run selector audits through `tools/audit_selectors.py`; use the configured formatter/linter and measure materially changed symbol complexity before/after. Split by real lifecycle/invariant ownership, not arbitrary line-count helpers.
- Public surface changes require the existing API/capability generators/checkers and affected declaration/manifest integration tests.
- Full-suite execution is justified only for cross-cutting collection/configuration/shared-fixture/import/packaging changes, not each local algorithm phase. The final mandatory dependency/import cutover is cross-cutting and receives its appropriately broader closure.
- Permanent tests own consumer-visible behavior, boundaries and failure semantics. Delete obsolete source-text/wiring/incidental-wording tests instead of repinning them. New tests use bounded deterministic/property domains and explicit diagnostic IDs.
- Optional-engine comparisons run only where the corresponding engine is installed/available; missing dependencies are recorded. Native tests run without optional-engine skips.

### 19.6 Native dependency-isolation gate

Build/install the final package and its **own `phydrax-meshcore` native kernel** in a clean environment with external geometry/mesh/partition engines absent. Meshcore is an in-house requirement for its native construction routes, not an engine to remove from the full native acceptance environment.

Exercise actual base import, native CAD decode/construction/query, surface/tetra/polyhedral/quad/hex generation, layers, adaptation, transfers, overset and solver workflows. Observe loaded providers/processes and resulting runtime identities; static import-string tests are not sufficient. Test a separate base-import/missing-meshcore environment only for truthful availability/refusal semantics, not as evidence that native C++ generation works without its implementation.

Move `meshio` loading to explicit interchange boundaries where core imports currently force it, and make optional-codec coverage/losses explicit. Do not make reimplementing every mesh-file codec a prerequisite for the native numerical mesher, and do not claim retained codec execution is in-house. Native CAD formats and native lifecycle persistence must have their planned native implementations.

### 19.7 Clean cutover and documentation map

| Existing or proposed file | Required final update |
| --- | --- |
| Existing `phydrax/meshing/__init__.py`, `providers/__init__.py`, geometry/B-Rep/interchange/discretization/graph facades | Explicit canonical exports, lazy optional-provider boundaries, no removed-name aliases/re-exports and no eager OCP/mesh-engine imports. |
| Existing `pyproject.toml`, `uv.lock`, `native/meshcore/pyproject.toml`, `CMakeLists.txt` | Correct native packaging/dependencies/platform builds, source/build identity and optional comparison groups. Preserve provider worker sources only for explicit external routes. |
| Existing `phydrax/meshing/_interop.py`, `phydrax/geometry/simplicial/_io.py`, `phydrax/discretization/fem/_io.py`, `_spectral_hp_io.py`, `phydrax/_mesh_file_profiles.py` | Remove the observed eager `meshio` import edges from native import paths; keep codec execution lazy at explicit file boundaries and preserve shared format admission, node ordering and loss reports. Move `meshio` to an explicit optional interchange dependency when these callers are migrated; native array artifacts and solver preparation must still work with that codec absent. |
| Existing `docs/guides_meshing.md`, `docs/api/meshing.md` | Full native source/operation/family/control matrix, certificate meanings, transfer matrix, failure/research/resource boundaries and optional comparisons. |
| Existing `docs/api/geometry.md`, `docs/guides_differentiable_geometry.md`, `docs/guides_multiregion_surfaces.md` | Native CAD/source query and approximation distinctions, fixed-epoch derivatives, junction/periodic semantics and explicit conversion routes. |
| Existing `docs/guides_file_interchange.md`, `docs/api/interchange.md`; proposed new `docs/guides_native_cad.md` | Exact format/entity coverage, native/external BRep distinction, units/occurrences, source digests, malformed-input/resource refusal and native persistence. |
| Existing `docs/guides_discretization.md`, `docs/guides_finite_elements.md`, `docs/guides_isogeometric_analysis.md` and affected solver/FV/AMR/lifecycle guides | Geometry/field-family transfer, accepted epoch/rollback, compatible flux/circulation, curved integration and restart semantics. |
| Proposed new `docs/guides_distributed_meshing.md`, `docs/guides_native_overset.md` | Owner-local generation/adaptation, communication/capacity/restart, hole/donor/motion behavior and interpolation versus conservation. |
| Existing `examples/meshing_native.py`, `adaptive_bisection_heat.py`, `adaptive_device_simplex.py`, `anisotropic_metric_adaptation.py`, `ale_conservative_remesh.py`, `cad_high_order_curving.py`, `boundary_layer_core_mesh.py`, `delaunay_voronoi.py` | Migrate generic demonstrations to native execution and real end-to-end acceptance; keep intentional low-level primitive examples. |
| Workstream-proposed new examples | Implement the native surface/tetra/periodic/image/reconstruction/quad/hex/polyhedral/overset/design smoke workflows with bounded inputs and observable scientific evidence. |
| Existing `examples/meshing_omega_h.py` and explicit provider examples/tests | Retain as clearly optional comparisons/interoperability, not examples presented as fully native execution. |
| Existing `README.md`, `mkdocs.yml`, `CHANGELOG.md`, `NOTICE` | Update navigation, dependency/coverage claims, public cutovers and mathematical/source provenance only after smoke proof. Publish no unmeasured superiority or unsupported universal guarantees. |

The replacement map is explicit:

- `NativeImplicitProvider`/its provider plan facade → `NativeMeshingProvider`/`NativeMeshingPlan`; geometry implicit discovery/realization remains.
- `FTetWildCompartmentProvider`, engine-coupled `CompartmentMeshingSpec` and wrapper `CompartmentMeshingResult` → geometry-owned `CompartmentMeshingSource` + native volume request + `CellMeshingResult.region_evidence`, including Neurofluid consumer migration; optional fTetWild comparison remains separately named.
- OCCT-owned native CAD import/query/partition/tessellation → W07 native owners; optional external-shape bridge remains explicit and lazy.
- Qhull/PyVista-owned native reconstruction → meshcore/native reconstruction owners.
- METIS-owned canonical GRAPH → native graph partition owner; METIS comparison is explicit.
- Gmsh-owned native layer core fill → W03/W09 native fill; Gmsh layer comparisons remain external.
- TIOGA-owned native moving overset → W13; TIOGA comparison remains external.
- Vertex-P1 transfer assumptions → W08 field/geometry/material-specific transactions; no compatibility shim applies a wrong transfer.
- `SimplexTopologyEdit` → canonical `CellTopologyEdit` for native mixed/polyhedral and existing simplex commits, without an alias; periodic quotient and owner-local storage descriptors extend the existing carrier rather than introduce second mesh types.

Each cutover updates all references through LSP plus docs/examples/generated-data discovery, removes the obsolete native path and tests, and leaves unrelated concurrent work untouched.

## 20. Research gates and integration checkpoints

| Gate | Required result | What does not close it |
| --- | --- | --- |
| Exact decisions versus constructions | Proven filter/error domain and independent postclassification of every accepted construction | “Uses exact predicates” while coordinates/coverage are unbounded approximations |
| Complete PLC recovery | Mandatory constraints/materials/cavities recovered under declared immutable/subdividable policies | Convex-hull tetrahedralization or centroid clipping |
| Difficult CAD | Native complete intersection/arrangement evidence for the fixed positive corpus, including tangencies/coincidences | Small residuals, an OCCT fallback, or blanket refusal of all difficult fixtures |
| Adaptive implicit topology | Source-bound enclosures and coverage sufficient for the topology claim | Dense samples, QEF convergence or a theorem-name string alone |
| General all-hex | Nontrivial nonsweepable corpus with family, topology, fidelity, validity and resource requirements met | Only box/sweep output, all-negative tests, or silent mixed-cell substitution |
| Curved global embedding/remap | Certified relevant domains/intersections and bounded geometric integration error | Positive corner Jacobians, node projection residuals or sampled overlap |
| Distributed independence | Local construction/commit/restart with complete collective semantics and no external mesher | Partitioned storage followed by a mandatory serial gather |
| Solver/state closure | Every field/material/history invariant and accepted-step transaction passes | Correct connectivity plus partial field transfer |
| Leadership | Preregistered like-for-like correctness/quality/cost evidence | Large API inventory, anecdotal fast cases or external-engine performance credited to native code |

Implementation checkpoints:

1. **Contract-safe baseline:** W00 regressions closed; corpus and evidence obligations frozen.
2. **Native constrained tetrahedral vertical slice:** W01–W03 execute a complete cavity/material workflow through a solver without external engines.
3. **Source and CAD independence:** W04–W07 plus native reconstruction/Boolean/interchange positive cases close their coverage ledger.
4. **Simulation lifecycle closure:** W08/W09 preserve curved geometry, all required state and rollback through repeated native remeshing.
5. **Cell-family closure:** W10/W11 meet the required quad/hex/polyhedral positive and research gates.
6. **Scale and assemblies:** W12/W13 complete native distributed generation/refine/coarsen/restart and moving overset.
7. **Physics-aware advantage:** W14/W15 demonstrate the declared physical-error/cost objective and qualify actual derivative scope.
8. **Final cutover:** all mandatory matrix entries, consumer migrations, docs/generated data, installed native isolation and comparative gates accounted for. No phase boundary by itself is the complete deliverable.

## 21. Implementation working rules

- A later `//code` request starts on a fresh worktree under the mandated `../phydra-labs/.worktrees/` location, using a neutral native-meshing branch/worktree name. Copy the authoritative ignored `AGENTS.md` into that worktree before implementation. `//code--` is the explicit current-worktree exception.
- Re-read touched source and resolve current LSP references before edits. This plan names observed owners, not permission to overwrite concurrent changes.
- Shared contracts are integrated first by one owner. Parallel slices receive exact interface/identity/evidence boundaries and skip mid-flight build/lint/tests/formatters; integrated verification runs once per completed slice as selected by the repository policy.
- Do not implement scaffolds, fake native fallbacks or public selectors backed only by refusals. New files listed here must own the stated invariant/algorithm; reuse an existing owner instead if implementation-time discovery shows it already exists.
- Update docs/changelog and regenerate public/capability data after smoke proof of each permanent cutover. Do not publish proposed capabilities as implemented.
- This plan is complete as a planning deliverable; the workstreams, new files, acceptance runs and leadership targets are future work, not claims about the current repository.

## 22. Primary comparison and algorithm references

- [Gmsh reference manual: mesh algorithms, CAD and structured/extruded routes](https://gmsh.info/doc/texinfo/gmsh.html#Mesh-module).
- [Pygalmesh upstream workflow and control coverage](https://github.com/meshpro/pygalmesh/blob/main/README.md).
- [CGAL 3D mesh generation: feature protection, restricted refinement and sliver optimization](https://doc.cgal.org/latest/Mesh_3/index.html).
- [CGAL periodic 3D mesh generation: quotient-domain construction and feature handling](https://doc.cgal.org/latest/Periodic_3_mesh_3/index.html).
- [Quality tetrahedral mesh generation with HXT](https://arxiv.org/abs/2008.08508).
- [Fast tetrahedral meshing in the wild](https://dl.acm.org/doi/10.1145/3386569.3392385).
- [Mmg native upstream operation and adaptation documentation](https://mmgtools.org/index.html).
- [VoroCrust domain-conforming Voronoi meshing](https://vorocrust.sandia.gov/).
- [Hex-Mesh Generation and Processing: a Survey](https://arxiv.org/abs/2202.12670).

These references guide independent implementation and comparison. They do not transfer another library's guarantees to Phydrax and do not authorize copying incompatible licensed code.
