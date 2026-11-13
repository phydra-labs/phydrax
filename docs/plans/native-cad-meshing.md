# Native CAD and meshing: clean integration plan

## 1. Decision, baseline, and scope

This is an implementation plan, not implementation authorization. No project code, tests, builds, benchmarks, or generators were run while preparing it. The source review used current `dev` at `7bdc501d0` and the donor branch `feature/native-meshing-20260929` at `c56614a9f`. Relevant paths were inspected in their existing worktrees. Source/test presence is not qualification evidence.

**Decision:** integrate the donor's mathematics and useful representations onto current `dev`; do not merge its operational architecture. Keep the full scientific CAD/meshing scope and explicitly include T-splines. Remove compulsory work/query/entity/scratch/byte quotas, per-object accounting, preparation credits, resource receipts, process-issued proof authority, and budget replay from native scientific operations. Do not replace them with enormous defaults, an unlimited mode, or a differently named resource framework.

Implementation starts only after `//code`, on a fresh worktree such as `../.worktrees/native-cad-meshing`, branch `feature/native-cad-meshing`, from the then-current branch. Copy the authoritative ignored `AGENTS.md` before editing. `//code--` alone authorizes staying in the current worktree. Leave the donor and parent checkout untouched. Do not implicitly commit, merge, or open a PR.

### Scientific scope

- Native analytic/rational/source-defined CAD: topology, trims/p-curves, queries, construction, intersections, Boolean/partition/sewing operations, supported offsets/blends/defeaturing, sketches/design, and interchange.
- Explicit native T-spline surfaces, genuinely local geometry-preserving refinement, shared CAD/IGA evaluation, qualified analysis, fixed-topology differentiation, and lossless native persistence.
- Curves, planar domains, parametric/implicit surfaces, constrained solids, labeled images/materials, reconstructed/imperfect surfaces, and their native initial meshes.
- Metric adaptation, high-order geometry, boundary layers and native core fill, mixed cells, structured/swept/multiblock meshes, quads/hexes, and restricted power/polyhedral meshes.
- Periodic quotient topology, native partitioning/distributed construction and adaptation, native moving overset connectivity, conservative/compatible transfer, solver-aware decisions, design, and complete accepted-state continuation.
- Current scientific consumers and existing native primitives remain supported. External engines remain explicitly optional comparisons/interoperability routes, never native fallbacks.

No GUI, general CAD document-management system, unrelated solver rewrite, or arbitrary loft/path-sweep authoring is inferred from this request. The donor's documented CAD authoring scope is primitives, line/conic sketch loops, extrusion/revolution, and supported exact operation trees. General quad/hex construction and difficult singular CAD remain real research-dependent parts of the scientific goal; they cannot be declared complete through all-negative tests or silent scope reduction.

### Observed facts that constrain the plan

1. Current `dev` already owns `CompiledGeometry`/`DesignState`, `CellMesh`/`CellGeometrySpec`, source selections, field spaces, transfers, and generic lifecycle. These remain canonical.
2. Current `MeshingLimits` already imposes resource ceilings. Removing only the donor's extra limits is insufficient. LSP found direct consumers in specs, adaptation/proposals, optional providers, worker adapters, tests, and facades; dynamic provider accesses require supplementary searches.
3. Current topology edits are `SimplexTopologyEdit`; the donor's generalized `CellTopologyEdit` is a clean replacement, not an already-existing current API.
4. Current `PeriodicCell` is a lattice owner, not a periodic `CellMesh` quotient representation. The latter must be integrated explicitly.
5. Current `native/meshcore` already exists. Extend its CMake/header/predicate/Delaunay/CDT/clipping owners; add only absent kernels. Preserve current Python/package requirements rather than copying donor metadata.
6. The donor already implements native n-ary B-Rep and planar partitions. Its `_partition.py:native_arrangement` dispatches native rectilinear cells or `full_overlay_partition_brep`. Current `dev` is the OCCT-backed side to replace. Do not mistakenly reimplement a supposedly missing donor partition algorithm.
7. Donor dual quads/hexes are useful topology conversions, not proof of general field-aligned quad or high-quality all-hex meshing. Supported fillets are family-bounded, not a general vertex-blend network.
8. Current `_tspline.py` is explicitly a rectilinear bicubic basis-only profile. `analysis_suitable_refinement_closure` closes inserted lines across the domain and constructs `TMesh2D.tensor_product`; this does not satisfy genuinely local T-spline refinement. Its `geometry_valid` checks the parameter partition, not a physical control-point/weight map.

## 2. Contracts fixed before parallel implementation

### 2.1 Scientific ownership

| Owner | Responsibility |
|---|---|
| `phydrax.geometry` | Authoritative source geometry, source entities/occurrences, oriented boundaries and regions, charts, projection/classification/intersections, source accuracy and design state. |
| `phydrax._interpolation` | Shared spline evaluation, extraction, knot insertion and local basis/refinement mathematics. No CAD-specific second T-spline or NURBS implementation. |
| `native/meshcore` | Robust geometric decisions/constructions and owned local topology kernels. Internal mutability and normal allocation ownership are permitted. |
| `phydrax.meshing` | Scientific requests, resolved native route, generation/adaptation, source associations, quality/compliance, and assembly of accepted canonical results. |
| `phydrax.discretization` | Cell topology, coordinate maps, field spaces, geometry transitions, field-specific transfer and runtime realization. |
| Existing `linalg`, `nonlinear`, `optim`, `sparse`, spatial and graph owners | Numerical solves, optimization, exact/enclosed matrix operations, reusable relations/search, and graph partitioning. |
| Existing lifecycle/model/interchange owners | Atomic accepted-state transitions, scientific codecs, safe persistence and independent reconstruction. No meshing-specific accounting runtime. |

A geometric source may have several realizations. Tessellation is never silently promoted to source CAD. A sampled implicit/image interpretation never acquires a stronger exactness claim merely because a mesh is valid. A source-parametric T-spline remains one surface across its extraction cells; knot lines are not automatically physical CAD edges.

### 2.2 Public request and result cutover

- Keep one native construction API: `NativeMeshingProvider(options=None).plan(source, specification, coordinate_contract=...).execute()`. Keep the existing `prepare_mesh_adaptation` / `execute_mesh_adaptation` pair. Do not add another `MeshEngine`, universal `mesh()` wrapper, or parallel result type.
- Retain `CurveMeshingSpec`, `SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, `VolumeMeshingSpec`, `CellMeshingTarget`, family policies, size/metric/feature/region/layer/periodic controls. Remove `limits` and `limits_id` from scientific requests and identities.
- Add a direct physical source-deviation request to surface/volume specifications where the existing source/control does not already own it. Exact mode and an explicit positive approximation tolerance are different requests. Do not invent a CAD approximation tolerance from available memory, a sampling grid, or an arbitrary route default. Contradictory tolerances are rejected at their owning boundary.
- `NativeMeshingOptions` contains algorithm choices and numerical schedules only. A missing route resolves deterministically during planning from source capabilities and requested family/semantics. Preserve explicit route selection. If the requirements do not select a valid supported route, explain the missing capability; do not try fallback algorithms after failure or relax the request.
- Accept existing source owners directly where they carry the complete input. Keep typed composite sources only where they add scientific data: fixed implicit discovery/grid, power sites/weights, mapped reference domain, block interfaces, sweep stations, or prepared layers/core. Normalize them into one prepared `MeshingDomain` view. Do not expose forwarding-only source wrappers or create a second source identity system.
- Add `MeshingScope.from_geometry(source, entity_ids)` and `MeshingScope.whole(source, dimension)` in `_scope.py`; retain the existing mesh `from_selection`. They own revision/occurrence/kind validation, not provider tag conversion. Remove `GmshProvider.entity_scope` / `whole_scope` after all callers migrate.
- Retain `CellMeshingResult`: canonical mesh + independent coordinate geometry + frame + region/patch/label/attribute organization + source associations + mathematical acceptance/quality/compliance + explicit derivative scope. Runtime provenance and optional phase timings are separate observations, not acceptance premises.
- Remove `MeshingRuntimeInfo.enforced_limits` / `unenforced_limits`. Keep actual provider/build/backend and determinism provenance. A real allocation failure, unsupported operation, numerical nonconvergence, and invalid geometry remain distinct outcomes.
- Remove `MeshingExecutionPolicy`. Keep `AbstractMeshingSession` and real native/provider lifetimes. Gmsh thread selection already belongs to `GmshOptions`; native worker selection belongs to existing execution configuration/options. No duplicated native timeout/payload/cleanup policy.
- Replace `SimplexTopologyEdit` with generalized `CellTopologyEdit` in the existing owner. Preserve `assemble_topology_edit` as the common accepted topology/lineage boundary. Every bisection/local-metric/adaptation caller migrates; no compatibility alias.
- Replace engine-coupled `CompartmentMeshingSpec`, `CompartmentMeshingResult`, and `FTetWildCompartmentProvider` with geometry-owned `CompartmentMeshingSource`, ordinary volume specification, native provider and canonical result/region evidence. Migrate Neurofluid and all other consumers. Keep the explicit FTetWild comparison provider for its own existing domain.

### 2.3 Identity and reconstruction

Separate source identity, topology identity, numerical geometry revision, scientific request, algorithm/prepared structure, and runtime observations. Reuse current identity utilities; do not globally rewrite `_identity.py` or `_fingerprint.py`.

- Source identity includes actual source definitions, units/frame, occurrence/branch identity and material semantics, not derived tessellation density or storage policy.
- `topology_id` retains incidence/orientation/scientific IDs. Numerical coordinate updates advance the existing geometry/numeric revision without inventing new topology.
- Plan identity includes the source/request and algorithm choices that affect numerical output. Worker count for a declared deterministic route, scratch chunk sizes, elapsed time, receipts and allocation history are not scientific identity.
- `result_id` binds accepted scientific content and the strength/scope of its claims, not execution timing or resource receipts. Keep provider/runtime provenance separately. Do not erase scientifically different approximate/exact methods merely because coordinates happen to coincide.
- Proof reuse requires matching immutable inputs and validated mathematical premises. A hash, a public `certified=True`, a cache hit, or process-local issuer membership is not proof. Restore either verifies a sufficient witness or recomputes the required mathematical check.
- These are intentional representation/API/fingerprint changes. Document them, migrate all readers/writers/templates, and reject obsolete structures rather than reuse historical IDs or add internal generation fields/shims.
- Fresh-process continuation reconstructs source + mesh/coordinate map + field/history schema and arrays + periodic/distribution state without the original object, caller-held Python template, or live issuer. Domain codecs use the generic model recipe/archive owners; they do not restore old spending credits.

### 2.4 Memory, termination, and safety

Use ordinary C++ RAII/standard containers and immutable host preparation, with growable private storage and reusable scratch. Preserve checked index/size arithmetic, native handle ownership, safe C buffers, real allocation failures and no-allocation atomic commit sections. Do not retain pointers beyond their owners or turn safe output-capacity checks into no-ops.

Stream complete broad-phase/candidate work in private chunks. Chunk size affects scheduling, not which valid problem sizes are accepted. JAX topology work uses stable capacity buckets; overflow triggers an explicit host-side growth/reprepare transition before retrying the unchanged scientific operation. No silent truncation and no per-candidate host synchronization.

For Bernstein/interval certification, use deterministic adaptive depth-first/chunked traversal rather than a breadth tree capped by arbitrary depth or piece counts. Every adaptive owner must implement a finite unresolved exit before its old caps disappear. In addition to a decisive bound, stop an undecided box when its extent reaches the caller's declared physical/parameter resolution or when its endpoints have no distinct interior point in the declared numerical evaluation format. Apply this numerical-resolution rule even when the host stores endpoints as exact rational numbers; unbounded Fraction subdivision is not an escape from termination. Use finite exact multiplicity/contact/threshold decisions where the supported algebra provides them; otherwise return UNRESOLVED, never a relaxed pass. Zero approximation tolerance does not request infinite subdivision: use a finite exact route or refuse an unsupported exact claim. The root/certificate/implicit-discovery owners retain the achieved enclosure and reason at this exit. Cancellation is an additional operational exit, not the sole way to stop a public proof. Numerical resolution is not a work counter or a cap on the number of cells; it cannot weaken the requested geometric acceptance criterion.

Acceptance must include a determinant exactly at its declared floor, a tangent/multiple root, a coincident zero set, a rational interval sequence that continues tightening indefinitely in exact arithmetic, and a thin positive feature requiring deeper refinement than old defaults. Verify finite UNRESOLVED for genuinely undecidable-at-resolution cases and successful completion for the resolved feature. Finite traversal can still be expensive; no universal complexity or termination-at-requested-accuracy claim is made.

Keep geometric tolerances, minimum determinant/rank margins, conditioning requirements, requested fidelity, metric anisotropy and layer thickness. They are mathematical/physical requirements, not logistical limits. Degenerate IEEE predicate-domain restrictions require a new arithmetic proof before removal.

Existing external process and untrusted-artifact boundaries retain actual descriptor/path/checksum/no-pickle/extent protections. Native scientific operations do not inherit their quotas. CAD semantic import policies lose entity/depth/byte/occurrence caps only after the corresponding decoder is made safe without them: rewrite STEP `_value`/`_list` and reference traversal, and BRep-text offset/trim/sweep carrier decoding, as explicit-stack iterative traversals; validate count products, offsets and available payload/reference lengths before allocating knot/pole/topology banks; preserve cycle/dangling-reference rejection. `_cad_io.py` uses descriptor-owned streamed or indexed input rather than an unconditional full-file bytes copy. Keep shared assembly definitions as a DAG and enumerate occurrences as needed, not mandatory exponential flattening. Propagate recoverable allocation failures without publishing a partial model. Optional caller-selected bounded ingestion remains outside geometry. Test deeply nested valid acyclic input, huge declared counts with short payload, cyclic references and injected allocation failure before deleting old depth/count guards.

## 3. File map conventions and ownership

`E` = extend/change an existing current-dev file. `A` = adapt an existing donor file absent on dev, at the same path unless stated. `N` = explicitly proposed new file. `X` = do not import donor infrastructure, or remove the obsolete current symbol after its callers migrate. A row naming several files assigns a coherent responsibility to each named member; it is not permission to copy a package wholesale.

Paths beginning `geometry/`, `meshing/`, `discretization/`, `interchange/`, `linalg/`, `graph/`, `lifecycle/`, or `applications/` are relative to `phydrax/`. Bare helper filenames in a grouped row share the explicit owner directory in that row/section; `providers/` in meshing rows means `phydrax/meshing/providers/`.

Integration owns public facades, `_contracts.py`, `_result.py`, `_canonical.py`, the central `_adaptation.py`, packaging and generated records. Kernel, spline, CAD, geometry-proof, generation, transfer and distributed owners each have exclusive implementation files. Route owners deliver complete helpers against these contracts, not concurrent edits to the central dispatch/result modules.

Any donor change not mapped to a scientific requirement is excluded by default. This prevents unrelated donor linalg, optimizer, compiler, model, application and policy deltas from entering through a broad merge. A newly discovered required helper must have its owning invariant and consumers added to this plan before adoption; do not carry unrelated changes transitively.

**Signature cutover obligation:** removing a budget owner means removing every import, argument, stored field, context activation, facade export and codec entry that refers to it—not passing `None`, retaining an ignored `budget=`, or introducing a shim. In particular, migrate donor coordinate-budget plumbing in `discretization/_cell_geometry.py`, `_cell_mesh.py`, `_cell_geometry_transfer.py`, `_cell_geometry_transition_types.py`, `_cell_geometry_predecessor.py`, `_chart_certification.py`, `_chart_execution.py`, `_coordinate_enclosure.py`, `phydrax/_geometry_predicates.py`, `applications/neurofluid/_model.py`, all CAD/meshing route consumers, and their model recipes. Retain only actual numeric preparation in those owners. An adopted vertical slice must import and execute with the excluded budget/receipt modules absent; no half-wired intermediate API counts as a completed slice.

## 4. Public API, current callers, and policy removal

| Files | Action |
|---|---|
| E `phydrax/meshing/_contracts.py` | Remove `MeshingLimits`; remove spec limit fields/parameters/hash inputs. Add curve request and missing scientific target/fidelity controls. Keep exhaustive family/operation/derivative selectors, physical control consistency and no-fallback admission. |
| E `phydrax/meshing/_scope.py` | Implement source-neutral constructors, occurrence-aware CAD scope conversion and current mesh selection. Preserve scientific IDs separate from positions. No publication-ticket membership. |
| E `phydrax/meshing/_controls.py`, `_sizing.py`, `_metric.py`, `_quality.py` | Reuse existing scientific controls, SPD property checks, gradation, size/quality measurements. Add missing native controls without duplicating selectors or metric algebra. Keep explicit repair policy for untrusted metric samples; never silently clip invalid metrics. |
| E `phydrax/meshing/_session.py` | Remove `MeshingExecutionPolicy`; preserve ephemeral session close/ownership behavior. |
| E `phydrax/meshing/_result.py`, `_trace.py`, `_canonical.py`, `_organization.py` | Build one accepted result with compact mathematical findings. Remove resource claims and receipt/clock dependencies from scientific IDs. Extend organization for exact material/interface and distributed relations; do not replace source identity with names or positions. |
| E `phydrax/meshing/_audit.py`, `_audit_topology.py` | Remove `maximum_connectivity_entries`, `maximum_coincidence_candidates`, `maximum_intersection_candidates`; stream complete scans. Preserve audit dispositions and their honest scope. Diagnostic welding does not mutate geometry or assign semantic identity. Native success cannot use RECORD/SKIP to conceal a required geometry failure. |
| E `phydrax/meshing/_adaptation.py`, `_proposals.py` | Remove policy quotas, work pricing and capacity-derived mark truncation. Reuse the existing request/transaction APIs; dispatch to scientific route helpers. A learned proposal is untrusted until geometry, transfer and physical reanalysis pass. |
| A `phydrax/meshing/_failures.py` | Extract only typed scientific/operational failure data needed by new routes; migrate existing failure imports together if this separation prevents cycles. Do not import cut-stage receipt graphs. Keep public exception categories stable where meaning is unchanged. |
| A `phydrax/meshing/providers/_native.py`, `_native_options.py`, `_native_source_type.py`, `_native_sources.py`, `_native_schedules.py` | Implement compact typed planning/dispatch and source normalization. Use explicit source/route/family capability checks. Remove preparation receipts, scope/workspace/budget arguments and issued-plan registration. Numerical schedules are not an alternate resource API. |
| X donor `meshing/_native_preparation.py`, `_measurements.py`, `_support_work.py`, `_metric_execution.py`, `_family_execution.py`, `_layer_core_resources.py` | Do not import accounting. Move any genuinely needed phase-local numerical preparation into the relevant scientific owner. No zero-valued ledger stubs. |
| A/adapt `providers/_native_session.py` only if a real reusable native session is necessary | Keep close/thread/handle ownership and CPU pool reuse, without timeout/allowance/receipt semantics; normal plan execution manages it internally. Do not require a user session just to mesh. |
| X donor `providers/_native_publication.py`, `_native_route_policy.py`, `_native_route_checks.py`, `_native_route_metadata.py`, `_native_support.py` as wholesale modules | Transfer their scientific admission/compliance checks into the canonical result/certification and route owners. Do not copy parallel publication policy objects or staged resource handovers. |
| E `phydrax/meshing/providers/_gmsh.py`, `_gmsh_execute.py`, `_gmsh_remesh.py`, `_gmsh_boundary_layer.py`, `_gmsh_preflight.py`, `_gmsh_constraints.py`, `_gmsh_elements.py`, `_gmsh_evidence.py`, `_gmsh_import.py`, `_gmsh_layers.py`, `_gmsh_sizing.py` | Remove spec limits/session policy and provider-owned scope methods. Keep external options, global-state lock, real source import correspondence and explicit provider failure. Native source comparison uses a verified explicit export, not an implicit OCP dependency in native CAD. |
| E `phydrax/meshing/providers/_conversion.py`, `_ftetwild.py`, `_mmg.py`, `_omega_h.py`, `_tioga.py`, `_vorocrust.py`, `_openvdb.py`, `_poisson.py`, `_manifold.py`, `_worker.py`, `_sources.py` | Remove native request limits and runtime limit-label fields. Preserve true array validation, source IDs, optional-provider semantics, worker error propagation and explicit interpolation/transfer losses. Remove `_check_result_limits` and quota-only `_check_arrays` branches, not their shape/finite/index checks. |
| E `phydrax/_external_runtime.py` narrowly; E `meshing/providers/_worker.py` | Add operation-timeout and input/output-exchange defaults to the existing `NativeWorkerPolicy`, using the old external-call defaults. `ProviderWorker.call` reads them from its constructor policy instead of `MeshingLimits`; retain `NativeWorker.call` enforcement and explicit call overrides. No new policy type, no native-core dependency and no global runtime rewrite. Runtime policy identity changes honestly; scientific request identity does not. |
| E `phydrax/meshing/_compartments.py`, `phydrax/geometry/_compartments.py`, `phydrax/applications/neurofluid/_model.py`, `__init__.py` | Clean native source/request/result cutover; delete the old compartment provider/spec/result symbols after migrations. Preserve all label, compartment, adjacency, frame and 3D–1D transport semantics. |
| E `phydrax/meshing/__init__.py`, `providers/__init__.py`, `geometry/__init__.py`, `geometry/brep/__init__.py`, `interchange/__init__.py`, `discretization/__init__.py` | Explicit lazy exports for the final owners; remove obsolete aliases/resource types and optional-engine eager imports. Preserve current configuration/bootstrap and package ownership conventions. |

**Known current caller sets:** LSP found `MeshingLimits` in `_contracts`, `_adaptation`, `_proposals`, `_compartments`, worker/conversion and MMG/Omega_h/TIOGA/VoroCrust/implicit providers; dynamic Gmsh/FTetWild accesses add further callers. Scope-method callers include CAD partition, Gmsh capabilities/layers/planar tests, `tools/meshing_qualification.py`, `tools/meshing_benchmarks.py`, and CAD examples. `CompartmentMeshingResult` is a real input of `NeurofluidCase`. `NativeImplicitProvider`/`ImplicitMeshingPlan` are removed after their surface users migrate to the common native provider. Re-run references against the implementation worktree immediately before exported edits; supplement with lazy-export, codec, fixture and generated-data searches.

## 5. Native robust kernel and execution boundary

### 5.1 Existing native files

| Files | Action |
|---|---|
| E `native/meshcore/include/phydrax_meshcore.h` | Define the clean C ABI for retained algorithms. Remove donor execution/batch/ticket/storage-accounting entry points and work/query/cavity/scratch quota fields. Keep actual pointer extents, output buffer capacities, statuses, opaque owned handles and build identity. Add a distinct real allocation-failure status if the existing status conflates it with policy exhaustion. |
| E `native/meshcore/src/capi_guard.hpp`, `core_capi.cpp` | Check sizes/strides/overflow before access; do not let C++ exceptions cross C. Preserve all-input validation before mutation. Remove spend/poll/receipt paths. |
| E `native/meshcore/src/predicates.hpp`, `predicates.cpp`, `expansion.hpp`, `filtered.hpp` | Preserve filtered/adaptive exact arithmetic, exact-zero and SoS rules and IEEE assumptions. Extract donor source/rational support without operation charging. Exact sign does not imply exact floating output construction. |
| E `native/meshcore/src/mesh.hpp`, `triangulation2d.hpp`, `delaunay2d.cpp`, `delaunay3d.cpp`, `cdt2d.cpp`, `spatial_sort.hpp` | Extend the current triangulation/CDT structures and source-witness behavior, not a second triangulator. Ordinary growable storage replaces policy allocators. Separate local reusable slots from scientific IDs. |
| E `native/meshcore/src/clip2d.cpp`, `clip3d.cpp`, `clip_common.hpp` | Preserve symbolic classifications, rounded output bounds/moments and direct-output validation. Remove arbitrary clip-vertex/scratch ceilings by growable staging where the algorithm permits, not unchecked writes. |
| E `phydrax/_meshcore.py` | Extend current typed ctypes bindings. Do not copy the donor's entire large module. Keep loader/ABI/build verification, dtype/shape conversion and handle cleanup. Add numerical wrappers only. Borrow inputs synchronously; persistent handles own copied/immutable source banks. Default export is sizes → exact NumPy allocation → checked copy → free handle. Zero-copy is not required for this cutover. |
| E `phydrax/_geometry_predicates.py` | Integrate exact represented-coordinate operations without budget arguments. Preserve filtered-device UNCERTAIN and explicit host exact resolution. Do not silently fall back after a native error. |
| E `phydrax/geometry/_triangulation.py`, `_supermesh.py`, `_tetra_intersections.py` | Migrate current low-level callers to the reduced ABI while preserving triangulation, clipping and conservative-overlap behavior. |

### 5.2 Donor-only numerical kernels

All following are `A` under `native/meshcore/src/` at their named paths. Port the mathematical bodies and scientific failure behavior, not allocator/work plumbing.

| Files | Responsibility |
|---|---|
| `dyadic.hpp`, `exact_scalar.hpp`, `exact_line.hpp`, `floating_point_mode.hpp`, `rounding.hpp` | Exact/rational calculations and correctly bounded publication; retain arithmetic preconditions. |
| `geometry3.hpp`, `topology_keys.hpp`, `boundary_policy.hpp` | Shared geometric primitives, canonical topological keys and fixed/conforming boundary semantics. |
| `cavity.hpp`, `triangulation3d.hpp` | Owned tetrahedral adjacency, ghost closure and staged validated cavity replacement. Reserve all changed banks before the no-allocation mutation block. |
| `plc3d.hpp`, `plc3d.cpp`, `feature_protection.cpp` | Exact PLC validation, acute-feature protection, constrained segment/facet recovery and region flood classification. No centroid-label or convex-hull substitute. |
| `source_construction.hpp`, `facet_coverage.hpp` | Source points/strata, exact parameters, rounded-carrier deviation and current source coverage. Preserve scientific history only where curved-source claims require it. |
| `tet_mesh.hpp`, `tet_mesh.cpp`, `tet_mesh_state.cpp`, `tet_mesh_source.cpp`, `tet_mesh_insertion.cpp`, `tet_mesh_collapse.cpp`, `tet_mesh_refinement.cpp`, `tet_mesh_improvement.cpp`, `tet_mesh_export.cpp` | Persistent source-aware tetrahedral construction/edit state, stale-generation checks, local operations and canonical export. |
| `refine3d.hpp`, `refine3d.cpp`, `improve3d.hpp`, `improve3d.cpp` | Refinement, local flips/ring reconnection/collapse/relocation/sliver improvement. Preserve unmet quality and real numerical termination status. |
| `clip3d.hpp`, `reference_tetrahedron_clip.inc`, `arrangement.cpp`, `exact_tetrahedra_arrangement.inc`, `intersections.hpp`, `intersections.cpp` | Robust clipping/contact/arrangement kernels, consumed through streamed spatial candidates. |
| `restricted_delaunay.hpp`, `restricted_delaunay.cpp`, `power_diagram.cpp`, `periodic.hpp`, `periodic.cpp` | Restricted duals, region-restricted power pieces, periodic image sufficiency and quotient topology. |
| `surface_mesh.cpp`, `surface_envelope.cpp`, `level_set.cpp` | Chart-aware surface edits, explicitly interpreted envelope construction and P1 zero insertion. |
| `quad_topology.cpp`, `hex_topology.cpp`, `mixed_refinement.cpp` | Dual/reference templates and oriented shared-face closure. Do not advertise these alone as general quad/hex generation. |
| `interval_program.cpp`, `quadratic_bounds.cpp` | Outward interval-program execution and continuous mapped-cell validity. |
| `graph_partition.cpp` | Native multilevel graph partitioning, owned publicly by `phydrax.graph`, not by an alternative mesh partition service. |
| `parallel.hpp`, `parallel.cpp` | Reusable CPU pool for independent batches with deterministic merge. Retain necessary worker configuration entry points without donor quota/64-slot admission machinery. No claim that serial topology insertion automatically becomes parallel. |

`X`: `bounded_memory.hpp`, `bounded_memory.cpp`, execution/batch/target portions of `parallel_capi.cpp`, and accounting-only fields in `counter_records.hpp`. Use ordinary containers/RAII, not an allocator-alias replacement framework. Keep convergence/quality/source failure data in the owning result structures.

**Required algorithmic performance work:** replace `plc3d.cpp:collect_edge_cavity`'s per-edge scan over every tetrahedron slot with a conservative dynamic tetrahedron-AABB index or complete adjacency traversal. Maintain it across accepted edits, preserve exact narrow-phase tests, and validate against exhaustive small cases including touching/coplanar constraints. Reuse the package spatial owners for Python-side broad phases; no dense all-pairs tables. Keep optional observations in benchmark code, not operation admission.

## 6. Shared coordinate, enclosure, and scientific certification

| Files | Action |
|---|---|
| E `phydrax/discretization/_cell_geometry_validity.py` | Remove default fixed depth/piece quota fields and identity inputs. Preserve determinant/planarity thresholds and CERTIFIED_VALID/INVALID/UNRESOLVED. Implement adaptive streamed Bernstein subdivision with representability/non-progress termination; preserve current family coverage. |
| A `phydrax/geometry/_mesh_certificates.py`, `_embedding_state.py`, `_embedding_cells.py` | Compact mathematical bindings/findings/status and cell/stratum scopes; no `MeshCertificateLimits`, resource fields, or issued-proof authority. |
| A `phydrax/geometry/_global_embedding.py`, `_planar_chain_embedding.py`, `_restricted_embedding.py`, `_prism_embedding.py`, `_boundary_degree_embedding.py`, `_mapped_embedding.py`, `_mapped_corners.py`, `_exact_polyhedral_geometry.py` | Preserve local-to-global embedding, legitimate shared contacts, exterior-degree and mapped/reference proofs. Only reuse local checks when their full premises justify a global conclusion. |
| A `phydrax/geometry/_domain_coverage.py`, `_domain_boundary.py`, `_domain_chart_chain.py`, `_domain_numerics.py`, `_domain_planar_boundary.py`, `_domain_preparation.py`, `_planar_coverage.py`, `_mapped_coverage.py`, `_mapped_reference_coverage.py`, `_mapped_reference_domain.py`, `_planar_primitives.py` | Exact/bounded source-region measure and oriented exterior/interface coverage. Total volume equality alone is not coverage. Treat domain descriptors as views over real source authority, not independently self-certified replacement geometry. |
| A `phydrax/geometry/_meshing_domain.py`, `_source_boundary_records.py`, `_surface_source_support.py`, `_mapped_source_support.py` | One revision/frame-bound source-query/stratum view and chart support, with batched source evaluation. No preparation ledger. |
| A `phydrax/geometry/_source_fidelity.py`, `_radial_source_fidelity.py`, `_source_interpolation.py`, `_approximate_source_coverage.py` | Two directed deviation bounds and actual source coverage. Approximate/sampled routes retain their weaker labels. Do not equate projected-node residuals with continuous fidelity. |
| A `phydrax/geometry/_interval_enclosure.py`, `_interval_native.py`, `_rational.py`; A `phydrax/discretization/_coordinate_enclosure.py` | Extract interval/rational/algebraic polynomial composition, Bernstein restriction and exact source maps. Retain numeric preparation only; remove credit/cache-issuer/byte logic from the donor coordinate owner. |
| A `phydrax/linalg/_exact.py`; E `phydrax/linalg/__init__.py` | Exact-field solve/determinant owner for rational/algebraic coordinate systems, singular refusal and exact pivot logic. No local duplicate Gaussian elimination in CAD or meshing. |
| E `phydrax/linalg/_hermitian_spectral.py`, `__init__.py` | Add donor `hermitian_log_enclosure` and `hermitian_exp_enclosure` mathematics needed by continuous curved metric lengths. Keep SPD lower bounds, spectral residuals, eigenframe and truncation error; ordinary floating log/exp is not an enclosure. Remove coordinate-budget signatures. |
| A `phydrax/meshing/_certification.py`, `_certification_inputs.py`, `_certification_schedule.py`, scientific pieces of `_initial_certification.py` | Compose required local validity, embedding, source coverage/fidelity and semantic checks once. Bind actual inputs. These are mathematical obligations, not new route policy/receipt systems. Canonical result assembly remains in existing owners. |
| E `phydrax/_interpolation/_bspline.py`, `_bspline_grid.py`, `_tensor_bspline.py`, `_rational_spline.py` only as needed | Shared span-local jets, knot insertion, extraction and denominator/bounds needed by CAD/T-splines/IGA. Preserve other interpolation consumers and numerical conventions. |
| X donor `phydrax/_resource_ticket.py`, `_issued_proofs.py`, `discretization/_coordinate_budget.py`, `_cell_geometry_transition_budget.py` | No import. Extract any genuine immutable numeric preparation into its mathematical owner instead of retaining no-op budget parameters. |
| X donor `_host_storage.py` accounting and broad `_fingerprint.py`, `_model/_structure.py`, `_model/_artifacts.py` deltas | Retain only a demonstrated necessary immutable-array/codec utility at an existing canonical owner. Immutability is required for safe reuse; host-byte estimates and paid owner traversal are not. Keep exact registered reconstruction and validation. |

`phydrax/geometry/_planar_embedding.py` may be adopted as the neutral owner of the existing `PlanarEmbedding` value when removing its current dependency on the OCCT partition module. Move all internal imports and keep the single intended public spelling; do not add an alternate coordinate-frame class.

## 7. Native CAD, design, and interchange

### 7.1 Source representation and operations

Unless marked E, the following B-Rep helpers are A at `phydrax/geometry/brep/`.

| Files | Action |
|---|---|
| E `_model.py`, `_patches.py`, `_source.py`, `_differentiable.py`, `_projection.py`, `__init__.py` | Integrate source-authoritative `BRepGeometry`, curves/surfaces/p-curves, oriented coedges/loops/shells/solids, occurrences and native query/projection. Separate source and tessellation IDs. Preserve fixed-topology design state, ambiguity and derivative margins. |
| `_constructors.py`, `_tessellation.py` | Native primitives, line/conic profile extrusion/revolution and supported exact operation trees. Tessellation is optional derived output and reuses common curve/surface generation, not a second mesher. |
| `_exact.py`, `_parameter_range.py`, `_interval_source.py`, `_interval_derivatives.py`, `_spline_box.py`, `_gap_evidence.py`, `_correspondence.py`, `_source_correspondence.py`, `_root_bindings.py`, `_conic_gaps.py`, `_trim_helpers.py` | Source bounds, periodic/root endpoints and coupled curve/surface correspondence. Remove budget arguments; never promote sampled fit to continuous source correspondence. |
| `_placement.py`, `_placed.py`, `_placed_affine.py` | Exact occurrence composition and source-frame interpretation, without coordinate-based occurrence welding. |
| `_query.py`, `_query_membership.py`, `_query_projection.py`, `_query_types.py`, `_projection_contracts.py` | Prepared native source queries with batched pointwise status, trim membership, lower/upper distance bounds and multiple-minimum handling. Remove `BRepQueryBudget` and point/operation/scratch caps. |
| `_affine_query.py`, `_analytic_carrier_closest.py`, `_analytic_cylinder_closest.py`, `_analytic_disk_closest.py`, `_analytic_edge_closest.py`, `_analytic_source_regions.py`, `_containment.py`, `_sphere_membership.py`, `_revolution_membership.py`, `_stationary_isolation.py`, `_clearance_bounds.py` | Reuse specialized analytic and general certified query bodies. Fast paths require their actual hypotheses; native B-Rep does not claim globally exact distance from a local projection. |
| `_intersection.py`, `_intersection_curve.py`, `_intersection_kernel.py`, `_contact_indicatrix.py` | Root isolation, regular/singular branch discovery, Krawczyk contraction and coupled 3D/p-curve continuation. Preserve source-defined non-rational intersection curves, branch connectivity/completeness, tangencies and coincidences. |
| `_boolean.py`, `_boolean_arrangement.py`, `_boolean_cells.py`, `_boolean_coincidence.py`, `_boolean_core.py`, `_boolean_full_overlay.py`, `_boolean_material.py` | Native source-face arrangements/DCEL/material classification, empty/disconnected results and exhaustive lineage. Shared interfaces remain one oriented source relation. No fitted-spline or triangle fallback for exact Boolean output. |
| `_sewing.py`, `_selection.py`, `_edge_classification.py`; E `phydrax/geometry/_cad_revision.py` | Proved correspondence, selection/feature classification and explicit split/merge/deletion lineage. Fold donor `_revision.py` conversions into the canonical CAD revision owner where possible. No fuzzy healing or guessed naming continuity. |
| `_measure.py`, `_normal_range.py`, `_section_clearance.py` | Source measure/normal/section/clearance calculations with actual error/convergence scope. |
| `_planar_arrangement.py`, `_planar_blend.py`, `_oblique_fillet.py`, `_oblique_chamfer.py`, `_closed_seam_blend.py`, `_parallel_cylinder_blend.py`, `_fillet.py`, `_fillet_radius_map.py`, `_planar_offset.py`, `_offset_result.py`, `_offset_rounding.py`, `_defeature.py`, `_deformation.py`, `_linked_parameters.py` | Retain the donor's supported analytic blend/offset/defeature/deformation features and parameter sensitivity contracts. Distinguish supported families from general rolling-ball/corner-network capabilities. Failure preserves the source. |
| `_curve_approximation_bounds.py`, `_curve_approximation_topology.py` | Continuous coupled export-error and topology-preservation proofs for explicitly requested approximation. Remove proof-cell/depth/byte ceilings as scientific acceptance requirements. |
| E `_partition.py`, `_planar.py` | Replace current OCCT execution with the donor's native n-ary region/void/precedence arrangements. Make partition computation in-memory: remove mandatory `destination` and serialization/reopen from the operation. Build named sets/association graph from actual native lineage, not a codec round trip. Persist/export explicitly afterward. |
| X `_coordinate_request.py`; adapt `_carrier_pieces.py`, `_request_preparations.py` only for mathematics/reuse | Remove object/JSON/PyTree byte pricing and public preparation-credit APIs. Retain source-span pieces and private call-local preparation only where needed to avoid repeated geometry work. |

**Required CAD positive cases:** non-rational regular quadric intersection through Boolean, coupled p-curves, lossless native archive and exact-export refusal/explicit bounded approximation; tangent/coincident/singular fixtures with correctly scoped positive and negative outcomes; seams/poles, cavities, assemblies, precedence and selected void targets. A small residual or successful source parse does not establish complete intersection discovery or native workflow closure.

### 7.2 Design, process and codec files

| Files | Action |
|---|---|
| E `phydrax/geometry/design/_schema.py`, `_constraints.py`, `_sketch.py`, `_continuation.py`, `_qualification.py`, `_reduced.py`, `_search.py`, `__init__.py`; A `_solve.py`, `_search_result.py`, `_sketch_certificate.py`, `_sketch_dimensions.py`, `_sketch_priority.py`, `_naming.py` | Reuse ParameterId/DesignState and native numerical solvers. Add source-profile lowering, root/DOF evidence, supported dimension sensitivities and explicit lineage-bound recipe names. Residual convergence is not a unique-root certificate. Keep existing `Sketch.to_source` meaning; distinct profile/model results are not aliases. |
| E `phydrax/geometry/process/_lower.py`, `_stack.py`, `phydrax/meshing/_layer_extrusion.py`, `_planar_bands.py` | Native in-memory construction/partition and explicit persistence calls. Preserve stack precedence, wall/cap/volume identity and declared shape changes. Remove incidental OCP dependence and required temporary-file round trips. |
| A `phydrax/interchange/_cad.py`, `_cad_io.py`, `_cad_carriers.py`, `_cad_placement_frame.py`, `_cad_placement_evidence.py` | Physical import/export and approximation policy, exact unit/placement handling and coverage/loss reports. No resource manifests, occurrence quotas or accounting receipts in CAD identity. |
| A `phydrax/interchange/_step.py`, `_step_metadata.py`, `_step_repair.py`, `_iges.py`, `_cad_brep_text.py` | Native declared entity/format coverage, reference graphs, trims, units and occurrences. Preserve independent external fixtures and explicit repair choices. Unsupported required entities identify their dependency chain. Do not silently linearize/fill missing p-curves. |
| A `phydrax/interchange/_dxf.py` | Native planar-profile exchange for the donor's lines/arcs/bulges/splines/circles/ellipses. Keep declared units and physical junction-gap semantics without snapping. No solid-BRep fallback or parser/byte ticket framework. |
| A `phydrax/interchange/_cad_archive.py`; E `_catalog.py`, `_report.py`, `__init__.py` | Lossless source geometry, root/branch/T-spline structure and occurrence persistence through existing archive ownership. Remove `BRepArchiveReceipt`/resource receipt wrappers; writers return the existing path/result convention with meaningful codec reports only. |
| Move E `phydrax/geometry/brep/_occt.py` → A `phydrax/interchange/_cad_occt.py`; A `_cad_occt_projection.py` | Retain explicitly named lazy OCCT interoperability/projection. Remove obsolete geometry-owner exports, not aliases. Native BRepSource/Boolean/meshing cannot accept an external tessellation as native carrier authority. |
| E `phydrax/geometry/surface/_model.py`, `_contracts.py`, `_interop.py`, `_high_order.py`, `_intersection.py`, `__init__.py`; A `_arrangement.py`, `_boolean.py` | Separate triangle-surface operations and native source-aware import. CAD-to-surface tessellation reports its loss; triangle export is not exact STEP/IGES CAD. |
| E `phydrax/geometry/reconstruction/_core.py`, `__init__.py`; A `_normals.py`, `_poisson.py`, `_poisson_octree.py`, `_robustness.py` | Native normal/reconstruction algorithms, including screened Poisson and octree routes. Preserve approximation/coverage/topology evidence and explicit repair interpretation. No hidden Qhull/Open3D/PyVista dependency on native routes. |
| E `phydrax/geometry/analytic/_sweeps.py`, `surface/_g1_multipatch.py` | Share source-map/seam capabilities without conflating analytic implicit sweeps, B-Rep authoring and swept mesh generation. |

Current partition caller migrations include `geometry/process/_lower.py`, `_planar_bands.py`, CAD/planar/process/layer tests, both meshing tools and examples. Gmsh comparisons explicitly serialize a supported native source when the external engine requires a file. Export failure is separate from successful in-memory CAD construction.

## 8. Explicit T-spline workstream

T-splines are not covered merely by saying NURBS. This workstream is required for the final CAD/meshing deliverable and has its own positive CAD-to-analysis witness.

### 8.1 Representation and shared substrate

- Reuse the actual existing `TMesh2D`, local knot, anchor, T-junction-extension, extraction and transfer mathematics. Move reusable topology/basis/extraction/refinement ownership into the shared spline substrate, not into a CAD-specific copy.
- Preserve the currently supportable rectilinear bicubic analysis-suitable profile as an explicit profile, while distinguishing it from arbitrary T-meshes and unsupported extraordinary configurations. This is not a claim of all T-spline variants or vendor format support.
- A source T-spline stores T-mesh incidence, T-junctions/extensions, exact knot structure/multiplicity, persistent anchor IDs, control points and positive rational weights. Persistent IDs are not assumed to equal coefficient-row positions.
- Structural basis metadata is immutable; control points and weights are dynamic numerical leaves. Extraction/gather arrays are reusable fixed-role numerical data, not array-bearing static closures.
- Add `TSplineSurfacePatch` as one source-parametric B-Rep carrier using the shared basis realization. Do not replace it permanently with disconnected tensor-product patches. Local rational Bézier pieces are a prepared exact/bounded query representation with source-cell/parameter ancestry.

### 8.2 Genuine local refinement

**Selected refinement algorithm:** use the bicubic analysis-suitable coupled-extension-graph local refinement of Scott–Li–Sederberg–Hughes, sections 4.2–4.3 of [Local refinement of analysis-suitable T-splines](http://staff.ustc.edu.cn/~lixustc/Figs/2012_refine.pdf), rather than a generic unspecified closure. The admitted profile is a rectangular rectilinear bicubic T-mesh with simple interior knot intervals, the paper's open-boundary/zero-interval convention, no extraordinary vertices, topology-derived local knot windows, and an analysis-suitable extended mesh. For each active control vertex/anchor, march in each topological direction until two vertices or perpendicular edges are crossed; use the traversed knot intervals, with the prescribed zero intervals at the outer boundary, to derive the five-knot cubic windows. Keep index topology distinct from equal-valued boundary knots. Caller-supplied windows are checked against this derivation, not accepted because they happen to give small residuals.

The numerical construction is:

1. For every marked active rectangle, stage a four-way split at the exact parameter midpoints u_mid = (u0 + u1)/2 and v_mid = (v0 + v1)/2, computed in the canonical rational coordinate owner, and insert their corresponding ordered index positions. Children have parent refinement depth plus one and canonical order lower-u/lower-v, upper-u/lower-v, lower-u/upper-v, upper-u/upper-v. Closure-induced splits likewise record their actual parent and increment refinement depth; this depth is not the alternating-bisection level of the Morgenstern–Peterseim algorithm. Preserve unchanged entity IDs, then allocate genuinely new IDs once in canonical parent/address order after closure, using the existing lineage allocator—not visitation or worker order. No source mutation occurs before acceptance. Derive the finite candidate-coordinate universe from the source plus these explicit marked splits.
2. Express source and candidate extensions in the candidate's common index space and construct the coupled extension graph, including source face extensions needed for nestedness.
3. Enumerate legal insertion edges at weighted T-junctions, compute the resulting coupled-graph weight, and select the minimum with canonical geometric/index/ID tie ordering. Insert only the selected missing skeleton segments, update affected topology/extensions/anchors and repeat until the coupled graph has zero weight. The graph weight may stay equal or increase on a step; do not impose a false strict-weight-decrease test.
4. Define the finite **atomic skeleton** from all source and marked-split u/v coordinates. A legal insertion must occupy at least one previously absent atomic segment and never remove an occupied segment. Splitting a stored topological edge changes its record count, not this occupancy invariant. Strict growth of this finite occupied-segment set proves termination without a work quota; a no-op or unavailable legal insertion is an explicit algorithm failure with source rollback. The full tensor skeleton is the mathematical terminal upper bound, not a fallback to construct eagerly or a required target for every refinement.
5. Build exact sparse knot-insertion routes in the common knot coordinate system and verify complete source-basis inclusion in the target on local polynomial pieces. Scott's M is coarse-basis-by-fine-basis; the target-by-source operator P used below is Mᵀ. Zero coupled-graph weight is an admissibility premise, not a substitute for checking the implemented coefficient map. Failure of inclusion, continuity or geometry verification rolls back the complete refinement.

Qualify all four T-junction orientations, interacting closures, boundaries and repeated boundary knots; repeated interior knots remain explicitly unsupported by this selected profile rather than silently given a C2 claim. Geometry regularity and trimming are separate checks. The [Morgenstern–Peterseim admissible-refinement analysis](https://arxiv.org/html/1407.6175v4) supplies a useful independent nestedness/closure reference for its level-bisection subclass, not a theorem for every arbitrary input or this implementation's runtime. In particular, locality is not a universal constant bound per marked cell: its linear-complexity statement is amortized over admissible refinement sequences. Benchmark actual closure propagation.

The refinement algorithm may propagate when suitability requires it, but cannot unconditionally rebuild the Cartesian product of all u/v breaks. A positive locality fixture must leave distant cells/anchors unchanged while preserving a non-tensor T-mesh.

Apply rational geometry refinement in homogeneous coordinates: transfer weighted control coordinates and weights together, then divide by positive target weights. Rational field coefficients require the corresponding weighted prolongation; directly applying polynomial coefficient transfer to Euclidean control points or rational field coefficients is not generally correct. Independent field spaces retain their own basis/weight contract.

For a sparse nested-basis map P, the contracts are polynomial coefficients c′ = Pc; homogeneous geometry H′ = PH with H = (wX, w); and a rational field with its own weights a, q′ = P(aq) / P(a). These are distinct actions. Verify the basis-inclusion identity on complete local polynomial pieces, not only the current sample residual. Do not drop small nonzero coefficients or identify different knots by decimal rounding.

**Exact scalar/storage contract:** use the shared represented-coordinate owner for canonical rational knot-insertion coefficients and exact target homogeneous coefficients, or its lossless exact source-expression composition—not binary64 coefficients qualified only by sampled residuals. Binary64 inputs are exact dyadics, but an insertion ratio such as 1/3 is not binary-representable. Preserve that ratio and the resulting homogeneous source bank in identity/persistence; prepare one bounded numerical realization for JAX, projection and analysis. CAD and IGA consume the same realization and its declared error, while exact-equivalence claims refer to the authoritative exact expression. Do not put Python Fraction objects in device leaves. A source update either flows through the retained exact map or explicitly establishes a new authored control bank and revision; rounded materialization alone cannot retain the old exact-equivalence claim. Reuse the exact rational/source-expression and archive conventions from section 6/11, not a new T-spline scalar serialization format. A nonuniform-weight, non-affine insertion with a 1/3 coefficient must survive native persistence with equality of complete exact polynomial pieces.

Keep raw authored weights in both CAD and IGA evaluation. Remove current `NURBSGeometryState` mean-one normalization in the shared IGA owner rather than retaining an independently rounded normalized view; separate floating divisions need not preserve one exact common weight ratio. Any conditioning scale must be a separately carried common factor with proved ratio preservation, or an explicit optimization-coordinate map back to the raw source before evaluation. Migrate tensor-NURBS and T-spline consumers together, including state reconstruction, volume/certificate/query/side-action/coupling tests, `tools/iga_h1_qualification.py`, `tools/iga_h1_benchmarks.py`, `tools/iga_closure_qualification.py`, IGA examples and documentation. This is an intentional canonical-state/fingerprint and floating-rounding change, not an identity-preserving cleanup. Verify same-source geometry within declared numerical error, raw-bank persistence, awkward nonuniform weights/non-power-of-two scales, and the weight-gauge derivative null direction.

A refinement is an atomic discrete event. Source geometry remains unchanged on failed admissibility/nestedness/regularity checks. A new basis topology/revision and explicit source-to-target relation are published; physically unchanged geometry is recorded as exact-equivalence evidence, not by reusing an incompatible structural ID.

### 8.3 Suitability, continuity and analysis

- Separate parameter-partition validity, analysis-suitability/dual-compatibility hypotheses, basis independence, partition of unity/nonnegativity, trace continuity, rational denominator positivity, physical map regularity, and global surface embedding.
- Current sampled partition/nonnegativity diagnostics remain diagnostics, not continuous or physical-geometry proof. Validate complete Bernstein coefficient identities and the selected ASTS theorem hypotheses. Distinguish the 16 bicubic Bernstein coefficients per polynomial piece from the number of active anchors: globally independent T-splines need not be locally independent on one piece. Remove `extraction.rank == extraction.local_width` as a universal ASTS admission premise; use local rank only for a claim that actually requires it. For example, the cited admissible-refinement result bounds supported functions by 2(p+1)(q+1), not (p+1)(q+1), on a cell. Allocate sparse local supports from their actual validated structure; no arbitrary 16-anchor cap and no global dense extraction/transfer matrix.
- Derive anchors/local knot windows from the admitted T-mesh rules, or validate explicit metadata against those rules; current noncrossing-extension and local-rank flags alone are insufficient. Replace each-cell/all-anchor scanning with a prepared local-support index and sparse cell-to-anchor routes. Fixed-route JAX evaluation must not call NumPy or select cells in a Python loop over query points.
- Verify cross-cell jets at their declared continuity order, including T-junction neighborhoods, repeated knots, outer boundaries and trimmed interfaces. A basis-only `geometry_valid` boolean must not authorize a folded physical control net.
- Implement T-spline continuity explicitly in `AbstractSurfacePatch.is_c1_on` and related consumers; do not inherit the donor's unhandled-family `True` default. Extend `SurfaceSourceRootAtlas` and native surface-meshing dispatch, which currently have `BSplineSurfacePatch`-only admissions, through validated source-piece capabilities rather than claiming tensor-spline checks also cover T-splines.
- Feed the same source/control state into qualified IGA realization. Add supported scalar surface H1 analysis first as part of this complete workstream; admit higher-regularity/shell or compatible-field consumers only when their actual requirements are implemented and independently verified. Do not broaden all IGA spaces merely because a carrier evaluates.
- Refinement/T-junction changes are discrete. Control-point/weight JVP/VJP and supported higher derivatives are valid only within fixed incidence/knots/active trim/query branches and map-regularity margins.

### 8.4 CAD, meshing and interchange

Trimming, projection, intersections, tessellation and meshing consume the same source patch and its prepared Bézier cell pieces. Internal extraction boundaries/T-junctions are not additional physical CAD features. Refinement preserves face identity and trim geometry through explicit source correspondence; derived query/mesh preparations are revalidated against the new representation.

Keep one explicit source-support owner: the native `BRepModel` contains its single per-face `patches` table together with `BRepGeometry`'s curves, coedges, loops and occurrences. Do not duplicate the surface table in both objects merely to integrate T-splines. Full physical source/query/certificate bindings must include the model's complete patch payload, T-mesh structure and numeric control/weight state; a curve/topology-only `geometry_id` cannot stand in for that complete binding. Representation refinement creates new revision-qualified entity IDs and an explicit correspondence for the same semantic face; never reuse a stale source revision because the geometric image is unchanged.

Native persistence retains the full T-mesh, knots, anchors, control values, weights, trims, occurrence and refinement relation. In a fresh process it reconstructs without the original object.

External export has three explicit outcomes: (a) geometry-preserving Bézier/NURBS decomposition with declared loss of T-mesh/refinement structure, only if numeric encoding and trim/seam correspondence actually preserve the represented geometry; (b) an explicitly requested bounded approximation with its error and topology evidence; (c) unsupported/refused. Sampling or a successful self-round-trip never proves exactness. Test nonbinary extraction/weight cases where finite external numeric encoding may prevent an exact claim.

A quadrilateral mesh with T-junctions remains a mesh. It is not accepted as a T-spline until knots, anchors, basis and admissibility are explicitly authored and validated.

### 8.5 T-spline file map

| Files | Action |
|---|---|
| E `phydrax/discretization/iga/_tspline.py` | Extract reusable spline ownership; replace global tensorizing closure, sampled-only acceptance shortcuts and rounded knot-key identity. Retain IGA-specific realization/consumer requirements only. No duplicate class implementation or old-location compatibility implementation. |
| N `phydrax/_interpolation/_tspline.py` | Canonical T-mesh/local-knot/anchor admissibility, local extraction and sparse local refinement mathematics shared by CAD and IGA. Reuse existing univariate/rational spline primitives. |
| E `phydrax/_interpolation/_bspline.py`, `_bspline_grid.py`, `_bspline_projection.py`, `_rational_spline.py`, `_tensor_bspline.py`, `__init__.py` | Reuse shared jets and existing knot-insertion routes; integrate donor Boehm/exact Bézier extraction and outward coefficient bounds here. Remove the separate collocation extraction/Cox–de Boor implementation from `_tspline.py`. No second CAD-specific rational evaluator or speculative parallel Bézier substrate. Preserve tensor spline/IGA users. |
| N `phydrax/geometry/brep/_tspline.py`; E `_patches.py`, `_model.py`, `_source.py`, `_differentiable.py`, `_tessellation.py` | `TSplineSurfacePatch`, control/weight state and native source integration; explicit parameter schema, oriented map and source-cell piece extraction. No tensor-grid flattening as the authoritative representation. |
| E/A `geometry/brep/_query.py`, `_query_projection.py`, `_query_membership.py`, `_intersection.py`, `_intersection_curve.py`, `_correspondence.py`, `_source_correspondence.py`, `_normal_range.py`, `_spline_box.py`, `_measure.py`, `_curve_approximation_bounds.py`, `_curve_approximation_topology.py`; E `geometry/_atlas.py` | Dispatch through the shared surface-piece/evaluation contract, including trims crossing extraction cells and local refinement. Implement actual continuity and one-sided jet semantics. Preserve ambiguous/singular status and same-parameter p-curves; analytic-only blend/defeature routes do not implicitly admit T-splines. |
| A/adapt `geometry/_source_fidelity.py`, `_surface_source_support.py`, `_domain_preparation.py`; A/adapt `meshing/_surface_generation.py`, `_surface_metric.py`, `providers/_native_surface.py`; E `_curving.py` | Extend tensor-only source-root atlas and surface admissions to validated local T-spline pieces. Retain one semantic surface, boundary semantics, source associations and continuity after refinement. Cell partitions do not become physical feature edges. |
| E `discretization/iga/_basis.py`, `_realization.py`, `_geometry.py`, `_topology.py`, `_identity.py`, `_plan.py`, `_manifold.py`, `_actions.py`, `_transfer.py`, `_adaptive.py`, `_cut.py`, `_query.py`, `_certificate.py`, `_field_view.py`, `_interfaces.py`, `__init__.py` | Connect local extraction to existing IGA preparation/assembly/query/trim/transfer, without a second PDE solver or tensor-product coercion. Parameter dimension comes from the basis, not the rank of a flat anchor/control bank. Separate basis and physical geometry evidence; integrate sparse weighted prolongation and fixed-topology AD. Expose T-mesh construction/refinement through this single public IGA facade only with the complete qualified route; the shared implementation lives in `_interpolation`. |
| A/adapt `interchange/_cad_archive.py`, `_cad_carriers.py`, `_step.py`, `_iges.py`, `_cad_brep_text.py`; E model codec registrations and facades | Lossless native T structure; explicit exact-geometry/structure-loss, approximation, or refusal outside native format. No invented standard T-spline entity. |
| E `tools/iga_closure_qualification.py`, `docs/guides_isogeometric_analysis.md`, `docs/guides_isogeometric_analysis_support.md`, `docs/api/discretization/isogeometric.md`; E/A geometry/meshing API guides; N `examples/native_tspline_cad_analysis.py` | Register and qualify the actual geometry/basis/space/formulation/backend/precision/derivative/restart tuple. Current public IGA plans do not admit T-meshes. Basis-only records or the words extracted Bernstein do not expand the qualified tensor-only surface. |
| E `tests/unit/discretization/test_iga_correctness_repairs.py`, `test_iga_structural_topology.py`, `test_iga_tensor_transfer.py`, `test_iga_single_patch.py`; N `tests/unit/discretization/test_tspline_basis.py`, `test_tspline_local_refinement.py`, `test_tspline_geometry.py`; N `tests/unit/geometry/test_tspline_brep.py`; N `tests/unit/interchange/test_tspline_exchange.py`; N `tests/integration/test_tspline_cad_analysis.py` | Independently verify extraction identities, complete traces/continuity, local refinement, nonuniform-weight geometry and rational-field preservation, physical-map failure, JVP/VJP/Hessian and gauge invariance, native projection/intersection/trim/mesh, and fresh restore/export. Replace the obsolete dense-anchor-ID test with duplicate/foreign-ID refusal and arbitrary persistent-ID correctness. Add only distinct consumer-visible scenarios. |

**T-spline end-to-end acceptance:** author a non-tensor admissible T-mesh with nonuniform positive weights; build a curved trimmed native surface; perform source projection and meshing; locally refine one region while preserving the geometry and distant topology; prolong a nonconstant field correctly; solve/refine a qualified manufactured surface PDE on the same source; check fixed-topology gradients independently; native archive/fresh restore; external decomposition with the correct structure-loss/exactness report. Negative cases include crossing extensions, incompatible local knots, dependent bases, invalid weights, folded/degenerate maps, stale extraction/trim bindings and an unqualified quad mesh presented as a spline.

## 9. Native generation and improvement

All named A files below are donor-only on current dev; E files already exist. Routes return construction data/edits to the common scientific acceptance owner, not private successful-result types.
**Approved mapped source/root overlay cutover:** retain the original PLC facets, edges, occurrences, material sides, frame-sweep stations and coordinate maps. Partition authored facets and edges exactly against root faces and edges, with canonical reduced rational piece coordinates and independently proved source and root coverage. Split pieces keep their original entity identities and carry distinct root-qualified chart identities; they are not new physical source entities. Relation, query, witness, association and result identities and their PyTree/archive representations consequently change; original source and numerical geometry identities do not. Cold restore reconstructs the authority and revalidates the exact partitions and identities, not transient proof tokens or replayed requests.


| Files | Action and acceptance |
|---|---|
| A `meshing/providers/_native_curve.py`, `_native_planar.py`, `_native_planar_construction.py`, `_native_planar_domain.py` | Curve arc-length/size-density placement and exact planar CDT with holes/features; shared junction identity and continuous chord bounds where claimed. Remove fixed trim/pair/root-work ceilings; numerical inability is explicit. |
| A `meshing/_surface_generation.py`, `_surface_construction.py`, `_surface_boundary_sources.py`, `_surface_boundary_operations.py`; A `providers/_native_surface.py`, `_native_surface_association.py` | Discretize shared source curves once, mesh trimmed charts, refine by physical length/angle/fidelity and assemble orientation/lineage. Support analytic/rational/intersection/T-spline carriers through the same query contract. |
| A `meshing/_plc_source.py`, `_plc_planar_support.py`, `_plc_mapped_support.py`, `_volume_generation.py`; A `providers/_native_volume.py`, `_native_plc_geometry.py`, `_native_cad_volume.py` | Native constrained solid fill with actual region/interface recovery and source ancestry. CAD surface-to-PLC approximation is explicit and independently bounded to the original source; never label affine fill as exact curved volume geometry. Preserve fixed boundary versus conforming subdivision. |
| A `geometry/implicit/_analytic_profile.py`, `_adaptive_discovery.py`, `_enclosure.py`, `_projection_cover.py`; E `_policy.py`, `_discovery.py`, `_curve_discovery.py`, `_realization.py`, `_projection.py`, `_neural.py`; A `meshing/_implicit_volume.py` | Separate established-reach restricted-Delaunay and enclosure-complete adaptive discovery/fill. Detect hidden components/tangencies/singularities or refuse the unproved claim. Keep fixed-lattice design realization distinct from adaptive topology. Remove byte-derived box limits and compulsory depth caps; use growable/streamed discovery with representability checks. |
| E `geometry/multiregion_surface/_label_extraction.py`, `_topology.py`, `_geometry.py`, `_validation.py`, `_transfers.py`, `_transaction.py`, `_topology_transitions.py`, `_seeding.py`, `_remesh.py`, `_views.py`; E `meshing/_compartments.py`; A `providers/_native_compartment.py` | Native labeled-image/interface construction with actual source affine, categorical interpretation and junction incidence. Preserve tiny required materials, reflections/shears and per-region coverage; no centroid relabeling. |
| A `meshing/_surface_envelope.py`; E/A reconstruction and triangle arrangement owners from section 7 | Explicit repair/wrap/reconstruction source followed by conforming fill. Keep changed-geometry and sampled/certified interpretation visible. |
| E `meshing/_metric.py`, `_local_metric.py`, `_optimization.py`, `_bisection.py`, `_device_metric.py`; A `_tetra_metric.py`, `_surface_metric.py`, `_surface_metric_controller.py`, `_bisection_kernel_host.py`, `_device_tetra_metric.py` | Reuse true tensor-metric split/collapse/flip/reconnection/relocation and conforming h-refinement/coarsening. Do not scalarize an SPD metric. Preserve COMPLETE/STALLED/numerical-stop distinctions; a valid but unmet candidate is not request success. |
| A `meshing/_mixed_adaptation.py`, `_polyhedral_adaptation.py`, `_polyhedral_adaptation_shared.py`, `_polyhedral_plane_restriction.py` | Actual mixed-family face-signature closure and same-region polyhedral refinement/agglomeration/site regeneration. No freeze-all-nonsimplex substitute. Nonnested regeneration uses actual conservative/compatible transfer, not invented parent stencils. |

**Required combination work, not merely admission labels:** source-aware tensor-metric finishing after initial tetrahedral fill; periodic-orbit-synchronized 3D tetrahedral edits; curved-source preservation and high-order coordinate transitions through selected edits; mixed-family refine/coarsen with layer/interface closure; labeled material junctions and original source fidelity through regeneration. Route/family/order/operation support is recorded from implemented positive cases, not an unrestricted Cartesian-product claim.

When ordinary native metric phases cannot progress, bounded coupled endpoint or
periodic-orbit models use native quadratic programming to propose source-positive
displacements. The selected edge must improve its own violation rung without
worsening higher-priority rungs. Frozen tensors only define the proposal:
reinterpolated source metrics, actual global progress, determinant floors,
protection, region constraints, exact periodic transport, and normal native
admission remain authoritative. Actual associated implicit-facet refusals may
add proposal planes; no geometry-specific oracle changes acceptance.

`TetraMetricProposalError` retains the actual unsuccessful solver result when one
exists, its policy, admitted demand, and source/generation/edge/source-cell
binding. Only an independently audited local infeasibility certificate prunes
that local model; numerical, certificate, and resource failures are not converted
to a zero proposal, `STALLED`, or global mesh infeasibility. Native dense QP
early stopping uses the same declared primal-infeasibility tolerance and
normalized rays as the independent audit, including custom-derivative routes.

Relocation previews retain the checked-star cells named by `removed_cell_ids`;
those cells are validation context, not retired topology. Overlapping moves
commit one deterministic affected-cell/source-ancestor union. Current-star
exact preparation is reused only within one unchanged search, never in a
global array cache. Native implicit-facet refusal diagnostics are ephemeral,
generation/source-bound, and cleared across attempts, success, source changes,
and handle lifetime; no persisted evidence or snapshot fields are added.

**Approved thin-channel sizing correction:** the original `planar-thin-channel`
positive uses a 0.02 m uniform target and corresponding 0.03 m maximum size,
instead of 0.8 m and 1.2 m. Preserve its authored geometry, 25° minimum angle,
hard median-size acceptance, and 50% relative size tolerance. The former
controls are incompatible: the neck and minimum angle require at least 29 mesh
edges, while the region area permits at most 12 edges of length at least 0.4 m,
too few for the requested median. The user explicitly selected a finer positive
workflow rather than converting this case into an expected rejection.

## 10. Layers, high order, structured and general cell families

| Files | Action and acceptance |
|---|---|
| E `meshing/_boundary_layer.py`, `_layer_extrusion.py`, `_curving.py`, `_motion.py`; A `_layer_curving.py` | Native advancing/exact-sweep layers and high-order coordinate maps. Preserve physical first-cell height, active columns and collision outcomes. Thickness reduction/termination/merge is allowed only by the requested scientific policy. Whole-cell validity, source fidelity and embedding gate accepted curving. |
| A `meshing/_layer_core.py`, `_layer_core_association.py`, `_layer_core_controls.py`, `_layer_core_mapped.py`, `_layer_core_periodic.py`, `_layer_core_regions.py`, `_layer_core_topology.py`; A `providers/_native_layer.py` | Fill the actual remaining domain natively while keeping accepted layer nodes/cap facets/region sides fixed. Integrate transitions and combined coverage. Fold donor `_layer_core_publication.py` scientific assembly into canonical publication; omit `_layer_core_resources.py`. |
| A `meshing/_structured.py`, `_sweep.py`, `_multiblock.py`; A `providers/_native_structured.py` | Transfinite/elliptically improved blocks, triangle→prism and quad→hex sweeps, explicit conforming block gluing. Preserve opposite edge counts, face permutations, frame/periodic seam maps and curved coordinate realization. No coordinate-nearness welding. |
| A `meshing/_quad_generation.py`, `_dual_extraction.py`; A `providers/_native_dual.py` | Preserve explicit dual conversion. Implement the separate real feature/cross-field-guided quad layout: singularity/valence placement, constrained patch decomposition, integer-grid extraction or validated recombination, boundary parity and quality optimization. A prepared cross field that does not influence topology cannot pass. |
| A `meshing/_hex_generation.py`, `_hex_dominant.py`, `_hex_cut_preparation.py`, `_hex_cut_generation.py`, `_hex_frame_map.py`, `_hex_frame_records.py` | Preserve dual conversion and structured mapped routes with truthful names. Complete both feature-aware balanced-grid/cut-cell all-hex closure and frame-field/block/integer-grid extraction research routes. Hex-dominant construction reports actual mixed transitions; all-hex requests cannot fall back to tets/pyramids. |
| A `meshing/_polyhedral_generation.py`; A `providers/_native_polyhedral.py`; E `geometry/_triangulation.py`, `discretization/_polygon_geometry.py`, `_polygon_query.py`, `_cell_complex.py`, `_cell_mesh.py` | Restricted Voronoi/power pieces over nonconvex/cavity/material domains, reciprocal oriented faces and source/site/component ancestry. Join only legal same-region pieces. Preserve exact/bounded coordinate maps and consumer decomposition/conditioning. |
| E `discretization/_tensor.py`, `_tensor_entities.py`, `_tensor_index.py`, `_hexahedral.py`, `multiblock/_core.py`, `_interpolation.py`, `_constraint_extension.py`; E spatial/AMR balanced-tree owners only where required | Reuse current layouts, oriented templates, explicit interfaces and balanced spatial topology. No second octree, AMR executor or structured mesh carrier. Nonconforming interfaces use actual existing mortar/transfer, never pretend conforming gluing. |

**Approved archived-ridge cap policy:** the user explicitly permits conforming cap subdivision for the original archived ridge case, with matching layer/core refinement. Preserve the authored source geometry, existing node positions, physical layer heights and growth, materials, and all quality targets. New cap nodes and facet subdivisions require exact source ancestry and conforming interface coverage. This is an explicit opt-in; other fixed-cap requests retain their fixed topology. The original fixed-cap request is geometrically infeasible: its unit cap edge borders a strip of height approximately 0.03, forcing an incident tetrahedron radius-to-shortest-edge ratio greater than 8.33 against the requested bound of 2, even after noncap-only refinement.

**Approved unit-box boundary policy:** the user explicitly permits conforming boundary subdivision for the original unit-box layer/core workflow, preserving the original source geometry, existing vertices, layer heights and growth, materials, and quality targets. Exact binary64-rational analysis found 128 violating noncap side facets and no violating cap facets, so prefer subdivision of noncap side facets while keeping the cap fixed; coordinated cap refinement is permitted only if needed. Subdivision must preserve exact source entity and occurrence ancestry and independently prove source and interface coverage. This approval is specific to the unit-box request, not a general change to fixed-boundary requests.

**Approved curved-hybrid boundary policy:** the user explicitly permits conforming boundary remeshing for the original curved periodic layer/core workflow, including matched cap refinement when necessary. Preserve the exact mapped source geometry, all existing vertex positions, periodic and material interfaces, physical layer heights and minimum thickness, visibility, stretch, the original lower growth bound, and unchanged mesh quality limits. The user additionally approved raising only the physical growth upper bound to 1.001: exact nearest-wall analysis of the four retained original columns establishes growth approximately 1.000798388786, above the original upper bound 1.0000000001. Reprove the original mapped root charts, source occurrences, interface coverage, periodic action and actual physical controls for every refined column. These opt-ins do not alter other fixed-boundary requests.

**Approved input-forced quality exemption:** the user explicitly approved that canonical tetrahedral acceptance exempts only quality rows proved to be forced by the authored input. A row is exempt only when its cell is exactly incident to an authored ridge whose exact input dihedral angle is below the requested dihedral target, and the failing quantity is that ridge dihedral, which no mesh can raise above the input angle. Each exemption is reported in the acceptance evidence with the ridge, its exact input angle and the achieved value. Every other quality row stays a hard gate. Rows near a small input angle that are not provably forced (for example radius-edge rows or dihedrals away from the ridge) are refinement defects, not exemptions. The original small-angle wedge (input 7.153°, target 10°) is the motivating case.

The user subsequently extended this approval to radius-edge rows forced by an authored face angle. A radius-edge row is exempt only when the cell has a face that triangulates an authored planar source facet at one of that facet's vertices whose exact interior angle α satisfies 1/(2·sin α) above the requested radius-edge bound. Every triangle of that facet incident to the vertex then has an angle of at most α, so it and every tetrahedron on it exceed the bound regardless of the mesher. The exemption covers only that radius-edge row. The record reports the source vertex, the exact angle, the forced lower bound 1/(2·sin α), the face triangle's own ratio and the achieved cell ratio. Radius-edge rows on cells without such a face remain hard findings.

**Approved shape-target strength:** the user approved an explicit strength for native tetrahedral shape targets (radius-edge bound and minimum dihedral), mirroring `SizeControlStrength`. Hard remains the default: today's gate plus the approved input-forced exemptions. A soft request still refines and improves toward the target, but publishes every unmet cell (criterion, achieved value, reason) as evidence instead of refusing. Validity, determinant, orientation, size, source-fidelity, coverage and physics gates stay hard regardless. An original profile may declare soft only where the donor source authored the target as an aim (published as unmet evidence, not gated), and each receipt cites that donor source. Soft strength enters request and result identities; hard requests keep their identities.

**Approved bounded carrier deviation for nested refinement:** the user approved
keeping exactly nested refinement when a boundary vertex's parents are rounded
source points. A created boundary vertex's exact source point is the exact
stencil combination of its parents' exact source points, which lies exactly in
its declared stratum. Its binary64 carrier keeps the nested position and may
differ from that source point by a recorded deviation bound. The bound is
propagated exactly from the parents' bounds plus the new rounding and stored
as `carrier_deviation` in the PLC source-point records. Certification checks
the carrier against that bound instead of requiring the correct rounding;
stratum membership stays exact, and exact nested transfer is preserved. Source
points without parents keep the correctly rounded contract. Stored records and
dependent identities change for refined vertices.

For planar bilinear boundary faces crossing distinct authored source strata,
use exact whole-face image evidence and separately identified physical overlap
pieces. Their target support is the pullback through the full bound face map,
not an affine virtual reference triangle. Extend the canonical source-fidelity
record and the typed support variants in `target_reference_pieces`; update
issuance, association validation/transfer, currentness, and archive replay
together. Prove source and target coverage without merging coplanar source
identities or setting false per-triangle correspondence errors to zero.

**General quad/hex research acceptance is mandatory and separate:** positive non-sweepable parts, multiply connected domains/cavities, curved feature intersections and material interfaces; whole-cell positivity, global embedding, exact required family, fidelity and quality. Trace frame singularities, boundary parity, decomposition and map inversion failures to exact obstructions. Develop and independently test transition templates and orientation/coverage. Dual subdivision, box/sweep success or all-negative admission does not close the general gate. If a required positive remains unresolved, report that gate uncompleted; do not rename the route or narrow the corpus.

**Approved validity-proof identity change:** use canonical directed interval
enclosures of the exact rational Jacobian and determinant when they prove the
existing validity obligations, retaining the existing exact method when
unresolved. Source geometry, coordinate realization, scientific thresholds,
failure semantics, and requested quality remain unchanged. Proof bounds and
certificate identities may change; regenerate affected certificates, archive
evidence, and qualification records. Do not imitate historical proof payloads or
describe this as identity-preserving optimization.

**Approved cross-platform identity policy (exact IDs only):** identities and
decisions that archive restore or acceptance recomputes use only exact or
decided data: meshes, topology, exact witnesses, policies and pass/fail
decisions together with their exact evidence. Floating-point diagnostics
(quality metrics, condition numbers, sampled minima, residuals) that such
recomputation would re-derive through platform arithmetic are recorded as
non-identity evidence. Values that are only hashed as given are stored with the
archive and restore unchanged. Acceptance gates that restore recomputes decide
exactly, so a stored result is accepted or refused identically on every platform
and archives restore across operating systems. The gate is a cross-OS restore
matrix over every result family; any cross-OS refusal is a required fix.
Certified enclosures are sound on every platform without relying on libm, XLA
or BLAS accuracy. Generation is deterministic on each platform but need not be
byte-identical across operating systems. Each platform produces a valid,
certified mesh, and cross-platform qualification compares certificates,
decisions and exact data rather than mesh bytes. Affected certificate, result
and model identities change once as a clean cutover: archives are rewritten and
there are no compatibility reads. The user first approved a stronger policy
requiring byte-identical generation across operating systems, then replaced it
with this one. Evidence that prompted the decision: XLA quality diagnostics
fingerprinted into result identities, BLAS-dependent SVD condition numbers in
T-spline certificate identities, and a Linux-written archive refused on macOS
at an identical native build.

**Approved self-contained audit reports:** `CellMeshAuditReport` stores its
`CellMeshAuditPolicy`, not only the policy identity, so an uncertified meshing
result made under any audit policy restores in an independent process by
recomputing its audit under that stored policy. `report_id` is unchanged
because it already binds the policy identity; restore verifies the stored
policy against that identity. The audit report's PyTree and archive
representation change once as a clean cutover: archives holding audit reports
are rewritten, with no compatibility reads. The user chose this over refusing
to archive such results, which would have left a valid public result without
lossless persistence.

**Approved implicit-route fill size semantics:** the implicit route's volume
fill interprets a requested hard size h as the donor did: the cell circumradius
target is h itself, with no edge-length conversion (√6/4) and no 0.5·maximum
radius cap, and the maximum size is enforced at acceptance (the donor's
tolerance on the maximum). The PLC route keeps its edge-length convention. The
convention is an explicit, closed fill parameter declared by the route, part of
the fill identity, never inferred. Implicit result identities change once.
Measured on saved carriers: torus L0 fills drop from 5956 to 1766 cells, median
edge from 0.84h to 1.15h, maximum edge at most 1.96h within the donor's 2.25h
acceptance, and the first metric epoch refines as the donor designed.

**Approved certified implicit volume ladder:** the native zero-set remesher keeps
its sagitta-centered carrier vertices, which have 3–8× smaller absolute volume
error than inscribed carriers because the outward and inward caps nearly
cancel. Since that cancellation is not monotone in the tolerance, the ladder
replaces the original "|V − V_exact| strictly decreases" test with a certified
one: at every level the exact volume error lies inside the certified enclosure
V(r ± ε) the donor already computes, and the enclosure strictly shrinks. The
certified fidelity bound must still strictly decrease. The user chose this over
donor-faithful inscribed carriers, which kept the original test but had larger
errors and two further defects (sphere L2 boundary-edge median, torus L1 fill
quality).

**Approved archive representation changes:** each is a one-time clean cutover
with no compatibility reads.
1. Omega_H adaptation evidence stores the canonical worker record and the source
   field identities its evidence identities bind, so Omega_H results archive and
   restore by recomputing those identities from stored inputs; evidence
   identities are unchanged.
2. Wall-clock telemetry (`MeshAdaptationResult.elapsed_seconds` and any other
   runtime timing outside scientific identity) is not persisted in model
   archives, so archive bytes are deterministic on each platform; restored
   results carry no wall time, and timing stays in run receipts and benchmarks.
3. Model archives share equal arrays across different parents: the first
   occurrence in canonical order is written, later equal arrays are content-keyed
   references with a byte-equality collision guard, and readers refuse repeated
   equal array definitions so each archive has one canonical form. Archive
   content identities, result identities and report identities are unchanged.
4. The aggregate archive byte limit counts bytes once per stored owner, as
   element counts already do, adopted only after measurement proves restore
   memory scales with stored rather than logical bytes; otherwise the result
   structure drops duplicate references instead. The limit value is unchanged.
   Measured: torus restore holds 202 MB of distinct arrays against 281 MB
   stored and 2.25 GB logical, so per-owner charging was adopted.
5. Large static integer and digest fields that were JSON lists in the archive
   manifest (`MeshingScope.unqualified_entity_ids`,
   `CellValidityCertificate.cell_map_digests`) are stored as integer and byte
   array columns through the existing column recipes; identities are
   unchanged. A global manifest-limit increase was rejected for this change.
6. The exact coordinate bank of `CellGeometrySpec.exact_coordinate_bank` and
   `_ExactCoordinateBankMap.exact_coordinates` is an immutable integer-dyadic
   bank (int64 odd significand and int16 exponent per entry, a sparse overflow
   table for wider significands and a sparse table of non-dyadic
   `RadicalElement` entries) instead of per-entry `Fraction` tuples. Only the
   static PyTree payload type changes: fingerprints, identities and archives
   (the existing `"exact"` column projection) stay byte-identical and earlier
   archives restore. `_RetainedCoordinateBank.exact_coordinates` holds the same
   owner. Measured on the STEP external-combined curved geometry: 132.6 MB in
   3.2M objects becomes 9.77 MB, and its identities and re-packed archive bytes
   are unchanged.
7. HP adaptation results retain a compact source-replay record rather than a
   complete prepared source epoch. It records the source topology and geometry,
   source epoch/mesh identities, actual current field element specifications,
   the adaptation request and original refinement/coarsening evidence. Restore
   rebuilds derived worksets, constraints and discretization through their
   canonical owners, replays the requested transition, and verifies geometry,
   lineage, target state and field transfer. Lost caller degree-offset intent
   must not be inferred from clamped target degrees. HP result identities and
   HP PyTree/archive representation change; non-HP result representation is
   unchanged. Existing scientific topology, geometry and lineage identities
   remain authoritative.

**Resource budget policy:** solver, factorization, Krylov and Hodge memory
budgets are explicit plan inputs and part of plan identity. Library defaults
remain documented constants for unbudgeted calls; they are never derived from
host memory, so a refusal never depends on the machine. Every qualification
profile declares its execution-memory budget from its run envelope and passes
it to every solve and factorization in the profile, writer and independent
reader alike. SPD operators use the canonical symmetric factorization. Native
searches that can grow without a structural bound (for example exhaustive
Steiner-free suppression in PLC recovery) take a declared node and byte budget
and refuse through their typed failure with search-count evidence.

The required `archived-layer-core` material profile separately declares a
user-approved 256 MiB manifest allowance: its measured canonical cycle metadata
occupies 191,299,479 bytes. The normal writer and independent cold reader derive
admission from the same profile declaration. Its 4 GiB container, aggregate, and
member byte limits, original element/axis limits, 4,097-member limit, and all dtype,
header, nesting, and central-directory checks remain unchanged. Other profiles retain
64 MiB manifests; library defaults, archive representation, and scientific
identities do not change.

Implicit chart certification schedules independent face and piece continuations
in 1,024-lane worksets. This bounds active numerical scratch and interval-query
buffers, not the total number of faces, subdivisions, or proof rounds. Required
input topology and completed proof records retain their actual sizes. Per-piece
mathematics and response sequences, output row order, crease refusals, and
binary64-resolution decisions remain unchanged. Consumed round replies are
released before the next refinement round.
Local injectivity proofs prepare the canonical float64 BVH once per refinement
generation for root-box collisions and reference-box chord neighborhoods. Queries
include touching boxes and restore original piece ordering before every reduction.
Query batches bound the possible pair working set; a single genuinely dense
neighborhood retains its actual output size. This replaces repeated all-piece
scans without changing determinant enclosures, reference-corner widening, or
blocking-piece refinement.

When the narrowed root-distance interval still straddles the requested tolerance,
certification also uses a centered mean-value enclosure of squared displacement.
Its gradient is `2(R-P)·(DR-DP)`, using the existing implicit root differential
and a certified center root on the same branch. This retains derivative
cancellation near a stationary distance, avoiding massive subdivisions caused
by independent coordinate-box bounds. Both valid enclosures are intersected;
an undecided differential does not manufacture a bound. Source geometry,
tolerances, strict refusal, and binary64-resolution handling remain unchanged.
This stronger mathematical proof can tighten reported bounds and change the
internal proof subdivision; it is distinct from the identity-preserving
workset and neighborhood scheduling changes above.

## 11. Geometry and field transitions, periodicity, distribution and consumers

### 11.1 Coordinate and field transfer

| Files | Action |
|---|---|
| E `meshing/_topology_edit.py`, `_lineage.py`, `_bisection.py`, `_local_metric.py`, `_adaptation.py`; A `_lineage_records.py`, `_entity_keys.py` only where needed | Generalized `CellTopologyEdit`, fixed/mixed/polyhedral blocks, oriented shared-face witnesses and stable IDs. Keep lineage assembly once. Array positions and reusable local slots never become scientific IDs. |
| E `discretization/_cell_geometry.py`, `_cell_mesh.py`, `_cell_complex.py`, `_topology.py`; A `_cell_geometry_transition_types.py`, `_cell_geometry_transfer.py`, `_nested_reference.py`, `_nested_reference_types.py`, `_reference_subdivision.py`, `_reference_cell_types.py`, `_source_geometry_realization.py`, `_cell_geometry_predecessor.py`, `_chart_certification.py`, `_mapped_geometry_measure.py` | Exact nested map composition/restoration, bounded reconstruction where explicitly requested, shared geometry-node continuity and current geometry revisions. Do not infer curved geometry from vertex stencils. |
| E `meshing/_association.py`, `_interface_binding.py`; A `_association_state.py`, `_association_composition.py`, `_association_routes.py`, `_surface_association_transfer.py`, `_implicit_association_transfer.py`, `_mapped_reference_association.py` | Preserve source stratum closure on splits/collapses and exact source occurrence/region sides. Reprojection verifies a known source class; it does not invent semantic correspondence. |
| E `discretization/fem/_topology_transfer.py`, `_de_rham.py`, `_form_elements.py`, `_form_reconstruction.py`, `_reference.py`, `_reference_operator.py`; A `_exact_form_moments.py`, `_mapped_form_transfer.py`, `_surface_chart_compatible.py`, `_surface_chart_transfer.py`, `_sphere_chart_transfer.py` only for their distinct mathematics | Current general-order form basis remains canonical. Implement actual H1/DG/H(curl)/H(div) actions with reference maps, Piola transformations and full orientation actions. Preserve reproduction, commuting and dual/adjoint distinctions. Do not restore obsolete lowest-order element owners. |
| E `geometry/_supermesh.py`; E `discretization/finite_volume/_automatic_remap.py`, `_unstructured_remap.py`, `_unstructured_motion.py`; A `_remap_csr_view.py`, `_remap_evidence.py`, `_unstructured_types.py` as justified | Physical overlap CSR, measures/errors, required moments, source/target coverage and conservative content transport. No global dense overlap matrix or conservative claim from constant-preserving interpolation. |
| A `geometry/_volume_material_map.py`, `_implicit_cavity_atlas.py`, `_sphere_material_atlas.py`; A `meshing/_projected_material_map.py`; A `discretization/_exact_plc_geometry.py`, `_exact_rational_geometry.py`, `_exact_algebraic_geometry.py`; A `finite_volume/_material_schedule.py`, `_material_chain_action.py` | Chronological curved material-following maps and compact reusable actions where required. Preserve real removed/created cell geometry and material correspondence; remove publishing-request limits/credits. Ordinary Eulerian remap does not retain every prior edit. |
| E `discretization/_transfer.py`, `_state_transfer.py`, `_topology_epoch.py`, `_distributed_field.py`, `_field_query.py`, `_views.py`; E `fem/_point_interpolation.py`, `_generic.py`, `_local_provider.py`, `_geometry_motion.py`; E `equations/fem/_lowering.py`, `_worksets.py` | Consume the actual geometry and field transition; retain heterogeneous worksets and stable compiled kernels. Reuse numeric coefficient/table banks rather than rebuild source×target Cartesian products. Import no general donor compiler/field-policy rewrite. |

**Material-map dispatch cutover:** change donor `_adaptation._prepare_volume_material_chain`, `_lineage.CellMeshTransition`, `_projected_material_map.py`, `finalize_native` and the accepted-state/field dispatch together. They currently hard-code a discovered single-region `ImplicitProjectionBoundarySource`, and `finalize_native` sets the ordinary transfer to `None` whenever any material chain exists. Replace those nominal special cases with the actual validated source-map/cavity capabilities required by each implemented implicit, CAD/mapped, PLC and multi-region route. Select or compose transfer **per field's declared semantics**: chronological material-following content, Eulerian physical overlap, nested/Piola FE transfer, or explicitly nonconservative interpolation. One material field must not disable transfer for unrelated nodal, Eulerian or compatible fields. Missing overlap/source-domain coverage or a missing required field action rejects the complete transaction; no zero fill or extrapolation. A mandatory witness commits one edit carrying material content plus an Eulerian and a compatible FE field, checks conservation and commuting obligations independently, then repeats after fresh-process restore. Keep source-specific discovery identity where it is a real premise, not as a universal requirement of every material map.

### 11.2 Periodic and distributed execution

| Files | Action |
|---|---|
| A `discretization/_periodic_topology.py`, `_periodic_exact.py`, `_periodic_geometry.py`, `_periodic_measures.py`; E `_periodic_cell.py` only for shared lattice semantics | One quotient owner: representatives, integer winding, exact lattice/isometry, orientation and original source identity. This is distinct from existing particle/spectral lattice use. |
| A `meshing/_periodic.py`, `_periodic_constraint.py`, `providers/_native_periodic.py`, `_native_periodic_algebraic.py`, `_native_periodic_constraints.py`, `_native_periodic_size.py` | Generate and adapt quotient complexes with orbit-wide edit closure. Preserve distinct winding entities even when endpoint representatives coincide. |
| A `geometry/_periodic_embedding.py`, `_periodic_affine_degree.py`, `_periodic_image_frame.py`, `_periodic_restriction.py`, `_periodic_mapped_restriction.py`, `_periodic_power_preparation.py`, `_affine_mapped_contact.py` | Geometry/coverage on the actual quotient and lifted neighborhoods; no post-hoc coordinate pairing. |
| E `meshing/_distribution.py`, `_device_adaptation.py`, `_device_status.py`; A `_distributed_generation.py`, `_distributed_planar_generation.py`, `_device_generation.py`, `_distribution_forest.py`, `_distribution_migration.py`, `_migration_execution.py`, `_restart_distribution.py` | Owner-local initial generation/adaptation and collective accepted-state commit. Expand neighborhoods when cavities/search exceed a halo; no presumed one-ring completeness. Stable semantic conflict priorities and neighbor exchange; no mandatory global mesh/candidate gather. |
| A `phydrax/graph/_partition.py`, `_distributed_partition.py`; E graph facade and `_facet_adjacency.py`; E `meshing/_metis.py` | Canonical native graph partitioning and owner-local refinement; preserve explicit METIS comparison, not fallback. Report cut/load/migration observations and actual feasibility. Remove work-unit admission but retain user-declared load-balance/target-share semantics. |
| E `meshing/_organization.py`, `_assembly.py`; A `_collective_geometry.py`, `_collective_organization.py`, `_collective_failure.py`, `_mesh_part.py`, `_publication_lowering.py` only for true owner-local data/result views | Compact distributed result views with canonical IDs and accepted geometry/field roles. Do not retain certificate/receipt authentication graphs or duplicate placement/ownership history. |

Exact resolution of uncertain device predicates is a batch barrier on the owning host before mutation. Hybrid host/device execution is reported honestly. Implement parallel independent cavity batches only after conflict/ordering and rollback invariants hold; keep serial execution as an independent reference. Real multi-process/multi-device qualification is separate from forced-device tests.

### 11.3 Overset, solver-aware design and persistence

| Files | Action |
|---|---|
| A `meshing/_overset.py`, `_overset_exchange.py`; E `_assembly.py`, `_coupling.py`, `_motion.py`; E `discretization/finite_volume/_unstructured_overset.py` | Native hole cutting, fringe/donor search and moving refresh. Deterministic donors cannot be invalid/fringe receptors; expose orphan coverage. Reuse PreparedFieldQuery for interpolative action/transpose. Conservative overlap transport is a separate route with coverage and content evidence. Extend owner-local search/communication where required, not merely sparse coefficient exchange. |
| A `meshing/_decision.py`; E `_proposals.py`; E `discretization/fem/_hp_runtime.py`, `_hp_solver.py`, `_spectral_hp_completion.py`; E IGA `_adaptive.py`, `_transfer.py`, `_certificate.py` | Scientific physical-error/QoI decisions among h, p, geometry order, metric and relocation using existing estimator/solver owners. Replace donor compulsory `DecisionBudget` with a scientific `MeshAdaptationObjective` (new symbol in this owner): tolerance/quantity and explicit numerical admissibility. Measured cost ranks feasible alternatives; no mandatory wall/memory/DOF admission ceilings. Existing explicit target-complexity controls remain mathematical objectives, not an execution quota substitute. |
| E `phydrax/solver/_finite_element_adaptivity.py`, existing FV topology-event consumers; E `applications/solid_mechanics/_topology_reanalysis.py`, `applications/neurofluid/_model.py` | Commit mesh, geometry, fields, material/history, solver refresh and ownership through existing accepted-state/CompositionRebind. Reject failed reanalysis with unchanged accepted state. Preserve current physics, not only mesh arrays. |
| E `phydrax/discretization/iga/_geometry.py`, `_basis.py`, `_actions.py`, `_manifold.py`, `_cut.py`, `_interfaces.py`; E contact `_surface.py`, `_iga.py`, `_implicit_geometry.py`; E meshfree `_surface_geometry.py`, `_surface_transfer.py` only at changed seams | Verify existing consumers against canonical new source/map data. Do not silently broaden unsupported contact/IGA/meshfree representations; implement each required new consumer contract rather than route through an incorrect corner/triangle surrogate. |
| E `meshing/_interop.py`; N `meshing/_model_codec.py`; E `phydrax/_model/_artifacts.py` narrowly; E lifecycle `_archive.py`, `_array_artifact.py`, `_distributed_checkpoint.py`, `_restart_topology.py` only for missing scientific roles | Register exact source/mesh/geometry/field/history/periodic/distribution recipes with generic model/array archives. Fresh-process decode reconstructs canonical constructors, validates restored Strict contracts, and verifies/recomputes necessary mathematics. No live template/issuer requirement. |
| X donor lifecycle `_meshing_checkpoint_binding.py`, `_meshing_field_records.py`, `_meshing_registration.py`, `_meshing_source_families.py`, `_meshing_source_records.py`, `_meshing_sources.py` as a subsystem | Extract only indispensable source/field codec definitions into the existing/domain codec owners above. Do not import request-history replay, paid warm/cold preparation, resource receipt bindings or a second checkpoint format. |

Fixed-topology geometry/weight/control derivatives are supported only under their qualified regularity and branch conditions. Boolean/topology/refinement/donor/acceptance changes remain discrete events. A field-action transpose is not automatically a derivative through geometry remeshing. Tests must distinguish primal conservation, raw transpose, Hilbert adjoint and fixed-event sensitivities.

## 12. Verification, benchmarks, documentation, and execution order

The detailed acceptance and execution matrix below governs implementation; no listed command has been run during planning.

### 12.1 Scientific workflow coverage

Use the donor's 13-family/114-profile registration as a source/case inventory. Preserve physical inputs, positive/negative intent, quality/fidelity targets and required feature combinations. Remove administrative resource-envelope requirements. Add T-spline CAD-to-analysis coverage explicitly. Do not make every case repeat every field/archive/provider combination; test distinct consumer contracts once with meaningful state/geometry coverage.

| Family | Required positive witness and edges |
|---|---|
| Curves/planar/surfaces | Junction network; hole/embedded features/thin channel; closed sphere/torus and open trimmed sheet; complete source association, topology and physical size/fidelity. |
| Constrained solids | Disconnected/nonconvex/cavity/material/triple-junction/small-angle/sliver-rich PLCs, fixed and conforming boundaries; independent coverage and an actual FE/FV solve. |
| Implicit | Sphere, hidden component, offset torus and adaptive changes; certified source fidelity, unresolved tangency/singularity semantics, source-faithful adaptation and physical convergence. |
| Labeled images | Two-slab and oblique/sheared/reflected multi-material junctions; exact declared interpretation, interfaces and transport inventory/flux balance. |
| Reconstruction | Noisy lobed sphere, torus, ellipsoid and imperfect envelope; explicit approximation/repair semantics, components/genus and original-source deviation. |
| CAD | Native construction/query/partition/serialize/read/mesh/curve/solve; trimmed rational and non-rational intersection branches, seams/poles/occurrences/units; no external engines. |
| T-splines | Full section 8 witness, including real local refinement, nonuniform weights, CAD trim/query/mesh, qualified PDE, AD and native/external exchange distinctions. |
| Periodic | Skew lattice/material interfaces, multiple winding entities with shared representatives, orbit-synchronized refine/coarsen/metric changes and compatible field transfer. |
| Hybrid layers | Convex/concave/curved/periodic walls, physical first-layer height, collision policy, fixed cap/core, high-order mixed adaptation and conservative/compatible state. |
| Quads/hexes | Structured/swept/multiblock and dual conversion separately; genuine field-aligned quads and nonsweepable/cavity/curved/material all-hex research corpus. |
| Polyhedral | Nonconvex/material-L/stretched/periodic sources, site regeneration and real FV/VEM consumption, independent measures and conservative nonnested remap. |
| Distributed | Owner-local generation, refine/coarsen/repartition, all required fields/history/RNG/clock, changed-placement fresh restart and continued solve; no hidden global gather. |
| Moving overset | Hole/donor coverage and accepted moving PDE update; interpolation vs conservative overlap, orphan failure and complete rollback. |
| Design | Independent derivative oracle and fresh physical reanalysis; geometry-error versus field-error decisions, analytic/uniform/learned proposals and event-boundary invalidation. |

### 12.2 Test file actions

- **Current API/consumer regressions:** update `tests/unit/meshing/test_contracts.py`, `test_scopes.py`, `test_proposals.py`, `test_provider_worker.py`, `test_implicit_provider.py`, `test_compartment_meshing.py`, `test_audit_quality.py`, `test_metric.py`, `test_lineage.py`, `test_distribution.py`, `test_interop.py`, `test_assembly.py`, `test_coupling.py`, `test_boundary_layer.py`, `test_cad_curving.py`; current CAD/planar/process tests and Neurofluid consumer tests. Remove obsolete API/quota expectations, not scientific assertions.
- **Native kernel tests:** retain/adapt existing `test_predicates.cpp`, `test_cdt2d.cpp`, `test_delaunay2d.cpp`, `test_delaunay3d.cpp`, `test_clip.cpp`, `test_capi_guard.cpp`; add donor `test_cavity.cpp`, `test_getters.cpp`, `test_plc3d.cpp`, `test_refine3d.cpp`, `test_improve3d.cpp`, `test_curved_collapse.cpp`, `test_intersections.cpp`, `test_tetrahedra_arrangement.cpp`, `test_power_diagram.cpp`, `test_periodic.cpp`, `test_surface_mesh.cpp`, `test_level_set.cpp`, `test_mixed_refinement.cpp`, `test_quadratic_bounds.cpp`, `test_quadratic_validity.cpp`, `test_interval_program.cpp`, `test_graph_partition.cpp`, `test_parallel.cpp`, `test_parallel_kernels.cpp`. Preserve `tet_mesh_fixtures.hpp` and independent reference clipping cases.
- **Native CAD/codec tests:** port scientific bodies from `tests/unit/geometry/test_native_profile_construction.py`, `test_brep_curve_carriers.py`, `test_brep_model_validation.py`, `test_brep_intersection.py`, `test_native_cad_boolean.py`, `test_brep_sewing.py`, `test_brep_analytic_closest.py`, `test_brep_query_boundaries.py`, `test_brep_analytic_blends.py`, `test_brep_defeature.py`, `test_brep_offset_evidence.py`, `test_recipe_names.py`; `tests/unit/interchange/test_cad_native.py`, `test_cad_brep_text.py`, `test_cad_export_evidence.py`, `test_step.py`, `test_iges.py`, `test_step_repairs.py`, `test_step_metadata.py`, `test_dxf.py`; surface arrangement/Boolean tests in their existing directory.
- **Generation/adaptation tests:** port `test_native_provider.py`, `test_native_curve_publication.py`, `test_curve_placement.py`, `test_native_planar_construction.py`, `test_native_planar_exact_ancestry.py`, `test_native_surface_generation.py`, `test_native_volume_generation.py`, `test_native_volume_support.py`, `test_native_cad_volume.py`, `test_implicit_analytic_volume.py`, `test_adaptive_implicit_volume.py`, `test_image_volume.py`, `test_surface_envelope.py`, `test_tetra_metric.py`, `test_tetra_metric_curved.py`, `test_surface_metric.py`, `test_surface_metric_arcs.py`, `test_mixed_adaptation.py`, `test_polyhedral_adaptation.py`, `test_cell_geometry_transfer.py` at their owning geometry/meshing paths. Do not retain resource-only cases mixed into those files.
- **Family/advanced tests:** port `test_native_layer_core.py`, `test_quad_generation.py`, `test_hex_generation.py`, `test_hex_cut_preparation.py`, `test_hex_cut_generation.py`, `test_hex_frame_map.py`, `test_structured_generation.py`, `test_multiblock_generation.py`, `test_sweep_generation.py`, `test_nonsweepable_family_source.py`, `test_native_polyhedral.py`, `test_native_periodic.py`, `test_native_periodic_size.py`, `test_native_periodic_vem.py`, `test_native_overset.py`, `test_native_overset_geometry.py`, `test_native_overset_fields.py`, `test_native_overset_conservative.py`, `test_native_distributed.py`, and `test_solver_aware_decisions.py`. Add independent positive cases for missing general algorithms; do not repurpose refusal as coverage.
- **Shared numerical proofs/transfers:** port `tests/unit/geometry/test_domain_coverage.py`, source/global embedding and source-fidelity tests at their actual owning paths, `tests/unit/meshing/test_volume_material_map.py`, `tests/unit/discretization/test_material_chain_action.py`, compatible mesh-transfer tests and `tests/unit/linalg/test_exact_field.py`. Extend current supermesh, FE de Rham/Whitney/H(div), unstructured motion/remap, spectral-matrix and spline tests when their owners change.
- **New gaps:** section 8 T-spline tests; `tests/unit/meshing/test_mesh_state_cold_restore.py` must launch a Python 3.13 writer process, author source/high-order map/fields/history and archive, wait for that process to exit, then launch a separate isolated consumer with no producer object, issuer or caller-held template. The consumer reconstructs, validates and continues the solve/adaptation; include T-spline exact structure/refinement, periodic state and changed-placement distributed variants where applicable. Merely deleting variables or decoding in the producer process cannot pass. Also add `test_periodic_mesh_ownership.py` for quotient-aware ownership and the focused native spatial-index/exhaustive-reference test in `test_plc3d.cpp`.
- **Optional-provider tests:** migrate Gmsh, FTetWild, MMG, Omega_h, TIOGA, VoroCrust, Manifold/OpenVDB/Poisson tests and missing-provider boundaries. Keep native tests unconditional; real providers run only in explicit installed environments.
- **Delete/do not import:** native `test_bounded_memory.cpp`, `test_execution_batch.cpp`, `test_execution_target.cpp`, `test_parallel_budget.cpp`, quota parts of `test_parallel_policy.cpp`; Python native execution-receipt/request-accounting/managed-envelope/publication-issuance and resource-only tests. Rescue actual ownership lifetime, failed-allocation atomicity, pointer/index/decode safety and cancellation cleanup assertions into their owning safety tests.
- **Typing:** adapt donor `tests/typing/cases/native_meshing.py`, `tests/typing/installed/meshing.py`; add T-spline constructor/evaluation/refinement/CAD/IGA typing cases within the established fixture layout. Remove obsolete quota/receipt types. Preserve deliberate misuse checks and installed-wheel API behavior.

Keep separately collectable consumer contracts. Do not create source-text/import-spelling/wiring tests or collapse scenarios to reduce item count. Because donor tests are being materially reshaped, keep an old→new scientific-contract map in the implementation record, coverage ratchets at changed owners, and deliberate faults for shared geometry/transfer helpers: inverted interior, omitted interface, wrong periodic winding, incorrect Piola orientation, broken overlap weight, stale source, broken spline extraction/weighted prolongation. No count-based coverage claim.

### 12.3 Runtime smoke and benchmark plan

Every completed implementation slice must run its actual changed public path before broader testing. Tests alone do not prove the native CAD/mesh/physics workflow. Engine-free witnesses use the actual installed wheels in a process without optional geometric engines; forced devices are not real accelerator/multi-host evidence.

Reuse `tools/meshing_benchmarks.py` and its case selection. Add missing CAD/T-spline stages there or in the existing spline/IGA benchmark owner, not another generic benchmark framework. `tools/iga_closure_qualification.py` and applicable IGA benchmark tools gain the new T-spline source/refinement/analysis cases. Donor `benchmarks/native_meshing_campaigns.json` is a registration, not measurements; retain only useful scientific case definitions.

Measure separately: decode/source preparation; spatial indices; intersection/Boolean; surface/volume topology; improvement/adaptation; geometric certification; canonical output conversion; transfer preparation/action; PDE preparation/solve. JAX cases also separate lowering, compilation, first execution, warm execution, temporary/output/code bytes and logical retained data. Measure actual process/device peaks with their honest scope; do not relabel logical bytes or allocator receipts as RSS. Benchmark instrumentation is optional observation outside production acceptance.

Scaling varies the controlling source/mesh complexity: entities/trim branches/occurrences, PLC edges/tetrahedra, material/interface complexity, refinement epochs, polynomial degree and extraction-local width, T-spline marked-region size versus total domain, periodic images, graph parts and distributed owners. The T-spline locality campaign must contrast local refinement with full tensor refinement at comparable physical error and preserve the same source geometry.

Baselines: current native primitives where comparable, donor mathematical routes with nonbinding administrative allowances, and explicitly installed external CAD/meshing/adaptation providers on the same physical requests. Record failures instead of comparing a weaker result or silently relaxing quality/fidelity. Compare total error-to-solution cost, success coverage and quality distributions, not only cells/second. No beyond-SOTA claim until the relevant independently validated comparison exists; missing providers/hardware leave that evidence unavailable, not fabricated.

### 12.4 Tools, documentation, examples and packaging

| Files | Action |
|---|---|
| E `tools/meshing_qualification.py` → A/adapt package `tools/meshing_qualification/` | One clean CLI `python -m tools.meshing_qualification`; remove old module, no shim. Retain mathematical scenario bodies; simplify `_controls.py`, `_campaign.py`, `_records.py`, `_runtime.py`, `_fixtures.py`, `_physics.py`, `_descent.py`, `_cold_worker.py`, `_progress.py`, `__init__.py`, `__main__.py`. Remove capacity/timeout/scratch/work/request-credit campaign schema. Keep process isolation only where the test requires it. |
| A/adapt `tools/_meshing_cases.py`, scientific `tools/meshing_qualification/scenarios/` bodies | Keep native lifecycle/surface/implicit/image/reconstruction/CAD/periodic/hybrid/family/polyhedral/distributed/overset/design scenarios and relevant existing bisection/remap/provider comparisons. Do not import package-level `phydrax.meshing.qualification` as a second runtime. Numerical algorithms used by scenarios belong to their canonical owners. |
| A/adapt `tests/data/meshing/campaigns.json`; E/A benchmark registrations | Retain source/case identity, expected outcome and scientific controls. Remove quotas, receipts and administrative completion labels. Add T-spline cases. Regenerate deterministic data only after final API/source choices stabilize. |
| A/adapt `tools/check_native_dependency_isolation.py`; N `tools/meshing_qualification/scenarios/tspline.py` | Set and assert the supported CPython 3.13 interpreter; build/install matching current Python/native wheels in an isolated environment. Register `native-tspline-cad-analysis` in the qualification CLI and isolation driver alongside CAD, source/meshing/physics and two-process restore witnesses. `--all-native` must include these actual positive workflows. Keep base import/missing-meshcore refusal separate from native execution; do not let missing engines skip the native witnesses. |
| E `docs/guides_meshing.md`, `docs/api/meshing.md`, `docs/api/geometry.md`, `docs/guides_differentiable_geometry.md`, `docs/guides_finite_elements.md`, `docs/guides_unstructured_finite_volume.md`, `docs/guides_numerical_interoperability.md` | Scientific pipeline, source/fidelity, genuine family coverage, high-order/transfer/periodic/distribution and derivative scope. Remove old quota/receipt vocabulary and stale examples. |
| E `docs/guides_isogeometric_analysis.md`, `docs/guides_isogeometric_analysis_support.md`, spline/IGA API docs; N `docs/guides_native_cad.md` | Explicit T-spline source/refinement/CAD/analysis/interchange profile, native CAD operations and limits of mathematical support—not logistical caps. Distinguish tensor NURBS, ASTS, arbitrary T-mesh and quad generation. |
| E `README.md`, `CHANGELOG.md`, `mkdocs.yml`, `docs/installation.md`, `docs/guides_discretization.md` and the CAD/interchange guides named above | Describe native default ownership, optional dependencies, breaking API/identity/persistence changes and actual evidence/nonclaims. Do not import donor historical completion/resource prose. |
| E `examples/meshing_native.py`, `boundary_layer_core_mesh.py`, `ale_conservative_remesh.py`, `differentiable_geometry_fem.py`; A/adapt `native_surface_meshing.py`, `native_tetrahedral_meshing.py`, `native_multiblock_meshing.py`, `native_quad_hex_meshing.py`, `native_polyhedral_meshing.py`; N `native_brep_workflow.py`, `native_tspline_cad_analysis.py` | Each demonstrates a distinct native source→mesh/analysis contract without internal budget objects. Keep explicit optional comparison examples such as `cad_high_order_curving.py` labeled and migrate their projection/scope/persistence calls. |
| E root `pyproject.toml`, `uv.lock`; E `native/meshcore/pyproject.toml`, `CMakeLists.txt`, package locator; A legal/format files as needed | Keep current Python 3.13/package/build conventions. Add required numerical sources/Threads; update ABI/header/source identity and matching package pin once coherent. Remove mandatory OCP from core only after every native caller migrates; keep optional OCP/build123d/stubs and external providers explicit. No unrelated dependency upgrades. |
| E `tools/generate_public_api_manifest.py`, `tools/check_public_api_manifest.py`, `tools/generate_capability_inventory.py`, `tools/check_capability_consistency.py`, `phydrax/qualification/_builtin_catalog.py`; generated `docs/data/public_api.json`, `capabilities.json`, `application_portfolios.json`, `capability_closure.json`, `source_absorption.json`, `docs/api/capabilities.md`, `capability_closure.md` | Regenerate with the owning tools after the final cutover, not by copying donor snapshots. New capability tuples need actual positive evidence; T-spline basis availability does not authorize CAD/IGA closure. Preserve existing generated filenames; do not add a second inventory format. |

No automatic source-module import hooks, second type checker, internal format generation suffixes or compatibility aliases. All new Strict classes use the canonical contract/validation conventions. Host/device roles and compiler-sensitive function identities are reviewed before wrapping or moving code.

### 12.5 Verification commands and selection

Commands below are proposed for the implementation worktree after the named files exist and APIs are migrated. Use the repository environment and pinned tools. Select the conservative affected set from changes since the latest merge into `dev`, with semantic consumer analysis, not only textual imports. One ordinary `pytest -n auto` job at a time; provider/topology-isolated runs separately.

```sh
python tools/check_typing.py check
python -m tools.check_installed_typing --python 3.13
python tools/audit_selectors.py
python tools/audit_contract_candidates.py --signatures
python tools/check_public_api_manifest.py
python tools/check_capability_consistency.py
```

Apply configured Ruff/annotation and C++ formatting to touched files before source-identity-sensitive checks. Run changed-surface lint, Strict declaration/import/generated consistency checks; measure changed symbol complexity before/after and decompose by real numerical/lifecycle responsibility, not arbitrary line chopping.

Representative focused Python wave (expand with each landed scientific owner; do not run missing/prospective files):

```sh
python -m pytest -n auto -q tests/unit/meshing/test_contracts.py tests/unit/meshing/test_scopes.py tests/unit/meshing/test_proposals.py tests/unit/meshing/test_provider_worker.py tests/unit/meshing/test_audit_quality.py
python -m pytest -n auto -q tests/unit/geometry/test_cad_partition.py tests/unit/geometry/test_planar_partition.py tests/unit/meshing/test_native_provider.py tests/unit/meshing/test_native_volume_generation.py tests/unit/meshing/test_native_cad_volume.py
python -m pytest -n auto -q tests/unit/discretization/test_tspline_basis.py tests/unit/discretization/test_tspline_local_refinement.py tests/unit/discretization/test_tspline_geometry.py tests/unit/geometry/test_tspline_brep.py tests/unit/interchange/test_tspline_exchange.py tests/integration/test_tspline_cad_analysis.py
```

Run the process-owned cold-restore test separately from xdist/device-topology workers:

```sh
python -m pytest -q tests/unit/meshing/test_mesh_state_cold_restore.py
```

Native build/test in an exclusive worktree-local build directory:

```sh
cmake -S native/meshcore -B .tmp/meshcore-release -DPHX_MC_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .tmp/meshcore-release --parallel
ctest --test-dir .tmp/meshcore-release --output-on-failure
```

Run separate ASAN/UBSAN and TSAN configurations for the changed native ownership/concurrency surface, with distinct build directories. No quota-ledger CTest targets remain. Test injected real allocation failure and no-mutation/no-leak behavior; do not simulate correctness by granting an unlimited quota.

Actual public workflows and independent qualification precede uncontended benchmarking:

```sh
python examples/native_brep_workflow.py
python examples/native_tspline_cad_analysis.py
python -m tools.meshing_qualification --scenario native-corpus
python tools/meshing_benchmarks.py --case native-meshing --resolution 8 16 --repeats 5
```

The installed engine-free gate is mandatory, not replaced by checkout examples. After integrating the driver and its new T-spline/two-process cases, run:

```sh
python tools/check_native_dependency_isolation.py --python 3.13 --all-native
```

The tool builds the matching Phydrax and meshcore wheels when explicit wheel paths are omitted, installs them into its fresh environment, and executes outside the source checkout with source-path/library overrides removed. If using prebuilt wheels, pass the actual matched artifacts through its existing `--wheel` and `--meshcore-wheel` arguments. Verify that CAD, meshing, T-spline CAD/IGA and fresh-process continuation ran; separate missing-meshcore refusal does not count as a positive workflow.

The qualification CLI is rewritten to consume scientific profiles only; remove old resource flags rather than preserve them as ignored options. Run the final scientific matrix after all relevant inputs stabilize. Run full repository tests only if the actual delta affects global collection/configuration, shared fixtures or similarly broad substrates; otherwise use the documented affected matrix. New spline/coordinate/linalg math requires all its real transitive consumers, not merely CAD tests.

After implementation stabilizes, generate once, then check the final public/capability data:

```sh
python tools/generate_public_api_manifest.py
python tools/generate_capability_inventory.py
python tools/check_public_api_manifest.py
python tools/check_capability_consistency.py
```

Then run installed typing, affected import-boundary and executable-doc/example tests, and strict MkDocs build with the configured environment. Do not regenerate repeatedly during incompatible intermediate states. Tests/benchmarks are not executed during this planning task.

### 12.6 Dependency-aware work order

1. **Shared contracts and safety:** public removal/migration map; canonical source/result/edit/identity interfaces; native ABI/ownership; scientific certificate input contracts; corpus physics requirements. Complete usable vertical slices, not placeholder dispatch or no-op limits.
2. **Independent numerical foundations:** native kernel/PLC/source witnesses; shared interval/exact/spectral math; native CAD carriers/queries/intersections; shared T-spline basis/local-refinement owner. One owner per shared file.
3. **First complete pipelines:** native planar and PLC→canonical mesh→FE/FV; native CAD→surface/volume→high-order solve; T-spline→local refinement→CAD→qualified surface analysis. Each earns focused runtime proof before expansion.
4. **Coupled scientific capabilities:** implicit/images/reconstruction; tensor metric and geometry/field transitions; layers/core/high order; periodic quotient; restricted power; structured/sweep/multiblock. Transfer and geometry-preservation contracts are prerequisites, not deferred add-ons.
5. **Research/generalization in parallel with mature routes:** real field-aligned quad construction; both general all-hex approaches; singular CAD coverage; missing periodic/curved/mixed combinations. Integrate only complete algorithms passing required positive cases. These investigations start after their mathematical interfaces are fixed, not at the end of the program.
6. **Scale and simulation closure:** owner-local construction/adaptation/partition, moving overset, accepted-state/field/history migration, source-complete fresh restore, solver-aware design and learned proposals.
7. **Single integrated acceptance:** all required source/family/consumer cases including T-splines, optional comparisons, installed native isolation, typing/audits/docs/generated artifacts and phase-separated scaling/leadership evidence. No phase alone is the complete deliverable.

Critical dependencies: native source truth precedes source-fidelity claims; coordinate-map transfer precedes high-order adaptation; shared T-spline mathematics precedes CAD/IGA adapters; periodic quotient precedes orbit-aware distributed edits; actual geometric overlaps/maps precede conservative field claims; collective accepted-state contracts precede distributed publication; final native binary and Python bindings must match before any integration evidence is trusted.

The approved extensions in section 13 follow blocking performance/correctness
closure. Reuse the stable foundations and focused evidence before starting their
implementation; they do not replace any original positive gate. Extend the final
acceptance matrix once these additions are integrated, rather than repeatedly
running broad qualification on incomplete prerequisites.

Use one compact implementation/verification record: changed ownership surface, exact command/scenario, material tree/backend/dependency identity, result and failing node/scenario. This is development evidence, not a new runtime accounting graph. Reuse passed evidence while its inputs are unchanged. No concurrent unrestricted test jobs or benchmarks under build/test contention.

### 12.7 Completion conditions

- Native advertised operations genuinely execute without external geometric engines.
- No arbitrary native problem-size/work/query/scratch allowance is required to obtain an otherwise valid result; private chunks/buckets do not truncate the problem.
- All scientific requirements remain independently checked, with complete source/region/field identity and honest failure.
- T-splines satisfy every section 8 requirement, including genuine local refinement and the complete same-source CAD/analysis path.
- General quad/hex and singular-CAD required positive gates are either genuinely closed or explicitly reported uncompleted; safe refusal cannot satisfy an admitted positive.
- Every current caller, example, type fixture, codec, generated record and affected consumer uses the clean API. No compatibility aliases, old quota signatures, fake fallbacks or resource-only tests survive.
- Fresh-process and changed-placement continuation reconstruct the real scientific state without original objects or live proof issuers.
- The approved CAD boundary-ancestry cutover carries canonical `GeometryAssociation`
  exact edge intervals and `SurfaceSourceCharts` through PLC refinement to final
  target edges. Coverage/homotopy retains explicit coedge incidence and native
  gauges where unique source binding cannot otherwise be proved. Revalidate the
  actual ancestry; nearby UV shadows or distance bounds are not seam identity.
  Regenerate changed witness/certificate/archive identities, with numerical arrays
  remaining dynamic.
- **Approved periodic witness cutover:** retain the actual exact vertex-coordinate
  witness separately in `PeriodicMeshTopology`, rather than only its hash.
  Preserve the full nonlinear coordinate map. Independently revalidate the
  witness, every incident exact mapped corner, deck actions, and oriented facet
  traces during currentness checks and restoration. Migrate producers, topology
  edits, codecs, tests, and affected artifacts in one clean cutover; no old
  missing-field shim or hash-as-proof. Keep numerical witness leaves dynamic,
  prevent carrier cycles, and preserve the distinction between incidence
  identity and geometry identity.
- Measured performance claims match actual comparable workloads and hardware. Lack of evidence is stated, not replaced with capability counts or cost-model counters.

## 13. Selected Solidcraft-derived extensions

Added by explicit user request after the source review of
[Solidcraft at `147d1cfaddcc6fe824399a58fea36a022278961e`](https://github.com/tangency-labs/solidcraft/tree/147d1cfaddcc6fe824399a58fea36a022278961e).
This is a plan addition, not a claim that the donor was built or its advertised
coverage independently reproduced. Adopt useful mechanisms within Phydrax's
canonical owners, not another kernel, solver, naming system, or document model.

### 13.1 Priority and readiness

1. Complete current safety fixes and blocking performance work first:
   incremental mathematical certification, bounded geometry/archive construction,
   and repeated-compilation removal. Continue independent original capability
   fixes alongside them.
2. Add selected operation-sequence reference cases to the existing qualification
   corpus as their owning routes stabilize. These provide evidence for the later
   extensions, not a separate testing framework.
3. Implement constructive sketch preparation and exact non-bijective recipe-name
   transport as independent extensions of existing design owners.
4. Apply placement-only invalidation/reuse within the existing preparation work;
   introduce no standalone cache subsystem.
5. Extend continuity-driven bridge construction only after the source-jet,
   whole-seam continuity, and corner-compatibility contracts below are explicit.
   This is a later mathematical extension, not an immediate broad G3/fillet claim.
6. Run integrated acceptance and uncontended phase-separated benchmarks after
   these inputs stabilize. No useful extension is considered implemented merely
   because its design, dispatcher, or negative tests exist.

### 13.2 Constructive sketch preparation and branch-aware continuation

**Mechanism:** recognize supported line/circle/arc relations; propagate certified
constructions and rigid clusters; use uncertain/tangent constructions only as
initial guesses; solve the remaining variables through native numerical owners.
Source references:
[construction](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/sketch/src/construct.rs),
[cluster decomposition](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/sketch/src/decompose.rs),
[solver/continuation](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/sketch/src/solver.rs).

**Owners:** extend `geometry/design/_sketch.py`, `_constraints.py`, `_solve.py`,
`_continuation.py`, and `_sketch_certificate.py`; lower factor-local residuals and
incidence through `optim/_residual_graph.py` and `_graph_least_squares.py` where
beneficial. Reuse `RootDOFEvidence`, native nonlinear linearization and linalg
rank/nullspace evidence. General `GraphIR` is not a replacement residual solver.

**Approved representation cutover:** extend canonical `Sketch` with explicit arc
incidence and orientation, including directed major sweeps. Retain preparation
evidence separately from numerical root/rank evidence, and persist sufficient
source-bound checkpoint/transition inputs for independent continuation replay.
Migrate callers and archives and regenerate changed identities; preserve the
existing numerical field partition rather than hiding arrays as static metadata.

**Required behavior and evidence:**
- Keep constructive eligibility, a numerical seed, numerical rank/DOF, residual
  convergence, and root uniqueness as distinct claims.
- Retain explicit reference layout and branch witnesses across edits. Gauge and
  least-motion choices are declared semantics, not hidden regularization.
- Unsupported constructive relations remain in the declared numerical problem;
  failed or non-transversal constructions never become exact elimination facts.
- Test rigid and underconstrained clusters, mirrored solutions, tangency,
  coincident/degenerate loci, redundant versus inconsistent constraints, parameter
  edits, rollback, and independent restoration.
- Compare the reduced route with the unreduced native residual system and an
  independent geometric oracle. Verify JVP/VJP on fixed regular branches;
  invalidate sensitivity at branch/rank/topology events.
- Measure preparation, derivative work, solve cost and retained memory separately
  as constraint count grows. Do not import the donor's dense LM, finite-difference
  Jacobians, local rank thresholds, or work caps.

### 13.3 Exact split/merge lineage and ambiguity-aware selections

**Mechanism:** compose kept/modified/split/merged/generated face and edge origins
through operations; distinguish output pieces by proved source-qualified
construction or adjacency roles when possible.
Source references:
[operation history](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/kernel/src/history.rs),
[naming semantics](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/docs/book/src/dev/naming.md).

**Owners:** extend `geometry/design/_naming.py`, `geometry/_cad_revision.py`,
B-Rep operation provenance/selection owners, existing `AssociationGraph`, and
`meshing/_scope.py`. Current `RecipeEntityName` transport deliberately requires
bijective lineage; extend it rather than introducing another naming vocabulary.

**Required behavior and evidence:**
- Distinguish single-entity intent from set-valued selection of exact descendants.
  A boundary condition attached to an authored wall may follow all proved
  descendants when that is its declared meaning.
- Preserve definition, occurrence, source revision, orientation, region side,
  and operation-role identity. Equal geometry, display labels, sizes, array order
  or nearby centroids do not establish correspondence.
- Single-entity ambiguity, deletion, insufficient provenance and incompatible
  merge semantics remain explicit unresolved/refusal outcomes. Never silently
  reattach a scientific boundary condition using a geometric fallback.
- Verify composed split/merge chains, symmetric ambiguous pieces, independent
  coincident definitions, repeated assembly occurrences, deletion and changed
  parameter branches. Exercise actual boundary/source selections downstream in
  meshing and FE/FV, plus cold persistence and tamper refusal.
- Provenance describes derivation; geometry and field claims still require their
  independent mathematical acceptance. Retain existing bijective behavior.
- The approved side-aware `MeshingScope` cutover retains occurrence-qualified
  region side and orientation through identity, scope operations and persistence.
  Use existing `GeometryAssociation`, `RegionBoundaryEvidence` and mesh region
  data to prove each selected facet's owner/neighbor relation; resolve through
  native FE/FV side integration and supported interface semantics. Do not treat
  an interior face as an ordinary physical boundary. Positive selections from
  both sides of a shared face, permuted storage order, orientation changes and
  cold tamper refusal are required; representation-only refusal is not completion.

### 13.4 Independent operation-sequence reference corpus

**Mechanism:** adapt selected authored recipes combining extrusion, shelling,
blends, patterned holes, parameter changes and interchange. Use independent
analytic or externally produced reference data, not donor success labels.
Source references:
[oracle specification](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/docs/oracle.md),
[Boolean locality and measure checks](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/kernel/src/boolean_guard.rs).

**Owners:** extend `tools/meshing_qualification` scenario/oracle registrations,
owning geometry/interchange tests and existing test support. Preserve deterministic
case identities, physical units, authorship, reference-generation provenance and
applicable fixture licenses. No new command language or corpus schema generation.

**Acceptance:** include thin walls, tangent/coincident contacts, body-splitting
cuts, mixed curved intersections and modifications of already-modified bodies.
Check appropriate combinations of volume, area, centroid, moments, bounds,
topology, region/material identity and interchange loss. Add independent Boolean
monotonicity/algebraic checks and preservation outside a tool's proved influence.
Use exact/enclosed or independently justified numerical oracles; apply deliberate
faults where plausible wrong geometry could preserve a single scalar measure.
Sampled distances/locality checks are diagnostics, never substitutes for continuous
source fidelity, coverage or embedding certificates. Optional comparison engines
stay outside mandatory package execution.

### 13.5 Continuity-driven surface bridges

**Mechanism:** use Hermite control-row constraints to organize boundary position,
tangent, curvature and higher-jet matching.
Source reference:
[bridge formulation](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/kernel/src/bridge.rs).

**Owners:** extend existing `geometry/brep` blend/match operations, shared spline
basis/extraction, differential operators, native linalg and enclosure owners.
Do not add a parallel spline fitter, surface representation or derivative engine.

**Readiness and acceptance:**
- State admitted source families, parameterization/reparameterization, regularity,
  boundary and corner-compatibility hypotheses before implementing a continuity
  order. G1/G2/G3 are geometric claims, not merely equality of sampled derivatives.
- Use analytic/AD/exact source jets with certified enclosures. Do not transplant
  sampled arc-length tables, finite-difference stencils or fixed refinement/shrink
  counts as proof of whole-seam continuity.
- Supply positive curved-source examples for each advertised order, full seam
  verification, denominator positivity, nonfolding/embedding, and incompatible
  corner/singular-source refusals. Check fixed-branch derivatives and native
  archive reproduction; exercise a smoothness-sensitive analysis consumer only
  when its actual formulation prerequisites are qualified.
- Document unsupported corner networks and general rolling-ball cases explicitly;
  the admitted bridge profile is not a claim of arbitrary fillet support.

### 13.6 Dependency-scoped placement reuse

**Mechanism:** explicitly identify placement-only edits whose dependencies admit
reuse of local geometry; unknown dependencies take ordinary evaluation.
Source reference:
[placement reuse](https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/crates/doc/src/placement_reuse.rs).

**Owners:** existing `CompiledGeometry`/`DesignState`, source preparation, exact
placement/occurrence, certification and lifecycle owners. Share this work with the
current prove-once/incremental wave; do not create a document timeline or another
cache framework.

**Acceptance:** explicit dependency/effect contracts distinguish local geometry,
placement, derived queries and coupled scientific consumers. Reuse in-hand
artifacts only against validated inputs and transformation premises; stale or
unknown dependencies re-evaluate. A proved rigid transform can preserve a
component's internal claims, but independently moved instances require renewed
assembly contact/embedding and affected PDE-coupling checks. Compare cold versus
reused outcomes, IDs where semantics are unchanged, derivative behavior, failure
and rollback; measure avoided work on repeated-instance motion without competing
benchmarks. No process-global proof memos or numerical arrays hidden as static
cache metadata.

### 13.7 Explicit exclusions and extension completion

- No wholesale Solidcraft/truck kernel, separate-branch SMLib migration, UI/MCP
  command system, document lifecycle, compatibility format or resource accounting
  layer is imported. Current native owners remain canonical.
- No heuristic reference repair, retry nudges, sampled-success certificates,
  arbitrary work caps or unsupported continuity/rank claims.
- Review file-level and fixture provenance/licenses before copying any material;
  the workspace's MIT-or-Apache-2.0 declaration does not replace vendored/asset review.
- Update authored docs, examples, typing/callers, canonical persistence and generated
  capability data with each accepted extension. Run the actual changed scientific
  workflow and its conservative dependent checks.
- These items extend the plan's acceptance obligations; none is marked implemented
  by this addition. Preserve all original section 12 completion requirements and
  explicitly track remaining extension work rather than silently dropping it.

## 14. Selected TPMS-Generator-derived extensions

Added by explicit user request after reviewing
[TPMS-Generator at `b9b2c841f827229886f13113d30ec893f15dcca9`](https://github.com/meysamznj/TPMS-Generator/tree/b9b2c841f827229886f13113d30ec893f15dcca9)
(MIT). The donor is a single-file PyQt application (`app.py`). It samples six
trigonometric nodal fields (gyroid and its mirror, Schwarz D, Schwarz P, I-WP,
lidinoid) or a string-`eval` custom formula on a float32 grid. It then extracts
`±τ` or `c` isosurfaces by marching cubes. Sheet walls use `τ = ½·k̄·t`, with a
hard-coded family "average gradient" k̄, so the requested thickness is a heuristic.
The constant-padded field is clipped against a box, cylinder or sphere with
Manifold3D Booleans. The donor reports porosity, surface area and specific surface
from the resulting STL, with optional QEM decimation.
Nothing in that pipeline is certified. This section adopts the useful
*scientific product*: periodic lattice metamaterials for AM, FEA and CFD. Every
claim is built on Phydrax's certified owners. Formulas are mathematics; no donor
code is copied.

### 14.1 Gap analysis against current owners

- Interval enclosures already cover `sin`/`cos` and gradients through the traced JAX
  program (`geometry/_interval_enclosure.py`), so TPMS fields are admissible to
  enclosure-complete adaptive discovery (`geometry/implicit/_adaptive_discovery.py`).
- Adaptive and fixed-lattice implicit discovery and implicit volume fill refuse
  periodic axes and `periodic_constraints` (`_discovery.py`, `_curve_discovery.py`,
  `providers/_native_implicit.py`). A periodic quotient TPMS cell is therefore not
  meshable today. This is the foundational gap.
- Sharp CSG with analytic envelopes exists (`geometry/analytic/_operations.SharpCSG`).
  Certified crease curves between an implicit field and an analytic envelope
  surface, plus the trimmed envelope caps they bound, do not.
- Orbit-locked periodic tetrahedral adaptation (`providers/_native_periodic_tet.py`),
  periodic FE transfer and `solid_mechanics.PeriodicHomogenizationCase`
  (Hill–Mandel evidence) exist, but never consume an implicit periodic cell.
- No source owns trigonometric-polynomial (Fourier) level-set families, critical
  levels, or sheet-thickness semantics.

### 14.2 Priority and readiness

1. Completion of the original section-12 corpus remains first; these extensions
   do not displace it.
2. Trigonometric nodal source family and certified critical levels (14.3), and
   periodic adaptive discovery (14.4): the foundations.
3. Periodic quotient surface and volume meshing (14.4), then lattice metrics (14.6).
4. Finite specimens clipped by analytic envelopes with certified creases and caps
   (14.5).
5. Periodic homogenization consumers and fixed-topology design sensitivities (14.7).
6. Qualification profiles, export, documentation and benchmarks (14.8), run in
   the strict tier order of the execution protocol.

### 14.3 Trigonometric nodal level-set sources

**Mechanism:** represent each family exactly as a finite trigonometric polynomial
on the lattice's reduced coordinates `u ∈ ℝ³/ℤ³`, as a canonical table of rational
coefficients times products of `sin`/`cos(2π·n·u)`. The gyroid, Schwarz P/D, I-WP,
lidinoid and the gyroid's mirror image are named tables with cited formulas.
Arbitrary user Fourier tables use the same representation. They are labelled
honestly as *nodal approximations* of the minimal surfaces, not exact minimal
surfaces; mean curvature is reported, not assumed zero.

**Owners:** N `geometry/analytic/_trigonometric_level_set.py` (a `GeometrySource`
with the family table, `PeriodicCell` lattice, level `c` and region selector
`network`/`complement`/`sheet`). Families and region selectors are closed
`Literal` selectors parsed once. Reuse `discretization/_periodic_cell.PeriodicCell`
for the lattice, including anisotropic and skew affine cells.

**Required behavior and evidence:**
- Evaluate on exactly reduced fractional coordinates, so that lattice-equivalent
  points give bitwise-identical field values and enclosures. Periodic orbit
  consistency must not depend on `sin(2π) ≈ 0` rounding.
- Take the exact Lipschitz and Hessian bounds from the coefficient table
  (Σ|a|·‖2πn‖ terms) for interval and Lipschitz enclosure policies.
- Certify critical levels: enclose all solutions of `∇F = 0` in the quotient cell
  by interval Newton/Krawczyk through the existing native interval and nonlinear
  owners, and enclose their critical values. A requested level `c`, or sheet band
  `±τ`, whose enclosure meets a critical-value enclosure is refused as singular.
  Topology changes (for example Schwarz P at `|c| = 1`) are certified events,
  never meshed silently.
- Symmetry facts used as oracles must be proved from the table, not assumed. For
  example, a measure-preserving lattice map `g` of the quotient cell with
  `F∘g = −F` gives an exact volume fraction of ½ at `c = 0`. For Schwarz P that
  is the half-lattice shift `u ↦ u + (½,½,½)`; for the gyroid it is inversion
  `u ↦ −u`.
- No string expression parser and no `eval`. Arbitrary user fields remain JAX
  callables admitted only through the existing interval-extension contract, which
  refuses unsupported primitives.

### 14.4 Periodic discovery and quotient meshing

**Mechanism:** discover the zero set on the three-torus quotient of the declared
lattice. The box cover is identified across opposite cell faces, and the exact
translation group comes from `PeriodicCell`. Then build a periodic surface carrier
and a periodic constrained tetrahedral fill whose face triangulations are
identical under the lattice action.

**Owners:** E `geometry/implicit/_adaptive_discovery.py`, `_policy.py`,
`_source_charts.py`, `_certified_implicit.py` (quotient-aware cover, scaffold
and topology); E `providers/_native_implicit.py`, `meshing/_implicit_volume.py`
(admit `periodic_constraints` for periodic sources); reuse
`discretization/_periodic_topology.PeriodicMeshTopology`, `_native_periodic_tet.py`
orbit maps, and native PLC/refine3d with paired periodic faces. Coordinate the
native periodic fill with the refine3d owner, and do not create a second periodic
topology representation.

**Required behavior and evidence:**
- Report quotient topology per cell (components, Euler characteristic and genus of
  the quotient surface; component counts of the network and complement) from the
  certified cover. Compare against cited literature values only for the declared
  primitive or conventional cell.
- Every periodic face pair is exact under the lattice action. Orbit
  representatives and windings come from the canonical periodic topology owner.
- Source fidelity, coverage and the approved hard/soft shape-target semantics are
  unchanged. Orbit-locked adaptation and compatible transfer on the quotient reuse
  the existing periodic adaptation owners.

### 14.5 Sheet semantics and finite specimens

**Sheet thickness:** the level band `{|F| ≤ τ}` has nonuniform wall thickness.
Expose it honestly through two explicit, separately named semantics:
1. A *level band* with requested `τ`. Its certified minimum wall thickness is
   `2τ / sup‖∇F‖` over the band, enclosed by intervals; the achieved thickness
   distribution is reported as measured evidence.
2. A *minimum-thickness target*. Solve for the smallest `τ` whose certified
   minimum thickness meets the requested printable wall `t`. This replaces the
   donor's `½·k̄·t` heuristic.

A true geometric offset (exact distance band) is a different source. It is added
only if certified distance queries to the nodal surface can carry it, and never
relabelled from a level band.

**Finite specimens:** intersect the lattice source with an analytic `Box`,
`Cylinder` or `Sphere` through `SharpCSG`, keeping exact set semantics. The crease
curves are the zero set of `F` restricted to each envelope surface chart. Discover
them with the existing two-dimensional implicit curve discovery on the envelope's
parametric chart. They become protected features and trim the envelope caps,
which seal the cut ends like the donor's Boolean caps, but with certified
geometry. Owners: E `geometry/implicit/_curve_discovery.py` (parametric-chart
mode), E `meshing/_surface_construction.py` and the implicit volume route (mixed
implicit-face and trimmed-analytic-cap boundary), reusing source-fidelity and
protected-feature evidence. Creases tangent to an envelope are certified
unresolved events, not meshed.

### 14.6 Lattice metrics

- Certified solid volume fraction (relative density, porosity) enclosure from the
  final discovery cover: inside, outside and unresolved leaf volumes, which are
  independent of the mesh. Report it alongside the accepted mesh measure and its
  source-fidelity bound.
- Surface area and specific surface area from the accepted mesh, reported as
  measured quantities with the fidelity evidence. They are not claimed certified
  unless an area enclosure is proved.
- Owners: E `geometry/implicit/_adaptive_discovery.py` (cover measure enclosure)
  and the existing meshing result measure/evidence owners. No donor-style metric
  dictionary.

### 14.7 Physics and design consumers

- Periodic homogenization: solve cell problems on the periodic quotient mesh with
  `PeriodicHomogenizationCase` and Hill–Mandel evidence for effective elasticity,
  and the scalar analogue for effective conductivity. Effective permeability needs
  a periodic Stokes cell problem and is planned only if that solver owner exists.
  Cubic symmetry of the effective tensor is a checked consequence for cubic
  families, not an imposed constraint.
- Design: lattice parameters (level `c`, `τ` or the thickness target, cell
  scales) are `DesignState` parameters. Realization uses the fixed-lattice,
  fixed-topology `ImplicitSurfacePlan` route, giving JAX sensitivities of volume
  fraction and homogenized properties. Certified critical-level crossings and
  topology changes invalidate sensitivities as events, consistent with the
  existing design event semantics.
- Spatially graded lattices (position-dependent `c` or cell size) are
  nonperiodic finite-specimen sources. They are a later extension and are never
  meshed as a periodic quotient.

### 14.8 Qualification, export and documentation

- N `tools/meshing_qualification/scenarios/native_tpms.py`, registered in the
  qualification CLI:
  - periodic gyroid sheet cell: quotient topology, exact periodic face pairing,
    the certified volume-fraction enclosure against an independent high-order
    quadrature of `{|F| ≤ τ}`, and homogenized conductivity with checked cubic
    symmetry;
  - Schwarz P network at `c = 0`: volume fraction ½ proved by symmetry, plus
    network/complement component counts;
  - Schwarz P at a level inside its critical-value enclosure: certified singular
    refusal;
  - diamond specimen clipped by a cylinder: certified creases and caps, a
    watertight oriented solid, FE linear-elastic compression, and volume/porosity
    oracles;
  - an independent-process cold restore of one periodic lattice result.
- Export: STL/3MF write of the accepted surface through the existing interchange
  catalog. Solid export requires a closed, oriented 2-manifold certificate, and
  STL is declared lossy with respect to the source.
- Guide section in `docs/guides_meshing.md`, an example
  `examples/native_tpms_lattice.py`, changelog, public API and capability
  regeneration, and typing fixtures for the new public source.
- Benchmarks: discovery, periodic fill, crease discovery and homogenization,
  phase-separated, scaling with cells per specimen and level near criticality.

### 14.9 Explicit exclusions

- No marching cubes, Manifold3D, trimesh, scikit-image, PyVista/VTK, PyQt or
  GPU-detection code. No float32 fields and no constant-padding closure trick.
- No string formula parser or `eval`.
- No average-gradient `τ` heuristic presented as wall thickness.
- No QEM decimation: size reduction stays the certified native remesh within the
  fidelity band.
- These items extend the plan's acceptance obligations; none is implemented by this
  addition, and the original section-12 completion requirements are unchanged.
