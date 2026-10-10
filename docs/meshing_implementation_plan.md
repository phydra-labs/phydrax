# End-to-end native meshing implementation plan

## 1. Deliverable and completion contract

This document began as the static implementation plan for the
`feature/native-meshing-20260929` worktree. The initial baseline was `dev` at
`62abd2db0`; later `dev` merges culminated in `a93b6f3c2bbeca95c64e5a571c98e94c848eeabb`.
The file-by-file sections below remain the design and acceptance record.
“Proposed” and imperative text describes a requirement, not a claim that the
symbol, workflow, qualification, or benchmark is complete. Current public
support and evidence boundaries are authoritative in
[`guides_meshing.md`](guides_meshing.md#evidence-vocabulary-and-current-closure-state)
and the generated capability catalog.

The deliverable is a Phydrax-owned geometry-to-simulation meshing system, not a facade over other meshers. It includes native initial generation, improvement, adaptation, geometry preservation, state transfer, distribution, persistence, and solver integration. It also includes native CAD construction/query/Boolean/interchange capability, rather than treating OCCT independence as an unspecified future project.

Ordinary numerical/runtime infrastructure such as NumPy, JAX, compilers, and communication libraries is permitted. Gmsh, CGAL/pygalmesh, TetGen, fTetWild, Mmg/ParMmg, Omega_h, VoroCrust, Manifold, OpenVDB, Open3D, PyVista/VTK reconstruction, Qhull, OCCT, and METIS must not perform a geometric/meshing/partitioning operation on a route advertised as native. Explicit optional comparison/interoperability providers may remain; there is no hidden fallback to them. File codecs are distinguished from mesh algorithms, and any retained external codec is declared at interchange rather than described as an in-house codec.

Completion requires all mandatory positive workflows in this plan to execute with those engines absent, not merely to refuse their inputs safely. A refusal is a correctness success only for a deliberate negative case; it is a functionality failure for an admitted positive case. Quality, geometry, semantic, and resource requests cannot be relaxed to improve benchmark success rates.

General automatic high-quality all-hex construction, robust singular CAD arrangements, and reconstruction of underspecified defective surfaces have research-dependent boundaries. Those workstreams remain in scope. Their acceptance domains and evidence must be explicit; an unresolved research gate does not become a completed capability by changing its label or silently narrowing the user's scope. Narrowing the final goal requires a separate user decision.

### Current disposition

The current implementation has focused owner verification for W02/W03,
W05/W06, W08, and W10--W14. W05 closes sheet, dirty, disconnected, and
marked-adaptation envelope cases under the unchanged 20,000,000-work request;
W06 includes rotational-volume and mixed rotation/screw compatible-field
lifecycles under unchanged limits. W04's focused matrix is 83/83. W07's
targeted source-only/loft/topology matrix passed, and exact rational
collapsed-pole radial splits plus locally refined per-interval trim ribbons
close the overlapping-sphere and curved-void positives. W08's surface/tetra metric
matrices, rational-trim continuation/archive, curved-sphere, and positional
archive-recipe checks passed; its implementation evidence is not a final W15
qualification campaign. W09 is implemented and
focused-qualified on corrected source `curved-periodic-narrow-gap-exact-x-orbits`
(revision `2afa0d24…`, digest `059b334c…`): the final complete repeats took
106.2345 and 108.5906 seconds under the unchanged 120-second limit. The
original period-x=1 source remains the immutable exact negative described
below; neither tolerance changes nor a
source rewrite reclassify it. W15 tooling/corpus checks are 46/46, but no final
qualification, like-for-like benchmark-leadership, authenticated release, or
leadership artifact has been produced. These are separate implementation,
test, qualification, and release facts.

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

### 1.4 Revisions required by the exterior-calculus merge

The `dev` merge at `f99cd90d2` replaces specialized compatible element owners and extends canonical mesh/form support. Meshing integration must retain both the new substrate and the geometry/lifecycle work already implemented:

- `form_element` and `FiniteElementDeRhamComplex` own compatible scalar, circulation, flux, and density spaces. Delete the superseded tetrahedral Nédélec and simplex RT/BDM owners; migrate generation, transfer, chart, solver, qualification, and archive consumers rather than restoring removed factories.
- Compatible DOF orientation is a complete entity transformation, not necessarily a signed permutation. Nested and common-refinement transfers must use the canonical form basis with the declared degree, twist, proxy, and coordinate map. Retain failure, reproduction, commuting, and content evidence; a merge must not downgrade general-order fields to lowest order.
- `SimplicialConnectivity` and canonical reference topologies admit arbitrary dimensions. Distributed storage, affine geometry construction, coordinate polynomial enclosures, and point locators must compose that support without narrowing the merged API to triangles and tetrahedra. Native generation routes retain their explicitly admitted dimensions.
- Full and restricted coordinate maps remain authoritative for physical DOF locations, mapped support, embedded geometry, and field queries. Form-based scalar coordinate elements must expose their actual polynomial basis to enclosure/validity preparation; do not identify them by old family strings.
- Scientific archives must inventory and restore the canonical `FormBasis` and proxy tabulators, including all array leaves and static form identity. Old specialized tabulator codecs must disappear with their deleted owners.
- Adaptive H(curl) capability retains the merged general-order de Rham space and the existing meshing accepted-transition lifecycle. Its native adaptation owns the explicit form-field transfer and `CompositionRebind`; refusal preserves the accepted state. Generic FE solver adaptation continues to use the existing explicit topology transaction.

Integration verification covers canonical form/de Rham behavior together with meshing field transfer, restricted/mapped/embedded queries, scientific recipe restoration, and an actual accepted native adaptation. Native geometry kernels are unchanged by the exterior-calculus merge; their bounded-allocation and atomic-refusal checks remain required.

Observed post-merge smoke: native bisection of one tetrahedron publishes two children; canonical trimmed order-two transfer reproduces degrees zero through three at interior queries with maximum value defect 1.6342482922482304e-13 and maximum exterior-derivative commutation defect 7.993605777301127e-15. Source-only PLC preparation preserves four authoritative vertices, six edges and four triangles with 18 actual work units and a 1288-byte retained-memory peak under the unchanged 1000000-byte cap. The rebuilt native suite passes all 17 CTest cases. These are integration observations, not completion of the full workflow or comparative-performance gates; Python contract and typing integration remain separate checks.

Subsequent integration checks: 34 scientific model-recipe cases pass, including immutable form identity and invalid-proxy refusal. A five-dimensional affine FE map reports determinant 32 and zero inverse-action defect, and rejects its inverted counterpart. The native conforming curve split/inverse-collapse smoke preserves all original source coordinates and facet IDs through ghost closure; protected-corner, work-budget and allocation-budget regressions pass in `test_improve3d`.

The resumed merged-transfer check passes 84 cases across compatible mesh
transfer, compatible field queries and cell-geometry transfer. The canonical
host-value optimizer check passes 20 cases covering nonlinear constraints,
strict evaluation allowances, finite feasible state retention and nonfinite
refusal. The selector audit reports zero findings. These scoped checks do not
replace the integrated typing or unchanged transport acceptance gates.

Resumed image preparation now proves exact coplanar voxel-face coalescing and preserves rounded oblique triangles when that proof fails. The unchanged two-material box uses 12 authoritative vertices and 11 source polygons instead of 27 vertices and 56 independently constrained triangles. Conforming PLC generation no longer promotes internal polygon triangulation diagonals into scientific feature curves; fixed-boundary obligations remain unchanged. Native memory counters cover the native allocation owner, not JAX or whole-process/device peaks.

The frozen native Neurofluid profile still requires exact `p50` and `p95` edge statistics at target 1.0. Shape-only construction and global circumradius sizing satisfy different subsets of those goals; neither is a complete positive result. The owning native metric pass manager is being joined to initial generation with the original statistical goal, exact source-boundary coarsening and actual work/resource evidence. Publication, continued transport, integrated typing and phase-separated benchmarks must pass before this workflow is marked complete.

### 1.5 Bounded integration checkpoint

The user approved collecting atomic owner handoffs, one integrated verification
pass, and an acceptance/blocker report before further qualifications. That
checkpoint is complete. The user subsequently requested completion of all
remaining tasks; implementation and prerequisite-controlled qualifications have
resumed. No new feature scope or long lifecycle qualification was launched during
the checkpoint pass. `AGENTS.md` was copied verbatim from `dev`; both copies
have SHA-256 `5c1d66087a2dacda0a3dbbac12171521d770f54979b0dfe3936967f71f9d12e8`.

Fresh checks on the collected implementation:

| Surface | Observed result |
| --- | --- |
| Native build and CTest | All 17 tests passed in 18.01 seconds |
| Focused model recipes, archive safety, checkpoint and native source restoration | 66 tests passed with `-n auto` |
| Checked signature audit | No redundant checked guards reported; identity-sensitive and static-only guards retained |
| Selector audit | Zero findings |
| Public API data | Regenerated; manifest check passed |
| Actual native planar publication and recipe restoration | Certified 5-vertex, 4-cell mesh; 87 recipe arrays; exact restored coordinates, connectivity, active masks, vertex/cell IDs and request identity |
| First-party typing | One invalid-argument-type diagnostic at `_meshing_field_records.py:408`: `Array \| ndarray` passed to an `Array` recovery kernel |
| First-party Ruff lint | 483 diagnostics; baseline-versus-change attribution not established |
| Collected substrate formatting | `_coordinate_enclosure.py`, `_mapped_embedding.py` and `_distributed_checkpoint.py` would be reformatted; numerical-relativity `_checkpoint.py` already formatted |

The runtime smoke's first attempt used a nonexistent `CellBlock.connectivity`
field in the throwaway harness. After changing that check to canonical `vertices`
and `vertex_valid`, the smoke passed. It is a serial publication/recipe check,
not different-process-count restart or continued FE/FV qualification.

Collected failures remain blockers, not completed positive capabilities:

- The already-running final CAD intersection/Boolean/B-Rep set finished with
  20 failures, 82 passes and 55 warnings. Failures include cone preparation,
  torus criteria, revolved/trimmed tessellation, cylinder trim and tangency
  discovery, coincident-cylinder face graphs, sphere material labeling and
  incorrect solid classification. One failure is an incidental message assertion.
- The interchange owners reported 22 STEP failures and 12 IGES failures from
  runs overlapping earlier edits. Source-only round trips do not establish
  realized interchange correctness; first-round native symbolic-period identity
  is not preserved by the external floating-period representation.
- Mixed adaptation passed 10 cases, but mapped DG retains two work-count
  assertion failures. Coverage retains an `unresolved` versus `violated` status
  mismatch. Other mapped-support/high-order failures remain unattributed.
- Packed checkpoint descriptor publication, restoration and corruption refusal
  have focused proof. Different-process-count restart and continued FE/FV
  execution on that new durable representation were not run.
- Periodic full lifecycle remains outside its 120-second wall gate; the observed
  epoch-zero L2 error was about 0.143, above the 0.05 target. Neurofluid exact
  p50/p95 sizing and continued transport remain incomplete.
- At the checkpoint, size length comparisons added `16 * float64 epsilon` to the
  requested relative tolerance. That implicit allowance has since been removed:
  comparisons use only the authored absolute and relative tolerances. An actual
  measured edge of `1.0000000000000002` is refused at zero tolerance and accepted
  at an explicitly requested one-ulp allowance; seven focused sizing regressions
  pass. Earlier passes using automatic slack do not qualify exact frozen goals.
- Resumed native publication measures the complete distinct retained scientific
  array footprint, not only vertex/connectivity carriers. A tetrahedral smoke
  retained 2540 bytes against 112 carrier bytes; a 2539-byte cap refused at
  canonicalization with the original requested/achieved data quantities and
  cumulative native execution measurements intact. Its permanent consumer
  regression passes. Source-only STEP sphere import retains both authored pole
  vertices, and sphere/plate native archives preserve exact model and geometry
  identity; these do not qualify realized external interchange.

This checkpoint does not mark W02–W15, the mandatory corpus, release qualification
or the original end-to-end deliverable complete. Resolve each workflow's scientific
and integration prerequisites before its next long qualification run.

#### Resumed scientific and lifecycle evidence

| Surface | Observed result | Still required |
| --- | --- | --- |
| Exact n-ary surface classification | Implicit construction-point ray classification replaces floating winding/trust margins; 63 scoped tests pass with explicit work and native byte refusals | Full source-solid validation remains a separate contract |
| Native-backed host arrays | Actual 96-byte float64 payload plus 40-byte owner metadata; views remain valid after scope exit, including last-view release on another thread; 20 scoped Python cases pass | Arbitrary NumPy, JAX, device and compiler allocations are not thereby measured |
| Execution identity and source restoration | Fresh current-default source, transfer, refinement, and separate-process reopen/coarsening complete for the unchanged periodic/material source, 31→35→31 cells, with exact original scientific arrays, source-coordinate banks, routes, columns, regions, and quotient data. Final source content `70f1c8e1…` and lifecycle content `f5afaaff…` use seven members/six dtype banks under unchanged 257-member/16-level/rank-8 defaults; `.tmp/archive_final31_{source_attempt5,refine_attempt2,coarsen_attempt2}.run.json` | Full field/history/PDE and mandatory corpus qualification remain separate; historical larger-limit or earlier native/representation receipts are not the current final receipts |
| Installed native dependency isolation | Normal-resolver installation outside the checkout realizes native STEP/IGES/BRep box queries and archives, area 22 and volume 6, with no external geometry engines loaded | Final global isolation after all remaining code lands |
| Earlier native ABI and quality | The matching wheel used for these records binds header contract `17a42093…`, source digest `b4815b31…`, and binary digest `5172440c…`. Both native configurations build; all 17 CTests pass. Twelve selected-edge/pole/resource Python consumers pass, and the unchanged ball-in-box native BRep source plus Gmsh layer consumer passes in 57.354798 seconds after explicit selected-edge paired-cavity closure; `.tmp/native_surface_cavity_build_wheel_attempt1.run.json`, `.tmp/native_surface_cavity_python_consumers_attempt3.run.json`, and `.tmp/gmsh_ball_native_source_layer_after_cavity_attempt1.run.json`. These and earlier decimal-split/115-case records remain preserved under their actual native identities | Later prepared-source changes use the current identities recorded below. Full original reconstruction, statistical sizing, curved-layer, periodic/material, surface, CAD and final isolation gates remain separate; mismatched binaries remain refused |
| Source-safe native predicates and resource ownership | Independent source banks, geometric subsegment bounds, shared-facet ancestry, real bounded shape refinement, exact-source cavity positivity/empty-ball decisions, and canonical carrier priority have native consumer proof. Native measured allocation counters remain six; two additional conservative host-storage bounds share the original cap without dummy buffers. Full first-party typing and both selector/signature audits are clean | The unchanged complete workflow corpus still requires its own runtime proof; later layer-source changes have focused typing proof and await final integrated checks |
| Native spatial curve density | Source-breakpoint quadrature and local cumulative-density residuals converge for both sphere-intersection density variants and the meridian. The formerly stagnating sixths root has residual `-1.787426315116441e-16`; all three actual spatial publication consumers pass after owned certainty-mask composition, `.tmp/cad_topology_spatial_consumers_attempt2.run.json` | Full curved CAD partition, sweep, interchange, and physical continuation remain separate |
| Registered archive authority and dynamic fidelity | Nine self-certified foreign PLC identity/bank/bound cases and four represented/parametric layer query/domain cases refuse at archive admission after genuine original-positive closures. Closed parametric originals retain all six scoped source-face checks; dynamic query numerical leaves are visible and fixed-role/nontrainable, with no static-array warnings. User-approved partition cutover changes whole-container/PyTree/model/receipt representation; `.tmp/archive_final_layer_authority_consumers_attempt7.run.json` | Closed restoration fixtures do not replace original open-wall lateral ancestry or curved W09 acceptance; those retain their original inputs and independent positive gates |
| Original W14 declared-field campaign | The unchanged resolution-4/capacity-20000/120-second/three-repeat/one-round campaign reports `passed` and `mandatory_workflow_completion: complete`: all uniform, analytic and learned trials complete, all-field reanalysis and learned admission publish receipts, analytic selection continues its solve, and geometry/PDE/transfer FD-duality plus event invalidation pass. Measured lifecycle 71.915896 seconds, wrapper 93.765553 seconds; `benchmarks/native_design_lifecycle.json` and `.tmp/w14_original_prepared_linearization_attempt1.run.json` | Earlier stale-marker and 129.482753-second timeout records remain preserved. Full h/p/metric corpus and final release evidence remain separate. Learned had higher observed cost and physical error than both baselines in this single ordered campaign; no superiority/generalization claim |
| Original C0 mixed-sketch extrusion | Authoritative knot-stratum interpolation closes the original native tessellation; source-u knot partition then corrects the Green measure defect from 1.330178878851034 to 1.3298672286276014 against independent analytic volume 1.3298672286269282. All seven independent signed-face comparisons pass. The unchanged default-tessellation/source-measure/save-load/no-external-kernel consumer passes in 564.4084 seconds under its original 600-second deadline, with exact model/source identities and restored-volume equality; `.tmp/cad_topology_green_original_source_measures_after_syntax_attempt1.run.json` and `.tmp/cad_topology_mixed_extrusion_green_after_attempt1.run.json` | Three original interval warnings remain visible. Fine/coarse reported error zero is not an outward integral certificate. Other CAD intersections, curved partitions, periodic realized interchange and the captured source-distance tail still require their separate original gates |
| Compatible nonnested transfer and second-order entity transforms | Eight original independent contracts pass under `-n auto`: all 24 tetrahedral order-two circulation base transformations, shared tetra/hex moment identity, arbitrary RT/BDM divergence moments, realizable 3D curl companions and transpose pairing, and incompatible-family refusal. Direct public native common-refinement/compiled transfer smoke passes for 74 circulation and 144 full-flux DOFs: affine reproduction errors 6.22e-15 and 4.77e-15, transpose errors 7.11e-15 and 6.66e-16; `.tmp/w08_compatible_original_moment_contracts_attempt1.run.json` and `.tmp/w08_compatible_public_smoke_attempt1.json` | Coverage bounds remain explicit (1.02e-10); 3D curl commutes with the realizable discrete curl-image companion, not an arbitrary discontinuous cellwise projection. This closes the scoped compatible-transfer/entity-transform item, not general curved nonnested remeshing, periodic fields or the all-state PDE corpus |
| Original curved-layer periodic source | Constructor-authored missing root period descriptor and current source archive pass in 271.958167 seconds with original physical bits/columns/intervals retained. The targeted original facet-18 orbit diagnostic completes in 158.15742 seconds; bottom vertices agree exactly, but original top traces retain x residuals -101121/2361183241434822606848 and -6620711477/77371252455336267181195264 under period-x = 1. `.tmp/layer_root_periodic_declaration_attempt1.json` and `.tmp/layer_first_periodic_trace_attempt1.json` | User explicitly chose to preserve the original source and strict periodic semantics. This original positive gate is blocked by mathematical source incompatibility; no rounding, source revision, tolerance waiver or false acceptance is permitted. Independent quality/resource/archive obligations remain actionable |
| W10 exact embedding work | After actual shared face-expression power preparation and eleven independent contracts pass, original attempt 10 still certifies the 1,952-cell source embedding at 1,958,596 cumulative work but refuses coverage at 2,000,000 completed units with 2,000,002 requested under the unchanged 2M-work/81,920,000-byte limits. Peak storage remains 33,819,669 bytes; actual retained storage increases to 27,204,533 bytes as more complete face data is retained. `.tmp/coordinate_face_composition_contracts_attempt1.run.json` and `.tmp/coordinate_original2m_source_target_attempt10.ledger.json` | Source support/publication, the 15,616-cell target, Q1 oracle, field/history/archive/coarsening remain unreached. Nineteen later ownership/action contracts and a compiled source-map smoke close their scoped invariants, not this workflow. Attribute each actual preparation key/work delta before a performance rewrite; two aggregate scope calls do not establish duplicate full traversal, and genuine Jacobian work remains charged. |
| Current exact source-key sharing | Per-key attribution observes 7,837 bank/key calls and 188,088 charged coefficient visits, including a second traversal inside multi-affine corner preparation. The validated complete bank/key now feeds corner construction directly. Eight host/exact-signed-bank/source-basis/changed-source/reference-map contracts pass in 29.69 seconds; the unchanged original source runtime measures exactly 46,848 fewer visits, embedding certified at 1,911,748 work, `.tmp/native_exact_source_key_sharing_contracts_attempt1.run.json` and `.tmp/coordinate_original2m_source_target_after_key_sharing_attempt1.run.json`, wrapper 28.570325 seconds | Coverage still refuses at 1,999,999 completed with 2,000,005 requested under the original 2M/81,920,000-byte limits; retained storage 27,617,580, peak 36,818,453 bytes. Genuine Jacobian work remains charged. The 15,616-cell target and physical field/history/archive continuation remain unreached; this measured sharing is not full W10 acceptance |
| Original envelope source archive | Original resolution-4/capacity-20000 raw sheet and wrapping policy publish their complete source closure and reopen in a separate process under unchanged default 257-member/16-level/rank-8 archive limits. Every raw vertex/triangle/feature value, policy, carrier, repaired model/realization, PLC binding and repair theorem identity is preserved; content `cbbe0bd1…`. Producer/cold wrapper times are 19.037107/18.958237 seconds; three independent repair-bound/topology-forgery and source-bound contracts pass in 29.37 seconds. `.tmp/w05_original_envelope_source_archive_produce_after_identity_attempt1.run.json`, `.tmp/w05_original_envelope_source_archive_cold_after_json_boundary_attempt1.run.json`, `.tmp/w05_envelope_archive_theorem_contracts_attempt1.run.json` | This is source-only archive completion, not the full W05 volume/field/PDE/adaptation lifecycle. Complete registered-part validation and original exact successor geometry/continuation remain separately required. Diagnostic-script owner-field and tuple/JSON-boundary failures remain preserved |
| Current original image sizing and native cavity ownership | Actual first-divergence attribution proves a non-compound source-witness insertion admits 3 old/6 predicted binary children but legitimately commits a 9-old/16-new native conflict cavity. Source-aware prepared inspection/commit fixes admission, retains the original source fraction and generation, and uses actual ambient native buffers. Matching C ABI `f52d53c8…` wheel builds; two affected CTests pass in 1.51 seconds. The unchanged size-two/capacity-4096/target-1/default-16-pass/zero-tolerance request now completes every metric pass and reaches certification/compliance with every independently counted committed edge incidence correct; `.tmp/native_prepared_source_split_build_attempt1.run.json` and `.tmp/native_image_metric_actual_prepared_cavities_attempt1.run.json`, wrapper 78.536585 seconds | Final honest `COMPLIANCE_FAILED`: p50 0.9814366945225572 and p95 1.1130049976183884 still differ from target 1. Sizing convergence and full image/Neurofluid transport remain open. Original source, schedule, controls and resource limits stay unchanged; no tolerance waiver or replay of unchanged failure. |
| Current owning-quantile proposal and prepared work | Four independent one-ULP crossing, forward/reverse padding and linear-interpolation cases pass in 34.70 seconds. Global proposals now use the owning requested linear quantiles, not a frozen/full equal-edge band. Native stored linearization retains the primal sort permutation; actual derivative actions reuse its gathers/scatters. The unchanged original request reaches its owning compliance check after the corrected accounting in 62.147456 seconds with 73,320,354 cumulative metric work (71,198,726 host, 2,121,628 native), `.tmp/native_image_owning_quantile_contracts_attempt1.run.json` and `.tmp/native_image_metric_prepared_sort_accounting_attempt1.run.json` | Final `COMPLIANCE_FAILED`/stalled after 12 passes: p50 0.9999768301466239 and p95 1.2134051921683116. Actual supporting edges have movable source tangents; no incompatibility theorem is inferred from a fixed corner. Preserve original LM controls and zero tolerance; inspect the retained last solve and native admission before another full run |
| Exact source/storage ownership and full P1 coefficient actions | Nineteen independent alias/lifetime/atomic-quota/live-scope/full-weight/source-cache/physical-measure/shared-incident contracts pass with `-n auto` in 30.69 seconds. Direct compiled full-weight tetrahedral physical map/JVP matches independent rational references with unchanged source coefficients; an actual ended native scope retains work/memory and explicit logical-unmanaged JAX evidence, `.tmp/native_source_ownership_contracts_attempt1.run.json` and `.tmp/native_source_ownership_smoke_attempt1.run.json`, smoke wrapper 20.831721 seconds. | This does not establish W10's original complete source/target work gate or Packed direct/device/collective/archive lifecycle. General Cartesian interpretation of nonunit coefficient rows and arbitrary curved-source extension remain refused. |
| Current bounded-source inspection and atomic publication | The original decimal RNE witness exposes an existing complete-star route that generic conflict preparation can refuse. Direct and inspected star routes now share source/protection/surface/work admission; inspection retains the exact finite 1-old/2-new cavity under its original allowance. Actual preview→commit smoke preserves original coordinates and certifies deviation 5.721958498152808e-17 against 1e-12, wrapper 18.551968 seconds. Eight prepared/direct/source-bound/fixed-source/unauthorized-carrier contracts pass in 28.57 seconds; two affected CTests pass after the matching wheel rebuild. Header ABI `f52d53c8…`, current source `16474257…`, binary `c069d833…`; `.tmp/native_prepared_source_star_build_after_nested_type_attempt1.run.json`, `.tmp/native_prepared_rne_source_smoke_after_reporting_fix_attempt1.run.json`, `.tmp/native_prepared_source_star_contracts_attempt1.run.json` | Raw smoke memory is measured; whole-operation work counts are explicitly incomplete. Original image sizing, broad refinement and final isolation remain separately required; this scoped proof does not relabel the prior image compliance failure |
| Direct and scheduled source-segment relocation | The original right-tetrahedron direct move to exact parameter 0.375 remains covered. Automatic improvement now generates bounded source-line candidates and commits a genuine physical-quality improvement while preserving original coordinates, source-chain ancestry, facet planes/areas, region volume and positive orientation. The current matching-wheel original multiface, vertex-removal, protected-vertex and scheduled-curve consumer smoke passes all four cases; the vertex-removal minimum remains unchanged | This scoped schedule/source proof does not qualify the complete material/cavity lifecycle, public resource declarations, broad W03 corpus or final robustness |
| Original periodic sparse hard-solve admission | After causal sparse Gram/block-inverse, native failure, sticky quota and warning-free affine BVH contracts, the full original source/request/options inventory and actual ended native scope are observed. Original attempt 11 preserves every scientific array and source identity but refuses before hard polishing: proposal work 18,441,298 plus admitted hard bound 39,324,230 exceeds the unchanged local remaining 21,394,151; root remaining 22,518,190 is not a replacement local allowance. Wrapper 21.585075 seconds; `.tmp/periodic_restart_material_sparse_schur_attempt11.run.json`. Ended root work is `[18441810, 0, 0, 0, 0, 0]`, managed peak 37,420 bytes and host-storage upper peak 8,210,286 bytes | No hard solve, exact placement, publication, field/PDE or archive positive follows from this refusal. Whole-work instrumentation remains incomplete. Native sparse schedule and scientific graph preparation optimization must fit original controls and both actual allowances |
| Original W11 polyhedral lifecycle | Native material-L generation, certified canonical adaptation, physical/history rebind, conservative overlap remap and continued mimetic diffusion pass: left inventory 4→4, right 5→5, physical error 1.0658e-14 against 1e-7, 20.663169 seconds, 63,350 logical retained bytes; artifact 5864 | Source construction still reports cavity/query/scratch guarantees as unenforced; broader quality/corpus/scaling and final release checks remain separate |
| Original W13 lifecycle | Full unchanged resolution-4, three-repeat, two-motion workflow passes in 105.289781 seconds under the 120-second/0.05 gates: conservative density 72→72, material inventories 18 each, history defect 5.684e-14, both accepted-boundary/PDE transitions and rollback, 22,769,635-byte checkpoint, source-authorized restart/reprepare, and continued-versus-uninterrupted PDE error zero; artifacts 5894/5903 | Scientific ambiguity counts 2/1 remain explicit; final global checks and broader corpus/scaling claims remain separate |
| Fresh original W12 lifecycle | One fresh two-producer → one-process restart run passes native generation, refine/coarsen/re-refine, GRAPH, all-role checkpoint, cold source recertification/repack/rearchive and continued FE/FV; all four hot/cold field errors are zero, essential masks exact, nine IDs/clocks/cursors preserved, both tamper proofs refuse; FV updates all eight cells with full-component inventory drift 5.551115123125783e-19 | Final global release checks and broader scaling claims remain separate |

The final W12 run finished in 964.676739 seconds under its unchanged
3600-second deadline. Its complete record is
`benchmarks/native_distributed_lifecycle.json`; it is not the earlier stitched
continuation. The process-lifetime resident peak was 13,543,211,008 bytes, not
a native-managed or device-memory measurement. Compiler/device peaks and
message payload bytes were not measured. Earlier failed and diagnostic records
remain preserved, rather than being rewritten as successful outcomes.

The final W13 checkpoint has content identity
`57623ce5a98a70c933b15085ef2ec18b63e7c0904b82c592e968b05fd8076868`.
Its measured lifecycle time includes both motions, checkpoint write/read,
restart preparation, and continued PDE execution. Process startup is separate
(124.83 seconds overall); earlier source-association and timeout failures remain
retained evidence, not successful outcomes.


### 1.6 Reassessment after stopping all jobs

All running agents and services were stopped at the user's request. Applied
worktree changes and actual positive/failing records remain intact. The
architecture is not being restarted: the remaining work combines incomplete
cross-owner implementation, unverified integrations and original positive
workflows that still fail. Completed administrative tasks are not a percentage
of scientific capability completion.

The original mixed-sketch extrusion/archive, ball-in-box source/layer consumer,
compatible nonnested differential transfer, W12/W13 lifecycles and W14 design
campaign have actual positive evidence above. They are not requeued merely to
confirm those observations. The original curved-layer source remains an explicit
incompatibility refusal under the user's preserve-source decision.

#### Dependency-first implementation sequence

1. **Stabilize shared source and resource ownership.** Complete actual distinct
   owner retention in `NativeHostStorageWorkspace` and escaped live-workspace
   accounting in `CoordinateEnclosureBudget`. Release discarded per-node
   composition/Bernstein workspaces while retaining genuinely live frontier
   charts and source banks. Reuse complete source preparation across embedding,
   face traces and coverage, keyed by explicit scientific identity. W10's small
   face-sharing change does not resolve the dominant original 2M-work failure.
2. **Close packed scientific lifecycles before dependent campaigns.** Complete
   approved 6/24 lineage with the actual original `CellMeshingResult` owner,
   exact source barycentric data and dynamic nontrainable leaves. Direct,
   device and distributed execution share the original allowance. Collective
   proof retains untouched compiled outputs and derives actual host inverse
   publication states; cold archive validation must replay that derivation.
   Finish envelope source-family admission and independently restored repair
   bounds, then original volume/field/solve continuation. No affine substitution
   or parent-source reconstruction from coincident dimensions is permitted.
3. **Resolve measured numerical blockers.** Finish canonical matrix-free SPD
   preconditioning for the original periodic hard-constrained solve: all
   observed MINRES solves exhausted 64 iterations, so finite directions are not
   convergence evidence. Integrate and verify cumulative image/implicit clocks,
   measured source-query accounting and complete failure evidence, then capture
   the original image metric state on its current global route before choosing
   a p95/preparation correction. Older failures do not identify the newer route.
4. **Close remaining geometry and cell-family gates.** W02/W03 retain their
   complete native source and constrained-volume consumers, including positive
   segment relocation and current schedule-level cavity/removal evidence.
   W07 closes original curved partitions, Boolean/intersection matrices,
   periodic source-query attribution and the default STEP/IGES/BRep writer
   campaign. W08 closes surface/tetrahedral/mixed/polyhedral transitions with
   exact successor geometry and all state roles. W10/W11 close structured,
   swept, multiblock, quad/hex/polyhedral original consumers and scaling.
   Reachable W09/W06 layers, high-order fields, quotient/distributed semantics
   continue; the preserved incompatible curved-periodic positive is not rerun.
5. **Qualify and publish the complete reachable contract matrix.** Complete
   W14 h/p/metric decisions and W15 corpus, independent physical oracles,
   unchanged resource gates and phase-separated capacity scaling. Update all
   callers, examples, source inventories, current recipe identities, docs,
   changelog and generated API/capability data. Finish typing/selector/signature
   audits and dependency-isolated installed execution after final integration.

Implementation uses bounded ownership slices, not another broad feature fan-out.
Every shared file has one integration owner. Numerical windows are serialized;
long original workflows follow causal fixes and prerequisite proof, never a
different input, higher allowance, proxy source or relabeled safe refusal.

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

The CPU-native execution boundary is `NativeExecutionBudget` in
`phydrax/_meshcore.py`, backed by the nestable native scope in
`bounded_memory.hpp`. It borrows the existing managed allocation pool. Managed
allocation counters remain measured; conservative external-host reservations
share the same cap and have separate live/peak upper-bound evidence.
Original construction/root-solve work
units and explicit source-evaluation queries are cumulative; internal geometric
primitive requests have separate telemetry and native deadline checks, not an
automatic debit to either original quantity. Separately owned host/JAX source
batches enter through `charge_native_geometry_queries`. Static pure-solver work
bounds are admitted without recording them as actual consumption, then actual
work is charged before candidate commit. Atomic tetrahedral calls narrow and
restore the original absolute work ceiling, and allocation-free work observation
remains available after refusal. Native steady-clock exhaustion reports
`TIMEOUT`, distinct from capacity or scientific refinement failure.
`NativeExecutionBudget.remaining()` observes original nested work/query/cavity
minima, available bytes after native allocations and live external-host bounds,
and the native deadline;
it never ends, renews or replaces a scope. External local edits bind their actual
touched-cell sets through `admit_cavity`, sharing the same cap and peak/refusal
evidence instead of substituting published-cell or fragment counts.
`current_native_execution_budget` exposes the actual ambient admission owner
without loading a library, so publication helpers reuse the existing root.
Native scopes remain creating-thread owned; original request and operation-clock
metadata belongs to the Python execution wrapper, not this accessor.

Exact rational reference-triangle clipping packs coefficients and results in
actual native-backed host arrays, narrows its local caps to the original live
allowances, and charges returned native arithmetic work once before publication,
including partial work on refusal. Packing visits check the original deadline;
they do not introduce CPU-instruction proxy work units. Fraction/int temporaries
retain explicit host-coefficient bounds in the same original scratch allowance,
without being reported as measured native allocation bytes.
A scoped square-overlap smoke gives exact area `1/4`, 203 native arithmetic work
units and 742 observed managed peak bytes; a parent allowance of 400 refuses the
second clip after a cumulative 354 units, while zero work refuses before any
payload allocation. The existing clipping ABI does not expose a primitive-query
count, so that telemetry is explicitly unavailable, not a measured zero.


## 6. W02 — Compile geometry/controls and generate curves, planar domains and surfaces

**Prerequisites:** W00/W01; native CAD query implementation from W07 for CAD-specific gates. PLC/analytic paths do not wait for all CAD interchange work.

### Algorithm decisions

- Discretize authoritative feature curves once under chord/normal/metric bounds. Share the resulting vertices and constrained chains across adjacent surface patches; do not independently remesh seams and weld them by proximity afterward.
- Expose standalone curve discretization through `CurveMeshingSpec`, including open/closed curves, embedded curves and explicitly declared network junctions. Source incidence determines whether a junction is legal; default surface/volume manifold assumptions cannot silently reject or merge a valid curve network.
- Planar domains reuse constrained Delaunay and add spatially varying size/metric requests, region classification and protected embedded points/curves. Preserve narrow features and holes under explicit work bounds.
- Curved surfaces use chart-aware constrained triangulation or restricted-surface refinement according to the source capability. Handle seam charts, poles and overlapping charts through existing atlas/trim ownership. UV quality alone is not physical-space quality.
- Source scalar sizing is resolved by the canonical size-control owner at actual source points, with principal curvature obtained by native Gram whitening, bounded solves and Hermitian spectra. Proximity uses source-bound lower-gap evidence: analytic plane intervals or prepared native BVH covers of outward source boxes, never a nearest-sample distance labeled continuous. Unknown source separation is refused. Scalar edge bounds, continuous source interpolation/normal bounds and ambient physical SPD metric edge/shape measurements remain independent refinement criteria.
- Embedded mapped content and complete mapped-edge arc length share the exact source-expression `sqrt(Gram)` owner. Genuine rational coordinates and signed rational weights retain canonical numerator/denominator expressions: positive Bernstein denominator proofs, exact polynomial reciprocal/binomial moments, and outward approximation, arithmetic, and publication bounds determine acceptance under the unchanged work, subdivision, and term budgets. Exact denominator-square extraction reduces reciprocal series degree, and exact pullback Jacobians remain outside the Gram root. Source coordinates retain the owning exact corner-limit proof; a bounded direction-dependent tangent Gram at a measure-zero triangular apex is integrated through its exact cone chart without inventing a corner value. Expansion scratch is released while queued actual coefficient data remains budgeted. Arc-length statistics consume positive source-edge integral enclosures rather than RNE chords or fixed sampled quadrature; the sizing owner retains the original statistical comparison/error authority.
- Embedded DG prepares one bounded source-density polynomial and shares exact rational monomial moment functionals across its actual source moments, target mass, and mixed forms. A proved near-square residual reduces genuine rational radicals algebraically; 128-bit dyadic radical endpoints, reciprocal/binomial tails, and final binary64 publication errors remain explicit. Precision allocation comes from the unchanged 64-ULP all-column content claim and native mass-condition amplification, not an enlarged acceptance tolerance. Nonlinear quad refinement and explicit coarsening bind authored polynomial reference actions, live source signatures, and original coordinate banks; positive Jacobians, oriented boundary cancellation, and exact parent measure establish the complete reference partition before the existing native mass solve and transpose are published.
- Scalar surface and sphere-chart content retain rational source Gram expressions through exact affine restrictions and genuine projective reference actions. Proven positive projective denominator powers and exact Jacobian factors remain separate from the radical; the same prepared exact density/moment owner encloses the reciprocal factor and final publication under the original error and resource requests. An actual quarter-cylinder chart and its nonidentity projective image enclose physical contents pi/4 and pi/8 rather than carrier-chord areas. This scoped consumer proof is not a complete sphere-remeshing qualification.
- `SurfaceMeshingSpec.background_metric` binds an explicit affine simplex background mesh, its exact vertex metric revision and physical coordinate contract. Native original-coordinate predicates and bounded native solves establish query coverage; the metric owner performs SPD interpolation. Reconnection consumes full ambient tensors for metric spacing and diagonal decisions rather than replacing anisotropy with a global area limit. Uncovered points and incomplete hard physical criteria refuse publication; an incomplete unrequested schedule shape aim is an explicit warning with actual construction counters.
- Source volume-region controls on a surface are retained as `RegionBoundaryEvidence` linking exact source-region scopes/materials/roles to overlapping boundary labels and ordered source-region sides of mesh patches. They are not fictitious exclusive volume-cell zones. Actual surface-region controls retain exclusive face zones and authored shared-curve interfaces. No coordinate equality, display name or array extent establishes source identity.
- Pole preparation retains one authored physical corner and all collapsed chart wedges. Ring depth is checked against continuous source interpolation and normal-turn bounds; adjacent seam nodes are shared before CDT. Refinement of a collapsed wedge uses an exact unconstrained radial split, not centroid insertion. The split's chart coordinate is retained as an exact rational restriction carrier even when no binary64 point represents it; the rounded vertex is only an execution representative, and explicit reciprocal pole splits are producer-validated against that exact authority rather than rounded-view legalization. Native zero-area child retention requires equal explicit pole identities on an existing constrained collapsed side; ordinary coincident coordinates are not pole evidence.
- General curve trims use `CurveTrimLoop.certify_topology` bound to the actual prescribed chord partition, native interval derivative bounds for the complete source/chord homotopy ribbon, exact oriented chart-chain cancellation, and cross-loop separation/containment for holes. Ribbon bounds are retained per source-curve parameter interval: a boundary interval whose ribbon exceeds the requested deviation is bisected and reconnected before publication, and only boundary-adjacent triangles inherit that local bound. Arc and candidate-pair capacities belong to the source proof and remain visible in chart-cover findings/resource counts rather than being relabeled as global certificate limits.
- Rooted B-Rep coedges retain their original implicit endpoint and common vertex definitions through domain lowering and curve atlases. Nominal endpoint parameters are bounded realizations, not exact junctions; UV joining uses source root identity/chart injectivity and root uncertainty contributes to continuous curve/ribbon bounds.
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

### Observed retained-trim query and source-authority closure

`MeshingDomainBoundarySource` now retains original trim curves for native exact
membership classification and samples authored boundary chains as well as the
interior chart grid. A converged projection outside the original outer trim or
inside a source hole is not a boundary-distance result; that sampled query keeps
its nearest actual source sample instead. A narrow diamond whose coarse grid
contains no interior points retains its original boundary samples rather than
reporting an empty complete source and failing nearest-seed selection. These
distances and covering-radius estimates remain explicitly **sampled**, not
continuous certificates. Source sample/query and scratch admission account for
grid and authored-chain candidates before evaluation under the original budgets.

The intended closed-circle fixtures now explicitly author
`NativePeriodEndpoint` values at zero and one mathematical turn. Numerical
endpoint representatives, circle definitions, radii, hole orientation and
physical requests are unchanged; arbitrary binary64 endpoints are not promoted
to periodic roots. The formerly unresolved closing arc of the annulus was a
missing source-period authority, not permission to weaken trim isotopy.
The corrected source authority digests are:

| Internal fixture | Previous domain authority | Corrected domain authority |
| --- | --- | --- |
| Annulus | `b5ab78d22782e2db0c6801e87ca8312cd81f284394329d3e91387b8f803a6e29` | `a2f1e5b41691c2c3c376c8f7a09061a127e31a75e19b6d34f3b4a63fe67aaf40` |
| Capped cylinder | `0f0f15dd8b1d221f19a2720b7ab5edb0dccae46750e722c99a8bfc60c6e4d8e6` | `0fb527e4a8c8f45581b4ddce672f518f5757abc43272ed6235a30c58838e78f4` |

No historical qualification identity or frozen source payload was repinned.
The public annulus route executed with the original size request `0.2`, hard
surface deviation `0.01` and default numerical schedule: 210 vertices,
362 triangles, Euler characteristic zero, complete original trim-chart coverage,
complete associations and certified two-sided deviation
`0.007414619309032796`. Three scoped consumer regressions cover hole/outer-trim
distance, boundary retention without interior grid samples, and scientific
recipe restoration followed by owning source validation. The last case retains
the original circle and exact period roots through the existing canonical
`Fraction` representation; it does not replace them with a polygonized source.

The reserved public sharp/narrow diamond case also executed with its original
four vertices, soft size `0.2`, hard surface deviation `0.01` and default
schedule. It published 12 vertices and 10 triangles, retained all four authored
sharp corners, and certified complete original trim coverage with two-sided
deviation `5.684341886080803e-14`. Its measured area was
`0.010000000000000009`, matching the exact represented diamond. No source-angle
waiver, clipped source, polygonized smooth surrogate or quality-gate change was
used.

Straight degree-one spline p-curves exposed another source-query boundary:
native curve-band classification kept subdividing an exact straight boundary
hit instead of resolving it, making the IGES surface-import path impractical.
The sampler now reuses the owning exact affine-coefficient proof for original
line and equal-weight, clamped two-control degree-one spline carriers, including
their exact UV normalization. It uses equivalent affine polygon predicates only
after proving exact source joins, absent endpoint uncertainty and exact binary64
publication of the source endpoints. Original spline and root banks remain the
authority. Unproved or nonlinear curves retain native bounded curve refinement;
no sample fit or tolerance increase authorizes this fast path.

A scoped affine-boundary regression resolves boundary/inside/outside cases with
zero subdivision allowance; a nonlinear quadratic-trim regression distinguishes
the original curve from its coarse chord cover. Both passed. The independent
interchange owner also observed the repaired native STEP and IGES box surface
imports passing the unchanged `0.1`/`0.5` deflections and default capacities,
with independent area 22 and volume 6. This is scoped interchange evidence,
not completion of the CAD corpus.

This evidence does not close all W02 requirements or the surface-PDE corpus.
Existing curve/network, rational-patch, metric, periodic, source-fidelity and
solver requirements retain their separate acceptance gates.


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

Each bounded improvement pass gives ordinary live sliver stars a
source/quality-admitted interior-removal opportunity before reconnections can
replace their links. Uncertain and determinant-floor entries retain the
existing construction-repair order. A retirement requires the original strict
minimum-dihedral gain, positive orientation, topological link, source/region
and protection checks; no inferior collapse is forced. Actual examined-cell
attempts and committed removals retain their existing counters and original
pass/work ceilings. Successful removals refresh the sliver queue before the
normal reconnection/relocation schedule; final unmet-quality evidence remains
mandatory.

Dimension-one smoothing constructs six bounded candidates on retained exact
or source-bound segment ancestry, checks every incident facet, and uses the
existing atomic relocation and witness admission. Fixed/protected and
nonrepresentable cases remain refusals. Interior removal proposals are ranked
by physical quality; the first source/link-admitted cavity is validated and
committed once through the original owned edit, with scientific-generation and
staging-revision guards. Losing quality candidates do not attempt cavity
admission; an optional shape outside the actual cavity cap is rejected before
attempting it, without resetting any real refusal.

The matching Main integration smoke passed the original multiface,
unprotected vertex-removal (`>=1` unchanged), protected-vertex and scheduled
curve source/quality cases: four passed in 109.18 seconds (142.50 seconds
overall). Selected native PLC/refinement/improvement targets passed three of
three in 1.82 seconds. The observed release source is
`a47f0dbb632a6378e8e0fc320c5c97c31b9031298ca622a7d723b6b96aa327b4`,
with binary
`769d00871b20515a6bd220a38b6107d15cac035ce62f6a6f6bd833f1342d7877`.
These scoped positives do not close the original material/cavity full-phase
qualification, public query/cavity guarantees or final robustness matrix.


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

Tetrahedral circumcenters now evaluate the existing original-coordinate
closed-form determinant and numerators using the shared expansion owner; volume
reporting uses the same exact orientation value. Nonfinite radius-edge
constructions remain queued or explicitly unmet. The retained reconstruction
cell with exact determinant `8.987998638190837e-22` reports positive volume
`1.4979997730318062e-22`, radius-edge ratio about `4.3684467700106616e16`,
and one sub-five-degree sliver. Its fixed source remains unchanged and the
unattainable shape target reports `REFINEMENT_LIMIT`, not an empty success.
This is a numerical/status regression proof, not qualification of the complete
reconstruction positive corpus. Native cap, nested-allowance and original
20,000-work triple-junction proofs retain their original quantities.


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
- `PeriodicRefinement` owns the complete original `source_geometry: CellGeometrySpec`, not only the source mesh and child-reference rows. Its constructor renews the original source bank, verifies independently rounded corner carriers, and binds the full source specification in the history identity. Native/mapped callers pass their actual specification; the bare-mesh topological refinement API alone may explicitly define its canonical affine source. Parent recovery never reconstructs an original specification from rounded vertices or inverts away authored coefficient actions. History and source-recipe identities therefore change honestly when this retained authority is added; old identities are not aliases. Parent/reference/generation banks follow actual published block-concatenated SCI order, including coefficient-action regrouping.
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

### Lifecycle preservation implementation

High-order curving lowers mesh-bound quotient nodes through the finite-element
owner's actual coordinate-element nodal numbering and canonical entity
isometries. It does not build a second geometry chart or merge winding entities
by position. Degree-three and degree-four deformed seam traces are checked in
the actual coordinate basis.

Advancing layers join smooth source sectors through quotient edge incidence,
carry explicit fan/corner and schedule ancestry into generated vertex orbits,
and synchronize thickness reductions and layer termination across vertex
orbits. Layer volumes and caps retain the source identification; cap closure is
measured on the quotient, not the displayed cut. Source feature associations and
material domains remain authoritative in the layer result's existing source
bindings. Scoped triangle and quadrilateral cases cover translational tori,
partially periodic strips, and rotational wedges, including transition pyramids,
physical schedule/volume checks and quotient boundary composition. A convex
ridge crossing a periodic seam retains its fan front without a spurious seam
boundary. The existing optional Gmsh quadratic-node comparison reports absent
comparison dependencies separately from native periodic execution.

These preservation checks do not replace continuous quotient source-fidelity
and periodic-image embedding certification, or construction-time periodic
layer/core filling. Those remain separate owning generation/certificate
obligations of the complete W06 gate.

The native skew-lattice point-source route now constructs representable frontal
sites whose measured hard p50 and p95 lengths equal the authored target exactly;
it does not add comparison slack. Finite-element geometry preparation and
compatible moment projection use fused numerical batches, and the periodic
manufactured forcing retains its lattice coefficients as dynamic leaves.
Repreparation of an unchanged carrier preserves the actual compiled static
structure and continued coefficients.

The geometry-owned periodic image certificate now encloses full polynomial and
rational source maps, enumerates sufficient translation/finite-rotation image
neighborhoods, verifies authored group identities and continuous quotient
traces, and uses the native mapped contact/root-restriction proofs. Its canonical
global certificate publishes image/pair/subdivision counts and coefficient
work/storage evidence. An independently authored curved quad/triangle quotient
strip certifies both mapped self-image embedding and exact two-sided source
fidelity; corner-positive cross-seam overlap and image/pair/subdivision/work/byte
exhaustion are refused. Every source facet, including seam gauges, remains in
the independent reference-partition proof.

`PeriodicCellGeometrySource` now retains the actual FE nodal numbering, original
bare source layout, dynamic representative coefficients and authored exact
isometry expressions. H1 quotient nodes use reference-owner labels and entity
permutations instead of physical nearest-node pairing. Runtime resolves their
RNE images; continuous source-basis bounds account separately for coefficient
rounding without modifying scientific seam equality or physical tolerances.
Degree-three/four deformed strips, representative-coordinate derivatives,
source renewal/reindexing and formerly failing rounded affine publication pass
their scoped contracts. The retained 31-cell prism/tet material layer candidate
also certifies full mapped image embedding under the original default limits,
using exact affine contact and continuous convex two-prism atlas proofs.

Represented translational sources have a source-stratified native CDT image
constructor. Native input-vertex maps carry root/image identity, protected and
material-interface segments are submitted before triangulation, exact rational
circumball bounds certify the image neighborhood, and exact positive-area
source overlaps bind target material ancestry. The material-feature fixture's
region scopes now refer to the containing canonical region entity set.
Its zero-tolerance p50/p95 realization is not yet accepted. Source construction
now removes unprotected fixed long edges and supplies the quotient edge count
required by immutable protected-edge quantiles before proposing coupled native
SQP root moves with authoritative quantile equalities, cell-area inequalities
and the authored maximum size. Explicit corners and native feature/material
segment-incidence vertices remain fixed; unprotected interior target copies
may move while the original source object, coordinates and identities stay
immutable. Source support is renewed by exact original-source classification.
The superseded penalty route is removed. SQP's incomplete detailed counters
are reported as such; admitted and executed-iteration work bounds are charged
without labeling them measured QP operations. The exercised material fixture
still refuses before publication, so the original 120-second/0.05 lifecycle
gate is not established.

`NATIVE_MIXED` now constructs periodic child orbits from exact reference-fraction
weights and source quotient incidences before canonical edit assembly. Its
finite template closure compares paired facet subdivisions in common group
charts, including finite rotations and fixed-axis vertices, without Cartesian
nearest-point matching. Complete coarsening retains parent source-corner
orbits, layer ancestry, retired winding-distinct entity IDs, and allocation
high-water marks. Nonlinear degree-three quad/hex torus and rotated
tet/prism/hex/pyramid restriction/restoration cases passed, as did invalid
cycle, incomplete-family, and exhausted-work refusal. A bounded actual
nonplanar two-material periodic layer/core candidate refined from 31 to 47
prism/tet cells and restored its original lifted and quotient topology and
coordinates, using 5,586 template work units. This candidate smoke used one
synthetic tangential marking group for all eight prisms, not the source's
authored physical column/interval identities; it does not establish their
schedule restoration. Nonlinear rotated-prism H1 and DG
refinement/coarsening passed their canonical transfer certificates; DG content
defects remained below 1.4e-17. A public audit-level unit-hex torus transaction
also refined and coarsened nonconstant Q2 damage history, retained density
inventory 2.0, and continued an independently compiled reaction PDE with a
maximum transferred-solution defect below 1.2e-14. The hierarchy and canonical
orbit witness now retain immutable, variable-arity quotient identity banks;
their new scientific contents intentionally change their fingerprints, with
no historical-field default injection. Multi-stage edits against an unchanged
source carry `PeriodicVertexOrbitWitness.allocation_prior`: the actual previously
bound target and its canonical witness. Assembly replays the entire prior chain
iteratively, checks exact complete keyed banks and cursors, and preserves bare
advanced-bank refusal; it does not infer authority from a larger count. Three
partition stages allocated at cursors 10, 13, and 21, preserving retired keys;
a bounded 1,102-link chain including 1,100 actual numeric relocations validated
without depending on Python's recursion limit. Forged prior and missing-prior
regressions refuse atomically. These template/scalar checks do not
establish accepted layer/core source publication, general mixed
H(curl)/H(div) coarsening, or the reserved full lifecycle gate.

The restored accepted 31-cell PL periodic two-material carrier separately passed
its initial Q2 harmonic diffusion solve (`u = y + z`, periodic in the authored
x-direction), with 85 DOFs and maximum nodal defect `6.22e-15`. Its owning exact
physical forms measured region volumes `0.020201436745944545` and
`0.030010193156276656`; the nonconstant quadratic damage profile remained in
`[0.5, 0.6252961092091588]`. This required authoritative logical labels from
`HybridReferenceFamily.nodal_reference_labels()`, consumed by the canonical
periodic H1 edge/triangle/quad trace dispatch. Rotated prism/tet P2 and
hex/pyramid degree-two/four consumers passed without coordinate rounding or
Cartesian pairing. These are initial-solve/numbering checks, not adaptation
continuation: the actual composite source association-transfer contract still
blocks the public refinement stage. No curved-W09 source claim or full
120-second/0.05 qualification follows.

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

The public source-chart metric route is `MeshAdaptationRoute.NATIVE_SURFACE_METRIC`.
It consumes `MetricMeshAdaptation` or `RelocationMeshAdaptation` with an explicit
`coordinate_contract=source.coordinate_contract`, original
`SurfaceAssociationTransfer` support, and a `CellGeometryTransitionPolicy`
selecting `bounded_chart_deformation`. The retained original fidelity tolerance,
not a new route default, bounds reconstruction and successor publication.
The actual chart deformation accompanies the geometry transition for existing
finite-element and finite-volume transaction consumers.

Eight-point Gauss lengths are edit proposals only. A COMPLETE metric result
requires outward full-source derivative integrals in the original, exactly
partitioned chart metric background. The canonical `phydrax.linalg`
`hermitian_log_enclosure` and `hermitian_exp_enclosure` owners enclose spectral
residuals, scalar-series remainders, and numerical eigenframe nonorthogonality.
Metric arc bounds carry scientific edge IDs and a source/background binding.
Native execution records retain dynamic work, memory, clock, and status leaves;
point-query counts are distinct from native predicate telemetry.

The exact spectral owners accept the original `coordinate_budget` explicitly:
they pre-admit rational term work and conservative CPython object-storage
workspace through `CoordinateEnclosureBudget`, without creating a renewed
private allowance or calling that host bound native measured memory. Callers
release completed transient scopes and recursively retain only genuinely cached
original log coefficients. A scoped recovery proof measured an operator defect
`1.371e-15` inside the certified `7.424e-14` radius, with 1,727 logical exact work
units, 28,500 retained object bytes and a 642,308-byte object-workspace upper
bound. Zero-work and 1,024-byte host allowances refused before coefficient
construction; a huge-radius exponential ball refused before allocating its
integer power. Arbitrary JAX/compiler and interleaved allocator workspace is not
included in this CPython object bound or falsely charged to the native pool.
The ordinary curved-surface arc owner now passes that same active ledger into
the actual logarithm, exponential, norm, and radical kernels. A varying-SPD
actual two-triangle torus-patch smoke certified COMPLETE arc bounds, recording
681,583 logical exact work units, 682,646 actual native work units, and a
944,328-byte host upper bound under its explicitly authored one-million-work
allowance. Public Sphere/Surface execution keeps the original ledger active
through source preparation and final association/fidelity publication, not only
the private producer. The original 640,000-work public sphere fixture currently
refuses its final native actual debit; it is not recorded as a positive result.

That earlier route's narrow observed proof covered a native cylinder remesh,
accepted chart/geometry/global/source publication, and constant and varying-SPD
analytic sphere arc enclosures. It did not at that checkpoint close the full
W08 acceptance gate. The final focused owner closure subsequently reported
32/32 surface-metric and 40/40 tetrahedral-metric checks, passing rational-trim
archive/continuation and curved-sphere smokes, and 12/12 canonical positional
archive-recipe checks. Dense accepted adapted targets reopen through the
existing `accepted_target` role with certification inputs, report,
associations, and one canonical source-to-target lineage event; unchanged
zero-displacement whole-target remaps reuse certified endpoint measures. These
results do not admit approximate carrier/source substitution, relaxed
size/quality/fidelity/resources, or generic topology generation, and they are
not final W15 qualification or release evidence.

The full-sphere material owner exposes
`SphereProjectiveReferenceMap.reference_expressions()` as actual rational
source-reference-to-target-reference arguments from its retained homogeneous
coefficient action and denominator. An actual native 102-triangle sphere
publication was admitted by `SphereMaterialCellAtlas`; an identity actual-cell
map evaluated these expressions exactly at `(1/7, 2/9)`. This verifies the
source-bound algebra interface, not a remesh or field/PDE continuation. Its
positive denominator, Jacobian, boundary and complete overlap partitions remain
with the genuine prepared sphere-piece owner, not an affine corner fit.

`prepare_sphere_chart_compatible_transfer` now consumes those genuine native
pieces for independent source-form and target-test rational pullbacks, with
exact complete target-entity partitions, source trace checks, and the canonical
boundary-preserving interior commuting correction. A native 102-to-64-cell
sphere correspondence retained 614 actual pieces and certified both complete
partitions. Prepared circulation (`k=1`, untwisted H(curl)) and normal-flux
(`k=1`, twisted H(div)) actions matched independent old-material pullbacks
against the actual embedded target physical edge moments within
`1.34e-15`; actual physical trace, returned companion cochain, and algebraic
transpose pairings also passed. The returned map's entire-column commuting
certificate was checked, not inferred from a top-density tensor shape.
The original 100-million work, 256-million-byte host, `1e12` condition, and
32-term rational enclosure bounds were unchanged. Completed per-piece
expression temporaries release their upper bound; only genuinely live chart
coefficients are recursively retained. This proves the compatible consumer,
not the public sphere metric producer or an all-field/history/PDE continuation.

The ordinary UV and Sphere consumers now use one canonical independent-chart
one-form action driver. The duplicated lowest-order-only UV kernel is removed.
UV integration reconstructs exact intersections from the original dyadic chart
cells and retained native candidate routing, not rounded simplex representatives.
Both complete reference partitions, actual full DOF transforms, arbitrary-order
entity moments, boundary-preserving corrections, and the returned two-form
companion are certified together. Full-chart orientation-bundle signs preserve
true twisted flux under reversed source charts; they do not relabel top density
as H(div). An actual embedded reversed-chart flux/cochain/transpose regression
passed, and order-three untwisted circulation and twisted flux independently
interpolated a material polynomial with moment errors below `2.85e-16` and
returned-companion errors below `4.94e-16`. Their whole-column commuting defect
was `2.435e-14`; the original error, condition, and resource requests were unchanged.

Canonical retained coefficient ownership now holds strong references and bounds
its CPython traversal/cache indices. A reproduced recycled-wrapper-ID bug had
counted only 424 bytes for 100 simultaneously live different polynomials; it
could omit actual source coefficients from the resource ledger. After repair,
two actual visits to all 100 banks retained a 121,612-byte upper bound and a
122,076-byte host peak upper bound. An original 10,000-byte policy now refuses
that live bank. Earlier CPython upper figures above and below were observations
with the prior retainer, not current resource proofs; native-pool measurements
are unaffected. The two higher-order consumer and live-bank refusal regressions
passed together. Compiler/device/process memory remains outside these host bounds.

Material field roles now share one original remaining work/query/time allowance
after the actual mesh execution record; no role renews its maximum. Their field
evidence carries the same actual ended `NativeExecutionRecord` as dynamic JAX
leaves. Adding that owning field intentionally changes whole evidence,
transfer-container, PyTree, recipe, and persistence identities, including the
explicit absent-record case. Archives do not inject historical fields or reuse
old identities. A scoped actual native execution charged 17 work units and
three queries, then published and restored its field evidence with unchanged
current evidence identity and exact counters. This proves dynamic record
publication, not an all-role transfer/history/PDE continuation or enforcement
of compiler/device allocations.

The mapped material FV route now binds exact content preparation and CSR
publication to the original `CommonRefinementPolicy` work, pair, and host
memory limits. A stricter stage ceiling overlays the same active coordinate
ledger; it does not alter its authored maxima or renew previous work. A
128-cell actual native sphere identity-correspondence FV smoke transferred
two component inventories with defect `2.14e-15`, used 48,512 logical exact
work units and a 6,927,137-byte host upper bound, and refused an original
one-work policy before modifying the accepted source. The native/mesh-global
allowance handoff and changed-mesh PDE lifecycle remain separate obligations;
these host bounds are not process, compiler/device, or native-pool measurements.

The dedicated `execute_sphere_metric_adaptation` producer now uses actual
material/radial cells, immutable original SPD supports, genuine radial local
operations, and full successor coordinate expressions for final arc and
whole-cell quality enclosures. Its separate `SphereGeometryReconstruction`
record is produced by `reconstruct_sphere_material_cell_geometry`, with actual
corner strata, complete original coordinate degree, source/target bindings,
node rounding bounds, and original validity/certificate controls. It is not a
Surface UV record. Exact small linear actions accept the same supplied
`coordinate_budget`; their dry work/storage upper bounds and actually observed
operation counts are distinct. Native attribution is charged once even across
nested periodic-controller refusals.

A scoped owning producer proof changed an actual 116-triangle native sphere
to 50 triangles using 33 collapses and two flips, retaining 226 exact overlap
pieces and both complete radial reference partitions. The original 640,000
work, 1,280,000 query, 81,920,000-byte workspace and 0.16 source-fidelity
allowances were unchanged. It reported 531,631 logical exact work units,
634,062 native work units, 327 geometry queries, a 45,627,283-byte CPython
workspace upper bound, and maximum fidelity 0.1464466094067357. The source
was unchanged. Its status was truthfully **STALLED**, not COMPLETE: this
is a valid changed private successor, not a converged public unit-mesh or PDE
continuation. Zero-ledger and stale-source refusal regressions also passed.
The CPython upper bound is not native-managed, compiler/device or process RSS.

That first private proof did not exercise organization inheritance. A later
public-route attempt exposed mixed edge-label ancestry. The producer now uses
the canonical closure of every original organization scope to gate collapse
ancestry, and actual vertex-scope classes to gate split supports; no patch,
zone or label is dropped or silently repaired. Its owning regression invokes
the real `inherit_mesh_organization` consumer. The changed proof passes on a
124-to-50-triangle native source with 37 collapses, 226 complete pieces,
maximum fidelity 0.1475575233925261, 434,729 logical exact work units,
544,790 native work units and a 31,359,111-byte CPython workspace upper bound
under the same original allowances. The status remains STALLED. Already
computed genuine validity/embedding certificates are reused only after their
actual geometry, original policy and original certificate limits bind exactly;
their computation is not repeated with a renewed budget.

Exact construction point/face/vertex orbits now have one canonical owner,
`PeriodicConstructionOrbits`, bound to the source topology and numeric coordinate
frame. Mixed-family template signatures compose that owner instead of defining
their own generic orbit algorithm. The actual hex-torus scientific topology and
geometry refine/restore regression passed after the cutover; this is not a
general surface split/collapse/relocation orbit producer or W08 PDE proof.

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
| Existing `phydrax/discretization/fem/_form_elements.py`, `_de_rham.py`, `_mapped_form_transfer.py` | Build H(curl)/H(div) transfer through canonical general FormBasis circulation/flux functionals and their complete entity transformation matrices, with explicit degree, twist, proxy and reference-composition identity. Refine and project complete source patches with certified full exterior-derivative commutation; preserve native solve rank/condition, integration-error and cumulative-budget evidence instead of restoring deleted specialized factories or misusing P1 interpolation. |
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

### Qualified checkpoint and remaining scope

The accepted piecewise-linear 31-cell prism/tetrahedron source is freshly
authored from the original periodic/material controls. All eight authored
identities and mesh identity
`ee5f30e3481c7033a35a00fd61bbc02b346b90e40e7d9164a4dc17229b5b9036`
are preserved; current certificates are regenerated, not copied from historical
records. Warped layer facets retain their original mapped face authority.

The current source archive `.tmp/native_accepted_layer_core31_current` has
content identity
`5f072cdfc211b287338f3a314af731996c8ddc27ae4630b448a1d598efa63d54`.
It restores under unchanged `DEFAULT_ARRAY_ARCHIVE_LIMITS`: 257 members,
nesting depth 16, rank 8, and default byte limits. Its seven members are the
manifest and six dtype banks. Shallow wire recipes preserve the original
logical recipe and aliases. The current transfer archive's semantic identity
`edd20a9908297cd811da07d80267b67cba5a772f44470400e0c03e48d97beb4b`
is identical to the historical transfer despite the wire-layout change.

Actual source adaptation refines 31→35 cells, publishes a durable archive
`de076f8beaf38e58cb07cdc79435b0b9ba31068f0546149d7b5feeb7f041dd40`,
then reopens it in a fresh process and coarsens 35→31. The lifecycle archive
`.tmp/native_accepted_layer_core31_lifecycle` has content identity
`36dabebca2ff40d781ccb224804d1c972dc1021fe70d5e72e7876f1f4fa07e5e`.
Source blocks, cell/vertex IDs, coordinates, and layer columns restore exactly.
Historical larger-limit source receipts remain evidence, not current defaults.

This checkpoint has no selected curved-facet fidelity certificate
(`scoped_fidelity=0`). It does not complete the original curved-source W09 gate.
Original-source H1/Q2, DG1, Hcurl2, and Hdiv2 physical reconstructions have passed,
as has the initial harmonic diffusion solve. The actual durable topology
refine/reopen/coarsen is now proven. Transported history and continued PDE
consumers still require the shared canonical public stage, using the authored
layer columns and intervals rather than synthetic tangential grouping.

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

### Observed layer/core source-composition closure

`NativeLayerCoreSource` retains the complete composite material namespace and
the exact core-local material map. Its periodic core vertex representatives,
integer phase shifts, and explicit oriented seam polygon pairs are validated
before fixed-boundary native fill; no nearest-orbit pairing is used.
Original face controls are composed through explicit source-facet ancestry.
Certification requests retain independently selected original-face queries,
actual target facet global IDs, and their unchanged tolerances in addition
to the whole-source certificate; whole-source evidence cannot replace a
requested selected-face proof.

`prepare_native_layer_association_transfer` binds three disjoint original
namespace plans: composite material ownership, source-core PLC strata, and the
immutable layer realization. `ComposedAssociationTransfer` reuses the existing
PLC lineage algorithm; original warped layer faces are proved by exact
source-root control identity, reference-face equality, and bounded reference
inequalities, not by a planar triangle replacement. Original cap global IDs
and orientation, material maps, and dynamic source coordinate leaves remain
retained through the source and transfer-plan archives.

The actual accepted 23-tetrahedron/eight-prism case has 534 numerical archive
members after complete source-bank publication; its three-plan archive has
143 numerical members. Trusted member capacity 6,144 and depth 32 retain all
generic byte/rank controls. The real marked mixed refinement now passes
source association propagation, current certification, and physical layer
markers. Its inverse coarsening still reaches a separate periodic
sibling/facet-orbit admission refusal; that does not qualify coarsening,
transported history, or continued PDE acceptance.

This case retains a piecewise-linear original source and has no selected-face
fidelity request. It does not close the mandatory original-curved-wall W09
workflow. The native CAD owner subsequently reports that the unchanged original
quadratic-profile extrusion constructor passes its default realized source
route in 129.38 seconds with six original placed-surface carriers. The complete
curved-source scoped-facet hybrid workflow still requires its own consumer proof.

The curved recipe does not inherit the unit-box qualifier's layer schedule or
120-second deadline. Its retained periodic/material counterpart uses two
0.01-thick layers and a 0.03 core offset inside the original 0.05 gap, with
unit translation along the extrusion direction. These source-authored controls
define the curved combined consumer; the unit-box routine remains a separate
qualification. No completed curved adaptation or physical continuation is
credited yet.

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

### Current exact mapped-source family contract

Mapped grid construction admits independently authored positive axis-aligned
affine reference roots with non-unit lengths. Physical size bounds divide each
source derivative column by the exact reference-axis length before directed
interval norm bounds; no sampled inverse or additional size tolerance is used.
Reference inverses now retain exact rational chart controls, including nonbinary
ratios, through the canonical polynomial-composition owner and exact solver.
Coverage preserves the full exact chart geometry rather than rounding its
reference vertices into a replacement affine carrier. An actual axis-length-3
two-material publication exercised denominator 3, nine pure hexes, authored
order 2, certified volumes 3/1.5, embedding and zero source fidelity, Q2 PDE
error below 4.7e-13, and FV residual below 9.1e-16. Ratios outside the canonical
integer-control representation are explicit capability refusals, not rounded
approximations or mathematical impossibility claims.
Region controls must name the complete original root-cell set of their declared
material. Patch controls name original root faces and must agree with their
actual material adjacency. Source restrictions retain the original coordinate
coefficients, including curved interfaces and cavity walls.

`MappedReferenceAssociationTransfer` owns exact source-image strata for host
nested hexahedral epochs. It validates original root expressions, controls,
scientific corner identity, whole-chart containment, shared-entity agreement,
orientation, current independent domain coverage, and actual topology lineage.
It does not project rounded corners onto a replacement linear source.

Family field optimization preadmits its declared iteration/Armijo batch bound
against the current original native execution scope. Only complete, actually
measured objective/gradient counts are charged after execution; the bound is
not published as performed work. Original Jacobian-bound, reference-location,
corner-image, and source-stratum batches consume that same source allowance.
An actual nested mapped publication consumed 70 externally charged work units
and 38 source queries; a subsequent query beyond the remaining original
allowance refused rather than renewing the quota. This establishes those
measured batch obligations, not complete compiler/host scratch accounting.
Host numerical scratch must use the canonical original-root reservation owner.
Selected explicit family host banks now allocate through that canonical owner:
dual node/support tables, grid region/face tables, reference parent/pullback
banks, correctly rounded physical corners, and original-source association
tables. An actual mapped publication and escaped array-view lifetime smoke
measured a 652-byte managed peak; this bounds those actual allocations only,
not untouched NumPy temporaries, Python rational objects, or JAX/compiler memory.

The family qualification helper now distinguishes
`curved-nonsweepable-hex`, `curved-cavity-hex`, and `curved-material-hex` profiles,
with independently authored Q2 source maps and the existing hard pure-family
and size policy. Adding these profiles is not a passed 120-second/0.05 lifecycle
record. A separate non-unit two-material publication smoke produced six actual
hexes, independently certified volumes 2 and 1, embedding and zero-fidelity
coverage, Q2 physical diffusion error below 1.4e-12, and stationary mapped FV
content-rate residual below 1.4e-15.
The unchanged unit-root seven-cell nonsweepable junction and 26-cell cavity
also published actual curved material hexes with certified embedding, coverage,
and zero-fidelity bounds. Their material volumes were respectively 6/1 and
17/9; geometry-aware Q2 diffusion errors were below 5.7e-12 and 1.2e-11,
and stationary mapped FV residuals were below 5.6e-16. Restricted-root embedding
reuses the existing continuous occupied-atlas extension theorem on declared
root topology after exact complete root-trace continuity, not a relaxed contact
subdivision budget. These small publication/consumer proofs do not replace the
frozen Q1 manufactured lifecycle and complete resource qualification.

The current family lifecycle implementation retains H1/Q2, DG1, Hcurl2 and
Hdiv2 scalar/history states through the canonical native spaces and transfers.
Compatible moments use deterministic physical gradients of the authored
scalar/history fields, not an invented RNG state. The original stationary Euler
state carries all five extensive components through conservative mapped-FV
remapping and independently checked original-region inventories. Accepted-epoch
and cold packets retain the actual arrays and case declarations under the
unchanged 257-member, 16-level and rank-8 archive limits. These consumer paths
are implemented, not qualified by the original family campaigns: current
dual-hex source coverage, mixed rational-face restriction and curved
nonsweepable coefficient-work failures still precede accepted field epochs and
fresh-process continued PDE verification.

The original resolution-4 dual-hex lifecycle was actually exercised under
20,000-cell capacity, 120 seconds, three repeats, target error 0.05, and one
refine/coarsen round. Initial publication produced 1,952 pure hexes and 2,450
vertices with accepted original material/cavity coverage. Original Q1 physical
error was 0.01572477, reached at 49.2381 seconds; generation took 25.5189 seconds
and geometry association 17.6163 seconds. The lifecycle nevertheless failed
during refinement at the unchanged geometry-transition basis-work cap
`2**26`, before a refined target, coarsening, or continued PDE was published.
Total measured elapsed time was 165.899 seconds. Prepared exact-basis batching
must close the real work and wall gates; neither increasing that cap nor
reporting the initial solve alone constitutes full qualification.

General quad extraction retains original polynomial and rational spline
coefficients through exact bilinear reference composition, never nodal fitting.
Feature/crease edges separate cross-field patches; curved frames use original
source jets. The existing `parametric_surface` route now publishes an original
nonconstant-weight rational spline quarter-arc extrusion as pure quadrilateral
topology. Its independent source knot-span atlas retains all six original
controls, knots, weights and revision. Whole-map algebraic equality plus
independently declared original UV-domain coverage establish zero two-sided
source fidelity. The exercised public request produced 42 quads with passed
audit/compliance/global/source certification, physical mapped-edge arc-length
evidence, and a geometry-aware surface Q1 harmonic solve with L2 error
`9.101360266624066e-7`; canonical full source-closure restoration preserved the
coordinate-geometry identity and repeated that error.

This positive original-source publication does not establish arbitrary CAD
coverage: exact source charts presently require affine original UV trim
coedges, positive UV orientation, and triangles contained in one original knot
span. Knot-line decomposition across spans, nonlinear/implicit trims and
general reversed source charts retain their separate coverage gates.
Nonpolycube cut templates, arbitrary varying-frame singularity graphs,
immutable-chain recombination, and the full mandatory curved all-hex research
corpus remain open until their actual positive publication, fidelity,
consumer, and resource gates pass.

### Tests, smoke, benchmark and gate

- Add **proposed new** `tests/unit/meshing/test_structured_generation.py`, `test_sweep_generation.py`, `test_multiblock_generation.py`, `test_quad_generation.py`, `test_hex_generation.py`; extend native topology and tensor/AMR/multiblock tests.
- Cases: opposing edge-count mismatch, reversed face maps, periodic block cycles, twisted sweep, fold inside a trilinear cell, extraordinary surface vertices, cavity closure, incompatible boundary parity, trimmed curved blocks, material boundaries and mixed transition interfaces.
- Independent positive all-hex corpus must include nonsweepable domains; pure family and quality requirements are checked on actual output, not requested options. Deliberate template faults must fail independent topology/validity checks.
- Add **proposed new** `examples/native_multiblock_meshing.py`, `native_quad_hex_meshing.py`; smoke generated blocks through existing FEM/FV/AMR consumers with conservative/compatible transfers.
- Benchmark decomposition/field solve, integer/topology extraction, placement, optimization and certification separately; vary controlling block/feature/singularity count and requested cells, not only box resolution.
- Gate: structured/swept/multiblock functionality and general quad/hex research gates have separate results. Full scope is not declared complete while the mandatory general all-hex positive corpus remains unmeshed or uncertified.

### Current swept-map and block-gluing contract

`NativeSweepSource.profile_geometry` retains a complete canonical scalar
polynomial triangle/quad coordinate source, not just its mesh corners.
The coordinate owner supplies the exact anisotropic profile basis times
`{1 - w, w}`. For authored absolute frames, each interval therefore represents
`(1 - w) * (R[k] * X(u, v) + t[k]) + w * (R[k+1] * X(u, v) + t[k+1])`.
This is affine interpolation of transformed source controls in the station
axis, not an inferred rigid interpolation on the rotation group. Profile
coefficients remain numerical leaves, participate in scientific identity,
and survive construction, publication, and FEM/FV coordinate preparation.
The profile's fixed corner coordinates must equal the correctly rounded
owning source corner images; the sweep does not move those corners to admit
an incompatible coordinate map.

Nonplanar exact sweep publication consumes an independently authored
`MappedReferenceDomain` and an explicit canonical
`CellGeometryRestrictionSource` supplied as `root_correspondence`.
Its named `sweep:<profile-block>` rows identify source roots in layer-major
column order. Exact canonical expression equality permits retaining an
independently authored equivalent root basis and its original coefficient bank;
it never permits replacing a curved map by its corners. Source geometry/topology
identities, full root controls, reference orientation, domain partition,
positive Jacobians, embedding, and boundary fidelity are checked by their
existing geometry owners. The published owning basis must meet the requested
geometry order; root binding cannot silently downgrade that hard request.
Generated corners or a generated boundary are never the source authority.
A curved, skew-profile frame twist has been published with certified coverage
and zero source-fidelity bounds; actual FEM linear-gradient reproduction and
FEM/FV volumes agree with the independently integrated source volume.

Conforming multiblock gluing retains every full coordinate element/control
bank and combines common-root ancestry without rebuilding curved maps from
corners. Explicit logical face permutations/flips determine node identity.
Ordinary corner-defined transfinite maps follow the node snapping explicitly
allowed by the interface tolerance; authoritative mapped and higher-order
source coefficients do not. Exact mapped trace continuity independently
rejects a cracked interface even when all corner coordinates agree.
Positive reversed 3D face gluing, curved cell interiors, and an annular block
cycle have scoped proofs. Independent nonconforming parts retain their
revision-bound explicit nodal couplings; a logical face cannot be consumed by
two separate couplings. No proximity-based coupling or duplicate field transfer
is introduced.

These are separate results from the general quad/all-hex research gate.
The transfinite route currently publishes its declared vertex-level
piecewise tensor map with independently bounded fidelity to authored
curves/surfaces; it does not claim an exact continuous curved Gordon–Hall map.
Sweeps currently admit unrestricted owning polynomial `FiniteElementSpec`
profiles, not rational, restricted, or polynomial-composed profile elements.
Revolution evaluates analytic rotations at stations and retains the declared
linear station space between them; exact trigonometric interiors are not
implemented. Exact mapped publication currently requires one authored root
chart for each constructed cell and one root basis per named output block;
arbitrary restriction to coarser/differently partitioned source charts is not
constructed automatically. Lifted periodic profile correspondence, automatic
decomposition, automatic nonconforming interpolation/mortar construction, and
gluing incompatible independent root authorities into one mapped domain remain
explicit capability boundaries. None is supplied by an external engine,
mixed-family substitution, or a hidden linear source surrogate.


## 15. W11 — Native domain-conforming Voronoi, power and polyhedral generation

**Prerequisites:** W01 exact clipping/regular triangulation, W02 domain strata, W03 certified domain decomposition where used, W06 periodicity and W07 for CAD fidelity.

Weighted power geometry binds `CellGeometrySpec.exact_source` to dynamic
site points/weights, carrier points/tetrahedra, and compacted equal-site/carrier
vertex witnesses. The ideal planes use original input expressions
`n = 2(q-p)` and `h = sum(q*q-p*p) + wi-wj`, not rounded plane coefficients.
Canonical host `source_coordinates()` preparation solves rational intersections
through the owning exact small-linear-action substrate and checks rank,
conditioning, carrier containment, all-site power inequalities, and true
round-to-nearest-even coordinates under explicit work and integer-bit bounds.
The numerical `resolve()` coordinate bank remains separate. Exact source facets
must prove planarity before any integration triangulation; triangulating,
projecting, or jittering a nonplanar rounded carrier is not that proof.
Fixed-topology site differentiation is not claimed by this host preparation.

Authored plane splitting retains an
`ExactPowerCellGeometryRestrictionSource`: a dynamic parent construction,
original normal/offset rows, and parent-edge/plane witnesses define each new
vertex through exact scalar intersection actions. Agglomeration retains the
same source geometry and compacts removed coordinate rows by explicit parent
vertex witnesses. Neither operation makes rounded face coefficients
authoritative. Restriction uses the existing convex-cell plane-split admission;
near-coincident cuts may create genuinely thin cells whose original validity
floor or VEM conditioning refuses publication, without jitter or added slack.
FV/VEM consume the exact source integration before numerical publication.
Common refinement proves coverage with rational measures and publishes separate
RNE overlap error bounds, rather than treating a floating coverage tolerance as
an exact source oracle.

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

### Native serial CSR partition implementation and measured limits

`phydrax.graph.partition_graph` owns canonical weighted CSR input, target shares,
integer capacities, the nonempty-part policy and independent Python evidence.
The native serial kernel currently implements:

- Stable component discovery and capacity-feasible component allocation.
  One-part-per-component allocation matches sorted weights to sorted capacities;
  other independent allocations use a deterministic contiguous-part dynamic
  program with `O(C k²)` attempted transitions and `O(C k)` retained state.
  Component candidates are accepted only under the original fine capacities.
- Eight seeded heavy-edge/coarsening starts, recursive bisections grown from at
  most eight seeds, greedy/FM refinement with rollback, and two restricted
  V-cycles. Starts are compared after fine projection by overload, then weighted
  cut; coarse quality is not substituted for the fine result.
- At most sixteen packing-repair rounds and sixty-four exchange candidates per
  vertex. Overload-reducing exchanges keep the destination within its exact
  capacity; required nonempty counts are preserved. Coarse capacity slack is
  never applied to the final input graph.
- At most three adjacent-part re-bisection rounds and two multiway rounds,
  capped further by `refinement_passes`. Each multiway round considers at most
  `k` deduplicated regions of at most eight strongly connected parts, with two
  multilevel starts and one V-cycle per regional solve. Fixed union membership
  makes external crossing edges invariant; publication requires an exact
  weighted global cut decrease, original capacities and requested nonempty policy.

The eleven native counters retain `adjacency_visits` at index 9 and
`candidate_evaluations` at index 10. `work_limit` caps their sum across initial
starts, repairs, rejected proposals and regional solves, including isolated
vertices. These units count visited CSR entries and vertex/candidate attempts,
not CPU instructions, elapsed time or peak bytes. Exhaustion is a resource
refusal, with no partial ownership publication. Python recomputes cut, weights,
counts, capacity status and indivisible-vertex evidence from returned owners;
unresolved balance is reported rather than hidden by changing the request.
This serial resident-CSR qualification does not establish a distributed,
owner-local or gather-free partitioning contract.

The 2026-09-29 parent-run campaign used the fixed command
`python tools/meshing_benchmarks.py --case graph-partition --resolution 256 512 --repeats 1`,
imbalance 1.03, part counts 2/8/32/64, six 2D/3D grid/Delaunay-dual/mesh-dual
families, and unit, vertex-weighted and disconnected variants: 72 cases per
resolution. METIS 5.1.0 with 32-bit indices was an explicitly selected comparison,
not a native fallback. Both routes passed the same independent exact-capacity
and nonempty checks before a cut ratio was considered comparable.

| Resolution | Native accepted | Comparable cases | METIS not accepted | Median native/METIS cut | Maximum ratio | Ratios above 1.03 |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 72/72 | 55 | 17 | 0.921569 | 1.0 | 0 |
| 512 | 72/72 | 53 | 19 | 0.907407 | 1.0 | 0 |

There were no zero-baseline-cut losses. Independent native disconnected-grid
regressions with explicitly constructed cut bounds 192 and 640 passed. The
cut-tail criterion is met on **all comparable cases in this finite campaign**,
not established for the excluded baseline outcomes or other graphs/scales.
All baseline failures remain in the output, and the unchanged campaign-wide
`quality_target_met` is **false** because it requires every baseline outcome to
be accepted. The reported two-resolution command wall time increased from
30.34 seconds before multiway refinement to 57.36 seconds afterward: this is a
quality/cost tradeoff, not a performance win or a native-versus-METIS timing
claim. Memory records distinguish retained graph bytes, traced Python bytes
(excluding native allocations), and cumulative process peak RSS; they do not
establish per-phase native allocation peaks or a memory Pareto advantage.

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
- Add a bounded native moving-overset workflow; smoke moving assembly → hole/donor refresh → accepted transfer → PDE update → restart. The source/archive campaign is qualification tooling, not a public example API.
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
- The native design qualification requires all preregistered uniform, analytic and learned trials to complete under unchanged campaign gates. It reports each observed total cost, represented-surface L2 error, exact affine-triangle integral QoI error, conditioning, DOFs and process-lifetime memory peak. Quadrature-refinement deltas are numerical diagnostics, not rigorous uncertainty bounds; a single ordered campaign does not estimate cost sampling uncertainty or establish learned generalization or superiority.
- Plain mapped global embedding now activates the canonical `CoordinateEnclosureBudget` under the authored certificate work/scratch caps. Cell and pair workspace is released separately from strongly retained exact source/control nets; exhausted preallocation remains unresolved. Certificates record actual source-expression work and peak storage, so their IDs intentionally renew from the former zero-cost records. No schema generation or compatibility path is retained. A captured 125-pair comparison proved unchanged separator decisions and full-control plane sequences before this accounting cutover; the full 452-pair scientific certificate was identical. This is bounded embedding proof, not completion of the physical source-realization event or uniform/analytic/learned campaign.

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

## EXECUTION STATUS — W15 qualification integration

This section records implementation and verification state without changing any
requirement above. No run, success rate, timing, memory improvement, release
authorization, or leadership result is claimed here. Integrated execution is
pending the owning native build, registered routes, and stable caller cutover.

- `tools/_meshing_cases.py` supplies deterministic analytic, source-digested
  planar-hole/embedded-feature, thin-channel, and cavity/material/internal-curve
  PLC cases. Deliberate stale-revision and capacity cases are negative cases;
  their expected refusals never satisfy a positive workflow.
- `tools/meshing_qualification.py --scenario native-lifecycle` executes native
  preparation/publication, independent source-domain coverage/global embedding,
  hard requested sizing/quality and embedded-feature checks, canonical
  adaptation, actual P1 state transfer and a manufactured diffusion solve.
  Transferred state seeds the next solve through the correction equation.
  Failed acceptance or native refusal remains a failed positive case.
- `--scenario native-corpus` preserves each attempt and the complete thirteen
  workflow ledger. Presence of the current routine cases does not close the
  larger mandatory corpus or any missing combined-control workflow.
- `tools/meshing_benchmarks.py --case native-lifecycle` varies resolution and
  entity/work capacities, uses isolated repeated attempts with a hard process
  timeout, and retains failed/timed-out attempts alongside accepted samples.
  Lowering, compilation, first execution, warmed samples, compiler memory,
  logical retention, process peaks, backend-reported device memory and
  end-to-end time to a fixed physical error are separate evidence fields.
- Optional Triangle comparison is explicitly selected by
  `--scenario comparison-triangle-lifecycle` or benchmark `--comparison triangle`.
  It runs Triangle as its primary generation algorithm and the same independent
  source/control checks and common native adaptation/transfer/solver consumer.
  It is not a native fallback. Missing Triangle is `missing-dependency`.
  Matching source/request identities alone do not establish equal construction
  work/scratch/query guarantees or authorize a leadership comparison.
- Neurofluid tools provide `--scenario native-generated-transport`: occupied
  labeled-image source → canonical native result with exact region evidence →
  Neurofluid admission/re-admission → bulk/network transfer → transport loop.
  Independent material/interface geometry, local/compartment/total inventory,
  positivity and host continuous-time physical-error checks accompany native
  lifecycle identities. Benchmark attempts support hard cancellation; failed
  preparation, generation and scientific outcomes remain evidence records.
- Built-in native capability entries bind exact conjunctions rather than a
  Cartesian product of flags. Source-inspected implementation/admission,
  mandatory positive completion, independent scientific evidence and
  release/leadership authorization remain separate facts. Internal capability
  profile generation fields are being removed through the canonical registry
  and caller cutover; external provider releases and standards remain explicit
  boundary metadata.

The following invocations are requested verification commands, **not observed
results** (run from the worktree with the newly built in-house kernel):

```sh
JAX_ENABLE_X64=1 python -m tools.meshing_qualification --scenario native-lifecycle --corpus-case planar-hole-feature --resolution 4 --capacity 20000 --timeout 120 --repeats 3 --target-error 0.05 --adaptation-rounds 1
JAX_ENABLE_X64=1 python -m tools.meshing_qualification --scenario native-lifecycle --corpus-case plc-material-cavity --resolution 4 --capacity 20000 --timeout 120 --repeats 3 --target-error 0.05 --adaptation-rounds 1
JAX_ENABLE_X64=1 python -m tools.meshing_qualification --scenario native-corpus --resolution 4 --capacity 20000 --timeout 120 --repeats 3 --target-error 0.05 --adaptation-rounds 1
JAX_ENABLE_X64=1 python -m tools.meshing_benchmarks --case native-lifecycle --corpus-case planar-hole-feature --resolution 2 4 8 --capacity 20000 40000 --repeats 3 --timeout 120 --target-error 0.05 --adaptation-rounds 1
JAX_ENABLE_X64=1 python -m tools.neurofluid_qualification --scenario native-generated-transport --size 2 --maximum-cells 4096 --maximum-vertices 4096 --oracle-capacity 2048 --steps 5 --dt 0.01 --repeats 3 --target-error 0.05 --balance-tolerance 1e-9
JAX_ENABLE_X64=1 python -m tools.neurofluid_benchmarks --scenario native-generated-transport --size 2 --maximum-cells 4096 --maximum-vertices 4096 --oracle-capacity 2048 --steps 5 --dt 0.01 --repeats 5 --attempts 2 --timeout 120 --target-error 0.05 --balance-tolerance 1e-9
```

All thirteen mandatory workflows remain required. In particular, an immersed
surface area/integral check is not a surface PDE; direct structured construction
is not independently accepted provider publication; an affine dual-hex example
is not the complete curved/nonsweepable all-hex corpus; stored partitioned
arrays are not distributed generation/refine/coarsen/restart proof. Research
gates for singular CAD arrangements, underspecified defective-surface
reconstruction and general all-hex remain incomplete until their prescribed
positive evidence exists. No safe refusal is being relabeled positive
completion, and no current tooling entry closes an unexecuted owner workflow.

### Subsequent W15 integration state

The canonical `CapabilityProfile` now contains no internal generation argument,
attribute, serialized field or restoration read. Implemented/admitted native
conjunctions use actual typed candidate profiles with `released=False`; mandatory
unclosed combinations remain explicit obligations, not qualified entries.
Actual constructor and semantic consumers were migrated, including battery
release/OED identity, commercial MPM/category identity and explicit external HEP
provider release metadata. Historical signed/generated battery evidence remains
historical; it cannot be converted into current release evidence by changing
dates or subject IDs. Generator-owned data requires the owning normal
regeneration and integrated verification after the source cutover.

Native phase observations now use the implemented execution-only
`NativeMeshingPhaseMeasurement` callback at provider preparation/execution.
Drivers preserve only emitted records and actual invocation/work counters.
Missing phases are unmeasured, not zero; inclusive construction intervals are
not summed with their contained subphases. The channel does not add timing to
source, specification, plan, trace, result or scientific identity.
Runtime records include the actual loaded C-API source digest, loaded binary
digest and compiled configuration, together with the Python package actually
imported and driver-source identity; a checkout digest is not used as a
substitute for a loaded native build.

Two additional registered scientific drivers are available for integrated
execution. The surface and benchmark commands below remain separate
qualification obligations; the original polyhedral qualification is observed
passing as recorded below:

```sh
JAX_ENABLE_X64=1 python -m tools.meshing_qualification --scenario native-surface-pde --resolution 4 --capacity 20000 --timeout 120 --repeats 3 --target-error 0.05
JAX_ENABLE_X64=1 python -m tools.meshing_benchmarks --case native-surface-pde --resolution 2 4 8 --capacity 20000 40000 --repeats 3 --timeout 120 --target-error 0.05
JAX_ENABLE_X64=1 python -m tools.meshing_qualification --scenario native-polyhedral-lifecycle --polyhedral-source polyhedral-material-L --resolution 2 --capacity 20000 --timeout 120 --repeats 3 --target-error 1e-7
JAX_ENABLE_X64=1 python -m tools.meshing_benchmarks --case native-polyhedral-lifecycle --polyhedral-source polyhedral-material-L polyhedral-stretched-L --resolution 2 4 --capacity 20000 40000 --repeats 2 --timeout 120 --target-error 1e-7
```

The surface driver performs an actual zero-mean scalar H1 Laplace–Beltrami
solve and independent radial-lift error measurement, not an area substitute.
Its hard size/source-deviation requests are distinct from any earlier
soft-sizing source-owner smoke and require their own run. It explicitly does
not qualify surface state adaptation. The polyhedral driver uses the native
two-material nonconvex power-cell route and canonical organized-source
regeneration, preserving source patch and material organization. Exact-source
common refinement transfers material inventory; an accepted-boundary
composition rebind publishes both physical state and history before continued
mimetic FV response. VEM and FV consume the same exact geometry owner.
The unchanged original command above passed with complete mandatory workflow
completion in 20.663169 seconds against the 120-second bound, physical error
`1.0658141036401503e-14` against `1e-7`, material inventories `4 → 4` and
`5 → 5`, and zero consumed inventory defect. Its actual published physical/
history receipt is
`b4925d48578e356b61cc7e12276cb03d72f153908287690c67b7cc7a0beba702`.
Resolution 2, entity capacity 20000 and three repeats were not changed.
Exact overlap work admission bounds actions before execution and charges
executed actions, not hypothetical plane-triple solves rejected by exact
separation. Soft-size requests, site derivatives, the larger benchmark corpus
and release/leadership claims remain explicit nonclaims.

Exact positive prerequisites remain: represented planar/PLC/envelope
associations need an owning authority-aware native lineage transfer rather
than the current B-Rep-only admission; implicit volume extraction must publish
a source-conforming result rather than a preparation workset; periodic
provider publication must be joined to orbit-preserving field-state/solver
closure; native distributed generation/refine/coarsen/repartition/checkpoint/
changed-placement restart must be joined to continued owner-local physics;
and design-state provider realization must be joined to accepted physical
reanalysis. Native CAD and repaired-envelope consumer helpers are retained
unregistered until their exact source/adaptation prerequisites and positive
runtime evidence are available. These facts keep the complete thirteen-workflow
gate incomplete rather than weakening or relabeling its requirements.

### Active mandatory caller joins and observed hard-control outcome

The earlier missing-driver ledger is superseded by concrete callable wiring:
`native-cad-lifecycle`, `native-envelope-lifecycle`,
`native-periodic-lifecycle`, `native-design-lifecycle`,
`native-family-lifecycle`, `native-hybrid-lifecycle`,
`native-implicit-lifecycle`, `native-distributed-lifecycle`,
`native-overset-lifecycle` and `native-image-lifecycle` are registered beside
the planar/PLC, surface PDE and polyhedral drivers. Their actual owning
generation, certification, source/state transfer, solver, motion or restart
APIs are called; scientific-owner gaps remain failed positive evidence, not
manufactured success. In particular the restricted implicit volume driver now
consumes the implemented native `CellMeshingResult` publication, never a
preparation workset. The periodic caller consumes owner-supplied coarsening
witnesses and compatible transfer. The overset caller generates both native
parts and retains original source, request, certificates and actual accepted
state through its source-aware motion and scientific archive.

The current explicit verification library is
`.tmp/build/ResumeIntegration/libphydrax_meshcore.dylib`; the installed virtual
environment copy was reported stale. Select the current library explicitly,
not by silently using the installed copy. A bounded per-workflow invocation is:

```sh
JAX_ENABLE_X64=1 PHYDRAX_MESHCORE_LIBRARY="$PWD/.tmp/build/ResumeIntegration/libphydrax_meshcore.dylib" .venv/bin/python -m tools.meshing_qualification --scenario native-periodic-lifecycle --periodic-case skew-lattice --resolution 4 --capacity 20000 --timeout 120 --repeats 3 --target-error 0.05 --adaptation-rounds 1
```

Additional frozen controls include CAD `--cad-format step|iges|brep`, periodic
`--periodic-case skew-lattice|skew-material-feature`, family
`--family-case dual-quad|dual-hex|transfinite-quad|multiblock-quad|sweep-hex`,
hybrid `--layer-count`, `--first-thickness`, `--layer-growth`,
`--geometry-order`, implicit radius/fidelity/discovery/root/refinement
capacities, overset motion/wall/donor/overlap capacities, and actual distributed
placement/halo/cavity/slack/checkpoint controls. Design decisions consume
explicit `--maximum-memory-bytes` and `--maximum-condition`. Qualification
and isolated benchmark drivers expose the same controls; the benchmark
enforces a hard process deadline and retains every failed/timed-out sample.
The corpus orchestrates the available thirteen workflow categories in
isolated bounded attempts rather than omitting unverified caller joins.

`native-distributed-lifecycle` is a two-role scenario. A single-process
invocation with a fresh shared `--distributed-checkpoint-root` launches
`--distributed-processes` producer processes through the canonical local
launcher with the same frozen request; each producer joins the distributed
runtime from the PHYDRAX rank environment, owns exactly one accepted
partition, generates the native unit square, accepts Morton-partitioned
owner-local refine/coarsen/refine epochs, projects every physical role into
its stable-ID allocated history bank, replays a native graph restart
repartition, recovers all roles across it, and publishes its process-local
checkpoint. The invoking process then restores every owner with the changed
`--restart-parts` owner count, recertifies the retained source, repacks the
allocated forest, recovers every role, rearchives the closure, and continues an
FE Dirichlet solve and an owner-local FV advance from the reopened archive:

```sh
JAX_ENABLE_X64=1 PHYDRAX_MESHCORE_LIBRARY="$PWD/.tmp/build/ResumeIntegration/libphydrax_meshcore.dylib" .venv/bin/python -m tools.meshing_qualification --scenario native-distributed-lifecycle --resolution 1 --distributed-processes 2 --distributed-parts 2 --restart-parts 1 --distributed-checkpoint-root /shared/fresh-root
```

The archived closure uses only the canonical source-closure roles: accepted
data carries exactly `fields` and `adaptation_results`. The accepted adaptation
policy is the retained result's `policy`; earlier topology epochs are reached
through the collective target's evidence chain, not a second accepted-data
role.

**Observed qualification failure, not completion:** the hard
`native-surface-pde` unit-sphere request at resolution 4/capacity 20000, h = 0.4,
median relative tolerance 0.5, maximum edge 0.6 and two-sided deviation 0.04
failed `target_size_p50` compliance before the PDE. The measured median edge
was 0.17352029242384795, below the required 0.2; maximum edge
0.39018064403225655 and rigorous deviation bound 0.03974638772319786 satisfied
their separate requests. The run reported 149 refused points and incomplete
refinement. Its actual callback/source/binary/configuration evidence was
retained. The hard requests were not relaxed, and the earlier soft-sizing
sphere/PDE smoke is not credited to this hard profile. The sizing owner was
given the failed-case evidence for correction. No leadership, release,
complete positive corpus, or complete thirteen-workflow outcome follows.

### Observed native overset routine and remaining physical failures

The authorized **post-cutover** `native-overset-lifecycle` routine completed
with exit 0 at resolution 1, capacity 20000, two motion steps, one warmed
repetition, timeout 120 seconds and wall/donor/overlap pair capacities 500000.
Its actual native sources, strict carrier publication, wall/donor refresh,
prepared field queries, refused and accepted composition rebinds, carried-state
Poisson updates, canonical source/request/physical-declaration archive and
restored-owner continuation were exercised. The measured residual/boundary
QoI was 1.9152779376365187e-13 and restart continuation error was zero. This is
**not** an analytic L2 error or a conservation claim for moving FE interpolation.
The separately certified stationary cell-average overlap profile had zero
inventory defect.

Evidence: `.tmp/overset-native-lifecycle-field-declarations.json`.
The canonical checkpoint contained 2307014 bytes with content identity
`d89a740e03dd12cf2b265a69cf72f8e8940b8cd413e24d1c80b489ff9b83477b`.
The loaded native source digest was
`41d0a0b9363753d3a590b52ff08dfc62c9d5e9633c745ffdcbb93d97e3e1a9bc`,
binary digest
`52528cf4862bec13622164ea1c89383f8dea9cd83af8cf2ad7aef0ac4a5d8e93`.
The single routine measured 85.2645241250284 seconds end-to-end and a
2851356672-byte process-lifetime peak; neither is a repeated comparative
benchmark or a leadership result. Earlier residual and nonfinite-static
archive failures remain historical evidence; their owning policy/codec fixes
did not relax the independent acceptance threshold or strip carrier evidence.

The repaired-envelope driver binds the actual owning source-table and
represented-PLC association transfer. An earlier changed-kernel attempt
achieved strict publication/global embedding/domain coverage, measured volume
0.1515250772265342 and complete 1052/1052 source-facet coverage before failing
the scalar H1 metric admission. Subsequent owning evidence (`artifact://2590`)
reports epoch-zero H1 solver success after 111 iterations and independent L2
error 0.001597291864532493, followed by a real `NATIVE_MIXED` adaptation refusal:
no family-preserving template matches all oriented shared-face subdivisions.
The frozen 100000-cell cap remains unchanged; global 24-child subdivision is
not an admissible replacement. The native face-preserving template owner must
close that prerequisite before the transferred-state solve can be credited.

Dual-hex source publication now proves actual canonical Q1 point, whole-edge,
whole-facet and whole-cell PLC membership against explicit source role/index
authorities before issuing fresh exact associations. A genuine dyadic rigid
translation of the current resolution-2, 712-cell source preserved authority/incidence tables
and domain-region identities, proved coordinate-polynomial coefficient identity,
and passed fresh geometry/coverage/stratum support plus an independent strict
complete-association watertight audit. The public volume transition dispatch also
passed an actual tetrahedral source translation with all four exact, complete
role/index association tables and parent provenance preserved. Source-class
revalidation also passes the actual source audit validity object and exact
accepted certification-request limits together with its embedding certificate
to the canonical prepared-evidence owner. An actual native tetrahedral source
and fresh-target identity-propagation smoke passed all four complete exact
strata with this paired evidence. These observations
do not credit the unchanged dual-hex lifecycle: the original resolution-2,
712-cell scalar PDE error 0.08438428094646884 misses the frozen 0.05 target.
A separate resolution-4 initial solve reached 0.03497457 but hit native
mixed-template closure at the first refinement epoch. Mixed refine/coarsen and accepted distributed
reconstruction therefore remain open thirteen-positive-workflow gates.

Owner-local planar PLC association transfer consumes current canonical global
embedding and domain-coverage premises through the `embedding` and `coverage`
keywords of `PlcAssociationTransfer.propagate`. Both certificates must bind the
actual target coordinate map, authoritative domain/region identities and the
collective partition evidence; local polynomial/stratum support still runs.
Accepted source classes consume their current coverage theorem rather than
rerunning whole-domain coverage on a partial owner carrier. An actual native
planar source smoke passed all three complete exact association tables with
the explicit target premises and rejected a source certificate presented for
translated target coordinates.

`geometry._collective_domain_coverage.certify_collective_premises` is the
fresh-target producer of those premises. Every owner calls it collectively with
its owner-local target, the original serial affine source with its certified
embedding/coverage and region request, and the restored numerical epoch. It
checks the complete disjoint dyadic partition, exact root restriction charts,
reciprocal facet incidence and source-boundary contact on the logical banks,
certifies the actual target maps with the canonical validity owner on every
partition, and transports exact root integrals; bank failures refuse. In an
actual two-process native planar refine, both owners obtained certified
validity, embedding (4 cells, 4 boundary facets) and coverage (achieved 1.0,
error 0.0, no subdivision) bound to the storage partition evidence. This does
not credit the cold distributed physical continuation gate.

### Observed hard surface PDE correction

The unchanged resolution-4/capacity-20000 unit-sphere workflow now completes
with hard h = 0.4, maximum edge 0.6, median relative tolerance 0.5 and two-sided
deviation 0.04. The accepted carrier has 216 vertices and 428 triangles; cell
validity, topology, quality, global embedding and source fidelity are certified.
Both directed deviation bounds are 0.02596559454955228. The actual zero-mean
Laplace–Beltrami solve has radial-lift L2 error 0.04183681101707906 against the
unchanged 0.05 target, mean 2.2451120315840452e-17 and successful native solver
status after 74 iterations.

The phase-diagnostic run measured 45.18909987504594 seconds for the workflow
and 93.05 seconds for the process including import and diagnostic instrumentation.
Lowering took 4.042996791074984 seconds, compilation 2.107307124999352 seconds;
three warmed solves took 0.061905583017505705, 0.053019083105027676 and
0.061366792069748044 seconds. Compiler temporary/output/argument bytes were
1262248/3766/45588, logical retained bytes 622593 and process-lifetime peak RSS
1783578624 bytes. These are one run's observations, not comparative leadership.
Evidence is `.tmp/native_surface_pde_postfix.json`; the loaded source/binary
digests are retained in that record. A preceding concurrent invocation timed
out at the hard 180-second launcher limit without a result; that failure remains
historical evidence. This positive PDE case does not close surface adaptation,
the full curved/trimmed corpus, or the thirteen-workflow completion gate.

### Observed resumed periodic hard-control refusal

The documented skew-lattice periodic lifecycle at resolution 4, capacity 20000,
three warmed repetitions and one adaptation round fails specification
compliance before its solver. The unchanged hard target is 0.25 with maximum
edge 0.375. Measured median and 95th-percentile edges are
0.2576941016011038 and 0.3125; minimum/maximum edges are 0.25/0.3125.
Quotient measure is 0.9999999999999999, coordinate-lift residual is zero, and
the native embedding check uses nine images and 37 candidate pairs. Those
source checks do not establish target-statistics compliance.

The failed workflow takes 4.505857417010702 seconds and reports a
1232699392-byte process-lifetime peak. Native phase observations and loaded
source/binary/configuration identities remain in its failure record. The
periodic owner must realize the original hard request through bounded,
orbit-preserving generation/adaptation; no seed replacement, tolerance
increase or soft-control downgrade is credited as a correction.

