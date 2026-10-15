# GPU spatial search and hierarchical execution: implementation plan

## 0. Authority, baseline, and evidence boundary

This is a file-by-file design and implementation plan. It covers reusable nearest-neighbor preparation, device BVH construction, real hierarchy refitting, bounded dual-tree scheduling, fused FMM execution, and separately gated Laplace translation improvements. The implementation record below states what was delivered; performance statements are limited to the recorded gates.

- Implemented on branch `feature/cuda-provider-hardening` (base `87a9e9e73`, later merged with `dev` at `e7b82d23c`) alongside the CUDA provider work it builds on.
- File inventories below are against the current working tree. Repeat LSP references and supplement configuration/artifact/example inventories immediately before each public cutover. Worktree-path LSP queries returned incomplete references; main-checkout queries supplied the project-wide inventory for unchanged base contracts, followed by reads of the worktree implementations.
- Files marked **new** are proposed, not existing capabilities. No native CUDA/HIP dependency is added by the required implementation.
- Performance proposals must pass the stated gates. A failed experiment is removed completely; it must not leave a placeholder, unused dispatch path, misleading backend name, or unsupported public selector.

### Implementation record

Decisions that differ from or resolve the plan text below:

- `MortonSourcePlan(address_plan, point_capacity)` takes no dtype and carries no epoch; the prepared artifact keeps the input precision, and `refresh` refuses dtype changes.
- Exact small-k selection failed its gate on the RTX 5090 (slower than the full sort at the default candidate envelope), so the full sort is retained; there is no `_nearest_selection.py` and no stable-ID rank metadata.
- `BVHSourcePlan`, `PreparedBVHSources`, and the BVH query types live in `discretization/spatial/_bvh_neighbor_query.py`, not `_neighbor_sources.py`. The MEDIAN layout keeps only O(depth) static metadata; topology and per-level segments are generated on the device. The default BVH target chunk fills a fixed slot budget.
- The exact nearest traversal and the bounded leaf-candidate traversal in `_bvh.py` select branch results arithmetically instead of through a batched `lax.cond`, which broadcast every closed-over tree array to the query lane count; admission filters only visited leaf items.
- The `frontier_widths` ladder ships as an explicit option with full-width default: on 20,000 points it cut warm traversal time 4–6x and temporary bytes 2x against full width with bit-identical relations, but tripled compilation time.
- Far M2L streaming and the Pallas leaf-pair fragment kernel ship; the JAX route stays the default. The register-local multipole Pallas experiment was not attempted.
- Zero-shift rotation M2M/L2L return their input exactly (complex GEMM kernels round identity products) while keeping the contraction tangent.
- The rotation route applies Wigner blocks and axial tables to coefficient vectors (no per-pair translation matrices) with an application-level custom JVP, and far M2L runs in the existing 256-event batches. Measured on the RTX 5090 it compiles 4–5x faster than the dense route at order 4, evaluates order 8 where the dense route did not compile within 30 minutes, and evaluated 2.7x faster at 2,048 sources, while its strength VJP was 2.4x slower there; dense stays the default.
- The FMM near/far streaming changed 20,000-particle float32 evaluation from about 63–78 s to 11–12 s warm (temporary bytes 261 MiB to 56 MiB), and the monopole field route from about 115 s to 10 s (JAX) or 7 s (Pallas fragment), matching the previous outputs to float32 rounding.
- Device-built BVH point queries (3 × 4,096 targets, k = 8, 3D) took 30–53 ms at 65,536 sources (uniform/clustered) and 32 ms at 262,144, against 407–2131 ms for the dense blocked query and 160–205 ms for Morton (which refuses the clustered cloud); device preparation took 11–45 ms against 0.6–2.6 s for host `prepare_bvh`.

### Prior evidence to reuse, not rerun merely to restate

The preceding investigation exercised JAX 0.11.2 on the installed NVIDIA GPU:

1. Morton kNN matched an independent NumPy reference on uniform float32 data: 2,048 sources, 64 targets, k=8. The derived candidate capacity was 929; the largest visited candidate count was 245.
2. The same plan refused 2,048 coincident sources with `CANDIDATE_OVERFLOW` on every target and returned no valid incomplete neighbors.
3. Existing host-built `PackedBVH` nearest queries and device-refitted queries matched NumPy under JIT. For 2,048 coincident points, leaf size eight, the BVH returned logical IDs zero through seven.
4. Portable upstream `jaxkd` matched a random 6D reference, but selected IDs `[2, 4]` rather than `[0, 1]` at a tied boundary. Its radius query was strict. Its coincident Euclidean-norm derivative was non-finite; that norm is nondifferentiable at zero.
5. The prior 20,000-particle FMM run found no speed benefit from Phydrax's current pointwise Pallas route. Those timings are evidence for that specific tree state, capacities, order, precision, and environment, not a prediction for the proposed kernels.

These probes establish current behavior, not comparative upstream throughput. Their throwaway scripts were removed. All new implementation inputs invalidate the relevant evidence and require new qualification.

### Pinned external references

| Reference | Inspected commit | Reusable ideas | Do not import unchanged |
|---|---|---|---|
| [jaxkd](https://github.com/dodgebc/jaxkd) | `96a229525bda6af415406921e2b572c3845a6a3f` | Capacity-fixed median construction, compact point-query organization | Traversal-order tied membership, strict radius boundary, unbounded query batching |
| [yggdrax](https://github.com/TobiBu/yggdrax) | `2758fa9b8f84aaf0250926e42e23edfc11c326b8` | Duplicate-aware radix construction, bounded wavefront widths, mutual pair scheduling | Host retry ladders, silent depth clipping, unchecked receive capacity, uncertified reuse |
| [jaccpot](https://github.com/TobiBu/jaccpot) | `5bc4959f41ff57f3955e5e3e373dd969a0387000` | Direct sorted-leaf kernels, bounded CSR chunks, register-local translations, rotation-covariant derivatives | Atomic-order accumulation, environment-driven route selection, acceleration-only potential VJP, step-count-only cache reuse |
| [jaxkd-cuda](https://github.com/dodgebc/jaxkd-cuda) | `d9e37557b9c823caf6962809825f11599934e1ca` | FFI integration cautionary reference | Builder allocation/synchronization, unchecked leaf split-dimension reads, float32-only bindings |
| [cudaKDTree](https://github.com/ingowald/cudaKDTree) | `37ce4c377170d13670632143b976192d5b69ed03` | Small-k candidate containers and memory/build tradeoffs | Input mutation, float32 ranking for wider data, non-XLA-stream operations |
| [cuBQL](https://github.com/NVIDIA/cuBQL) | `20f9db19cdb160e2b83d8f05dfdb0437fe11fe24` | Primitive BVH queries, equal-key leaf splitting, shrinking-radius organization | Host-synchronized builders, global allocator changes, silently truncated stacks, sample float32 ranking |

Source mechanisms to consult: `jaxkd/tree.py:257-385`; cuBQL `builder/cuda/radix.h:432-466` and `traversal/shrinkingRadiusQuery.h:272-337`; cudaKDTree `cukd/knn.h`; yggdrax `_interactions_impl.py:5012-5520`; jaccpot `pallas/nearfield_leafpair_csr.py`, `pallas/m2l_real_csr_lanes.py`, and `operators/_transverse_degeneracy_jvp.py`. Conceptual reuse does not authorize copying code without preserving applicable MIT/Apache notices. Add entries to `NOTICE` and `LICENSES/` only for code actually incorporated.

## 1. Scope ledger and decisions

| ID | Required end-to-end result | Phase |
|---|---|---|
| S01 | Explicit reusable source artifacts for exact Morton kNN, radius relations, and shell witnesses | P1 |
| S02 | Graph, particle, meshfree, halo, PTV, gravity, and distributed caller cutover | P1, P7 |
| S03 | Exact small-k selection preserving full-width identity, certification, and refusal | P1, performance-gated |
| S04 | Capacity-fixed JAX MEDIAN BVH construction and an explicit exact BVH point-query route | P2 |
| S05 | Real source-order/hierarchy refresh that skips rebuilding when its invariants hold | P3 |
| S06 | FMM geometry refresh with current far-pair admissibility and atomic acceptance | P3, P5 |
| S07 | Bounded accepted-pair storage and width-aware dual-tree work without altered coverage/order | P4 |
| S08 | Streamed JAX and actual CUDA Pallas near-field evaluation, including potential derivatives | P5 |
| S09 | Bounded far-message execution through the existing sparse streaming owner | P5, performance-gated |
| S10 | Analytic Laplace translations in native coefficient conventions with complete admitted derivatives | P6, numerical/performance-gated |
| S11 | CPU/GPU numerical, AD, resource, compiler, identity, and consumer proof | Every phase |
| S12 | Clean public/representation cutover, authored docs/examples, deterministic generated data | P7 |

### Binding design choices

1. Existing owners remain canonical. No tree registry, generic provider plugin framework, duplicate CSR runtime, global numerical cache, or second array/axis vocabulary.
2. Source preparation has its own lifecycle. Use `MortonSourcePlan.prepare` and `BVHSourcePlan.prepare`; do not add identical `prepare_sources` forwarding methods to every query plan.
3. Query plans consume prepared artifacts only. Remove the old raw-source low-level query signatures in one complete cutover; no raw/prepared shim overload on those methods.
4. `graph.query_neighbors` keeps its raw dense/blocked behavior and may also accept a prepared artifact in the source argument with an explicit matching plan. It never accepts redundant raw and cached coordinates together.
5. MEDIAN is the first device builder because its topology depends only on capacity and leaf size. Host Morton/SAH preparation stays valid and host-only. A dynamic-layout device radix builder is not required by this plan.
6. BVH point queries are explicit. An overflowing Morton query still refuses; no automatic fallback changes its resource or execution contract.
7. The initial periodic BVH lower bound is zero on periodic axes and the ordinary conservative AABB bound on other axes. This is exact but weak; fully periodic queries may approach a full scan. Morton remains the recommended periodic route. Tighter periodic pruning is a separate proved optimization, not an assumed epsilon margin.
8. Existing stable route/reduction order is preserved where the interaction partition is unchanged. Do not replace route IDs with endpoint-derived IDs merely to make refit and fresh rebuild cuts comparable; those cuts need not match.
9. Scientific formulas stay with their owners. Backends own loads, tiling, execution, and seeded accumulation, not a competing gravity or harmonic definition.
10. The whole `UniformFMMPlan.evaluate` route remains reverse-mode-only unless separately extended and qualified. The monopole/fragment route must retain its existing JVP/VJP and mixed-derivative behavior. Do not infer forward-mode support from a `custom_vjp`.
11. Public and representation changes are intentional feature changes, not identity-preserving typing cleanup. Do not keep historical fingerprints by retaining obsolete payload keys or ignoring added numerical fields.
12. Defaults change only after a measured gate. No throughput claim follows from a CUDA selector, interpret-mode parity, or an upstream README.

### Non-goals

No all-metrics ML-neighbor rewrite; approximate nearest neighbors; new periodic-image enumeration; new integration/block-step solver; wholesale geometry/meshing migration to GPU; new MPI transport; native FFI dependency; default TF32; arbitrary unqualified high expansion orders; automatic HIP/TPU/multi-GPU acceleration claims. Existing host validation, exact predicates, cell lists, Verlet caches, sparse layouts, and domain-owned support policies remain in force.

## 2. Contracts common to every phase

- New concrete classes inherit `StrictModule`, are final by default, and validate metadata before assignment. Use runtime-resolvable `phydrax.typing` forms, nominal dimensions, a fresh `Scope` per operation, and `__strict_contract__ = True` where required.
- Array coordinates, IDs, masks, topology, numerical tables, and status remain dynamic leaves. Static metadata is immutable and absent from device runtime buffers. Fixed-role metadata, not static numerical storage, excludes conditioned arrays from parameter discovery.
- Source capacity, target capacity, case axes, node axes, and physical coordinate axes are distinct contracts. Scientific identity comes from the declared address/support/IDs, never equal dimensions or display strings.
- Floating geometry supports explicitly selected float32/float64. Preserve full supplied integer ID values/dtypes separately from narrowed execution ranks. Constructor/shape/kind mismatches raise the owning `TypeError`/`ValueError`; data-dependent invalid geometry and resource failure reach the result as evidence.
- Discovery, ordering, routing, and acceptance decisions are nondifferentiable. Recompute continuous geometry from live numerical leaves on the accepted frozen topology. Do not globally stop gradients on stored coordinates.
- No Python loops over runtime-sized array axes, runtime JIT-wrapper construction, scalar synchronization in an iteration, or hidden device-to-host conversion. Static degree/level loops and immutable host topology preparation need explicit rationale.
- Restore remains within the registered reconstruction owner, checks exact type/fields, and validates the complete restored value. Do not create a persistence format merely because a new prepared class exists.
- Measure changed-symbol cyclomatic/cognitive complexity before and after. Extract substantial preparation/execution/evidence responsibilities; do not preserve huge ordinary helpers through trivial forwarding or source-copy duplication.
- Permanent tests assert consumer contracts. No tests of source text, absence of a particular sort call, forwarding, import spelling, or incidental default counts. Compiler/transfer inspection belongs in benchmark evidence and throwaway diagnostics.

## 3. P0 — baseline, contracts, and adoption experiments

### Files

- `tools/spatial_query_benchmarks.py`, `tools/spatial_hierarchy_benchmarks.py`, `tools/particle_fmm_benchmarks.py`, `tools/multipole3d_benchmarks.py`, `tools/multipole_plane_benchmarks.py`: extend the owning drivers rather than add parallel harnesses.
- `benchmarks/_runtime.py`, `benchmarks/_memory.py`: reuse runtime identity, separate lowering/compile, synchronized timing distributions, compiler memory, and adoption gates. Change these shared utilities only if a genuinely missing invariant is demonstrated.
- `benchmarks/memory_graph_queries.py`, `benchmarks/memory_geometry_queries.py`, `benchmarks/streamed_relation_scaling.py`: retain their affected memory scenarios and extend only the scenario needed by the new route.
- **New** `tests/_support/spatial_references.py`, only if three or more test families need the same independent host oracle. It may own stable distance/ID ordering and pair-coverage assertions; it must not mirror the package's tree construction or acceptance logic.

### Baselines and campaigns

Record source/build/driver fingerprints, backend/device/plugin identity, dtype, precision configuration, capacities, masks, distribution, cache policy, exact commands, and status. Separate host metadata preparation, numerical build, refit, query, interaction preparation, near/far evaluation, lowering, compilation, first execution, warmed execution, and transformed execution.

Search matrix: dimensions 1/2/3 for Morton and 1/2/3/6 for BVH; source capacities 4,096/65,536/1,048,576 subject to admission; k=1/8/32; varying target/source ratios and repeated target batches; uniform, Gaussian clusters, anisotropic slabs/filaments, lattice ties, coincident points, inactive slots, periodic seams. Do not run the full Cartesian product automatically; choose each row for an independent failure or scaling meaning.

FMM matrix: the existing 20k scenario plus admitted 65k/262k/larger capacities; explicit leaf, queue, far, near, receiver, and edge bounds; uniform and clustered distributions; order and opening-angle sweeps; signed strengths; isolated primal/strength-JVP/position-VJP/mixed-derivative runs. A sampled direct reference uses all sources for the selected targets, not a different reduced physical problem.

Spherical matrix: native Laplace order 0/1/2/4/8 and the largest admitted order; real and complex strengths; payload channels; axial and near-axial shifts, both signs, coordinate scaling, and both topology execution routes. Screened kernels remain separate unchanged-control rows.

### Gates

- Exact selection ships only where it is measurably preferable to the full-sort baseline; account for rank storage and compiler lowering, not just asymptotic source code.
- Device MEDIAN is admitted as a correct explicit GPU construction capability. Recommend it for rebuild-heavy workloads only after build-plus-query, host-transfer-inclusive comparisons. No requirement or claim that it beats Morton on uniform clouds or brute force on small/high-dimensional workloads.
- The accepted-pair storage rewrite must preserve output order and resource refusal and demonstrate its bounded retained/temporary storage as frontier capacity grows.
- Width scheduling becomes a default only with a material traversal benefit and no material regression on the controlling uniform/reference workloads. The simple full-width execution remains the conservative choice until then.
- Streamed execution must bound compiler temporary/reverse storage by the declared tile working set rather than full near/far capacity. Use the existing memory gate where applicable and report the runtime tradeoff; a proposed universal percentage improvement is not an established requirement.
- The Pallas fragment route must beat the streamed JAX route in its named target envelope before being described as accelerated. It stays explicit otherwise; unqualified experiments do not ship under an accelerated name.
- Spherical rotation ships only after coefficient, field, derivative, conditioning, and resource proof and a useful order/size envelope. Keep the existing dense route as a distinct supported method, not a disguised fallback.

Run benchmarks without concurrent tests/builds/agents. Discard evidence invalidated by code/configuration/dependency/topology changes; do not rerun valid unaffected checks merely for a handoff.

## 4. P1 — prepared Morton sources and exact selection

### Public API and numerical state

**New** `phydrax/discretization/spatial/_neighbor_sources.py` owns source preparation, source-plan metadata, prepared artifacts, and refresh transitions. Proposed names are final planning choices, not existing symbols:

```python
source_plan = MortonSourcePlan(address, source_capacity, dtype=np.float64)
sources = source_plan.prepare(points, active_mask=active, stable_ids=ids)
knn = MortonNeighborQueryPlan(source_plan, target_capacity, maximum_neighbors=8)
result = knn.query(sources, target_points, target_mask=target_active)
```

`dtype` is an explicit preparation choice. Query arithmetic uses the prepared geometry precision; do not silently narrow a wider target input. Normalize mixed inputs to their intended common precision before source preparation, or refuse an incompatible prepared-query precision. Existing in-repo mixed-input callers migrate with their previous common precision. Graph preparation remains float64.

`PreparedMortonSources` stores live logical `points`, wrapped sorted coordinates, active mask, full-width stable IDs, reversible int32 permutations, sorted codes/IDs/activity, execution-only stable-ID ranks, logical finite/domain flags, source validity evidence, metadata identifying the source plan, whether IDs were explicitly declared, and a dynamic epoch. Avoid retaining duplicate logical code/integer-coordinate tables unless an identified consumer needs them. `MortonEncoding` stays the public address-encoding result.

Query plans bind the source plan and own only query-specific capacities/backend/chunking. Matching address values alone are insufficient: validate the declared source-plan identity/capacity/precision. kNN, radius, and shell plans built from one source plan share an artifact.

Low-level signature cutover:

- `MortonNeighborQueryPlan(source_plan, target_capacity, maximum_neighbors, ...)`; `query(sources, targets, *, target_mask, target_stable_ids, exclude_self, radius, target_radii)`.
- `MortonRadiusRelationPlan(source_plan, target_capacity, maximum_pairs, ...)`; `query(sources, targets, radius, *, target_mask, target_stable_ids, exclude_self, pair_once)`.
- `MortonRadiusShellWitnessPlan(source_plan, target_capacity, ...)`; `certify(sources, targets, radius, half_width, *, target_mask, target_stable_ids, exclude_self)`.

No low-level method accepts raw sources or source mask/ID overrides after preparation. Preserve the existing self-exclusion and shell declared-ID relationship refusals. Query outputs keep logical indices, valid masks, per-target counts/status, and existing completeness/resource evidence.

### Files and edits

| File | Change |
|---|---|
| **New** `phydrax/discretization/spatial/_neighbor_sources.py` | `MortonSourcePlan`, `PreparedMortonSources`, source preparation evidence/transition; typed validation, canonical ordering, live geometry, ranks, and refresh |
| `phydrax/discretization/spatial/_morton.py` | Keep addressing/encode/decode and canonical ordering; remove obsolete `_MortonPointOrder` and `_canonical_morton_point_order` after all consumers migrate; make tie ordering explicit only where current owning behavior is insufficient |
| `phydrax/discretization/spatial/_neighbor_query.py` | Remove `_SortedSources` and duplicate source preparation; build target rows/search views from the prepared artifact; cut over all three constructors/methods; preserve certificates and status precedence |
| **New** `phydrax/discretization/spatial/_nearest_selection.py` | Exact selection invariant and independently gated partial-selection implementation; no public algorithm switch |
| `phydrax/discretization/spatial/__init__.py` | Explicit exports of the new source lifecycle, alongside the existing query types |
| `phydrax/discretization/__init__.py` | Update its existing spatial export aggregation; no compatibility names |

Source preparation checks IDs once and retains their order rank. Original ID arrays remain authoritative in results, self-exclusion, pair packing, and cross-shard merges. Ranks are local execution metadata; never use them as scientific identity or compare them across artifacts. The new artifact's actual field/PyTree representation is nevertheless part of its current representation identity.

### Exact small-k candidate

1. Mask excluded candidates with infinite distance; find the exact k-th threshold with `lax.top_k`.
2. Assign priority tiers: valid distance below threshold, valid distance equal to threshold, absent.
3. Select k smallest integer `(tier, stable-ID rank)` keys. Choose int32/int64 key storage from a static capacity bound so arithmetic/negation cannot overflow. Never negate full signed IDs or truncate them.
4. Sort only the selected k by distance and original full-width ID, then apply validity.
5. Preserve required-candidate counts, overflow decisions, retry radius, and completeness certification exactly. Test fewer than k, k equal to capacity, one-ulp float64 differences, periodic duplicate-cell avoidance, and all tied distances.

The inspected JAX 0.11.2 `lexsort` delegates to `lax.sort`, whose default is `is_stable=True`; current sort stability is not an established bug. Logical-index tie keys may explicitly complete an order if its owner needs that invariant, but avoid an unnecessary extra sort key. Measure valid-output behavior and identity effects. If the partial-selection route does not win, remove it and keep the existing full sort.

### Caller cutover inventory

| Existing file | Migration |
|---|---|
| `phydrax/discretization/spatial/_plane_schedule.py` | Use the prepared source artifact for `point_order`; replace `encoding.coordinates[storage_to_logical]` with its live sorted coordinates |
| `phydrax/discretization/spatial/_point_hierarchy.py` | Reuse source preparation before constructing cell topology |
| `phydrax/discretization/spatial/_distributed_relations.py` | Construct one shard source plan; prepare once in the shell execution and share it with local/remote engines, owner evidence, and reply geometry; migrate `_engine_candidates`, `_remote_answers`, and `_owner_evidence` |
| `phydrax/discretization/spatial/_plane_distributed.py` | Preserve public ownership/communication behavior; update only signatures/annotations actually affected by source binding |
| `phydrax/graph/_query_batch.py` | Planned-source dispatch and artifact input; update the three graph query/batch APIs and shared geometry/evidence assembly |
| `phydrax/discretization/meshfree/_neighbors.py` | Prepare padded certification-precision sources once; reuse them for selection and continuous offsets; migrate radius queries |
| `phydrax/discretization/meshfree/_exterior.py` | Prepare the shell witness sources with their explicit active mask and point IDs |
| `phydrax/discretization/particle/_morton_neighborhood.py` | Bind the source plan at preparation; build/refresh the numerical sources at the particle epoch; query radius pairs from that state |
| `phydrax/applications/cosmology/_halo_finder.py` | Prepare the halo radius sources with active flags/IDs |
| `phydrax/bubble_dynamics/_cloud.py` | Prepare radius sources without fabricating explicitly supplied IDs |
| `phydrax/velocimetry/synthetic/_ptv.py` | Prepare pixel/visibility sources per camera; reuse inside the camera's query work |
| `phydrax/solver/_particle_gravity.py` | Migrate Ewald radius calls; each physically different image-offset source still needs its own preparation; consume new plane sorted-coordinate fields |
| `phydrax/discretization/meshfree/_schwarz.py` | Confirm its canonical-order consumer preserves numerical/ID ordering |
| `phydrax/operators/integral/multipole/_laplace3d.py` and `phydrax/operators/integral/vortex/_fmm_complete.py` | Audit plane-state field/identity consumers; migrate changed fields without changing kernels |

Private minimum-image/evidence imports in meshfree transport/exterior and PTV do not change merely because source preparation changes.

Distributed scope is deliberately bounded: eliminate duplicate preparation within one shard query. Do not add a numerical cache to `DistributedPointLayout`, change migration/epochs, or change target/reply collectives. Global result merges continue to compare original IDs.

### Graph and batched preparation

- Raw dense/blocked graph inputs retain the existing float64 conversion, workspace policy, source-index ties, and radius boundary.
- A prepared graph source requires a matching explicit query plan, float64 logical points, and default logical-index IDs. Both raw planned and prepared graph routes require a float64 source plan; refuse an incompatible plan rather than narrow graph geometry. A prepared source also refuses a competing `source_mask` or custom-ID semantics; the artifact already owns source activity and graph ties remain source-index ties.
- Homogeneous source cases may be stacked with explicit mapping axes. Map coordinates, masks, permutations, ranks, geometry-dependent leaf payload/bounds, and evidence; share address metadata and capacity-fixed BVH topology arrays. Do not broadcast shared topology into every case or blindly map the whole plan.
- An unbatched prepared cloud may be shared across compatible target cases without copying its source arrays. Distinct prepared source cases must share plan metadata or use the existing signature-grouped execution boundary; case correspondence is explicit, not inferred from equal source sizes.
- Query individual cases with bounded `lax.map`/owning case execution so Morton retry conditionals do not accidentally evaluate every branch under unbounded `vmap`. Source preparation may use explicit homogeneous vectorization with its working set recorded.
- Continuous offsets/distances come from `sources.points`, not a stopped-gradient selection view. Test squared-distance derivatives at coincidence and ordinary distance derivatives away from zero; do not promise a derivative of Euclidean norm at zero.

### Tests

Migrate existing owning tests in `tests/unit/discretization/spatial/test_neighbor_query.py`, `test_distributed_plane.py`, `tests/unit/backends/test_spatial_backend.py`, `tests/unit/discretization/test_meshfree.py`, `tests/unit/discretization/meshfree/test_local_stencil_lifecycle.py`, and `tests/unit/graph/test_graph_query.py`.

**New** `tests/unit/discretization/spatial/test_prepared_neighbor_sources.py`: shared artifact reuse across query kinds, mismatch refusals, masks/ID validity, batching, live geometry, old behavior parity, empty active set, source immutability, and epoch transitions. New selection cases may live in the existing query file: shuffled/extreme signed and unsigned IDs, threshold ties, fewer than k, full capacity, duplicate IDs refused, periodic seams, and unchanged overflow/certification.

Retain affected distributed, particle-Morton, meshfree exterior/transport/surface, halo feedback, bubble cloud, PTV, and flow-integration consumer tests. Select their exact owning scenarios; do not run unrelated ML-neighbor methods.

## 5. P2 — capacity-fixed device BVH and explicit point queries

### Builder and representation

`PackedBVH` stays the canonical primitive tree. Its existing field/PyTree layout and host MEDIAN/MORTON/SAH outputs need not change. A median cut is `start + (stop-start)//2`, independent of geometry, so topology, level offsets, depth, and leaf capacities can be prepared from integer capacity.

| File | Change |
|---|---|
| `phydrax/_bvh.py` | Own `_MedianBVHLayout` preparation and `_build_median_bvh`; isolate substantial new build work in **new** `phydrax/_bvh_build.py` only if complexity demands it, with no alternate public tree type |
| `phydrax/_bvh.py` | Extend the existing bounded nearest traversal for item-admission, full-width tie keys, node bounds, empty-node guards, and completeness/work evidence; do not duplicate the traversal to avoid an identity audit |
| `phydrax/discretization/spatial/_neighbor_sources.py` | `BVHSourcePlan`, `PreparedBVHSources`; retain live logical points, activity/IDs, the private built tree, explicit node occupancy and minimum-ID metadata, build evidence, and epoch |
| **New** `phydrax/discretization/spatial/_bvh_neighbor_query.py` | `BVHNeighborQueryPlan(source_plan, target_capacity, maximum_neighbors, ...)`, result/evidence/status, and one stable compiled entry point |
| `phydrax/discretization/spatial/__init__.py` | Explicit source/query exports; no alternate spellings |
| `phydrax/graph/_query_batch.py` | Add explicit BVH plan dispatch and prepared-source cases using the existing graph contract |

The device builder performs one segmented deterministic sort per dividing level, leaves completed ranges in place, compacts inactive members within each final leaf, and computes bounds through the owning level-reduction substrate. Topology decisions use stopped-gradient geometry; stored source coordinates remain live. All-valid host/device parity is a numerical contract, not a source-text test.

Invalid/inactive items become trailing `-1` leaf entries. Masked trees stay inside the prepared point-source/query boundary until every generic primitive predicate has an explicit empty-node contract; do not hand such a tree to host ray/overlap/hierarchical callbacks accidentally. No existing geometry constructor is forced to switch build routes.

Node emptiness is an explicit Boolean, not `minimum_id == iinfo.max`: the maximum ID value is a valid supplied identity. Minimum-ID metadata enables exact equal-distance pruning when the lower bound cannot beat the current k-th tuple. Keep a checked stack capacity derived from depth and a node-visit bound; any structural/resource exhaustion invalidates the row, never returns partial neighbors.

Storage follows the selected input precision with no device narrowing. New point-route coordinate magnitude/domain checks must ensure the declared squared-distance arithmetic is representable, with data-dependent invalid status. Conservative non-periodic pruning owns rounding/underflow assumptions and must be proved against the computed distance rule, including allowed compiler reassociation. Never declare an unproved epsilon multiplier an exact bound. Periodic axes use the safe zero bound initially; a stronger bound needs a separate proof and adversarial campaign.

### Point-query semantics

- Full-width stable-ID ties; logical indices returned.
- Active/source validity and duplicate-ID checks; inactive/invalid target precedence documented.
- Explicit self exclusion using the declared IDs; preserve the existing equal-capacity default logical-slot convention and unequal-capacity refusal.
- Inclusive static radius and dynamic per-target radii; no hidden unsupported keyword branch.
- Float64 candidates ranked in float64, not recomputed after float32 selection.
- No active sources produces a complete empty result. Build/traversal failure produces no valid neighbors and explicit evidence.
- The balanced depth bound is structural; benchmark visit counts are not a permanent arbitrary threshold test.

### Intentionally unchanged host consumers

These retain host preparation and their existing device refit where present:

- `phydrax/_geometry_predicates.py`;
- `phydrax/discretization/_point_cloud_view.py`, `_polygon_query.py`, `_simplicial_locator.py`, `fem/_form_reconstruction.py`, `meshfree/_surface.py`, `splatting/_mesh.py`;
- `phydrax/enforcement/_ansatz.py`;
- `phydrax/geometry/_bvh_overlap.py`, `_supermesh.py`, `_triangle_ray.py`, `simplicial/_bvh.py`, `brep/_projection.py`, `implicit/_discovery.py`, `multiregion_surface/_geometry.py`, `_validation.py`, `_topology_transitions.py`;
- `phydrax/imaging/tomography/_core.py`, `phydrax/optics/geometric/_graded_index.py`;
- `phydrax/meshing/_audit_topology.py`, `_boundary_layer.py`, `_coupling.py`, `_local_metric.py`, `_sizing.py`.

Several consume topology on the host for exact predicates, variable-length CSR, or storage/stack certification. They are regression consumers of any shared traversal/refit edit, not candidates for speculative constructor migration. The contact-owned LBVH and ML metric owners also remain untouched.

### Tests and identity

- `tests/unit/test_packed_bvh.py`: public numerical host/device parity for capacities at leaf boundaries, dimensions, dtype, duplicate centers, signed zero; masks/all-inactive data; every valid item appears once; transfer-guard JIT smoke; nearest/refit regression.
- **New** `tests/unit/discretization/spatial/test_bvh_neighbor_query.py`: independent exact point-query oracles; extreme ID values including the maximum value; ties, radius, self exclusion, masks, invalid geometry, periodic/out-of-box points, non-power-of-two capacities, and explicit refusal. Include the coincident-cloud case without a performance-count pin.
- `tests/unit/graph/test_graph_query.py`: BVH/dense planned behavior and geometry derivatives, prepared source reuse, case batching, plan/period mismatch; retain the Morton overflow refusal.
- Existing segment/triangle/overlap/ray/locator tests owning a touched shared kernel: regression selection only, no blanket meshing suite.

Unchanged `PackedBVH` fields do not prove unchanged program identity. Audit `query_neighbors` consumers in `nn/operator/architectures/geometric/_geometry_operator.py`, `_local_operator.py`, `nn/operator/layers/_graph_transfer.py`, `_regional_processor.py`, and every source-addressed consumer reported by LSP. Document intentional callable/prepared/whole-container ID changes; do not duplicate code or fake historical payloads to avoid them.

## 6. P3 — genuine refit and certified structure refresh

### Source order

`MortonSourcePlan.refresh(previous, points, *, active_mask, stable_ids)` returns a transition with a candidate, accepted state, acceptance flag, reuse/rebuild flags, and evidence. Omitted mask/ID arguments retain the previous declared source identity, not silently return default IDs.

A sort-free candidate is legal only when the previous state was valid, logical IDs/activity are unchanged, new active points are in domain, and newly encoded keys read in the previous permutation remain in complete canonical order. Carry the inverse permutation, uniqueness evidence, and ID ranks; refresh coordinates/codes and epoch. Otherwise build a fresh candidate. Failed candidates do not replace the previously accepted state.

### Hierarchy and plane schedule files

- `phydrax/discretization/spatial/_point_hierarchy.py`: split source order from topology construction. Reuse additionally requires every active point to remain in its owning leaf prefix. Preserve counts, connectivity, leaf maps, and cell geometry; update codes/epoch. Crossing a leaf or changing order/mask/IDs rebuilds.
- `phydrax/discretization/spatial/_plane_schedule.py`: split order, capacity-fixed group tables, and numerical geometry. Reused ordering/membership may update common prefixes, bounds, centers, scales, padding, and evidence without sorting. Prove which group/connectivity tables remain fixed from the actual grouping algorithm; if that proof fails, rebuild. Stop describing compare-after-build as a fast refit.
- Keep public hierarchy/schedule refresh signatures unless a genuine shared prepared-input boundary requires a clean cutover. `MortonPlaneScheduleState.point_order` changes to the new source artifact; no compatibility property recreates its removed encoding object.

### FMM refresh files and contracts

- `phydrax/discretization/spatial/_plane_interactions.py`: extract one owning acceptance calculation used by build and current-geometry recheck. Recheck all retained valid far pairs against opening angle, node-radius limit, distinctness, and finite geometry; preserve pair coverage and exact near completion.
- `phydrax/solver/_particle_gravity.py`: add `refresh_structure` and an owned transition for `PreparedUniformFMMStructure`. Reuse numerical topology only with unchanged particle/leaf membership, activity, IDs, and admitted source ordering. Update live geometry and recompute the stored far acceptance evidence; reuse prepared reduction/stream schedules only when their topology binding is still valid.
- A short-range cutoff can have omitted pairs that become relevant after motion. Without an existing proved cutoff-plus-skin certificate, geometry changes force an interaction rebuild. Do not certify cutoff reuse by checking only the retained far list.
- A failed refresh returns the old accepted structure and a failed candidate/transition. The consumer must observe the failed update; retaining the old accepted artifact is rollback, not a claim that the new geometry succeeded.
- Use capacity-bounded stream schedules for FMM so host-prepared and JIT-rebuilt structures have identical static fields/shapes. Do not impose a new requirement that callers prepare the original structure only under JIT.

Refit and fresh rebuild may choose different valid far/near cuts. Tests assert exact pair coverage, current admissibility, resource status, and independent numerical/error evidence, not equality of those different cuts. Source-order/hierarchy fast-path output may be compared with fresh construction where its canonical topology is mathematically identical.

### Tests

Extend `test_point_hierarchy.py`, `test_plane_schedule.py`, `test_plane_interactions.py`, `tests/unit/solver/test_uniform_fmm_monopole.py`: unchanged order, order-changing moves, leaf crossing, mask/ID changes, rejected domain updates, MAC boundary violations, cutoff rebuild, repeated strengths, transaction rejection, and geometry AD. A test of a counter or mocked builder call is not proof of refit; the benchmark must show the eliminated work and its compiler/runtime cost.

## 7. P4 — bounded dual-tree storage and work

### Files

- `phydrax/discretization/spatial/_plane_interactions.py`: replace step-array accumulation/concatenation with fixed-capacity far buffers and stable prefix append. Keep the same step/frontier/child order, required counts, maximum accepted ratio, and fail-closed list invalidation.
- The same owner prepares a small tuple of supported static frontier widths, with the last width equal to the queue capacity. Full-width execution remains the default until the width campaign passes. Width changes do not silently grow resources.
- `phydrax/solver/_particle_gravity.py`, `phydrax/operators/integral/multipole/_laplace3d.py`, and `phydrax/operators/integral/vortex/_fmm_complete.py`: pass the owned traversal preparation where needed and incorporate its current identity. Do not add gravity MAC logic to generic traversal.
- `tools/spatial_hierarchy_benchmarks.py`: record interaction preparation separately, live/required capacity, temporary/output/retained bytes, compilation, and sparse/dense frontier distributions.

Branches/rungs must have compatible result signatures and preserve accepted-pair order. Avoid an unbounded vmap of conditionals that evaluates every width. Inspect actual transfers/control-flow lowering; source use of a conditional is not proof of zero synchronization. If width selection loses, remove the candidate and keep bounded full-width traversal.

Tests in `tests/unit/discretization/spatial/test_plane_interactions.py` compare complete scientific pair coverage and ordered valid relations with the baseline, including capacity boundaries at seed and each refinement phase. Exercise same-support, rectangular, cutoff, and maximum-node-radius consumers. Do not require identical resource-plan IDs after changing their representation.

## 8. P5 — streamed near/far execution and real Pallas fragment kernels

### One execution topology

`PreparedUniformFMMStructure` replaces global point-pair routing arrays with a prepared sparse relation over source/target leaf slots. A near event contributes a bounded vector of target-leaf particle outputs. Leaves have fixed prepared occupancy and contiguous sorted ranges; pad only the final particle storage to its fixed leaf capacity and keep explicit masks.

Use `StreamedRelationPlan` receiver/edge tiling, fragment source maps, stable route grouping, seeded high/correction accumulation, replay, poisoning, and one receiver epilogue. No new CSR builder or replay runtime belongs in the solver/backend.

### File-by-file changes

| File | Change |
|---|---|
| `phydrax/solver/_particle_gravity.py` | Prepare leaf-slot relations/stream schedules; remove `_near_pair_routes` and obsolete near point-route fields; keep `prepare_structure`, `_near_field`, `_field`, evidence, and consumers consistent |
| **New** `phydrax/solver/_particle_fmm_kernels.py` | Only if needed for complexity: scientific Plummer/split pair formulas, reference near-event messages, scalar kernel specification, and complete potential/gradient semantics |
| `phydrax/backends/spatial.py` | Own actual Pallas fragment loads, bounded lane/leaf loops, seeded output, and a complete derivative rule linked to the same scientific body |
| **New** `phydrax/backends/_spatial_fragments.py` | Extract substantial lowering/derivative implementation if necessary; keep `backends.spatial` the owning boundary, not a second public runtime |
| `phydrax/sparse/_streamed.py` | Add an explicit capacity-bounded preparation option for schedules if the current compact-host/traced-bound split prevents compatible FMM refresh; retain compact default behavior for other consumers |
| `phydrax/sparse/_execution.py` | Reuse seeded reduction/two-sum invariants. Change only a proved missing shared operation; never copy the reducer into a backend |
| `phydrax/backends/__init__.py` | Export only genuinely public new capabilities; internal FMM fragment bodies need no convenience re-export |
| `phydrax/discretization/spatial/_plane_interactions.py` | Consume current admitted interaction evidence and topology binding |

Existing `spatial_squared_norm` and `spatial_pair_acceleration` remain legitimate explicit array primitives. Stop using the latter for global FMM pair expansion, but do not delete a useful public primitive merely because one consumer migrated. Its independent numerical/AD tests remain. Remove only obsolete FMM routing and duplicate formulas.

### JAX and Pallas execution

- JAX reference: a per-event message over one leaf pair, streamed over bounded fragments. Working arrays may scale with fragment width and leaf occupancy, never total near capacity times leaf occupancy squared.
- Pallas: one program per bounded receiver slot/target subtile. Load target particles, follow the fragment's receiver-major lane ranges, load source leaf particles/masks/strengths, evaluate interactions, and accumulate before storing. Do not pass an already expanded global displacement tensor to a pointwise kernel.
- Source gather may be fragment-local or use direct sorted-array references. Choose the lower measured working-set route without padding the whole cloud to a power-of-two Cartesian product. Bound all local source/target tile dimensions explicitly.
- Pair validity is dynamic support data: lane validity, source/target masks, and exact same-leaf diagonal exclusion. Invalid/padded lanes use safe operands and are numerically inert.
- Preserve existing source-minus-target direction, signed strengths, softening, split radius including softening, inclusive cutoff, and the current refusal of split monopole potential. No kernel substitution changes Plummer gravity to unsoftened Laplace.
- Preserve event order and existing accumulation modes. Compensated accumulation carries high and correction across fragments; do not add independently reduced subtotals or use atomic order as a deterministic route.
- Public defaults stay JAX. The advertised new fused route initially targets real CUDA, float32/float64, with separate CPU interpret correctness. Do not inherit a TPU/HIP claim from the simpler spatial primitive's capability declaration.
- Compile-smoke the actual dynamic leaf loop and split `erfc` on CUDA before final admission. If split lowering cannot execute faithfully, construction explicitly refuses that Pallas combination; no hidden JAX primal fallback. A static bounded masked loop is an acceptable compiled execution choice if its resource/performance gate passes.

### Derivatives and failure

Provide a `custom_jvp` or complete primitive rule with a JAX reference tangent on the same bounded fragment, so potential and gradient cotangents, geometry/strength JVP/VJP, and the admitted mixed/Hessian actions remain correct. Preserve seeded-accumulation derivatives; integer work counts have `float0` tangents and are not fed to floating reduction arithmetic. Declare derivative execution as JAX/reference where that is what runs; do not advertise accelerated backward performance without measuring it.

Keep the outer `UniformFMMPlan.evaluate` rematerializing reverse contract distinct from `evaluate_monopole`. Do not copy an acceleration-only reverse onto a potential-returning function.

Schedule, interaction, non-finite, and resource failures reach the final FMM result. Propagate the sparse owner's poisoning on failed streamed execution; never transform failure into a successful zero force. The current cosmology test pins a zero payload for exhausted interactions: remove that incidental assertion when the streamed result changes, retain/strengthen refusal and invalidity assertions, and update the documented consumer boundary rather than re-pin another incidental value.

### Far messages

Replace the global `far_local[far_capacity, coefficient_count]` temporary with an event callback using the existing Cartesian `m2l` and a receiver epilogue through the same streamed owner, provided the phase-separated memory/runtime gate passes. This is an execution change, not the spherical numerical rewrite. Preserve far event/reduction order for the unchanged partition and status/resource evidence.

### Tests and consumers

- `tests/unit/backends/test_spatial_backend.py`: actual fragment scenarios with split receivers, empty slots, masks, coincident particles with positive softening, non-power-of-two local dimensions, float32/float64, and all accumulation modes; independent direct values, repeated deterministic results, JVP/VJP duality, potential-only cotangents, mixed derivatives, and explicit unsupported-route refusal.
- `tests/unit/solver/test_uniform_fmm_monopole.py`, `test_cartesian_fmm_gradients.py`, `test_cartesian_fmm_algebra.py`: prepared strengths, source/target geometry, full output semantics, resource refusal, refit/rebuild admissibility, and direct softened references.
- `tests/unit/applications/test_cosmology_tree_gravity.py`, `test_cosmology_force_closure.py`: TreePM/split semantics and evidence at consumer boundaries. Do not assert interpret mode proves a CUDA route.
- `tests/unit/test_sparse_streamed.py`, `test_sparse_streamed_derivatives.py`: required only if the shared schedule/replay owner changes; test compact behavior unchanged and explicit capacity-bound preparation/refit compatibility.
- `phydrax/operators/integral/vortex/_fmm_complete.py` and its owning tests: only shared traversal/source-state regressions; no scientific vortex rewrite.
- `tools/particle_fmm_benchmarks.py`, `tools/spatial_hierarchy_benchmarks.py`, `benchmarks/streamed_relation_scaling.py`: phase-separated primal/transformed scaling and compiler/retained bytes, compared under identical material inputs.

## 9. P6 — separately gated native Laplace translations

This phase is not a replacement of Cartesian softened FMM and does not accelerate Helmholtz by calling Laplace. It is a complete, optional numerical/execution route under the existing spherical multipole owner. No partial route ships under a full-translation name.

### Ownership and API

Extend the existing `TranslationRoute3D` selector with one qualified rotation route after its gate passes; retain `dense` as the existing supported method. Parse the closed selector once, store/fingerprint it, and dispatch exhaustively. Do not add aliases for upstream basis names.

Public coefficients stay the native complex, spin-zero `SphericalModeLayout`, including complex strengths and payload channels. Internal real-basis work uses `special.real_harmonic_basis` and native normalization, with explicitly derived input/output maps for moment and local conventions. Do not import Dehnen coefficient scaling as if it were our orthonormal layout.

### File map

| File | Change |
|---|---|
| `phydrax/operators/integral/multipole/_laplace3d.py` | Route declaration/parsing, translation preparation, M2M/M2L/L2L dispatch, field/resource/evidence and prepared identity |
| **New** `phydrax/operators/integral/multipole/_laplace_rotation.py` | Native axial translation coefficients, rotate-to-axis/scale/application, complete derivative rules, bounded preparation and numerical evidence |
| **New** `phydrax/operators/integral/multipole/_laplace_projection.py` | Only if needed to isolate the existing reference preparation/translation responsibility; no source-copy forwarding facade |
| `phydrax/discretization/spectral/_spherical_algebra.py` | Reuse `SphericalRotationPlan`/`PreparedSphericalRotation`; extend owning block/generator preparation only if an invariant/capability is missing |
| `phydrax/special/_real_harmonic.py`, `_solid_harmonic.py` | Reuse the existing Cartesian-safe basis and exact complex-to-real maps; change only a proved owning capability gap |
| `phydrax/operators/integral/multipole/_radial3d.py` | Keep outgoing/decaying kernel routes and dimensionless-argument limits intact; migrate only shared preparation/layout changes; refuse an unsupported rotation route rather than route to Laplace |
| `phydrax/operators/integral/multipole/_adapters.py` | Preserve density/weight conversion, dipole normal orientation, far-only QBX ownership, and complex payloads |
| **New** `phydrax/backends/_multipole_fragments.py`, only after the numerical route passes | Separately gated register-local Pallas translation fragments over the existing prepared far stream; no independent multipole physics or CSR owner |
| `phydrax/operators/integral/multipole/__init__.py`, `phydrax/operators/integral/__init__.py`, `phydrax/operators/__init__.py` | Update only real public selector/contract exports and their existing lazy/explicit facade conventions |

Separate topology preparation from kernel-specific translation data if needed. Rotation preparation must not allocate unused projection quadrature/pseudoinverse data. Radial preparation must not acquire an unrequested Laplace numerical method through inheritance. Any resulting prepared field/PyTree change is an explicit current-representation cutover.

Where dense calibration is materially changed, replace raw inverse/pseudoinverse application with a prepared native `phydrax.linalg.MinimumNormProblem` solve/factorization, preserving the intended cutoff, rank/support evidence, and operation contract; prepare once and reuse multi-RHS execution. Do not add a private solve or jitter. Existing projection references remain independent controls, not the sole oracle for their replacement.

### Numerical and derivative proof

- Derive native z-axis M2M/M2L/L2L coefficients with all degree/order/normalization factors. Prove coefficient-map equivalence using moments and local polynomial reproduction, not equal output shapes.
- Use the owning rotation blocks/generators and degree/radius scaling. Bound tables, code growth, exponent/radius range, and compiler working memory separately. High order is admitted only within a declared numerical/resource envelope.
- M2L at coincident centers is singular and refused. Zero M2M/L2L shift is a valid identity translation with nonzero displacement derivatives; implement its analytic polynomial limit, not a constant identity with zero tangent.
- Prefer a rotational-covariance directional derivative for all nonzero shifts: angular contributions are representation-generator commutators and radial contributions follow the translation's degree scaling. Calibrate every sign in the native active-ZYZ convention.
- Do not copy a finite-band axis patch that changes the derivative of an unchanged primal without an error contract. At axial/near-axial shifts the selected derivative must be the derivative of the same mathematical operator, with first and admitted higher-order checks.
- Preserve real/complex strength linearity, source/target center derivatives, dipole/source-normal actions, local evaluation, and status. No silently dropped potential or coefficient cotangent.

### Qualification

Extend `tests/unit/operators/test_multipole3d.py` and `test_multipole_layer_adapters.py`; **new** `tests/unit/operators/test_laplace_rotation_translation.py` owns the new route's coefficient/axial/zero-shift/error/resource contracts. Reuse `tests/_support/differentiation.py`.

Include independent direct sums, exact low-degree moments/polynomials, translation composition where mathematically exact, rotational covariance, complex payloads, positive/negative axis shifts, approach-to-axis sweeps, JVP/VJP duality, mixed derivatives/Hessian actions, source normals, order convergence, and resource refusal. Existing screened-kernel rows must remain unchanged controls. Run affected spherical/real/solid-harmonic tests only if their owner changes.

`tools/multipole3d_benchmarks.py` and `tools/multipole_plane_benchmarks.py` separate table/topology preparation, each translation, complete field evaluation, lowering, compilation, warmed execution, and derivatives. No public rotation selector, default change, or accelerated claim is admitted until the complete route passes.

### Register-local GPU experiment after the numerical gate

Once the native rotation route is qualified, evaluate the upstream one-pair-per-lane layout as a private far-fragment implementation: coefficients remain local through rotation, axial translation, and inverse rotation; receiver reductions remain with the prepared sparse stream. Declare an explicit translation-execution backend rather than repurpose `execution="plane_dual"` as a CUDA selector. A Pallas-selected primal must really execute that kernel and refuse unsupported dtype/order/payload tuples, never silently call JAX.

Reuse the same prepared native basis/translation tables as dynamic nontrainable leaves; do not hide numerical arrays inside static callbacks. Complex strengths require complete real/imaginary channel handling, not an assumption of coefficient reality. Measure register pressure, spills, code bytes, compilation, primal/derivative time, and memory as order and channel count increase. Keep JAX as the default and reference derivative execution unless a complete accelerated derivative primitive is separately justified. Add kernel/consumer tests only if this experiment passes and ships; otherwise remove it entirely. This experiment is not a prerequisite for the useful source/BVH/streaming phases.


## 10. P7 — complete cutover, documentation, typing, and generated data

### Authored documentation and examples

- `docs/guides_sparse_spatial_hierarchies.md`: explicit source lifecycle, exact selection/capacity semantics, BVH choice and periodic limit, real refresh invariants, bounded traversal, and streamed execution.
- `docs/api/operators/graph.md`: raw versus prepared source contract, matching source/query plan, float64/source-index identity, batching, exact radius, no automatic fallback, and resource evidence.
- **New** `docs/api/discretization/spatial.md` if the new source/query public surface needs a canonical API page; link it through the existing `docs/api/discretization/index.md` and `mkdocs.yml` navigation. Do not scatter duplicate API entries into unrelated meshfree pages.
- `docs/api/discretization/meshfree.md`, `docs/api/discretization/particle.md`, `docs/api/execution.md`: update actually changed usage/provenance, not a global prose rewrite.
- `docs/guides_integrals.md`, `docs/api/operators/integral.md`, `docs/api/backends.md`: fused near/far ownership, potential and derivative execution, unsupported split/platform combinations, precision, and qualified spherical route.
- `docs/api/geometry.md`: keep host geometry build/refit capabilities truthful; do not claim generic masked/device geometry support from a private point builder.
- Existing examples and inline snippets discovered by constructor/signature inventory: migrate in the same phase. If no authored runnable example covers prepared reuse, **new** `examples/prepared_spatial_queries.py` should demonstrate repeated targets, explicit Morton/BVH choice, matching masks/IDs, and refusal evidence, not a thin forwarding sample.
- `CHANGELOG.md`: public signature break, source/query/whole-container identity impact, genuine refit, device MEDIAN capability, explicit periodic BVH limit, bounded FMM execution, derivative support/nonclaims, and any removed obsolete fields.
- `NOTICE`, `LICENSES/`: only actual source incorporation notices, not automatic entries for reading repositories.

### Typing, declarations, identity, and persistence

- **New** `tests/typing/cases/spatial_queries.py`: accepted raw/prepared graph and low-level source/query signatures; wrong source family/capacity/precision; truthful inferred result types; narrow deliberate misuse ignores paired with runtime refusal tests.
- **New** `tests/typing/cases/multipole_translation.py` only if P6 changes public annotation contracts.
- `tests/unit/test_contract_declarations.py`, `tests/unit/test_role_declarations.py`, `tests/unit/typing/test_strict_integration.py`: run the affected declaration/construction gates; do not invent source-text declaration tests.
- New source/query/prepared representation IDs include current preparation configuration, precision, and actual owned representation. Enclosing particle/distributed/FMM IDs change where they depend on that state. Do not retain old key names just to produce historical IDs.
- Keep logical scientific IDs separate from rank/index execution metadata. This does not justify ignoring actual field/PyTree changes in whole-container fingerprints or compiler identity.
- Run existing registered artifact/restore/array-model tests only for the actual owner registrations touched. If no class is persisted, do not create a serialization subsystem. Saved templates whose actual structure changes require an explicit clean cutover, not a version field or compatibility decoder.
- Source-addressed functions and imported helpers require before/after identity and compiler-admission analysis; an unchanged jaxpr does not prove unchanged identity. Preserve guards that own identity unless the change is intentional and recorded.

### Generated consistency

After the last implementation change, run `tools/generate_public_api_manifest.py`, then `tools/check_public_api_manifest.py`. Update `docs/data/public_api.json` deterministically. If capability declarations are actually extended, update their existing owning declarations, run `tools/generate_capability_inventory.py` and `tools/check_capability_consistency.py`, and label only exercised tuples. Do not automatically mark a numerical method released or update an unrelated catalog.

The runtime-contract TSV files in `docs/plans/` are dated audit baselines, not a new generated registry to hand-edit. Refresh only through their documented owning workflow when the implementation actually updates that audit.

## 11. Verification waves and concrete acceptance

### Focused waves, per completed slice

1. Configured formatting/lint on changed files, changed-surface static diagnostics, and applicable selector/contract/declaration audits.
2. Direct runtime smoke with the changed public API, not tests alone.
3. Exact affected regression/consumer nodes. Fix focused failures before widening.
4. Phase-separated benchmarks and compiler/retained-memory proof with no competing processes.

Required direct smokes after implementation:

- Reuse one prepared source across several target batches and across kNN/radius/shell, exercise mismatches, inactive data, ties, radius/self exclusion, and rejected refresh.
- Build a BVH on the actual GPU, query the coincident cloud and a clustered/random reference, exercise the explicit periodic bound and failure evidence, and observe transfer/capture behavior.
- Run a complete FMM on CUDA with the new fragment path, signed monopole potential/gradient and strength refresh, resource refusal, geometry refit/rebuild, and potential/position/strength derivative actions.
- If P6 ships, run complete Laplace and layer/QBX evaluation with the selected translation route, axial/zero-shift cases, complex strengths, and independent values/derivatives.
- Run source-shard preparation/reply behavior on the existing dedicated distributed harness; unchanged target transport is not an excuse to omit consumer proof.

### Conservative integration matrix

Include the spatial query/order/hierarchy/schedule/interaction/distributed families; affected graph and geometry consumers; selected meshfree/particle/halo/bubble/PTV/Ewald consumers; FMM algebra/monopole/gradient/cosmology consumers; affected streamed tests if that owner changed; P6 multipole/adapters and only harmonic/rotation owners actually touched; typing fixtures; declaration/role gates; public/generated consistency; and owner-specific identity/restore tests.

Selection is based on the then-current delta since the latest merge into `dev`, including any prior unverified or invalidated CUDA work, not only the last edit. Reuse valid unaffected prior results. No full suite unless global collection/configuration/shared-fixture or another genuinely cross-cutting change requires it.

CPU tests use one invocation with `-n auto` and a global worker cap (for this environment, `PYTEST_XDIST_AUTO_NUM_WORKERS=5`). GPU tests use a dedicated one-GPU invocation with at most two workers and `XLA_PYTHON_CLIENT_PREALLOCATE=false`; large resource cases and forced topology tests run separately. Do not launch several unrestricted xdist jobs. The observed container limit is about 62 GB; retain the existing 54 GB watchdog policy and reject an admitted case before it exhausts the process.

For scripts/tool drivers, set `PYTHONPATH` to the implementation worktree so they do not import the parent checkout. Never run plain `uv sync` in this environment; use an explicit `--inexact` synchronization only when a required dependency change is approved.

### Final gates

- `python tools/check_typing.py check` must report zero first-party diagnostics. The current environment's previously observed missing optional packages/unused ignores are not a passing gate; resolve genuine prerequisites rather than suppress them.
- `tools/audit_selectors.py` and `tools/audit_contract_candidates.py --signatures` pass, with no redundant checked-boundary guards or unresolved selected annotations.
- Public-annotation changes additionally run the existing installed-wheel typing qualification where material.
- Every newly claimed route has direct runtime and failure evidence, complete derivative support or explicit refusal, and measured resource bounds. CPU interpret does not qualify CUDA; one GPU does not qualify multiple GPUs, TPU, or ROCm.
- All affected callsites/tests/docs/examples and generated records are cut over together; no deprecated aliases, dead routing arrays, experimental scaffolds, or compatibility generations remain.

## 12. Dependency-aware implementation ownership

- Source/query owner: P1 plus P3 source-order/hierarchy work. Own `_neighbor_sources.py`, `_morton.py`, `_neighbor_query.py`, `_nearest_selection.py`, and tightly coupled hierarchy migrations.
- BVH owner: P2 private layout/build/traversal and BVH point-query implementation. Coordinate the prepared-artifact contract with the source owner; do not edit graph simultaneously.
- Graph/consumer integration owner: all public caller/type/export migrations and P7 records. Own `_query_batch.py` and each shared facade during integration.
- Hierarchical execution owner: P4 and P5, with `_particle_gravity.py` one active owner; sparse-owner changes reviewed against its existing consumers.
- Spherical numerical owner: P6 after its conventions and admission gates are fixed; no concurrent edits to the shared multipole prepared-owner file.
- One integration owner records evidence, resolves interfaces, regenerates once, and runs the integrated affected matrix once. Subagents skip broad build/lint/tests mid-flight; integrated checks happen after a coherent slice.

Ready independent work: baseline drivers and private MEDIAN layout can proceed alongside source API preparation; P1 caller migrations depend on source signatures; graph BVH integration depends on both P1 and P2; genuine FMM refresh depends on P3 schedule and P4 acceptance recheck; fragment execution depends on the final leaf/stream ABI; P6 is a separate numerical gate, not a prerequisite for useful search/streaming improvements. Follow existing explicit `//merge` and `//pr` rules; no implicit merge or commit point is introduced.

## 13. Remaining gates, not unresolved design placeholders

The implementation choices above are resolved. These facts must be established by code/runtime evidence before a claim is admitted:

1. Whether exact partial selection wins on the installed compiler/device and where its static crossover lies.
2. Whether segmented MEDIAN construction beats host build plus transfer for the intended rebuild workload; correctness alone admits the explicit capability, not a speed claim.
3. Whether a tighter periodic BVH bound can be proved; until then zero periodic-axis bounds are the selected correct implementation.
4. Whether width-aware queue execution preserves ordering and improves actual GPU work without harmful conditional lowering.
5. Whether split `erfc` and dynamic source loops lower faithfully in the selected Pallas backend; otherwise the exact unsupported combination is refused, with no fake fallback.
6. Whether streamed defaults and far-message execution pass compiler/retained-memory and runtime gates under the same numerical partition/order.
7. Whether native rotation translations preserve normalization, complex payloads, axial and zero-shift derivatives, conditioning, and the selected resource envelope.
8. Which source-addressed/container identities change. Actual representation/source changes must be recorded, not hidden by historical fingerprints.

If a gate fails, retain the existing qualified route and remove that experimental implementation. Finish all reachable accepted work, and report the exact unqualified tuple without inventing performance, topology, or scientific support.
