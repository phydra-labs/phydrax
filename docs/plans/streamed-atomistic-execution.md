# Streamed atomistic execution: native MACE and shared-substrate closure plan

## 0. Baseline, authority, and evidence boundary

- Repository baseline: `dev`, `d720a65fb090f32c40a9d8c89472bb1d15a24d9e`, merge PR #389; meshfree implementation commit `d7e3989fc`. The preceding merge is PR #388, enforced periodic constraints. The working branch was `dev`; the tracked working tree had no reported changes when scoped.
- Reference checkout: `.tmp/symmetrix-xl`, previously inspected commit `da331142729978d27b2a52f6b34e1628593df3d0`. Reference sources are pinned for this plan, not assumed identical to a moving upstream branch.
- This is a design and implementation plan, not implemented support or qualification evidence. Familiarization used source/document reads, repository metadata, and LSP reference queries. No package code, examples, tests, benchmarks, builds, formatters, or linters were run.
- This Markdown plan is the only repository mutation authorized by this request. A later `//code` implementation starts in a fresh worktree under `/Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/streamed-atomistic-execution`, branched from the then-current branch. Copy the authoritative ignored `AGENTS.md` into the worktree before implementation. All implementation changes stay there. `//code--` is the explicit exception.
- Reconcile intervening changes first; never undo another author's work. Re-read affected owners and run LSP references before every exported-symbol change. Do not carry this snapshot's line numbers or consumer counts forward as an exhaustive migration inventory.
- Files marked **new** are proposed files, not existing capabilities. Existing files listed below were located through repository tools or source investigations.
- New scientific structure is a closure exception under `docs/CAPABILITY_LIFECYCLE.md:35-48`: MACE, arbitrary-degree O(3) execution, image-aware learned graphs, and accelerated/distributed inference need named owners, exact support tuples, dependencies, obligations, and permanent nonclaims. Planning does not authorize release.
- Implementation closure, numerical/derivative/performance/provider qualification, and signed release admission are separate milestones. Missing hardware, reference weights, rights, or release signatures remain explicit blockers; they do not justify fake implementations, narrower undocumented support, or a production claim.

### References

- Symmetrix-XL: <https://github.com/bonan-group/symmetrix-xl>.
- Paper: <https://arxiv.org/abs/2610.01036>, especially Sections 3–5 and Appendix B.
- Streaming tensor-product antecedent: <https://arxiv.org/abs/2607.18074>.
- Pinned source anchors: `docs/streamed_edge_execution.md`, `docs/reference/execution_support_matrix.md`, `libsymmetrix/source/compact_radial.cpp:326-365`, `symmetrix/source/symmetrix/extract_mace_data.py:983-1082,1136-1153`, `symmetrix/source/symmetrix/jit_codegen.py:884-1109`, and `pair_symmetrix/README.md:75-129` in the reference checkout.
- Native policy/authority: `AGENTS.md` as supplied in context, `docs/CAPABILITY_LIFECYCLE.md`, current `docs/plans/meshfree-closure.md`, and the canonical qualification catalog.

## 1. Goal and full scope ledger

Implement native, trainable, architecture-faithful standard MACE; bounded learned-graph inference and force training; multi-image periodic geometry and cell stress; safe checkpoint conversion and durable native artifacts; owned multilayer distributed execution; real accelerated kernels; and usable ASE/i-PI/frozen-export deployment. Absorb execution invariants into their existing Phydrax owners, not an alternate C++ engine.

| ID | End-to-end requirement | Owning phases |
|---|---|---|
| C01 | Canonical identities, supported envelopes, refusal/derivative semantics, breadth exception | P0, P12 |
| C02 | Prepared receiver/source schedules; variable degree, empty rows, masks and duplicate routes | P1 |
| C03 | Bounded nonlinear edge evaluation plus receiver epilogue; seeded reductions | P1, P2 |
| C04 | Recomputed reverse, coordinate/parameter JVP/VJP and force-loss mixed derivatives | P2, P6, P7 |
| C05 | Explicit image routes, nonzero self images, triclinic/partial periodic geometry | P3 |
| C06 | Image-aware Verlet lifecycle, wrap/cell certificates, rollback and capacity growth | P3, P7 |
| C07 | General real O(3) coupling, explicit parity/basis/path normalization | P4 |
| C08 | Symmetric contraction/product graph with preserved source parameterization | P5, P6, P8 |
| C09 | Exact radial execution plus qualified smooth tabulation and active-species binding | P5, P6 |
| C10 | Native standard MACE interactions, product bases, residuals, readouts, E0/scale/head semantics | P6 |
| C11 | Unified finite/periodic E/F/S prediction, learned program stress and training | P7 |
| C12 | Faithful imported OMAT-0, MPA-0, MP-0b-family, OFF23, standard MH-0 checkpoints | P8, P12 |
| C13 | Pickle-free native model persistence and fresh-process MD/training continuation | P8, P11 |
| C14 | True owner-local graph execution, forward feature halos and reverse cotangent return | P9 |
| C15 | Topology/ownership migration and bounded distributed continuation | P9, P11 |
| C16 | Actual accelerated forward/reverse tensor-product execution, with explicit target admission | P10 |
| C17 | ASE calculator, native i-PI provider, fixed-shape E/F/S export | P11 |
| C18 | Native PaiNN/NequIP and selected graph/meshfree consumer cutovers | P2, P4, P12 |
| C19 | Phase-separated scaling, compiler/retained/device/halo memory and capacity evidence | P2, P10, P12 |
| C20 | Documentation, generated API/catalog, source rights and exact release dossier | Every phase, P12 |

### Model boundary: architecture admission, not a brand-name promise

The required import campaign covers the standard interaction families already discussed: real-agnostic and residual interactions, their density-normalized variants, one- and two-interaction standard checkpoints, standard multihead readouts, Bessel/polynomial cutoff and Agnesi transforms, and optional ZBL. Native models may use a statically declared greater interaction depth if the same admitted layer contracts compose; test at least three layers rather than hardcoding the paper's two.

Required external model rows: OMAT-0 small/medium, MPA-0 medium, MP-0b/0b2/0b3 standard variants including a degree-two hidden case, OFF23 small/medium/large, and standard MH-0 with explicit head selection. These rows exercise distinct semantics, not redundant copies of the same path. Verify each checkpoint's actual classes, tensors, provider release, digest, and rights before declaring admission. A row is not complete when its importer merely recognizes a model name.

Pinned campaign rows (external release identifiers and SHA-256 from the inspected source manifest; they are not native schema generations):

| External model | Source SHA-256 |
|---|---|
| mace-omat-0-small | `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5` |
| mace-omat-0-medium | `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a` |
| mace-mpa-0-medium | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` |
| mace-mp-0b-small | `7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0` |
| mace-mp-0b-medium | `ab8baff639a8f295f3eccad3d3ccf574efb6fb63220bd52cd88664211569e521` |
| mace-mp-0b2-small | `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9` |
| mace-mp-0b2-medium | `a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf` |
| mace-mp-0b2-large | `348390e758e1c90011c7675e864850f8e9c5b3e7c217f79a2b4bad1baaa657ad` |
| mace-mp-0b3-medium | `2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012` |
| mace-off23-small | `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f` |
| mace-off23-medium | `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` |
| mace-off23-large | `a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4` |
| mace-mh-0, head omat_pbe | `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d` |

Source URLs are retained in `.tmp/symmetrix-xl/benchmarks/foundation_model_support_manifest.json`; admit bytes through the canonical external-resource policy and verify these hashes before trusting a row. These pins have not been downloaded or runtime-qualified here. At P0, also identify an actual one-interaction source fixture and pin its source configuration/provider/weights digest. The reference's `symmetrix/test/test_single_layer_mace.py:28-67` requires locally supplied checkpoints; it does not bundle a usable fixture. A tiny deterministic source-provider model exercising the separate invariant readout product is a lawful test oracle, but does not replace qualification of a named external checkpoint.

Permanent nonclaims for this plan: MACE-MH-1's nonlinear interaction architecture, MACEField polarization/field-response extensions, arbitrary custom e3nn modules, checkpoint conversion without declared architecture metadata, unrestricted high angular degree/body order, differentiability through neighbor discovery/image enumeration/ownership changes, universal numerical accuracy of a pretrained model, or matching the paper's capacity/speed/scaling numbers. These are different architecture/scientific contracts, not silent missing standard-MACE support.

### Planned execution support matrix

These are implementation and qualification obligations, not observed support:

| Route | Planned envelope | Required derivative/property proof |
|---|---|---|
| Native exact JAX | CPU float32/float64; JAX GPU targets qualified separately | E/F/S, parameter/coordinate JVP/VJP, mixed force/stress-loss gradients, admitted coordinate HVP |
| Native tabulated JAX | Same exact model, explicit radius/species/continuity envelope | E/F/S; higher derivatives only under an established knot/cutoff regularity contract |
| Accelerated Pallas | Explicit NVIDIA CUDA target, float32 and float64 separately qualified | Forward/source/geometry kernels, E/F/S and complete admitted native transform behavior; interpret mode is not GPU performance evidence |
| Distributed native | Real two-device/rank execution first; multi-host and each GPU/provider are separate tuples | Partition-local multilayer E/F/S, exactly-once adjoint, migration/restart and declared derivative orders |
| ASE / i-PI | Host protocol over a loaded native artifact and its selected qualified runtime | Actual units/tensor/property semantics, cache/lifecycle and failure handling |
| Frozen IREE | Current host-runtime export boundary, fixed shapes/capacities | Native E/F/S/status parity as explicit outputs; no deployment-time AD or neighbor discovery |

No automatic CUDA/HIP/TPU equivalence, FP64 acceleration, multi-host qualification or exported training follows from a selector or a CPU test. Register only measured tuples. The first accelerated target is CUDA because a real Pallas execution route is already compatible with the repository's JAX-native direction; HIP/TPU acceleration is not part of this plan's required backend implementation.

No new LAMMPS pair style, Kokkos runtime, NVRTC/hipRTC code generator, ONNX training route, or global SPH/DEM rewrite. LAMMPS interoperability, when exercised, uses existing i-PI transport and is labeled as such. The native JAX reference and accelerated route are different qualified executions of one scientific architecture.

## 2. Updated facts on current dev

| Area | Current source evidence | Consequence |
|---|---|---|
| Learned models | `nn/atomistic/_painn.py`, `_nequip.py`; only those models exported | MACE is genuinely new; do not claim e3nn-compatible NequIP |
| Prediction/training | `_prediction.py` and `_training.py` are dense finite-batch paths; training has E/F but no stress targets | Large/periodic training needs a prepared graph path, not just a new model class |
| Periodic learned runtime | `LearnedGraphPotentialTerm(allow_periodic=True)` already permits fixed-cell particle graphs; tested in `test_dynamics_periodic.py` | Preserve this existing route; earlier blanket 'no periodic learned potentials' was too broad |
| Graph geometry | `_graph.py` uses receiver-minus-sender and optional minimum image; dense realization ignores preserved cells; `edge_slots` is rebuilt with lexsort | Explicit image topology and prepared schedule reuse are required |
| Stress | `_stress.py` refuses directed learned graphs; program dynamic cells are restricted, including isotropic-only validation | Removing one refusal is insufficient; cell deformation and all geometry must use one convention |
| Sparse execution | `_execution.py` reduces already-produced route values | No callback-fused streamed edge/epilogue exists |
| Seeded sums | `reduce_key_groups`, `KeyGroupAccumulation` retain high and correction components across updates | Reuse seeded accumulation; do not collapse tile subtotals |
| Linear schedules | `_linear.py` prepares target/source row-gather layouts; optional CSR storage is a separate materialization route | Share preparation invariants, not CSR coalescing semantics for nonlinear edges |
| O(3) | `_o3.py` owns Cartesian degree 0–2; `_o3_tensor_product.py` supports `uvw`/component normalization | Preserve existing physical Cartesian fields; share the coupling owner when adding general layouts |
| Active O(3) consumer | LSP found `_constitutive.py:706` in `discretization/meshfree`, in addition to NequIP and tests | The newer learned meshfree laws are affected by any O(3) change |
| Harmonics/CG | Complex `special` harmonics; private real geodesy table; SU(2) CG; sparse spectral scalar-modal CG | Reuse coefficient/basis mathematics; don't equate a spectral product with a multiplicity-aware neural product |
| Polynomial/interpolation | Total-degree monomials exist, but no shared sparse product graph; Hermite interpolation uses interval search; B-spline grids and projection already exist | Extend owners, don't build a second polynomial or interpolation vocabulary |
| Meshfree closure | Certified support epochs, bounded local fitting, learned constitutive laws, owner-local distributed relations and halo transpose | Reuse current support/halo/lifecycle evidence rather than designing from the earlier snapshot |
| Distributed atomistics | Existing slab routes and reductions are real, but `halo_short_range_evaluate` evaluates the global potential then masks outputs | Not an owner-local multilayer MLIP; a true local graph evaluator is missing |
| Persistence | `_checkpoint.py` persists runtime state and requires a recreated matching potential; ML portable artifacts require `AbstractArrayModel` | A registered atomistic model recipe/array envelope is needed |
| Deployment | ASE structure adapters, external-provider i-PI transport, fixed-shape host IREE/ONNX executables | No native ASE MACE calculator; exports are not differentiable native models |

LSP observations at planning time: 19 references to `AtomisticGraph`; 21 references to `O3TensorProductPlan`. Counts include declarations and facades and are not coverage guarantees. Re-run at implementation time and include opaque/serialized/generated consumers through owner inventories.

### Binding corrections to the preliminary brainstorm

1. Do not predict a per-atom memory number or speed ordering from source. The earlier rough 10 KB/atom and millions-of-atoms estimates were unmeasured. Exact arrays, compiler buffers, derivative route, model widths, degree, neighbor density, allocator overhead, and deployment buffers determine capacity.
2. A receiver/source transpose schedule routes cotangents; it is not the derivative of a nonlinear edge function or epilogue. Recompute the local primal and apply its actual derivative.
3. A `custom_vjp` alone blocks normal forward-mode AD through that callable. Force training requires mixed coordinate/parameter derivatives; phonons require coordinate HVPs. Use ordinary JAX or a complete JVP/transpose primitive, not a first-order-only shortcut.
4. `lax.map` or rematerialization syntax does not prove a compiler memory bound or reproduce exactly three physical sweeps. Bound graph-wide intermediates and graph-wide carry tapes, then measure the transformed executable. More-than-two-layer replay is an explicit schedule.
5. Generic cubic Hermite with arbitrary nodal slopes is C1, not globally C2. The pinned reference computes slopes by a not-a-knot/clamped cubic-spline solve (`compact_radial.cpp:326-365`), not by JVP of the radial network. Force conservativity alone does not qualify Hessians or force training at knots/cutoff.
6. Imported generalized CG/U bases cannot be regenerated in an arbitrary equivalent basis while reusing W. Record the exact source basis, transform it explicitly, and retain the fixed W-to-polynomial map. Treating merged polynomial coefficients as independent trainable weights expands the source parameter space.
7. Graph displacements are receiver-minus-sender, whereas native particle pair displacement has the opposite convention. MACE's external convention must be read from its admitted provider, not guessed. The importer applies one audited basis/direction transform, including odd-degree signs.
8. For row lattice H and the usual column-vector Cartesian deformation F = I + strain, H' = H @ F.T. Current `_stress.py:87` uses `contract("ij,kj->ki", F, H)`, which is exactly H @ F.T, not F @ H. Preserve that convention; an earlier abbreviated source interpretation was wrong. Independent affine-deformation/shear controls are required, but this is not evidence of a strain-orientation bug or permission to change the current map.
9. A position-force moment about a mass center is not a general periodic cell virial. A self-image edge can exert zero net positional force but nonzero cell stress. Derive periodic stress from the same scalar energy with explicit image vectors.
10. Minimum-image guards must remain for classical pair-once terms. A new image-aware learned graph cannot silently change pair multiplicity or admit classical mixtures outside their declared support.
11. The source project is not uniformly MIT: its separate `pair_symmetrix` integration is GPLv2. Weight licenses, data terms and provider dependencies are separate rights. A readable URL or SHA-256 does not grant redistribution or establish trust in a pickle.
12. No JAX persistent cache is currently configured by Phydrax. A deployment environment may enable it, but compiler cache hits do not establish scientific identity or portability. Reuse existing artifacts/signatures rather than a second cache manager.

## 3. Architecture and invariants

### Ownership

- `phydrax.sparse`: endpoint relations, stable route grouping, bounded edge/node schedules, reduction order and routed derivative execution.
- `discretization` periodic/particle/spatial owners: lattice/image enumeration, neighbor candidates, support certificates, ownership and halos.
- `nn.operator.representations` and `nn.operator.layers`: real O(3) layouts, component bases, parity, coupling legality, normalization and tensor-product execution.
- `_polynomial`: sparse product graph; `nn.atomistic` owns MACE's generalized coupling/basis parameterization.
- `_interpolation`: spline grids, preparation/evaluation and derivative continuity; `nn.atomistic` owns radial-family admission and approximation fidelity.
- `atomistic`: shared energy/prediction/training/dynamics/stress and durable scientific model/state boundaries.
- `execution`, `lifecycle`, `qualification`, `export`, `_external_runtime`, `_array_archive` and registered model reconstruction remain canonical owners of their existing invariants.

### Implementation style and contracts

New modules inherit `StrictModule`, are final by default, validate before assignment, use runtime-resolvable nominal `phydrax.typing` contracts and `__strict_contract__ = True` where required, and keep all functions/helpers truthfully annotated. Reuse `checked` only at audited effective boundaries; preserve source-addressed callable/compiler identity or explicitly invalidate it. Constructor-bypassing restore remains inside the registered reconstruction owner and validates the full restored value once. No static numerical arrays, blanket annotation hooks, alternate shape vocabulary, aliases, internal schema generations or guessed scientific identities.

Shared helpers own validation/preparation/execution/commit/evidence invariants, not trivial forwarding. Measure changed-symbol cyclomatic/cognitive complexity before and after; decompose substantial owners without changing source-faithful formulas or reduction order gratuitously. Explicit dtype and host/device boundaries apply to package, tests, tooling, examples and manifests.

### Static versus dynamic

Static preparation: declared species order, l/parity/multiplicity, basis convention, legal coupling/monomial incidence, capacities, tile schedule shape, chosen algorithm/precision/accumulation, callable code identity and derivative route. Device-visible numerical leaves: weights, node state, coordinates, cell matrices, fixed numerical coefficient tables, masks/indices when runtime-bound, and source/target data. Fixed role is not static numerical storage.

Structure/schedule identity and bound topology identity are distinct. `RelationExecutionState.execution_id` does not currently fingerprint route index/mask content. The new binding must carry the relation owner schema/epoch/content identity; do not mistake equal shapes/capacities for the same topology. Active cutoff masks may change within an accepted candidate epoch; route membership/order and integer images change only through rebind/reprepare.

Source semantics, exact native weights, source-to-native transform, exact versus tabulated radial realization, execution signature, bound graph/cell/owner epoch, and numeric revision are separately identifiable. Changing weights invalidates folded/table artifacts without recompiling a same-signature native kernel; changing static architecture/backend/capacity requires a new signature. Never fake historical IDs after changed source, fields, or partitioning.

### Bounded execution

For a fixed architecture and capacities:

`live storage = persistent topology/geometry + inter-layer node boundaries + admitted outputs + one bounded edge/receiver/channel workspace + replay/halo/parameter-cotangent storage`.

This is not O(1) total memory. Requested edge latents are O(E times latent width) public/inter-layer outputs and cannot be eliminated without changing model semantics. Source/parameter cotangents cannot be returned as one graph-sized partial per tile and then summed: accumulate into one live target/parameter state or a bounded reduction tree.

Receiver tiles own complete epilogues. High-degree receivers may need several edge fragments; maintain their seeded high/correction accumulator across fragments and commit the epilogue only after the final fragment. Do not assume every row fits a single tile, or allocate N times maximum degree padding without a charged bound. Channel/coupling/monomial scratch has its own cap.

### Differentiation

Integer route discovery, image enumeration, sort/permutation, owner assignment and topology epochs are not differentiable. Within an accepted fixed candidate topology, differentiate numerical geometry, cell matrices, learned parameters and node states. Smooth cutoff support and padded-safe callback domains are required. A failed scientific/resource state must propagate failure/derivative-invalid evidence; finite partial values do not imply success.

Support paths: energy; E/F/S; coordinate JVP/VJP; parameter JVP/VJP; mixed force/stress-loss parameter gradient; coordinate HVP for elastic/phonon observables on a suitably smooth route. No full Jacobian/Hessian when the consumer needs an action.

Mixed force-loss parameter differentiation needs differentiability of the first coordinate derivative with respect to parameters; it does not automatically require a globally C2 coordinate function. Coordinate HVP/elastic/phonon derivatives do require their own spatial regularity. Preserve current PaiNN/NequIP cosine cutoffs and report their piecewise spatial second-derivative boundary honestly; do not smooth a checkpoint's cutoff silently. Padded callbacks use explicitly admitted domain-safe sentinels (for example, nonzero dummy harmonic directions), not the assumption that a zero input is valid for every scientific function.

## 4. Dependency graph and work ownership

| Phase | Dependencies | Deliverable |
|---|---|---|
| P0 | current dev | Contract/support/migration ledger and source rights |
| P1 | P0 | Shared prepared streamed relation execution |
| P2 | P1 | Transform-safe replay/resource accounting and current-consumer cutovers |
| P3 | P0, P1 | Explicit periodic image graph and cached geometry lifecycle |
| P4 | P0 | General real O(3) layouts/coupling with existing Cartesian consumer preservation |
| P5 | P4; interpolation may proceed earlier | Sparse product graph and smooth radial preparation |
| P6 | P1, P2, P4, P5 | Native exact/trainable and prepared standard MACE |
| P7 | P3, P6 | Complete prediction, force/stress training and periodic dynamics |
| P8 | P4–P7 | Trusted source conversion, pickle-free model artifact, fidelity campaign |
| P9 | P1–P3, P6, P7 | Owner-local multilayer distributed execution and migration |
| P10 | P1, P2, P4–P8 | Real accelerated specialization and transformed kernel evidence |
| P11 | P7–P9; P10 for accelerated deployment | ASE/i-PI/frozen export and fresh-process continuation |
| P12 | all | Numerical, scientific, performance, operational and release dossier |

One integration owner controls shared facades, graph/program result types, O(3) identity changes, generated inventories and final checks. After P0 contracts are frozen, P1/P3/P4 can develop concurrently; radial preparation and artifact/security work can run independently of kernel optimization. Children skip build/lint/tests/formatting mid-flight; run each selected integrated check once afterward. Do not parallelize overlapping state/identity migrations without an integration owner.

## P0. Contract, support and migration inventory

### Files

- `phydrax/atomistic/_potential.py`, `_types.py`, `_graph.py`, `_prediction.py`: write the accepted fixed-route E/F/S, species, units, precision, cell/image and provenance contracts before implementation.
- `phydrax/sparse/_relation.py`, `_execution.py`, `_key_groups.py`: record current grouping/reduction/safe-index invariants; define schedule content binding and chunk-seeded semantics. Preserve ordinary existing `reduce` behavior; it need not become a costly callback forwarding wrapper.
- `phydrax/_identity.py`, `_trainable.py`, `_execution_resources.py`, `_execution_plan.py`: reuse, rather than redesign, executable/numeric/artifact identities and resource admission. Extend only a genuinely missing structured owner fact.
- `phydrax/qualification/_catalog.py`, `_builtin_catalog.py`, `_builtin_sources.py`, `_source_reference.py`: declare the closure exception and source/rights ownership. Keep architecture profiles, exact backend tuples and signed release authority distinct.
- **new** `phydrax/atomistic/_mace_profiles.py`: domain-owned candidate support rows, derivative/geometry/backend envelopes and permanent refusals, registered through the existing catalog.
- `NOTICE`, `LICENSES/`, `pyproject.toml`: attribution for any actually ported code; optional import-only provider dependencies with admitted releases. Do not add torch/e3nn/mace as package-runtime dependencies or copy the GPL pair style.

### Prerequisite correctness repair: zero-valued active event derivatives

Source inspection of `_execution.py:438,447,466-470` shows `_sum_segments` branches on `subtotal != 0` or `value != 0`. [INFERENCE] Ordinary JAX differentiation through those branches suppresses tangents/cotangents of valid zero-valued events, and seeded fast mode suppresses a nonzero tangent when primal events cancel to a zero subtotal. This must be repaired in the owning additive-reduction contract before migrating PaiNN/NequIP or using this reducer as the derivative oracle.

Preserve established primal zero/no-op and high/correction semantics where those serve retention/padding consumers, but make additive-sum JVP/transpose depend on route/event validity, not numerical primal zero. Implement a transform-complete native derivative rule if retaining the primal branch requires one; do not replace it with `custom_vjp` alone or a numerical-zero test on tangent values. Explicit structural retention/padding events remain inert through their masks. Reuse the owner rather than invent a model-local corrected sum.

Extend `tests/unit/test_sparse_substrate.py` with exact-zero and cancellation cases for seeded/unseeded fast/deterministic/compensated sums, JVP/VJP and a mixed derivative. An active edge `m(theta,x) = theta*x + x*x` at x=0 has coordinate derivative theta and mixed coordinate/parameter derivative 1; its primal message is exactly zero. Contrast with an invalid padded event whose derivative is zero. Include nonzero opposite events that cancel in a seeded chunk. Keep primal retention/correction and existing projector/grouped consumers covered; actual smoke/regression proof belongs to implementation, not this read-only finding.

### Inventory obligations

LSP-reference all changes to `AtomisticGraph`, `AtomisticGraphExecutionPlan`, `AbstractAtomisticPotential`, `O3TensorProductPlan`, `O3Representation`, prediction/training types and stress functions. Supplement LSP with registered artifact types, serialized recipes, source-addressed callables and generated inventories. Retain a consumer/contract migration map in the final implementation report; delete stale examples/refusals only when new behavior is exercised.

### Acceptance

Every C01–C20 item has a named owner and observable workflow; every support tuple specifies architecture, geometry, precision, derivative order, execution backend, provider and capacities. Failed/refused/unsupported states cannot be relabeled as accepted inference. Native model construction needs no optional importer/provider.

## P1. Shared bounded streamed relation execution

### Files and changes

- **new** `phydrax/sparse/_streamed.py`: `StreamedRelationPlan`, `PreparedStreamedRelation`, and a typed result/evidence record. Plan construction validates receiver, edge, channel, schedule/fragment and derivative workspace capacities. Prepare target-major and source-major stable route orders, row offsets/counts, safe endpoints, fragment-finalization masks and reversible route maps. Accept `EdgeRelation` and `RowRelation` through their existing validation.
- `phydrax/sparse/_execution.py`: expose preparation through the existing execution owner and share seeded reducers. Carry `KeyGroupAccumulation` high/correction state across fragments. Do not collapse events into independently summed chunk totals for deterministic/compensated mode. Leave unrelated min/max/mean contracts intact; the nonlinear streamed scientific primitive is initially additive, with epilogue normalization explicitly model-owned.
- `phydrax/sparse/_key_groups.py`: reuse stable key/group identity and reversible permutations. Factor missing shared schedule preparation from this owner only if it owns a real invariant. Duplicate numerical edges must remain distinct callback events; CSR coalescing is not generally legal.
- `phydrax/sparse/_linear.py`: reuse/refactor receiver/source row-schedule construction where valid. Preserve its sparse coefficient refresh and optional CSR materialization ownership. Do not make all linear operators use a nonlinear evaluator.
- `phydrax/sparse/_relation.py`, `_ops.py`: minimal shared types/validation additions only; image shifts are not generic endpoint-relation fields.
- `phydrax/sparse/__init__.py`: explicit exports for the new canonical streamed capability; no graph-local alternate runner.

### Execution contract

The prepared evaluator takes dynamic source/receiver/edge data and a PyTree-capable pure edge callable plus receiver epilogue. Typed protocols specify accepted payloads, fixed output shape and callback admissibility; callables carrying arrays remain dynamic leaves with fixed-role metadata as appropriate. Prepare compiler/output signatures once. No new jit wrapper inside evaluation, host hash of weights inside a kernel, hidden array capture, callback output probe or shape-dependent Python numerical loop.

Evaluate safe finite dummy inputs for padded lanes, mask before numerically unsafe operations, accumulate only valid route payloads and commit receiver outputs once. Receiver epilogues may emit node state and scalar/per-node observables; optional edge outputs must be explicitly requested and charged. Reject callbacks whose dependencies require undeclared global state, variable output width or unsupported semantics.

Oversized rows are either split into bounded fragments with seeded state or refused at the declared fragment/schedule limit; never truncated. Empty rows still receive their lawful zero-aggregate epilogue and node mask. Prepare transpose ordering independently; swapping endpoints does not sort them.

### Behavioral tests

- **new** `tests/unit/test_sparse_streamed.py`: independent small reference for irregular/duplicate routes, case isolation, masked out-of-range padding, empty rows, fragmented single high-degree receiver, receiver residual/update, scalar/block/PyTree payloads, all accumulation modes, and route reorder under stable event IDs.
- Extend `tests/unit/test_sparse_substrate.py` for shared seed/boundary changes only; preserve current linear transpose/adjoint coverage.
- Test a receiver epilogue with direct receiver/residual dependence together with a requested edge-output loss and a node-observable loss. Verify their combined source/receiver/edge/parameter cotangents against an independent per-edge reference; returning edge latents must not drop their cotangent or reverse the epilogue twice.
- Test cancellation split across fragments so dropping correction or summing subtotals fails; test a nonlinear epilogue whose incorrect per-edge application visibly changes the output.

Smoke: evaluate an irregular nonlinear graph with edge capacity larger than a tile, receiver degree larger than a fragment, a padded row and a duplicate edge; observe outputs, successful evidence and the intended capacity refusal through the public sparse API.

## P2. Transform-safe replay, resource admission and current consumer cutovers

### Files and changes

- **new** `phydrax/sparse/_streamed_derivative.py`: bounded transformed execution rules only where ordinary JAX cannot preserve the required ownership/memory/order contract. The native reference uses ordinary differentiable JAX. Reverse the receiver epilogue, recompute local edge values, apply actual local VJP/JVP and route source/receiver/edge/parameter cotangents. Share one `PreparedLinearization` when primal/JVP/VJP act at one point; never separately call linearize and vjp at that point.
- `phydrax/_numerics/_checkpointed_scan.py`: reuse existing replay modes/schedules; extend only if the new relation schedule requires a demonstrably missing leaf/payload contract. Rematerialize edge AND node-product/readout internals. Avoid a scan whose graph-wide carry is retained once per tile. Parameter cotangents and global source cotangents have one admitted live accumulator or a bounded reduction tree.
- `phydrax/_execution_resources.py`, `_execution_plan.py`: attach persistent, scratch, replay, derivative, parameter-reduction, output, halo and compilation evidence. Use existing `memory_basis` distinctions and certified-resource admission. Caller-supplied budgets select qualified fits; no-fit is refusal, not an attempt to allocate the smallest plan.
- `phydrax/nn/atomistic/_painn.py`, `_nequip.py`: migrate message computation and receiver update onto the shared prepared relation route. Preserve current sinc/cosine radial formulas, parameter order, gating, precision, masks and semantic finite/periodic distinctions. Remove superseded direct receiver scatter helpers, not the underlying scientific architecture.
- `phydrax/graph/_equivariant.py`, `_neural_operators.py`: migrate `EquivariantGraphConvolution` and `GraphKernelIntegral` aggregation where callbacks meet the contract. Attention/global normalization retains its explicit ownership; do not silently convert a global operation into tile-local normalization.
- `phydrax/graph/_architectures.py`: migrate `MeshGraphNetBlock` through the same evaluator. Its returned/persistent edge latents remain graph-wide requested outputs; bound extra hidden activations, not the edge output itself. Preserve edge/node residual and global features.
- `phydrax/discretization/meshfree/_constitutive.py`, `_conservation_solve.py`: migrate the measured nonlinear edge-law gather/evaluate/accumulate path, preserving edge reversal/action-reaction, coverage, monotonicity/coercivity, support epoch and implicit derivative evidence. Do not rewrite local fitting, all transport limiters, SPH contact/history, or DEM without a separate concrete need.


### Reverse scheduling decision

Default bounded reference reverse replays one receiver tile, reverses its epilogue once, computes local edge VJPs, groups that tile's source events by prepared stable source IDs, and updates the live graph-wide source cotangent with seeded accumulation. It does not retain every receiver's wide product/message cotangent or emit one full parameter-gradient tree per tile. Its declared canonical event order includes receiver-tile order; within each source's events the prepared stable order is explicit.

A globally source-major optimized reverse is admissible only when required receiver cotangents are available as charged node-boundary data or can be recomputed within an admitted work budget. Merely switching the outer loop to sources can replay an entire receiver epilogue for every neighbor and create degree-squared work, or require a forbidden graph-wide wide-message tape. Charge that tradeoff and benchmark it; the plan does not prescribe a source-major outer loop at the expense of its bounded-memory contract. MACE-specific source/edge kernels must match the same local derivative and accumulation semantics.
### Derivative and memory acceptance

Reference and optimized routes must cover primal, JVP, VJP, parameter update, gradient of force loss, gradient of stress loss once P7 exists, and coordinate HVP. Every specialized custom rule has an ordinary-JAX reference and nested transform tests. Determinism means a defined event/owner order on the admitted device/backend; fast GPU scatter is not called deterministic. Compensated accumulation is not exact real arithmetic.

Compiler memory analysis must be run on E/F/S and force-loss transforms, not only the forward. Evidence separately records logical retained arrays, argument/output/alias/temp/code bytes, sampled peaks and device headroom. A syntax-level remat or a sampled peak is not a bound. Failures retain diagnostic evidence and never yield a finite partial value with valid derivatives.

### Tests, smoke and benchmarks

- Extend existing `tests/unit/atomistic/test_painn.py`, `test_nequip.py` for tiled/reference E/F, masks, cutoff and mixed parameter/coordinate derivatives.
- Existing graph tests: `tests/unit/graph/test_native_route_message_passing.py`, `test_graph_architectures.py`, `test_graph_equivariant.py`, `test_graph_neural_operators.py`; preserve consumer outputs rather than callback wiring.
- Existing meshfree tests: `tests/unit/discretization/meshfree/test_constitutive_laws.py`, `test_local_stencil_lifecycle.py`, and `tests/integration/test_meshfree_learned_flux_training.py`.
- **new** `tests/unit/test_sparse_streamed_derivatives.py`: nonlinear finite differences, duality, shared array-bearing callback parameters, mixed force-loss derivative, complex transpose versus adjoint for the linear subcase, and invalid derivative propagation.
- **new** `benchmarks/streamed_relation_scaling.py`: benchmark forward, reverse and mixed derivative separately across edge count, degree skew, channel count and tile capacities. Reuse phase/memory utilities from current meshfree/sparse campaigns, not a new metrics format.
- `tools/sparse_execution_benchmarks.py`: extend its existing campaign entry points/reporting to the shared runner without source-text tests.

Smoke: actual PaiNN and NequIP energy/force calls, a force-supervised optimizer update, a MeshGraphNet block retaining edge state, and the meshfree learned-flux training example on the changed path. Observe parity/status and real compiled transformed memory.

## P3. Periodic image graph, geometry and neighbor lifecycle

### Representation decision

Keep `ParticlePairRelation`'s unordered, distinct-particle, pair-once contract. Add an image-aware directed relation in the particle/spatial owner; do not weaken every classical consumer to permit self or repeated image pairs. Atomistic graphs consume both canonical relation kinds under explicit preparation semantics.

For row cell H, define one native graph displacement:

`d_e = r_receiver - r_source + n_e @ H`.

Partial periodic axes force corresponding integer components to zero. Exclude `(source == receiver, n == 0)`; include every nonzero self image and every admitted distinct-pair image exactly once per directed route. Route identity is stable source/receiver identity plus integer n and case, never slot or distance coincidence. Reversal maps `(source, receiver, n)` to `(receiver, source, -n)`.

### Files and changes

- `phydrax/discretization/_periodic_cell.py`: add bounded image-translation enumeration/certification separate from minimum image. Bound the required integer stencil from the lattice/inverse geometry and cutoff-plus-skin, with native rank/condition evidence; reject unbounded/ill-conditioned or over-capacity enumeration. Its existing image stencil for nearest-image search is not automatically a complete cutoff stencil. Support fractional wrapping and fixed shifts under changing H.
- **new** `phydrax/discretization/particle/_image_relation.py`: typed directed image relation and candidate/completeness evidence. Host immutable preparation uses NumPy; dynamic geometry uses JAX. Charge images, candidate slots, edges and receiver degree separately.
- **new** `phydrax/discretization/particle/_image_neighborhood.py`: prepare bounded image-aware spatial candidates by reusing native cell/Morton query owners. Do not enumerate N² times image count except an explicitly capped dense reference requested for validation. Route packing uses stable endpoint/image keys, valid masks and case isolation.
- `phydrax/discretization/particle/_neighborhood.py`, `_verlet.py`: compose image-aware candidate state with existing lifecycle. Cache relation, shifts, reference geometry/cell, active mask, epoch/rebuild count, schedule binding and certificate margin. A cell-deformation bound must account for each stored image coefficient; the current minimum-image scalar deformation formula cannot be assumed sufficient.

Image completeness under cell deformation also covers images absent from the cached relation and integer translations outside the current enumeration stencil. A bound only on stored n cannot establish absence of newly entering images. Bind a certified deformation envelope with a lower lattice singular-value/reciprocal bound and a conservative excluded-image shell; re-enumerate/refuse when that witness expires. Test shrinking/shearing a cell so a previously absent image enters r_c plus skin from outside the old stencil, not only movement of already retained routes.
- `phydrax/discretization/particle/_cell_list.py`, `_metric_cell_list.py`, spatial query owners: add only missing candidate/query capability at the canonical owner. Retain unique-image guards for classical pair plans.
- `phydrax/atomistic/_graph.py`: separate graph topology preparation from numerical geometry binding. Carry image shifts, candidate versus active masks, bound schedule, cell/image identity and overflow evidence. Reuse schedule until a candidate epoch changes; remove per-force-call lexsort/edge-slot reconstruction. Distances/directions can be computed inside tiles when consumers do not need graph-wide geometry; requested GraphIR geometry remains explicitly charged output.
- `phydrax/atomistic/_types.py`, `_system.py`: support per-case cell/partial-PBC binding without making scientific identity depend on equal shape or species order. Preserve stable IDs and element/atom-type distinction.
- `phydrax/atomistic/_dynamics.py`, `_potential_program.py`: term-specific geometry admission. Pure image-aware learned programs may exceed the unique-image radius; mixed classical programs still obey every term's own guard. Derivative closure binds fixed shifts and indices before differentiation.

### Wrap, strain and overflow

Rewrapping atoms changes the representation of the same physical image edge. Either evaluate candidate geometry with consistent unwrapped/reference coordinates or update n by the exact endpoint image-count difference; never keep old n with newly wrapped coordinates. Verify identical displacements before/after crossing a periodic face without a full graph rebuild when the skin certificate remains valid.

Topology rebind and capacity replacement are explicit lifecycle transactions. For a capacity-only failure, retain accepted coordinates, RNG/thermostat state and force cache; choose the next declared capacity ladder entry on the host, rebuild and retry the same physical attempt. No silent retry of a scientific failure, no host synchronization inside the numerical loop, and no unbounded growth. Active-species changes invalidate species-prepared tables when unsupported; they are not silently mapped to another element.

### Tests and acceptance

- **new** `tests/unit/discretization/particle/test_image_neighborhood.py`: brute-force host lattice enumeration oracle, skew/partial PBC, duplicate image exclusion, self images, reversed shifts, reorder/padding/case isolation, image/candidate/edge/degree overflow and conservative cell/wrap certificates.
- Extend `tests/unit/atomistic/test_types_and_graph.py`, `test_dynamics_periodic.py`, and existing particle dynamic-relation tests for composed lifecycle behavior.
- Demonstrate primitive-cell/supercell E-per-atom and forces equivalence on a small periodic model; a one-atom cell has zero positional force and a nonzero strain response. This catches missing self images and incorrect factor-of-two counting.

Smoke: a five-atom periodic crystal with cutoff larger than its unique-image radius, a triclinic cell, a face-crossing Verlet update and a failed-capacity transaction. Confirm actual route counts against the independent oracle.

## P4. General real O(3) representation and coupling

### Design decision

Retain `O3Representation`/`O3Features` as the truthful Cartesian scalar/vector/STF physical-field representation already used by NequIP, EqGINO and meshfree laws. Add a general ordered real-irrep layout for arbitrary admitted l/parity/multiplicity. Both lower into the same canonical coupling preparation/execution in `O3TensorProductPlan`; there is no second MACE tensor-product evaluator or duplicate coupling generator. Distinct physical Cartesian fields and general packed irreps are not compatibility aliases.

Preserve existing low-degree block order, basis, multiplicity/path indexing and learned parameter shapes. Preserve current analytic low-degree coefficients/operation order where unchanged. Adding source-addressed code, metadata or fields can still change fingerprints: record actual invalidated IDs and rebuild affected artifacts; do not claim a bitwise identity-preserving refactor without evidence. This plan does not authorize rotating existing NequIP/meshfree weights into a new hidden basis just to simplify MACE.

### Files and changes

- `phydrax/special/_spherical_harmonic.py`, `_solid_harmonic.py`, `special/__init__.py`: expose a prepared real Cartesian harmonic evaluation with explicit normalization, basis ordering and admitted maximum degree. Reuse existing regular-solid recurrence/basis mathematics where numerically appropriate; avoid chart singularities from theta/phi differentiation at poles. Zero-length valid directions have an explicit domain policy, not guessed values; inactive padding is safe before evaluation.
- **new** `phydrax/nn/operator/representations/_irreps.py`: `O3IrrepLayout` with nominal block identities, l, parity, multiplicity, real orthonormal component basis and explicit packed ordering. Canonicalize declarations without losing repeated distinct block/path identity. Scientific component/multiplicity axes use native axes.
- `phydrax/nn/operator/representations/_o3.py`, `__init__.py`: expose a preparation-time block description/basis map for the existing Cartesian layout, not a new forwarding public representation. General operations prepare once; no per-edge conversion allocation.
- `phydrax/tensor_network/_su2.py` and `phydrax/discretization/spectral/_spherical_algebra.py`: reuse shared host coefficient preparation and complex/real basis transformation. Extend the coefficient owner only for generally useful missing real-basis capability; retain spectral scalar-modal multiplication semantics. General real phases are derived and tested, not a blanket heuristic phase rule for every basis.
- `phydrax/nn/operator/layers/_o3_tensor_product.py`: accept either native layout through one normalized preparation contract. Support explicit per-path `uvw` and `uvu` incidence, multiplicity equality requirements, weighted/unweighted paths, native component normalization and explicit imported path scales. Bound path count, weights, coefficients, instructions and execution working set before allocation. Sparse rows omit only mathematical zeros or explicitly qualified coefficient truncation; no unexplained 1e-12 pruning.
- `phydrax/nn/operator/layers/_o3.py`: general pointwise/gated operations over prepared blocks as needed for MACE; existing Cartesian methods keep their meaning. MACE scalar nonlinear readouts are not replaced by NequIP gates.
- `phydrax/axes`, `phydrax/ein`: use existing semantic identities and ordinary native contractions; do not add a general sparse branch to opt-einsum or a parallel axis runtime.
- `phydrax/discretization/meshfree/_constitutive.py`: reconcile its owned path-weight and tangent-layout checks against the shared plan, with no semantic weakening. Inspect `_edge_product` and `_require_edge_product` specifically.

### Basis/import invariants

Source e3nn/SpheriCart conventions include direction choice, component ordering, normalizations and phases. Preserve the source convention metadata at the interchange boundary; transform all harmonics, CG/U coefficients and feature blocks consistently into the native declared basis. Equal irreps dimensions are not a basis map. Test inversion/reflection parity and not just SO(3) rotations. The target need not publish external provider-specific aliases to expose a coherent native API.

### Tests and smoke

- Extend `tests/unit/nn/test_o3_tensor_product.py`: maintain existing scalar/dot/cross/STF contracts; add l=3 and l=4 coupling, `uvu` multiplicity requirements, illegal parity/degree/path refusal, scalar weights versus dynamic tensor weights and coefficient/parameter budgets.
- **new** `tests/unit/nn/test_real_irreps.py`: explicit orthogonal basis conversion, unitary real rotation/reflection, general packing and no accidental equal-size interchange.
- Extend appropriate existing special/spectral/SU2 tests found by LSP; independent low-degree Cartesian formulas and high-degree harmonic addition/orthogonality controls avoid a same-generator self-oracle.
- Keep meshfree constitutive rotation, edge-reversal, monotonicity and owned-weight tests; static typing cases cover both accepted layouts and invalid kinds.

Smoke: actual tensor products using Cartesian physical fields and a general degree-three irrep layout, each with parameter/coordinate JVP/VJP; existing meshfree edge law remains numerically valid.

## P5. Sparse product graph and radial realization

### Product graph

- **new** `phydrax/_polynomial/_product_graph.py`: bounded canonical monomial/product-factor preparation. A node references a parent and one factor; merge equal commutative monomials with deterministic multiplicity accounting. Plan count/degree/level/live storage before construction. Evaluate levels with JAX indexed products and bounded receiver/channel tiles, not Python loops over atoms/edges.
- `phydrax/_polynomial/__init__.py`: expose only a genuinely reusable substrate needed by consumers. Existing total-degree/PCE evaluators are not automatically rewritten; preserve their formulas/ordering unless a measured consumer warrants a separate migration.
- **new** `phydrax/nn/atomistic/_symmetric_contraction.py`: MACE's output irrep, correlation order, species-conditioned W, generalized U basis and sparse W-to-polynomial map. The native trainable model retains original W and a fixed coupling map. Imported original/reduced source bases are explicit. Frozen execution may bind merged coefficients, tied to the original parameter revision, but may not optimize them as independent source parameters.
- Reuse `SparseLinearMap`/native route contraction for coefficient binding. Bound basis generation work and avoid dense dimension-to-correlation-power arrays. Optional source extraction must charge provider U tensors before transfer. Runtime degree/correlation loops are static architecture loops with measured code growth.

Tests: **new** `tests/unit/polynomial/test_product_graph.py` for independent polynomial values/gradients/HVPs, repeated factors, zero inputs (no derivative by division), merge multiplicity, coefficient updates and resource refusal. **new** `tests/unit/atomistic/test_symmetric_contraction.py` for equivariance, original versus explicit transformed U basis, source W parameter gradients and forbidden stale binding.

### Radial realization

Exact radial network is the native training and reference default. Frozen inference may explicitly select a qualified tabulated realization; it has different numerical identity, not different learned parameters. Unsupported families fail rather than approximate by a convenient Bessel/cutoff shape.

- `phydrax/_interpolation/_piecewise.py`: reuse cubic segment value/derivative evaluation; prepare uniform span lookup once and use constant-time arithmetic for admitted uniform grids. Retain general nonuniform search. Do not hide host `is_uniform`/array synchronization inside runtime.
- `phydrax/_interpolation/_bspline_grid.py`, `_bspline.py`, `_bspline_projection.py`: reuse grid continuity/jet and native prepared solve substrates. Add a prepared uniform spline evaluator and constrained C2 preparation where missing, not a local atomistic tridiagonal solver. Factor/prepare once and solve all radial channels/pairs as multi-RHS.
- **new** `phydrax/nn/atomistic/_radial.py`: exact Bessel, polynomial envelope, optional Agnesi transform, radial MLP, admissible postprocessing and source scales. Validate source checkpoint semantics including bias/activation/normalization and cutoff placement.
- **new** `phydrax/nn/atomistic/_radial_projection.py`: domain-owned projection declaration and retained fidelity evidence. For source-style cubic splines, use the actual not-a-knot/clamped preparation, with continuous values/first derivatives and the resulting C2 interior. Check the cutoff extension separately: clamped first derivative alone does not establish matching second derivative to zero outside. A higher-derivative admitted table must enforce/source-match boundary jets through the existing constrained spline/linalg owner; otherwise Hessian/force-training use remains exact-network execution, explicitly reported.

Choose projected-width versus embedding-width tables by an explicit execution objective and evidence. Table banks are prepared only for ordered species pairs whose source transform/network requires them; do not assume pair symmetry. No S² full-domain table allocation merely because a checkpoint supports many elements. Active-species binding may not discard the full model's scientific species domain. Training weight updates invalidate frozen tables; training uses exact networks unless a separately declared differentiable projection recomputation route is intentionally admitted and qualified.

### Qualification gate

Source exact network versus table: nodes, mid-spans, independent linear/logarithmic radii, lower-bound behavior, just-below/at/above cutoff, first derivatives and requested higher derivatives. Record held-out maximum errors in internal feature units. Separately compare end-to-end energy/force/stress and NVE drift; internal feature error is not an eV/force bound. Refuse out-of-domain radii under the declared table support; no invisible exact fallback under a tabulated name. Qualification gates and tolerances are fixed before collecting results and retain failures.

Tests: extend existing interpolation tests with uniform/general parity, boundary support and one-sided knot jets. **new** `tests/unit/atomistic/test_radial_projection.py` covers source slope semantics, conservative table forces, second-derivative availability, exact-versus-table identity, ordered pair binding, stale weights and failed gate. Smoke a same-model exact/table E/F/S call and cutoff sweep; benchmark radial preparation separately from warm table use.

## P6. Native standard MACE and prepared inference

### Files and changes

- **new** `phydrax/nn/atomistic/_mace.py`: final `MACEPotential(AbstractAtomisticPotential)`; immutable configuration and dynamic parameter leaves. Match the shared native model/scale/precision/species boundary, not an unrelated inference-only class.
- **new** `phydrax/nn/atomistic/_mace_interaction.py`: source-faithful interaction kernels and prepared plans. Source embedding/linear_up, radial-weighted `uvu` coupling, exact neighbor normalization or admitted density function, linear_down, species self-connection, residual and product update have separate coherent ownership. Density-normalized and residual variants are exhaustive admitted selectors; unknown variants refuse.
- **new** `phydrax/nn/atomistic/_mace_readout.py`: per-interaction scalar readouts, nonlinear last readout, source E0/scale/shift, head-specific offsets/scales and optional ZBL. Preserve source placement of scaling: atomic reference energies must not be accidentally multiplied twice. Atomic-number identity and checkpoint species order are explicit.
- One-interaction invariant readout is a separate owned product, not merely the two-layer architecture with its second interaction removed. `_mace_readout.py` must execute the admitted `readout_reshapes` -> `readout_products` symmetric contraction -> product linear -> scalar nonlinear readout sequence, with its own correlation, U basis and W parameters, self-connection policy and layout binding. The source fixture has readout correlation two. `_mace_prepare.py` keeps this node-local product/readout inside the tiled epilogue and accounts for its derivative workspace. Native construction, source conversion, artifacts and accelerated execution all represent this explicit stage.
- **new** `phydrax/nn/atomistic/_mace_prepare.py`: system/species-bound frozen execution using the shared relation and whole-receiver epilogue. Compose consecutive linears only across a boundary that is genuinely linear and has no intervening residual, activation, head dependence or field input. Bind every fold/table to source numeric revision and transformation evidence.
- `phydrax/nn/atomistic/__init__.py`: explicit `MACEPotential` and needed configuration/preparation exports, with import-only optional providers kept lazy/outside package runtime.
- `phydrax/atomistic/_potential.py`: minimally extend prepared-model capabilities as needed, retaining ParameterOwner discovery and NumericRevision as canonical. Fixed U/table inputs are nontrainable dynamic leaves; original source W/MLP/embeddings/readouts remain parameter leaves. Do not mark the whole prepared model nontrainable if its path supports training.

### Mathematical pipeline

Each interaction first prepares source features; each receiver tile recomputes radial/angular edge factors, accumulates legal sparse coupling contributions, then runs the complete symmetric product, self/residual linear update and readout. Only inter-layer node states and requested outputs escape the tile. Do not enumerate neighbor tuples to obtain higher body order.

First-layer species-only contractions and adjacent linear composition are preparation optimizations, never assumptions about every model. The exact trainable model is usable before folding. If preparing a new execution does not preserve the declared architecture, fail; no reduced width/body-order substitute.

### Tests and smoke

- **new** `tests/unit/atomistic/test_mace.py`: independent tiny scalar/vector MACE calculations, E0/scale-shift/head precedence, residual/density variants, degree/parity equivariance, species reorder, padding, atomic energy sums, cutoff continuity and capacity refusal.
- **new** `tests/unit/atomistic/test_mace_execution.py`: exact reference versus streamed/folded execution, multiple layers, high-degree receiver, tiles crossing species/head blocks, parameter mutation invalidating preparation, E/F/JVP/VJP/mixed derivative parity.
- Extend `tests/integration/atomistic/test_training_and_rmd17.py` with native MACE force-supervised fitting using the existing trainer. A small bounded synthetic contract is sufficient here; scientific foundation-model accuracy belongs to P12.

Smoke: construct, train on E/F labels, evaluate and prepare a native model; observe a real accepted update and changed predictions. Exercise one-, two- and three-layer E/F, not merely constructors or nonempty outputs.

## P7. Unified prediction, stress, training and periodic MD

### Files and changes

- `phydrax/atomistic/_prediction.py`: preserve the single `energy_and_forces` public entry point while completing its prepared graph/cell support and optional requested stress. Extend prediction/provenance with lawful stress shape/convention/availability and image/schedule/replay evidence. Unsupported requested stress refuses; no zero matrix labeled valid. A finite call without requested stress remains simple and unchanged semantically.
- `phydrax/atomistic/_potential_program.py`: context binds the actual fixed image topology and runtime cell vectors. Replace blanket directed-graph cell refusal only for terms with real cell-derivative capability. Preserve classical/reciprocal/bonded/site requirements and mixed-term admission. Evaluate shared energy and derivative work without independent duplicate linearization at the same point.
- `phydrax/atomistic/_stress.py`: retain named homogeneous Cartesian strain at fixed fractional coordinates and fixed n. For row H and column deformation F, verify H @ F.T against independently deformed Cartesian positions. Compute stress from strain gradient/volume under one named tensile/virial convention. Derive periodic virial from the same derivative, including image terms; finite moment diagnostics retain their distinct meaning. Migrate `_driven_stress.py`, `_crystal_elasticity.py` and controlled-free-energy consumers only for actual new capability/result assumptions, not an invented strain-orientation correction.
- `phydrax/atomistic/_training.py`: extend the existing full-batch/training-kernel owner to prepared finite/periodic sparse graphs and optional stress labels/masks/weights/scales. Preserve train-only normalization, accepted-update counters, stable RNG addressing, validation/continuation and explicit overflow termination. Prepared training structure does not hash traced updated parameters. Force/stress labels differentiate the scalar over fixed topology; graph rebuild is an outer data/preparation event.
- `phydrax/atomistic/_active_learning.py`: retain existing frame cell/stress labels when lowering into the training problem; campaigns no longer discard stress. Capability, scale and prepared graph identity enter continuation/promotion evidence.
- `phydrax/nn/atomistic/_painn.py`, `_nequip.py`: use capability-based validated periodic/image inputs through the shared graph, not simply delete finite-model rejection. Any newly admitted periodic/stress support is tested; historical minimum-image program evaluation remains valid.
- `phydrax/atomistic/_dynamics.py`, `_barostat.py`, `_hybrid.py`, `_alchemical.py`, `_rerun.py`, `_rollout.py`, `_crystal_elasticity.py`: migrate only actual affected learned geometry/cell/result assumptions found by LSP. Existing thermostat/integrator/alchemical mathematics stays canonical. Enable dynamic-cell learned methods only after their actual algorithm and image certificates are qualified; fixed-cell NVE/NVT and strain evaluation alone do not establish NPT support.

### Stress and derivative edge cases

Self images: coordinate cotangents cancel at the same stable atom, but cell cotangents remain. Triclinic off-diagonal strain and rotated cells must agree with an independent affine-deformation energy derivative. Virial/stress sign, units, volume and Voigt ordering are boundary metadata, not inferred from equal shape. Partial PBC stress uses a declared embedding/cell-volume meaning; if a physical volume is absent, refuse a volume-normalized bulk stress rather than invent thickness.

Force-loss training needs mixed coordinate/parameter derivatives; stress-loss training needs mixed strain/parameter derivatives; phonon/elastic response needs admitted continuity and HVP. Discrete branch changes remain outer epochs. Failed support/overflow state produces an invalid derivative, not a successful zero gradient.

### Tests and smoke

- Extend `tests/unit/atomistic/test_dynamics_periodic.py`, `test_controlled_free_energy.py`, `test_crystal_elasticity.py`, `test_types_and_graph.py` and active-learning campaign integration.
- **new** `tests/unit/atomistic/test_mace_stress.py`: diagonal/shear independent finite differences, row-cell convention, rotation covariance, primitive/supercell parity, zero-force/nonzero-stress self-image, partial-PBC refusal/admission, masks and image overflow.
- **new** `tests/integration/atomistic/test_periodic_mace_workflow.py`: E/F/S plus force/stress fitting, fixed-cell NVE/NVT, Verlet face crossing, cell-deformation certificate and exact rejected-state rollback.

Smoke actual periodic MD with multiple neighbor rebuilds, force/stress-supervised accepted update and a failed graph-capacity attempt. Check physical E/F/S and state transitions; finite arrays alone are not proof of conservativity, stress accuracy or stable MD.

## P8. Safe source conversion and native model artifacts

### Files and changes

- **new** `phydrax/atomistic/interchange/_mace_checkpoint.py`: explicit host source converter. Inspect admitted external model classes/configuration, tensors, U basis, species table, heads, radial family, cutoff and numeric normalization. Preserve original W and provider U or an audited exact sparse transform; no guessed tensor transpose or dimension-based identification.
- **new** `phydrax/atomistic/interchange/_mace_worker.py`: optional pinned provider worker, reusing `_external_runtime.PinnedExecutable`/`run_pinned_command` for bounded process lifetime/output and pinned environment. A subprocess is not a security sandbox.
- `phydrax/atomistic/interchange/_core.py`, `__init__.py`: reuse AdapterReport/loss/provenance patterns; publish one converter, not competing aliases. Import without provider packages remains available for native artifact inference.
- **new** `phydrax/atomistic/_model_artifact.py`: atomistic architecture recipe and pickle-free arrays plus source/projection/binding metadata. Reuse `phydrax/_model/_structure.py`, `_array_archive.py`, `_artifact_security.py` and registered reconstruction ownership; do not force `AbstractAtomisticPotential` through `AbstractArrayModel`.
- `phydrax/atomistic/_checkpoint.py`: preserve runtime-state versus model-artifact distinction. Bind checkpoints to a matching artifact/numeric revision/graph preparation. A portable bundle carries the native model artifact plus dynamics/training recipe and complete state; a dynamics-only checkpoint still requires its explicitly matching prepared model.
- `phydrax/atomistic/__init__.py`, `pyproject.toml`: public save/load atomistic-model boundary and optional converter-only dependencies with verified installed provider releases. Do not pretend a published full torch model is a safe weights-only file.

### Trust and rights

Default native loading is bounded and pickle-free: exact member/byte/shape/dtype/field inventory, no arbitrary import/object construction, registered exact type/field reconstruction and one full `phydrax.typing.validate` after constructor bypass. Verify digests and scientific identities, not merely archive readability.

Structural typing alone is not scientific/domain validation. After registered reconstruction, invoke the same owning validators used by construction/preparation for positive finite cutoff/scales, exact species/head maps, admitted selectors, coefficient/path/basis consistency, parameter finiteness, W-to-U shape and parameter-space constraints, table interval/continuity/boundary policy, and source numeric-revision binding. Recompute cheap invariant evidence and require trusted bound evidence where verification is not cheap; serialized success flags do not authenticate themselves. A malformed object with structurally valid annotations must still refuse. Do not add a second local validator or widen types to avoid these checks; expose/reuse the real constructor owner invariants. Add corruption cases that preserve shape/dtype and recalculate ordinary payload digests but violate domain/scientific meaning.

External safe state-dict forms require an admitted architecture declaration and exact tensor mapping. Published full-object torch checkpoints may need executable deserialization: only an explicit trusted-source capability permits that path, before loading, with source digest/provider environment and rights recorded. Refuse absent trust; never fall back to unsafe deserialization after a safe loader fails. Do not advertise pinned process isolation as protection from malicious pickle. Conversion outputs contain no executable provider payload.

Large U/coupling extraction has explicit maximum tensor/work/archive bounds. Unknown/custom modules, missing source fields, unsupported postprocessing, inconsistent head/species layouts and unsafe untrusted formats fail before publication. Atomic publication uses existing archive/lifecycle semantics. Models and provider code have independent licenses; no external weights are bundled without verified redistribution permission.

### Fidelity and parameter-space preservation

For every required external row, compare source and native exact execution on identical finite/periodic configurations: total/per-atom energy, forces, stress, direction/basis normalization, species ordering, selected heads and residual/density semantics. Also compare source parameter directions where the provider contract is trainable. A merged-polynomial inference artifact does not establish source parameter-gradient parity.

Freeze campaign tolerances before running. The pinned reference manifest's tolerances are external evidence inputs, not a universal native guarantee; declare native per-dtype/per-property mixed absolute-relative gates and downstream physical tolerances beforehand. A failed row remains failed with retained diagnostics; never loosen gates just to pass. Source fixture provenance, checkpoint hashes, provider releases, coordinate/cell identities and reduction/spline policies are recorded.

### Tests and smoke

- **new** `tests/unit/atomistic/test_model_artifact.py`: reconstruction, wrong species/units/parameter revision, stale projection, corruption, wrong registered type/fields, oversized arrays/member count and atomic failed publish.
- **new** `tests/interchange/test_mace_checkpoint.py`: admitted source layouts/head/density variants, unsafe-source refusal, tensor/U capacity refusal, original W parameter-space preservation and independently sourced parity. Provider absence skips only at its canonical boundary; installed provider failures do not skip.
- **new** `tests/integration/atomistic/test_mace_restart.py`: fresh subprocess loads only declared model/state data, resumes MD and training, compares to uninterrupted accepted evolution, rejects altered model/source/graph/owner bindings.

Smoke actual trusted source conversion and native E/F/S, then load/evaluate after removing provider dependency from the execution environment. Native exact and table artifacts remain distinguishable and independently reconstructible.

## P9. Owner-local multilayer distributed inference and migration

### Files and changes

- `phydrax/discretization/spatial/_distributed_relations.py`: reuse existing owner layouts, complete relation admission, deduplicated bounded halo columns, forward gather and exact reverse return. Bind per-layer payload shape/dtype to the prepared plan, not a new all-to-all implementation.
- `phydrax/discretization/particle/_distributed.py`: extend actual geometry ownership to the admitted periodic/triclinic cell contract; do not represent oblique cells as diagonal ParticleBox slabs. Reuse spatial owner epochs and expose a declared fractional owner partition where required. Include periodic image aliases without inventing new physical atom identities.
- `phydrax/atomistic/_distributed.py`: add true learned owner-compute execution. Keep the old global-evaluate/mask route labeled as a reference, not an optimized local evaluator. Owners compute their receivers with source halo features; exchange boundary states at each layer; reverse layer cotangents in reverse order; return coordinate/cell contributions exactly once. Include owner energy/per-atom outputs, global reduction order, finite/capacity/collective status.
- **new** `phydrax/atomistic/_feature_execution.py`: only domain-specific layer schedule/capability binding over the canonical halo owner. No duplicate generic collective packet or halo API.
- `phydrax/nn/atomistic/_mace_prepare.py`: apply the same owner-local layer kernels as single-device execution, with owned receiver masks and halo source tables. Intermediate feature halos remain at r_c plus skin rather than a silently widened interaction radius.
- `phydrax/lifecycle` and atomistic migration/checkpoint owners: carry stable IDs, image/candidate schedules, ownership epoch, model binding, positions/images, velocities, thermostat/RNG/constraints/bias histories and accepted force-cache revisions through migration. Reprepare feature halos once on accepted epoch change. Collective-mode migration is not declared supported until complete payload exchange/rollback exists.

### Acceptance

The local-reference simulator executes the same partition-local graph and exchanges as real collective execution; it must not evaluate the global model and then mask results. Compare it to independent single-device global MACE. Test a two/three-layer path whose dependency crosses owners at successive layers, so missing feature exchange or double reverse return changes forces. Node features are deduplicated by stable owner; multiple image edges still contribute multiplicity.

Reference-lane gather and reverse-return mechanics also remain in the canonical halo owner; the atomistic layer scheduler supplies payloads and layer ordering only. Independently test the shared-cell cotangent/global stress reduction (including nonzero self-image stress with cancelling position forces) so exactly-once positional force return cannot mask missing or doubled cell contributions.

Forward and reverse missing-owner, incomplete-route, payload-width, halo-message byte and migration capacity failures propagate to the consumer and preserve accepted state. Distributed derivative qualification is separate from inference qualification. Actual accelerator/multihost evidence is required before claiming those tuples; simulated owner lanes are a numerical oracle only.

### Tests, smoke, benchmark

Extend `tests/unit/discretization/spatial/test_distributed_relations.py`, `test_distributed_plane.py`, `tests/unit/atomistic/test_distributed_atomistic_completion.py`. Add **new** `tests/unit/atomistic/test_distributed_mace.py` and **new** `tests/integration/atomistic/test_distributed_mace_workflow.py` for multi-hop halo/state/adjoint, repeated image sources, migration commit/rollback and force-training mixed derivatives on admitted routes.

Smoke real collective execution on at least two devices/ranks with moving atoms crossing ownership boundaries, graph rebuild, feature exchange and reverse force return. Benchmark fixed total N and fixed N per device separately; report actual edge/halo/rebuild/message counts and communication cost, not inferred scaling efficiency.

## P10. Real architecture-specialized accelerated execution

### Scope decision

Implement a real JAX-integrated accelerated route for the admitted MACE sparse tensor-product hot path, not a selector that calls ordinary JAX under an accelerated name. Pallas is the preferred fit with current native spatial use. This is an actual planned deliverable; unavailable hardware is a qualification blocker, not permission to deliver a stub. Unsupported targets/architectures/derivatives refuse explicitly. The ordinary-JAX route remains the canonical portable reference.

### Files and changes

- **new** `phydrax/backends/atomistic.py`: backend capability/probe/admission owner for the actual accelerated problem kinds, precision, architecture, tile/working-set and derivative signatures. Do not overload spatial distance/P2P capability IDs or assume a `triton` selector proves an implementation.
- **new** `phydrax/nn/atomistic/_mace_kernels.py`: fixed model-structural coupling rows, widths and layouts lower into real receiver-owned forward, source-owned reverse and edge-geometry cotangent kernels. Learned weights, model-conditioned numerical tables and geometry are explicit array operands. Structural coefficient specialization uses audited immutable literals only where they really are mathematical structure; no array-bearing callbacks hidden static.
- **new** `phydrax/nn/atomistic/_mace_kernel_derivatives.py`: complete admitted JVP/transpose/higher-order transformations for the accelerated operation. Verify coordinate HVP and mixed force/stress-loss parameter gradients against ordinary JAX. No custom_vjp-only first-order claim.
- `phydrax/_execution_plan.py`, `_execution_resources.py`, `_identity.py`: bind actual admitted backend/target/runtime/build identity and measured compiler/workspace facts. Use current callable transparency/opacity rules. Do not add a custom persistent compiler cache manager; deployment prepares stable callable signatures and may use JAX's environment-configured cache.

### Optimization gate

Measure channel/receiver/edge tile sizes, occupancy/register/shared-memory use when available, code growth and arithmetic order. Preserve fully connected multiplicity when declared; low-rank radial factorization may reorganize computation but may not truncate a trained rank or path. Dense projection stays once per receiver after complete aggregation. Recompute local factors in reverse without broad graph-wide edge/path storage.

Compare accelerated and pure-JAX streamed routes at identical model, precision, topology, requested properties, derivatives and rebuild policy. Optimize only measured bottlenecks. Admission requires correct/failure/derivative evidence plus a documented useful performance or memory benefit in its exact envelope; slower experimental tuples remain labeled research or are removed, never marketed as faster.

### Tests and smoke

- **new** `tests/unit/backends/test_atomistic_backend.py`: real unsupported-target refusal, precision/working-set/resource refusal and identity invalidation.
- **new** `tests/unit/atomistic/test_mace_acceleration.py`: all legal path families, residual/readout interaction, duplicate/empty rows, masks, dtype, gradient, mixed derivative and source/geometry ownership against native reference.
- **new** `benchmarks/mace_execution_scaling.py`: lowering/compile/warm E/F/S and force-training timings, compiler buffer/code bytes, retained bytes and runtime peaks across N, degree, channels, l/correlation, species and tile capacities. Include exact versus tabulated preparation costs and failure boundaries.

Smoke actual accelerated E/F/S, real force-loss gradient, a parameter update and subsequent prediction with the same executable signature but new numeric revision. Demonstrate the kernel is the selected executed route through retained runtime/backend evidence, not import spelling or source-text assertions.

## P11. Deployment, operational continuation and user workflows

### Files and changes

- **new** `phydrax/atomistic/interchange/_ase_calculator.py`: optional ASE Calculator over one loaded native model/artifact, prepared candidate graph/Verlet state, explicit Å/eV mapping, energy/free_energy/per-atom energy/force/stress property semantics and provenance. ASE caching must invalidate on coordinates, cell, PBC, species, model revision or preparation changes. Unsupported stress/property requests raise the contract error; no valid-looking zero.
- `phydrax/atomistic/interchange/_ase.py`: reuse detached structure identity/loss conventions; do not change `from_ase_atoms` into a model loader.
- `phydrax/atomistic/interchange/_ipi.py`, `phydrax/atomistic/_hybrid.py`, `_born_oppenheimer.py`: implement a real native provider adapter over the model/program without a second energy/force loop. `ExternalAtomisticEvaluation` is owned by `_hybrid.py`, not `_born_oppenheimer.py`. Current `_ipi.py:344-347` stores a received wire virial directly in its `stress` field, while `:374` transmits that stress directly as virial or substitutes zero. Establish one explicit native tensile-stress versus configurational-virial contract and convert at the wire boundary, including volume/sign, atomic-unit versus system-unit scaling, cell/inverse-cell orientation and tensor ordering. Reuse native unit/scale owners. Refuse unavailable required virial; preserve socket framing/endian/partial-read bounds, not the existing dimensional ambiguity. LSP-migrate affected `_active_learning.py`, `chemistry/_atomistic.py`, hybrid/provider tests and `examples/atomistic_ipi.py` together; do not relabel every external tensor without auditing its actual provider semantics.
- **new** `phydrax/export/_atomistic.py`: fixed-capacity E/F/S inference ABI using existing native inference/export owners. Inputs explicitly carry coordinates, cell, species, endpoints/images/masks or a declared host-prepared graph boundary; statuses/active counts are outputs. Exporting model evaluation does not export neighbor lifecycle or training.
- `phydrax/export/_iree.py`, `export/__init__.py`: reuse current artifact publication, digest-pinned load, compiler/runtime identity and native parity. Loaded host executables remain nondifferentiable. Accelerated Pallas operations may not lower through IREE: declare export's real native-JAX route and qualify it, never substitute an undeclared fallback. ONNX is not a second required deployment route here.
- `phydrax/atomistic/_checkpoint.py`, model artifact and lifecycle owners: portable model-plus-state reconstruction includes complete preparation recipe, model numeric revision, species/head/radial realization, graph/owner/replay epochs and transient histories. Runtime compiled cache is disposable, not required for restart correctness.
- **new** `examples/atomistic_mace.py`, `examples/periodic_mace_dynamics.py`, `examples/mace_checkpoint_conversion.py`, `examples/atomistic_mace_deployment.py`: real native construction/training, periodic E/F/S+MD, trusted conversion and ASE/i-PI/export workflows. No network download/import on package import; external artifacts supplied explicitly.

### Operational tests and smoke

- Extend `tests/unit/atomistic/test_ase_interchange.py`, `test_ecosystem_advanced.py` for actual calculator/property/virial behavior; new integration model-restart/export files as warranted by independent ownership.
- **new** `tests/integration/atomistic/test_mace_deployment.py`: loaded native artifact through ASE and i-PI returns physical E/F/S consistent with native reference, changes cell/species/model revision and rebuilds safely, rejects missing virial, and resumes in a fresh process.
- Existing export contract tests plus **new** `tests/unit/export/test_atomistic_export.py`: fixed ABI/property/status parity, digest/dtype/shape refusal and explicit no-grad behavior.
- Exercise ASE optimization and moving-cell/property calls, i-PI loopback with actual native inference, exported frozen E/F/S and fresh-process MD/training continuation. A mock echo or archive member-count test is not workflow proof.
- Independent raw-wire i-PI controls use an independently specified skew cell and known length/energy/force/tensor conversions, plus an energy with known nonzero strain derivative. Inspect actual transmitted/received bytes and compare the required virial sign/volume/unit transformation and cell/inverse orientation against that independent physical/protocol oracle. Same-code loopback alone cannot detect a shared stress-as-virial or unit error. Include partial reads and unavailable required virial; do not treat an echo of our own wrong bytes as parity.

## P12. Qualification, documents, generated surfaces and completion

### Consumer-visible regression matrix

| Boundary | Required controls |
|---|---|
| Sparse schedules | stable ordering, duplicates, high-degree fragmentation, empty rows, masked invalid inputs, seed/correction continuity |
| Derivatives | JVP/VJP duality, finite differences, mixed coordinate/parameter and strain/parameter gradients, coordinate HVP, failure-invalid derivatives |
| O(3) | rotation and reflection, path normalization, real basis mapping, low-degree dot/cross/STF preservation, arbitrary-degree/resource refusal |
| Symmetric product | repeated-factor multiplicity, zero-input derivatives, fixed U/W parameter space, sparse/dense tiny oracle, bound-before-allocation |
| Radial | exact source family, slope construction, knot/cutoff jets, bounds, conservative table derivatives, distinct approximation identity |
| Geometry | all image routes, self images, primitive/supercell parity, skew/partial PBC, route reversal/wrap/cell certificates |
| Model | E0/scale/head precedence, residual/density updates, multiple interactions, species IDs rather than string/dimension coincidence |
| Runtime | neighbor reuse/rebuild, failed attempt rollback, capacity replacement, stable RNG, stale model/graph/artifact refusal |
| Import/security | explicit source trust, no unsafe fallback, provider/output bounds, exact tensors/U basis, native no-provider use, restore validation |
| Distribution | real owner-local execution, successive-layer halos, exactly-once reverse force/cell return, missing-owner refusal, migration restart |
| Deployment | correct units/sign/Voigt/virial mapping, requested-property refusal, native/frozen parity and real cache invalidation |
| Performance | full transformed route memory, capacity scaling, prepare/cold/warm separation, matched rebuild policy, actual backend/provider |

Permanent tests cover consumer-visible invariants and genuine edge matrices, deterministic and isolated. No source-text/import-spelling/wiring/default-copy tests, giant scenario loops, or permanent performance thresholds tied to one device. Tiny independent references are allowed as explicit bounded results; package runtime stays matrix-free/sparse. Keep plausible failing-before/passing-after regressions for bugs found during implementation, including missing i-PI virial if confirmed by independent controls; independent strain/shear controls verify the retained map rather than assuming it is wrong. Do not weaken tests to preserve a wrong result.

### Minimal conservative execution policy

During implementation only, select tests from actual changed symbols/consumers since the last merge into dev. Run affected ordinary tests with `-n auto`; use dedicated invocations for provider/device topology/accelerator isolation. Full suite is required only if collection/shared-fixture/architecture changes justify it. Shared sparse/O(3) changes need all affected consumers, including meshfree laws; whole particle/contact/application suites are not automatically relevant.

Planned integrated gates:

- `python tools/check_typing.py check` — zero diagnostics over all first-party roots.
- `python tools/audit_contract_candidates.py --signatures` and `python tools/audit_selectors.py` — no redundant effective checked guards or duplicate selector conventions.
- Configured Ruff formatter/linter over touched code, not a competing style.
- Runtime-resolvable strict declarations; static typecheck fixtures for new public layouts/model/graph/import/export boundaries. Installed-wheel typing checks where applicable to public annotation changes.
- Targeted pytest node selection and real smoke programs covering every changed runtime boundary.
- Public API and capability generators/checkers after facade/profile changes.

No gate above has been run for this plan, and a plan-file change does not justify running package checks.

### Benchmark campaign and memory accounting

Separate: source/model conversion; coefficient/product-graph preparation; radial projection; neighbor build; grouping/tile schedule; lowering; compilation; first invocation; warmed E/F/S; transformed force/stress training; neighbor rebuild; feature communication; deployment result assembly. Synchronize only declared benchmark boundaries.

Vary N and E independently through density/degree; also degree skew, l/correlation/channel width, species count, tile/channel caps, precision, interaction depth and ownership. Benchmark original native consumers before cutover and reference/streamed/accelerated counterparts after. Use a same-checkpoint/source oracle for MACE, not changed widths or discarded stress. Exact and tabulated realizations are separate records.

Retain compiler argument/output/alias/temp/code bytes for every transformed route; logical retained bytes for topology/images/node states/tables/artifacts; bounded halo/parameter/replay bytes; sampled allocator/RSS peaks labeled observations; environment/build/device/provider/topology identities and device headroom. Code size/compile cost count against optimization success. No blanket linear-complexity guarantee: search/order/basis construction and capacity-dependent work are measured independently.

Capacity campaign brackets last successful and first refused/failed configuration with identical protocol in fresh processes. Distinguish one evaluation, repeated static calls, moving NVE/NVT and sustained MD with rebuilds. Distributed strong/weak scaling includes communication and moving-neighbor lifecycle. The paper's 11.24M atoms, 3.1–5.0x and 93.8% numbers remain external reported results, not acceptance targets or promises.

### Scientific and external prerequisites

Pinned source/checkpoints/provider releases with rights; independent E/F/S and parameter-direction references; finite crystals, primitive cells and skew/partial-PBC structures; downstream NVE/cutoff/diatomic/phonon controls; real accelerated and multi-device/multihost hardware; exact artifact/export runtime versions; independent release review/signatures. Record unavailable prerequisites precisely after finishing all reachable implementation. Synthetic fitting and invariance tests verify algorithms, not universal material accuracy or trusted release.

### Documentation and generated files

Update after smoke proof, not before implementation claims:

- `docs/guides_atomistic.md`: canonical model/support/API and native training.
- `docs/guides_atomistic_dynamics.md`, `guides_atomistic_distributed_execution.md`: image/cell/neighbor/owner lifecycle and exact unsupported methods.
- `docs/guides_atomistic_interop.md`, `guides_atomistic_active_learning.md`: source trust, model artifacts, E/F/S labeling, ASE/i-PI and provider-only boundaries.
- `docs/guides_sparse_spatial_hierarchies.md`: streamed relation preparation, event ordering, derivative/resource evidence and requested edge-output limits.
- `docs/api/atomistic.md`, `atomistic_advanced.md`, `docs/api/nn/architectures.md`, `docs/api/execution.md`: truthful public surfaces, no duplicate canonical owner.
- `docs/cookbook/atomistic_dynamics.md`, `atomistic_interop.md`: real runnable workflows.
- **new** `docs/guides_mace_execution.md`: source architecture/basis fidelity, exact/table/folded routes, capacity objectives, derivative continuity, hardware and rights boundaries.
- `mkdocs.yml`, `README.md`, `CHANGELOG.md`: navigation and exact support changes; invalidate superseded claims and identities explicitly.
- `docs/data/public_api.json` and canonical capability/source/closure data: regenerate using `tools/generate_public_api_manifest.py` and `tools/generate_capability_inventory.py`; run `check_public_api_manifest.py`/`check_capability_consistency.py`. Do not hand-edit generated JSON or add schema generations.
- `phydrax/qualification/_builtin_catalog.py` and source/candidate providers: source ledger and support profiles generated through canonical authority. Numerical artifact creation never self-promotes a candidate.

### Deletion and cleanup

After integrated smoke evidence, remove superseded direct model scatters, duplicate edge-preparation helpers, old stress/periodic refusals now covered by real capabilities, stale provider/model examples and source-text/incidental tests. Retain legitimate unsupported-boundary tests; replace only those whose public capability intentionally changed. Delete temporary scripts/scaffolds. Do not delete unrelated code or broaden into contact/DEM/SPH refactors.

## 5. Final end-to-end acceptance and delivery

Completion requires all of the following, not a model constructor or plausible compiled scaffold:

1. Native model construction, exact E/F/S, E/F/stress training and a real accepted parameter update using shared training/runtime owners.
2. Periodic primitive-cell/supercell and self-image semantics, triclinic strain stress and smooth derivative routes with independent physical controls.
3. Shared streamed execution used by native MACE, PaiNN/NequIP and the selected graph/meshfree consumers, with bounded transformed memory, seeded ordering and explicit failures.
4. Every required source checkpoint family has an actual admitted conversion/fidelity result or a precise external qualification blocker; original parameter-space identity is retained.
5. Native artifacts load without external model providers and reconstruct complete matching model/state/preparation in a fresh process; stale/corrupt/trust failures refuse.
6. Distributed execution is truly partition-local, exchanges intermediate features and reverse cotangents, and supports complete accepted migration/continuation on declared tuples.
7. The accelerated selector runs real kernels, has complete admitted derivative behavior, reports target/provider evidence, and never hides an ordinary-JAX substitute.
8. ASE/i-PI and frozen-export workflows provide actual E/F/S with correct boundary conventions and property refusal, not mock forwarding.
9. Targeted tests, real runtime smokes and phase-separated scaling are observed and retained; docs/facades/manifests/callers are migrated with obsolete paths removed.
10. Delivery names implementation worktree/branch, actual exercised commands/artifacts, intentional identity/numerical changes, exact support profiles, and remaining external qualification/release blockers. No unobserved speed/capacity/accuracy/release claim.

Recommended implementation order: shared streamed execution and transformations first; image topology and O(3) preparation in parallel once contracts are frozen; then exact trainable MACE, E/F/S integration, faithful source conversion/artifacts, owned distribution, accelerated specialization and deployment. These are dependencies inside the complete deliverable, not permission to stop after the generic substrate.

## 6. Design review disposition

Two independent source-grounded reviews covered architecture/compiler/derivative/distribution contracts and model/basis/radial/import/security/deployment contracts. Both reported no remaining material design blockers after amendment. This is plan review, not implementation or runtime qualification.

Resolved findings: valid-zero/cancellation derivatives in seeded reduction are a P0 prerequisite; one-interaction MACE has a separate invariant readout product; restored artifacts require owning scientific/domain validation beyond structural typing; image certificates cover absent/excluded translations; reverse scheduling accounts for wide epilogue state and degree-squared replay; combined edge/node losses retain all cotangents; halo references stay canonical and shared-cell cotangents are tested independently; i-PI has independent raw-wire unit/cell/virial controls.

Remaining external implementation/qualification prerequisites are explicitly retained: admitted provider releases and trusted/rightful source bytes; a pinned actual one-interaction source fixture (plus its E/F/S and original-W parameter-direction parity); accelerated and real distributed hardware; matched frozen-export runtime; and independent release evidence/authorization. None is asserted available or passed by this plan.
