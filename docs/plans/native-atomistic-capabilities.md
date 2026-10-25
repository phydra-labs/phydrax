# Native atomistic capabilities: full implementation and qualification plan

## 0. Authority, baseline, and completion boundary

This plan implements the user's native-only goal: Phydrax must cover the supported scientific and operational capabilities of both inspected ALCHEMI repositories, then establish stronger composability, derivative, failure, resource, and measured performance guarantees. An external evaluator, wrapper, converted call-through, or optional ALCHEMI backend does not satisfy a native implementation requirement.

- Inspected repository: `dev` at `c8ef7e8034eeb146a8c309543f54bea90b492150`, merge PR #425, `exact-gp-execution`. The tracked tree was clean when scoped. This was the latest merge into `dev` at planning time; implementation must refresh the comparison point and affected-test matrix after intervening merges.
- Toolkit source: [`f59e28dc6701583a13bb294e536076c2a7379e98`](https://github.com/NVIDIA/nvalchemi-toolkit/tree/f59e28dc6701583a13bb294e536076c2a7379e98).
- Operations source: [`10e12340c39fd12a2375d6301768e72d2cd12868`](https://github.com/NVIDIA/nvalchemi-toolkit-ops/tree/10e12340c39fd12a2375d6301768e72d2cd12868).
- FourierD3 formulation: [A fast summation method for the DFT-D3 dispersion correction](https://arxiv.org/html/2607.15103), inspected October 18 revision. Publication and source-revision identifiers are external provenance, not native schema generations.
- Evidence here is source/document/test-definition inspection and repository metadata only. No package code, tests, examples, builds, formatters, linters, or benchmarks were executed. A session interruption was recovered from saved worker transcripts; unfinished research was not treated as a completed implementation or test result.
- This Markdown plan is the only intended repository change. No worktree or implementation branch is created by `//plan`. A later `//code` starts a fresh worktree under `../phydra-labs/.worktrees/native-atomistic-capabilities/` from the then-current branch, with a neutral branch name and the authoritative ignored `AGENTS.md` copied before implementation. If this plan is not yet committed, carry its approved contents into that worktree too. All implementation work stays there. `//code--` is the explicit exception.
- Existing `docs/plans/streamed-atomistic-execution.md` and `docs/plans/gpu-spatial-search-execution.md` provide design history, not the current capability inventory. Native MACE, streamed relations, periodic image graphs, stress supervision, Pallas coupling, owner-local learned execution, and native artifacts now exist. Do not rebuild them as missing features or inherit an old plan's exclusions when they conflict with this expanded scope.
- Files labeled **NEW** below are proposed implementation files. They are not claimed to exist. New symbols are design choices, not shipped public APIs. Existing files were mapped from source; exact references must be refreshed before changing them.
- No language server was configured during familiarization. At implementation start, check again and use LSP references for every changed exported/cross-module symbol when available. While unavailable, use explicit import/caller searches, registries, persistence recipes, generated inventories, and consumer tests. The seed migration inventory below is not permission to omit later-discovered callers.

### Native-only and legal boundaries

Product inference, training, simulation, optimization, sampling, generation, and distributed numerical execution must use Phydrax-owned JAX/native/Pallas routes. Do not add an ALCHEMI dependency, call its Warp kernels, or run Torch models behind a native label. Existing host file-format libraries remain legitimate interchange dependencies; their use does not authorize an external numerical engine.

Separately pinned source environments may supply independent development/qualification oracles and explicit one-time admitted weight extraction. Converted artifacts contain native arrays and registered recipes only and run without the source framework. Source code, parameter tables, training data, and checkpoint weights have separate rights. No implicit downloads, unsafe-load fallback, unlicensed redistribution, or claim that a digest grants permission.

Arbitrary user-provided native models/samplers must have extension contracts comparable to the upstream bring-your-own-component workflow. Executing an arbitrary Python/Torch program is a different interoperability requirement and cannot be claimed by renaming a native callback. Each concrete upstream architecture/configuration admitted to the parity campaign needs an actual native implementation, not a distilled substitute.

### What counts as completion

Track these independently: implementation completeness; numerical qualification; derivative qualification; resource/performance qualification; hardware/topology qualification; scientific validation; operations/restart qualification; rights/security admission; signed release authorization. A planned or registered profile is not retained evidence. Follow `docs/CAPABILITY_LIFECYCLE.md`; new model architectures and the crystal domain need explicit closure exceptions naming their owner, support tuples, dependencies, evidence obligations, and nonclaims.

The scope is the pinned supported capability envelope, not every possible future upstream feature or every mathematically invalid Cartesian product of options. Build a source-derived operation matrix before implementation: model/configuration, physical method, geometry/PBC, requested properties, derivative coordinates/order, precision, layout/capacity, backend, distribution, and lifecycle. Unsupported source combinations must remain distinguishable from required-but-unimplemented native rows. Never silently narrow a required row to make a test pass.

"Better" has explicit gates: more scientifically valid native combinations; complete declared derivatives; no silent truncation or plausible failed outputs; bounded/reusable preparation; durable scientific continuation; and measured improvements on declared end-to-end workloads at matched physical definitions and accuracy. Do not promise that one route dominates every atom count, precision, device, or derivative order. Do not count a changed Hamiltonian, weaker tolerance, omitted neighbor work, or different requested outputs as a performance gain.

## 1. Complete scope ledger

| Requirement | Required end-to-end result | Plan owners |
|---|---|---|
| C01 | Explicit species/material IDs, units, conditioning, properties, model ports, parameter roles, and derivative requests | P0, P1, models |
| C02 | Native model extensibility plus source-faithful supported MACE families, actual AIMNet2, and actual UMA | Models, distributed |
| C03 | Lawfully admitted pretrained conversion, native artifacts, complete multi-model simulation/training restart | Models, learning, P12 |
| C04 | Native prepared neighbor methods, selected sources, dual cutoffs, periodic images, layouts, rebuilds, custom pair actions | Interactions, P1 |
| C05 | Native segmented reduction and spline/particle-grid operations with correct adjoints, masking, and bounds | Interactions |
| C06 | Point-charge Coulomb, DSF, Ewald, PME, and admitted slab semantics | Interactions, distributed |
| C07 | Public multipole/response capabilities with exact basis, self/background/exclusion, geometry, and derivative contracts | Interactions, models, distributed |
| C08 | Conventional admitted two-body D3(BJ) and explicitly distinct FourierD3 definition, preparation, E/F/S and requested derivatives | Interactions |
| C09 | Complete NVE/NVT/NPT/NPH, thermostat/barostat choices, constraints, learned dynamic-cell state and failure semantics | Dynamics |
| C10 | Device-resident batched FIRE2/L-BFGS coordinate/cell relaxation and grouped NEB/climbing-image paths | Dynamics |
| C11 | Heterogeneous bounded cohorts, inflight refill, stages, force reprime, whole groups, backpressure, exact publication and continuation | Dynamics, learning, distributed |
| C12 | True owner-local domain execution, layer halos and reverse cotangents, migration, reciprocal distribution | Distributed, interactions, models |
| C13 | Data-parallel training and distributed stage pipelines, including admitted combinations with domain execution | Distributed, learning |
| C14 | Streaming atomistic training: objectives/optimizers, E/F/S and additional properties, validation, EMA, freezing/adaptation, accumulation | Learning, models |
| C15 | Offline/on-policy distillation with native teachers, replay, embedding/HVP matching, and admitted distribution objectives | Learning, models, dynamics |
| C16 | Atom/edge-budget batching, weighted source mixing, Zarr/other storage boundaries, bounded prefetch/staging, reporting/profiling | Learning, dynamics |
| C17 | Metric-correct CVs, umbrella/walls, metadynamics/RMSD histories, ABF, dynamic-cell bias response | Sampling |
| C18 | Bias-aware replica exchange, walker/state/shared history ownership, thermodynamic rebinding and reweighting boundaries | Sampling, dynamics, distributed |
| C19 | Native conditional generation and resumable bounded streams over existing transport/sampling owners | Generation |
| C20 | Native crystallographic identity/ASU expansion, conformer packing/contact relief, deduplication, physical relaxation/ranking | Generation, dynamics |
| C21 | Consumer-visible failure, resource, provenance, RNG, observation, cancellation, and full restart guarantees across all families | Every owner, P12 |
| C22 | Complete caller/test/docs/example/public API/capability cutover and independent scientific/performance/release evidence | Every owner, P12 |

No row can be satisfied by an exported class, a source-text test, an unexecuted benchmark producer, an external numerical provider, or a different model with a similar name.

## 2. Current source facts that determine the design

| Current owner | Established behavior | Required consequence |
|---|---|---|
| `atomistic/_potential_program.py` | OR-merges geometry requirements and reduces cutoffs to one maximum; one physical neighborhood supplies pair and learned graph geometry | Bind independent scientific resources without weakening pair multiplicity or unique-image guards |
| `atomistic/_sites.py:575-579` | Existing `AtomisticNeighborhoodBundleState` contains two pair-only fields; it is not consumed by program/dynamics | Move and evolve this owner; do not create a second bundle vocabulary |
| `atomistic/_barostat.py`, `_dynamics.py` | Learned NPT is rejected; MC barostat requires dense pair authority; main integrator admission is Verlet or BAOAB | Implement complete ensemble methods and cell-aware continuation, not just delete a refusal |
| `chemistry/_optimization.py`, `chemistry/reaction/_path.py`, `chemistry/_atomistic.py` | Host-gradient molecular optimization and host FIRE/NEB exist; atomistic chemistry PES rejects periodic cells | Add native device/periodic execution while preserving the honest host-provider path |
| `_execution_pool.py`, `_execution_workset.py`, `_iteration.py`, lifecycle owners | Real refill, signature buckets, semantic keys, observation and revision-bound checkpoint mechanisms exist | Compose atomistic scientific state with these owners; no replacement scheduler/checkpoint framework |
| `nn/atomistic`, `_prediction.py` | PaiNN/NequIP/MACE and conservative E/F/S exist; standard charge/spin/task/embedding contracts and native AIMNet2/UMA are not established | Extend the canonical model contract and implement real new architectures |
| `atomistic/_training.py` | One-potential, full-batch Adam E/F/S frontend over shared training kernel | Expose reusable shared minibatch/EMA/selection/distribution capabilities, not clone operator training |
| `_training_kernel.py`, `_training.py`, `_training_checkpoint.py`, `_data_plane`, `nn/parameters`, operator training | Composite objectives, delayed/EMA targets, parameter subspaces, deterministic data epochs, bounded prefetch and richer training controls already exist | Reuse their ownership and preserve existing consumers |
| `atomistic/sampling`, `_thermodynamic.py` | CVs, basic biases, scalar ABF, replica exchange/SAMS exist; nonempty bias IDs are rejected during thermodynamic table preparation | Close typed bias cross-evaluation/history/rebinding and metric/measure semantics together |
| `applications/_coordinate_generation`, `atomistic/_diffusion.py`, transport | Real flow/diffusion and biological fixed-support generation exist; current composition rejects periodic cells | Generalize domain admission/streaming, not the numerical evolution stack |
| Chemistry/crystal mapping | No native crystallographic ASU/packing/equivalence workflow owner was found | Introduce one explicit domain composition, not an external packing/symmetry engine |
| `_force_field.py`, `_polarizable_force_field.py` | LJ/Tang–Toennies dispersion and PSD-only LJ-PME exist; D3 is a distinct missing method | Keep D3/Fourier scientific identity and signed factors separate from existing dispersion definitions |
| `backends/atomistic.py`, `_mace_kernel_derivatives.py` | Pallas execution has explicit architecture/precision/order admission | Qualify full mixed derivative requests; HVP training generally needs order three, not merely HVP inference |

## 3. Shared architecture and invariants

### 3.1 One owner per responsibility

- `phydrax.atomistic`: scientific system/condition/property contracts, energy programs, atomistic state transitions, model data and domain orchestration.
- `phydrax.nn.atomistic`: concrete scientific architectures and their exact/source/accelerated realizations.
- `phydrax.discretization.particle` and `.spatial`: periodic geometry, image/search completeness, candidate topology, spatial ownership and halo routes.
- `phydrax.sparse`: relations, stable route groups, bounded gather/reduce/streamed fragments and transposes; no scientific pair reinterpretation.
- Native interpolation/spectral owners: B-spline transfer, FFT/reciprocal execution and normalization; `phydrax.linalg` owns solves, factorizations, spectral/property evidence and reusable linearizations.
- `phydrax.optim` / `nonlinear`: optimizer/root algorithms and method evidence. Chemistry/atomistics bind physical coordinates, cell charts and convergence; they do not implement second private solvers.
- Shared model ports, derivative contracts, trainable roles, training kernel, data plane, execution, lifecycle, artifacts and qualification remain canonical. Extend a shared owner only for a genuine reused invariant.
- Chemistry/reaction and the selected crystal-domain owner own chemical/path/symmetry semantics. Flow/diffusion integration, rotations, periodic cells and packing primitives stay in their existing numerical owners.

### 3.2 Preparation, dynamic leaves, and identity

Scientific model definition, parameter revision, numerical realization, static executable signature, bound resource/topology/cell epoch, and accepted run/replica/state identity remain distinct. A compile-compatible shape does not establish chemical identity, source-weight fidelity, valid periodic coverage, or checkpoint compatibility.

Static values: immutable selectors, capacities, declared units/bases/axes/role identities, architecture choices, supported derivative envelope, communication layout and callable code identity. Numerical values remain dynamic leaves: parameters, fixed coefficient tables, cells, coordinates, charges, masks, routes, optimizer/history state and array-bearing callbacks. FIXED is a parameter-discovery role, not permission to hide arrays in static metadata or closures.

Constructors validate before assignment. New concrete modules inherit `StrictModule`, are final by default, and use runtime-resolvable native `phydrax.typing` contracts with `__strict_contract__ = True` when required. Use nominal dimensions and explicit scientific axis identity, not coincident sizes. Apply `checked` only after tracing callable/compiler identity consumers; it does not replace scientific/capability checks or complete restore validation. Closed selectors are declared once and parsed at their owner. No internal schema generations, compatibility aliases or silent reconstruction coercions.

The bundle/conditioning/state cutovers intentionally change PyTree and persistence structure and may change fingerprints and executable identities. Update registered recipes and every consumer in one cutover. Refuse old/stale artifacts under the new contract; do not preserve historical IDs by inventing payloads or keep both representations behind a union. Existing explicit lifecycle migration remains an owner, not an excuse for automatic compatibility.

### 3.3 Resource bundle: selected design

**NEW** `phydrax/atomistic/_neighborhood_bundle.py` is the canonical atomistic neighborhood requirement/preparation/lifecycle owner. Move `AtomisticNeighborhoodBundleState` out of `_sites.py` and evolve it here; delete the obsolete definition/import path. Add `AtomisticNeighborhoodBundlePlan` and `PreparedAtomisticNeighborhoodBundle` in this owner.

Each request binds scientific role, support/coordinate domain, periodic/image and pair-counting convention, cutoff/skin, declared selection policy, exclusions where applicable, precision/layout needs and explicit capacity. Requests are keyed by owning term/role identity, never display text or shape. Required roles cover classical MIC pairs, directed image graphs, explicitly counted image-pair sums, coordination neighborhoods and interaction-site pairs. A model-defined nearest-neighbor cap is scientific selection, not buffer overflow. Reciprocal plans remain term-owned and bind the same system/cell/epoch; do not put FFT/SCF/model state into a generic resource registry. The bundle can be prepared from explicit role requirements without attaching a potential, so crystal contacts and CVs reuse it without a fake energy term.

Preparation resolves compatible requests into certified source searches and fixed-capacity derived views. Reuse `PreparedSearchGroup`, native particle plans, `EdgeRelation`/`RowRelation` and `StreamedRelationPlan`. A shared source must cover the consumer's actual cutoff plus reuse margin and have compatible event semantics. Matrix/COO conversion, half/full counting and repeated images are explicit operations; no coalescing of distinct nonlinear events. Runtime cutoff masks use current geometry. Do not cache a geometry-filtered compact view across motion without a new certificate.

The bundle state retains every role's accepted search/cache state, build geometry and image frame, completeness certificate, topology epoch, required/available capacities and failure status. Candidate refresh is a pure function of accepted bundle, proposed positions/cell/image counts and explicit rebuild policy. Certificates cover both stored and absent images under displacement and cell deformation; wrapping updates integer shifts exactly. Keep capacity failure separate from scientific/domain/cell-conditioning failure. Capacity ladders may replay only the same physical attempt from accepted state and only for eligible capacity failures.

`PreparedAtomisticPotentialProgram` resolves each term to static resource slots during preparation, binds numerical geometry once, then evaluates the ordered energy/property dependencies. It must not replace global mutable neighbor attributes before each term. Pure graph-only prediction can keep its direct graph preparation boundary; do not force low-level geometry users through a redundant bundle wrapper.

`AtomisticDynamicsState` carries one bundle state instead of the legacy `neighborhood` plus `neighborhood_cache` alternatives. Force-cache validity binds geometry/cell, all required role epochs, Hamiltonian/model revisions, conditioning and bias/thermodynamic state. No cache can be reused merely because an unrelated role did not rebuild.

### 3.4 Model/property dependencies and derivatives

Retain `AbstractAtomisticPotential`. Extend its native conditioning and requested-observable contract through existing model ports and derivative contracts, not a second plugin language. A prepared term consumes a declared field identity and its derivative relationship; predicted charges, multipoles, embeddings and corrections cannot be wired solely by untyped string names.

For a conservative model/correction group, derive coordinate/cell/parameter response from the same scalar energy, including predicted-field response, polarization stationarity and D3 coordination/self terms. Clearly separate declared direct/nonconservative outputs and algorithmic force modifications such as NEB/ABF from physical energy gradients. They require their own work and sampling contracts.

Use native reusable linearizations for repeated JVP/VJP/HVP actions. Never materialize a full Hessian when only actions are needed. A requested dense Hessian is an explicit, pre-admitted scientific output with atom/case block axes and a byte bound. Qualify force/stress training and HVP distillation at their full mixed derivative order. A first-order force attachment or `custom_vjp` alone is not transformation closure. Discrete topology/ownership/rebuild/refill/acceptance/deposition decisions remain outside pathwise derivative claims.

Every failed admitted path preserves status/evidence and invalidates dependent numerical values and derivatives. Do not return a plausible zero or infer success from finiteness. Approximation choices, accumulated error bounds and unavailable evidence remain explicit.

### 3.5 Scientific transitions, scheduling and observations

Scientific owners expose their complete proposed/evaluated/accepted transition, convergence and continuation state. Worksets own signature grouping and bounded placement; lifecycle owns durable commits, lineage and rollback. Stage transitions specify which force/optimizer/thermostat/bias resources are invalidated or re-primed. Never transplant arbitrary mutable hooks that can change physics without entering identity, state and replay.

RNG addresses depend on stable experiment/task/replica/material identity and committed action counters, not lane, batch order, rank or refill schedule. Whole logical groups such as reaction paths move together. Output publication and its durable receipt precede refill; a failed sink cannot silently lose a completed system. Restarts retain queue/cursor/group membership, pending publication, state/parameter revisions and all continuation payload, not just coordinates.

## P0. Capability inventory, reference admission and necessary baselines

### Files and changes

| File | Planned change |
|---|---|
| **NEW** `phydrax/atomistic/_capability_closure.py` | Declare the operation-level C01–C22 closure/support matrix using existing `CapabilityClosureRequirement`, `CapabilityGapResolution`, `SupportTuple` and evidence vocabulary. Separate implemented/candidate/released and missing prerequisite rows. Do not create a second registry or mark source inspection as behavior evidence. |
| **NEW** `phydrax/atomistic/_profiles.py`; existing `_mace_profiles.py` | Own exact new model/interaction/runtime/workflow candidate tuples; retain and deliberately requalify affected MACE tuples rather than renaming them into an umbrella success. |
| `phydrax/qualification/_builtin_catalog.py`, `_builtin_closure.py`, `_builtin_sources.py` | Register domain profile/closure producers; add inspected source/paper/table records with actual archive/license hashes when obtained. Preserve `code_inspected` versus `behavior_inspected` and rights status. |
| `tools/mace_checkpoint_campaign.py`; model-specific campaign tools named in the model section | Pin lawful source architectures/configurations/weights and source runtime identities. Capture source references in isolated environments; no product-runtime dependency or silent catalog download. |
| Existing owning benchmark drivers and `benchmarks/_runtime.py`, `_memory.py` | Record necessary correctness/performance baselines before performance changes. Reuse common measurement/identity utilities; change them only for a demonstrated missing measurement invariant. |
| `NOTICE`, `LICENSES/`, external-artifact admission owners | Add notices only for actually reused material under verified rights; keep source code, model weights and D3/crystallographic tables separate. No invented redistribution permission. |

Build a named model/configuration and operation matrix from the pinned upstream public surfaces and working example scenarios, not a keyword count. Record required scalar/vector/tensor outputs and derivatives, applicable distribution modes, periodic conventions, precision, convergence semantics and memory bounds. Model wrapper acceptance may be broader than known checkpoint fixtures; distinguish source architecture discovery from numerical fidelity admission instead of silently calling all arbitrary modules supported.

Use the user's observations as the baseline if an actual failure is reported. Otherwise reproduce the narrow relevant defect before fixing it. Do not run broad baseline suites. For performance changes record equivalent numerical/status behavior and phase-separated baseline measurements under the same material environment. External source/weight/parameter data and accelerator/multi-host prerequisites remain explicit gates; finish reachable native work without substituting a fake provider or reducing scope.

## P1. Shared contract and representation cutover

### File-by-file changes

| File | Concrete change and migration |
|---|---|
| `phydrax/atomistic/_potential.py` | Replace the single merged cutoff/geometry-boolean assumption with typed per-role requirements; extend model capability/observable/conditioning and derivative admission using common contracts. Preserve scalar-energy ownership and explicit nonconservative capabilities. |
| **NEW** `phydrax/atomistic/_neighborhood_bundle.py` | Implement the selected requirement resolution, certified preparation sharing, pure candidate refresh, complete bundle state/evidence and role-specific failure/capacity accounting. |
| `phydrax/atomistic/_sites.py` | Remove the obsolete pair-only bundle definition; retain coordinate realization, site identity and exact force pullback. Route site searches through their own admitted bundle role rather than unbounded incidental dense pairs. |
| `phydrax/atomistic/_potential_program.py` | Bind static term resource slots and typed predicted-field dependencies; stop collapsing independent cutoffs. Construct one numerical context, evaluate one conservative energy, derive E/F/S and requested actions, retain per-term status and failure provenance. |
| `phydrax/atomistic/_types.py`, `_system.py`, `_graph.py`, `_prediction.py` | Bind explicit conditioning/observable axes, system/case identity and image conventions; preserve heterogeneous masking and native graph-only prediction. Reuse live JAX geometry, not detached foreign distances. |
| `phydrax/atomistic/_dynamics.py`, `_barostat.py`, `_thermal.py`, `_ensemble_advanced.py`, `_rollout.py` | Consume one accepted bundle state; update force validity and complete continuation fields. Introduce no compatibility union with old neighborhood/cache forms. Complete the methods in the dynamics section before removing their scientific refusals. |
| `phydrax/atomistic/_classical.py`, `_many_body.py`, `_electrostatics.py`, `_force_field.py`, `_polarization.py`, `_polarizable_force_field.py` | Migrate each term to its own prepared scientific role; retain exclusions, minimum-image versus explicit-image multiplicity, field/polarization status and unit conventions. |
| `phydrax/atomistic/_alchemical.py`, `_hybrid.py`, `_stress.py`, `_rerun.py`, `_committee.py` | Preserve controlled-Hamiltonian/force-group/scalar-blend semantics while binding independent resources and all relevant force-cache identities. RESPA and rerun cannot bypass bundle validity. |
| `phydrax/atomistic/_born_oppenheimer.py`, `_checkpoint.py`, `_model_artifact.py` | Migrate native deployment preparation and artifact recipes; serialize every referenced native model/condition/resource state with content identity. Remove MACE-only assumptions at generic artifact/restart boundaries without weakening family-specific source proofs. |
| `phydrax/export/_atomistic.py`, `_atomistic_export_worker.py`; `atomistic/interchange/_ase_calculator.py`, `_ipi.py`, `_trajectory_io.py` | Carry the canonical changed request/resource/status contracts. Frozen export remains epoch/capacity-bound unless its scope is explicitly extended and qualified; do not mislabel it as live neighbor discovery or trainable export. |
| `phydrax/atomistic/__init__.py`, `phydrax/nn/atomistic/__init__.py`, owning chemistry/execution facades | One canonical export per public symbol; update lazy resolution and remove superseded imports/aliases. |

### Required consumer cutover inventory

Beyond the direct owners above, inspect and migrate actual changed-contract consumers in `_active_learning.py`, `_crystal_elasticity.py`, `_driven_stress.py`, `_hydrodynamic_brownian.py`, `_kinetics.py`, `_nanoflow_observables.py`, `_observables.py`, `_observer.py`, `_reporter.py`, `_ring_polymer.py`, `_semigrand.py`, `_soft_matter.py`, `_stochastic_runtime.py`, `_topology_epoch.py`, `driven_flow/_sllod.py`, `free_energy/_switching.py`, and `sampling/{_bias,_free_energy_learning,_multistate,_multistate_checkpoint}.py`.

Chemistry consumers include `_atomistic.py`, `_optimization.py`, reaction path owners and `periodic/_lattice_force_constants.py`. Application consumers include protein cotranslation boundary/protocol and physical dynamics workflows, rotamer free-energy terms, and the shared biological generation interfaces. Do not edit unrelated same-named dynamics modules merely because a textual import search matched them. Refresh actual references, constructors, serialized field inventories and public facade users before cutover.

### Acceptance and focused proof

- Replace the old mixed learned/classical rejection scenario in `tests/unit/atomistic/test_program_stress.py` with positive admission of independently valid pair and image resources. Retain negative tests for an invalid classical image relation and expired current-cell unique-image certificates.
- **NEW** `tests/unit/atomistic/test_neighborhood_bundle.py`: independent role rebuilds, compatible-source sharing, incompatible pair/image refusal, exact wrapping, deformed-cell absent-image coverage, selected sources, role-specific overflow, candidate rollback, capacity-only replay and stale force refusal. Assertions concern returned geometry, status and accepted state, not private wiring or source strings.
- Update existing graph/image/Verlet, E/F/S, dynamics, hybrid/alchemical, artifact/restart and consumer tests. Add a small independent lattice oracle with repeated images and a self-image whose positional force is zero but stress is not.
- Demonstrate a real native mixed Hamiltonian evaluation and one accepted/rejected dynamics attempt, then serialize/restore/replay the same complete state in a fresh process. Numerical masks/IDs/tables remain dynamic leaves, and parameter discovery excludes conditioned state without making it static.
- Public field/annotation/recipe changes require declaration, role, typing, identity/fingerprint and constructor-bypassing restoration coverage. Record intentional representation changes and invalidated evidence; never re-pin incidental historical hashes as a substitute for correct semantics.

## P2. Native neighborhood, segmented-reduction and spline execution

### Scientific scope and selected design

The required surface is prepared scalar/tiled reference search, cell-list and explicit cluster routes; single/heterogeneous cases; selected receiver rows; full/half and fixed-capacity matrix/COO/segmented layouts; multiple cutoff tiers; per-case reuse/rebuild; differentiable pair actions; segmented sum/dot/multiply/mean/RMS/matvec; and particle-grid spline spread/gather/gradient/channel operations. Reuse current image searches, Morton/BVH preparation, sparse schedules and interpolation rather than porting Warp wrappers.

Two distinctions are mandatory:

1. A **scientifically declared graph selection** (for example UMA's nearest-neighbor cap and degeneracy band) is part of the model definition. It is not a capacity failure. A buffer that cannot hold the scientifically selected result is a separate failure. Default unrestricted-radius graphs still never truncate silently.
2. A **classical explicit-image pair sum** is a separate relation from a minimum-image pair relation. Add an `image-pair-sum` requirement with an exact reversal/half-view rule; it may cover radii larger than the unique-image radius without weakening the MIC contract. For symmetric pair energies, a complete directed sum has twice the canonical half-sum, including paired nonzero self-images. Do not apply that identity to arbitrary directed message functions.

Receiver selection uses stable material IDs and a prepared compact-row mapping. Radius tiers share one certified outer source but retain separate active masks, counts, output capacities and failure evidence. Per-case rebuilds update only the requested initialized segments; retained topology still binds current geometry and current cell/image certificates. An explicit image shift is part of route identity, not an attribute that may be coalesced away.

### File-by-file changes

| File | Planned implementation |
|---|---|
| `phydrax/discretization/particle/_image_neighborhood.py` | Extend `AbstractImageRouteSearch` and the existing dense/cell implementations with prepared receiver-subset mapping, per-case segment budgets and selective rebuild. Preserve separate cell/edge/degree/image evidence. Add declared learned-graph selection with canonical stable-ID/image tie handling; do not use capacity clipping to implement it. |
| `phydrax/discretization/particle/_image_relation.py` | Add exact canonical half views and reversal maps using material endpoints and integer image sign; retain self-image multiplicity and per-case segment ownership. Half views expose their counting convention. |
| **NEW** `phydrax/discretization/particle/_image_radius_tiers.py` | `ImageRadiusTierPlan` and prepared tier views: one outer search, ordered radii, independent tier capacities, explicit physical-cutoff versus candidate-skin masks and completeness evidence. |
| **NEW** `phydrax/discretization/particle/_cluster_image_search.py` | `ClusterTileImageRouteSearch` under the existing image-search interface. Reuse Morton ordering; construct bounded fixed-width groups, image-aware bounding volumes, candidate group pairs and exact pair queries. Charge group/tile intermediates independently of final edges. Native reference must cover the declared fp32/fp64 and triclinic/partial-PBC envelope; qualify any narrower acceleration envelope explicitly. |
| `phydrax/discretization/particle/_verlet.py` | Per-case displacement/cell/absent-image certificates, exact wrap re-expression, pure candidate rebuild/select, initialization checks and complete accepted-state retention. No host synchronization to decide each case inside a compiled step. |
| `phydrax/discretization/particle/_cell_list.py`, `_morton_neighborhood.py`; spatial search owners used by these routes | Reuse/extend source preparation and query capabilities only where a missing invariant is shared. Preserve existing query ordering, radius-boundary, metric and tie contracts for non-atomistic consumers. |
| `phydrax/backends/spatial.py` | Add actual cluster/pair kernel specializations only after the native reference and matched phase-separated crossover evidence. Declare target, dtype, layout, derivative and resource envelopes; never advertise an accelerated selector that executes an ordinary reference fallback. |
| `phydrax/sparse/_ops.py`, `_execution.py`, `_streamed.py` | Complete the required segmented operations using existing stable gathers/reducers and reusable support/count state. Dot/multiply/matvec compose canonical gather and `phydrax.ein`/ordinary matrix multiplication; avoid new forwarding spellings. Mean/RMS must declare empty-segment and zero-norm behavior: masked padding is inert, while a requested undefined scientific mean/norm derivative remains unavailable/refused rather than silently repaired. Preserve first and mixed second derivatives and qualify higher requested orders. |
| `phydrax/discretization/splatting/_bspline.py`, `_assignment.py` and owning transfer code | Extend cardinal B-spline support through degree 5 (order 6), with channel payloads, exact deposition/gather transpose and admitted mesh extents. Compute continuity from degree and actual joins: degree p has C^(p−1) basis continuity at ordinary knots; force/Hessian continuity is lower after differentiation. Do not infer a universal derivative-order bound by blindly changing a constant. |
| **NEW** `phydrax/atomistic/_pair_kernel.py` | Bind a declared native scalar pair-energy function and dynamic parameter PyTree to MIC or explicit-image pair semantics, unit/cutoff/regularity contracts and the existing streamed relation machinery. Do not introduce another callback compiler or force-only shortcut. |
| `phydrax/atomistic/_neighborhood_bundle.py`, `_graph.py` | Resolve these routes/tiers/selection policies into prepared resource slots and runtime graph bindings; source selection and storage capacity remain distinct identities. |

### Tests and direct smoke

- Extend `tests/unit/discretization/particle/test_image_neighborhood.py` and the existing dynamic-relation/Verlet tests: exact route multisets against bounded NumPy lattice enumeration, triclinic and partial PBC, repeated/self images, zero/nonzero shifts, receiver subsets, reversal, tier containment, independent overflow, retained versus rebuilt cases, cell deformation, wrapping and uninitialized selective reuse.
- **NEW** `tests/unit/discretization/particle/test_cluster_image_search.py`: dense/cell/cluster route parity, highly skewed occupancy, empty cases, tiles overflowing before edge storage, fp32/fp64, and candidate-image budget refusal. Do not test incidental storage order unless deterministic ordering is the declared consumer contract.
- Extend `tests/unit/test_sparse_substrate.py`, `test_sparse_derivatives.py`, `test_sparse_streamed.py`, `test_sparse_streamed_derivatives.py`: empty and masked segments, active zero/canceling contributions, seeded deterministic/compensated accumulation, matvec orientation, JVP/VJP duality and mixed derivatives. Keep requested undefined zero-norm derivatives distinct from safe masked padding.
- Extend `tests/unit/discretization/test_particle_grid_splat_bspline.py`: orders 1–6, partition of unity, polynomial moments, deposition/gather adjoint identity, periodic wraps, finite differences within cells and declared join behavior, channel isolation, mesh/capacity refusal.
- Run a direct compiled native multi-case build → selective move/deformation → refresh → scalar pair energy/force evaluation; observe capacity rejection without advancing accepted state. Separately launch the real accelerated route on each claimed target.

### Benchmarks, docs and downstream coverage

**NEW** `benchmarks/atomistic_neighbor_routes.py` measures preparation, lowering, compile, first/warm query, build/reuse/selective rebuild, role-view formation and full force-evaluation cost. Sweep atom count, density/degree skew, batch heterogeneity, image count, cutoff tiers, receiver fraction and all controlling capacities; record logical schedules, compiler temporary/output/code bytes, process/device peaks, transfer and synchronization evidence. Pair-centric/grouping alternatives may be selected by measured total cost, not asymptotic naming alone.

Update `docs/guides_particle_data_plane.md`, `guides_sparse_spatial_hierarchies.md`, `guides_atomistic_dynamics.md`, `guides_mace_execution.md`, the owning discretization/sparse API pages and relevant examples. Consumers of changed splat/reduction/search substrates require semantic impact selection, including existing PME, LJ-PME, multipoles, particle-grid users, graph/meshfree users and native MACE/PaiNN/NequIP. Do not run only the new atomistic tests if a common primitive changes.

## P3. Native electrostatics, multipoles, D3 and FourierD3

### Methods and definitions

Keep separate scientific identities for: finite direct Coulomb; screened/shifted-force DSF; periodic Ewald; particle-mesh Ewald with a declared influence/modulus rule; admitted slab correction; point versus Gaussian-smeared multipoles; conventional truncated-CN D3(BJ); and FourierD3's modified-CN definition. Parameter estimation is immutable preparation with achieved/reference error evidence, not a guaranteed force-error certificate from a heuristic tolerance.

Charges and moments may be fixed system data or typed predictions of a preceding native model in the same energy dependency graph. All coordinate/cell/parameter derivatives include that dependency. A prepared reciprocal cache is bound to the actual cell, mode support, charge/moment revision where relevant, numerical policy and topology epoch. Reuse fixed structure while refreshing numerical coefficients through its owner; never detach a current-cell quantity merely to make caching easy.

Periodic Coulomb requires an explicit neutrality/background and macroscopic-boundary convention; its k=0 treatment is not FourierD3's. Slab support requires exactly the admitted two-periodic-axis geometry, the actual triclinic normal/projected height and its correction/background definition. Invalid PBC combinations must refuse, not return a zero correction that looks valid. Pair exceptions, self images, reciprocal self terms and per-atom decomposition availability are explicit. A global energy term must not advertise zero per-atom energies as a complete local decomposition.

Multipoles require declared component basis/order, normalization, moment origin and trace convention. Native Cartesian traceful moments and a source real-spherical traceless layout are not interchangeable because they share a shape. Any trace projection must be explicitly requested and reported. Gaussian width, overlap/self terms, field features, polarization status and stationary-response derivatives remain scientific contracts.

### D3 and FourierD3 invariants

- D3 scope is the upstream two-body C6/C8 BJ treatment, its admitted cutoff/switch policies and coordination interpolation. ATM/C9 is not included or silently substituted. Tables, covalent-radius scaling, damping, species support, units and padding are authenticated at preparation.
- Conventional CN truncation and the Fourier modified counting function are distinct definitions. Preserve the actual source epsilon/cutoff behavior when qualifying source fidelity; record and convert any dimensionful regularization. A tiny jump or underflow is not a proof of exact C1 continuity. Derive the supported regularity from the implemented formula and branch domain, including the transition and cutoff. A smoother mathematical alternative must be explicitly named, separately identified and qualified—not silently called the same method.
- Signed C6 separation uses the reference species/coordination block matrix, not an instantaneous atom×atom decomposition. Do not clip negative eigenvalues or route the result through PSD-only LJ-PME. Select rank by evaluating the actual reconstruction criterion; signed errors need not be monotone. Store achieved residual, chosen rank and exact factors. A capped rank that misses a required tolerance refuses or returns an explicitly unsuccessful preparation.
- Coefficient-table residual, CN-definition error, mesh/interpolation error, reciprocal truncation, roundoff and measured force/stress discrepancy are separate evidence. The coefficient tolerance alone does not bound every observable.
- FourierD3 retains the k=0 contribution, species-pair channels, Hermitian mode weights and coefficient-dependent self energy. Self energy participates before differentiating CN-dependent coefficients; otherwise its force and virial contributions are lost.
- Full directed CN neighborhoods and complete species coverage are verified by the bound resource/table contracts. No uncovered element is silently treated as padding.
- Mesh residency is bounded by batch × species × rank-chunk × grid; full per-atom rank coefficients and admitted outputs are also charged. Use bounded scan/map and checkpointed replay rather than unrolling arbitrarily many rank chunks into the compiled graph. No total-constant-memory claim.

### File-by-file changes

| File | Planned implementation |
|---|---|
| **NEW** `phydrax/atomistic/_reciprocal_mesh.py` | Domain-specific prepared mesh/influence/mode binding composed from native spline and spectral owners. Own row-cell reciprocal vectors, volume, half-spectrum weights, explicit discrete-modulus versus other admitted influence definitions, FFT-friendly host sizing, bounded channel execution and resource evidence. Instantiate it per term; it is not a global cache or another FFT library. |
| `phydrax/atomistic/_electrostatics.py` | Complete screened direct Coulomb and DSF terms; generalize Ewald real-space support to either the preserved MIC role or the distinct explicit-image pair sum; add prepared reciprocal-mode coverage under cell changes, dynamic context charges, admitted slab/background policies, higher-order PME and requested E/F/S/charge/cell actions. Keep alpha/cutoff/mesh selection outside dynamic iterations and report the numerical definition. |
| **NEW** `phydrax/atomistic/_multipole_basis.py` | Exact declared Cartesian/real-spherical maps, normalization/trace/origin contracts, Gaussian basis normalization and small self-overlap preparation. Reuse existing O(3) and native linalg owners; do not add another harmonic vocabulary. |
| **NEW** `phydrax/atomistic/_multipole_electrostatics.py` | Complete Gaussian-multipole direct-k/Ewald/PME and field-feature contracts through public degree 2, plus the current point-multipole definition as a separate method. Real-space, reciprocal, self/background and moment/position/cell derivatives compose through native resources. No public octupole claim from private higher-degree helpers. |
| `phydrax/atomistic/_polarization.py` | Consume complete point/Gaussian interaction actions as appropriate; remove orthogonal-cell restrictions only after triclinic energy/force/stress/response proof. Preserve PCG/TCG distinctions, convergence and force-residual gates, warm starts, implicit or explicitly algorithmic derivatives and rank/condition evidence. Do not replace a converged method by a fixed-step approximation under the same name. |
| `phydrax/atomistic/_force_field.py` | Reuse the reciprocal mesh owner for LJ-PME while retaining its own dispersion definition and PSD coefficient contract. Do not alter it into FourierD3 or drop nonbonded exceptions. |
| **NEW** `phydrax/atomistic/_d3_parameters.py` | `D3ReferenceTables`, damping/CN-policy records and `FourierC6Decomposition`: source digest/rights, explicit units, species/reference-slot masks, bounded host decomposition, exact achieved residual and restore validation. Never download tables during import or tracing. |
| **NEW** `phydrax/atomistic/_dispersion_d3.py` | `D3BJPotential` and prepared energy term: conventional admitted two-body definition, stable coordination interpolation, separate CN/dispersion role radii, switch policy, explicit image/self counting, full native derivatives within the declared regularity domain. |
| **NEW** `phydrax/atomistic/_dispersion_fourier_d3.py` | `FourierD3Potential`: modified CN, signed coefficient channels, closed-form radial Fourier kernels with justified small-argument series, exact self/CN chain, term-owned mesh and bounded rank execution. Distinct scientific and approximation identities; no direct-sum fallback under a mesh method name. |
| **NEW** `phydrax/atomistic/_dispersion_ts.py` | `TSDispersionPotential` for admitted AIMNet output definitions: typed predicted C6/polarizability fields, the exact source combination/damping and limiter policy, unit conversion, image-pair counting and connected model response. It is distinct from D3's coordination-table interpolation; preserve source-defined kinks and constants in identity. |
| `phydrax/atomistic/_potential.py`, `_potential_program.py`, `_neighborhood_bundle.py` | Typed charge/moment dependencies, `coordination` and `image-pair-sum` roles, per-term numerical artifacts and availability/status records. |
| `phydrax/atomistic/_profiles.py`, `__init__.py`; qualification/source catalog owners | Add exact method×geometry×precision×derivative×backend profiles and canonical exports. Existing `_mace_profiles.py` stays MACE-owned; do not create redundant registries for every fragment. |

### References, tests and scientific acceptance

Use **NEW** `tests/_support/atomistic_long_range_reference.py` only for small, independent host references: explicitly converged Ewald under the same macroscopic convention, finite-cutoff DSF, direct-lattice dispersion with independently formed coefficients, numerical radial Fourier quadrature, and finite differences. A naive conditionally convergent Coulomb image sum is not an independent periodic oracle. `chemistry.periodic.PeriodicEwaldPlan` is an additional native reference, not the sole independent one. Source-framework outputs remain isolated optional oracles.

- **NEW** `tests/unit/atomistic/test_long_range_electrostatics.py`: triclinic and admitted slabs; nonneutral/background conventions; small cells and self images; exceptions; per-case energy layouts; weighted cotangents; predicted-charge chain rule; first/cell HVP and mixed training derivatives; mode/cache coverage refusal; mesh/order/cutoff refinement. Replace broad loose-tolerance assertions with convergence evidence without deleting distinct existing failure contracts.
- **NEW** `tests/unit/atomistic/test_multipole_electrostatics.py`; migrate `test_advanced_polarization.py`: basis/origin/trace controls, Gaussian-to-point limits where defined, direct-k versus PME refinement, field equivariance, cell-strain and moment derivatives, stationary-response residuals and stale cache refusal.
- **NEW** `tests/unit/atomistic/test_dispersion_d3.py`: lawful reference tables, supported/uncovered species, padding, finite/periodic multiplicity, independent E/F/S and damping derivatives, documented cutoff joins and mixed derivatives away from unsupported branches.
- **NEW** `tests/unit/atomistic/test_dispersion_ts.py`: independent source combination-rule and unit controls, active reference coverage/padding, declared limiter branches, image counting and finite differences through predicted dispersion coefficients.
- **NEW** `tests/unit/atomistic/test_dispersion_fourier_d3.py`: signed-factor reconstruction and negative eigenvalues, capped-rank failure, same-definition direct lattice and radial-quadrature checks, total/frozen-CN/self forces, finite-strain virial, chunk equivalence within declared reduction tolerance, mesh-refinement translation artifacts, full-CN admission and species failures.
- Direct runtime smokes: mixed native MACE + PME + FourierD3 with independently valid resources; a finite/slab predicted-charge case; a large-radius small-cell D3 image sum; triclinic Gaussian multipoles; a rejected capacity/cell attempt preserving accepted state. Force/stress training and HVP-distillation smoke the actual requested mixed order, not only the primal.

**NEW** `benchmarks/atomistic_long_range.py` and `benchmarks/atomistic_dispersion.py` compare matched physical definitions and error targets. Separate prepare/lower/compile/first/warm, charge/moment/cell refresh, transformed execution, complete MD-step cost and memory. Sweep atoms, density, batch/cell heterogeneity, reciprocal modes, spline order, species count, rank and chunk capacity. Retain failed/OOM/refused rows and the exact source/build/runtime identity. Use `tools/audit_host_sync.py` and `tools/audit_memory_patterns.py` as candidate audits, supplemented by actual traced-path transfer/resource evidence; source matches alone are not runtime proof.

Update `docs/guides_atomistic_force_fields.md`, `guides_atomistic_polarization.md`, `guides_atomistic_dynamics.md`, their API/cookbook pages and scientific examples. Add a dedicated dispersion guide documenting the CN distinction, no ATM, table rights, signed factors, uncertainty/error budget and complete derivative matrix. D3 data rights and each real hardware/topology tuple are gates, not permission to label an unqualified route supported.

## P4. Native model families, conditioning, observables and source fidelity

### Source pins and actual model admission

Additional inspected architecture sources are [AIMNetcentral 0.2.0](https://github.com/isayevlab/aimnetcentral/tree/b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55), [fairchem-core source](https://github.com/FAIR-Chem/fairchem/tree/bb0a062a9a08d8b3ad3995aca5b9765e07804990), mace-torch 0.3.15 used by Toolkit and 0.3.16 used by the current native converter, and [latent Ewald source](https://github.com/ChengUCB/les/tree/35b971a650cf30adb84fef7138761e81aa47b3a2). Pin source/package/recipe identities independently of checkpoint weight digests. Provider environments are isolated because their e3nn and other dependency requirements need not coexist.

Inventory the actual wrapper contract, not every class present in a source package. The pinned Toolkit `MACEWrapper.adapt_output` indexes `raw_output["energy"]` and advertises E/F/S/atomic energies. Thus source `AtomicDipolesMACE`/`AtomicDielectricMACE` classes are not automatically supported energy-wrapper routes merely because they exist. Energy-bearing MACE/ScaleShiftMACE families, named MH-1 nonlinear variants, compatible energy-plus-property variants and MACELES must be assessed from their real forward signatures/configurations. Direct-only property models use declared native model ports and cannot be inserted into a scalar potential by inventing zero energy. The generic property/extension contract remains required; no blanket "any Torch module" claim follows.

The old streamed-MACE plan's exclusion of nonlinear/attention/joint-embedding families is not carried forward as a reason to omit a required pinned wrapper/catalog row. Complete the actual admitted families, while keeping architecture support, catalog checkpoint fidelity and custom native extensibility separate.

### Conditioning and observable contract

**NEW** `atomistic/_conditioning.py` owns `AtomisticConditioningSpec`, dynamic `AtomisticConditioning`, `AtomisticObservableRequest` and the typed requested-output/evidence container. A channel binds an existing `ValuePort`, case/atom level, dtype, representation, physical meaning, categorical versus continuous interpretation and an explicit default policy. Canonical tuples replace unordered native sets in identity payloads.

- Total charge, AIMNet2 spin multiplicity, UMA's task-specific spin/category index and continuous/categorical joint-embedding features are not conflated. In particular, a source UMA spin index of zero is not a physical multiplicity of zero. Map each source task's convention explicitly and record default use.
- Refuse missing required conditions and out-of-range/unmapped categories. Dynamic invalid cases carry status; never clamp a category into a supported range.
- Requests select energy, complete or scoped per-atom energy, forces, stress, charges/spin charges, dipole/quadrupole/polarizability where supported, named embeddings, Born-charge responses and Hessian actions/blocks. Each observable carries its actual units, origin, component/irrep basis, parity and availability. Quadrupole trace conventions are model-specific, not globally assumed traceless.
- Requests compose the existing `DifferentiationRequest`/`DerivativeContract` vocabulary. Do not create a second atomistic derivative-order language in training. Continuous charge conditioning may be differentiable; categorical source indices are not silently treated as continuous physical variables.
- Change the canonical graph evaluation method from `graph_energy(...)` to `evaluate_graph(AtomisticGraphInputs, request) -> AtomisticModelOutputs`, and migrate every implementation/caller together without an alias. Preserve `AbstractAtomisticPotential` as the energetic owner; unsupported scalar energy/conservative requests refuse before execution. Existing prediction functions retain distinct preparation/evaluation responsibilities rather than multiplying `predict`/`forward` synonyms.
- Prepared model-specific reductions return typed `system_partials`; the native executor combines owned contributions into `system_context` at declared stages. This serves AIMNet charge updates, UMA composition/conditioning/channel balance and their reverse rules. Model-locality declarations, not a generic wrapper, govern distributed admission.

### Native architecture algorithms

**AIMNet2.** Implement the actual AEVSV Gaussian scalar/vector radial expansion and source cutoff, scalar/vector `ConvSV` contractions including d2 features, species embedding/padding semantics, GELU MLP passes, iterative charge-feature feedback, single-channel and NSE two-channel modes, output heads and atomic shifts. For NSE, the source channel targets are Q/2 ± (multiplicity−1)/2. Keep rc_s/rc_v and embedded short-range Coulomb radius as explicit roles/tier views. Preserve the source output-module ordering and whether Coulomb/dispersion is embedded, subtracted or externally composed; do not add corrections twice.

Source NQE/NSE normalization adds epsilon to the system weight sum. Therefore its corrected charges do **not** sum algebraically to Q in every case. Match the actual source formula and expose its charge residual; do not write an exact-charge assertion that would force an unannounced model change. A separate exact-charge policy would be a different declared model definition. Downstream electrostatics consumes the actual predicted charges under its explicit neutral/background admission.

Source output inventories also include `DispParam` and `D3TS`. Implement their actual admitted behavior: species reference C6/polarizability multiplied by exponentiated bounded network outputs, followed by the source TS combination rule and BJ-like C6/C8 pair energy. The source feature/denominator bounds are declared model nonlinearities with kink regularity and unit-scaled constants, not silent numerical repairs. Bind the resulting native `TSDispersionPotential` from P3 so coordinate/parameter derivatives include the predicted coefficients. Validate active-species reference data; padding gets safe declared dummy values. Do not relabel this source-specific method conventional coordination-dependent D3.

**UMA/eSCN-MD.** Implement the Gaussian radial smearing, edge species embeddings, edge-degree embedding, source polynomial envelope, edge-aligned real harmonic rotations and m truncation, SO(2) convolution blocks, gating/S² activations, admitted normalization choices, grid/spectral atom-wise feed-forward blocks, residual placement, final norm and dataset-specific E/F/S heads. Reuse native O(3) representation/coupling owners and explicit source basis maps; no guessed normalization from matching dimensions.

Implement charge/spin/dataset embeddings, the actual mixture-of-linear-experts routing and mixed linear operators, and post-block system channel balancing. A generic mixture of complete model predictions is not equivalent. Any source-specific composition normalization, including `include_self` behavior, is a mathematically named policy in architecture identity—not a hidden model-release compatibility flag. Source-faithful and corrected scientific definitions must never share an identity.

The source nearest-neighbor cap (including its degeneracy band) belongs to the learned-graph selection policy. Candidate capacity must cover the selected graph and its tie band. Merged MoLE is an explicitly prepared realization tied to composition, charge/spin/task conditions and parameter revision; changed bindings refuse until explicit re-preparation. Preserve element references and output normalizers exactly. Mixed-task batches are admitted only where fixed execution signatures and scientific task ports are compatible.

**MACE expansion.** Extend the existing implementation, not a second MACE: attention-residual/nonlinear interaction families, relevant readout variants, Chebyshev/Gaussian radial choices and Soft transform, SO(3)/edge-irrep and first-layer conventions, joint categorical/continuous node/system embeddings, head selection and requested energy/property readouts. Preserve source W/U parameterization and actual parameter maps. Accelerated coupling may refuse a new family until its equations/derivatives are implemented and qualified; it may not silently run a reference implementation under the accelerated name.

**MACELES.** Add native latent readout heads and a separately defined `LatentEwaldPotential` composed with native reciprocal/finite interaction owners. The inspected source uses Gaussian-smoothed charge kernels: finite-space erf(r/(√2σ))/r, and periodic nonzero reciprocal modes weighted by exp(−σ²k²/2)/k², with explicit mode cutoff, normalization and self policy. These learned latent channels are not automatically physical partial charges. Preserve source normalization, self treatment, finite/periodic branch semantics and requested Born-response projection policy; do not silently add a Coulomb neutralizing background or physical-charge constraint. Use bounded finite-pair/mode streaming or an explicitly qualified accelerated summation, not hidden N×N arrays. The source's `node_energy` omits its global latent correction, so report scoped per-atom decomposition plus the global energy ledger instead of inventing atomic shares.

### Hessian and higher-response ownership

**NEW** `atomistic/_hessian.py` owns only the domain binding, not a new differentiation engine. `AtomisticHessianOperator` binds model revision, conditioning, geometry, cell, resource epochs and regularity, then uses `phydrax.linalg.prepare_linearization` on the energy gradient/force map. With forces F = −∇E, the Hessian action is **−DF·v**. Reuse the prepared linearization for repeated actions and native adjoints.

Dense per-case Hessian blocks are explicit bounded outputs with atom-output/atom-input/component axes; never form cross-case zeros or a full matrix for an action request. Mixed parameter derivatives of HVP losses require the actual third-order path. Coordinate/strain/mixed actions have separate admission. Frozen tables retain their proven continuity limits. Test rotational covariance by differentiating the rotated gradient; rotational zero modes apply at a stationary point, not at arbitrary geometry.

### File-by-file changes

| File | Planned implementation |
|---|---|
| **NEW** `phydrax/atomistic/_conditioning.py`, `_hessian.py` | Contracts and domain response binding described above, composed with existing ports/derivative/linalg owners. |
| `phydrax/atomistic/_potential.py`, `_types.py`, `_prediction.py`, `_potential_program.py` | `evaluate_graph` cutover, explicit conditioning/output/derivative requests, per-term field dependencies and honest local/global energy availability. Streamed numerical payload separates structural compile signature from host record provenance without weakening scientific identity. |
| **NEW** `phydrax/nn/atomistic/_aimnet2_features.py` | Source AEVSV and ConvSV formulas, fixed coefficient leaves, bounded native streamed contractions, source normalization and cutoff regularity. |
| **NEW** `phydrax/nn/atomistic/_aimnet2.py`, `_aimnet2_source.py` | `AIMNet2Architecture`, actual native potential/charge passes/output modules; exact reconstruction from a complete source tensor/declaration inventory; every source key consumed or explicitly refused. |
| **NEW** `phydrax/nn/atomistic/_uma_so3.py`, `_uma_layers.py` | Source rotation/mapping/grid conventions, SO(2) convolutions, normalization, activation and feed-forward blocks with explicit resource caps and native basis-map evidence. Reuse native arbitrary-degree O(3), not duplicate it. |
| **NEW** `phydrax/nn/atomistic/_uma_conditioning.py`, `_uma.py`, `_uma_source.py` | Task/charge/spin/dataset conditioning, actual MoLE routing/mixed linears and prepared merge, channel balance, E/F/S readouts, normalizers/references, native potential and exact source reconstruction. |
| `phydrax/nn/atomistic/_mace.py`, `_mace_interaction.py`, `_mace_readout.py`, `_radial.py`, `_mace_source.py`, `_mace_prepare.py`, `_mace_kernels.py` | Complete required source families and migrate `evaluate_graph`; preserve distinct exact/tabulated/folded/accelerated definitions and parameter spaces. |
| **NEW** `phydrax/nn/atomistic/_mace_embedding.py`, `_mace_les.py` | Joint embedding semantics and latent-head composition. LES numerics belong to a native atomistic term, not imported source execution. |
| **NEW** `phydrax/atomistic/_latent_electrostatics.py` | Latent Ewald definition, channel units/normalization, finite/periodic mode/self policies and global energy/response evidence using P3 owners. |
| `phydrax/atomistic/_feature_execution.py`, `_distributed.py` | Typed system-reduction stages and reverse accumulation interface specified in P11. Do not substitute a depth×cutoff geometry halo for actual per-layer feature/charge communication. |
| **NEW** `phydrax/atomistic/interchange/_source_admission.py` | Canonical family-neutral source envelope and rights/binding evidence, composed with family-specific parameter/basis proofs. Reuse external-artifact and resource-set validation; no second trust system. |
| `phydrax/atomistic/interchange/_mace_checkpoint.py`, `_mace_worker.py` | Explicit pinned source-release/configuration admission, new required energy-bearing families and exact optimized-layout normalization. Compiled/cuEquivariance source extraction must recover a real parameter mapping; opaque executables are not converted by wrapping them. |
| **NEW** `phydrax/atomistic/interchange/_aimnet2_checkpoint.py`, `_aimnet2_worker.py`, `_uma_checkpoint.py`, `_uma_worker.py` | Isolated source extraction, safe state/tensor admission, exact config/normalizer/reference/grid inventory, bounded pickle-free result. Trusted executable legacy formats require explicit digest-bound trust; safe load never falls back to unsafe load. |
| `phydrax/atomistic/_model_artifact.py`, `_checkpoint.py` | Register native families/validators and generic source envelope; persist every referenced model and dynamic condition in composed restarts. Keep family-specific proofs, full restore validation and actual numeric revision. |
| `phydrax/nn/atomistic/__init__.py`, `phydrax/atomistic/__init__.py`, owning interchange facade; `_mace_profiles.py`, `_profiles.py` | Canonical lazy exports and exact family/route support. Retain MACE-specific profile ownership; new families use the central new native profile owner, not a second registry. |

### Clean cutover, tests and artifacts

Refresh references to `graph_energy`, `_energy_unchecked`, `AbstractAtomisticPotential`, graph inputs, model artifact/source types and prediction results. Known consumers include `_prediction.py`, `_potential_program.py`, `_mace_prepare.py`, `_coarse_graining.py`, `_training.py`, `_distributed.py`, `benchmarks/mace_execution_scaling.py`, `benchmarks/atomistic_streamed_consumers.py`, model execution tests and `tests/integration/test_metal_models.py`. Audit all source-addressed callables and artifact recipes; no old-method aliases or forged historical IDs.

- **NEW** `tests/unit/atomistic/test_aimnet2.py`, `test_uma.py`, `test_atomistic_conditioning.py`, `test_atomistic_hessian.py`; extend existing MACE/radial/basis/source/acceleration/artifact tests.
- Independent small references cover AEV/charge-update algebra (including epsilon residual), Gaussian expansion, channel balance, source normalization and signed/native basis actions. Validate permutation/case/padding isolation, translation/rotation/reflection covariance appropriate to each irrep, declared cutoff regularity and source graph selection ties.
- Derivative cases: coordinate/parameter JVP/VJP duality; E/F/S finite differences; Hessian symmetry and translational identities; parameter-HVP finite differences; continuous-charge response; categorical refusal; changed topology/model/condition invalidation; nonconservative/property-only request refusal.
- **NEW** `tests/fixtures/aimnet2/`, `tests/fixtures/fairchem/` and their isolated generators: small actual source architectures, not mocked echoes. Preserve exact external release/digest in provenance, not native schema generations. Extend existing source MACE fixtures for newly admitted configurations.
- **NEW** `tests/interchange/test_aimnet2_checkpoint.py`, `test_uma_checkpoint.py`; extend the existing MACE conversion tests after locating their actual owner. Cover malformed/oversized archives, unknown classes, source dtype/layout/basis refusal, required rights/trust, parameter-direction fidelity, round-trip and provider-free fresh-process inference/training.
- **NEW** `tests/integration/atomistic/test_native_model_families.py`: each native family performs actual E/F/S and a small training/continuation step; connected predicted-charge electrostatics; source-to-native artifact load without source packages; native MACELES finite/periodic energy/response.
- **NEW** `benchmarks/atomistic_model_families.py`: architecture/cutoff/degree/channel/neighbor/case scaling, exact versus admitted prepared realizations, MoLE refresh/staleness, E/F/S and HVP/third-order training, prepare/lower/compile/first/warm and full memory evidence. Do not inherit source TF32 presets or uniform precision assumptions.

Update `docs/guides_mace_execution.md`, `guides_atomistic_interop.md`, `guides_atomistic.md`, `docs/api/atomistic.md`, `docs/api/nn/architectures.md`, examples and typing cases. **NEW** `docs/guides_atomistic_models.md` documents architecture versus checkpoint support, task/condition semantics, output origins/bases, derivative envelopes and rights. Preserve the old plan as history but mark its superseded model exclusions explicitly when implemented; historical benchmark JSON remains historical evidence.

Required artifact gates: AIMNet ensemble-member digests and weight rights; UMA gated access/weight terms and exact source configuration; MACE ASL-bearing model rights where applicable; optimized-source parameter-map admission; independent architecture/source fidelity; and each hardware/derivative tuple. These are not replaced by a successful random tiny model. No pretrained accuracy, arbitrary Torch conversion, derivative-through-neighbor-selection or automatic cross-backend claim is made.

## P5. Complete native dynamics, thermodynamic control and dynamic cells

### Prerequisite correctness and measure ownership

Source inspection shows the current pressure diagnostic uses `AtomisticForceState.virial`, while the program documents that field as a finite center-of-mass position moment, not periodic strain response. Dynamics also does not request stress in its ordinary configuration evaluation. Before changing it, reproduce the narrow periodic self-image/strain case and retain a failing-before/passing-after regression. Pressure-capable and cell-evolving plans must request and cache the actual strain derivative from the same scalar Hamiltonian.

Extend `AtomisticForceState` with declared availability and the cell/strain response needed by pressure and cell forces. In row-cell convention, configurational tensile stress is `sym(dE/dstrain)/V`; internal pressure is kinetic momentum-flux contribution minus tensile stress. Missing requested response is a refusal/unavailable status, never a fallback to the position moment. Constraint and bias contributions must enter according to the selected measure and splitting.

`AtomisticPhaseSpaceMeasurePlan` remains the single authority for physical degrees of freedom, constraints, mass metric, COM/momentum restrictions, coordinate-map and cell measure. Use the **rank of independent admitted constraints and restrictions**, not merely their count; redundant or singular constraints cannot silently change thermostat DOF. Rank/condition evidence belongs to native constraint/linalg owners and is prepared/certified rather than recomputed by hidden dense host operations every step.

### Numerical method choices

- Preserve native velocity Verlet and BAOAB and their existing accepted-state/RNG contracts.
- Bind Nosé–Hoover chains to the thermodynamic row and admitted DOF, with complete chain positions/momenta, source-required bounded chain lengths and Yoshida–Suzuki compositions, substeps and numerical mass policy. No separate hidden temperature on a thermostat plan.
- Preserve and correctly integrate native stochastic velocity rescaling where already supported. Add the upstream deterministic velocity-rescaling operation as an explicitly noncanonical equilibration/control method; do not advertise it as a canonical sampler.
- Consolidate duplicate GLE execution onto the existing discrete-transition/FDT-checked owner in `_stochastic_runtime.py`. Migrate PIGLET/noisy-force consumers only with preserved physical definition and proof; do not replace an Euler method by a different discretization under an unchanged identity.
- Implement complete MTK-type NPT/NPH with isotropic, diagonal/semi-isotropic where admitted, and triclinic cell control. The chosen cell chart, rotational gauge, barostat inertia, particle/cell chains, splitting order, Jacobian/measure corrections and conserved/work quantities are explicit numerical facts. NPH has no thermostat; any reference energy/temperature used only to set inertia is a declared numerical parameter, not a hidden 300 K physical target.
- Derive the factorized position/momentum/cell/chain updates from an identified MTK reference for the chosen chart before coding. Native matrix exponential/phi/solve owners implement nontrivial small cell actions; do not insert an ad hoc eigen or explicit-inverse formula. A more accurate exponential drift is a separately identified numerical realization, not alleged stepwise equality with the source's linear cell update.
- Generalize the existing isotropic Monte Carlo volume method to bundle-backed learned/classical programs. Acceptance includes proposed total energy, the exact coordinate/volume Jacobian, forward/reverse proposal density and the declared scaled entities. A scalar hydrostatic pressure contributes P·V; an arbitrary tensor target cannot be inserted into this scalar formula. Anisotropic mechanical-stress relaxation and a thermodynamic external-stress ensemble require distinct, derived work/measure contracts.
- Preserve valid existing deterministic pressure-feedback operations as mechanical control/relaxation when appropriate; they do not establish NPT. Delete an old implementation only when a new owning method actually replaces its semantics, not merely because both change a cell. Do not add unrelated thermostat/barostat families solely for breadth.
- A thermodynamic run has exactly one cell-evolution owner: an integrator-owned extended-system method or a scheduled MC move. MC NPH and other undefined combinations refuse. Required source-supported constraint combinations must be inventoried; richer constrained extended-cell profiles require their full constrained measure and projection proof, not a blanket promise from SHAKE availability.

### State, transition and RNG

Replace raw thermostat/barostat arrays with typed continuation records for particle and cell chains, GLE memory, barostat momentum/chart/reference cell, move/deposition counters and cell epoch. Their numerical arrays are dynamic fixed-role leaves. `prime_state` evaluates all required energy/force/strain resources without consuming a physical step, stochastic address, stage budget or adaptive-history event.

A candidate step operates on one accepted neighborhood bundle, complete thermodynamic/controller state and energy ledger. Every substage uses the same declared unit system and current state. Commit kinematics, cell, bundle, force, constraints, controllers, counters and ledger atomically only when all required contracts pass. A capacity-only retry uses the same physical attempt and RNG address. Scientific failure is not repaired by redrawing noise or silently changing algorithms.

All stochastic addresses use run/replica identity, realization, committed action/step, operator identity and stable particle or scalar-draw identity. Global thermostat draws and per-particle draws are distinct. Placement changes do not redefine noise. Bitwise numerical replay is claimed only for a qualified fixed runtime/execution profile; stable addresses alone do not imply bitwise equality across changed shapes/devices.

### File-by-file changes

| File | Planned implementation |
|---|---|
| `phydrax/atomistic/_dynamics.py` | Consume one bundle state; typed controller/cell continuation; force strain availability and complete cache identities; execute declared thermostat/barostat/cell-update schedule stages; explicit priming; periodic pressure correction; per-role retry evidence. Decompose validation, numerical transition, commit and result construction by invariant rather than extend an oversized monolith. |
| **NEW** `phydrax/atomistic/_cell_dynamics.py` | Native extended-system cell plan/state and complete NPT/NPH splitting, chart/gauge/mass/measure metadata, native small matrix actions and conserved/work evidence. It does not own another generic integrator runtime. |
| `phydrax/atomistic/_ensemble_advanced.py`, `_thermal.py`, `_stochastic_runtime.py` | Complete source-required thermostat schedules, thermodynamic-row binding, exact admitted DOF and semantic scalar/particle addressing; consolidate only obsolete duplicate GLE behavior through its canonical owner. |
| `phydrax/atomistic/_barostat.py` | Bundle-backed MC proposal/evaluation/acceptance and rollback; actual total energy including fixed bias; preserve each classical role's current-cell unique-image gate while image roles may have larger radii. Remove dense/learned refusal only with the complete new path exercised. |
| `phydrax/atomistic/_thermodynamic.py` | NPH and cell-method/measure admission, explicit pressure versus stress semantics, complete controller rebase requirements and qualification binding. |
| `phydrax/atomistic/_constraints.py`; native constraint/linear owners it consumes | Independent-constraint rank/DOF evidence, compatible mass-metric projections and required cell-scaling rules. Preserve finite iteration/convergence/status semantics and use native solves. |
| `phydrax/atomistic/_rollout.py`, `_hybrid.py`, `_checkpoint.py`, `_quantum_nuclear.py`, sampling and driven-flow consumers | Carry every new state field, cell owner and ledger component; migrate MC scheduling, RESPA/segment execution, GLE/PIGLET, replay and fresh-process restore without legacy unions. |

### Tests, smokes and qualification

Extend `test_dynamics_foundations.py`, `test_dynamics_thermal.py`, `test_dynamics_periodic.py`, `test_dynamics_rollout.py`, `test_dynamics_electrostatics.py`, `test_dynamics_hybrid.py`, relevant advanced/GLE/quantum consumers and `tests/integration/atomistic/test_dynamics_workflow.py`.

**NEW** `tests/unit/atomistic/test_dynamics_npt.py` and `test_dynamics_thermostats.py` cover: actual periodic pressure/self-image stress; harmonic and ideal-gas controls; extended conserved quantity and timestep refinement; correctly derived volume distributions for the selected measure/proposal; tensor/cell chart conventions; chain/DOF constraints; warm controller rebinding; one cell owner; finite/nonfinite/constraint/capacity rejection; exact counter and accepted-state rollback; same-profile restart and cross-placement RNG identity.

Statistical tests use predeclared finite domains, seeds, burn-in, correlation handling and acceptance intervals. A thermostat's finite-step ensemble bias must be measured/qualified; do not assert an exact canonical law for BAOAB at arbitrary step size or repeatedly run a failing statistical test until it passes. Physical qualification is distinct from a tiny analytic unit test.

Direct smokes: learned MACE plus PME under a changing triclinic cell, induced independent bundle rebuild, a capacity-only replay, checkpoint/restart, and NPH versus NPT controller behavior. No production claim follows from the smoke. Update dynamics/sampling/force-field API and cookbook pages, `examples/atomistic_dynamics.py`, phase-space and replay documentation, and source/qualification profiles.

## P6. Native relaxation, grouped paths and staged heterogeneous campaigns

### Optimization and cell charts

`AtomisticRelaxationPlan` binds a prepared native energy program, admitted coordinate/cell chart, constraint/tangent operator, native optimizer and convergence policy. Keep the existing explicit-host-gradient chemistry surface path for genuinely external/host providers; it is not replaced by hidden callbacks inside JIT.

- Implement FIRE/FIRE2 in the canonical `phydrax.optim` owner, with actual power/velocity/mixing/timestep state, group reductions, step caps and source-required variant semantics. Source defaults are configured method parameters, not tests of literal constants.
- Reuse/refactor the existing L-BFGS two-loop/history owner. Provide stepwise native execution and the source-equivalent trust-scaled force-driven method with no line search as a distinct method; it must not claim monotone energy decrease. Preserve the native line-search method for true scalar objectives. Weak curvature, non-descent direction, trust bounds and convergence before/after evaluation have explicit statuses.
- Atomic variables use unwrapped coordinates plus image counts so wrapping does not corrupt optimizer history. For a unit-cell or logarithmic deformation chart, derive the **full generalized gradient by the chart's native pullback** of the same energy/enthalpy. `−V·stress` is not a universally correct cell-force formula away from the identity chart. Fréchet/log-chart actions use native matrix-function/linearization owners.
- Hydrostatic/fixed-volume/selected-strain constraints are explicit tangent/retraction contracts, with projected and full residuals. Use sparse relations/`LinearOperator` actions for symmetry and coordinate constraints; do not allocate a dense `(3N+6)×K` basis except as an explicitly bounded reference output.
- Converged-minimum sensitivities require a stationary, regular constrained/gauge-fixed problem and a successful native Hessian/implicit solve. CONVERGED alone is not proof of Hessian invertibility. A converged saddle has a different inertia contract; validate it before an envelope/implicit derivative claim.

### Reaction paths

Extend `chemistry/reaction/_path.py` and the native atomistic PES bridge to bounded device execution of whole image bands. Reuse native FIRE/L-BFGS rather than retain a private optimizer loop. Preserve improved energy-weighted tangents, climbing-image selection, fixed endpoints, force/energy consistency from one accepted iterate and individual/group convergence evidence.

Periodic fixed-cell paths require explicit atom correspondence and a declared image/winding branch. An optional shortest-MIC initialization is a named policy with an ambiguity certificate, not an automatic replacement of the user's path. Every image must share the required chemistry/cell/scale contract. Whole paths are admitted, moved, checkpointed and published as groups; individual image slots are not independently refilled. Variable-cell solid-state NEB is not implied by coordinate/cell relaxation and is outside the pinned upstream fixed-cell NEB requirement.

### Campaign execution and continuation

**NEW** `atomistic/_campaign.py` is the atomistic composition over existing execution/workset/lifecycle owners, not another scheduler. `AtomisticStagePlan` records the scientific operation, input/output ports, budget, convergence/failure outcome routing and cache invalidation/reprime policy. It supports native MD, relaxation, grouped paths and prepared generation/label actions through the same accepted-transition contract.

Static signatures include layout/capacity, resource roles, model architecture/realization, derivative needs, precision, method-state extents and execution placement. Numeric model revisions/conditioning and task identity remain separate bindings. Do not compile record identity into every streamed batch, and do not hide identity by reusing stale metadata. Compatible stages may share a force evaluation only when their request and Hamiltonian/condition/bias context actually agree; line-search trials and different forces are not magically one shared call.

A cohort round is: evaluate admitted active/reprime lanes; form proposed accepted transitions; classify terminal/group outcomes; stage outputs; obtain durable publication receipts or park behind bounded backpressure; commit scientific/stage state; then refill eligible slots through native pool/frontier helpers. Reprime consumes no physical budget or adaptive event. Budget exhaustion, scientific convergence, capacity refusal and numerical failure remain different outcomes.

A continuation packet carries stable task/material/run/replica IDs, group/image membership, stage/counters, complete kinematics/cell/controller/optimizer/bias state, force and model/resource identities, RNG addresses, ledger and pending-publication state. Receiver re-preparation/reprime is an explicit transfer transition. A crash restores the last durable scientific/workset commit; it does not erase output acknowledgements or duplicate accepted labels.

### File-by-file changes

| File | Planned implementation |
|---|---|
| **NEW** `phydrax/optim/_fire.py`; `phydrax/optim/__init__.py` | Actual FIRE/FIRE2 init/step/solve and grouped reduction contracts, with deterministic state/evidence. |
| `phydrax/optim/_bounds.py`; **NEW** `_lbfgs_kernel.py` only for the real shared recursion/history invariant | Refactor the existing algorithm into one reusable step kernel; retain projected/line-search semantics and add the explicitly trust-scaled force-driven method. Confirm and test unbounded-domain admission instead of using arbitrary huge bounds. |
| **NEW** `phydrax/atomistic/_relaxation.py` | Prepared relaxation, generalized coordinate/cell charts, sparse/operator tangent constraints, native optimizer binding, physical versus projected convergence, resource/cell status and implicit-sensitivity admission. |
| `phydrax/chemistry/_atomistic.py`, `_optimization.py`, `reaction/_path.py` | Add the native periodic/batched PES path and device NEB; preserve honest host-surface execution; migrate the inline FIRE algorithm and result construction into the owning method/result contracts. |
| **NEW** `phydrax/atomistic/_campaign.py` | Scientific stage/workset composition, task/group continuation, compatible evaluation sharing, terminal outcomes, exact output-before-refill protocol and checkpoint envelopes. |
| `phydrax/_execution_pool.py`, `_execution_workset.py`, `_iteration.py`, `lifecycle` publication owners | Extend only demonstrated shared group-admission/acknowledgement/observation invariants. Keep nonlinear and cardiovascular pool consumers correct; no atomistic physics in the generic executor. |
| `phydrax/atomistic/_checkpoint.py`, `_reporter.py`, `_observer.py`, `_frame.py`, `_rollout.py` | Complete domain continuation, output receipts and observation scopes; maintain physical time versus optimization iteration versus generation pseudotime. |

### Tests, smokes, docs and benchmarks

- **NEW** `tests/unit/optim/test_fire.py`; extend the actual L-BFGS/constrained optimizer tests: independent analytic objectives, curvature history, trust limits, group state, and meaningful stepwise-versus-solve equivalence. Do not assert optimizer default literals.
- **NEW** `tests/unit/atomistic/test_relaxation.py`: analytic minima, target-pressure objective, cell-chart pullback finite differences, symmetry/constraint actions, full/projected residual distinction, wrapping/history, failed trial rollback, stationary/gauge sensitivity and failure statuses.
- Migrate `tests/unit/chemistry/test_constrained_reaction_qmmm.py`, `test_advanced_reaction.py` and existing path consumers. Test an independent two-dimensional saddle, explicit periodic winding, ambiguous atom correspondence, band-atomic failure, climbing-image transitions and one-iterate result consistency. Use HVP/inertia evidence on the non-gauge tangent space for saddle qualification.
- **NEW** `tests/unit/atomistic/test_campaign.py`: mixed task sizes/stages, incomplete groups, parked publication, capacity rebucketing, model revision changes, reprime, no duplicate outputs, interruption at every transaction boundary, exact semantic keys and scoped numerical replay.
- Direct smokes: periodic cell relaxation; a fixed-cell periodic migration path; heterogeneous relax → NVT → NPT → label campaign with one failed task, one capacity-limited task, output backpressure and a fresh-process restart.

**NEW** `benchmarks/atomistic_dynamics_campaigns.py` measures complete step/relaxation/path/campaign throughput, active-lane utilization, rebuild and refill cost, publication backlog, preparation/compilation and memory across atom/edge/method-state/group capacities. Extend existing dynamics benchmark entry points where they already own a scenario rather than retain duplicate drivers with the same purpose. Update dynamics/chemistry/optim/execution API pages, `docs/cookbook/atomistic_dynamics.md`, and add `examples/atomistic_relaxation_campaign.py`. Any removed algorithm/API/test gets an explicit old-to-new behavioral contract mapping.

## P7. Atomistic data, training, fine-tuning and operational continuation

### Reuse and clean public cutover

Keep `fit_atomistic_potential`, `AtomisticTrainingProblem`, `AtomisticTrainingPolicy` and `AtomisticTrainingResult` as the canonical frontend names, but replace the old frozen full-batch E/F/S-only input representation in one clean cutover. The problem binds immutable sources/mixtures, loader/workload policy, graph/program preparation, objective requests and validation membership. Remove obsolete supervision-split constructors once every caller has migrated; do not keep both representations behind overloads or aliases.

The frontend constructs the existing role-aware training kernel with one contribution per genuine objective, uses existing optimizer/parameter-subspace/target policies, and owns only atomistic preparation, physical normalization, selection, reporting and continuation. Operator training remains a consumer of shared owners, not a copied implementation or an atomistic special case.

### Objective, precision and training semantics

- Each objective returns numerator and support under an explicit reduction. Preserve current structure-balanced per-atom energy MSE as a named policy and provide source atom-weighted/total-energy variants. Force graph-balanced versus component-balanced losses and stress component masks are explicit. Accumulation/distribution merges numerators/support before normalization; never average unequal batch/rank means.
- Support MSE, MAE, Huber and source-required residual forms, explicit scalar composition and constant/linear/cosine/piecewise schedules. Schedules advance with the declared committed-update convention and stay fixed over an accumulation window. Literal versus normalized objective weights is explicit. Signed mathematical terms must be represented in a declared objective with appropriate semantics; do not infer that every negative coefficient is necessarily unbounded or silently normalize/drop it.
- Missing labels have per-field masks and availability; a nonfinite label is not silently interpreted as absent unless an explicit ingest policy declares that transformation and retains the loss report. Zero-support padding is inert. Failed model/resource evaluation invalidates the update, not merely one conveniently omitted loss component.
- Storage dtype, model compute precision and loss/metric accumulation precision are separate. Preserve high-precision labels through ingest; a requested lower loss dtype is an explicit numerical choice, not an automatic cast to coordinate precision. Geometry and higher derivatives cannot inherit an AMP dtype merely because a dense network matmul supports it.
- Optimizer transforms, clipping, schedules, parameter selection/freezing and accepted-update recurrence use the shared kernel. Validation-driven plateau changes are committed rule-state transitions, not mutable host edits hidden from checkpoint identity.
- EMA recurrence, parameter view used for evaluation, and best-model selection are separate policies. Keep fixed scientific tables and conditioned state out of parameter EMA unless a separately declared model-state policy requires them. Selection records the exact view/revision and metric definition.
- Low-rank adaptation acts within admitted parameter/irrep multiplicity spaces. A generic dense adapter cannot mix unequal irreps or be declared source-equivalent merely by shape. Reference-energy fitting uses native rank-revealing least squares and retains rank/residual evidence, source/split/head identity and the scientific energy convention.

### Immutable structure source and store

**NEW** `atomistic/_structure_store.py` owns `AtomisticStructureStoreSchema`, the `AtomisticStructureSource` protocol/in-memory implementation, `AtomisticStructureStoreWriter`, `AtomisticStoreCommit` and `open_atomistic_structure_store`. Reuse canonical resource admission, unit/axis identity and lifecycle publication.

Field declarations include semantic identity, level, dtype, trailing axes, unit/convention, extensivity, parameter/data role and availability. Required levels cover structures, atoms, edges, logical groups, custom segments and bounded product levels such as per-case Hessian blocks. Parent-level/cardinality relationships and integer-index rebasing are explicit; a field is not an index merely because it is integral. Cell, image, atom identity, source lineage and conditioning survive reads, transforms, grouping and storage.

Use Zarr's authoritative external format through an optional host I/O dependency, with **one canonical native representation** and no internal schema-generation fields. Writes seal immutable content-addressed shards and atomically publish a committed manifest through lifecycle/resource-set owners. Readers see a complete prior or new revision; failed append cannot publish a torn pointer/data set. Retraction/compaction publishes a new revision and preserves lineage. Ingest legacy/external stores only through explicit checked interchange, not an automatic repair path.

Field projection and coalesced bounded reads preserve requested order and duplicates. Validate metadata cardinalities/dtypes/byte counts before allocation/decompression. Optional cached edges use local stable mappings and exact geometry/cutoff/image/selection identities; a cache is never authoritative solely because its dimensions fit. Configurable compression/chunking/storage layout is execution metadata, not a different scientific object.

### Loader and staging

**NEW** `atomistic/_training_data.py` composes `IndexEpochPlan`, `DistributedIndexEpochPlan`, `BoundedPrefetchIterator`, native worksets and placement:

- `AtomisticSourceMixture` binds canonical source order, nonnegative probabilities, replacement/epoch definition and the intended sampled or importance-corrected objective. Record actual draw probabilities; repeated draws have distinct draw IDs but retain source structure identity.
- `AtomisticWorkloadBuckets` bounds cases, atoms, per-role edges/images, cells and bytes. A deterministic budgeted epoch plan supports atom-only, edge-only and combined budgets. Oversize records refuse or enter an explicitly configured larger bucket; they are not dropped silently.
- Derived edge-count indices bind source geometry revision, cutoff/skin, image/selection policy and transforms. Geometry-changing augmentation invalidates them unless a certified envelope covers it. Random transformations use semantic sample/epoch/draw addresses, not batch position.
- `AtomisticLoaderCursor` binds epoch/stream/batch/source cursors and committed accumulation progress. Resume does not merely skip the same number of rows in a possibly reordered source.
- Read/prepare ahead only for declared background-safe sources. Host queue depth and device staging bytes are bounded separately. Native placement/device transfer and synchronization are explicit; a producer thread alone is not a claim of overlapping GPU transfer.
- Separate host provenance envelopes from the static numerical compile signature. Preserve material IDs as needed dynamic leaves and validate binding at preparation; do not strip identity or call content changes identity-preserving to avoid retracing. Qualify one reusable executable per declared bucket/layout rather than per record.
- Train/validation/test separation is based on declared independent-unit lineage plus exact source/configuration identifiers. Detect duplicates and prohibited ancestry; do not claim that matching counts or arbitrary group labels prove scientific independence.

### File-by-file changes

| File | Planned implementation |
|---|---|
| `phydrax/atomistic/_training.py` | Extend the canonical problem/policy/result to sources, buckets, objective collection, shared optimizer/subspace/target controls, evaluation/selection and distributed binding. Remove superseded full-batch-only branches/constructors after caller migration. Full-batch becomes an ordinary one-batch policy, not a second trainer. |
| **NEW** `phydrax/atomistic/_training_objectives.py` | Domain objectives, physical reductions/schedules, normalization/support and request-union construction using P4's single observable/derivative vocabulary. Reference-energy fit with native least-squares evidence. |
| **NEW** `phydrax/atomistic/_training_data.py`, `_structure_store.py` | The loader/source/store invariants above, including bounded product-level data, actual index rebasing and atomic publication. |
| `phydrax/_training_kernel.py`, `_training.py`, `_training_checkpoint.py`, `_data_plane` | Reuse existing machinery; add only needed committed rule-state, subgroup epoch or common precision mechanisms with all consumers migrated. Do not put atomistic data structures in these generic modules. |
| **NEW** `phydrax/_training_loss_scale.py`; operator `_loss_scale.py`, `_fit.py`, facades | If required for shared AMP, move the existing loss-scale policy/state/scale-unscale invariant to this owner, give it one canonical generic name and migrate operator plus atomistic callers. No duplicate or compatibility alias; preserve current operator numerical behavior. |
| `phydrax/nn/parameters/_selection.py`, `_low_rank.py`; owning O(3) layers | Reuse exact leaf/alias/subspace semantics; add only mathematically valid irrep-aware adaptation, merge and identity contracts. |
| `phydrax/atomistic/_active_learning.py`, `_coarse_graining.py`, `_checkpoint.py` | Label sets expose native sources; campaign/member fitting and force-only coarse-graining use the new problem; preserve acquire/label/qualify/promote transactions and full training/dynamics restart. Existing external labeling remains honest interop but does not satisfy the native-teacher requirement. |
| `phydrax/atomistic/_reporter.py`, `_observer.py`; `_iteration.py`, `_execution_sampling.py` | Typed training/stage/campaign metrics, structured output and profiler/phase-memory boundaries through existing observation owners. Physics-changing controls remain identified transitions, not logging hooks. |
| `pyproject.toml` | Add only the necessary optional atomistic storage/interchange dependency group (for example current supported Zarr). No ALCHEMI/Torch/Warp runtime extra. |

### Migration and proof

Migrate known users in `_active_learning.py`, `_checkpoint.py`, `_coarse_graining.py`; `examples/atomistic_mace.py`, `atomistic_active_learning.py`, `atomistic_coarse_graining.py`; `tools/atomistic_rmd17_benchmarks.py`; MACE deployment/source test support; `tests/typing/cases/atomistic_execution.py`; and the training/stress/restart/deployment/active-learning integration families. The public docs and cookbook examples must change in the same cutover.

**NEW** tests: `tests/unit/atomistic/test_training_objectives.py`, `test_training_data.py`, `test_structure_store.py`. Cover independent float64 loss references, atom/structure/component weighting, partial/zero support, schedule boundaries, label precision, accumulation/distributed normalization, rejected-update counters, source probabilities, repeated reads/draws, edge-only budgets, oversize refusal, augmentation invalidation, leakage evidence, cursor identity and partial store publication. Fixed-profile accumulation/replay comparisons distinguish mathematical equality from bitwise guarantees that depend on reduction grouping.

Extend shared kernel/data-plane and operator precision/selection/adapter tests when their owners change. Add runtime import isolation and fresh-process continuation, actual memory-bounded streaming, and a repeated-bucket compilation-count experiment rather than a permanent source/wiring test. Direct smoke: heterogeneous finite/periodic E/F/S training with minibatches, accumulation, clipping, EMA evaluation, selected parameter subspace, failed update and restart. A separate actual distributed smoke uses P11, not an unchecked local `vmap` substitute.

## P8. Native offline and on-policy distillation

### Objective and measure contracts

Native teachers are frozen parameter-role components, not external evaluators. Coordinate/strain derivatives remain available for force/HVP labeling; excluding teacher parameters from the optimizer must not detach the input geometry. Teacher revision, conditions, source frame/geometry and requested label/probe definition enter immutable label identity.

- E/F/S and other supported property matching use P7 objectives and P4 requests.
- Atom-energy matching is a decomposition-bound regularizer, not universal physical supervision. Bind whether the source decomposition includes global corrections; do not compare incompatible gauges as an observable.
- Embedding matching binds layer/representation/axis identity. Scalar channels may use a learned dense projector; nontrivial irreps require an equivariant multiplicity-space map. Projectors are explicit trainable components with their own parameter space and artifact identity.
- HVP matching stores the actual Gaussian/Rademacher probe and its semantic address with the teacher label. Reuse `AtomisticHessianOperator`/native prepared linearization. Student parameter gradients of coordinate-HVP losses generally require mixed order three; strain-HVP variants have their corresponding mixed request. Preparation refuses unqualified accelerated/table routes; choosing a native exact route is explicit, never a fallback.
- Distribution matching reuses **`AtomisticPhaseSpaceMeasurePlan`, `PreparedThermodynamicStateTable`, `AtomisticCanonicalSamplingQualification`, `ReducedPotentialDataset` and native free-energy/overlap owners**. Do not introduce another ensemble-measure or equilibrium certificate. Energy autocorrelation/ESS diagnostics supplement, but do not prove, equilibrium admission.
- The base Boltzmann objective groups samples by exact phase-space measure and **one generating Hamiltonian/teacher-student revision**, evaluates the source-compatible self-normalized finite-sample KL objective with declared forward/reverse mixing, and records that it is an estimator rather than the population KL. Atoms of equal count are not the same system. Relaxation paths and adapting bias histories are not equilibrium samples.
- `−βU_old` is an unnormalized reduced energy, not a known normalized `log q_old`. Mixed-revision replay cannot combine importance ratios as if the unknown partition constants canceled across different generators. Permit such distribution objectives only through the existing multistate normalization/reweighting owner with cross-evaluated reduced potentials, mixture probabilities, overlap and uncertainty evidence; otherwise use valid single-generator groups or pointwise objectives. Do not label self-normalized finite-sample importance estimates unbiased.
- NPT or biased distributions use the owning reduced-potential and coordinate-measure convention, including P·V and any applicable Jacobian/frozen bias. Do not invent a second formula in the trainer. Gradient treatment of fixed sampled coordinates versus differentiating through sampling is an explicit objective definition.

### On-policy transaction

**NEW** `atomistic/_distillation.py` owns `AtomisticTeacher`, `AtomisticLabelRequest`, bounded `AtomisticExperienceBuffer`, `AtomisticOnPolicyDistillationPlan/State` and `run_atomistic_distillation_segment`. It composes existing campaign, store, training and qualification owners.

The state machine is ROLLOUT → LABEL → TRAIN → QUALIFY → PROMOTE. Retain complete walker/campaign continuation, generating policy revision, label cursor, immutable replay-store revision, committed-update/microbatch cursor, selected model/EMA state, qualification outcome and pending publication. Labels at cadence plus final frame are idempotent by semantic identity, not row count. Failed labeling/training/qualification does not replace the previously promoted student or consume an uncommitted cursor.

Required replay is bounded FIFO with explicit draw probabilities, admission and policy-age semantics; use existing acquisition/source-mixing owners for richer selection only when needed. Do not confuse experience replay with `AtomisticReplayPolicy` for deterministic trajectory recomputation. Teacher labels from different definitions/revisions remain distinguishable. A changed student either explicitly drives the next segment or waits for promotion according to the declared policy.

### Files, tests and deliverables

| File | Planned change |
|---|---|
| **NEW** `phydrax/atomistic/_distillation.py` | The native teacher/label/replay/controller transaction above; exact phase-boundary continuation and promotion. |
| `phydrax/atomistic/_training_objectives.py`, `_training_data.py`, `_structure_store.py` | Typed teacher fields/probes/representations, replay sources, per-generator grouping and objective reductions. No second loader or optimizer loop. |
| `phydrax/atomistic/sampling/_multistate.py`, free-energy/UQ owners | Reuse and, if genuinely missing, extend canonical sample/measure/normalization evidence for distribution objectives. Sampling owns equilibrium/reweighting semantics. |
| `phydrax/atomistic/_active_learning.py`, `_campaign.py`, `_checkpoint.py` | Native labeling and shared output/promotion/continuation integration; retain valid authoritative external-label interop separately from native completion claims. |
| **NEW** `tests/unit/atomistic/test_distillation.py`; P7 objective/data tests | Teacher parameter immutability after a real student update while coordinate derivatives remain correct; E/F/S/embedding/HVP independent controls; mixed third-order refusal/proof; replay eviction/admission; every phase interruption; stale/reordered source refusal; rejected promotion; no duplicated labels. |
| **NEW** `tests/integration/atomistic/test_native_distillation.py` | Offline store labeling → student fit → provider-free restart; on-policy native dynamics → teacher → buffer → train → qualify/promote with saved walker and replay state. |

Distribution-objective tests use independent discrete/harmonic controls with known normalization, same-count/different-chemistry refusals, single-generator stale-policy ratios, and mixed-generator cases requiring MBAR/overlap evidence. Embedding tests exercise the actual representation map; HVP tests compare a small independent dense Hessian and finite parameter perturbations. A random teacher example demonstrates mechanics, never pretrained or chemical accuracy.

**NEW** `benchmarks/atomistic_training_streaming.py` measures data read/prepare/staging, lower/compile/first/warm update, accepted updates, atoms/edges, padding, memory and HVP-versus-force cost. Extend `tools/atomistic_rmd17_benchmarks.py` for the native source/bucket frontend; preserve actual accuracy comparisons at matched settings. Distillation campaigns additionally report rollout/label/train/reweight/qualification/promotion costs, not training throughput alone.

**NEW** `docs/guides_atomistic_training_data.md`, `guides_atomistic_distillation.md`, `docs/cookbook/atomistic_distillation.md`, `examples/atomistic_streaming_training.py`, `examples/atomistic_distillation.py`. Update `guides_atomistic.md`, active-learning/coarse-graining/MACE guides, API, cookbook, examples, typing cases and checkpoint documentation. Source-weight rights, third-order accelerated support and hardware/distribution tuples remain explicit qualification gates.

## P9. Enhanced sampling, adaptive histories and bias-aware exchange

### Required scope and prerequisite reproductions

The pinned Toolkit supports periodic pair CVs, full-matrix umbrellas, walls, standard/well-tempered metadynamics with cadence/ramp, shared/walker/state histories, preallocated/growing/FIFO history policies, fixed-correspondence RMSD-history kernels, force-only pair-distance ABF, and synchronous temperature or equal-temperature window exchange. Native closure must cover these workflows, not only the corresponding class names. Native metric, image, derivative, restart and measure handling should be stronger; asynchronous exchange and an unqualified time-dependent reweighting formula are not implied.

Current source already has the relevant CV/bias/multistate/free-energy owners. Before edits reproduce the narrow affected behaviors: periodic biases currently subtract raw values rather than use their CV metrics; group geometry is evaluated from positions without always retaining an unwrapped molecular branch; biased forces are patched around the integrator rather than entering the complete conservative program; dynamic-cell bias stress is not included; nonempty thermodynamic bias IDs are rejected. [INFERENCE] The ABF sign/bin-grid correspondence, bias kick after momentum projection, and exhausted-history repeated failure need exact failing runtime scenarios before being recorded as confirmed defects. Keep independent regression controls, not source-text assertions.

### Selected contracts

**Bias as part of the scalar Hamiltonian.** Add `AtomisticBiasTerm` to the existing program. It consumes explicit control values, a read-only dynamic history view and current geometry/cell from `AtomisticPotentialContext`. Every force kick, constraint projection, rebase, barostat proposal, stress evaluation and reduced-potential cross-evaluation sees the same bias definition. Remove the obsolete force-patching path only after this cutover is complete. Physical forces needed by an estimator come from an explicitly requested physical force group or a correct scalar decomposition, not an incorrectly signed subtraction from total force.

**Metric and geometry.** Add `CollectiveVariableGeometry` carrying current cell vectors, supplied unwrapped positions/image branch and coordinate domain. Periodic group COM/Rg/RMSD/path CVs use the native accepted unwrapped representation or a declared, certified topology-based reconstruction. An anchored MIC reconstruction is allowed only under its explicit extent/uniqueness certificate; it is not a general solution for long molecules or winding polymers. Local triclinic MIC uses `PeriodicCell`'s owning certificate, never componentwise fractional rounding masquerading as shortest physical distance.

CV results report zero-length/collinear/alignment-gap/image/periodic-branch/nonfinite status as appropriate. Masks are numeric-safe; runtime boolean slicing is removed. Higher derivatives require the relevant branch and regularity certificates. For RMSD, compute the aligned MSD using a native quaternion/self-adjoint spectral or equivalent admitted alignment owner, retain the eigengap/conditioning, and derive higher response through that owner. A stopped optimal rotation justifying a first envelope derivative is not a general Hessian rule. RMSD at zero and MSD-based history energy have distinct regularity.

**Bias forms.** Umbrellas use the declared metric displacement and full/scalar/diagonal stiffness, with PSD evidence from native linalg. Walls/flat bottoms specify their branch interval and exponent/continuity; periodic walls without a meaningful branch refuse. Source-compatible nearest-image Gaussian hills retain their seam/branch limits. A richer smooth periodic kernel is a separately selected wrapped-Gaussian/Fourier representation with a truncation/derivative tail certificate; a finitely truncated image sum must not be called exactly periodic and smooth merely because omitted tails are small.

Cell-dependent bias derivatives come from the same strain map. For example, a volume umbrella `V_b = κ(V−V0)²/2` contributes tensile stress `κ(V−V0)I` and the opposite pressure contribution; do not swap pressure and stress signs. A moving center and a changing history have separate protocol-work increments evaluated under an explicitly fixed boundary-state convention.

**History ownership and events.** `AtomisticBiasHistoryPlan/State` bind kind, owner axis (shared/walker/state), capacity, deposition cadence, ramp, well-tempered parameters and continuation. Arrays remain dynamic fixed-role state. Each force evaluation holds history fixed; deposits occur only at declared committed boundaries. For a simultaneous multi-walker event, compute all heights against the pre-event history and insert in stable semantic order, all-or-nothing. Never deposit during a rejected proposal, cross-evaluation, force prime or rebase.

Capacity policies are explicit: terminal exhaustion without repeated automatic retry; FIFO with its free-energy-estimator limitations; and bounded grid representation with approximation evidence where supported. Source unbounded growth maps to explicit finite host capacity-ladder transactions or continuing segments, never allocation inside a compiled loop. Rebinding preserves history content, owner identities and counters under a new prepared identity; it is not silent compatibility. Warm references are a declared new-run initial condition with provenance, whereas resume restores exact continuation; do not confuse approximate reconstructed history with an exact checkpoint.

**ABF.** Required pair-distance parity includes the geometric/entropic correction, half-open bins, sample thresholds/ramp, force caps, no out-of-range clipping into edge bins, and the correct bias sign. Use an independent radial reference `A(r)=U(r)−2kT log r` to establish it. The physical conditional mean-force estimator is not the raw projected force alone.

For the explicitly supported general-CV extension, let `Jξ` map physical tangent perturbations to CV perturbations. Construct dual vector fields through native actions and the small Gram solve, satisfying `Jξ bα = eα`, and evaluate `dA/dξα = E[∇U·bα − kT div_μ bα | ξ]` under the **declared base measure μ**. Do not form a full dense Hessian for its trace. Holonomic cases require tangent fields, rank/condition evidence and the actual constrained/coarea/Fixman measure factors derived for that profile; there is no universal extra determinant weight that may be inserted without identifying the measure. Start qualification with pair distance and bounded named multi-CV/constrained controls; reject unqualified measure/rank cases explicitly.

Force-only ABF is a nonconservative force port with work evidence and refuses energy-based exchange/barostat/free-energy admission. An integrated bias is a separate realization: native interpolation/integration constructs `−A` from the admitted mean-force field (including support/connectivity/integrability evidence), and it is conservative only for fixed history. Frozen integrated ABF can participate in ordinary energy workflows; do not blanket-refuse all ABF modes or pretend force-only ABF has an energy.

**Bias-inclusive multistate execution.** Reuse `PreparedThermodynamicStateTable`, `PreparedAtomisticMultistate`, replica exchange/SAMS and native reduced-potential datasets. Each row's potential identity binds physical Hamiltonian controls, bias controls, history owner and fixed/history revision. Required cross-evaluation includes the actual bias at the trial configuration/cell; any relevant label/history/control change invalidates force caches. Histories are selected by their declared owner, not blindly swapped with coordinates.

Define one atomic iteration order: propagate with H_n → eligible cell proposal with H_n → required reduced potentials/assignment move with H_n → rebase all continuation state → eligible deposition/update using the declared origin/target labeling convention → force refresh if history changed → publish/commit. Implement the required source temperature and window-exchange modes; combined or walker-dependent Hamiltonian moves need their own detailed-balance derivation before admission. Controller memory, momenta, cell state and cache identities rebind together.

Adaptive samples are not ordinary equilibrium samples. Preserve `AtomisticCanonicalSamplingQualification`; mark both SAMS and bias-history adaptation in dataset activity. Frozen-history segments may be authenticated and analyzed through existing MBAR/reweighting owners, including a zero-sample unbiased target row when supported. Standard/well-tempered/ABF profile diagnostics retain convergence, owner-temperature and storage-policy assumptions; mixed-temperature history or FIFO does not automatically support the usual profile formula. No second free-energy estimator or self-issued equilibrium certificate is introduced.

### File-by-file changes

| File | Planned implementation |
|---|---|
| `phydrax/atomistic/sampling/_collective_variable.py` | Current geometry input, metric-aware branches, safe masked operations, explicit status and regularity, group-coordinate and optional aligned path behavior. Preserve distinct physical coordinate domains. |
| **NEW** `phydrax/atomistic/sampling/_cv_geometry.py` | Certified group/image geometry binding and topology-based whole-molecule reconstruction using native cell/topology owners; no second neighbor search. Large coordination CVs request the bundle CN role. |
| **NEW** `phydrax/atomistic/sampling/_alignment.py` | Domain alignment/MSD/RMSD binding with native small spectral/linearization evidence, fixed correspondence and derivative-gap admission. |
| `phydrax/atomistic/sampling/_model_collective_variable.py` | Forward the same geometry/metric/identity context; retain array-bearing learned CVs as dynamic fixed-role leaves. |
| `phydrax/atomistic/sampling/_bias.py` | Canonical conservative form plans and `AtomisticBiasTerm`, metric-aware evaluation and full cell response; remove obsolete independent force augmentation after migrating callers. Avoid retaining competing old/new bias forms as aliases. |
| **NEW** `phydrax/atomistic/sampling/_bias_history.py` | Bounded hills/reference/grid histories, cadence/ramp/well-tempered deposition, explicit owner maps, freeze/digest, capacity-rebind and event work. |
| **NEW** `phydrax/atomistic/sampling/_abf.py` | Pair-distance estimator and admitted generalized/constraint measure actions, consistent bins/ramp/cap, force-only versus integrated realization and native integration/solve status. |
| **NEW** `phydrax/atomistic/sampling/_biased_dynamics.py` | Move the scientific biased-step transaction/checkpoint owner here from the overloaded `_bias.py`; it composes P5 transitions and history events, not a second integrator. |
| **NEW** `phydrax/atomistic/sampling/_adaptive_profile.py` | Explicit diagnostic adaptive profile results with support/convergence/storage/temperature assumptions; never mislabeled equilibrium `FreeEnergyResult`. |
| `phydrax/atomistic/sampling/_free_energy_learning.py` | Bind existing learned conservative bias/committee gate as a program term; preserve the owning restrained mean-force estimator. |
| `phydrax/atomistic/sampling/_multistate.py`, `_multistate_checkpoint.py`; `_thermodynamic.py` | Typed bias cross-evaluation, owner/history/control-aware potential IDs, full rebase, adaptation activity and authenticated continuation. Replace the blanket nonempty-bias refusal only when this path is complete. |
| `phydrax/atomistic/_potential_program.py`, `_dynamics.py`, `_barostat.py`, `_checkpoint.py` | Context history/control view; conservative and explicitly nonconservative requests; bias-inclusive strain/MC energy; complete force-cache and ledger identity; no deposition on probes. |
| `phydrax/atomistic/free_energy/_switching.py`, `phydrax/uq/_atomistic_free_energy.py` | Admit supported static/frozen biased definitions through existing work/reduced-potential conventions; retain explicit refusal of unqualified adaptive reweighting. |
| Owning facades and application CV consumers | Migrate actual geometry arguments and results in protein/nucleic observations/torsions/cotranslation; preserve legitimate nonperiodic positions-only uses. |

### Tests, direct scenarios and evidence

**NEW** focused files under `tests/unit/atomistic/`: `test_collective_variable_geometry.py`, `test_bias_forms.py`, `test_bias_history.py`, `test_abf.py`, `test_biased_dynamics.py`, `test_bias_multistate.py`. Preserve/migrate the contracts currently in ecosystem/advanced, kinetic CV, learned-bias, multistate/free-energy, controlled switching and biological observation tests.

Controls include: independent triclinic image enumeration; explicit molecular image branches across boundaries; long/winding molecule ambiguity rather than incorrect anchor reconstruction; RMSD versus independent Kabsch values; degeneracy and zero-distance derivative refusal; periodic torsion seam behavior; exact volume-umbrella stress sign; moving/history work closure; cadence/owner/FIFO/exhaustion/rebind; no deposit during rejected/probe/rebase calls; constraint tangent residual under bias; radial ABF entropic term and double-well flattening; consistent bin boundaries; constrained/general-CV mean-force quadrature; harmonic-window exchange; fresh-force equality after label/history changes; frozen-history MBAR controls; and refusal of unqualified adaptive or force-only energy claims.

**NEW** `tests/integration/atomistic/test_enhanced_sampling.py` exercises a periodic umbrella with cell response, well-tempered history, fixed-correspondence RMSD exploration, pair-distance ABF, window exchange, shared/walker/state ownership, freeze-to-analysis and fresh-process restart. Keep these separately collected scientific scenarios, not one broad soft-assertion test.

Extend `tools/atomistic_sampling_benchmarks.py` where it owns scenarios, and add **NEW** `benchmarks/atomistic_enhanced_sampling.py` only for the phase-separated scaling campaign it does not own. Vary history size/owners, CV support, grid size, replicas/states and cross-evaluation capacity. Report prepare/lower/compile/first/warm, deposition/rebase/refresh and full propagation overhead, memory and statistical evidence. A small smoke does not establish free-energy convergence. Qualification comparisons use independent analytic/quadrature controls and declared correlation/uncertainty policies; no retry-until-green statistics.

Update `docs/guides_atomistic_sampling.md`, `docs/api/atomistic_sampling.md`, `docs/cookbook/atomistic_sampling.md`, `examples/atomistic_sampling.py`, `examples/atomistic_learned_bias.py`, relevant dynamics/thermodynamic documentation and generated profiles. The existing distinction between adaptive history and equilibrium samples remains, now with a complete native bias-aware execution path.

## P10. Native generation streams and crystallographic search

### Domain ownership and required surface

Add one public subpackage, **`phydrax.atomistic.crystal`**, alongside the existing sampling/interchange domains. It owns crystallographic setting/ASU semantics, molecular crystal input, contact relief, comparison and ranking. Numerical rotation, periodic-cell, sparse search, optimization, transport, data-store, execution and lifecycle machinery remain with their current owners. Register the new domain as a capability-lifecycle closure exception; do not introduce another top-level simulation framework.

The required source scope includes generic conditional/custom-sampler generation, bounded and continuing streams, fixed/sampled space-group policies, standard-setting general-position molecular packing with integer Z′, multiple components/conformers, contact rules and formula-unit volume, ASU/expanded storage, distributed quotas, radial screening/deduplication, recipe persistence and physical relaxation handoff. Native source-to-conformer execution must be included where the upstream convenience workflow creates conformers; requiring callers to provide them would silently narrow that workflow.

Additional all-setting/special-position, confirmed crystal-equivalence and exact continuation modes below are explicitly designed native improvements with separate support/qualification rows. They are not implied by merely accepting a space-group number or by a radial score.

### Native generation protocol and lifecycle

**NEW** `atomistic/_generation.py` owns the atomistic support/condition/proposal/stream composition, not another flow/diffusion solver. Reuse P4's typed condition ports and existing `ScoreContext` semantics rather than another shape-only condition dictionary. Distinguish fixed chemistry, bounded variable composition with explicit element vocabulary, fixed versus variable cell, representation, mass/charge/topology admission, units and resource policy.

A prepared sampler is a native array-bearing callable/scientific transition with declared state, input/output ports and resources. Stateless one-shot proposals can complete in one transition; stateful packing/iterative samplers retain actual continuation. `PreparedCoordinateSampler`, native reverse diffusion/probability-flow/continuous transport and `PreparedCrystalPacker` consume this contract. Typed condition transforms and pure filters replace unrestricted mutating hook behavior; every rejection retains reason/provenance rather than silently disappearing.

The stream uses existing campaign/workset and publication owners. Keys are derived from campaign/support/model identity, stable proposal ordinal and role, never from lane/chunk/rank/completion order. Ordinal-order publication uses a bounded pending-result bank and watermark. A checkpoint includes budget/cursor, active sampler/campaign state, pending results/receipts, model/condition/source revisions and RNG addresses. Recomputing from a watermark is allowed only for an explicitly pure idempotent one-shot sampler; it cannot replace saving the optimizer/packer/history state of a general stateful sampler.

Unlimited source-style iteration is expressed as an explicit chain of finite continuation segments with bounded resident work, not an unbounded on-device allocation or hidden retry. Zero-target/zero-work calls have specified empty results and counter behavior. Generated proposals become typed structure-store records and, when requested, complete native relaxation tasks; geometry screening alone is not physical qualification.

### Periodic generation and representation

Extend native atomistic diffusion/coordinate-generation admission deliberately. Periodic coordinates are fractional plus an explicit cell representation and image/branch convention. A `FlatTorusManifold` may supply the geodesic transport contract missing from the existing torus state geometry, but its metric must be explicit:

- componentwise fractional wrapping is valid for a declared fractional Euclidean representation;
- it is **not** the shortest physical displacement in a general triclinic cell;
- physical-metric logarithms use the native lattice/MIC owner and its uniqueness/branch evidence.

Cell coordinates use a positive-volume chart with explicit height/condition bounds and native retraction/linear algebra. Gauge, centering, cell and representation Jacobians are required for any density/likelihood statement; pseudotime is not MD time. A categorical species decoder does not infer valid bonds, charge state, stereochemistry or molecule identity. Those enter a declared chemistry admission/qualifier. Preserve fixed-support biological generation and migrate its sampler/conditions/identity handling without replacing its numerical model or silently asserting permutation equivariance.

### Crystallographic identity and tables

`SpaceGroupSetting` binds International number, exact Hall setting/symbol, origin/axis/cell choice and canonical rational affine operations. Store integer rotations and rational translations in a deterministic exact representation; a denominator-24/int8 compression may be used only after the table generator proves its range and completeness. General exact arithmetic is the authority, not a guessed fixed bound or float32 table.

Use admitted immutable Hall/operation data with provenance and rights (for example the inspected spglib database as data/test oracle), but implement expansion, closure, setting conversion and numerical execution natively. The required 230 standard-setting map matches the pinned source's actual Hall choices, including origin and rhombohedral/hexagonal settings. Additional 530-setting support is explicit and checked operation-by-operation. Derive chirality/Sohncke and other symmetry properties from operations, not group-number ranges or a hard-coded label list.

For row lattice H, use its metric and the exact operation action to derive the invariant metric subspace. Recover a positive cell through native factorization and report condition/positivity evidence. Check the Cartesian realization of every operation. A monoclinic/rhombohedral setting cannot be handled by a generic crystal-system projection that ignores its setting.

General-position multiplicity includes centering and all admitted operations. Special-position support is a separately qualified exact stabilizer/coset mode: solve affine congruences with integer-shift ranges derived from the operation on the unit cell; form/deduplicate the finite intersection/stabilizer lattice; bound the search and return incomplete/undetermined evidence on exhaustion. Do not use arbitrary shifts `[-2,2]³` or enumerate all operation subsets without a resource bound. Native exact-algebra/linalg owners supply reusable elimination/integer operations; do not invent a second generic solver.

Molecular special positions additionally require an admitted molecular symmetry/permutation action and orientation-stabilizer compatibility. Expansion uses coset representatives and records multiplicity and rational Z′ without duplicate copies. Wyckoff letters come only from an admitted authoritative mapping, not a guessed labeling algorithm. Enantiopure/chirality constraints are explicit: improper operations cannot silently turn a conformer into an unrequested enantiomer.

### Molecular input, native conformers and contact relief

The molecular input binds formula-unit composition, atom/bond/stereo identity, conformer pool/local frames, masses, charge/spin, contact-distance and volume policies. Source graph/file parsing may be optional host interchange, but no external embedding, force-field optimization, packing or symmetry engine counts as native numerical execution.

Native conformer construction uses a bounded distance-geometry/torsion workflow: admitted bond/angle/stereo/rotatable-bond constraints → bound propagation/triangle consistency → explicitly approximate low-rank metric embedding with native spectral evidence → native constrained error/energy optimization → stereo/ring/geometry qualification. Do not silently repair an indefinite distance Gram matrix or label an approximate embedding exact. Use a declared finite molecule/constraint/attempt envelope and existing native optimizers; no private gradient/Newton loop. A supplied native physical potential owns energetic cleanup; missing parameters/support is explicit. Preserve native input-conformer support as well.

`CrystalPackingPlan` combines fixed/weighted group selection, Z/Z′, component/conformer choices, volume/shape/height policy, finite attempts/step budgets and explicit atom/image/contact/workspace capacities. Weighted priors and atomic-volume/radius tables are separate admitted data, not scientific facts inferred from element names. Unknown elements need declared data or refuse; do not copy a silent provider fallback.

Candidate state contains ASU fractional centers, proper rotations/quaternions, cell metric/chart coordinates, conformer choices, counters and overlap evidence. Expansion retains source ASU atom, molecule, component, conformer, operation and lattice-image mapping. Centers may be wrapped while molecular atoms remain on their declared whole-image branch. Identity follows attempt ordinal and scientific input, not rank allocation.

Contact relief evaluates a declared overlap penalty with native image searches. Exclude only intramolecular pairs in the same copy with zero net image shift; different copies and periodic self-images remain. Use an exact counting convention so an unordered-pair and directed-image sum is not accidentally double-counted. Pull gradients/torques/cell actions back to ASU variables through native geometry; small inertia solves use native linalg. Step limits, cell compression/deformation and refusal policies are explicit numerical choices. The overlap tensor is not physical stress, and overlap convergence is not energy convergence.

The packer is a stateful campaign operation with real partial-budget shortfall and terminal reasons. No hidden unlimited resampling, acceptance of truncated contacts, or fabricated success when the target count is not reached. Bundle preparation must accept explicit role requirements without requiring a potential object, so contact and descriptor consumers reuse it directly.

### Symmetry-constrained relaxation, equivalence and ranking

Symmetry-constrained physical relaxation supplies **operator/sparse tangent and adjoint actions** for ASU/site/strain degrees of freedom to P6. It must not materialize a dense `(3N+6)×K` basis for a large expanded crystal. Local site projectors and small strain subspaces are prepared once. Nonlinear rigid/orientation charts use genuine retraction; an affine displacement constraint is not a rigid-body manifold. Retain both projected and full physical residuals and verify symmetry after relaxation. An unconstrained run may break symmetry; its source setting then remains provenance, not a false property of the result.

Radial screening preserves source typed/untyped modes, hydrogen selection, inclusive threshold and explicit greedy non-transitivity. It is a screening score, not proof of crystallographic equivalence. Add a bounded native confirmer:

1. admissible lattice reduction with returned integer basis transform and boundary/tolerance evidence;
2. bounded compatible basis/sublattice enumeration, including explicit supercell policy;
3. candidate origin shifts from matching type classes;
4. periodic type-preserving assignment using the existing native assignment owner;
5. molecular topology/stereo correspondence and RMS/max residual checks.

Return `EQUIVALENT`, `DISTINCT` or `UNDETERMINED`; exhausting a basis/shift/automorphism/assignment budget is not proof of distinction. At degenerate reduction boundaries, several reduced cells may be valid; test equivalence and returned transforms, not a false universal unique floating representation. Improper mappings and enantiomorphism are explicit. Keep actual structure IDs distinct from a tolerance-based equivalence certificate.

Deduplication uses deterministic declared priority, a persistent store-backed representative bank and confirmed matches when requested. Greedy representatives and connected-component closure are distinct policies; a tolerance relation need not be transitive, so a component does not certify every member pair. Ranking compares converged/qualified structures under the same formula-unit and Hamiltonian definition, with residual/symmetry/provenance and uncertainty evidence. Different compositions require explicit chemical-potential/reference semantics. Energy ranking is not finite-temperature polymorph stability.

### File-by-file changes

| File | Planned implementation |
|---|---|
| **NEW** `phydrax/atomistic/_generation.py` | Typed support, sampler transition, conditions/filters, bounded stream/cursor and complete continuation composed with P6/P7. |
| `phydrax/applications/_coordinate_generation/_native.py`, `_support.py`, `_decoder.py`; protein/nucleic generation facades | Consume semantic per-sample keys/ordinals, typed conditions and shared stream/sink lifecycle. Move all-sample host materialization to explicit output policy. Preserve biological support/stereo admission and registered artifacts. |
| `phydrax/atomistic/_diffusion.py`; `phydrax/metrix/_flat_torus.py` and facade | Admitted periodic support and metric-explicit manifold actions; retain distinction between species corruption and chemical-generation admission. Reuse transport diffusion/flow/guidance/evolution owners. |
| **NEW** `phydrax/atomistic/crystal/__init__.py`, `_setting.py`, `_hall.py`, `_space_group_data.py` | Canonical facade, exact setting/group closure/transforms and generated admitted operation data. |
| **NEW** `phydrax/atomistic/crystal/_lattice.py`, `_wyckoff.py` | Invariant metric/cell charts, native bounded reduction/sublattice actions, stabilizer/coset/site admission and truthful completeness evidence. |
| **NEW** `phydrax/atomistic/crystal/_input.py`, `_conformers.py`, `_atom_types.py` | Explicit molecular/stoichiometric/contact/volume input; complete native conformer workflow; topology/stereo-aware type refinement and bounded correspondence. Reuse chemistry coordinate/topology, graph/combinatorial and native optimizer owners. |
| **NEW** `phydrax/atomistic/crystal/_asu.py`, `_packing.py`, `_symmetry_basis.py` | ASU state/expansion/provenance, bounded contact-relief transitions and sparse/operator symmetry-tangent actions for native relaxation. |
| **NEW** `phydrax/atomistic/crystal/_equivalence.py`, `_dedup.py`, `_ranking.py` | Certified bounded matching, persistent dedup and same-definition physical ranking. No external numerical matcher/packer in product runtime. |
| **NEW** `phydrax/atomistic/interchange/_cif.py`, `_rdkit.py` where needed | Explicit host graph/format admission only, with complete units/stereo/setting/provenance and declared losses. RDKit must not supply numerical conformer embedding or relaxation. Refuse unresolved disorder/occupancy rather than invent a structure. |
| **NEW** `tools/generate_space_group_tables.py` | Deterministic generation/check of exact canonical data from an admitted manifest; verifies ranges, closure, setting identities and rights before publication. |
| **NEW** `phydrax/atomistic/crystal/_profiles.py`; `atomistic/_profiles.py`, facades and catalog | Domain-specific crystal profiles plus shared generation profiles, exact source/setting/geometry/derivative/resource envelope and closure exception. |
| `NOTICE`, `LICENSES/`, reference-artifact manifests | Verified code/data notices. CSD prior, COD volume, Hall/operation/Wyckoff and optional parsing data have separate rights gates. No source code license is treated as data permission. |

### Tests, scenarios, docs and benchmarks

**NEW** `tests/unit/atomistic/crystal/test_setting.py`, `test_lattice.py`, `test_wyckoff.py`, `test_asu.py`, `test_packing.py`, `test_symmetry_basis.py`, `test_equivalence.py`, `test_dedup.py`, `test_ranking.py`, `test_conformers.py`; **NEW** `tests/unit/atomistic/test_generation_stream.py` and the owning torus-manifold tests. Keep each independent scientific failure meaning separately collectable.

Independent controls include standard-setting operation counts/closure and actual Hall choices; Cartesian metric invariance; nonstandard-axis/origin transformations; group multiplicity and special-site cosets; skew-cell ASU expansion; proper/improper chirality; explicit multi-image contacts/torques; a molecule crossing a boundary without false intramolecular contacts; conformer ring/stereo/rank failure; operator tangent/adjoint identity; basis/origin/atom/molecule/supercell equivalence; enantiomorphism; bounded-search UNDETERMINED; radial score versus independent sorted-distance references; greedy versus component semantics; zero-work and partial-budget results; ordinal-addressed RNG; stateful stream interruption and idempotent publication.

Source spglib/pymatgen/RDKit tools may be isolated admitted test oracles, never runtime substitutes. Table equality alone is not sufficient: use independent group laws, geometry/correspondence checks and published small crystal/CIF controls. Data rights must be established before committing fixtures. Niggli boundary tests accept equivalent certified reductions, not a tautological copy of the source output order.

Required direct scenarios: molecular input → native conformer → general-position packing → ASU store → expansion → native physical relaxation → screening/confirmed dedup → same-definition ranking; chiral/racemic admission; skew/small cells with self images; budget shortfall; stateful restart; fixed/variable-cell periodic generation with explicit chemistry qualification. Separately qualify richer all-setting/special-position and constrained-relaxation rows; they are not inferred from general-position success.

**NEW** `benchmarks/atomistic_crystal_packing.py`, `atomistic_crystal_dedup.py` measure preparation/compile/first/warm, complete attempts and accepted qualified outputs, contact/relaxation/dedup costs and memory across atom/operation/image/contact/pool/attempt capacities. Compare equal input proposals or declared sampling distributions, tolerance, budgets and physical quality; raw attempts/s alone can reward a worse generator.

**NEW** `docs/api/atomistic_crystal.md`, `docs/guides_atomistic_crystal_prediction.md`, `docs/guides_atomistic_generation.md`, `examples/atomistic_crystal_prediction.py`, `examples/atomistic_generation_stream.py`; update biological generation guides/API/callers, atomistic interop and MkDocs navigation. Document packing versus physical ranking, projected versus full minima, radial screening versus certified equivalence, and the absence of unqualified finite-temperature stability/likelihood claims.

Required gates are actual admitted crystallographic/chemical tables and rights, appropriate native physical models/parameters, all declared setting/molecule envelopes, and observed distributed/hardware evidence. Optional caller-supplied tables are a valid explicit input contract, not permission to mark an unavailable packaged reference/prior campaign complete.

## P11. Distributed training, domain execution, reciprocal work and stage pipelines

### Current route distinctions and selected cutover

The current owner-local MACE path performs real per-layer feature communication and reverse cotangent return. Its global-host ingress, fixed-cell assumptions, replicated controller arrays and host-gather checkpoint limit the supported execution envelope. In contrast, global-evaluate-then-mask helpers and certification of a reciprocal result supplied by the caller are not owner-local classical/PME/polarization execution. Some existing collective wrappers do communicate/reduce data; do not describe the entire old slab module as fake, but do not count those helpers as a complete distributed scientific solver either.

Consolidate spatial atomistic execution on one **`AtomisticDomainPlan` / prepared runtime / state** contract built from the existing owner-local implementation and P1/P5 state. Migrate and remove the superseded slab and `OwnerLocalAtomistic*` public families only after their valid numerical, migration, reduction, status and checkpoint scenarios have complete replacements. Global/dense reference evaluation remains explicitly labeled reference functionality in the owning test/reference path, not a second product runtime or compatibility shim.

**Retain both required spatial strategies:**

- **Halo domain decomposition:** partition-local geometry/receivers, role-specific physical halos and per-layer feature/adjoint exchanges.
- **Replicated-geometry graph partition:** explicitly replicate the admitted geometry/topology while partitioning receiver/model work and performing the required feature collectives and adjoints. Charge its replicated memory. Halo decomposition plus data parallelism does not automatically replace this source capability or dominate its performance.

One periodic system per domain plan matches the source baseline; independent systems use data/workset lanes and explicitly composed meshes. This is not a promise that every heterogeneous system or model is automatically distributable.

### Native communication and model semantics

Reuse `ExecutionGroup`, `ExecutionPlan`, native collective providers/leases, `DistributedOwnershipPlan`, `DistributedPointLayout.from_blocks`, `DistributedHaloPlan`, and the existing distributed spectral owner. No new communicator framework or vendor runtime is introduced.

Per-role searches share material ownership but have their own cutoff/image/selection/capacity and cell-deformation certificates. Ownership is defined in the declared fractional partition, not guessed from cell diagonal lengths. Add process-local ingress; global host materialization is not the only supported entry. Migrate complete continuation leaves and commit globally only when every owner admits the candidate.

Extend `AtomisticLayeredModel` in `_feature_execution.py` with prepared system-reduction stages: owned receivers produce typed `system_partials`; the executor combines them into `system_context` consumed by the next declared operation. AIMNet2 needs actual per-pass charge normalization; UMA needs composition/condition/MoLE and channel-balance reductions. Per-layer communication, rather than a single depth-times-cutoff ghost expansion, remains explicit.

The transpose of a replicated system-context consumption sums each owner's local cotangent once and returns it once to the owned partial producers. Do not multiply the response by owner count, reduce ghost energies twice, or discard parameter/cell cotangents. The graph-partition route implements the matching all-gather/reduce-scatter pair through native collectives; add a declared `psum_scatter`/equivalent capability to the canonical provider if it is absent. Do not fake it by retaining every rank's unnecessary full gradient buffer under a partitioned-memory claim.

Model locality and global features are declared per architecture and observable. A position-force route does not imply cell, stress, HVP or mixed training support. MACE with no system-reduction stages must retain its current numerical contract. Requested shared-parameter reductions, total charge, moment/bias/thermodynamic scalars and per-term energy/strain contributions have an exactly-once owner and declared reduction order.

### Actual reciprocal and polarization distribution

**NEW** `atomistic/_reciprocal_distribution.py` binds particle-owner patches to the native spectral slab/pencil layout:

1. spread owned charges/moment/channel payloads with the admitted spline support into bounded local patches;
2. route and sum patch contributions into owned FFT-layout cells with explicit overlap/halo/packet capacities;
3. execute the existing distributed transform and locally owned mode kernel;
4. reduce required energy/strain/system contributions once;
5. execute the inverse transform and exact adjoint redistribution;
6. gather the requested potential/field/force actions on owned atoms.

Derive patch coverage from the current cell, spline support and reuse certificate, not an orthorhombic `skin/h` shortcut applied to every skew cell. Use the same term-owned scientific definition and influence normalization as P3. Ewald without a mesh distributes mode/structure-factor work with its own bounded reduction; PME, multipole, LJ-PME and FourierD3 channels share the redistribution substrate, not each other's physics.

A replicated mesh is an explicitly named bounded comparison/execution strategy where useful, not labeled a distributed FFT. Complete owner-local polarization uses the native linear/operator solve, global convergence/force-residual evidence, warm-start state and selected implicit/algorithmic sensitivity. A certify-only interface does not satisfy it, and fixed-iteration TCG cannot silently replace converged PCG.

### Data-parallel and combined execution

**NEW** `atomistic/_training_distribution.py` owns atomistic case placement/objective reduction bindings over the shared training kernel. Use group-local ranks and immutable global epoch/draw identities; `_data_plane/_distributed.py` must not assume the entire process world when invoked on a subgroup.

Reduce global objective numerators/support before normalization and apply the same declared operation to parameter contributions. Empty-support shards are inert. Fast global-array execution and ordered/reproducible modes have explicit identities. **An ordered sum of already-reduced shard totals is not partition-invariant arithmetic.** If bitwise invariance to partition is a required profile, retain canonical atomic contribution order/accumulator semantics and qualify all affected primitives; otherwise state a roundoff-bounded numerical contract.

DP×domain and stage×domain combinations use explicit named resource groups/mesh axes and qualified collective schedules. Do not infer them from separate DP and DD unit tests. Semantic RNG addresses are rank-independent; numerical equality across different layouts is separately tested and scoped.

### Distributed dynamics and constraints

Use P5's typed controller/cell/force/ledger components, with explicit owner-blocked versus replicated storage. Global kinetic energy, momentum, virial, thermostat DOF, CV and optimizer dot products are native collective actions of the scientific method. Migration transports all applicable per-atom, per-group, controller/history, optimizer and stage state—not just coordinates.

Constraint communication has its **own support topology and capacity**. It is not bounded by Verlet skin, which is a temporal reuse margin. Prepare complete constraint group/edge communication and use native constrained solves/projections with global status. Unsupported constraint/ensemble/distribution combinations refuse before a trajectory starts; do not silently perform rank-local projections that violate the global constraint. RNG/counter and failure/rollback rules match the single-device scientific method.

### Stage pipelines and restart

Stage groups are explicit `ExecutionGroup` children. Use a prepared root-group fixed-capacity transport schedule with typed P6 continuation packets and credits; there is no implicit second communicator. The scheduling contract specifies group participants, collective order, packet schemas, publication receipts and termination/cancellation consensus. Nonparticipants cannot enter incompatible child collectives, and empty rounds still obey the root schedule. Prove the subgroup/transport primitive in a tiny real multi-process probe before building larger campaigns; if the current canonical provider lacks a required operation, implement/admit it there explicitly rather than choose a hidden fallback.

A full downstream sink parks upstream completed tasks without consuming new physics or overwriting their packets. Group transfers are atomic. Rank failure is fail-coupled by the existing launcher/supervisor: the job terminates within its bound and restarts from the last durable commit; it cannot promise a returned collective rollback state after a rank has disappeared.

Distributed continuation uses `lifecycle/_distributed_checkpoint.py` and existing repository/resource publication. Publish per-process array pieces and replicated fields from their canonical owners, assemble complete coverage, then atomically publish one global commit pointer/receipt. Partial coverage never becomes restorable. Include active tasks, transport credits/pending packets, source/loader cursors, controllers, histories, resource epochs and model revisions.

Overlap-only restore must retain integrity: use authenticated chunk/region records or another owning bounded scheme, not skip a whole-file checksum merely because only part of a shard is wanted. Restore only needed certified regions, reassemble global logical identities and reprepare changed ownership/resources through the existing topology-restart/migration owner. Same-profile ordered replay can be bitwise where qualified. Changed owner/process counts preserve scientific state and RNG lineage with explicit new preparation/migration evidence; future floating-point trajectories are not universally bitwise identical.

### File-by-file changes

| File | Planned implementation |
|---|---|
| `phydrax/atomistic/_distributed.py`; **NEW** `_domain_plan.py`, `_domain_state.py`, `_domain_runtime.py`, `_domain_checkpoint.py` | Decompose by admission/preparation, scientific state, execution and durability; migrate the owner-local core into the single `AtomisticDomain*` contract and retire superseded routes only with the full behavioral migration map. Remove obsolete private forwarding/re-export modules after all callers move. |
| **NEW** `phydrax/atomistic/_graph_partition.py` | Required bounded replicated-geometry receiver partition route, feature all-gather/reduce-scatter adjoints, global-condition handling and exact memory evidence. Shares model and scientific state contracts; it is not another model implementation. |
| `phydrax/atomistic/_feature_execution.py`; native model modules | Typed system reduction stages and correct transpose, per-layer halo/graph-partition execution, requested property/parameter/strain responses and model-specific locality admission. |
| **NEW** `phydrax/atomistic/_reciprocal_distribution.py` | Certified particle-patch↔spectral-layout transfer, mode/channel ownership, packet/status/resource evidence and exact adjoints. |
| `phydrax/atomistic/_reciprocal_mesh.py`, `_electrostatics.py`, `_multipole_electrostatics.py`, `_polarization.py`, `_force_field.py` | Bind the native reciprocal distribution to the same single-device scientific definition and complete polarization solve; eliminate certify-only substitutions. |
| **NEW** `phydrax/atomistic/_training_distribution.py`; `_training.py`, `_data_plane/_distributed.py` | Case/subgroup placement, global objective support, declared reproducibility, complete training checkpoint bindings and DP×domain composition. |
| `phydrax/backends/distributed.py`, owning collective provider/lease/program modules | Add only actually required native collective capabilities such as graph-partition reduce-scatter; declare effects, scopes, adjoints, device/provider and ordering. No ad hoc host MPI/NCCL layer in atomistics. |
| `phydrax/discretization/spatial/_distributed_relations.py`, spectral `_distributed.py` | Extend only missing route/patch/layout operations and differentiability/resource invariants; preserve other spatial/spectral consumers. |
| `phydrax/lifecycle/_distributed_checkpoint.py`, `_restart_topology.py`, repository/publication owner as needed | Complete global commit publication, authenticated overlap/chunk restore and explicit topology-change lineage without weakening corruption/refusal checks. |
| `phydrax/atomistic/_campaign.py`, `_dynamics.py`, `_neighborhood_bundle.py`, `_checkpoint.py` | Typed stage packets/credits, distributed scientific reductions/constraint support, full candidate commit and continuation integration. |
| Owning facades, `_mace_profiles.py`, `_profiles.py`, capability catalog | One canonical domain API plus an explicit graph-partition strategy; per-tier/model/method/derivative support instead of one umbrella distributed claim. |

### Verification and performance

Keep a precise old-to-new map for the existing distributed MACE, slab/migration/reduction/checkpoint and ecosystem scenarios. Do not delete `test_distributed_atomistic_completion.py` wholesale before moving every consumer-visible contract. Update documentation that currently describes supplied reciprocal evidence as if it were distributed PME computation.

Required tests include existing `test_distributed_mace.py` and new focused files for system reductions, graph partition, reciprocal distribution, distributed dynamics, atomistic training distribution, stage transport and distributed restart. Independent global native references must use the same scientific model, not global-evaluate-then-mask as the claimed implementation. Cover empty owners; repeated/self images; predicted charges; mixed moments; exactly-once context cotangents; strain/HVP/mixed derivatives; subgroup epoch sampling; zero-support shards; migration of every continuation field; asymmetric capacity failure; incomplete publication; tampered regions; changed model/measure/topology; and credit/backpressure saturation.

Dedicated multi-process integration tests use `launch_local_processes`, natural device topology and finite failure deadlines. A killed rank must fail the coupled job without hanging or publishing a partial commit. Same-topology restart and changed-topology migration are different tests. Forced CPU devices and lane-vmap are functional oracles only, not accelerator or multi-host proof.

Extend/rename `benchmarks/distributed_mace_scaling.py` to **`benchmarks/distributed_atomistic_scaling.py`** in one clean consumer/doc cutover if its scope becomes family-neutral. Preserve old JSON as historical evidence. Add declared model, halo/graph-partition, reciprocal, process/host and campaign scenarios. Measure strong and weak scaling; prepare/lower/compile/first/warm; each layer's communication; patch redistribution/FFT; full MD/training step; rebuild/migration; publication/restore; logical/compiler/process/device memory. A warmed evaluation table excludes neither cost silently nor failed/OOM rows.

**NEW** `tools/atomistic_distributed_qualification.py` reuses existing runtime identity and observed-host/topology admission. Qualify separately: lane reference; one process/multiple physical devices; multiple processes on one host; and at least two observed physical hosts. No specific unavailable GPU or host is assumed present during this plan. Multi-host hardware, transport/provider admission and matched source artifacts remain explicit gates.

Update `docs/guides_atomistic_distributed_execution.md`, distributed/model/training/operations API and cookbook pages, relevant examples, typing cases and generated profiles. Actual end-to-end acceptance includes halo and graph partition for supported model families; native distributed reciprocal interactions; learned dynamic-cell dynamics; DP×domain training; a backpressured multi-stage campaign; failure/restart; and admitted topology-change continuation.

## P12. Integration order, verification, documentation and completion gates

### Dependency-aware implementation waves

Phase numbers identify ownership, not an instruction to serialize independent work. The shared contracts are agreed first and each implemented slice must be a complete native vertical path, not a registry entry or compiling scaffold.

| Work item | Minimum ready prerequisites | Complete deliverable / parallelization boundary |
|---|---|---|
| P0 | Current tree/source/rights inventory | Source-derived operation/checkpoint/support matrix, exact baseline and explicit missing external prerequisites. Reference admission and independent mathematics research can proceed without accelerator availability. |
| P1 | P0, agreed contracts from all tracks | Migrate existing MACE/PaiNN/NequIP/classical programs to real typed requests and a functioning resource bundle; preserve existing prediction/dynamics/restart behavior plus the new mixed-resource scenario. P1 includes the shared conditioning/request definitions described in P4, avoiding a P1↔P4 cycle. |
| P2 | P1 resource/search contracts | Complete native search/layout/tier/selective/segmented/spline routes and focused proof. Model algorithm/source work can proceed independently against the fixed contract. |
| P3 | P1 and the required P2 transfer/search operations | Complete each electrostatic/multipole/dispersion definition with derivatives and evidence. Source-specific AIMNet TS and MACE latent terms coordinate with P4 but do not share model/runtime ownership. |
| P4 | P1 model/condition/observable contract; P3 only for the corresponding correction-dependent scenarios | Native architecture families and artifacts, each with actual inference/training/derivative/fidelity workflows. Do not block unrelated model kernels on a gated pretrained weight file. |
| P5 | P1 accepted state/strain/measure contracts and required P2 certificates | Complete each thermostat/cell method over existing native models; P3/P4 add their coupled acceptance cases as ready. Cell physics is not postponed until a campaign API exists. |
| P6 | P1 and the relevant P2/P5 transitions | Native optimization/path execution and whole-task stage/refill/receipt lifecycle. Optimizer work can proceed alongside P3/P4; no shared-file concurrent edits. |
| P7 | P1 request/identity contract and existing shared kernel/data owners | Complete source/store/minibatch/fine-tuning/control/restart frontend using existing model families, then qualify each P4 family. |
| P8 | P7, native P4 request/response capability, relevant P5/P6 rollout | Complete offline and on-policy distillation, including full requested derivative and sampling-measure admission. |
| P9 | P1/P5 conservative bias/transition/measure contracts | Complete metric/history/cross-evaluation/analysis paths. Existing free-energy owners stay canonical; no dependency on a separate trainer for basic sampling. |
| P10 | P1/P2 role search; P6 relaxation and P7 store for the complete workflow | Symmetry/input/conformer algorithms can start from P0 independently. Full generation/packing/relaxation/dedup/ranking closes after the shared runtime/storage interfaces are ready. |
| P11 communication/reference routes | P1 stable packet/model/resource contracts | Root/subgroup transport probe, process-local ingress, halo and graph-partition execution, exact collective transposes and checkpoint publication. Start with existing MACE; do not wait for every new scientific method. |
| P11 coupled routes | Relevant P3/P4/P5/P6/P7/P9 owners | Add native reciprocal, model system reductions, controllers/constraints/bias, training and stages as each scientific contract is ready; qualify the combined meshes, not just the isolated pieces. |
| P12 | Stable integrated behavior and final generated outputs | Cross-family scientific/operations/performance matrix, docs/examples/API/capability consistency and exact release-evidence disposition. |

### Shared-file ownership and integration rules

The integration owner exclusively edits shared `_potential.py`, `_types.py`, `_conditioning.py`, `_prediction.py`, `_potential_program.py`, `_neighborhood_bundle.py`, `_system.py`, `_graph.py`, `_dynamics.py`, `_thermodynamic.py`, `_checkpoint.py`, public facades, shared source/closure catalogs and generated inventories. Domain owners submit exact requested interface changes and consumer inventories; they do not independently add competing fields or parallel context/bundle types.

Specialists own disjoint numerical/model/method/source files. Generic sparse/spline/linalg/training/execution/lifecycle owners review changes to their invariants and all direct consumers. The integration owner coordinates the common loss-scale extraction, subgroup data-plane changes and authenticated distributed restore, which affect non-atomistic consumers.

No implicit commits, merges or PRs are part of this plan. Follow the repository's explicit `//merge`/`//pr` workflow if requested later. A merge invalidates only evidence affected by its actual delta, but common schema/configuration/fixture/owner changes may invalidate a broad matrix. Never reuse stale evidence merely because a test name is unchanged.

During implementation, keep one verification record with exact command/scenario, source/build and test/driver fingerprint, environment/dependency/backend/device/topology/precision identity, result, exact failing nodes and material timings. A passed check is reusable while all relevant inputs remain unchanged. Do not rerun valid checks just to restate a handoff.

### Mandatory clean cutovers and retirements

| Old assumption/surface | Replacement and completion condition |
|---|---|
| One program cutoff and one physical neighborhood/cache pair | Per-role certified bundle, resolved term slots and one accepted bundle state; migrate all direct program/dynamics/provider/consumer and checkpoint recipes together. |
| Pair-only unused bundle definition in `_sites.py` | The single `_neighborhood_bundle.py` owner; old definition/import path removed, not aliased. |
| `graph_energy` and missing conditioning/property context | One `evaluate_graph` contract with typed ports/conditions/requests and honest energy/property availability; every native model, prepared realization and source converter migrated. |
| Fixed full-batch supervision-only training representation | The canonical training problem over immutable sources, workload policy and objective collection; remove old constructors/branches after migrating examples, labels, coarse-graining, tests, typing and restart. |
| Bias forces patched around unrelated kicks | Bias term in the scalar program plus separately identified adaptive events/nonconservative port; preserve constrained projection, cell energy, work and history state. |
| Independent thermostat/cell update functions counted as complete ensembles | Complete admitted method/splitting/controller/measure state. Retain valid mechanical controls under truthful semantics; do not silently replace one discretization by another. |
| Inline FIRE/duplicated L-BFGS loops in domains | Native optimizer owner used by complete relaxation/path runtimes with physical chart/convergence evidence. |
| Slab/reference/certify-only functions counted as owner-local scientific execution | Complete domain/reciprocal/polarization routes and explicit reference functions; migrate each valid old scenario before deleting obsolete names. |
| One-host/global-array checkpoint or row-count resume presented as full continuation | Content/identity-bound loader/campaign/controller/pipeline state, complete atomic publication and qualified distributed restore. |
| Source name/shape/count used as scientific identity | Exact architecture, parameter, units/basis, measure, conditioning, material/source and state identity, with canonical ordering. |

Maintain an old-to-new contract map for materially refactored tests. Preserve independent failure meanings, intermediate states, status/work/evidence and rollback. Delete implementation-spelling/default-wording/source-text tests rather than re-pin them. Do not reduce item counts with loops/soft assertions or call a source-provider echo an independent oracle.

### Minimal conservative verification matrix

Select from the actual delta since the latest merge into `dev`, not just the last edited file. The paths below are a seed matrix; refresh references and semantic consumers during the cutover. Do not run the full repository suite unless global collection/configuration, shared fixtures or a comparably cross-cutting change actually requires it.

| Change owner | Required focused/integration coverage | Unaffected scope not automatically included |
|---|---|---|
| Shared atomistic context/state/units/requests/artifacts | Relevant existing and new `tests/unit/atomistic`, `tests/integration/atomistic`, source interchange, typing/declaration/role/restore tests; actual chemistry, protein/nucleic, coarse-graining, driven-flow and free-energy consumers | Unrelated finance, imaging or PDE families that do not consume changed shared primitives |
| Image/search/sparse/segmented/spline owners | Owning particle/spatial/splat/sparse tests plus every changed consumer family (graph, meshfree, particle-grid, electrostatics, learned atomistics and other direct users) | Unchanged metrics/providers/routes merely sharing a broad package name |
| Native model architectures/parameter maps | Family math/symmetry/regularity/derivatives, source fidelity, parameter-role/artifact tests, native training/dynamics/deployment; existing O(3) users if those owners change | A model-specific new file does not by itself require every unrelated neural architecture |
| Shared training/data/loss scale/selection | Kernel/objective/data-plane tests and all frontends importing changed helpers, especially operator precision/loader/selection/adaptation, atomistic fits and active learning | Unchanged training clients only when their inputs/contracts remain genuinely unaffected |
| Optimizer/constraint/linear helper changes | Owning method tests plus chemistry relaxation/path and actual other consumers; status, support/rank and sensitivity | No broad linalg suite when only a domain caller changes and the native owner is unchanged |
| Bias/CV/multistate changes | Sampling/free-energy/controlled-switching, learned biases, biological CV users, checkpoint/measure/status cases | Unrelated UQ estimators if no shared estimator owner changes |
| Generation/crystal changes | Native domain tests, biological generation regression, exact symmetry/chemical data, new conformer/packing/equivalence workflows and storage continuation | Existing crystal plasticity/magnetism code only if it adopts changed shared operations |
| Generic execution/lifecycle/distributed changes | Existing workset/pool/nonlinear/cardiovascular users; collective effects/leases, distributed spatial/spectral, data groups, shard restore/publication and exact failed-rank scenarios | No hardware/topology claim inferred from passing CPU or lane tests |

Each coherent implementation slice follows: configured formatting → changed-path direct runtime smoke → focused regression nodes → changed-surface lint/static/selector/contract/declaration/import audits → only then the conservative broader affected matrix. A confirmed reported failure is not rerun merely to confirm it. Exact failing nodes are rerun after a fix; statistical failures are investigated, not retried until green.

Run ordinary affected tests with `-n auto` in **one globally coordinated worker pool**. Forced or physical topology, real source providers, native builds, distributed processes and accelerator qualification use separate dedicated invocations. Benchmarks run without competing tests, builds or agent execution. Keep one isolated build directory per native owner if any compiled extension becomes necessary; no new FFI dependency is presumed by this plan.

Required planned commands, executed only during implementation at the appropriate wave:

```text
ruff format <changed first-party Python files>
ruff check <changed first-party Python files>
python -m pytest -n auto <selected affected test files or node IDs>
python tools/check_typing.py check
python tools/audit_selectors.py
python tools/audit_contract_candidates.py --signatures
python tools/check_import_boundaries.py
python -m tools.check_installed_typing
```

Angle-bracket arguments above denote the concrete touched-file/node inventory chosen for that wave, not commands claimed to have run. The current installed-typing tool defaults to Python 3.13, matching the current package support metadata; do not fabricate older-version passes or change Python support merely to run an obsolete matrix. Run the declaration/role/strict-construction tests located in `tests/unit/test_contract_declarations.py`, `test_role_declarations.py`, `test_strict.py` and relevant typing fixtures when their contracts change. Use only pinned ty and Ruff annotation checks, not another static checker.

Host-sync and memory audits are read-only candidate inventories. Run `python tools/audit_host_sync.py --category traced-body` and the existing memory audit with a touched-owner/root selection, then justify remaining bounded/static/host-topology/output-size cases. Confirm actual hot-path behavior with native transfer guards/profiling and compiler/runtime memory evidence. Measure touched cyclomatic/cognitive complexity, NLOC and parameter counts before/after; decompose by validation/preparation/execution/commit/evidence ownership, not arbitrary line cuts. Do not refactor source-faithful straight-line formulas solely to improve a metric.

### End-to-end acceptance matrix

Each scenario has an executable native smoke plus owning behavioral/scientific tests. Source-provider comparisons are isolated optional reference executions, never part of product execution. A missing required provider artifact/hardware/rights gate leaves the relevant row unqualified, not silently skipped as complete.

| Scenario | Required observable proof | Main scope rows |
|---|---|---|
| A01 mixed resources | Classical MIC interaction plus learned multi-image graph and coordination/reciprocal needs use independently valid resources; wrapping/deformation preserves geometry; invalid role refuses before state commit | C01, C04, C06–C08, C21 |
| A02 native family fidelity | Actual MACE/AIMNet2/UMA source-supported configurations, conditions and outputs; parameter-direction/derivative parity; source artifact converted and restored in a fresh environment without numerical source packages | C02, C03, C14 |
| A03 complete cell dynamics | Learned and mixed programs under admitted NPT/NPH/thermostats; correct pressure/strain/measure/controller state; capacity/constraint/scientific failure separation; full restart | C09, C21 |
| A04 relaxation and paths | Native coordinate/cell optimization and periodic fixed-cell NEB reach the independently known minimum/saddle with valid projected/full residuals, image branch and one-iterate output consistency | C10 |
| A05 heterogeneous stages | Different atom/edge/method-state sizes, whole path groups, reprime, partial convergence/failure, parked output/backpressure, refill and exactly-once publication | C11, C16, C21 |
| A06 streaming training | Mixed finite/periodic partial-label sources, actual budgets/staging, proper global normalization, freezing/adaptation/EMA selection, failed-update rollback and mid-epoch/accumulation continuation | C14, C16 |
| A07 distillation | Native teacher E/F/S/embeddings/HVP, actual mixed-order student update, replay and on-policy state; distribution objective with admitted measure/generator normalization, not atom-count identity | C15, C18 |
| A08 enhanced sampling | Metric/image-correct bias plus cell response, correct ABF sign/measure, history ownership/cadence, bias-aware exchange, freeze-to-analysis and restart | C17, C18 |
| A09 generation/CSP | Native conditional sampler and stateful stream, graph-to-conformer where required, exact general-position setting/ASU, multi-image contact relief, physical relaxation, bounded confirmed dedup and same-definition ranking | C19, C20 |
| A10 spatial strategies | Both halo and replicated-geometry graph partition execute native models with exact owner/ghost/context adjoints, per-term resources and failure evidence | C12, C13 |
| A11 reciprocal and combined meshes | Actual distributed spectral/particle transfers, point/multipole/dispersion channels, system reductions, DP×domain and stage×domain correctness where declared | C06–C08, C12, C13 |
| A12 crash and topology continuation | Controlled failure during computation, labeling, output, checkpoint and transfer never publishes partial state; same-profile replay and explicit changed-topology migration preserve the correct distinct guarantees | C03, C11, C13, C16, C21 |

Scientific references must be independent of implementation code where they could share a plausible defect. Use analytic finite systems, independently converged lattice/Ewald sums under the same convention, finite differences/adjoint identities, known equilibrium measures and exact geometric/group controls. Source-framework parity adds fidelity evidence but is not by itself chemical accuracy, a correct ensemble, or proof of the reference's own scientific assumptions. Include deliberate-fault adequacy for new shared test helpers when multiple families rely on them.

### Performance and superiority acceptance

Predeclare representative workloads and evidence thresholds before tuning. Match model/weights/physical correction, CN/cutoff/PBC/background definition, requested outputs, derivative order, dtype/precision, tolerance and source data. Keep source-framework environment conflicts isolated. Native-versus-source comparison includes required preparation, conversion, topology rebuild, communication and publication costs at their declared cadence; a static preallocated kernel benchmark is not an end-to-end MD/training result.

Record at least:

- immutable host preparation and numerical binding/refresh;
- lowering, compilation and first execution separately from warmed execution;
- graph/mesh/history/optimizer persistent bytes, requested outputs and scratch/replay/cotangents;
- compiler temporary/output/generated-code bytes and observed process/device allocator peaks separately;
- transfers/synchronizations, halo and reciprocal work, rebuild/migration and checkpoint/restore;
- complete accepted MD steps, converged relaxations/paths, accepted training updates, qualified generated candidates and successful campaign throughput;
- refused, failed, OOM and exhausted-budget rows, not just successful plot points.

Sweep controlling capacity, not only one small example: atoms/density/neighbor skew/images, heterogeneous batch and group sizes, cutoff tiers, channels/angular degree/depth, mesh/order, species/rank/chunk, history/replica/state count, optimizer memory, dataset/prefetch/staging and device/process/host count. Strong and weak scaling are different campaigns. Keep cold and warm cache policy fixed and recorded. Donation is allowed only for explicitly owned scratch/continuation buffers; public caller-owned state remains reusable unless donation is part of its contract.

A capability may be numerically complete before its acceleration profile is recommended. Keep the portable native route, but never execute it under an accelerated method name. Performance promotion requires observed workload-level benefit or a clearly reported accuracy/memory/throughput tradeoff; do not substitute larger feature count or theoretical asymptotics for measurement. Scientific and release promotion remain separate.

### Documentation, examples and deterministic generated outputs

Authored docs change with their implementation after direct smoke proof. The central update inventory includes:

- `docs/guides_atomistic.md`, `guides_atomistic_dynamics.md`, `guides_atomistic_force_fields.md`, `guides_atomistic_polarization.md`, `guides_atomistic_interop.md`, `guides_atomistic_sampling.md`, `guides_atomistic_uncertainty.md`, `guides_atomistic_active_learning.md`, `guides_atomistic_distributed_execution.md`, `guides_mace_execution.md`, and affected biological generation/chemistry guides;
- `docs/api/atomistic.md`, `atomistic_advanced.md`, `atomistic_force_fields.md`, `atomistic_interop.md`, `atomistic_sampling.md`, `chemistry.md`, `optim.md`, `execution.md`, `linalg.md`, relevant discretization/sparse/nn API pages and new crystal/model interfaces;
- matching cookbook pages and existing `examples/atomistic_*.py`, plus the new model, streaming, distillation, relaxation, generation and crystal examples named in each phase;
- **NEW** authored model/training-data/distillation/generation/crystal/dispersion guides, with exact current capability limits and no wording that treats source examples or random teachers as accuracy evidence;
- `mkdocs.yml` navigation for new public guides/API pages; `CHANGELOG.md` for the clean cutovers and intentional representation/identity changes; `NOTICE` and admitted `LICENSES` entries for actual reuse.

Examples use public canonical imports and explicit supported policies; no private convenience re-exports, automatic weights/data downloads, placeholder potentials or silent fallbacks. Run the changed examples as actual programs, inspect their outputs/status and build the affected documentation with its configured toolchain when the public surface is stable. Historical benchmark records remain immutable evidence; new source revisions require new measured records rather than hand-edited old successes.

After the last implementation or requested merge change, centrally regenerate the actual owners:

```text
python tools/generate_public_api_manifest.py --output docs/data/public_api.json
python tools/generate_capability_inventory.py
python tools/check_public_api_manifest.py
python tools/check_capability_consistency.py
python tools/check_import_boundaries.py
```

The current capability generator writes `docs/data/capabilities.json`, `docs/api/capabilities.md`, `docs/data/application_portfolios.json`, `docs/data/capability_closure.json`, `docs/data/source_absorption.json` and `docs/api/capability_closure.md`. These existing artifact names remain canonical; no second native-capability registry or schema generation is added. Public module discovery should use existing lazy facade traversal; extend `EXPLICIT_PUBLIC_MODULES` only if the new public subpackage is genuinely unreachable through that canonical traversal.

The proposed `tools/generate_space_group_tables.py` gets a real admitted-source `--source`, `--output` and `--check` contract when implemented. Generate canonical symmetry data only after source/rights/setting decisions are fixed; do not hand-format generated tables. Update runtime-contract TSV inventories only through their owning documented generator if affected, after locating its actual invocation; do not invent a generator or edit audit manifests by hand.

### Explicit prerequisite and nonclaim register

| Prerequisite / boundary | Required handling |
|---|---|
| Pretrained weight digests, access and rights | Obtain exact admitted files/terms; source code licenses do not license weights. Architecture/random-fixture proof does not close named pretrained fidelity or scientific accuracy rows. |
| D3, radius/volume/prior and crystallographic data rights | Record exact source, units, transforms, digest and permission before bundling. Caller-supplied admitted data is explicit; no automatic fallback/download. |
| Optimized/compiled source model layouts | Recover complete semantics and a real source-parameter map. An opaque executable or lost parameterization is not converted by wrapping it; keep that row blocked until a lawful reconstructible source is available. |
| Higher-order/regularity boundaries | Qualify each coordinate/strain/parameter pattern and join/branch domain. Do not call epsilon underflow exact smoothness or attach detached forces as higher derivatives. |
| Multi-host/device/transport availability | Require observed physical hardware and actual provider/collective execution; emulator/lane/reference results are separately labeled. Missing access leaves qualification blocked, not native implementation replaced by a mock. |
| Scientific validation and release authorization | Independent physical/data campaigns and exact signed support profiles are still required. This plan cannot promote itself. |
| Extra physics not in the pinned scope | No implicit ATM, arbitrary octupoles, arbitrary adaptive reweighting, variable-cell NEB, universal reactive chemistry, or finite-temperature crystal-stability claim. Richer modes explicitly described in this plan have their own complete implementation/qualification gates. |
| Arbitrary foreign programs | Native extension ports do not execute arbitrary Torch/Python code. Exact import of a named source architecture and native custom-model extensibility are separate obligations. |

### Definition of done

All C01–C22 rows must have a completed native workflow or a specifically identified external prerequisite that prevents its qualification; no unimplemented numerical path can be hidden behind a provider, fallback, placeholder or broad success flag. Every supported profile names its exact model/method/geometry/precision/derivative/resource/distribution/lifecycle envelope. All affected callers, typed signatures, recipes, tests, fixtures, docs, examples and generated artifacts have undergone the clean cutover. Required direct smokes, focused/integrated tests, scientific references and phase-separated performance campaigns have actual retained evidence under the final relevant tree. Implementation, qualification and release dispositions are reported separately, and any remaining prerequisite is stated rather than silently shrinking the user's goal.

For this planning request, completion is the self-contained file-by-file document with these decisions, dependencies, tests, benchmarks, migration and evidence gates. No implementation or runtime qualification is claimed by writing it.
