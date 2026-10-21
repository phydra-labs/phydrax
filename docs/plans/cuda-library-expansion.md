# Selective CUDA library expansion: implementation plan

## 0. Status, scope, and baseline

**Status: closed at the user's request with peer-dependent qualification blocked.**
The implementation and passing observations are preserved; this is not a claim
that all eight-GPU workflows are qualified. No host-staging workaround was added.
The original planning task was read-only. The subsequent `//code` instruction created the
`accelerator-capabilities` worktree from updated `dev` commit
`b42e720f0`; no implementation edits were made on the parent checkout.
The file maps below are the design baseline, not claims that every proposed
filename must exist independently of its canonical owner.

Original planning baseline: `dc39e1bd8c57229e95635da2fa490f589857241b`.
Implementation retains the intervening B300 qualification, cuDSS failed-refresh
recovery, and explicit CPU-extra work. The current runtime is CPython 3.13.15,
JAX/JAXlib/CUDA plugin 0.12.0, driver 595.91.07, and eight A40 GPUs (sm86).
Historical H100/H200 and single-device B300/Spark records do not qualify this
host or the new representations.

### 0.0 Implementation reconciliation

- The core companion contains separate small optional wrappers. cuEST's C
  redistributable required a separate optional `phydrax-cuest` runtime wheel;
  the vendor Python wheel is not a substitute for that C-library closure.
- cuEST geometry is explicitly bound and reused through SCF and associated
  forces. Its dynamic content certificate and guarded recipe replace repeated
  full-recipe transfers; native registry ownership remains canonical.
- Real tiny-molecule execution exposed a singular DIIS Gram solve. The owning
  SCF implementation now uses native affine-constrained minimum-norm least
  squares and retains the rank/constraint result, without jitter or silently
  disabling acceleration.
- Vendor collectives require explicit, owned staged programs and authenticated
  site-MPI supervision. A source-proven JAX 0.12.0 cached-AOT ordered-token
  defect required a narrowly owned dispatch boundary over the same compiled
  executable, with exact-release admission and provenance. No global JAX patch,
  fake token, alternative executable, or ordering relaxation is used.
- Initial peer-integrity failures affected outgoing transfers from physical
  devices 1 and 2 in both CUDA and JAX; GPU1 subsequently became CUDA-unavailable
  with a reported reset requirement. Later read-only discovery again found eight
  devices, and all 1,008 directed copy observations and the complete physical
  process/device matrix passed. No host changes were performed by the agent;
  recovery's cause is not established, and the initial failures remain evidence.
- Full-device installed execution exposed AmgX's process-global device-memory
  pool reuse. Its pinned integration now owns pools, allocation bookkeeping,
  thread streams, and library handles per device, with owning-device shutdown.
- Runtime evidence remains scope-specific: device-local mathematics, host-staged
  slices, same-host collectives, and multi-host execution are distinct claims.
  Candidate catalog entries are not automatically promoted by a small benchmark.

### 0.0.1 Retained verification scope

`benchmarks/a40_qualification.json` records hardware identities, artifact
identities, scientific results, and limitations. Two-, four-, and eight-GPU
cuDSS MG/MGMN and cuDecomp routes passed. Device-local installed qualification
is recorded separately from peer integrity and communication qualification.

The installed core and all eight per-device local/hostile optional-provider
profiles passed. The cuTENSOR multi-GPU host-slice gate remains failed: reversed
GPU1/GPU0 placement lost GPU1's complete contribution in float32 and float64.
A later standalone program importing only JAX/NumPy reproduced incorrect
GPU0-to-GPU1 peer and forced-copy transfers at five sizes; explicit host
transport passed. The precise driver/JAX/hardware cause is not established.
Passing reruns, extra waits, or sanitizer-only observations do not resolve it.

The user selected **“Stop with blocked qualification.”** Further debugging is
stopped. Implementation, diagnostic scripts, and raw evidence remain in the
worktree. No silent transport fallback, tolerance change, or host reconfiguration
was applied. The numerical-consumer record also preserves the original
3600-second campaign timeout: 19 call-phase passes plus one independently
completed final FMM case are not relabeled as a successful original process.

Isolated capacity measurements cover five tensor path/executor combinations
at two sizes, local/hybrid/MG/MGMN sparse solves, equal-topology native and
cuDecomp slabs, and six cuEST s/p/d molecular cases spanning 4–40 orbital and
10–500 auxiliary functions. The cuEST energy and analytic-force comparisons
use independently matched density fitting, not exact HF as a substituted
oracle. These bounded measurements do not establish a general speedup.

Unknown custom-call FLOP and traffic totals remain unknown even when XLA's
aggregate is positive; raw compiler estimates, known executable storage,
native workspace bounds, and sampled process memory remain distinct.

### 0.0.2 Complexity ownership

The verification record retains AST NLOC/physical-line/parameter measurements,
repository-pinned Ruff C901, and `cognitive-complexity` measurements against
`b42e720f0`. The coupled-launch controller was decomposed into launch,
per-rank terminal aggregation, and local scientific acceptance. Unknown exit
codes and supervisor failures cannot be overwritten by successful rank records.

The following long functions retain explicit, bounded responsibilities rather
than being split solely to reduce line counts:

| Symbol | Retained responsibility |
| --- | --- |
| `cuda_candidate_profiles` | Deterministically ordered profile declarations, not numerical control flow. |
| Molecular HF unrestricted/ROHF and KS unrestricted solve methods | Existing method-specific SCF formulas and iteration order; the cutover adds integral-definition provenance without rearranging their numerical kernels. |
| `DistributedSpectralExecutionPlan.__init__` | One admission transaction couples topology, layouts, precision, resource limits, provider binding, and canonical identity before assignment; the explicit controls do not introduce a parallel policy object. |
| `linalg._runtime._solve_prepared` | Existing exhaustive method execution and status/result construction; the narrow distributed-RHS changes preserve method ordering, work evidence, and source-sensitive numerical routes. |
| `build_contraction_schedule` | One ordered SSA scan owns live operands, shape propagation, cost accounting, and refusal ordering, including unary steps. |
| `measure_cuest_kernels` and distributed spectral `_run` | Explicit buffer/reference/cache lifetimes span timing and memory snapshots; splitting those scopes would obscure which allocations remain live. |
| `run_mgmn_cases` | One coupled lease lifecycle retains intermediate solve, derivative, refresh, retirement, and reconstruction evidence with fail-fast checkpoints. |

### 0.1 Objective

Add optional, explicitly selected numerical implementations beneath existing Phydrax scientific owners. A candidate must provide a missing capability, a demonstrably larger solvable problem, or a repeatable end-to-end advantage at the same admitted scientific accuracy. Installing a library is not support, and an upstream performance claim is not Phydrax evidence.

The primary implementation tracks are:

1. Eight-A40 baseline qualification and fail-closed qualification tooling.
2. Shared optional CUDA packaging, dependency attribution, resource, and lifecycle contracts.
3. cuTENSOR execution and cuTensorNet fixed-path execution/optional path planning through native prepared tensor contractions.
4. cuDSS memory admission, single-process multi-GPU execution, and separately qualified rank-local distributed execution.
5. cuEST molecular integral execution through native mean-field chemistry, completing density-fitted RHF energy and same-energy analytic nuclear forces.
6. A narrow cuDecomp distributed transpose provider, selected instead of adding cuFFTMp at the same time.

Conditional tracks have their own file maps and admission gates: cuDensityMat, cuOpt, cuSOLVERMp, and MathDx. They are not mandatory dependencies or implicit additions to the primary delivery. They start only when their stated workload and compatibility gates pass and that track is selected for implementation. No empty backend, unimplemented exported method, or candidate catalog entry is an implementation substitute.

A failed prerequisite blocks the corresponding primary support claim; it does not authorize silently replacing the requested provider, dropping its accepted scenario, or advertising a smaller successful subset as the original deliverable. Record the blocker and continue independent reachable tracks. Changes to an agreed implementation scope require explicit approval.

### 0.2 Workspace and impact

- A subsequent `//code` starts a fresh worktree from the then-current branch under `../phydra-labs/.worktrees/`, with a neutral name such as `accelerator-capabilities`. Copy the authoritative ignored `AGENTS.md` before work. `//code--` explicitly stays in the current worktree.
- Preserve concurrent changes. Re-establish the actual source/dependency baseline and the latest merge into `dev` once; use that comparison point for affected-test selection.
- No merge, commit, publication, system-driver change, IOMMU/ACS change, or package downgrade is authorized by this plan.
- Impact classes: optional packaging/build, native runtime, shared execution, public preparation contracts, scientific approximation/derivatives, identity/persistence, qualification, and performance.
- Language-server navigation was unavailable during planning. At implementation time use LSP references when configured; otherwise enumerate direct imports/calls, registries, persistence, generated manifests, and scientific consumers before each public cutover.
- `NEW` in the file maps denotes a proposed path, not an existing file. Other listed paths were located in this checkout. Proposed API names are design names, not claims of existing functionality.

## 1. Binding design decisions

### 1.1 Canonical owners and unchanged defaults

| Capability | Canonical Phydrax owner | Selected boundary |
| --- | --- | --- |
| Dense tensor network contraction | `phydrax.tensor_network` | Prepared executor and independently selected path planner |
| Ordinary contractions and matrix products | `phydrax.ein.contract`, ordinary `@` | Intentionally unchanged; no global interception |
| Sparse direct solves | `phydrax.linalg` | Existing plan/template/bind/solve/refresh/release lifecycle |
| Distributed ownership and collectives | Existing execution substrate | Explicit group-bound live communicator leases, only where required |
| Fourier transforms and redistribution | `phydrax.discretization.spectral` | Explicit transpose provider; local FFT and numerical conventions stay native |
| Molecular Gaussian integrals | `phydrax.operators.quantum.gaussian` | Prepared one-electron/J-K/contracted-gradient implementation |
| Electronic SCF and forces | `phydrax.chemistry.electronic_structure` | Native occupations, convergence, energy, stationary-force admission, and results |
| Conditional open quantum dynamics | Quantum generator-action and differential owners | Operator action only, not a new integrator or CPTP certifier |
| Conditional optimization | `phydrax.optim` | Existing convex/mixed-integer lifecycle and independent audits |
| Conditional fused small kernels | `phydrax.linalg` and existing consumer | One profiled specialization, not a second kernel framework |

No automatic CPU fallback, provider switching, densification, TF32 enablement, precision narrowing, RNG replacement, or loss of status/evidence. An unavailable explicit selection raises the owning capability error. Portable native implementations remain genuine selectable implementations, not compatibility aliases.

### 1.2 Numerical and transformation contracts

- Preserve x64 and highest float32 contraction defaults. Resolve storage, multiply, accumulation, decision, and output precision explicitly; FP64 buffers do not prove FP64 internal computation.
- Static metadata contains topology, layouts, policies, and immutable execution descriptions. Numerical arrays remain dynamic leaves. No device pointers, communicator addresses, callbacks containing arrays, or live native handles in durable fingerprints or static PyTree fields.
- Preparation and tuning are bounded host phases. Execution reuses prepared structure. No planner, JIT-wrapper construction, host hashing, or accidental per-iteration preparation in numerical loops.
- Every advertised transformation must work compositionally. FFI registration alone does not supply JVP/VJP/batching; `custom_vjp` alone does not supply forward mode or higher derivatives.
- Analytic forces, mathematical solve sensitivity, arbitrary autodiff, and algorithmic derivatives are separate capabilities. Unsupported compositions fail before entering an unsafe provider path.
- Keep ordinary contraction, adjoint, complex transpose, FFT sign/normalization, and scientific axis conventions distinct. Equal dimensions are not identity.
- New Strict classes follow existing `StrictModule`, nominal typing, runtime-resolvable annotations, `checked` boundaries, and constructor/reconstruction validation rules. Closed selectors have one runtime declaration and native parsing. No new blanket annotation enforcement or generic provider-discovery framework.

### 1.3 Identity and persistence cutovers

Distinguish mathematical structure, scientific approximation, prepared numerical binding, and execution identity:

- Provider/build, compute algorithm, derivative lowering, communicator/rank map, workspace policy, and transpose schedule belong to execution identity.
- Density-fitting basis/metric/conditioning, screening, normalization, range separation, and precision changes belong to the corresponding scientific/numerical definition.
- New prepared fields change PyTree structure and downstream IDs. Reprepare affected execution artifacts; do not reuse old IDs or add schema generations/compatibility aliases.
- Preserve mathematical structure IDs only when the structure is actually unchanged. Native source-addressed callables may also change identity after edits; inspect their compiler/checkpoint consumers rather than calling a refactor identity-preserving.
- Do not change the current `phydrax.ein.contract` callable. It is a direct `opt_einsum.contract` export; the new tensor route does not require replacing it.
- Serialize portable scientific arrays and reconstruction recipes, never native handles. Restoration validates complete values and re-admits the selected implementation. A missing provider is a refusal, not permission to substitute one.

### 1.4 Memory and lifecycle

Every candidate reports logical retained bytes, XLA compiler estimates, external device scratch/cache/resident allocations, host/pinned storage, communication buffers, and observed samples separately. A sample is not an allocation bound. Unknown external usage is unknown, not zero.

Admission must cover preparation/analysis, tuning, forward/derivative execution, refresh with overlapping generations, eviction with in-flight leases, reconstruction, and teardown. A cache-entry count alone is not a byte budget. Do not count one host's available RAM once per rank or treat eight device memories as a single interchangeable heap.

Reuse the existing bounded registry for local reconstructible memos. Keep shared ownership while work is in flight. Drain all vendor work that may reference public buffers before publishing completion, including error paths. Releasing a memo does not invalidate immutable input recipes. Distributed reconstruction and retirement require all participating ranks to agree; they cannot occur independently inside a local LRU destructor.

## 2. Gate A: eight-A40 baseline and truthful qualification

### 2.1 Required observed baseline

Before evaluating new libraries:

1. Record all eight GPU UUIDs, visible/local ordinal mapping, compute capability, driver/runtime/plugin versions, physical memory, current allocations, CPU/NUMA topology, and peer connectivity. Retain exact commands and source/environment identities.
2. Use a fresh Python 3.13/current-JAX CUDA13 environment and an audited sm86-capable companion. Generate the checkout's ignored lockfile before benchmark identity capture; do not reuse another machine's lock or wheel without verifying it.
3. Check contents of all 56 directed peer transfers using both CUDA-runtime and JAX transfers. Use exact distinguishable patterns, changed inputs, bounded size sweeps, and nonzero device subsets. Capability reports alone do not pass this gate.
4. Exercise native JAX collectives and the weighted-reduction/PCG smoke at 1/2/4/8 GPUs and process layouts 1x8, 2x4, 4x2, 8x1. Require disjoint ownership, uniform group IDs, numerical agreement, and clean exit.
5. Run core installed-wheel provider qualification with `--required-gpus 8`. Distinguish concurrent device-local cuDSS, serialized in-process AmgX, and independent-process QUDA. Sequential QUDA per-device coverage is not eight concurrent jobs or distributed QUDA.
6. Exercise native slabs `(8,)`, pencils `(2,4)` and `(4,2)`, native-JAX MACE owner comparisons, PIC migration/rollback/restart, and relevant FMM derivatives. Existing test parametrization does not expand merely because eight GPUs are visible.
7. Do not inherit another host's NCCL/UCX flags. Any necessary workaround is explicit environment identity and narrows the qualified transport claim. Peer corruption is a host blocker; disabling NCCL P2P is not a JAX peer-copy repair.

Compiled Pallas MACE currently refuses sm86. Preserve that refusal in this library-expansion baseline and report the route as unsupported, not passed-by-skip. A separate evidence-backed sm86 admission is not supplied by adding cuTENSOR or cuEST.

### 2.2 File-by-file baseline work

| File | Planned change |
| --- | --- |
| `tools/distributed_substrate_benchmark.py` | Gate finite errors and PCG convergence before successful exit; preserve complete result evidence. Validate requested process/device ownership and all-rank terminal agreement. Keep declared memory admission labeled declared. |
| `benchmarks/pic_distributed.py` | Make accepted physics, rejection reason, finite reference differences, and explicit route-specific scientific tolerances authoritative exit gates. Preserve rejected-step evidence; do not benchmark an unchanged accepted state as a successful step. |
| `tests/unit/test_distributed_execution_substrate.py` | Add distinct 1x8/2x4/4x2/8x1 GPU scenarios and nontrivial visibility subsets. Retain small CPU functional coverage without calling it GPU qualification. |
| `tests/unit/discretization/test_distributed_spectral.py` | Add eight-way slab and 2x4/4x2 pencil forward/inverse, padded-layout, complex adjoint, resource refusal, and composed-executable scenarios. Keep 2x3/3x2 tests: those expose different topology cases. |
| `tests/unit/atomistic/test_distributed_mace.py` | Add eight real owner devices with energy/force/strain agreement and explicit supported ingress. Do not imply multi-process MACE ingress from the single-process driver. |
| `tests/unit/test_distributed_scaling_benchmarks.py` | Extend owner selection and true physical-topology admission to eight; test numerical failure cannot become a successful campaign. |
| `tests/unit/solver/test_distributed_pic.py` | Extend selected reduced/spectral routes, overflow rollback and restart to eight; retain existing 2x2x2 cochain scenario as a separate contract. |
| `tests/benchmarks/test_pic_distributed.py` | Add 1/2/4/8 sizing and same-science acceptance, plus a failing physics/reference scenario that makes the command fail. Do not add exact-default/source-spelling tests. |
| `tools/qualify_cuda_wheels.py` | Preserve required per-case/device completeness; route subprocesses through the existing bounded process helper; forward one explicit hostile-library-path contract to audit and lifecycle children. Require actual competing-library evidence before labeling a run hostile. |
| `tests/integration/test_cuda_native_lifecycle.py` | Consume the qualifier's explicit hostile path rather than a different hardcoded directory; record challenge and mapped-library origins. Keep normal/hostile teardown independently collectable. |
| `tests/unit/test_cuda_wheel_qualification.py` | Cover eight-device completeness, mandatory skip/missing/duplicate refusal, custom hostile-path propagation, and absent-challenge nonqualification. |
| NEW `tools/qualify_cuda_execution.py` | One bounded physical-topology campaign coordinating existing workers/drivers. Validate requested scenarios, rank/device UUIDs, numerical results and completion; produce failed evidence on timeout/partial runs. Reuse `tools/_accelerator_process.py`, not another subprocess framework. |
| NEW `tests/_support/cuda_execution_worker.py` | Focused physical peer-copy/collective worker; explicit launch topology, size/case IDs, changed inputs, numerical oracles and bounded outputs. No import-time device discovery. |
| NEW `tests/unit/test_cuda_execution_qualification.py` | Consumer-visible qualification refusal for missing ranks/pairs/scenarios, incorrect contents, forced CPU topology, timeout and narrowed workload. Real GPU evidence is separate from synthetic parser fixtures. |
| `benchmarks/_runtime.py` | Extend environment/provider evidence only for newly material settings and actual dependency identities. Reuse phase timing, compiler memory and source-build fingerprints. |
| `docs/installation.md` | Add the observed A40 profile only after execution; separate peer integrity, native collectives, native providers, optional MPI, unsupported kernels, and performance. |
| `.gitignore` | Admit only completed canonical benchmark/qualification records selected for retention. |
| NEW `benchmarks/a40_qualification.json` | Generated from observed execution, never prefilled as a planned pass. Retain failed profiles and limits alongside successful scopes. |

The current local launcher waits without a whole-job deadline. This campaign must supervise the entire launcher process group through `tools/_accelerator_process.py`; do not add an unrelated launcher redesign merely to run qualification. Vendor distributed execution later needs its own coupled-failure contract in the execution owner.

### 2.3 Baseline acceptance

No successful qualification with an unobserved mandatory case, swallowed exception, skipped required hardware test, incorrect finite result, orphaned child, or timeout. Optional absent providers remain explicitly absent; their absence must not contaminate core-provider results. No multi-host, RDMA, zero-host-copy, or eight-way scaling claim from one-host correctness.

## 3. Gate B: optional provider packaging and common runtime

### 3.1 Distribution choice

Keep `phydrax[cuda13]` and the core cuDSS/AmgX/QUDA runtime dependency set intact. Ship small additional Phydrax wrapper DSOs separately within the existing lockstep companion where portable binary packaging is feasible. Never link new optional vendor libraries into common code or an existing core component.

Use canonical provider extras for optional vendor runtimes, beginning with `cutensor`, `cutensornet`, and `cuest`. Document composed installation such as `phydrax[cuda13,cutensornet]`; the provider extra installs its complete provider dependency closure, including cuTENSOR where needed, while `cuda13` supplies accelerator intent/plugin/core companion. No all-libraries extra. Missing CUDA setup is an explicit availability error, not successful GPU installation. Conditional extras are added only with their completed provider track.

Resolve actual CUDA13 distribution names, wheel contents, C headers, supported architectures, licenses, SONAMEs, minimum runtime patch, and Python/JAX compatibility before fixing dependency pins. Use one qualified pin set in the lock/build record; no speculative version pins in this plan and no relaxing constraints to force installation. Inspect both Linux x86-64 and aarch64 availability. An unsupported optional provider must not remove the core ARM companion or become a successful no-op install.

cuDecomp and cuDSS communication shims may depend on a site MPI ABI. They are explicit site-build artifacts using the canonical loader/identity validation; do not advertise them as universal companion-wheel components or bundle arbitrary MPI. A site component requires an exact supplied prefix/ABI, build manifest, hash, and qualification. Keep site components out of portable release defaults until their deployment contract is actually portable.

### 3.2 Shared file map

| File | Planned change |
| --- | --- |
| `pyproject.toml` | Add only completed optional provider extras and truthful optional typing boundaries. Preserve core CUDA13/Python/JAX constraints and mutually exclusive accelerator environments. |
| `native/cuda13/pyproject.toml` | Add qualified header/build dependencies for portable selected wrappers without adding their optional vendor libraries to core runtime requirements. Preserve wheel-only and lockstep release contract. |
| `native/cuda13/_build_backend.py` | If component-scoped build requirements are necessary, resolve them once from explicit build selection through the existing scikit-build hooks; retain sdist refusal. No new build framework. |
| `native/cuda/CMakeLists.txt` | Extend explicit component selection and source/build provenance. Portable release selection contains its declared completed wrappers; site-MPI components require explicit opt-in. Keep CUDA runtime shared. |
| `native/cuda/cmake/PhxPipCudaToolkit.cmake` | Recognize pinned optional vendor RECORD layouts, including packages not prefixed `nvidia-`. Discover headers/library directories from verified distribution records; preserve no-system-toolkit and no-site-packages-mutation rules. |
| `native/cuda/cmake/PhxCudaComponent.cmake` | Reuse separate DSO construction/exports/RUNPATH. Extend only for a real artifact requirement; no cross-component vendor linkage. |
| `native/cuda13/python/phydrax_cuda13/__init__.py` | Extend component lookup for packaged wrappers without loading DSOs. Missing selected component remains an error. |
| `phydrax/_cuda_native.py` | Extend canonical component/handler/dependency tables, exact identity validation and actionable reinstall hints. Load only the selected closure. Preserve device partitioning, bounded reconstructible memos, and explicit completion/cleanup. |
| `phydrax/_cuda_environment.py` | Attribute optional NVIDIA libraries whose distribution names are not `nvidia-*`; extend known SONAME families and detect CUDA-major contamination by RECORD ownership, not library version numbers. Keep static inspection device-inert. |
| `phydrax/_elf_dependencies.py` | Reuse dependency resolution; change only if qualified optional/site dependencies expose an unsupported real ELF requirement. Never weaken unresolved/foreign dependency refusal. |
| `phydrax/_config_types.py` | Extend `LibrarySetting` only for implemented component overrides; runtime-resolvable closed selectors. |
| `phydrax/_config_environment.py` | Add matching `ProviderConfig.library_paths` environment mappings with existing startup/freeze behavior. |
| `phydrax/backends/_types.py` | Reuse availability/distribution/transfer evidence. Add a field only for a demonstrated missing invariant, with explicit PyTree/API migration. Do not use one boolean to claim all JAX transformations. |
| `phydrax/backends/__init__.py` | Add lazy canonical exports for completed availability/capability owners; no vendor imports on facade import. |
| `native/cuda/src/common/phx_cuda_common.hpp` | Reuse local registry and RAII ownership. Extract genuinely shared completion/resource helpers only when multiple new providers use the same invariant. |
| `native/cuda/src/common/capi.cpp` | Preserve exact C ABI/status behavior. If new common evidence requires an ABI change, cut over every component and loader together, without a versioned parallel ABI. |
| `native/cuda/include/phydrax_cuda.h` | Match any justified common ABI cutover; no pointer/communicator serialization. |
| `tools/build_cuda_wheels.py` | Audit chosen component set, optional dependency closure, shared cudart, native ELF architecture, RUNPATH, and actual device code. Replace the cuDSS-only host-wrapper exception with an explicit per-component classification; a host wrapper does not qualify vendor SASS. |
| `tools/check_cuda_extras.py` | Check core-only and composed optional installs, non-`nvidia-*` CUDA distributions, foreign-major refusal, and missing vendor/companion artifacts. Test installed dependency behavior, not just METADATA string spelling. |
| `tools/qualify_cuda_wheels.py` | Add explicit component/profile selection with unchanged core mandatory cases. Selecting a provider makes all its required scientific/lifecycle cases mandatory; no broad skip-text heuristic as the only completeness gate. |
| `tests/unit/test_cuda_environment.py` | Optional wheel attribution, mixed-major, missing dependency, duplicate ownership and inert inspection scenarios. |
| `tests/unit/test_cuda_native_loader.py` | Real loader contract: selected-only dependency loading, exact handler/version validation, missing optional runtime isolation, release/rebuild and invalid artifact refusal. |
| `tests/unit/test_cuda_build_tools.py` | Component set, optional closure, header discovery, host-wrapper/device-code classification and architecture coverage. |
| `tests/unit/test_configuration_environment.py` | New explicit library settings and round-trip projection/freeze behavior. |
| `tests/integration/test_cuda_import_contract.py` | Core and CPU-only imports must not load or initialize new providers. An installed but unselected extra cannot alter execution. |
| `tests/integration/test_cuda_native_lifecycle.py` | Each completed new DSO: live state, queued work, release/rebuild, concurrent invocation, partial preparation failure, normal and actual hostile-path exit. |
| `tests/typing/cases/cuda_environment.py` | Truthful new public diagnostics/availability annotations if changed. |
| `native/cuda/README.md` | Portable versus site components, exact build/pin/artifact scope, optional installation, resource and teardown semantics. |
| `NOTICE` | License/provenance obligations from the exact shipped wrapper/source assets. Add license files only where the actual redistribution requires them. |

### 3.3 Runtime admission and evidence

A host-side descriptor cache may survive invocations; a numerical-intermediate cache must never trust a pointer, unchanged static `prepared_id`, or stale revision after transformed dynamic leaves change. Safe tensor default: invalidate numerical cache per invocation and retain descriptor reuse. Vendor backward storage must belong to its exact forward invocation.

Keep `cuda_native_resources()` scoped to loaded components. Existing count-then-index snapshots are not guaranteed coherent under concurrent native mutation; use quiescent observation for qualification. Do not advertise them as an atomic monitor or add unrelated telemetry. If admission later requires atomic byte snapshots, implement one common snapshot contract and migrate all consumers rather than silently relying on the present interface.

A site communicator library is pinned and validated before use. Never trust a process-global `CUDSS_COMM_LIB`, arbitrary `LD_LIBRARY_PATH`, or imported MPI from another ABI as sufficient provenance.

## 4. Track C: cuTENSOR and cuTensorNet through prepared contractions

### 4.1 Public preparation contract

Keep three independent concepts: mathematical network, path source, and executor.

- Extend `plan_contraction` with an explicit imported `path` or a typed `path_planner`; these are mutually exclusive with an explicitly requested native optimizer.
- Add a proposed `ContractionExecutionPolicy` accepted by `prepare_contraction`; provider values are `native`, `cutensor`, and `cutensornet`. Native remains the default, regardless of installed packages.
- Add a proposed `ContractionPathPlanner` for bounded cuTensorNet planning. Path provenance is not executor identity.
- `refresh_contraction` preserves admitted structure/execution choices but refreshes numerical binding and invalidates value-dependent cache eligibility.
- Host preparation/refresh must reject traced content-addressed input. The existing host array fingerprint path cannot establish current numerical identity inside tracing.
- Forward, reverse, and sliced execution must use the same selected executor; native low-level `execute_schedule` remains an explicitly native operation.

### 4.2 File map

| File | Planned change |
| --- | --- |
| `phydrax/ein/__init__.py` | Intentionally unchanged: preserve `contract` callable identity and ordinary API. |
| `phydrax/tensor_network/_contraction.py` | Extend `ContractionPlan`, `PreparedContraction`, planning/preparation/refresh, result evidence, and forward/reverse dispatch. Centralize path admission so every planner passes the same structural and resource checks. |
| NEW `phydrax/tensor_network/_contraction_provider.py` | Scientific execution policy, path-planner request, immutable executor descriptor, capability/resource admission and reconstruction metadata. No live handles or universal plugin registry. |
| `phydrax/tensor_network/_schedule.py` | Validate evolving-live-list path indices and terminal operations; separate native SSA schedule from provider lowering identity and derivative workspace evidence. |
| NEW `phydrax/tensor_network/_cutensor.py` | Lower admitted SSA operations to explicit mode/extents/strides/compute descriptors and stream-aware FFI. |
| NEW `native/cuda/src/cutensor/CMakeLists.txt` | Separate optional wrapper DSO through the canonical build helper. |
| NEW `native/cuda/src/cutensor/cutensor_ffi.cpp` | Descriptor preparation, bounded workspace, contraction/reduction/permutation execution, per-device memo reconstruction and release. |
| NEW `phydrax/tensor_network/_cutensornet.py` | Separate shape-only path planning from exact-fixed-path network execution; explicit scratch/cache, data rebinding and tuning. |
| NEW `native/cuda/src/cutensornet/CMakeLists.txt` | Separate optional wrapper, linked only to its selected vendor closure. |
| NEW `native/cuda/src/cutensornet/cutensornet_ffi.cpp` | Pinned network-centric C API, fixed path injection/readback, compute/precision configuration, cache invalidation and resource-safe lifecycle. |
| NEW `phydrax/tensor_network/_contraction_ad.py` | Native-owned multilinear JVP, transpose and batching rules over admitted contraction lowerings, including second-order composition and derivative memory admission. |
| `phydrax/tensor_network/_slicing.py` | Stop silently replanning an explicitly selected residual path. Add proposed `PreparedSlicedContraction` binding and `prepare_sliced_contraction`; prepare residual descriptors once, preserve ordinal execution/checkpoint semantics. |
| `phydrax/tensor_network/_placement.py` | Bind device-local residual executors to native slice placement. Account operand replication, concurrent lanes, external scratch/cache, casts and host gather. Preserve fail-closed aggregate behavior. |
| `phydrax/_tensor_network_precision.py` | Resolve actual provider multiply/reduction precision and child evidence. Audit current accumulation behavior; changed numerical semantics require new identity and qualification rather than evidence-only relabeling. |
| `phydrax/tensor_network/_peps.py` | Explicit opt-in through prepared exact-network execution; include selected executor/replay identity. Keep local ordinary `ein.contract` calls native unless separately prepared. |
| `phydrax/tensor_network/_platform_support.py` | Add precise implementation/derivative/distribution support coordinates for completed routes. |
| `phydrax/tensor_network/_platform_runtime.py` | Propagate executor/resource/precision identity into runtime and checkpoint admission. |
| `phydrax/tensor_network/_platform_archive.py` | Persist portable data and required execution provenance; no native handles. |
| `phydrax/tensor_network/_platform_interchange.py` | Validate complete reconstructed plans and implementation availability. |
| `phydrax/tensor_network/__init__.py` | Explicit exports of scientific policy/preparation types; lazy vendor use. |
| NEW `benchmarks/cuda_tensor_contraction.py` | Same-path executor and separate path-planner comparisons, local and independent-slice campaigns. |

MPS/MPO preparation helpers in `_contraction.py` must accept explicit execution selection. Existing tensor-renormalization code calls native `execute_schedule` directly; keep `phydrax/tensor_network/_tensor_renormalization.py` explicitly native and do not advertise accelerated TRG/HOTRG. Charge/graded/anyon/SU(2) owners remain native: a dense `ContractionStructure` is not a symmetry descriptor, and implicit `.to_dense()` is prohibited.

### 4.3 Complete lowering and AD obligations

Admit real/complex 32/64-bit routes only where explicit compute descriptors are qualified. Cover scalar outputs, outer products, retained shared modes, hyperedges, repeated-label diagonal/trace semantics, singleton reductions, and output permutations. Any normalization/gather/cast required by a provider is an explicitly recorded lowering with bounded memory, not hidden dense copy tensors or a silent native fallback. Refuse unsupported arity before execution; splitting a multi-input SSA step changes the lowering and cannot be called the same path without evidence.

For cuTensorNet execution, inject the admitted native path, disable hidden slicing/distributed execution, and rebind current buffers each invocation. For optional cuTensorNet planning, bound search/time, record seed/objective/controls/build, isolate uninterruptible search with existing bounded process ownership, recompute native schedule/resource admission, and keep proposed slicing explicit. Native greedy-search limits are not vendor graph-search bounds.

Implement JVP as multilinear tangent contractions and reverse rules with correct diagonal scatter/broadcast/reduction/permutation semantics. Ordinary complex contraction is bilinear; do not add unconditional conjugation. Test native JAX's actual transpose convention and real losses of complex arguments. Derivative paths are prepared statically, not searched during tracing. Qualify JVP, VJP, bounded batching, mixed second derivatives and Hessian-vector products. Refuse higher/unimplemented transforms explicitly. Vendor backward is not required for this contract and must not be assumed to support arbitrary slicing or higher AD.

Keep host checkpoint construction/merge and existing distributed orchestration honestly host-only. Pure local numerical slice kernels may be JIT/AD capable; that does not make `device_get`/NumPy placement orchestration differentiable. Preserve global ordinal reduction independent of task completion order. A merge over partitioned partial sums has its own declared grouping; do not promise bitwise equality to another grouping.

### 4.4 Tests and acceptance

Extend `tests/unit/tensor_network/test_contraction.py`, `tests/unit/tensor_network/test_network_hpc_portfolio.py`, `tests/unit/tensor_network/test_symmetry_portfolio.py`, `tests/unit/tensor_network/test_tensor_platform_runtime.py`, and `tests/integration/test_tensor_platform_qualification.py`. Keep `tests/unit/test_ein.py` behavior unchanged; use import-isolation tests, not source-spelling assertions.

Add NEW `tests/unit/tensor_network/test_contraction_provider.py`, NEW `tests/unit/tensor_network/test_contraction_provider_ad.py`, NEW `tests/integration/test_cuda_tensor_contraction.py`, and NEW `tests/integration/test_cuda_tensor_slices.py`.

Mandatory cases: invalid imported paths; requested versus actual compute precision; independent real/complex oracles; cancellation; zero tangents; repeated operands; second derivatives; current-data changes under one executable; refresh; concurrent forward/backward cache isolation; release/eviction; insufficient scratch/cache; invalid layout/dtype; fewer slices than devices; uneven slice counts; inactive padded lanes; one failed device; deterministic checkpoint grouping; no symmetry densification.

Benchmark five distinct comparisons: native path/native execution; same path/cuTENSOR; same path/cuTensorNet; cuTensorNet path/native execution; that changed path/cuTensorNet. Keep slice choices fixed for executor comparisons. Report planning, preparation, tuning, lowering, compilation, first/warm execution, derivatives, refresh/rebuild, transfer/reduction, external memory and end-to-end costs. A changed path is an algorithm comparison, not a kernel-only speedup.

## 5. Track D: cuDSS capacity and distributed execution

### 5.1 Three separate contracts

The current `SparseLDLT.memory_mode="hybrid"` already enables the vendor hybrid-memory flag. It does not establish bounded host/device consumption. Do not present the existing flag as a new capability or relabel it certified.

1. **Memory-admitted local/hybrid execution:** separately declared/enforced device and host requirements covering all phases.
2. **MG:** one process, explicitly reserved ordered GPU set, fully resident CSR/RHS on one initiating device, no implicit gather.
3. **MGMN:** one process per GPU, explicitly owned local CSR rows and outputs, native global failure/residual/derivative semantics.

Preserve the public lifecycle: `plan`, `prepare_template`, `bind_numeric`, `solve`, `refresh`, `release`. Add proposed `CudssMemoryPolicy` and `CudssExecutionPolicy` through `SparseLDLT`; keep `memory_mode` as the sole memory-mode selector. Do not add a second synonym. Execution policy carries group/layout reservations, not live communicators.

### 5.2 File map

| File | Planned change |
| --- | --- |
| `phydrax/linalg/_policies.py` | Add typed mode/group/memory policies and explicit unsupported-combination admission. Preserve local default and true multi-RHS versus matrix-batch distinction. |
| `phydrax/linalg/_plans.py` | Extend sparse-direct validation, method configuration and estimates; bind provider/group/layout/memory identities and refuse unsupported modes before allocation. |
| `phydrax/linalg/_costs.py` | Separate external resident/analysis/factor/solve/communication host/device bytes, bounds versus estimates, and cache/overlap costs. |
| `phydrax/linalg/_prepared.py` | Retain canonical prepared representation; incorporate reconstructible execution binding without live pointers. |
| `phydrax/linalg/_binding.py` | Group-aware symbolic/numeric binding, refresh and complete restore validation. |
| `phydrax/linalg/_runtime.py` | Route new modes through existing preparation/solve/release lifecycle and propagate evidence. |
| `phydrax/linalg/backends/_provider.py` | Dispatch local/MG/MGMN implementations while preserving mathematical solve differentiation and failure authority. |
| `phydrax/linalg/backends/_cudss.py` | Carry memory/phase evidence, true residual/refinement and partial-inertia nonclaims. |
| `phydrax/linalg/backends/_cudss_ffi.py` | Extend configuration/tokens and owned phase ordering. Numeric data remains dynamic; local memo IDs remain reconstructible hints. |
| `native/cuda/src/cudss/solver.cpp` | Apply hybrid configuration before analysis, query device minimum, enforce admitted factorization limits, and make completion/error ownership explicit. |
| NEW `phydrax/linalg/backends/_cudss_mg.py` | MG adapter within existing lifecycle; group reservation, initiating-device arrays, output/status and mathematical AD. |
| NEW `native/cuda/src/cudss/mg.cpp` | MG handles/streams, phase handlers, reconstruction and group-safe teardown. |
| NEW `phydrax/linalg/_distributed_sparse.py` | Proposed `DistributedCSRStorage` and `DistributedCSRLinearOperator`; explicit local rows/global columns and native sparse action/transpose/remote-coordinate exchange. |
| `phydrax/linalg/_sparse_contract.py` | Distinguish local canonical CSR from distributed CSR; never reinterpret an existing shape silently. |
| `phydrax/linalg/_distributed.py` | Reuse existing coordinate layouts/pairings; do not create another ownership partition language. |
| NEW `phydrax/linalg/backends/_cudss_mgmn.py` | Collective analysis/bind/refresh/solve/release, coordinated cache misses and rank evidence. |
| NEW `native/cuda/src/cudss/mgmn.cpp` | Global-dimension/local-buffer descriptors, qualified communication layer, collective phases and errors. |
| `native/cuda/src/cudss/CMakeLists.txt` | Build sources/handlers for qualified modes; isolate any site-MPI linkage from portable local execution, using a separate site DSO if needed. |
| `phydrax/linalg/_results.py` | Extend existing provenance/diagnostics only with required execution/phase/resource evidence; reuse portable statuses and derivative validity. |
| `phydrax/linalg/_sparse_providers.py` | Report mode-specific capability/refusal and preserve unreliable zero-inertia limitation. |
| `phydrax/backends/cudss.py` | Accurate availability/distribution contract; installed cuDSS is not automatic MG/MGMN admission. |
| `phydrax/linalg/__init__.py` | Canonical explicit exports of completed policies/storage. |
| NEW `benchmarks/cudss_execution.py` | Local/device/hybrid/MG/MGMN phase-separated comparisons with equal matrix/RHS/residual policies. |

### 5.3 Host/device bounds are a feasibility gate

cuDSS's hybrid device limit can be set after analysis, and the documented allocation handler is device-memory-specific. Neither proves a bound on analysis or host factor allocations. Before claiming certified bounded hybrid execution, obtain supported enforcement or a defensible full-phase upper bound for host and device allocations, including pinned pages and in-flight retired factors.

If that evidence cannot be established, reject a request for the certified bounded-hybrid capability. Do not replace the requested bound with RSS sampling, a post-allocation check, or an unsafe assumption about free RAM. Existing explicitly declared hybrid execution may remain available under its truthful prior scope. Hybrid execute mode and hybrid-plus-MG/MGMN combinations are separate unqualified modes, not implied by individual vendor features.

### 5.4 MG and MGMN semantics

MG runs one full solve, not one duplicate solve per shard. Only the initiating XLA stream is provided by the FFI; do not borrow nonexistent peer JAX streams. Use owned/vendor MG streams with established input readiness and complete vendor work before publishing output. Reserve all participating GPUs and account for external allocations on each. Refuse distributed input, matrix-batch `vmap`, and unsupported capture/overlap.

MGMN storage has local numeric values, local row pointers, global column indices, global shape, a row layout, partition identity and structure identity. First support is real float32/float64, int32 sorted unique CSR, certified symmetry, disjoint contiguous nonempty row intervals, and one GPU per process. Do not insert fake rows or duplicate entries for padding. Vendor-overlapping row ownership is not admitted without a native ownership/adjoint definition.

Rank-local buffers and global JAX arrays are distinct public contracts. An explicitly global row-sharded adapter is allowed only with a proven homogeneous shard/capacity representation and real custom partitioning. Otherwise refuse that route; do not infer global ownership from shape. Residuals/refinement/value gradients exchange needed remote vector coordinates through native sparse ownership rather than silently all-gathering the full vector.

Reuse mathematical sensitivity: tangent solves use the residual tangent, adjoints use the transpose solve, and sparse value gradients accumulate owned contributions exactly once. Preserve derivative tolerances and failure status. Refuse algorithmic factorization AD. Multiple RHS is not matrix batching. Qualify rematerialization, zero cotangents, repeated calls and supported higher compositions so one rank cannot skip a collective required by another.

### 5.5 Shared collective lease prerequisite

| File | Planned change |
| --- | --- |
| NEW `phydrax/_execution_collectives.py` | Internal immutable lease specifications and bounded live communicator ownership bound to existing execution groups; explicit ordered invocation contract. |
| `phydrax/_execution_runtime.py` | Bind/release live leases after existing bootstrap. No second distributed initializer or implicit independent world. |
| `phydrax/_execution_control.py` | Group-qualified preparation consensus, safe points and coupled failure; nonparticipants must not enter a different collective sequence. |
| `phydrax/_execution_resources.py` | Account provider/communicator resources with existing requests/evidence; distinguish limits, estimates and samples. |
| `phydrax/_execution_plan.py` | Admit actual per-device/per-host external footprint and ownership; missing remote evidence remains refusal for certified admission. |
| `phydrax/backends/distributed.py` | Reuse communication ABI/capability validation without making mpi4jax a mandatory runtime for vendor collectives. |
| NEW `native/cuda/src/common/collective_lease.hpp` | Native lifetime/phase ownership of qualified communicator bindings; no independently initialized coordinator. |
| NEW `tests/unit/test_execution_collective_leases.py` | Membership/configuration/order/lifetime refusal and failure propagation as consumer-visible contracts. |
| NEW `tests/integration/test_cuda_collective_leases.py` | Real repeated, unused-output, concurrently submitted, failed and rematerialized collective execution on the target JAX stack. |

A side-effect flag, Python trace-time serialization, native mutex, or registry integer alone is not communicator-wide ordering. Prove an actual JAX 0.12 effect/token/dependency chain and matching all-rank invocation sequence before MGMN or cuDecomp integration. Qualify separate compiled executables and asynchronous dispatch. If JAX cannot represent the contract, record an upstream prerequisite rather than fabricate a pure operation.

Validate configuration and ordered rank/device mapping before vendor entry. Isolate control and numerical communicators. A rank loss inside a collective fails the coupled group under bounded process supervision; attempting another ordinary status collective is not guaranteed recovery. Collective cache misses rebuild on all ranks; collective retirement occurs only at agreed safe points. Closed-lease reuse is a refusal. Do not hide collective communicator destruction in process-local atexit behavior.

### 5.6 Validation

Extend `tests/unit/linalg/test_cudss_provider.py`, `tests/unit/backends/test_backend_contracts.py`, `tests/unit/test_execution_resource_admission.py`, core import/lifecycle tests and relevant typing consumers. Add NEW `tests/unit/linalg/test_cudss_execution_contracts.py`, NEW `tests/integration/test_cudss_mg.py`, and NEW `tests/integration/test_cudss_mgmn.py`.

Cover SPD and symmetric-indefinite matrices; singular/near-singular/nonfinite inputs; int32 overflow; one/multiple RHS; refresh; structure mismatch; release/rebuild; rank-asymmetric eviction; tiny device/host limits; one-rank preparation failure; queued-work failure; correct transpose/value/RHS derivatives; unavailable MPI/NCCL ABI; inconsistent modes; stale/closed group; partial output refusal. Preserve current local batches and partial-inertia behavior. `phydrax/linalg/_inertia.py` and `phydrax/optim/_structured_ipm.py` must not acquire reliable zero-inertia claims from MG/MGMN support.

## 6. Track E: cuEST molecular integrals and native SCF

### 6.1 Complete initial scientific contract

The selected complete profile is **nonperiodic, all-electron, real-FP64 density-fitted RHF energy and analytic first nuclear forces**, with native SCF and typed energy/force/PES workflows. It is not an energy-only vendor call or an end-to-end vendor SCF implementation.

Admit integer ground-state occupations, an explicit orbital basis, spherical auxiliary basis, full-range Coulomb fitting, explicit screening/conditioning and resource policies. Cartesian or spherical orbital representations are allowed when individually qualified; mixed representations are refused. DF-UHF and native-grid DF-RKS/UKS are subsequent explicit extensions. Vendor Vxc, ECP, PCM, range separation, excited states, arbitrary AD, analytic Hessians and periodic chemistry are not implied.

### 6.2 Owning API and file map

Introduce a proposed `MolecularIntegralPlan` at the Gaussian owner. It selects an implementation and explicit representation, auxiliary basis/fitting/screening policies and resources. Preparation binds canonical basis/system identity; geometry binding supplies dynamic positions. Its operations return typed one-electron quantities, J/K for the actual density, and contracted nuclear-gradient contributions with status and definition/geometry identity.

Migrate `MolecularHartreeFockPlan` to one `integrals` configuration. Remove superseded parallel `factorized_eri`/`direct_jk` constructor dispatch after migrating every caller. Preserve native dense/direct/factorized implementations as explicit implementations of that one boundary, not compatibility parameters. Exact ERI primitives stay exact and are not redirected to density fitting.

| File | Planned change |
| --- | --- |
| `phydrax/operators/quantum/gaussian/_shell.py` | Reuse canonical active primitives, contractions, representation and angular-component definitions. |
| `phydrax/operators/quantum/gaussian/_basis.py` | Reuse `GaussianBasisPlan`/`PreparedGaussianBasis`, normalization, stable center IDs and shell ordering; expose only needed validated layout information. |
| `phydrax/operators/quantum/gaussian/_spherical.py` | Retain canonical real-spherical ordering/phase and test vendor bridge against it. |
| NEW `phydrax/operators/quantum/gaussian/_provider.py` | Integral preparation/geometry interfaces and typed one-electron/J-K/contracted-gradient results; precise derivative and representation admission. |
| NEW `phydrax/operators/quantum/gaussian/_basis_layout.py` | Validated packed-shell layout and sparse/block AO transformations derived from native basis definitions; no basis-name identity inference. |
| NEW `phydrax/operators/quantum/gaussian/_cuest.py` | Lazy cuEST preparation, explicit precision/fitting policy, workspace queries, FFI binding and geometry refresh/reconstruction. |
| `phydrax/operators/quantum/gaussian/_integrals.py` | Keep canonical native one-electron and exact ERI primitives; adapt through the new owner without hidden approximation. |
| `phydrax/operators/quantum/gaussian/_direct.py` | Adapt `PreparedDirectJK` into the shared integral boundary, preserving screening/status. |
| `phydrax/operators/quantum/gaussian/_screening.py` | Preserve geometry-validity/evidence ownership; no invented vendor reuse certificate. |
| `phydrax/operators/quantum/gaussian/_factorization.py` | Keep DF metric/rank and factor identities explicit. Fixed factors cannot claim moving-geometry forces without corresponding reconstruction/derivatives. Do not redefine periodic factor semantics. |
| `phydrax/operators/quantum/gaussian/_derivatives.py` | Preserve tensor-valued integral derivative API; contracted vendor gradients do not satisfy it. Refuse unsupported derivative requests. |
| NEW `native/cuda/src/cuest/CMakeLists.txt` | Optional tiny wrapper using qualified headers/runtime closure. |
| NEW `native/cuda/src/cuest/cuest_ffi.cpp` | Basis/geometry setup, S/T/V, DF J/K, contracted gradients, bounded memo/workspace and failure-safe destruction. |
| `phydrax/chemistry/electronic_structure/_molecular_hf.py` | Cut over `_integrals`, `_jk`, SCF preparation, final physical Fock, stationary-force admission, gradients and downstream capability checks as one scientific change. |
| `phydrax/chemistry/electronic_structure/_mean_field.py` | Add integral-definition/representation provenance to mean-field states/evidence where needed; migrate every constructor including stability and KS consumers. |
| NEW `phydrax/chemistry/electronic_structure/_molecular_provider.py` | Canonical molecular mean-field adapter into existing electronic provider/result protocols; native SCF remains the owner. |
| `phydrax/chemistry/_numerical.py` | Bind density-fitting, analytic-Lagrangian derivative route and component definition IDs; no representation alias. |
| `phydrax/chemistry/_provider.py` | Distinguish analytic first forces from differentiability, response, Hessians and execution scope. |
| `phydrax/chemistry/_model.py` | Resolve explicit orbital/auxiliary basis definitions and chemistry semantics; provider choice is not a physical-method alias. |
| `phydrax/chemistry/_calculation.py` | Propagate numerical/provider definition into calculation identity and admission. |
| `phydrax/chemistry/_result.py` | Preserve native unit/status/force/convergence evidence and implementation provenance. |
| `phydrax/chemistry/_lifecycle.py` | Restore/restart checks include changed numerical definitions; no old identity relabeling. |
| `phydrax/chemistry/_production_qualification.py` | Add precise candidate support coordinates; promotion requires actual scientific/provider evidence. |
| `phydrax/operators/quantum/gaussian/__init__.py` | Canonical integral policy/result exports without vendor loading. |
| `phydrax/chemistry/electronic_structure/__init__.py` | Export completed native mean-field provider/preparation contract. |
| `phydrax/chemistry/__init__.py` | Update only owned public facade entries required by the result/provider workflow. |
| NEW `benchmarks/cuest_molecular.py` | Basis/geometry/DF setup, changed-density J/K, SCF, force and changed-geometry timing plus external memory. |

### 6.3 Scientific corrections required before vendor insertion

At the planning baseline, `_jk` can use factorized/direct integrals, but `analytic_gradient_atomic_units` differentiates exact dense ERIs unconditionally. Stability and response contain analogous bypasses. Inserting cuEST only into `_jk` would produce energy and forces for different approximations. Reproduce the narrow native mismatch during implementation, retain a behavioral regression, and route each derivative through the same representation or refuse it explicitly.

The K builder's occupied-factor interface must represent the **actual SCF density**, including damping, not merely the newest orbitals. Reuse valid orbital factors when they reconstruct that density; otherwise obtain an admitted PSD factor through native linalg. Never silently clip an indefinite density or change damping. Include this factorization in cost/resource evidence. Handle zero density without invoking an unsupported nonzero-occupation vendor operation.

For RHF, keep the spin-summed-density energy convention and its exact Coulomb/exchange factors; forces combine one-electron, DF two-electron, overlap/Pulay and nuclear contributions. Verify vendor gradient scale conventions against that native energy. Rebuild the final unshifted physical Fock, check stationary residual and density/orbital/energy-weighted-density consistency, and require complete SCF success before admitting conservative forces. A successful integral call is not SCF convergence.

### 6.4 Basis, approximation and lifetime contracts

- Derive the bridge from native basis arrays, active primitive masks and stable particle IDs. Split general contractions only with a reversible AO mapping.
- Native normalization and vendor primitive/angular normalization are not interchangeable. Apply an explicit validated basis transformation to matrices, densities, coefficient factors and Pulay densities. Store structured maps rather than an avoidable global dense conversion matrix.
- Preserve Cartesian component order and real spherical order `0,+1,-1,+2,-2,...`, phases, center mapping and unit conversion. The boundary uses bohr/hartree; native result construction owns user units and force sign.
- Admit the stricter documented per-center angular-contiguity condition until the pinned vendor API resolves inconsistent overview wording. Do not add fake shells. Qualify supported angular momentum separately for values and gradients.
- Set native FP64 math explicitly. Set fitting cutoff, absolute/relative conditioning, inversion algorithm, auxiliary basis and screening explicitly; native and vendor defaults differ. Rank evidence must be obtained or marked unavailable, never fabricated.
- Distinguish matched-DF implementation error from DF-versus-exact approximation error. A discarded metric eigenvalue is not an energy-error certificate.
- Rebuild geometry-dependent pair lists/DF plans after coordinate changes. Reuse within all SCF iterations and associated forces at one geometry. Count overlapping geometry generations and cache leases.
- Query every creation/compute workspace before allocation. A vendor variable scratch-buffer limit is not a total-memory limit. Preserve query/compute settings exactly.
- Initially advertise analytic nuclear forces and `differentiable=False` for the cuEST electronic provider. Reject arbitrary JVP/VJP, basis/exponent derivatives and analytic Hessians; native exact routes retain their existing transformation support.

### 6.5 Downstream cutover map

| File | Required disposition |
| --- | --- |
| `phydrax/chemistry/electronic_structure/_response.py` | Refuse provider/DF states unless the response uses the same integral definition. No silent exact CPHF/CPKS substitution. |
| `phydrax/chemistry/excited/_hf_response.py` | Preserve reference/response Hamiltonian agreement; refuse unsupported DF excited response before dense reconstruction. |
| `phydrax/chemistry/excited/_ks_response.py` | Apply the same rule when KS integration is enabled. |
| `phydrax/chemistry/electronic_structure/correlation/_orbital.py` | Bind integral-store provenance and require an explicitly matching definition or explicitly declared different correlated Hamiltonian. Downstream correlation solvers consume this boundary. |
| `phydrax/chemistry/electronic_structure/_molecular_ks.py` | Migrate state construction in the main cutover; keep the cuEST execution route gated until J/K, XC/grid semantics and forces are implemented together. |
| `phydrax/chemistry/electronic_structure/_derivatives.py` | Explicit finite-difference-of-forces Hessians may use qualified force surfaces; do not relabel them analytic vendor Hessians. |
| `phydrax/chemistry/_surface.py` | Preserve typed energy/force capability, per-evaluation failure, unit conversion and geometry optimization integration. |
| `phydrax/chemistry/electronic_structure/_hartree_fock.py` | Keep existing `NativeRHFPlan` consumers explicit; reuse result-adapter conventions, not finite-difference work-count assumptions. |
| `phydrax/chemistry/qmmm/_core.py` | Existing embedded native RHF remains unchanged unless explicitly migrated; no inherited embedding claim. |
| `phydrax/chemistry/periodic/_gamma.py` | Existing periodic DF remains unchanged; molecular cuEST restrictions do not redefine shared periodic factors. |

### 6.6 Tests and complete workflow

Extend `tests/unit/operators/quantum/gaussian/test_general_gaussian.py`, `tests/unit/chemistry/test_general_mean_field.py`, `tests/unit/chemistry/test_contracts.py`, `tests/unit/chemistry/test_correlation.py`, `tests/unit/chemistry/test_excited_advanced.py`, and `tests/integration/chemistry/test_ground_state_workflow.py`.

Add NEW `tests/unit/operators/quantum/gaussian/test_basis_layout.py`, NEW `tests/unit/operators/quantum/gaussian/test_integral_provider.py`, NEW `tests/integration/chemistry/test_cuest_integrals.py`, NEW `tests/integration/chemistry/test_cuest_mean_field.py`, and NEW `tests/integration/chemistry/test_cuest_lifecycle.py`.

Use independent native/host integral and matched external DF references with recorded provenance. Cases include s/p/d and higher admitted shells; general contractions; reordered centers; Cartesian/spherical conventions; matched S/T/V; random/damped/rank-deficient/zero PSD densities; RHF closed-shell molecules and charged cases; finite-difference energy gradients over several step sizes; independent analytic forces; translation/rotation/net-force/net-torque checks; changed geometry; loose/nonconvergent SCF; rank/cutoff crossings; invalid basis/periodicity; unavailable provider; resource failure; gradient capability refusal; and typed PES geometry optimization. Native exact versus vendor DF is not the parity oracle.

DF-UHF adds spin-resolved and empty-spin-channel energy/force qualification. DF-RKS/UKS must first retain native XC/grid and migrate J/K plus forces together. Vendor Vxc, ECP and PCM require separate functional/grid/cavity/effective-charge definitions and full derivatives through existing `_functional.py`, `_grid.py`, Gaussian `_ecp.py`, and electronic `_environment.py` owners; they are not silently included in the RHF support claim.

## 7. Track F: choose cuDecomp, not two FFT runtimes

### 7.1 Decision and supported profile

Choose cuDecomp as the first candidate because it can replace an explicitly admitted redistribution while preserving local JAX FFTs, normalization, precision and scientific owners. Do not add cuFFTMp dependencies/files in this primary implementation.

The complete initial profile is regular divisible 3D direct slab redistribution, one process per GPU, explicit complex64/complex128, exact validated source/target/rank ordering, bounded scratch and an explicitly qualified GPU-aware MPI transport. No hidden autotuning, NVSHMEM heap, automatic workspace growth, or second coordinator.

The native full-mesh xy-pencil-to-z-slab schedule is not automatically equivalent to cuDecomp's conventional X/Y/Z pencil sequence. Preserve native pencil execution. Explicit cuDecomp selection for an unsupported canonical pencil mapping must refuse during preparation, not silently fall back or reintroduce the historical subgroup schedule. The baseline still qualifies native 2x4/4x2 pencils. A later cuDecomp pencil mapping requires exact endpoint proof, full packing/workspace accounting and new execution identities.

### 7.2 File map

| File | Planned change |
| --- | --- |
| NEW `phydrax/backends/cudecomp.py` | Lazy availability and exact site MPI/build/transport capability admission. |
| NEW `phydrax/discretization/spectral/_cudecomp.py` | Descriptor preparation, `cudecompGetPencilInfo` layout proof, explicit packing, inverse/transpose derivatives and lease binding. |
| NEW `native/cuda/src/cudecomp/CMakeLists.txt` | Explicit site-build component with qualified MPI closure; not a core portable default. |
| NEW `native/cuda/src/cudecomp/transpose.cpp` | Stream-taking transpose, bounded workspace and group-owned descriptor lifecycle. |
| `phydrax/discretization/spectral/_distributed.py` | Add explicit transpose implementation selection. Update preparation/report/resource/result/sequence identity; array conveniences must not discard failures. Keep native full-mesh pencil arithmetic unchanged. |
| `phydrax/discretization/spectral/__init__.py` | Export scientific policy/evidence only, no vendor handles. |
| `phydrax/discretization/spectral/_distributed_les.py` | Propagate provider/preparation evidence and exact execution/sequence identity. |
| `phydrax/applications/incompressible_flow/_distributed_les.py` | Production/restart admission must reject incompatible execution bindings. |
| `phydrax/solver/maxwell/spectral/_psatd.py` | Explicitly admit the transform implementation in global PSATD preparation and IDs. |
| `phydrax/solver/_distributed_pic.py` | Preserve PSATD/PIC execution/restart ownership and test repeated composed transform histories. |
| `phydrax/applications/cosmology/_distributed_mixed.py` | Keep numerical restart compatibility separate from concrete shard/provider execution identity. |
| `benchmarks/distributed_spectral.py` | Add explicit implementation/transport/layout reporting, same-science provider comparisons and external memory. |
| `tools/distributed_execution_qualification.py` | Package observed provider evidence into existing qualification contracts; this tool is not the GPU execution driver. |
| NEW `tests/integration/test_cudecomp_spectral.py` | Real permutation, transform/adjoint, lifecycle, repeated-history and failure tests on the qualified layout. |

The same communicator/effect prerequisite as MGMN applies. Trailing payloads and padded layouts need explicit support and memory accounting; they are not an assumed vendor batch. Transpose JVP is the same redistribution; transpose/VJP is the inverse map with exact declared layouts. Preserve complex Hilbert adjoints and FFT normalization independently.

Extend existing distributed spectral, LES, PIC, cosmology and mixed-workflow tests rather than treating standalone round trips as end-to-end evidence. A same-host 8x1 MPI transpose does not qualify 1x8 vendor execution, mixed GPUs-per-process, multi-host, or RDMA.

cuFFTMp remains a separately gated alternative if full distributed FFT capacity/performance is demonstrably missing after the native/cuDecomp comparison. Reopen selection only with exact NVSHMEM/topology, layout, differentiation, memory and identity evidence; never ship both merely to avoid choosing.

## 8. Conditional tracks

### 8.1 cuDensityMat: finite-factor generator actions

Entry gate: a real finite-factor density workload, pinned upstream JAX extension compatible with Python 3.13/JAX 0.12/CUDA13, and bounded lifetime/resource behavior. Prefer the upstream JAX integration; do not write another quantum AD stack by default. Upstream documentation disagrees on some transformation support across extension releases. Pin actual code and retain conservative restrictions until tested.

| File | Conditional change |
| --- | --- |
| `phydrax/operators/quantum/_register.py` | Reuse `HilbertRegisterLayout` identities and ordered wires. |
| `phydrax/solver/_local_hamiltonian.py` | Reuse local/product Hamiltonian terms; no global Hamiltonian/Liouvillian materialization. |
| `phydrax/operators/quantum/_open_contracts.py` | Reuse `QuantumGeneratorAction` and physicality evidence. |
| NEW `phydrax/solver/_lindblad_action.py` | Local jump definitions, native factor-action reference, explicit provider policy, preparation/refresh and existing IVP composition. |
| NEW `phydrax/backends/cudensitymat.py` | Lazy pinned upstream extension, capability/resource/lifetime admission and native-to-provider operator lowering. |
| `phydrax/solver/_differential.py` | Compose the action with `DifferentialProblem`, not a new integrator. |
| `phydrax/solver/_diffrax_backend.py` | Admit only integrator/adjoint transformations supported by the selected action. |
| `phydrax/solver/_diffrax_state_packing.py` | Reuse complex real-coordinate packing and validate its derivative requirements. |
| `phydrax/solver/__init__.py` | Export completed scientific action preparation only. |
| NEW `tests/unit/backends/test_cudensitymat_backend.py` | Version/import/resource/transform/lifetime refusal. |
| NEW `tests/unit/solver/test_lindblad_action.py` | Analytic damping, unequal/reordered factor dimensions, empty/zero-rate jumps, changed coefficients, supported gradients and integration refinement. |
| NEW `tools/cudensitymat_qualification.py` | Action/backward/integration/external-memory campaign through existing helpers. |

Use dimensionless jump operators plus explicit nonnegative rates, and adapt existing collapse operators without multiplying their already embedded square-root rate twice. States have declared ket/bra axis order. Dynamic operators, coefficients and states remain JAX leaves. Initially refuse unqualified forward/higher AD, nested batching and batch-of-gradient compositions, unsupported precision/representations and distributed density evolution.

Keep `phydrax/solver/_lindblad.py`, `phydrax/solver/_finite_cptp.py` and trajectory owners explicit and unchanged. An integrated generator action is not a complete CPTP channel. Preserve trace, Hermiticity, positivity and solver evidence without clipping or renormalization. Exit requires a complete native IVP workflow and workload benefit, not one operator call.

### 8.2 cuOpt: audited LP/QP/MILP

Entry gate: assembled optimization workload exceeding current native/MPAX/SCIP capability/performance, a compatible C-API runtime closure, and enforceable declared resource behavior. External solve is initially host-orchestrated; do not promise JIT/vmap or algorithmic AD.

| File | Conditional change |
| --- | --- |
| NEW `phydrax/backends/cuopt.py` | Lazy availability, explicit algorithm/precision/device selection, preparation and provider lifetime. |
| NEW `phydrax/optim/_programming/_cuopt.py` | CSR lowering, objective/row/bound conversion, original-coordinate duals and native KKT/status audits. |
| `phydrax/optim/_programming/_policy.py` | Explicit `CuOptPDLP`, `CuOptBarrier`, `CuOptConcurrent` methods and method-specific support. |
| `phydrax/optim/_programming/_lifecycle.py` | Canonical plan/template/bind/refresh/solve dispatch. |
| `phydrax/optim/_programming/_quadratic.py` | Update ordinary and differentiated dispatch consistently; native one-half quadratic convention must be converted to vendor convention exactly. |
| `phydrax/optim/_programming/_audit.py` | Audit original-coordinate rays/certificates; no fabricated infeasibility evidence. |
| `phydrax/optim/_programming/_conic_sensitivity.py` | Permit native KKT sensitivity only for audited regular witnesses with matching plan/binding and fixed active-set assumptions. |
| NEW `phydrax/optim/_programming/_cuopt_mixed_integer.py` | MILP incumbents/bounds/gaps/termination mapping through existing proof authority. |
| `phydrax/optim/_programming/_mixed_integer_policy.py` | Explicit linear-only cuOpt mixed-integer method. |
| `phydrax/optim/_programming/_mixed_integer_lifecycle.py` | Preparation, refresh and solve dispatch; candidate audit. |
| `phydrax/optim/_programming/_mixed_integer.py` | Preserve provider-reported versus independently certified result semantics. |
| `phydrax/optim/_programming/__init__.py` | Canonical method exports. |
| `phydrax/optim/__init__.py` | Explicit facade exports without runtime loading. |
| NEW `native/cuda/src/cuopt/CMakeLists.txt` | Optional qualified C-API wrapper only if required by selected binding. |
| NEW `native/cuda/src/cuopt/cuopt.cpp` | Owned model/buffer/solver lifetime and complete error propagation. |
| NEW `tests/unit/backends/test_cuopt_backend.py` | Lazy availability, execution/precision/resource refusals. |
| NEW `tests/unit/optim/test_cuopt_backend.py` | Sparse LP/QP correctness, objective scale, original-coordinate KKT, refresh and sensitivity admission. |
| NEW `tests/unit/optim/test_cuopt_mixed_integer.py` | Enumerable MILP optimum, early termination with/without incumbent, bounds/gaps and proof authority. |
| NEW `tools/cuopt_qualification.py` | Packing, prepare, refresh, solve and audit measured separately at equal independently verified tolerances. |

Native QP uses one-half x-transpose Q x; vendor QP does not. Convert the matrix including explicit regularization and test off-diagonal storage to catch double counting. Concurrent LP includes CPU simplex; it is explicit hybrid execution, not GPU-only. Documented two-GPU LP portfolio is not an eight-GPU distributed solve. Refuse MIQP, unsupported cones, matrix-free provider input, nonconvex QP, and MILP derivatives.

Extend existing convex lifecycle/bounds/linear/quadratic/conic-sensitivity/mixed-integer tests and policy-accepting control/power/energy-planning consumers. Do not allow `prepare_qp_sensitivity` to inherit support accidentally from ordinary solve dispatch. Exit requires original-problem audit and full scientific consumer behavior.

### 8.3 cuSOLVERMp: only a demonstrated dense-capacity need

Dependency: the collective lease/resource gate in Section 5.5. This is never a sparse or matrix-free fallback.

NEW `phydrax/linalg/_distributed_dense.py` owns explicit block-cyclic layout and distributed dense operator; NEW `phydrax/linalg/backends/_cusolvermp.py` and NEW `phydrax/linalg/backends/_cusolvermp_ffi.py` implement the selected factor/solve lifecycle; NEW `phydrax/backends/cusolvermp.py` owns availability; NEW `native/cuda/src/cusolvermp/CMakeLists.txt` and NEW `native/cuda/src/cusolvermp/solver.cpp` own the optional wrapper.

Extend existing linalg policy/plan/provider/runtime/result owners, not a separate solver API. Start with one explicitly selected documented dense factor/solve contract, column-major local storage and explicit leading dimensions on a 2D block-cyclic layout. Keep input/output distributed; charge redistribution explicitly. Eigen/SVD, complex/dtype variants and rank-deficient problems need their own contracts and are not inferred from library presence.

Add NEW `tests/integration/test_cusolvermp.py` and NEW `benchmarks/cusolvermp.py`: independent solution/residual/adjoint comparison, refresh, singular/failed solve, non-square process grids, workspace/refusal and collective teardown. Pin the actual current initialization API rather than copying older CAL/MPI examples.

### 8.4 MathDx: one measured kernel, not a framework

Entry gate: profile the existing positive discrete-velocity energy-equilibrium workflow and establish that repeated tiny solves or intermediate storage materially limit it. The concrete candidate is its masked 2x2 covariance solve; XLA may already fuse it better. No measured benefit means no MathDx feature is added.

| File | Conditional change |
| --- | --- |
| `phydrax/linalg/_small_batched.py` | Explicit native/MathDx selection preserving existing full status, scaling, determinant/condition, residual and refinement semantics. |
| NEW `phydrax/linalg/_small_batched_mathdx.py` | One admitted shape/dtype specialization, FFI, bounded batching and mathematical derivative rules. |
| NEW `native/cuda/src/mathdx/CMakeLists.txt` | Qualified build-only MathDx headers and inspected target code; no mandatory runtime SDK. |
| NEW `native/cuda/src/mathdx/small_linear.cu` | Fused scale/pivoted solve/refinement/evidence kernel using JAX-owned buffers. |
| `phydrax/discretization/discrete_velocity/_energy_equilibrium.py` | Accept explicit small-solve plan; preserve Newton masks, positivity, conservation and linear-failure propagation. |
| `phydrax/discretization/discrete_velocity/_spatial.py` | Preserve changed energy-plan identity through runtime and learned binding admission. |
| `phydrax/closure_data/_kinetic_equilibrium.py` | Refuse stale learned artifact bindings rather than relabeling plan identity. |
| NEW `tests/unit/linalg/test_small_batched_mathdx.py` | Singular/scaled/nonfinite/mixed-success batches, inactive lanes, numerical evidence, AD and unsupported selection. |
| NEW `tools/mathdx_small_solve_qualification.py` | Whole equilibrium workflow plus isolated kernel phases; include evidence cost and compiler/external memory. |

Retain native dimensions/dtypes/multiple-RHS support unchanged. Extend learned-energy/spatial/closure-data consumers and relevant native small-solve regressions. Vendor `info==0` is not native success evidence. cuFFTDx/cuBLASDx specializations require separate measured consumers; this plan does not authorize blanket substitution.

## 9. Test architecture, benchmark methodology, and acceptance matrix

### 9.1 Minimal conservative selection

Build one affected matrix from changes since the latest merge into `dev`. Include changed behavior and direct consumers; add shared native-provider tests for common loader/ABI edits, linalg consumers for policy/status changes, typing/Strict declarations for new contracts, and persistence/fingerprint/reconstruction consumers for changed prepared representations. Do not run unrelated domains merely because a CUDA library was added.

CPU/static tests run with CUDA hidden and bounded `pytest -n auto` workers. Provider and topology qualification runs separately with `-n 0` where one process owns devices/communicators. Never overlap unrestricted xdist invocations, builds, GPU campaigns or typing gates with performance measurements. Optional providers skip at their canonical boundary during ordinary CPU testing; explicitly selected qualification profiles reject missing/skipped required cases.

A new test must catch a consumer-visible behavior, boundary, transition, error or numerical invariant. No source-text/import-spelling/wiring tests, no tests of incidental defaults, and no large wrappers that obscure independent failures. Preserve independent scientific references and tolerances; use deterministic fault cases for failure/rollback/cache freshness. Remove touched incidental tests rather than re-pinning them.

### 9.2 Runtime matrix

| Surface | Mandatory physical scope | Key observable acceptance |
| --- | --- | --- |
| Native transport/execution | 1/2/4/8 GPUs; 1x8, 2x4, 4x2, 8x1; all directed peer pairs | Correct contents, ownership, collective results, convergence, completion and teardown |
| Core local companions | Every selected GPU; existing provider-specific concurrency | Numerical parity/status, release/rebuild, cache/device isolation and normal/hostile exit |
| Tensor local providers | Each selected device and all admitted real/complex precisions | Same-path primal/AD, changed data, cache freshness, resources and refusals |
| Native-owned tensor slices | 1/2/4/8 devices in supported host orchestration | Exact slice coverage, bounded lanes, declared ordinal/grouped reduction and failed-slice rejection |
| cuDSS bounded memory | One device plus explicit supported capacity profiles | Analysis/factor/solve/refresh bounds, minimum-memory refusal and truthful unknowns |
| cuDSS MG | One process with 2/4/8 GPUs | One solve, multi-RHS, no hidden gather/batches, group-wide memory and completion |
| cuDSS MGMN | 2/4/8 processes, one GPU each | Local CSR ownership, global descriptors, uniform status, distributed residual/AD, coordinated rebuild/release |
| Native spectral baseline | Eight-way slab and 2x4/4x2 pencils across native process layouts | Padded/unpadded reference, JVP/VJP, composed LES/PSATD/PIC histories and restart |
| cuDecomp selected route | 2/4/8 processes, one GPU each, validated direct slabs | Exact permutation, transform/adjoint, site ABI, bounded workspace and unsupported-pencil refusal |
| cuEST | One device per molecular calculation; per-device isolation | Matched DF S/T/V/J/K, converged native SCF, same-energy conservative forces and PES workflow |
| Conditional providers | Only their explicit admitted profiles | Complete named workflow and failure/AD/resource evidence; no inherited eight-device claim |

Changing visibility, runtime, provider, flags, executable source or generated inputs invalidates dependent evidence. Reuse valid unchanged checks. A single-host result never certifies multi-host failure behavior, fabric performance or network RDMA.

### 9.3 Performance comparisons

Run uncontended and record before/after on matched hardware, dependency versions and scientific settings. Vary controlling capacity and device count, not only a small fixed example. Keep preparation, path planning, tuning, lowering, compilation, first execution, warm samples, numeric refresh, reconstruction, communication, audits and teardown separately identified.

Report per-rank samples and corresponding-rank maximum latency for collective operations. Separate strong and weak scaling. Keep logical retained arrays, XLA temporary/output/code bytes, vendor device/host workspace and observed memory samples distinct. Record algorithm changes separately from same-algorithm backend changes. Preserve failures and numerical gates; never compare a lower-precision or failed run as a speedup.

Promotion requires either meaningful new capacity/capability or repeatable end-to-end improvement without losing admitted science. Hardware qualification can be successful while performance remains a nonclaim. Do not adopt a library solely because its isolated kernel is faster.

### 9.4 Required diagnostics and generated consistency

During implementation, after focused runtime smoke and affected regressions:

- Apply configured Ruff formatting/lint to changed files before identity-sensitive verification.
- Run `python tools/check_typing.py check`; it is the only repository static-checker gate.
- Run `python tools/audit_selectors.py` and `python tools/audit_contract_candidates.py --signatures`; select declaration tests including `tests/unit/test_contract_declarations.py` and relevant `tests/unit/test_strict.py` contracts.
- Run `python -m tools.check_installed_typing` for public annotation changes; the inspected current tool defaults to Python 3.13, consistent with the actual package requirement.
- Measure changed-symbol complexity before/after and decompose according to repository thresholds, without moving numerical operations or reduction order merely to lower a metric.
- Generate final public/capability data once, after the implementation stabilizes, with `python -m tools.generate_public_api_manifest` and `python -m tools.generate_capability_inventory`.
- Check it with `python -m tools.check_public_api_manifest` and `python -m tools.check_capability_consistency`.
- Run affected documentation/examples and installed-wheel provider smokes. No full suite unless impact analysis demonstrates a genuinely global collection/configuration/shared-substrate change requiring it.

These are future commands, not executed verification from planning.

## 10. Documentation, examples, and generated artifacts

| File | Implementation-stage update |
| --- | --- |
| `README.md` | Optional-provider discovery and bounded support claims; no blanket GPU acceleration statement. |
| `docs/installation.md` | Core versus optional extras, site-MPI components, A40 evidence and unsupported scopes. |
| `docs/api/backends.md` | Exact provider availability, array/process model, transformations, transfer and failure contracts. |
| `docs/api/configuration.md` | Added library overrides and immutable startup behavior. |
| `docs/api/execution.md` | Collective leases, ownership, failure boundaries and external memory evidence. |
| `docs/api/linalg.md` | cuDSS modes/memory/AD/status; conditional dense and small-kernel routes only when delivered. |
| `docs/api/tensor_network.md` | Path source versus executor, prepared slice binding, precision, AD and replay identity. |
| `docs/guides_tensor_platform.md` | Full prepared tensor examples, independent-slice distribution and support coordinates. |
| `docs/guides_ein.md` | Point to explicit prepared execution; ordinary `ein.contract` remains unchanged. |
| `docs/api/ein.md` | Same unchanged-default contract and prepared-execution cross-reference. |
| `docs/api/discretization/spectral.md` | Native versus selected transpose provider, exact supported layouts and refusals. |
| `docs/guides_spectral_methods.md` | Same-science provider use, normalization/adjoints, resource and topology constraints. |
| `docs/guides_spectral_pic.md` | Provider-bound PSATD/PIC execution and restart identity where exposed. |
| `docs/guides_computational_chemistry.md` | Exact versus DF representation, provider preparation, same-energy forces and unsupported response. |
| `docs/guides_production_computational_chemistry.md` | Typed energy/force/PES workflow, stationary admission and production qualification scope. |
| `docs/api/chemistry.md` | Canonical integral/provider selection and result semantics. |
| `docs/api/chemistry_production.md` | New prepared Gaussian/integral and mean-field provider contracts. |
| `docs/cookbook/production_computational_chemistry.md` | Complete explicit DF-RHF energy/force/geometry workflow with references and refusal behavior. |
| `docs/guides_quantum.md` | Conditional generator-action integration, rate/axis semantics and physicality nonclaims. |
| `docs/api/solver/final_open_quantum.md` | Conditional action versus channel distinction and transformation limits. |
| `docs/api/metrix/quantum_density.md` | Conditional action integration versus existing dense exponential methods. |
| `docs/api/optim.md` | Conditional cuOpt objective convention, hybrid execution, KKT sensitivity and MILP proof authority. |
| `docs/guides_lattice_boltzmann.md` | Conditional measured small-solve selection and invariant preservation. |
| `docs/api/discretization/lattice_boltzmann.md` | Conditional explicit energy-equilibrium small-plan selection. |
| `docs/all-of-phydrax.md` | Update support overview only for delivered, accurately scoped routes. |
| `phydrax/qualification/_builtin_catalog.py` | Precise candidate profiles and evidence requirements, never automatic release claims. |
| `docs/data/public_api.json` | Regenerate from canonical exports. |
| `docs/data/capabilities.json` | Regenerate provider/profile inventory. |
| `docs/data/application_portfolios.json` | Regenerate dependent portfolio inventory. |
| `docs/data/capability_closure.json` | Regenerate actual closure state, not planned completion. |
| `docs/data/source_absorption.json` | Regenerate only from the owning provenance ledger; external source/license facts must be real. |
| `docs/api/capabilities.md` | Regenerate from inventory. |
| `docs/api/capability_closure.md` | Regenerate from closure owner. |
| `mkdocs.yml` | Add navigation only for new public guides actually created during implementation. This planning document need not alter public navigation. |
| `CHANGELOG.md` | Record delivered providers, intentional representation/identity changes, unsupported transforms, deployment restrictions and actual evidence scope. |

Prefer complete examples in the existing guides above over a parallel set of provider-only tutorials. Add a standalone example only if it demonstrates a genuinely distinct runnable workflow not already represented. New benchmark JSON records are generated only after real execution and are unignored individually; no fabricated baselines, versioned schemas or empty success manifests.

## 11. Implementation dependency order and ownership

One integration owner controls shared loader/configuration/build files, execution-resource contracts, generated artifacts and the task-level verification record. Give each scientific subsystem one active owner. Siblings must not independently edit shared files or launch competing GPU campaigns.

1. **Baseline gate:** environment/artifact admission, truthful command gates, transport and current native workflows on eight A40s.
2. **Shared local provider gate:** optional packaging/dependency attribution, separate DSOs, local lifetime/resource contract, installed core-only regression.
3. **Independent ready local tracks:** prepared tensor execution and molecular integral/SCF contract work may proceed concurrently with bounded CPU/native build resources. cuDSS local-memory work is another independent owner.
4. **Tensor completion:** fixed-path local executors, transformation closure, then optional planner and native slicing/consumer workflows.
5. **Chemistry completion:** basis/DF primitive parity, then native SCF plus same-energy forces/PES; migrate representation consumers before promotion.
6. **Distributed prerequisite:** establish collective lease/effect/ABI/resource semantics before MGMN or cuDecomp implementations. Local MG can proceed independently only after its group reservation/stream ownership is established.
7. **Distributed completion:** MGMN and selected cuDecomp route, each with real failure/derivative/composed-workflow evidence. Native mixed-process/pencil baseline remains distinct.
8. **Conditional tracks:** start only after their entry gates and scope selection. They must not delay or silently broaden primary contracts.
9. **Integrated verification:** focused failures first, then conservative affected tests, installed typing/wheels, scientific qualification and isolated performance campaigns. Regenerate final manifests once and run only invalidated checks afterward.

Use one total CPU/memory/build budget, one active GPU correctness campaign at a time unless disjoint allocations and memory/CPU isolation are explicitly established, and no competing activity during benchmark measurements. Every native owner has an isolated build directory. Share only immutable downloads/toolchain caches, not mutable runtime/tuning caches.

## 12. Known prerequisites and stop conditions

| Prerequisite | Resolution owner / acceptance | If unresolved |
| --- | --- | --- |
| Actual eight-A40 driver/topology/peer correctness | Baseline campaign with retained per-pair/per-rank evidence | No multi-GPU qualification or provider performance claim |
| Optional vendor wheel/header/license/ABI compatibility | Shared packaging gate on actual pinned artifacts | No extra or wrapper advertised as installable |
| JAX 0.12 collective ordering and owned communicator lifetime | Execution lease proof through real compiled/rank scenarios | MGMN/cuDecomp blocked; no fake pure collective |
| cuDSS full-phase host/device bounds | Documented enforcement or valid conservative bound, including analysis/cache overlap | Certified bounded-hybrid request refused; sampled memory is not a substitute |
| cuDecomp canonical pencil endpoint mapping | Exact layout/rank/packing proof and composed transform qualification | Explicit unsupported provider selection; native pencil remains available |
| cuEST normalization, actual-density K factor, matching DF forces | Independent basis/J-K/energy-gradient evidence and native stationary checks | No conservative-force/PES support claim |
| cuEST fitting rank/conditioning evidence | Explicit policy and available trustworthy metadata | Refuse a rank-bound contract that cannot be established |
| cuDensityMat extension/version/transform disagreement | Pin code and qualify requested compositions/resources | Keep conservative refusals; do not inherit release-note claims |
| A candidate lacks end-to-end benefit or new capability | Matched benchmark/capacity evidence | Do not promote or add a speculative dependency |

These are explicit implementation gates, not requests for the user to supply repository-discoverable information. The implementation owner must exhaust available code, vendor artifacts and narrow runtime experiments before reporting a blocker. A plan may name a real unresolved runtime prerequisite; it must not invent a successful resolution.

## 13. Completion criteria

A primary track is complete only when its named installed end-to-end workflow executes with correct numerical/status/derivative/resource/lifetime behavior, all affected callers and identities are migrated, unsupported routes refuse explicitly, relevant docs/examples/manifests are current, and observed qualification/benchmark evidence is retained. Preparation scaffolding, visible GPUs, passing parser tests, a skipped numerical case, a green process exit without scientific gates, or a vendor marketing result does not count.

The final implementation report must enumerate completed and blocked provider profiles, exact exercised commands/scenarios, source/dependency/backend/topology identities, failures/skips and their meaning, numerical accuracy/status, memory provenance, and phase-separated measurements. It must distinguish installation, correctness, performance, same-host distribution, multi-host distribution, and release promotion.

## 14. Primary technical references

Vendor pages define candidate APIs, not Phydrax qualification. Pin the actual selected release and archive the relevant documentation/build identity during implementation.

- [cuTENSOR primitives and compute policy](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
- [cuTENSOR accuracy, plan cache, and graph restrictions](https://docs.nvidia.com/cuda/cutensor/latest/user_guide.html)
- [cuTensorNet path injection, slicing, workspace, and network execution](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/overview.html)
- [cuDSS 0.8 advanced memory and MG/MGMN features](https://docs.nvidia.com/cuda/cudss/0.8.0/advanced_features.html)
- [JAX FFI call contract](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html)
- [cuDecomp layout and communication model](https://nvidia.github.io/cuDecomp/overview.html)
- [cuDecomp preparation and lifecycle](https://nvidia.github.io/cuDecomp/basic_usage.html)
- [cuFFTMp topology/NVSHMEM prerequisites](https://docs.nvidia.com/cuda/cufftmp/usage/requirements.html)
- [cuEST scientific scope](https://docs.nvidia.com/cuda/cuest/)
- [cuEST installation and tested configurations](https://docs.nvidia.com/cuda/cuest/installation.html)
- [cuEST basis API](https://docs.nvidia.com/cuda/cuest/c-api/basis-and-grid/basis/index.html)
- [cuEST density-fitting conventions](https://docs.nvidia.com/cuda/cuest/density_fitting.html)
- [cuEST exchange input contract](https://docs.nvidia.com/cuda/cuest/c-api/methods/exchange/index.html)
- [cuEST contracted J/K gradient conventions](https://docs.nvidia.com/cuda/cuest/python-api/methods/jkgrad/index.html)
- [cuDensityMat-JAX supported transformations](https://docs.nvidia.com/cuda/cuquantum/26.09.0/python/jax/cudensitymat_jax.html)
- [cuQuantum JAX extension release notes](https://docs.nvidia.com/cuda/cuquantum/26.09.0/python/jax.html)
- [cuOpt LP/QP methods and conventions](https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html)
- [cuSOLVERMp layouts and requirements](https://docs.nvidia.com/cuda/cusolvermp/getting_started/index.html)
- [MathDx device extensions and toolchain](https://docs.nvidia.com/cuda/mathdx/index.html)
