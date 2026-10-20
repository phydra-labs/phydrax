# Apple GPU acceleration: end-to-end implementation plan

## 0. Status, objective, and baseline

**Status: implemented and locally hardware-qualified; not published.** The original plan below records the design and acceptance obligations. Sections 12–13 and the retained M1 Max qualification/optimization records contain the actual artifacts, checks, performance, and nonclaims.

Objective: support a useful, explicitly qualified Apple Silicon scientific-computing path through `pip install "phydrax[metal]"`, starting with the available M1 Max. Preserve Phydrax's JAX arrays, scientific APIs, native substrates, transformation contracts, ordering, failure evidence, and ownership boundaries. FP32 is useful where the application's error budget admits it; installing an accelerator is not authorization to reduce a scientific precision requirement.

### Current repository transition

During planning, concurrent work changed `pyproject.toml`, `native/cuda13/pyproject.toml`, `docs/installation.md`, and `CHANGELOG.md` from JAX 0.11 to **JAX >=0.12.0,<0.13**. The lockfile still contained JAX/JAXlib 0.11.2 at the last planning inspection. This is an observed in-flight transition, not a request to undo or independently repair that work.

Implementation must start from the completed, internally consistent JAX 0.12 transition. Refresh the exact lock and source state once in the implementation worktree. Use the matched JAX/JAXlib 0.12.0 pair if that remains the canonical lock. Do not downgrade the core package, maintain a Metal-only older Phydrax branch, widen an upstream dependency range without rebuilding, or relabel historical JAX 0.11.2 qualification as current evidence.

The inspected upstream `jax-mps` 0.11.0 and `metaljax` 0.11.9 distributions both declare JAX <0.12. Neither is an installable, qualified dependency for the new core requirement as-is. A current-JAX native build and lowering audit is the first implementation dependency.

The requested `dev` merge during implementation advanced this worktree to
`13d0a3076`, requiring standard GIL-enabled CPython 3.13 and S2FFT 1.5. Metal
packaging, tooling, fixtures, and qualification now target that baseline.
Earlier Python 3.12 CPU checks and the first native GPU smoke are historical
baseline evidence, not qualification of the merged stack. The address-free
`AxisArray` representation change also invalidates affected source-derived
identity evidence; no historical identities are relabeled.


### Implementation workspace

- A subsequent `//code` uses a fresh worktree from the current branch under the repository-required `../phydra-labs/.worktrees/` location. Copy the authoritative ignored `AGENTS.md` before work. `//code--` stays in the current worktree.
- Use a neutral name such as `metal-execution`; do not use upstream library names in the branch/worktree name.
- Establish the latest merge into `dev` as the required comparison point for affected-test selection. Record the actual tree state; do not invent it from this source inspection.
- Preserve concurrent changes. No implicit merge, commit, package publication, or upstream contribution is authorized by this plan.

### Impact classification

Public packaging contract; native provider/runtime integration; optional constructor API; configuration admission; device identity; scientific qualification; benchmark evidence; authored/generated documentation. The plan deliberately avoids a new scientific array API, a global precision change, a new general-purpose compiler, and a global test-fixture rewrite.

## 1. Binding product and numerical contracts

### 1.1 Installation versus execution

1. The public extra is `metal`.
2. Install exactly one selected, tested PJRT implementation. Do not install multiple Apple plugins and let discovery order choose a provider.
3. The delivery vehicle is a lockstep, wheel-only **`phydrax-metal` companion**, derived from an existing PJRT implementation with a small, explicit patch queue. It is not a new compiler written from scratch.
4. The companion release matches the Phydrax release. It pins the exact tested JAX/JAXlib pair; the base package can retain its current supported range.
5. Publish Darwin-arm64 wheels only. Keep the companion requirement in the extra unmarked so an unsupported host fails installation rather than silently receiving CPU-only behavior. The base CPU installation remains cross-platform.
6. Installing the extra does not silently select a lower numerical precision or take over the default CPU workflow. Select the Apple runtime explicitly through the existing `RuntimeConfig.platforms` or startup environment.
7. Preserve the selected backend's actual platform token: `mps` for the inspected JAX-MPS implementation, `metal` for the inspected alternative. The extra name is not a runtime alias. Freeze one token and one provider in the selected source/build record before editing the public profile and examples.
8. Missing binaries, incompatible JAX, conflicting plugins, unsupported hardware, and unsupported operations fail with actionable evidence. No automatic CPU or alternate-plugin fallback.

The target user experience is one installation command followed by normal Phydrax configuration and explicitly typed numerical construction. For the JAX-MPS source candidate, the intended configuration is `config.configure(config.RuntimeConfig(platforms=("mps",), enable_x64=True))` before numerical construction. If the selection gate chooses the other source, use its real `metal` token consistently everywhere; never ship both spellings for one implementation.

### 1.2 Floating-point policy and integer identity

- Preserve Phydrax's default x64 enablement and highest matmul precision. Do not infer numerical policy from the installed extra, device name, or operating system.
- The mandatory scientific profile uses **float32 numerical arrays, complex64 where specified, and exact int64/uint64 metadata/address arithmetic**, with x64 enabled.
- Morton queries already require x64 for codes, sort keys, and stable IDs while accepting float32 coordinates. Retain those guards and full-width representations.
- Actual float64/complex128 device inputs, runtime constants, intermediate computation, and results must be refused consistently. A warning followed by narrowing is not a valid implementation. Explicit compile-time casts of real literals or bounded, fully constant scalar setup graphs into supported real precision are normalized after allocation admission and inlining; this is declared host preparation, not wide runtime arithmetic or general floating-point optimization. Storage-only handling, if ever exposed, must preserve every bit and cannot masquerade as supported arithmetic; it is not needed for this deliverable.
- Integer operations must remain integer operations. Never compute integer contractions through float32. Implement an exact backend route where a required consumer uses it; otherwise refuse that operation instead of returning rounded integers.
- Do not narrow gather/scatter indices without proving representability and preserving the declared out-of-bounds semantics. Test high-bit keys independently from array-index capacity.
- Existing `Float64[...]` forms and `_dtype_names` follow the active JAX canonicalization policy. Do not globally rewrite these forms, change their meaning, or confuse an x64-off construction with FP64 support. The mandatory Metal qualification runs x64-on.
- FP16/BF16 are not required for the scientific support claim. Do not preserve an unqualified dtype in the Apple profile merely because the old declarative profile listed it.
- Precision-sensitive CPU stages remain explicit CPU workflows or declared host preparation. Do not add per-operation CPU numerical fallbacks inside a program advertised as Metal numerical execution.

Metal has no native FP64 arithmetic. Software FP64, double-single arithmetic, automatic mixed CPU/GPU iterative refinement, and automatic precision selection are not included. Existing application precision policies remain the authority over acceptable accuracy.

### 1.3 Execution, differentiation, and failure

- Keep scientific arrays as JAX dynamic leaves. Do not introduce parallel MLX model/array classes or hide arrays in static callbacks.
- Preserve ordinary `jit`, `eqx.filter_jit`, batching, JVP, VJP, and higher derivatives on the mandatory workload matrix.
- Do not monkeypatch `jax.nn.gelu`, `jax.nn.dot_product_attention`, or other scientific callables. Register only platform-specific lowering rules required by the selected runtime. Source-addressed callable identity and forward-mode AD must remain intact.
- Do not silently change the global Shardy setting. The selected backend must admit the current single-device lowering emitted by the canonical JAX stack; unsupported multi-device constructs are refused.
- `eqx.error_if` and the existing Phydrax compiled-refusal paths are mandatory, including the accepted branch and the actual error branch after synchronization. Do not delete guards, substitute `checkify`, or return a plausible zero because a callback target is unsupported.
- User/error callbacks are explicit host effects, not a license to run unsupported numerical operations on CPU. Preserve effect ordering, errors, lifetimes, and executable ownership.
- Preserve deterministic/compensated sparse accumulation and its derivatives. Backend fusion/fast-math may not reassociate away compensation or invalid-lane masking.
- No multi-device, multi-host, CUDA-aware MPI, Pallas-Mosaic, Triton, cuDSS, AmgX, or QUDA claim follows from installing the extra.
- The plugin may have host-orchestrated control flow. Qualify and disclose that behavior separately from GPU arithmetic; do not call such a recurrence wholly device-resident. Do not add host synchronization to Phydrax numerical loops to accommodate it. Measure backend loop synchronization rather than hiding it.

### 1.4 Required end-to-end scope

The implementation must complete all of these paths, not just a matrix multiply:

| Required path | Numerical contract | Required observable behavior |
| --- | --- | --- |
| Classical MLP/PDE residual | Explicit float32 parameters, coordinates, residuals; x64-on | Forward, coordinate Hessian/Laplacian, parameter gradient of a second-derivative loss, and an optimizer update |
| Prepared Morton neighbor query | Float32 geometry; full-width signed/unsigned IDs and Morton codes | Exact neighbor/tie ordering, preparation reuse, refresh, masks, periodicity, invalid and capacity-overflow evidence |
| Ordered sparse execution | Float32 values; deterministic and compensated policies | Cancellation-sensitive result, fixed-order replay, JVP/VJP/curvature, failed-schedule invalidation |
| Native-JAX MACE | Coordinate, compute, reduction, and output roles explicitly float32 | Energy/forces, strain/stress where exercised, parameter derivatives, second-order action, ordered replay and refusal behavior |
| Fourier/spectral operation | Float32/complex64 | Mode and Nyquist semantics, normalization, phase, JIT and derivatives |
| Native iterative solve | Float32 operator/RHS; native matrix-free route | Convergence/true residual, iteration limit, breakdown/nonfinite status, mathematical and admitted algorithmic derivatives |
| Temporal recurrence | Float32 state/time/parameters | Analytic terminal value and derivative, status, bounded trajectory/workspace, separately reported control-flow cost |

Required paths cannot be silently dropped after a failing test. Fix the owning backend/substrate or report the precise unresolved prerequisite; any reduction of this matrix requires explicit approval. A passed path does not qualify every consumer of that substrate.

The following remain explicit nonclaims: arbitrary float64 scientific workflows; every initializer/activation/model architecture; all Morton hierarchy/voxel/contact/AMR users; implicit temporal methods requiring an unimplemented decomposition; and CUDA-specific accelerated routes. These are bounded support claims, not hidden fallback implementations.

## 2. Gate A: select and bind the existing PJRT substrate

### 2.1 Decision procedure

Evaluate `jax-mps` and `metaljax` as **source candidates**, not as compatible JAX 0.12 wheels. Use separate environments, source/build directories, and processes. Never load both in one qualification process.

The primary source candidate is `jax-mps`, because it was the requested concrete option and has a small explicit PJRT/StableHLO-to-MLX boundary. The alternative already has callback and JAX 0.11.2 integration work that may reduce the current-JAX adaptation. Do not choose by a test badge, stars, one ResNet result, or an assumed difference in effort.

Before core integration, select the candidate that meets the following gates with the smaller maintainable correction surface:

1. Rebuild against the exact current JAX/JAXlib PJRT, XLA, LLVM/MLIR, and StableHLO dependency chain. Execute current emitted StableHLO, not an old hand-written module alone.
2. Initialize without global callable replacement, global compiler-policy changes, or unsafe external library discovery.
3. Execute `eqx.filter_jit` with a real guarded Phydrax operation and propagate an invalid-input error correctly.
4. Execute FP32 arithmetic and exact int64/uint64 operations in one x64-on program; refuse actual f64/c128 at every boundary.
5. Demonstrate ordinary JVP, reverse mode, and a second-order composition without a custom-VJP-only replacement for the user's callable.
6. Preserve duplicate-index, stable-sort, ordered-reduction, and nonfinite semantics required below.
7. Produce truthful completion, buffer ownership, memory, and no-copy interoperability behavior.

Freeze one source commit, native dependency closure, patch list, runtime platform, and build identity after this gate. A failed gate is not waived by labeling the extra experimental. If both candidates need substantive missing functionality, complete that functionality at the existing backend owner; do not disguise a prototype as completed support. Changing the selected source after freezing it requires updating this file map and invalidates affected evidence, not a runtime fallback switch.

### 2.2 Known source corrections to evaluate

**JAX-MPS source paths** (upstream paths inside the pinned build tree, not existing Phydrax files):

| Upstream owner | Planned correction or verification |
| --- | --- |
| `src/jax_plugins/mps/__init__.py` | Replace its installation role with the companion-owned loader. Remove system/Homebrew search, permissive mismatch handling, duplicate swallowing, Shardy mutation, and global JAX function patching from the shipped path. |
| `scripts/setup_deps_llvm.sh`, `CMakeLists.txt`, `macos-deployment-target` | Bind the current-JAX dependency chain and reproducible Darwin deployment target; make missing runtime metallib a build failure. Do not run an unpinned upstream bootstrap script as a release build. |
| `src/pjrt_plugin/mlx_buffer.cc`, `mlx_executable.cc`, `type_utils.cc` | Refuse unsupported f64/c128 consistently for transfer, constants, computation and output; preserve exact integer representation and strides/layout semantics. |
| `src/pjrt_plugin/ops/linalg.cc` | Remove integer-through-float behavior. Honor the supported FP32 precision request. Refuse CPU LAPACK routes in the admitted Metal numerical profile unless a genuine GPU implementation is provided. |
| `src/pjrt_plugin/ops/gather_scatter.cc`, `reduction.cc`, `sort_fft_complex.cc` | Complete required batching, bounds, duplicate, full-width key, stable tie, complex64/FFT, and accumulation contracts. Backport only the exact needed upstream fixes with provenance. |
| `src/pjrt_plugin/ops/control_flow.cc`, `stablehlo_parser.cc` | Current single-device IR admission, guarded callback/error support, effect order, correct predicates, and explicit host-control behavior. Unknown targets remain errors. |
| `src/pjrt_plugin/pjrt_buffer.cc`, `pjrt_event.cc`, `pjrt_executable.cc`, `pjrt_device.cc` | Completion and error propagation, lifetime/donation behavior, supported no-copy boundaries, allocator counters, truthful unsupported compiler-analysis APIs. |

**Alternative source paths, only if selected at Gate A:** `src/jax_plugins/metal/__init__.py`; `plugin-native/metal/metal_dtypes.cc`, `metal_client.cc`, `metal_buffer.cc`, `metal_executable.cc`, `metal_lowering.cc`; and `plugin-native/runtime/{host_callback,host_lapack,control,dtypes}.cc`. Remove f64/c128 ingress narrowing and any opt-in demotion bypass from the supported profile; adapt platform lowerings to JAX 0.12; refuse hidden CPU numerical routes; preserve callback ordering and bound callback-registry lifetime to live executables. Do not mark process-local callback indices as portable export identities. Audit precision environment mutations and M5-specific architecture overrides rather than carrying them into an M1 build.

The alternative build uses its existing native build system; do not port it to CMake just to make the source candidates look alike. Only one source implementation and its actual build path ship.

### 2.3 OpenFPM and in-house boundary

Do not import OpenFPM's CUDA/HIP-to-SPIR-V/MoltenVK toolchain. It is a bounded C++ source portability layer, not a PJRT backend or a way to port NVIDIA binary libraries and Pallas kernels. Reuse its lessons: compiler-derived ABI layout, bit-preserving storage, explicit atomic-scope/type refusal, and visible synchronization.

Do not add bespoke Phydrax Metal numerical kernels before the required ordinary-JAX paths are correct and measured. If a measured bottleneck later justifies a kernel, give it a separate accepted plan under the owning substrate, with a real device-buffer/lifetime interface and derivative/resource evidence. This deliverable does not depend on that later optimization.

## 3. Packaging, binary provenance, and publication file map

**New paths below are proposed files.** Packaged dylibs/metallibs and build manifests are generated outputs, not hand-authored source or committed fake artifacts.

| File | Planned change |
| --- | --- |
| `pyproject.toml` | Add `metal = ["phydrax-metal==0.2.2"]` for the inspected release, using the actual matching release at implementation. Do not add upstream Apple plugin dependencies alongside it. Add the developer path source, `metal_native` test marker, and `native/metal` first-party typing coverage. Keep base/CUDA JAX 0.12 policy. |
| `uv.lock` | After the JAX upgrade owner has completed the core lock, resolve the one selected companion and exact current-JAX pair. Preserve unrelated optional extras. Do not create a second application lock or rewrite it manually. |
| **New `native/metal/pyproject.toml`** | Lockstep project metadata, standard GIL-enabled CPython 3.13 support, exact qualified JAX/JAXlib dependency pair, one PJRT entry point, Darwin-arm64 wheel packaging, and license inclusion. No dependency on the unpatched upstream Python plugin. |
| **New `native/metal/_build_backend.py`** | Follow the existing wheel-only backend convention: refuse sdist hooks; validate the supported build host and complete native inputs; package only a verified native build. Keep every first-party Python annotation complete. |
| **New `native/metal/CMakeLists.txt`** | For the primary source candidate, configure the pinned native source/dependencies and install the dylib/metallib/manifest. Use this only if that is the selected native build system; the alternative uses its existing native build entry point instead. No general multi-provider build abstraction. |
| **New `native/metal/sources.lock.json`** | Canonical, content-hashed source closure: source URLs, exact commits/archive hashes, dependency/toolchain inputs, deployment target, and patch order/hashes. No schema-generation/version field. Missing hashes block release. |
| **New `native/metal/patches/`** | Reviewable patches grouped by actual invariants: current-JAX ABI/lowering, dtype refusal, exact integer/index semantics, effects/lifecycle, and runtime packaging as needed. Each has a rationale and upstream source identity. Do not vendor an untraceable whole fork. |
| **New `native/metal/python/phydrax_metal/__init__.py`** | Package metadata only; no device discovery, Phydrax import, numerical initialization, or public alternative scientific API. |
| **New `native/metal/python/phydrax_metal/plugin.py`** | Narrow JAX registration and validated package-local loading. No root Phydrax import during JAX discovery, no global function monkeypatches, no broad exception suppression, no system-library probing, no duplicate acceptance. Private JAX integration is narrowly pinned and tested. |
| **New `native/metal/python/phydrax_metal/py.typed`** | Include the companion's truthful first-party typing marker. |
| **Generated companion `lib/` and build manifest** | Ship the selected PJRT dylib, all required non-system runtime libraries and metallibs, and source/patch/toolchain/build identities. Keep private native dependencies private; system Apple frameworks remain system references. |
| **New `tools/build_metal_wheels.py`** | Bounded build from an explicitly provisioned interpreter/toolchain; isolated build directory; exact source acquisition; command evidence; Mach-O/deployment/signature/rpath audit; wheel hash and manifest even on audit failure. No implicit package installation in the builder. |
| **New `tools/check_metal_extras.py`** | Check built wheel metadata and resolver results for the supported host, unsupported hosts, base CPU install, current JAX bounds, absent native wheels, and companion version mismatch. Test actual resolution; pip cannot express arbitrary conflicts with independently installed plugin packages. |
| **New `tools/qualify_metal_wheels.py`** | Install built artifacts in a fresh isolated environment on real M1 hardware; verify installed-package provenance; execute the mandatory runtime/science/lifecycle cases; reject skipped, missing, duplicate, failed, or mismatched cases even if a child returns zero. Retain per-case results and exact software/hardware identity. |
| `tools/check_typing.py` | Add `native/metal` to the first-party Python inventory; keep pinned ty/Ruff policy. Do not exempt the companion loader from typing because it calls a dynamic provider. |
| `NOTICE`, `LICENSES/` | Preserve selected upstream authors, license text, patch provenance, and the notices of every redistributed/static native dependency. Reuse existing exact license texts where applicable; add missing texts using the established naming convention. Do not attribute the upstream implementation to Phydrax or label the binary PNPL-only. |
| **New `native/metal/README.md`** | Developer build prerequisites, exact source pinning, deployment target, wheel inspection/signing, clean qualification, loader contract, patch update process, and publication order. |

### Planned tool interfaces

- `tools/build_metal_wheels.py --python PATH --build-dir PATH --wheelhouse PATH --output PATH`: use the explicitly provisioned build interpreter and the one canonical source lock; write the audited wheel plus build evidence. The builder does not choose a provider dynamically or install its own dependencies.
- `tools/check_metal_extras.py --phydrax-wheel PATH --wheelhouse PATH --output PATH`: inspect wheel metadata and run the supported/unsupported platform resolution matrix without loading a GPU. Record resolved versions and failed resolutions independently from runtime results.
- `tools/qualify_metal_wheels.py --python PATH --phydrax-wheel PATH --companion-wheel PATH --work-dir PATH --output PATH`: create a fresh environment using the supplied interpreter, install the exact artifacts, run the required installed-wheel scenarios serially, and retain fail-closed evidence. The companion's recorded native platform is an asserted expectation, not an autodetection fallback.

Each tool exits nonzero for an incomplete or failed obligation and retains bounded command/error evidence. There is no `--allow-fallback`, `--skip-required`, or assumed-success flag. Package-index credentials and signing secrets are not written to evidence.


### Binary release rules

- Establish the minimum supported macOS from the built dependency closure and a real test host. The inspected upstream wheel target is macOS 14, but a build on a newer OS is not evidence for that minimum. Do not claim macOS 14 if the selected current-JAX/MLX build requires newer APIs.
- Inspect Mach-O architecture/deployment target and dependency/load commands with the owning tools (`lipo`, `vtool`, `otool`), verify the required PJRT exports and runtime assets, then verify code signatures after all relocation/repair steps. Reject absolute build/Homebrew paths and unexpected external native dependencies.
- A missing metallib is fatal, not a warning followed by a wheel that imports but cannot execute.
- Register below CPU priority to preserve automatic CPU selection. Priority alone does not prevent JAX from discovering/loading optional plugins. Require `import phydrax` and host-only configuration/diagnostics to remain device-inert; during CPU backend discovery, separately record any JAX-driven plugin loading and prove there is no Metal numerical execution or global semantic mutation. Do not promise that priority alone prevents `dlopen`.
- If the implementation introduces conditional registration to avoid CPU-side loading, it must preserve explicit configuration before initialization and have a tested JAX-supported selection boundary. Do not add a hidden second environment/configuration parser.
- Direct raw-JAX use of arbitrary competing plugins is not qualified. Phydrax's explicit Metal selection must reject conflicting installed entry points before JAX can suppress a registration error and choose another implementation.
- Publish the qualified companion before the core release advertising the extra. Publishing packages requires its own explicit authorization. If a new core release is necessary for lockstep matching, use the repository's release process, not a fabricated companion-only version.

## 4. Runtime, identity, and capability file map

| File | Planned change |
| --- | --- |
| **New `phydrax/_metal_environment.py`** | Own immutable host facts, pure compatibility assessment, explicit collection, installation admission, and actionable Metal issues. Follow the CUDA facts/assessment separation without copying ELF/CUDA-major logic. |
| `phydrax/_config_state.py` | In `_admit`, invoke host-only Metal installation validation only for an explicit selected Apple runtime, including startup requests decoded from environment. Validate before applying any setting; preserve `ConfigurationApplicationError` field/applied-state semantics. No discovery from `snapshot()` or CPU configuration. |
| `phydrax/_config_environment.py`, `phydrax/_jax_config.py` | Reuse the current platform parsing, precedence, projection and readiness functions. No new `metal=True`, no automatic x64-off, no new parallel config path. Change these files only if an actual current-JAX adapter gap is established; any additional setting must live in the sole owning adapter and be explicit. |
| `phydrax/_execution_device.py` | Recognize the selected Apple runtime from actual platform/client/build identity, not `device_kind` display text. Preserve raw native platform identity and unknown-plugin refusal. Do not map all generic `gpu` devices to Apple or rename an `mps` device to `metal`. |
| `phydrax/_execution_runtime.py` | Reuse discovery/inventory and single-device execution groups. Expected unchanged structure: consume corrected vendor identity and actual allocator evidence. No synthetic devices or mixed CPU/GPU mesh. |
| `phydrax/_execution_sampling.py` | Reuse actual PJRT allocator counters and `None` for unavailable measurements. Prefer fixing incomplete PJRT reporting at the selected backend; change the adapter only for an evidenced contract gap. Never turn unimplemented compiler memory analysis into zero or a fabricated estimate. |
| `phydrax/backends/distributed.py` | Cleanly update `APPLE_METAL_PROFILE` to the actual selected provider/platform and admitted numerical dtypes/scope. Do not add an alias for the old decorative `jax-metal` identity; do not widen `JAX_NATIVE_DISTRIBUTION` into an Apple multi-device claim. |
| `phydrax/backends/__init__.py` | Explicitly export the Metal facts/issues/report and the two diagnostic operations below, preserving lazy optional-dependency behavior and canonical symbol ownership. |
| **New `phydrax/backends/_metal_profiles.py`** | Own the Metal candidate capability declaration/profiles and exact support coordinates using existing `CapabilityDeclaration`, `SupportTuple`, `QualificationRuntimeIdentity`, and evidence types. No second release registry. |
| `phydrax/qualification/_builtin_catalog.py` | Register the one owner-local Metal capability producer lazily. Do not mark a candidate as released from device visibility or package installation. |
| `phydrax/qualification/_builtin_sources.py` | Record the selected upstream source/provenance where required by the existing source ledger. Do not create a second source inventory. |
| `phydrax/backends/_types.py`, `phydrax/_cuda_environment.py`, `phydrax/_cuda_native.py` | Intentionally unchanged. `BackendAvailability.environment_issues` is currently CUDA-specific; do not insert Metal records into that field or launch an unrelated generic-provider refactor. Metal runtime diagnostics have their own canonical owner. |

### Diagnostic API

Export through `phydrax.backends`:

- `MetalEnvironmentFacts`, `MetalEnvironmentIssue`, `MetalEnvironmentReport`.
- `MetalIssueCode` and `MetalIssueSeverity`, declared once with the existing selector/validation conventions.
- `assess_metal_environment(facts: MetalEnvironmentFacts) -> MetalEnvironmentReport`: pure, deterministic, no imports/probes beyond its supplied facts.
- `metal_environment(*, initialize_backend: bool = True) -> MetalEnvironmentReport`: explicitly collects host/runtime facts; `False` must not initialize any backend.

Facts distinguish host OS/architecture, installed distributions and entry points, selected platforms, exact core/companion/JAX/JAXlib versions, binary/source/build identity, initialization/probe state, observed device/runtime capabilities, and failure reason. Keep metadata static and canonically ordered. No arrays, live clients, compiler objects, or mutable registry owners in these records.

Use issue families for unsupported host, missing/mismatched companion, incompatible JAX/PJRT build, conflicting plugins, missing native assets, unsafe runtime-policy overrides, and failed initialization. **x64 enabled is not an unsafe-policy issue**: it is required by the scientific profile's integer paths. Report whether a device was actually probed; environment coherence is not numerical qualification.

A configuration refusal must happen before backend creation when metadata already proves it invalid. Native loading/compilation failures must preserve their original cause; do not catch them and report CPU success.

### Identity and memory invariants

- The selected provider, build, patch set, precision policy, hardware/OS, and actual runtime settings participate in Metal qualification identity. Adding a real Apple vendor/profile intentionally changes the corresponding Apple execution identity; never reuse the old decorative profile's ID.
- No new static/dynamic partition or model field is needed for Metal selection. CPU/CUDA payloads and plans are not migrated merely for a new optional provider.
- Keep host RSS, allocator use/cache/limit, logical live-array bytes, and compiler estimates distinct. Unified-memory allocations overlap host/device views; never add them to manufacture physical memory capacity.
- Unavailable compiler temporary/code-byte evidence remains unavailable, with a reason. Retain actual logical sizes and sampled allocator peaks without presenting samples as a bound. An execution policy demanding unavailable certified capacity must refuse.
- Use `benchmarks/_runtime.py` identity/timing helpers unchanged where possible. LSP found broad usage of `capture_environment`; do not change every benchmark record format to add Metal-only details. Compose its identity with the companion's exact runtime/build evidence in the Metal record.

## 5. Scientific construction and consumer file map

### 5.1 Explicit MLP parameter dtype

This is a concrete missing public construction capability, not a Metal-specific model implementation.

| File | Planned change |
| --- | --- |
| `phydrax/nn/layers/_linear.py` | Add keyword-only `dtype: DTypeLike | None = None`. Validate/normalize the explicit dtype through the existing dtype owner before numerical allocation. Pass the effective dtype to weight, RWF-scale and bias construction. Preserve the old no-argument path, key splitting, parameter roles, field order and numerical defaults. Do not add a stored dtype field or cast inputs inside every call. |
| `phydrax/nn/models/_mlp.py` | Add the same optional construction argument and pass it to every `Linear`, including zero-hidden-layer, final-layer and residual-projection paths. Preserve scan packing, KFAC layout, dropout structure and keys. The parameter describes parameter construction; it is not a promise to cast arbitrary inputs, callbacks, or all stochastic internals. |
| `phydrax/nn/_initializers.py` | Reuse the current JAX initializer registry. Do not add another initializer vocabulary or rewrite aliases as part of this task. Modify only if a truthful type annotation is needed for the dtype argument. |
| `tools/pinn_model_benchmarks.py` | Add an explicit float32 MLP case/option and construct all evaluation coordinates/constants in the workload dtype. Its current float64 evaluation grid is a real blocker. Do not advertise that every other benchmark architecture accepts the new option; reject unsupported architecture/dtype combinations rather than ignore it. |
| `docs/api/nn/layers.md`, `docs/api/nn/architectures.md` | Document construction dtype, unchanged defaults, RWF/bias/projection behavior, active x64 policy and the distinction from accumulation/compute policies. |

The new dtype path must be additive, not a cleanup of unrelated constructor flags or model structure. `Linear` and `MLP` have many cross-module consumers; LSP references were inspected. Existing callers omitting dtype remain intentionally unchanged and retain their existing initialization behavior. Update every caller that is newly advertised as a Metal example or benchmark to pass explicit precision.

Active dropout deserves a separate negative/compatibility check: its current `jr.bernoulli` probability is a Python scalar, and x64-on intermediate dtype must be observed rather than assumed from the input array. Do not silently change RNG precision or addressing to make it run. The mandatory PDE example uses no dropout; the published support matrix must not imply active dropout support unless its existing contract is preserved and qualified. Orthogonal initialization similarly needs a real decomposition or explicit host preparation; it is not evidence of GPU factorization.

Trace source/identity consumers before changing constructors or imported initializer helpers. Preserve default numerical leaves and PyTree structure, but do not promise unchanged source-derived IDs merely because values match. Regenerate only affected identity-bearing artifacts through their owners; never pin historical source payloads by hand.

### 5.2 Existing numerical owners

The normal action for these files is **reuse and qualify**. Make an implementation change only when the required consumer exposes a real dtype/backend defect; fix its owning invariant without adding `if metal` scientific branches.

| Existing owner | Required use and boundary |
| --- | --- |
| `phydrax/discretization/spatial/_morton.py`, `_neighbor_sources.py`, `_neighbor_query.py` | Prepared JAX neighbor path with float32 geometry and exact full-width keys. Retain x64 guards, stable-ID ordering, capacity refusal, source identity, and refresh checks. |
| `phydrax/sparse/_execution.py`, `_ops.py`, `_streamed.py`, `_key_groups.py` | Native deterministic/compensated reductions and their derivative paths. Preserve `two_sum`, seeded correction, valid-zero derivatives, invalid-lane masking and packed-key identity. |
| `phydrax/nn/atomistic/_mace.py`, `_mace_prepare.py`; `phydrax/atomistic/_graph.py`, `_feature_execution.py` | Native-JAX MACE, not accelerated coupling. Select float32 for every numerical role, retain source revision/preparation checks, forces from energy derivatives and ordered geometry/feature reductions. |
| `phydrax/_spectral/_fourier.py` | Float32/complex64 Fourier resampling, mode transfer, phase and derivatives. No replacement FFT convention. |
| `phydrax/linalg/backends/_native_krylov.py`, `phydrax/linalg/_runtime.py`, `_results.py` | Native iterative operator solves, true residual/status and derivative validity. No local solver loop or direct dense fallback. |
| `phydrax/solver/_diffrax_backend.py`, `phydrax/_temporal_precision.py` | Existing explicit/fixed recurrence with float32 state/time/coefficient roles and native failure evidence. Retain precision matching and sensitivity contracts. |

**Approved MACE representation decision:** the user selected explicit float32 coupling-basis storage with distinct, honest fingerprints, rather than static high-precision byte storage or a separate prepared-model vocabulary. Native host certification retains float64 arithmetic; runtime coefficients use the requested compute dtype. Float64 remains the default. The actual rounded coefficient representation owns identity and replay. Requested equivariance tolerance, measured residual, and the separately derived storage-rounding bound remain distinct evidence. This is an intentional precision/representation change, not an identity-preserving typing cleanup.

Particle measures retain float64 accumulation by default. The Metal atomistic profile explicitly selects float32 mass accumulation through the canonical measure/particle/structure preparation APIs; it must not silently lower the accuracy of existing CPU particle workflows.

Do not qualify all hierarchy geometry from one neighbor query. `MortonAddressPlan.cell_geometry`, sparse voxel centers, and some vortex/hierarchy callers currently construct float64 bounds under x64-on. Their broader dtype contracts remain outside the declared neighbor profile unless individually brought through the same complete change/qualification process. Do not remove guards or globally downcast their geometry.

Keep `phydrax/backends/atomistic.py` targets (`cuda`, `cpu_interpret`) and their architecture admissions unchanged. No `metal` target is added to a CUDA Pallas implementation. Keep the native CUDA providers and MPI contracts unchanged.

## 6. Permanent tests and exact qualification scenarios

### 6.1 New and extended test files

| File | Consumer-visible contracts |
| --- | --- |
| **New `tests/unit/test_metal_environment.py`** | Pure assessment, explicit CPU selection, unsupported host, missing/mismatched companion, current-JAX mismatch, competing entry points, canonical ordering, unprobed versus observed state, unsafe override refusal. x64-on must not be refused solely for enabling int64. |
| **New `tests/unit/test_metal_build_tools.py`** | Real archive/Mach-O metadata fixtures for missing runtime assets, wrong architecture/deployment target, unexpected dependency/rpath, incomplete source lock and manifest/binary disagreement. No assertions on source spelling or internal call forwarding. |
| **New `tests/unit/test_metal_extras.py`** | Built metadata and solver outcomes: exact companion/current-JAX selection, unsupported-host/missing-wheel failure, and base CPU independence. |
| **New `tests/unit/test_metal_wheel_qualification.py`** | A qualification result with a skipped/missing/duplicated required case, wrong device/provider/build, stale evidence or nonzero child exit cannot pass. Test aggregation semantics, not mock output echoes. |
| **New `tests/integration/test_metal_import_contract.py`** | Root/config/host diagnostics stay device-inert; selected Metal initialization leaves CPU configuration and callable identity intact; ambiguity refuses before wrong-provider use; explicit CPU still computes correctly with the companion installed. |
| **New `tests/integration/test_metal_runtime.py`** | Real-device dtype refusal at transfer/constant/result boundaries; exact integers, primitive transforms, compiled errors, callback effects, completion, and precision controls. |
| **New `tests/integration/test_metal_science.py`** | Separately collected end-to-end scenarios from the required matrix, with actual device placement, scientific/status checks, and independent or separately executed CPU references. |
| **New `tests/integration/test_metal_native_lifecycle.py`** | Repeated compilation/execution, input reuse versus explicit donation, buffer alias/copy contracts, callback/executable release, allocation failure, and process exit with live/in-flight state. Use bounded subprocesses for destructive lifecycle cases. |
| **New `tests/unit/nn/test_mlp_precision.py`** | Explicit FP32 parameters/output and higher derivatives with x64-on; RWF/bias/no-bias, residual projection, zero hidden layers, homogeneous scan and non-scan paths; invalid dtype refuses before allocation. |
| `tests/unit/nn/test_initializers.py`, `test_model_array_roles.py` | Extend genuine dtype/parameter-role contracts where setup is shared. Retain default initialization behavior and independently meaningful existing cases. |
| `tests/unit/test_execution_resource_admission.py` | Extend `test_accelerator_vendor_comes_from_the_loaded_runtime` with admitted Apple identity and spoofed/unknown runtime refusal; test actual platform/vendor resource admission. |
| `tests/unit/test_execution_memory_sampling.py` | Add Apple allocator evidence only if its adapter is touched; unknown values remain unknown and overlapping unified-memory observations are not totaled. |
| `tests/unit/test_configuration_runtime.py`, `test_configuration_environment.py` | Explicit Metal admission before effects, environment/config precedence, late-request refusal, and no global numerical reset. Preserve existing CPU cases. |
| `tests/unit/test_distributed_execution_substrate.py` | Update the Apple profile's real identity/admissible scope/dtype contract. Retain refusal of float64 and multiple-device scopes; no new fake distributed Metal case. |
| **New `tests/typing/cases/metal_environment.py`** | New diagnostic API types and invalid kinds under pinned ty. |
| **New `tests/typing/cases/mlp_precision.py`** | `DTypeLike` constructor inputs and truthful model results; use `assert_type` and narrow negative expectations. |
| `tests/typing/installed/consumer.py` | Extend installed-wheel API consumption for the new constructors/diagnostics without requiring the optional GPU provider. |

A test-only support file may be added under `tests/_support` only for a repeated real invariant such as isolated process/result validation or adjoint comparison. Do not create a second scientific implementation, a broad soft-assert scenario wrapper, or permanent source-text tests.

### 6.2 Required edge cases

**Installation and initialization**

- Base CPU installation without Metal; companion installed but CPU explicitly selected; explicit Metal with matching artifacts.
- Unsupported OS/architecture, unsupported minimum OS, missing native assets, mismatched core/companion/JAX/JAXlib, and a foreign plugin registering the selected platform.
- No successful-looking result from a failed plugin import, missing callback hook, invalid binary, or partially registered lowering table.
- Repeated initialization is deliberately idempotent only for the exact same owned registration; do not swallow an unrelated `ALREADY_EXISTS` error.
- Current JAX default Shardy/compiler settings and ordinary `jax.nn` callables remain semantically unchanged.

**Precision and primitives**

- Explicit float32 and complex64 arrays with x64-on; declared output dtype is checked, not inferred from finite values.
- Float64/complex128 host transfer, closed-over constant, conversion, intermediate and result each refuse without narrowing. Use values not representable in float32 so a plausible truncation cannot pass.
- Exact int64/uint64 values beyond 2^32 and 2^53, signed boundaries, high valid Morton bits and shifts; key order must not pass through floating point.
- Gather/scatter bounds, masked invalid indices, duplicate destinations, batched window scatter, stable ascending/descending ties and lexicographic keys.
- NaN, infinity, signed zero, overflow and subnormal behavior. Do not infer numerical compliance from float32 storage. If hardware flushes subnormals, record the limitation and refuse scientific claims depending on gradual underflow; do not relax a required nonfinite/status contract.
- Threefry typed-key split/fold-in and deterministic raw bits; do not silently switch PRNG implementations. Other PRNGs are admitted only with their own exact state/sequence evidence or are refused.
- Accepted and failing compiled `eqx.error_if`, including within nested conditional/loop transformations. Check the observed synchronized exception, not merely construction.
- Highest-precision FP32 contraction against an independent FP64 host reference; no implicit FP16/TF32 execution or compensation-destroying reassociation.

**Scientific scenarios**

- PDE: explicit float32 construction, batched input derivative, Hessian/Laplacian and parameter gradient of that residual; an optimizer step changes the model and agrees with the CPU reference update.
- Neighbors: empty active set, masks, periodic seam, coincident/equidistant points, signed/unsigned stable-ID ties, invalid geometry, boundary radius, candidate/pair overflow, stale source plan, and refresh preserving declared precision.
- Sparse: cancellation across fragments, duplicate/invalid stable IDs, zero-valued valid events with nonzero derivatives, invalid NaN padding with no contribution, fixed replay and derivative failure invalidation.
- MACE: all-role FP32, atom permutation/padding, energy-force consistency, cell/strain and parameter derivatives, second-order action, unknown species and unsupported radial support; retained source/preparation identity.
- FFT: representable complex modes, even Nyquist split/merge, multiaxis phase shift, real/complex dtype, normalization and linear adjoint/derivative checks.
- Linalg: small well-conditioned operator against a separate host solve; genuine nonconvergence, breakdown and nonfinite inputs; true residual checked at the consumer and failed derivative validity not replaced by finite zeros.
- Temporal: analytic decay/oscillation with parameter derivative, zero/short/long trajectories, termination/status, bounded workspace and explicit backend control-flow/synchronization accounting.

### 6.3 Reference and tolerance policy

Generate host input data once, round to the declared input dtype once, and give identical data to the Metal and CPU runs. CPU FP64 references must operate on those same rounded inputs, not a different better-conditioned problem. Keep CPU references outside the timed GPU path and outside the selected Metal process when needed to prevent accidental fallback or implicit dtype construction.

Use analytical formulas, independent host arithmetic, existing native scientific references, adjoint identities and forward/error bounds. Retain fixed model parameters when comparing higher derivatives; do not compare different random initializations and call it backend error.

Existing periodic float32 MACE cases provide a 1e-4 relative precedent for the same exercised workload, not a universal tolerance. The existing GPU matmul regression uses a relative error below 1e-5. Existing complex128 FFT and float64 sparse 1e-12/1e-9 gates are not float32 gates: retain them unchanged and derive a separate bound for the explicit lower-precision scenario before measuring it. Do not loosen tolerances after observing an unexplained failure.

Use deliberate faults for shared comparison helpers where an incorrect mask, order, failure status or zero derivative could otherwise pass. No statistical retry-until-green acceptance.

### 6.4 Conservative affected CPU regression set

Start with changed-file tests and the following direct families; select exact affected nodes where files contain unrelated scenarios:

- Configuration: `tests/unit/test_configuration_{runtime,environment,records,resolution}.py`.
- Execution: `tests/unit/test_execution_resource_admission.py`, `test_execution_memory_sampling.py`, and the existing Apple-profile case in `test_distributed_execution_substrate.py`.
- Neural construction: `tests/unit/nn/test_initializers.py`, `test_model_array_roles.py`, `test_tensor_value_sizes.py`, `test_low_rank_adaptation.py`, `test_parameter_transforms.py`, `test_model_execution_contracts.py`, and `test_models/{test_scan_behavior,test_complex_output}.py`.
- Direct structural consumers: affected MLP cases in `tests/unit/solver/test_kfac_layout.py`, `tests/unit/nn/test_low_rank_adapter_artifact.py`, and `tests/unit/ml/test_artifacts.py`; the compact Poisson consumer in `tests/integration/test_kfac_pinn.py` if the new dtype path reaches that workflow.
- Spatial: `tests/unit/discretization/spatial/{test_morton,test_neighbor_query,test_prepared_neighbor_sources}.py`, especially full-width tie ordering, periodicity, invalid geometry, overflow and refresh.
- Sparse: the affected cancellation/order cases in `tests/unit/test_sparse_streamed.py`, derivative/failed-schedule cases in `test_sparse_streamed_derivatives.py`, and `test_sparse_substrate.py::test_packed_key_groups_compare_every_word_and_int64_event_order`.
- Atomistic: affected float32/ordered cases in `tests/unit/atomistic/test_mace.py`, `test_mace_execution.py`, and benchmark contract cases in `test_benchmark_tool.py`.
- Spectral: the affected mode/Nyquist/JIT/derivative cases in `tests/unit/signal/test_fourier_resampling.py`.
- Iterative/temporal: affected cases in `tests/unit/linalg/{test_algorithmic_work,test_bicgstab,test_linear_solve_derivative_contract,test_checked_solve_contracts}.py` and `tests/unit/solver/{test_time_integrators,test_temporal_derivative_contract}.py` if those owners change.
- Packaging/catalog: new Metal tool tests, existing CUDA build/resolver unit tests, import-boundary tests, and `tests/unit/qualification/test_capability_catalog.py` when registration changes.
- Public typing fixtures and installed-wheel typing for the new APIs.

Intentionally do not run every application importing `Linear` merely because a keyword-only optional argument was added while the no-argument path is unchanged. Expand the selection if source/default/role/identity changes actually affect those consumers. Do not run the full suite under x64-off. Do not rerun the unrelated CUDA hardware campaigns on a Mac; record hardware-unavailable specialized gates honestly.

## 7. Runtime smoke and performance file map

A passed test suite is not the runtime smoke. Run the installed wheel through the advertised example and the complete scientific paths; observe the actual arrays, outputs, statuses, derivatives and device identities.

| File | Planned action |
| --- | --- |
| **New `examples/metal_scientific_execution.py`** | One executable, bounded FP32 PDE/MLP workflow using normal Phydrax configuration and APIs; explicit x64-on, actual Apple device verification, residual/higher derivative and optimizer update. Report meaningful numerical results and refuse unsupported setup. No CPU fallback or fake output. |
| `tools/pinn_model_benchmarks.py` | Explicit MLP dtype path, same-input CPU reference and phase-separated timings; add existing source/environment identity utilities rather than a new generic runner. Retain default scientific metrics and other architecture behavior. |
| `tools/spatial_query_benchmarks.py` | Reuse `--scenario nearest --dtype float32`; separate preparation, refresh and repeated query. Capture full build/environment/runtime identity and validate exact neighbors/status. Do not use the default float64 primitive scenario as Metal evidence. |
| `benchmarks/streamed_relation_scaling.py` | Add explicit dtype selection and a separately justified FP32 reference bound while retaining the existing higher-precision default. Sweep bounded receiver/degree/channel/tile capacity and both ordered policies. |
| `benchmarks/mace_execution_scaling.py` | Reuse native `streamed`/`dense` and model `exact`/`streamed` routes with all-role float32. Make accelerated `--target` required only when an accelerated route is selected; a native-only run must not resolve a CUDA/interpreter target. Do not introduce a fictional Metal Pallas target or label native runs `cpu_interpret`. Preserve existing explicit CUDA commands. |
| `tools/spectral_pde_benchmarks.py` | Add a declared float32/complex64 scenario with independent host formula and existing identity/timing helpers, retaining strict high-precision scenarios. |
| `tools/time_integrator_benchmarks.py` | Select the qualified explicit/fixed recurrence rather than running every implicit method; add dtype/route selection only where missing, retaining each route's current contract and reference. |
| **New `tools/native_krylov_benchmarks.py`**, only if no current driver covers the selected route | One focused native float32 iterative-solve driver reusing `benchmarks/_runtime.py`; include status/true-residual checks, preparation reuse, cold/warm execution and scaling. Do not introduce a generic workload framework. |
| `benchmarks/_runtime.py` | Reuse `capture_environment`, `capture_benchmark_identity`, `measure_lower_and_compile`, `measure_repeated`, `compiler_evidence`, and logical-byte accounting. Expected unchanged public record shape. Metal-specific evidence is composed in the campaign record. |
| **Generated `benchmarks/m1_max_qualification.json`** | Retain only actual installed-wheel results, exact device/OS/runtime/build identities, selected scenarios, commands, failures/nonclaims and raw timing references. Never prepopulate it with planned pass results. |

### Measurement protocol

1. Select bounded small/medium/large capacities before the campaign. Record actual M1 memory capacity and reserve room for the OS; do not assume a particular M1 Max RAM configuration.
2. Use one GPU-owning process; no simultaneous tests, native builds, agents doing numerical work, or competing benchmarks. Use stable shapes/dtypes and the owning cache policy.
3. Record host preparation/topology/model construction, lowering, compilation, first execution, warm samples, explicit transfer/synchronization, and amortized repeated execution separately.
4. Verify synchronization through the actual PJRT completion contract. A synchronous execute can legitimately have an already-ready future; a no-op wait is not by itself proof of asynchronous mismeasurement.
5. Record compiler temporary/output/code bytes when exposed, logical retained bytes, and sampled allocator/RSS evidence separately. Unknown compiler fields remain unknown, not zeros and not estimates manufactured from array sizes.
6. Compare the same workload/input/precision against CPU on the same machine. Do not compare Metal FP32 against CPU FP64 as a speed claim; keep FP64 only as the independent accuracy reference.
7. Sweep the controlling capacity and retain slow as well as fast cases. A single favorable fixed GEMM is not a scientific acceleration claim.
8. Diagnose whether loop dispatch, host predicates, CPU preparation or numerical kernels dominate. Do not change algorithms, masks, tolerances or derivative order to improve a timing row.
9. Audit touched numerical code for new Python runtime-sized loops, hidden host synchronization, dense materialization, repeated preparation/factorization and full sorts. Document the legitimate bounded/static/host-preparation cases.

No arbitrary speedup factor is assumed in this plan. Publication must show at least one useful end-to-end workload with repeatable acceleration before advertising an acceleration benefit, and must report workloads that remain CPU-favored. Functional support and performance qualification remain different evidence dimensions; lack of a speedup does not justify falsifying either one.

## 8. Documentation and generated surfaces

| File | Required update after implementation smoke |
| --- | --- |
| `README.md` | Add the one-command Apple installation and concise supported profile; retain current CPU/CUDA instructions and precision defaults. |
| `docs/installation.md` | Darwin/architecture/OS/Python/JAX wheel matrix, explicit runtime selection, diagnostic examples, unsupported-platform behavior, conflicting-plugin remedies, and current versus historical qualification. |
| `docs/api/backends.md` | Document the Metal diagnostic API, actual provider/platform identity, capability versus qualification, and absence of CUDA native-provider parity. |
| `docs/guides_configuration.md`, `docs/api/configuration.md` | Explicit runtime-selection example with x64-on FP32 construction; no global precision reset, no new Metal-specific configuration vocabulary. |
| `docs/api/nn/layers.md`, `docs/api/nn/architectures.md` | New construction dtype API and its boundaries, with existing defaults and parameter/scan roles preserved. |
| **New `docs/guides_metal_execution.md`** | One authoritative support/qualification guide: required scenarios, actual runtime behavior, precision/index distinction, known refusals, host-control/CPU-preparation boundaries, memory interpretation, benchmark methodology and retained evidence. |
| `docs/guides_sparse_spatial_hierarchies.md` | Qualified prepared-neighbor scope versus unqualified hierarchy/voxel/dyadic consumers; full-width keys and unchanged x64 requirement. |
| `docs/guides_mace_execution.md` | Native-JAX Apple execution separate from CUDA Pallas acceleration; actual precision/derivative/replay profile. |
| `docs/api/linalg.md` | Qualified iterative route and preserved status/derivative semantics; do not claim GPU LAPACK/direct-factorization support. |
| `mkdocs.yml` | Link the authored Metal guide under the existing installation/execution organization. |
| `CHANGELOG.md` | Record the extra, diagnostics, explicit constructor dtype and intentional Apple execution/profile identity change. Preserve the concurrent JAX 0.12 migration text. |
| `docs/data/public_api.json` | Regenerate after public exports stabilize with `tools/generate_public_api_manifest.py`. |
| `docs/data/capabilities.json`, `docs/api/capabilities.md`, and other outputs of the existing capability generator | Regenerate once after capability/profile/source updates stabilize. Do not hand-edit IDs or omit downstream generated records whose catalog identity changed. |
| This plan | Record implementation outcomes, exact retained evidence and unresolved prerequisites without converting historical observations into new pass claims. |

Use the existing generators and consistency checker. `tools/generate_capability_inventory.py` also owns application portfolios, closure/source records and generated closure documentation; allow its complete deterministic output instead of updating only the visible capabilities table. No invented compatibility aliases or schema-generation suffixes.

## 9. Dependency-aware implementation and verification waves

| Wave | Work | Exit condition |
| --- | --- | --- |
| A: baseline and provider binding | Refresh completed JAX 0.12 tree/lock; establish exact native source closure; compare isolated candidates against current lowering, effects, integer and precision gates | One selected source/build path, bounded patch contract, real current-JAX execution; no public extra advertising unqualified binaries |
| B: native correctness and packaging | Complete dtype/effect/lifecycle corrections, own loader, reproducible wheel build, Mach-O/runtime-asset audit and negative resolver cases | A self-contained installable companion that preserves required contracts and refuses unsupported paths |
| C: independent integration slices | Runtime diagnostics/admission/device identity; explicit MLP dtype; consumer-specific dtype defects and benchmark interfaces | Coherent implementation slices, CPU regressions and direct smoke on each changed surface |
| D: installed-wheel scientific qualification | Fresh M1 environment, mandatory primitive/effect/science/lifecycle matrix, no skips or source-tree import leakage | Every required case passes for the exact built artifact; explicit refusal cases fail in the intended way |
| E: isolated performance campaign | Phase-separated CPU/Metal comparisons and bounded scaling on the same machine | Honest retained accuracy, status, synchronization and performance evidence; no blanket speed claim |
| F: final integration and release readiness | Authored docs/examples, typing/audits, generated manifests, final wheel/provenance consistency and only invalidated checks | Acceptance checklist complete; companion/core publication can be requested separately |

### Ownership and parallelism

One integration owner owns this map, public interfaces, `pyproject.toml`, `uv.lock`, config/device boundaries, generated artifacts and the verification record. One native owner owns `native/metal` and its source patch queue. One scientific owner owns `Linear`/`MLP`, their direct consumers and dtype regressions. Qualification/benchmark work can proceed independently once the selected plugin and dtype interface are fixed, with explicit file ownership.

Do not start expensive dependent work before current-JAX ABI/effect gates are resolved. CPU-only source work can proceed while a single native build uses a bounded build budget. Do not run multiple unrestricted `pytest -n auto` jobs, native builds and Metal campaigns together. Each native owner uses a separate build directory; immutable downloads may be shared, mutable compilation/measurement state may not.

### Verification commands and scope

Commands below define the implementation obligations. Actual command results and artifact identities are retained in the section 12 verification record.

- Apply the configured formatter and lint to changed files before identity-sensitive verification.
- Run the exact new/failing CPU nodes, then the conservative affected CPU matrix with `pytest -n auto` in one invocation. Keep genuine numerical/type/rank cases under `strict_jax`; optional-provider integration retains the provider's declared runtime policy.
- Run `python tools/check_typing.py check` for the repository-required zero-diagnostic gate, including the new companion Python files.
- Run `python tools/audit_selectors.py`, `python tools/audit_contract_candidates.py --signatures`, and the existing import-boundary checks where their surfaces changed. Retain necessary provider/decoder/scientific guards; do not delete them to satisfy a nominal check.
- Run the public annotation fixtures and `python -m tools.check_installed_typing --python 3.13` for the currently supported installed-wheel consumer contract.
- Build using `tools/build_metal_wheels.py`, resolve using `tools/check_metal_extras.py`, and qualify using `tools/qualify_metal_wheels.py` with explicit core wheel, companion wheel, workspace and output paths. The qualifier must run hardware tests serially (`-n 0`) in the isolated environment and verify exact mandatory node/scenario coverage.
- Exercise `examples/metal_scientific_execution.py` from the installed artifact and observe numerical output/state, not just device enumeration.
- Run benchmark commands only after correctness succeeds, without resource contention; use each updated driver's documented dtype/route/capacity controls and retain exact commands in evidence.
- After implementation and authored documentation stabilize, generate once with `python tools/generate_public_api_manifest.py` and `python tools/generate_capability_inventory.py`, then run `python tools/check_capability_consistency.py`; check authored documentation/example references and the generator's full affected output.
- Run CUDA resolver/import regression checks affected by the packaging/configuration delta. Real CUDA hardware verification is a separate supported-host obligation if those semantics change; do not claim it was exercised on the Mac.

Adding one optional marker does not justify a full numerical suite. A broader suite becomes necessary only if implementation actually changes global collection, fixtures, core dtype behavior, shared numerical semantics or other cross-cutting inputs. Never run the repository-wide suite under x64-off as a substitute for selecting FP32 contracts.

Maintain one verification record containing exact command/scenario, relevant tree and wheel hashes, package/native/OS/device identity, pass/failure/skip, exact failed nodes, duration and evidence paths. Reuse passing results while relevant inputs remain unchanged. A source or dependency merge invalidates only the affected evidence; a native/JAX/MLX/precision/compiler change invalidates the corresponding Metal runtime/scientific/performance results.

## 10. Acceptance checklist and stop conditions

Completion requires all of the following:

- [x] Current Phydrax/JAX 0.12 dependency and native lowering/ABI state is coherent; no downgraded core or misleading dependency range.
- [x] `phydrax[metal]` installs the one exact matching companion on the supported Darwin-arm64 environment; unsupported hosts cannot silently succeed as CPU-only Metal installs.
- [x] No user LLVM/MLIR/Bazel/CMake/Homebrew source build is required for installation; required native libraries and metallibs are present and audited.
- [x] Root imports and pure diagnostics are device-inert; explicit CPU behavior and scientific callable identities remain unchanged with the companion installed.
- [x] Explicit Metal selection cannot select a conflicting plugin or silently fall back; diagnostics retain the exact cause.
- [x] FP32/complex64 execution and exact int64/uint64 paths work together with x64-on; f64/c128 never silently narrow at any native boundary.
- [x] Current-JAX JIT/batching/JVP/VJP/second-order paths and successful/failing compiled guards pass on the actual M1 Max.
- [x] Every required scientific scenario in section 1.4 passes with dtype, placement, accuracy, ordering, failure and derivative evidence.
- [x] Model constructor defaults/key addressing/field order remain unchanged; new dtype callers and source-sensitive artifacts are handled through their owners.
- [x] No numerical CPU fallback, invented compiler/memory evidence, hidden CUDA parity, or relabeled interpreter result appears in support claims.
- [x] Lifetime, completion, donation/copy/no-copy contracts and teardown are exercised; callback registries do not retain dead array-bearing owners indefinitely.
- [x] Phase-separated, same-precision CPU/Metal measurements and bounded scaling are retained; acceleration claims name the actual workloads and nonclaims.
- [x] Required typing, focused/affected tests, applicable audits, installed-wheel example, docs and generated public/capability artifacts are complete.
- [x] Publication order and exact release artifacts are ready without having published anything implicitly.

Hard stop examples: inability to build the current-JAX PJRT/StableHLO chain; compiled errors suppressed or impossible to lower; exact keys corrupted; unsupported precision narrowed; masks or compensation invalidated by backend optimization; required scientific path only passes through hidden CPU math; incomplete installed-wheel evidence. Do not convert these into skips, weaken the scenario, or declare a dependency extra done. Finish independently reachable work and state the precise missing prerequisite if one remains.

## 11. Evidence and source references

Local implementation ownership was established from the files named above. LSP reference checks included `APPLE_METAL_PROFILE`, `device_vendor`, `sample_device_memory`, `capture_environment`, `_config_state._admit`, `Linear`, and `MLP`. Broad references to the latter constructors and benchmark environment function are the reason this plan preserves default construction and existing record shapes instead of rewriting consumers indiscriminately.

External source observations, not local execution proof:

- [JAX-MPS repository](https://github.com/tillahoffmann/jax-mps), [inspected stable source](https://github.com/tillahoffmann/jax-mps/tree/4eb634832e03ff99331e8a03bf3517b76b12312a), [plugin initialization](https://github.com/tillahoffmann/jax-mps/blob/4eb634832e03ff99331e8a03bf3517b76b12312a/src/jax_plugins/mps/__init__.py), [native build](https://github.com/tillahoffmann/jax-mps/blob/4eb634832e03ff99331e8a03bf3517b76b12312a/CMakeLists.txt), [linear algebra handlers](https://github.com/tillahoffmann/jax-mps/blob/4eb634832e03ff99331e8a03bf3517b76b12312a/src/pjrt_plugin/ops/linalg.cc).
- [metaljax repository](https://github.com/eterevsky/metaljax), [0.11.9 package metadata](https://pypi.org/pypi/metaljax/0.11.9/json), [dtype/transfer representation](https://github.com/eterevsky/metaljax/blob/main/plugin-native/metal/metal_dtypes.cc), [callback bridge](https://github.com/eterevsky/metaljax/blob/main/plugin-native/runtime/host_callback.h), [plugin lowerings](https://github.com/eterevsky/metaljax/blob/main/src/jax_plugins/metal/__init__.py). Main-branch URLs are observations; replace them with exact source pins for an implementation build.
- [Apple Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf): supported types, mathematical modes and numerical-compliance distinctions.
- [Apple JAX plugin documentation](https://developer.apple.com/metal/jax/): experimental scope and unsupported floating/complex types; not evidence for current Phydrax/JAX 0.12.
- [OpenFPM PR 18](https://github.com/mosaic-group/openfpm/pull/18), [pinned device backend scope](https://github.com/mosaic-group/openfpm_devices/blob/0a1c41eaa3ad6353db7436d7207a8eb12b71efd6/README.md), [explicit FP64/system-atomic refusal](https://github.com/mosaic-group/openfpm_devices/blob/0a1c41eaa3ad6353db7436d7207a8eb12b71efd6/src/spirv_molten/MoltenVKKernelABI.cpp). The independent SPH report is a PR comment, not Phydrax or M1 Max qualification.
- [IREE JAX integration](https://iree.dev/guides/ml-frameworks/jax/): a separate AOT/PJRT direction, not the runtime implementation selected by this plan.

At planning time, current-JAX compatibility, compiled effects, precision/index semantics, deployment, acceleration, and exact dependency pins remained unexecuted obligations. Section 12 records their implementation outcomes rather than treating upstream source observations as local proof.

## 12. Initial implementation outcome

The implementation remains on worktree/branch `metal-execution`, after merging
`origin/dev` at `13d0a3076`. The selected native source is JAX-MPS commit
`127ab63d0cab71320702e562703c4aaa0c6570d1`, with the owned patch queue and pinned
MLX/OpenXLA source closure. This is an owned companion distribution, not a new
array API, a CUDA translation layer, or an independent compiler built from scratch.

Exact local artifacts:

| Artifact | SHA256 |
| --- | --- |
| `phydrax-0.2.2-py3-none-any.whl` | `12c907531751e43911b4e68ccdb7ca386c3993b2469a2eca29a46657fcec74f7` |
| `phydrax_metal-0.2.2-py3-none-macosx_14_0_arm64.whl` | `ce2e9dc316a0ddf70ce2d0b79dd3530ef746f97dee89f87529ea41e90e7e35cf` |

Native source identity is
`b7e3a5b2322c42f7d704c554e7c230e904416f9c37aee71d134cb9d1a4657730`;
build identity is
`f35ed164383fadf2c410965cdbc6991cc3de47b037ceda82c26ea771a10b2d55`.
The frozen wheels and build evidence are under
`.tmp/metal-native/frozen/b7e3a5b2322c42f7/`.

`benchmarks/m1_max_qualification.json` is the retained verification record.
It contains actual build/resolver/installed-wheel results, exact commands,
raw phase measurements, complete comparison outcomes, failed attempts,
source snapshots for changed reporting drivers, complexity measurements,
and runtime-audit rationales. The fresh environment outside the checkout
passed all 151 mandatory cases without skips and the standalone example.
All 11 resolver scenarios passed. Installed typing on the exact core wheel
passed its three fixture files; the final repository typing gate also passed.
No package was published, and candidate capability records were not relabeled
as released qualification.

The user approved explicit float32 dynamic MACE coupling coefficients, with
distinct actual-value fingerprints and unchanged float64 defaults. No hidden
static float64 coefficient payload or historical-ID reuse was introduced.
The host preparation comparison covered 107 real coupling families and preserved
their default float64 inputs/bases bitwise; this is not a universal bitwise
claim for arbitrary host SVD shapes.

The measured performance outcome and its precision limits are documented in
[Apple GPU execution](../guides_metal_execution.md#local-m1-max-results).
The width-256 Poisson PINN is the one accuracy-qualified faster workload in this
campaign. Slow cases and the original/additional wide-PINN comparison failures
remain in the record. Metal is not recommended for the measured MACE, sparse,
iterative, spectral, or neighbor workloads.

### Complexity and runtime audit

Measurements compare the merged core baseline and pinned native upstream against
the changed symbols, using Lizard and Python AST cognitive complexity. C++
cognitive complexity is unavailable, not reported as zero. The retained record
contains all 764 changed-symbol measurements.

The severe tail is retained deliberately rather than refactored incidentally:

- `AtomisticBatch.__init__` preserves ordered per-structure validation/batching;
  its added branches carry the explicitly selected accumulation precision.
- `evaluate_dem_barrier` and `DifferentialSolution.__init__` retain their existing
  control-flow complexity. Their changes defer default array creation or select
  the owned clock dtype without reordering formulas, fields, or validation.
- `_solve_prepared` retains its central scientific dispatch and derivative
  ownership; new branches distinguish weak tolerances from explicitly typed
  arrays. No solver loop or status ordering was rewritten.
- `run_pinn_model_benchmark` is an ordered measurement-lifecycle coordinator,
  keeping initialization, stable compiled closures, first execution, warm
  updates, and reporting in one lifetime. Its reporting loops are not device
  numerical iterations.
- The wheel qualifier's `main` owns one environment/admission/execution/evidence
  transaction; reusable process, hash, environment, and coverage operations are
  separate helpers.
- Native `HandleWhile`, `HandleCase`, and `HandleCustomCall` retain upstream
  evaluator structure while correcting token/effect ownership. Unrelated custom
  call branches are not rewritten.
- `HandleDotGeneral` retains explicit dtype/precision dispatch. The bounded
  scalar-constant interpreter retains visible fail-closed opcode admission
  rather than hiding its accepted operations in a registry.

New host SVD/rank and O(3) loops are immutable preparation. Constant evaluation
is bounded compiler preparation. Model/topology transfer, control predicates,
explicit callbacks, independent references, and reporting readback remain
visible boundaries. No selected sparse or iterative route gained a hidden dense
runtime copy, repeated runtime factorization, or Python loop over a dynamic array
axis. FFT tables are shape-derived preparation; caller FFT data executes on GPU.

## 13. Native optimization outcome

The subsequent `//code--` stayed in `metal-execution`. Changes are owned by
`native/metal/patches/12-bounded-numerical-execution.patch`,
`13-mlx-output-argument-separators.patch`, the companion's platform-specific
lowering registration, and the exact installed-wheel qualification inventory.
No new Phydrax array API, precision demotion, CPU numerical fallback, global JAX
monkeypatch, or source-generation compatibility alias was introduced.

The optimized native source is
`f346b2ade03dd5b73fe0542791411f2d2c154489cbcfc0b5da90795086d7f4b8`.
Its companion wheel SHA256 is
`cf55c97b949a8e2a178f533e9530260ab8473905c3b7474e9b8ce03abf60e377`.
The core wheel is byte-identical to the initial qualified artifact. Frozen
artifacts and build/qualification records are under
`.tmp/metal-native/frozen/f346b2ade03dd5b7/`; permanent comparison evidence is
`benchmarks/m1_max_optimization.json`.

All 293 mandatory installed-wheel cases and the standalone example passed.
The bounded runtime smoke additionally exercised the long mixed-magnitude sum,
ordered indexed map, full-width scatter cotangents, and token-bearing callbacks.
The campaign retains 12 primitive and 81 scientific comparisons, including
regressions and failed wide-PINN agreement. Measured outcomes and limits are in
[Apple GPU execution](../guides_metal_execution.md#local-m1-max-results).

The cutover distinguishes implementation-defined ordinary reduction trees from
explicit scientific event order. Parallel ordinary sums are admitted with bounded
scratch; deterministic/compensated loops retain their sequence. Singleton
scatter windows avoid sorting. Pure static-trip regions have explicit operation,
state, temporary, work, and argument bounds. Weighted `uvu` fusion retains staged
arithmetic, dynamic coefficients, later derivative consumers, and a 64 MiB total
output bound.

Qualification exposed and fixed a canonical MLX multi-output signature defect
and JAX 0.12 fill-gather index narrowing. The latter is corrected only through
Metal's owning lowering registration. Ordered callback case outputs now retain
their token dependency markers. Failed development artifacts are retained as
failures, not relabeled as qualification.

The severe compiler-complexity tail has explicit ownership rationale in the
evidence. `RegionEmitter::emit` is a fail-closed opcode dispatcher for one bounded
pure dialect; `TryWeightedUvu` owns one closed graph pattern, its SSA-use
admission, and literal staged shader. The existing `HandleWhile`, `HandleCase`,
and `HandleDotGeneral` remain the authoritative general dispatchers. Their
numeric loops and effect contracts were not fragmented or weakened to satisfy
metrics. All source-defined limits and unsupported-pattern routes remain visible.
