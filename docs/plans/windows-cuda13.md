# Native Windows CUDA 13 implementation plan

Status: implementation is in the `windows-cuda13` worktree. Authenticated source
preparation, CPU/reference execution, Windows file/process primitives, and
installed public typing have been exercised. Native compilation and GPU
qualification remain blocked on verified compiler/SDK inputs; the Windows CUDA
installation is not declared complete or hardware-qualified.

## 1. Goal and completion contract

Keep one public installation interface:

```text
pip install "phydrax[cuda13]"
```

On supported native Windows, this must install a complete, matched, wheel-based
stack that executes Phydrax scientific workloads on the NVIDIA GPU without WSL,
a user-installed CUDA toolkit, a compiler at installation time, manual wheel
URLs, or manual DLL-path configuration. A compatible NVIDIA driver and the
standard Windows runtime prerequisites of the selected CPython/JAX wheels remain
external prerequisites; the installer must not modify them.

The initial Windows qualification target is standard-GIL CPython 3.13, Windows
x86-64, one process, and one RTX 5090 (compute capability 12.0). Windows ARM,
free-threaded Python, additional GPU models, and additional device/process
layouts are not qualified by that result. Artifact architecture coverage and
observed hardware qualification remain separate facts.

Preserve the current Linux x86-64 and aarch64 installation contract and the
mandatory cuDSS, AmgX, and QUDA companion contract. A Windows JAX-only install is
not completion. A failure to port or qualify a mandatory component blocks the
Windows release; it does not authorize removing that component.

### Proposed capability boundary

| Surface | Required Windows behavior |
| --- | --- |
| Ordinary JAX-based scientific execution | Actual GPU execution, including supported float32/float64, complex64/complex128, differentiation, FFT, dense/sparse numerical routes, and wide indexing. Preserve native solver status and evidence. |
| Internal compensated accumulation | Use the existing canonical ordered JAX recurrence when Mosaic is absent; this still executes on the GPU. Preserve seeded components, cancellation behavior, derivatives, and nonfinite semantics. |
| Internal MPM APIC moments | Use the existing canonical JAX gather/product/reduction lowering when Mosaic is absent. Preserve masks, indexing, NaN propagation, derivatives, and declared error bounds; do not promise bitwise agreement between different reduction schedules. |
| cuDSS, AmgX, QUDA | Mandatory Windows companion components, with unchanged scientific APIs, precision contracts, FFI targets, failure evidence, and lifecycle obligations. |
| Optional cuTENSOR | Port the wrapper and support the existing `cutensor` extra using the exact Windows vendor wheel. Qualify separately when selected. |
| Explicit Mosaic MACE / Pallas spatial and FMM acceleration | Refuse unsupported compiled requests before staging when the installed compiler lacks the requested lowering. Ordinary native-JAX routes remain distinct available choices, not automatic substitutes. Never select `cpu_interpret` implicitly. |
| cuTensorNet and cuEST | No inspected Windows vendor artifacts exist at the pinned versions. Explicit installation requests must not become successful no-op extras; require the real distribution and fail resolution when no compatible wheel exists. Provider calls must also refuse clearly. |
| NCCL, site MPI, cuDSS MG/MGMN, cuDecomp, multi-process/multi-host execution | Not supported by this Windows profile. Refuse requests at the owning admission boundary. A one-device campaign is not peer or collective qualification. |

This is an explicit proposed Windows support contract, not a claim of complete
Linux feature parity. The unsupported optional compiler/provider routes above
must be visible in installation documentation and capability declarations. They
must not be used to weaken the mandatory companion requirement.

### Non-negotiable invariants

- No silent CPU fallback after installed CUDA intent, except deliberate CPU
  selection through the existing configuration interface.
- CPU-only base installs remain valid. Explicit CPU workers with an intact CUDA
  installation do not initialize a CUDA client or Phydrax native components.
- Do not disable or weaken the JAX indexing source guard or collective dispatch
  guard. Do not falsify source, build, runtime, device, or qualification identity.
- Keep CUDA's scientific platform identity `cuda`; do not invent a Windows-only
  numerical backend vocabulary. Add host/compiler provenance where execution
  identity requires it.
- No numerical formula changes, precision narrowing, implicit TF32, reduction
  reordering beyond the already declared route semantics, hidden host numerical
  substitution, or changes to RNG addressing.
- No new public installation alias, schema generations, or compatibility shims.
- Missing or skipped mandatory evidence, incorrect finite results, timeouts,
  unknown process status, and failed teardown cannot qualify a release.

## 2. Gate 0: user-directory boundary and authorization

The original restriction prohibited all local reads and writes outside
`C:/Users/Agent`. The user subsequently explicitly authorized read-only access
to required OS, installed-toolchain/SDK, and NVIDIA driver dependencies solely
for implementation tools and verification.

All writes, source checkouts, downloads, caches, build products, and evidence
remain under `C:/Users/Agent`. The exception does not authorize unrelated files,
other users' data, registry/system configuration changes, or driver changes.
Compiler inputs outside the profile may be read only when covered by that
explicit toolchain permission; private build staging remains profile-local.
Do not infer further permission changes from a later implementation instruction.

When implementation is authorized:

1. Follow the repository's worktree rule. A normal `//code` implementation gets
   a fresh worktree under
   `C:/Users/Agent/phydra-labs/.worktrees/windows-cuda13`; `//code--` is the
   explicit exception. Copy the authoritative `AGENTS.md` into the new worktree
   before implementation. Do not modify the parent checkout.
2. Keep source checkouts, downloaded archives, tool staging, virtual
   environments, Bazel output/user roots, repository caches, compiler temporary
   directories, logs, wheelhouses, test scratch, JAX caches, and QUDA tune caches
   under the authorized profile. Use short profile-local build paths where
   Windows path length matters.
3. Pass explicit interpreter/compiler/SDK/tool paths. Do not use upstream
   configure scripts' machine-wide discovery unchanged. Set profile-local
   `TEMP`, `TMP`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, and build-cache roots;
   disable unrelated tool configuration discovery with the owning tool's
   supported option, such as `uv --no-config`.
4. Reject paths or reparse points that escape the authorized work root before
   following them. A path validator is not an OS sandbox; do not claim that it
   constrains arbitrary third-party child-process I/O.
5. Never modify the registry, TDR/watchdog settings, drivers, system CUDA,
   machine-wide PATH, services, other users' processes, or other users' files.
   Do not reset the GPU. An unavailable dependency or WDDM resource failure is
   evidence to report, not a reason to change the shared host.
6. Limit all process cleanup to descendants owned by the current operation.
   Resource-intensive builds, tests, and benchmarks must not compete with each
   other; do not interfere with unrelated work on the shared GPU.

## 3. Evidence and architectural decisions

### 3.1 Current barriers

- `pyproject.toml` requires Python `>=3.13,<3.14` and JAX `>=0.12.0,<0.13`.
  The CUDA extra pins Linux JAX/JAXlib to `0.12.0`, installs the major-13 marker
  on all platforms, and requires the numerical companion only on Linux.
- `tools/check_cuda_extras.py` currently rejects NVIDIA/plugin/companion
  packages in a Windows resolution. The corresponding tests explicitly pin
  marker-only Windows behavior.
- `_cuda_environment.py` recognizes upstream `jax-cudaN-plugin` distributions,
  uses a set of major numbers, loads `libcuda.so.1`, and inspects ELF and Linux
  library paths. Missing plugins are warnings on Windows.
- `_cuda_native.py` and the companion locator assume `.so` libraries, ELF
  dependencies, and Linux CUDA-runtime names.
- `backends/_mosaic.py` admits Mosaic based on target compute capability, not
  the actual compiler implementation. Compensated accumulation and MPM moments
  therefore need attention even for ordinary scientific workflows.
- Windows qualification is also blocked by POSIX process cleanup, hard-coded
  `venv/bin/python`, symlinked harness staging, Linux host inventory commands,
  and the unconditional `resource` import in `benchmarks/_memory.py`.

### 3.2 Matched compiler stack

Use stock `jax==0.12.0` and `jaxlib==0.12.0` on Windows. Published winjax
`0.11.2` pins an incompatible JAX frontend and is not an installable dependency
for this checkout. JAX 0.12's base metadata permitting JAXlib 0.11.2 is not CUDA
qualification and is not a reason to introduce a mixed stack.

The owned Metal source lock already records the relevant upstream pins:

- JAX commit `8dcb7aeced770bd22531b42d0b925c7342787c27`;
- XLA commit `d867969800d4a8fd045505eabc65e1e66f30794f`;
- the pinned JAX `.bazelversion` is **9.2.0**, not winjax 0.11.2's Bazel 8.7.0.

Rebase the necessary winjax Windows portability patches onto that XLA/JAX build
workspace. Preserve JAX's own pinned dependency patches. Drop patches already
upstream; do not blindly apply the entire old series. Installed Python JAX
sources remain the stock wheel sources.

### 3.3 Distribution layout

Use two lockstep Phydrax binary distributions on Windows:

| Distribution | Ownership |
| --- | --- |
| Existing `phydrax-cuda13` | Numerical companion DLLs: mandatory cuDSS/AmgX/QUDA and the optional cuTENSOR wrapper. Same scientific component contract as Linux. |
| New `phydrax-cuda13-pjrt` | Windows loader, XLA CUDA PJRT DLL, CPython 3.13 CUDA kernel extensions, runtime manifest, and licenses. Owns the Windows JAX integration boundary. |

A single CPython-tagged Windows runtime wheel can contain both the PJRT DLL and
kernel extensions because Phydrax supports one Python minor. Measure its actual
size; do not assume it fits the package index quota. Exceeding the quota is a
publication prerequisite, not permission to omit assets.

Keep the upstream-required import package `jax_cuda13_plugin` for the compiled
kernel extensions. The distribution name differs deliberately; this is an
external JAX discovery contract, not a compatibility alias. Use one JAX plugin
entry point targeting `phydrax_cuda13_pjrt.plugin`; do not also introduce a
second independent namespace loader.

The existing major-13 marker remains unchanged and is not republished. Versions
of the two owned binaries must equal the actual core release version. Do not
invent a future version or overwrite an already published wheel.

### 3.4 Bootstrap and DLL ownership

JAX 0.12 catches errors from plugin `initialize()`. Its backend loop can also
skip `cuda` through `hardware_utils.has_visible_nvidia_gpu()` even when CUDA was
explicitly selected. Therefore, merely raising an exception in the loader does
not establish a no-fallback contract.

The Windows loader must:

1. Be importable without importing `phydrax`, initializing devices, or loading
   the PJRT DLL. Avoid the JAX-to-Phydrax initialization cycle.
2. Validate the exact supported JAX integration boundary and establish a
   fail-loud CUDA backend registration before failure-prone native bootstrap.
   Retain the original bootstrap error in a refusal factory if bootstrap fails.
   Successful `register_plugin` replaces that guard with the real factory.
   This preserves JAX's plugin callback ordering; do not defer all C-API
   registration until after discovery callbacks have already run.
3. Install the narrow, source-verified Windows hardware-detection adaptation
   needed by this JAX release. Actual driver/device failure must raise through
   the fail-loud backend path rather than return a value that makes JAX skip
   CUDA. Do not fabricate observed device facts. Explicit CPU selection avoids
   this CUDA initialization path.
4. Prepare CUDA kernel-extension DLL dependencies at the early
   `jax_cuda13_plugin` import boundary, before JAX imports its `.pyd` modules.
   This is separate from PJRT client initialization.
5. Resolve CUDA DLLs from their owning installed wheel RECORDs. Preload the
   required closure by absolute path and retain DLL-directory/library handles
   for the required process lifetime. Do not depend on unspecified ordering of
   multiple added DLL directories, cwd search, or a global toolkit installation.
6. Detect already resident incompatible DLL families before accepting them.
   Do not silently adopt a PyTorch-owned incompatible cuDNN family. Report
   import-order conflicts and recommend separate processes rather than
   replacing DLLs already loaded into a process.
7. Load `GetPjrtApi`, register the C-API capsule, CUDA custom calls/types, and the
   plugin profiler using the matched JAX release. Retain real error/status
   paths. Do not add an unqualified global JAX numerical callable replacement.
8. Supply `ptxas`/libdevice discovery from the owned wheel layout. Any required
   PATH changes are process-local and limited to compiler-tool directories;
   never add arbitrary cuDNN/toolkit directories to machine-wide PATH.
9. Use on-demand allocation as the Windows WDDM default through the owning
   plugin options when the user has not supplied an allocator choice. Preserve
   explicit user settings and record the effective choice. Do not alter x64,
   matmul precision, or watchdog policy.

Core startup performs metadata-only admission of installed Windows CUDA intent
before numerical work, including missing/corrupt/competing runtime packages.
Keep a metadata-only collection phase separate from explicit driver/device
observation. A successful static report is not hardware qualification.

## 4. File-by-file implementation map

Paths marked **new** are proposed files. All other paths were inspected or
located in the current repository. Changes below are dependency-ordered;
individual files have one active owner during implementation.

### A. Source-pinned Windows PJRT runtime

| File | Planned change and acceptance |
| --- | --- |
| **New** `native/cuda13_pjrt/pyproject.toml` | Declare the Windows x64, standard CPython 3.13 runtime distribution, exact JAX/JAXlib pair, matching core release, explicit NVIDIA Windows dependencies, and one JAX plugin entry point. No NCCL/NVSHMEM dependency for this single-device build. CUDA toolkit/compiler dependencies are build-owned, not user installation requirements. |
| **New** `native/cuda13_pjrt/sources.lock.json` | Pin upstream JAX/XLA and Windows-port source revisions, external dependencies, Windows redistributables, toolchain inputs, archive hashes, patch order/hashes, and license sources. Reuse the deterministic source-lock and source/build identity conventions from `native/metal`; no schema generation field. Record actual full hashes when acquiring artifacts, never guessed values. |
| **New** `native/cuda13_pjrt/patches/` | Rebased Windows XLA/compiler/kernel-wheel patches only. Preserve the necessary MSVC-ABI evaluation-order, FFI attribute, allocator-progress, and profiler behavior where not already fixed upstream. Distinguish XLA's internal Triton code generation from unsupported JAX Pallas/Triton APIs. Missing Pallas does not justify disabling ordinary XLA fusion indiscriminately. |
| **New** `native/cuda13_pjrt/toolchains/` | Explicit-path Bazel toolchain definitions and templates, derived from the Windows port but adapted to Bazel 9.2.0 and the current module graph. Generated paths stay in the build directory. No unbounded host discovery or patching of unrelated global caches. |
| **New** `native/cuda13_pjrt/_build_backend.py` | Package only a verified staging tree, following the owned Metal packaging boundary. Refuse source-distribution/editable/source-build fallback and incomplete or tampered staging. Write deterministic wheel members, RECORDs, metadata, and license inventory; emit a `cp313-cp313-win_amd64` wheel. |
| **New** `native/cuda13_pjrt/python/phydrax_cuda13_pjrt/__init__.py` | Lightweight runtime identity/asset boundary, with no import-time device initialization. Do not import core Phydrax. |
| **New** `native/cuda13_pjrt/python/phydrax_cuda13_pjrt/plugin.py` | Own the bootstrap sequence in section 3.4, exact private-hook source validation, registration state, C-API/profiler/FFI registration, and refusal evidence. Never log an error and return as if CUDA had not been requested. |
| **New** `native/cuda13_pjrt/python/phydrax_cuda13_pjrt/runtime.py` | Own process-local Windows DLL inventory, dependency loading, resident-family checks, handle lifetime, driver observation, and owned compiler-tool paths. It must not import JAX or core Phydrax merely to inspect files. Core CUDA loader/diagnostics reuse this owner rather than implementing a second DLL-loading policy. |
| **New** `native/cuda13_pjrt/python/jax_cuda13_plugin/__init__.py` | Early dependency bootstrap for the upstream-named CUDA kernel package. Call the runtime owner before `.pyd` imports. Package the actual compiled kernels, not replacements or forwarding substitutes. |
| **New** `native/cuda13_pjrt/README.md` | Build inputs, bootstrap phases, supported target, exact nonclaims, CRT/driver prerequisites, publication, and the distinction between stock frontend and patched native compiler. |
| **New** `tools/build_cuda_pjrt_wheels.py` | Build the locked PJRT DLL and CUDA kernel wheel from the matched workspace, stage the runtime package, inspect PE assets and required exports, verify licenses/source identity, and invoke the packaging backend. Use the existing bounded command-evidence runner. No end-user compiler invocation. |

The runtime manifest records the exact JAX/JAXlib pair, upstream source and patch
identities, host architecture, compiler/toolchain flags, CUDA component versions,
asset hashes, real compiler capabilities, and build identity. Explicitly record
that Mosaic/Pallas and distributed support are absent when they are not built.
A declaration is not a substitute for successful registration and numerical
qualification of capabilities declared present.

### B. Binary inspection and CUDA environment admission

| File | Planned change and acceptance |
| --- | --- |
| **New** `phydrax/_pe_dependencies.py` | A bounded, host-only PE32+ x86-64 metadata reader: machine, ordinary and delayed imports, named/ordinal exports and forwarders, checked RVA/file ranges, malformed/truncated input refusals. It performs no native loading. Keep this module standalone and import-safe so the Windows runtime wheel can carry a byte-identical, manifest-hashed copy, following the existing Metal shared-helper pattern. There is one authored parser, not two implementations. |
| **New** `phydrax/_native_dependencies.py` | Canonical owner of the format-neutral dependency evidence records currently defined in `_elf_dependencies.py`. Move the machine selector here as `NativeMachine`, and give `DependencyClosure` an explicit object-format field so PE evidence is not mislabeled as ELF evidence. Own explicit ELF/Windows inspection dispatch and conversion of standalone runtime evidence; use lazy format-owner imports to avoid cycles. Do not pretend a Windows controlled preload model is a full emulation of the OS loader. |
| `phydrax/_elf_dependencies.py` | Retain ELF parsing and glibc resolution semantics unchanged. Move shared evidence records and the machine selector to their canonical owner, migrate every import/annotation, and remove the old `ElfMachine` spelling without an alias. ELF results explicitly identify their format. ELF-specific operations remain ELF-specific rather than acquiring Windows-shaped fake results. |
| `phydrax/_cuda_environment.py` | Recognize individual plugin installations rather than deduplicating major numbers. Introduce immutable `CudaPluginInstallation` records with distribution, version, and CUDA major, and replace `CudaEnvironmentFacts.plugin_majors` with `plugins` in a clean cutover. Detect two CUDA-13 implementations as duplicates, not one plugin. Verify the official Linux plugin/PJRT pair and the owned Windows runtime by their actual contracts. Add metadata-only installation collection for startup admission; keep explicit observation separate. Add Windows DLL attribution, driver/runtime facts, unavailable-evidence reasons, and resident/shadowing diagnostics without invoking ELF discovery on Windows. A missing Windows plugin under installed CUDA intent is an error. Preserve pure `assess_cuda_environment` and explicit CPU semantics. |
| `phydrax/_config_state.py` | Add CUDA installation admission to the existing startup/configuration owner, including implicit Windows CUDA intent when platform selection is unset. Do not initialize a GPU during metadata admission. Preserve existing startup ordering, frontend patch installation, explicit CPU behavior, and refusal of changes after backend initialization. Do not globally change Linux selection policy. |
| `phydrax/_cuda_native.py` | Dispatch companion filenames/dependency inspection by host format. On Windows reuse the runtime's owned DLL session and absolute paths; retain exact release, component, major, build hash, handler count/name/pointer, and site-identity checks. Use the admitted CUDA-runtime handle for synchronization and retirement instead of searching for `libcudart.so`. Preserve owning-device restoration and failed-teardown evidence. Refuse unsupported Windows components before loading their binaries. |
| `phydrax/backends/_availability.py` | Recognize native Windows extensions and component dependencies through the canonical inspection boundary. Preserve probe-versus-load separation and actionable `environment_issues`; never skip dependency admission merely because the plugin distribution has a different name. |
| `phydrax/backends/__init__.py` | Export the new public installation record explicitly and migrate the typing branch/lazy export table. No deprecated `plugin_majors` alias. |
| `tests/_support/cuda_execution_worker.py` | Consume the migrated native dependency owner and obtain the CUDA-runtime/driver entry points from admitted runtime ownership. Keep Linux UUID/peer behavior unchanged; do not invent a second Windows DLL locator in the worker. |

`MappedCudaLibrary.soname` can remain the existing loader-name field, documented
as an ELF SONAME or a Windows DLL import basename. CUDA-major attribution must
come from unambiguous format metadata and distribution ownership, not from
cuSPARSE/cuSOLVER/cuDSS version digits or equal-looking filenames.

Windows inspection must distinguish owned wheel libraries, already resident
libraries, driver/OS dependencies, unresolved imports, and uninspected state.
An empty module inventory is not proof of a clean process. Do not scan unrelated
user directories or recursively inspect arbitrary system installations.

### C. Compiler capabilities, scientific routes, and execution identity

| File | Planned change and acceptance |
| --- | --- |
| **New** `phydrax/_cuda_capabilities.py` | One private, immutable compilation-support boundary for the actual lowering client. Resolve runtime provenance, supported Mosaic/Pallas entry points, sparse CUDA kernels, and topology limits from the admitted implementation. Metadata-only preparation and runtime observation remain distinct. Do not cache numerical arrays or keep clients alive indirectly through weak-cache values. |
| `phydrax/backends/_mosaic.py` | Require positive Mosaic support in addition to the existing compute-capability test. Otherwise lower the already registered canonical JAX reference on the destination backend. Do not execute a probe kernel or synchronize the host inside a numerical iteration. |
| `phydrax/_numerics/_compensated.py`, `phydrax/_numerics/_compensated_native.py` | Preserve recurrence, shape guards, primal/transpose/JVP/batching rules, and primitive identity. Only adjust admission integration or observable lowering provenance where needed; no numerical rewrite. |
| `phydrax/discretization/mpm/_moments_native.py` | Preserve canonical gather/mask/NaN semantics, live-output rules, derivative rules, and bounded route schedule. Integrate truthful lowering provenance without copying the reference into a new Windows implementation. |
| `phydrax/backends/atomistic.py` | Require actual Mosaic support for `target="cuda"`. Include the admitted runtime/compiler identity in `AtomisticKernelTarget.target_id`; do not infer equivalence from GPU name, capability, or JAX version alone. Unsupported compiled requests refuse; the explicit CPU interpreter contract is unchanged. |
| `phydrax/backends/spatial.py` | Require the actual Pallas lowering capability for the selected backend. Keep the TPU route separate and intact. Unsupported Pallas spatial/FMM requests refuse before compilation. |
| `phydrax/linalg/_sparse_providers.py`, `phydrax/linalg/_plans.py` | Admit/auto-select the `jax-cuda` sparse route only when the selected runtime provides its required kernels. Preserve explicit provider requests, status, resource limits, and native-JAX iterative routes. Do not silently substitute a host solver. |
| `phydrax/backends/cudss.py`, `phydrax/backends/cudecomp.py`, `phydrax/backends/distributed.py` | Apply the live runtime's topology/component restrictions at existing availability/admission boundaries. Preserve the broad NVIDIA vendor envelope as an envelope, not evidence that a Windows runtime supports every listed scope. Explicit cuDSS MG/MGMN, site MPI, and cuDecomp requests refuse on this profile. |
| `phydrax/_execution_bootstrap.py`, `phydrax/_config_state.py` | Refuse unsupported Windows distributed bootstrap before starting a coordinator or partially initializing JAX. Reuse the existing bootstrap owner and errors. |
| `phydrax/backends/_cuda_profiles.py` | Add host/runtime-qualified support tuples and Windows single-device candidates without promoting them to qualified evidence. Preserve existing Linux support and distinguish unavailable components from unqualified hardware. |
| `benchmarks/_runtime.py` | Extend current environment identity with Windows PJRT/kernel/build identity, compiler capabilities, effective allocator choice, and relevant Windows CUDA overrides. Reuse existing phase timing and fingerprint infrastructure. |

Intentionally unchanged numerical implementations include MACE kernel bodies,
FMM fragments, the frontend indexing correction, and ordered collective
source guards. Audit dense linalg's advertised device routes against actual
Windows kernel availability, including upstream MAGMA/host-offload behavior;
do not label a workload GPU-only because its input array is on a CUDA device.
If a requested GPU-only algorithm has no implementation, refuse it in its
owning linalg policy rather than globally forcing an unrelated JAX option.

### D. Native companion Windows build and portability

| File | Planned change and acceptance |
| --- | --- |
| **New** `.gitattributes` | Add narrowly scoped canonical line-ending rules for the affected native build inputs and new runtime source. Preserve patch bytes with explicit `-text` rules where necessary, including the existing CRLF-sensitive AmgX patch. Do not renormalize unrelated repository files. |
| `native/cuda/CMakeLists.txt` | Choose explicit platform-supported component defaults: Linux retains its current set; Windows builds cuDSS/AmgX/QUDA/cuTENSOR. Explicit unsupported component requests are errors, never filtered out. Record host/compiler/runtime linkage and redistribution identity in the build manifest. Preserve canonical source ordering and release architecture semantics. |
| `native/cuda/cmake/PhxPipCudaToolkit.cmake` | Keep the Linux owner intact and dispatch to Windows preparation. Discover compiler tools, headers, DLLs, and import libraries from verified wheel RECORDs and pinned Windows redistributables. No modification of site-packages or machine toolkit selection. |
| **New** `native/cuda/cmake/PhxWindowsCudaToolkit.cmake` | Own the Windows toolkit-view invariant: explicit nvcc/cl/LLVM/SDK paths; a profile-local view without privileged symlinks; real DLL/import-library pairing; pinned driver-link libraries and cuobjdump. Prefer existing import libraries; generate a real import library from validated exports only when the vendor wheel omits it. Never use fake functionality/stub exports to make linking pass. |
| `native/cuda/cmake/PhxCudaComponent.cmake` | Restrict GNU linker flags/RPATH/version scripts to Linux. Add Windows DLL runtime installation, proper imported DLL/import-library properties, consistent runtime linkage, and host-compiler flags. Keep shared cudart. |
| `native/cuda/include/phydrax_cuda.h` | Add Windows `__declspec(dllexport)` for the existing C ABI and handler declarations. Preserve names, integer widths, ownership, and calling conventions. Windows x64 does not require replacing every `ctypes.CDLL` with `WinDLL`. |
| `native/cuda/cmake/PhxAmgX.cmake` | Preserve the exact pinned solver configuration and numerical patches. Supply upstream Windows defines omitted by the custom recipe, including `WIN32`, `NOMINMAX`, and static-library API handling. Remove `m`/`pthread` and GCC-only options only on Windows. Use one explicit patch executable and consistent CRT/compiler flags. |
| **New** `native/cuda/patches/amgx-windows-static-api.patch` | Prevent upstream Windows static AmgX objects from exporting an independent `AMGX_*` DLL ABI or using inappropriate `dllimport` decoration. Keep Phydrax's public C ABI authoritative. |
| `native/cuda/cmake/PhxQuda.cmake` | Use the explicit patch tool, add the pinned Windows portability patch, select compatible host/device compiler flags, and preserve the existing QUDA features, architecture set, static internal linkage, and shared-cudart requirement. |
| **New** `native/cuda/patches/quda-windows-portability.patch` | Port the pinned upstream's timers, hostname calls, filesystem/directory checks, exclusive tune-cache locking, backtrace/error diagnostics, and GNU-specific function-name/compiler assumptions. Preserve units, lock atomicity, failure behavior, and numerical code. Do not replace failed operations with success. |
| `native/cuda/src/quda/quda.cu` -> `native/cuda/src/quda/quda.cpp` | Keep host FFI, QUDA API adaptation, cache preparation, serialized stream ownership, handlers, and shutdown in the host translation unit. Replace POSIX cache/home/environment handling with an explicit profile-local Windows path and correct UTF-8/Windows path conversion. Do not use lossy path conversion for non-ASCII profiles. |
| **New** `native/cuda/src/quda/quda_kernels.cu`, **new** `native/cuda/src/quda/quda_kernels.hpp` | Move the actual `any_unowned` and `decode_momentum` kernels and their launch boundary out of the FFI translation unit. The current wrapper is not kernel-free. Preserve launch dimensions, legacy default stream, ordering, blocking ownership readback, and error checks. Keep XLA FFI headers out of nvcc's MSVC host pass. |
| `native/cuda/src/quda/CMakeLists.txt` | Compile the host wrapper with the chosen C++ compiler and the two device kernels with nvcc; link both into the same component on every platform. No alternate Windows numerical implementation. |
| `native/cuda/src/cudss/CMakeLists.txt`, `native/cuda/src/cutensor/CMakeLists.txt` | Use correct Windows imported DLL/import-library targets. Keep scientific implementations in `solver.cpp` and `cutensor_ffi.cpp` unchanged unless compilation exposes a specific portability defect. |
| `native/cuda13/pyproject.toml` | Keep the existing distribution. Add the Windows runtime dependency under a Windows marker, platform-appropriate build dependencies, and a Linux marker for unavailable cuTensorNet build inputs. Verify every mandatory Windows dependency has a compatible wheel. |
| `native/cuda13/_build_backend.py` | Intentionally retain the existing scikit-build delegation and sdist refusal. Platform build admission belongs to CMake/build tooling; do not introduce a second packaging backend merely for Windows. |
| `native/cuda13/python/phydrax_cuda13/__init__.py` | Locate platform-correct DLL or ELF component files from the installed component manifest. Explicitly refuse components not shipped for that platform; distinguish unavailable components from missing/corrupt required assets. |
| `tools/build_cuda_wheels.py` | Add the Windows build/audit path without routing it through auditwheel, patchelf, or readelf. Reuse canonical PE inspection and existing device-code inspection. Audit actual component set, exact exports, shared cudart, import closure, architecture code, and build identity. Retain Linux audits and container behavior. |

#### Toolchain and QUDA details

Use a matched MSVC ABI. The proposed companion build uses clang-cl for host FFI
translation units and nvcc with supported MSVC `cl.exe` for CUDA translation
units. Do not assume Windows nvcc supports clang-cl as its host compiler.
The exact qualified compiler/SDK versions and hashes are source-lock inputs.
The XLA PJRT build follows its separately pinned clang Windows recipe.

The QUDA split preserves this existing order: acquire the component mutex,
synchronize the incoming XLA stream, initialize/use QUDA, launch the two helper
kernels at their original points on the legacy default stream, retain the
blocking ownership readback/error checks, and finish with the existing device
synchronization. Moving launches to a different stream would be a numerical and
lifecycle change, not a portability fix.

Confirmed upstream QUDA portability work includes `lib/tune.cpp`,
`include/timer.h`, `lib/util_quda.cpp`, `lib/interface_quda.cpp`,
`lib/comm_common.cpp`, `lib/targets/cuda/malloc.cpp`, and
`include/util_quda.h`. A real compile may expose additional platform assumptions;
those belong to this mandatory port, not a reduced provider subset.

Prefer consistent dynamic CRT linkage for the companion and its internal static
libraries, with an explicit audit of the selected runtime dependencies. Do not
assume `/MT` is safe merely because the outer ABI is C: allocator callbacks,
returned strings, `FILE*`, exceptions, and CRT environment snapshots have owners.
Do not claim static CRT linkage solves the system-read restriction. Verify the
stock JAX/CPython and plugin CRT prerequisites in the clean-install gate; no
system redistributable installer is run automatically.

The Windows wheel must contain inspected sm120 code for AmgX/QUDA. Preserve the
existing full release target list where building a release-architecture wheel;
`--architectures native` or a `120-real` developer build is not evidence for
other GPUs. cuDSS/cuTENSOR architecture support comes from the selected vendor
binaries and must be exercised, not inferred from wrapper SASS.

### E. Packaging, resolution, and installed intent

| File | Planned change and acceptance |
| --- | --- |
| `pyproject.toml` | Add Windows markers to the same `cuda13` extra: exact stock JAX/JAXlib 0.12.0 plus the lockstep numerical and PJRT companions. Retain Linux dependencies and the existing CUDA-major marker. Add Windows cuTENSOR dependencies. For explicit Windows cuTensorNet/cuEST requests, require their actual distributions rather than silently erasing dependencies through Linux-only markers; absence of a compatible wheel is a resolver refusal. Update local-source/build-tool integration without making ordinary wheel installs compile native code. |
| `tools/check_cuda_extras.py` | Validate Windows metadata and actual wheel resolution, exact runtime/companion/JAX versions, no foreign CUDA major, no competing upstream/winjax import-package owner, and absence/mismatch refusal cases. Keep Darwin/unsupported-host and Linux architecture expectations distinct. Reuse existing wheel metadata/RECORD and owned-runtime validation patterns rather than assertions about TOML spelling. |
| `tools/check_metal_extras.py`, `tools/check_accelerator_extras.py` | Update only shared-helper consumers affected by the CUDA resolver changes. Preserve their platform-specific policies and historical evidence. |
| `tools/check_typing.py`, `pyproject.toml` typing configuration | Include the new first-party native runtime Python and packaging backend in the owning typing/annotation gate where existing native companion roots are configured. Do not add another checker or broad suppressions. |
| `tools/check_installed_typing.py` | Use the installed environment's correct interpreter path for Windows while retaining the repository's supported-version policy. Do not add unsupported Phydrax interpreter versions solely for this port. |

The index must contain the Windows dependencies before publishing a core wheel
that advertises the extra. No missing-companion source fallback. Universal lock
resolution and platform wheel selection both need coverage; a lock entry is not
proof that a Windows wheel can actually be installed.

### F. Resource observation, bounded process supervision, and qualification

| File | Planned change and acceptance |
| --- | --- |
| `phydrax/_execution_sampling.py` | Add a Windows host/process memory provider using documented current-process and host-memory APIs. Report working set, high-water working set, physical/available memory, method, and resolution separately. Failed/unsupported observations remain unavailable, never zero. Reuse a host-only measurement boundary without initializing JAX solely to read process memory. |
| `benchmarks/_memory.py` | Remove the unconditional POSIX `resource` import. Reuse the Windows process-memory owner; preserve existing POSIX measurement semantics where practical. Keep unavailable evidence explicit and do not fabricate a numeric peak. |
| `phydrax/_execution_runtime.py` | Keep unknown Windows multi-process host identity unknown. Do not invent a boot ID from hostname or rank. Single-process inventory and allocator-capacity semantics remain unchanged; unsupported distributed requests are already refused at admission. |
| `tools/_accelerator_process.py` | Preserve bounded retained logs, exact status, deadlines, and cleanup of all owned descendants. Dispatch Windows execution to a race-free Job Object implementation. Do not turn `AttributeError`, an unknown exit code, or a parent-only kill into successful cleanup. |
| **New** `tools/_windows_process.py` | Own the Windows child-lifecycle invariant used by the existing qualification runner: create the child suspended, assign it to a non-inherited kill-on-close Job Object, resume only after successful assignment, wait with a deadline, terminate owned descendants, and close handles. Assignment failure terminates the still-suspended child and refuses. Avoid the race in assigning a running `Popen` child to a job. Use an explicit inherited-handle allowlist and no shell. |
| `tools/qualify_cuda_wheels.py` | Add explicit interpreter/work-root inputs, `Scripts/python.exe`, copied harness staging without privileged symlinks, isolated installed-wheel execution, Windows clean/hostile DLL audits, and runtime-wheel identity binding. Add a separately labeled installation/refusal mode for a no-GPU Windows runner; it cannot satisfy hardware qualification. Keep positive `--required-gpus` and complete mandatory provider coverage for hardware mode. |
| `tools/_cuda_optional_qualification.py` | Reuse platform-aware runtime/bootstrap and hostile-library policy. Qualify cuTENSOR's full selected mathematical profile. Explicitly report unavailable Windows cuTensorNet/cuEST requests; do not omit requested providers or label their refusal as numerical qualification. |
| `tools/qualify_cuda_execution.py` | Add an installed-interpreter/work-root path, Windows-safe host inventory, and actual one-device scientific execution. Do not require `lscpu` on Windows. Record empty peer matrices as not applicable, never peer safety evidence. Preserve Linux multi-device campaign gates unchanged. |
| `tests/_support/cuda_execution_worker.py` | Use admitted runtime entry points for device identity and CUDA operations. Keep helper ownership and reference checks; no system DLL guesses. |
| `tools/distributed_substrate_benchmark.py` | Exercise the existing 1x1 weighted-reduction/PCG scenario through the installed runtime. Change only portability/identity plumbing needed for that path. Do not start a distributed coordinator for one process. |
| **New** `benchmarks/windows_cuda.py` | A bounded campaign driver composing existing numerical and memory/timing owners, not a new numerical implementation. Cover ordinary scientific routes, compensated/MPM behavior, native component cold/prepared execution, and optional cuTENSOR. Record preparation, lowering, compilation, first call, warm execution, derivatives, transfers, compiler temporary/output/code bytes, and logical retained bytes separately. |
| **New generated evidence** `benchmarks/rtx_5090_windows_qualification.json` | Retain actual installed-wheel, runtime, precision, lifecycle, refusal, and benchmark evidence after qualification. Never prepopulate success or copy Linux measurements. Bind exact source/build/wheel/driver/backend identities. |

`tools/_accelerator_process.py` already owns qualification command supervision.
The generic external-runtime API currently relies on POSIX descriptor execution
and does not provide the required Windows process-tree guarantee; do not route
these commands through it and imply stronger isolation. This plan does not
refactor unrelated commercial-provider workers. `tools/_pytest_outcomes.py`
remains unchanged unless an actually selected Windows qualification path invokes
its process runner; if that becomes necessary, reuse the same lifecycle owner
and include its direct consumer tests.

A Job Object is process ownership/cleanup, not a filesystem sandbox. The path
and authorization gate in section 2 remains separate.

### G. Tests and typing fixtures

Existing tests should be extended where they already own the behavior. New tests
below cover new failure boundaries; do not add source-text, import-spelling,
forwarding, default-value, or mock-echo tests.

| File or family | Contract coverage |
| --- | --- |
| `tests/unit/test_cuda_environment.py` | Migrate installation records; supported Windows missing-plugin refusal; same-major duplicate implementations; wrong version/major; metadata-only versus observed state; explicit CPU selection; DLL attribution; incomplete inspection evidence; stable diagnostic ordering. |
| `tests/unit/test_cuda_native_loader.py` | Preserve all identity/handler refusals; add real Windows DLL fixtures and preload/teardown failures. Compiler absence may skip a developer-only fixture test, but the release job must provision the compiler and cannot count that skip as qualification. |
| `tests/unit/test_elf_dependencies.py` | Migrate shared record imports and preserve ELF resolution semantics exactly. |
| **New** `tests/unit/test_pe_dependencies.py`, **new** `tests/_support/pe.py` | External-format fixtures for PE import/export bounds, delayed imports, ordinals/forwarders, architecture mismatch, truncation, and ambiguous ownership. Fixtures encode the PE contract independently of the parser and are not executable stand-ins for GPU providers. |
| **New** `tests/unit/test_cuda_pjrt_packaging.py` | Actual staged-wheel metadata/assets/RECORD/license validation, malformed manifests, missing or tampered binaries, incorrect Python/platform tags, conflicting module ownership, and unavailable source-build paths. Reuse relevant existing packaging test support. |
| **New** `tests/integration/test_windows_cuda_import_contract.py` | Fresh-process startup and DLL behavior from section 6, including no-driver/default-selection refusal, corrupt runtime refusal, explicit CPU behavior, and supported/late import orders. GPU rows run only on the real Windows target; negative installation rows run without a GPU. |
| `tests/integration/test_cuda_import_contract.py` | Keep Linux mapped-library checks; share only actual cross-platform contracts. Do not turn Linux skips into Windows success. |
| `tests/unit/test_configuration_runtime.py`, `tests/unit/test_configuration_resolution.py`, `tests/unit/test_configuration_environment.py` | CUDA installation admission before backend initialization; explicit CPU; unchanged precision defaults; late-setting refusal; unsupported distributed startup fails before side effects. |
| `tests/unit/test_compensated_reductions.py` | Reuse cancellation, seeded recurrence, complex, empty, nonfinite, nonlinear derivative, and batching cases. Exercise the real Windows reference GPU lowering and retain Linux native/reference agreement. Do not weaken exact component comparisons. |
| `tests/unit/discretization/test_mpm_apic_streaming_moments.py` | Reuse the independent NumPy oracle/error bound, masked/garbage/wide indices, zero-weight NaNs, tile tails, empty cases, derivatives, structural sparsity, and compiler-identity checks. Test the canonical Windows GPU route, not a CPU interpreter. |
| `tests/unit/backends/test_atomistic_backend.py`, `tests/unit/backends/test_spatial_backend.py`, `tests/unit/atomistic/test_mace_acceleration.py`, `tests/unit/atomistic/test_mace_execution.py` | Correct availability/refusal and provenance for explicit accelerated targets; native-JAX scientific execution stays separate. Existing explicit interpreter tests retain their reference-only meaning. |
| `tests/unit/test_jax_indexing_frontend.py`, `tests/unit/test_jax_indexing.py` | Stock frontend source admission, full-width indexing and cotangent behavior, import ordering, and unchanged tamper refusal. No Windows exception to source guards. |
| `tests/unit/linalg/test_cudss_provider.py`, `tests/unit/backends/test_amg_backends.py`, `tests/unit/backends/test_lattice_providers.py`, `tests/unit/backends/test_backend_contracts.py` | Mandatory numerical precision/status/reference cases, sparse preparation/refresh, actual device ownership, unsupported topology refusal, exact companion admission, and module-import ordering. Migrate plugin-record consumers. |
| `tests/integration/test_cuda_native_lifecycle.py`, `tests/integration/test_cuda_tensor_contraction.py`, `tests/integration/test_cuda_tensor_teardown.py`, `tests/integration/test_cuda_tensor_slices.py` | Normal and challenged hostile-path exit, live/in-flight state, repeated preparation/rebuild/retirement, optional tensor transformations, and teardown failures. Retain per-device evidence. |
| `tests/unit/test_cuda_build_tools.py`, `tests/unit/test_cuda_wheel_qualification.py`, `tests/unit/test_cuda_optional_qualification.py`, `tests/unit/test_cuda_execution_qualification.py` | Windows resolution/build-audit contracts, positive hardware coverage, refusal-only report separation, exact wheel selection/hash, challenged hostile audits, requested-provider completeness, and timeout/status propagation. Replace the obsolete Windows marker-only acceptance case. |
| `tests/unit/test_accelerator_wheel_qualification.py` | Exercise bounded logs, timeout, parent exit with surviving descendants, process crash, and Job Object assignment failure. Prove owned descendants terminate without touching unrelated processes. Retain POSIX coverage. |
| `tests/unit/test_execution_memory_sampling.py`, `tests/unit/test_execution_resource_admission.py`, `tests/unit/test_benchmark_memory.py`, `tests/unit/test_benchmark_runtime.py`, `tests/unit/test_benchmark_comparison.py` | Windows measurement method/units/unavailability, no invented capacity, allocator versus host memory distinction, runtime/build identity differences, and benchmark comparison refusal across incompatible environments. |
| `tests/unit/test_metal_extras.py`, `tests/unit/test_accelerator_extras.py`, `tests/unit/test_metal_wheel_qualification.py` | Direct consumers of shared resolver/qualification helpers; preserve non-Windows contracts. |
| `tests/typing/cases/cuda_environment.py` | New installation-record inputs, public facts clean cutover, issue/inspection fields, and static misuse. Update other typing cases only where LSP references show changed annotations. |

Before exported/cross-module changes, run LSP references under the authorized
profile-local tooling setup and complete the caller map. The planning session
intentionally did not launch a language server. Supplement references with
registry, source-identity, persistence, documentation, and generated-data checks.

### H. Documentation and generated surfaces

| File | Required update |
| --- | --- |
| `README.md` | Same installation command on supported Windows/Linux, exact prerequisites, and a concise platform capability boundary. |
| `docs/installation.md` | Windows fresh-install instructions, exact JAX/runtime pins, driver/CRT prerequisites, DLL ownership, explicit CPU selection, import ordering, WDDM allocation/watchdog limitations, PyTorch/cuDNN coexistence refusal, optional-provider availability, and links to actual Windows evidence. Preserve historical Linux records and their nonclaims. |
| `native/cuda/README.md` | Windows compiler split, toolkit view, component set, cache paths, DLL inspection, architecture evidence, core provider portability, qualification, and publication. |
| `packaging/cuda-major/README.md` | Windows installed intent now denotes a required actual runtime stack; marker version/content remain unchanged. |
| `docs/plans/single-step-cuda-install.md` | Update the current contract section to include the Windows runtime companion. Preserve the historical RTX 5090 Linux observations explicitly as historical. |
| `docs/api/backends.md`, `docs/api/configuration.md`, `docs/api/execution.md` | Public facts migration, Windows diagnostics and capability admission, component paths, lifecycle/resource evidence, and CPU/configuration precedence. |
| `docs/guides_mace_execution.md` | Distinguish native-JAX Windows execution from unavailable explicit Mosaic acceleration. No reinterpretation of Linux or CPU-interpreter results as Windows GPU evidence. |
| `NOTICE`, relevant `LICENSES/` and runtime-wheel license payload | Preserve attribution for the port, XLA/JAX/LLVM and other shipped native code. Identify actual NVIDIA/CRT redistribution obligations. Do not assume one Apache license covers every bundled dependency. |
| `CHANGELOG.md` | Record Windows support, exact limitations, the public CUDA-facts cutover, compiler/runtime identity changes, and re-preparation requirements. |
| `tools/generate_public_api_manifest.py` -> `docs/data/public_api.json` | Regenerate through the owning generator after public exports/annotations stabilize. Do not hand-edit generated data. |
| `tools/generate_capability_inventory.py` -> `docs/data/capabilities.json`, `application_portfolios.json`, `capability_closure.json`, `source_absorption.json`, and generated capability API pages | Regenerate after support declarations stabilize. Keep candidate profiles separate from observed qualification. Update generator code only if the actual declared contract requires it. |

## 5. Dependency order and parallel ownership

The integration owner maintains one change/verification map. No shared file has
multiple active owners. Agents do not independently run overlapping builds,
formatters, lint, or test jobs; the integration owner schedules the gates once
per relevant tree state.

1. **Authorization and source gate.** Establish permitted execution environment,
   worktree, exact toolchain paths, source/patch/license locks, dependency wheel
   availability, and the Windows runtime package contract.
2. **Early feasibility builds.** Prove the matched PJRT/kernel build and the
   mandatory AmgX/QUDA Windows compilation before changing the published extra.
   These are release gates, not permission to deliver a partial implementation.
   PJRT and companion investigation can proceed independently, but heavy builds
   must be scheduled to avoid resource contention.
3. **Shared inspection/identity contract.** Complete the PE reader, shared
   dependency evidence, runtime manifest, plugin installation record, and
   bootstrap state/error interfaces. One owner integrates these shared contracts.
4. **Independent implementation slices after those contracts are fixed:**
   - Windows PJRT loader/kernel packaging;
   - native companion/toolchain/QUDA/AmgX port;
   - core admission, compiler capabilities, and scientific callers;
   - qualification/process supervision/resource observation and their tests.
5. **Integrated installed-wheel path.** Assemble both companions and core,
   complete resolver tests, then no-GPU installation/refusal checks, then actual
   RTX 5090 numerical/lifecycle qualification. Do not proceed to benchmark
   interpretation while a correctness/refusal gate fails.
6. **Linux integration and performance evidence.** Rebuild affected native
   artifacts, qualify the affected Linux x86-64/aarch64 surfaces, and run bounded
   uncontended measurements on the required hardware. Shared source/build-hash
   changes invalidate old binary qualification for the new artifact.
7. **Final declarations and release.** Update authored docs and examples,
   regenerate canonical public/capability data once, run invalidated consistency
   checks, and complete package-index installation verification before claiming
   the single-command public installation works.

## 6. Verification matrix and observable acceptance

All commands and scenarios in this section are future obligations, not executed
results. Gate 0 applies to every invocation.

### 6.1 Minimal conservative selection

At implementation time, select tests from the changes since the latest merge
into `dev`, plus the direct/transitive consumers named above. Reuse valid evidence
only while its implementation, tests, configuration, dependencies, compiler,
backend, device topology, and material environment remain unchanged.

- CPU-only unit/metadata/parser/configuration checks: one bounded `pytest -n auto`
  invocation, with the total worker budget constrained for the shared host.
- GPU, native lifecycle, import-order, and topology scenarios: dedicated
  invocations with one owned visible GPU and no concurrent unrestricted xdist
  process. Keep CPU references and callbacks available where the workflow
  contract requires them.
- Shared subprocess-runner changes require its direct qualification consumers'
  tests on Windows and POSIX. They do not justify unrelated scientific suites.
- Shared compensated/MPM lowering admission requires direct consumer correctness
  and derivative coverage on CPU, Linux CUDA, and Windows CUDA. Expand to their
  affected scientific families according to the final diff; do not use only a
  synthetic availability test as coverage.
- Do not run the full suite by default. Expand only if actual changes reach
  collection/configuration/shared fixtures or another demonstrated global surface.
- Measure changed complex symbols before/after and decompose by real lifecycle
  invariants. Do not grow a single bootstrap/build function into a large registry
  or hide complex behavior behind trivial forwarding helpers.

### 6.2 No-GPU Windows installation/refusal gate

Use a fresh profile-local environment and the actual built wheels. This gate
must carry an installation/refusal scope and cannot satisfy the hardware gate.

Required scenarios:

1. Ordinary base Phydrax and explicit CPU selection remain usable.
2. Windows `cuda13` resolves both owned companions and the exact stock JAX pair.
   No Linux wheel, foreign major, or second owner of `jax_cuda13_plugin` appears.
3. A missing runtime wheel, missing numerical companion, wrong lockstep version,
   unsupported Python/architecture, and unavailable optional vendor wheel refuse
   installation or admission, rather than producing a marker-only install.
4. With installed CUDA intent and default platforms, missing driver/no visible
   GPU yields a concrete error, not a successful CPU default.
5. Missing/corrupt PJRT assets and failed plugin initialization also refuse with
   default platforms. This specifically covers JAX swallowing discovery errors.
6. Pure fact assessment and metadata admission do not create a device/client.
7. A correct CUDA installation with explicit CPU selection does not initialize
   PJRT/Phydrax GPU components. Upstream kernel-extension imports may load CUDA
   user-space dependencies; do not assert an impossible blanket absence of every
   CUDA DLL merely because CPU was selected.
8. DLL-name comparisons obey Windows rules, while artifact identity preserves
   canonical paths/bytes. Cover spaces, non-ASCII profile paths, duplicate
   basenames, wrong architecture, and rejected path escape.
9. Process timeout, crash, and parent exit leave no owned descendants running;
   unrelated sentinel processes remain untouched.

### 6.3 Actual RTX 5090 installed-wheel gate

Require one observed RTX 5090, correct runtime/build identity, and CUDA-backed
outputs from actual operations. Run outside the source checkout, with an
isolated interpreter and the staged harness. Verify installed bytes against the
selected wheel hashes; a checkout import must not satisfy this gate.

| Scenario family | Required observation |
| --- | --- |
| Startup and placement | Default supported installation selects CUDA; explicit CPU remains explicit; frontend patch accepted before tracing; late Phydrax import retains its existing refusal. Record real device identity and runtime version. |
| Numerical basics | Float32/float64 and complex64/complex128 arithmetic, highest-precision matrix products, bounded scans/reductions, and FFT values agree with independent references within established dtype/conditioning tolerances. No blanket tolerance chosen solely to pass the port. |
| Wide indexing | Out-of-range wide indices, gather fill, scatter cotangents/high bits, negative-index behavior, and transformed variants retain the current contracts without large allocations merely to exercise high-bit values. |
| Compensated/MPM | Existing cancellation/seeded/NaN/empty/complex cases and MPM mask/gather/derivative/error-bound cases run through the actual Windows GPU backend. |
| Differentiation | JVP/VJP duality, nonlinear first derivatives and selected second-order/HVP scenarios through native numerical owners; no CPU-interpreter replacement. |
| Dense/sparse linalg | Native dense factor/solve and supported decomposition routes, native-JAX iterative sparse solve, and the JAX CUDA sparse route if advertised. Check residuals, rank/conditioning/status and failed cases, not finiteness alone. |
| cuDSS | Float32/float64 analysis, factorization, refactorization, true multiple RHS, supported derivative paths, failure cases, bounded resident-cache reuse, eviction/rebuild, and release. |
| AmgX | Existing numerical and convergence/status cases, hierarchy reuse/refresh/rebuild, device ownership, resource bounds, and release/shutdown. Preserve all configured algorithms. |
| QUDA | Complex64/complex128 gauge action, force and Wilson dslash against the existing independent references; tune-cache identity and concurrency; one-process/device rules; shutdown and in-flight work. |
| cuTENSOR when selected | Full supported precision family, transformations, workspace/cache/refusal contracts, and lifecycle checks. One-device host slicing must remain explicitly host-orchestrated, not a collective claim. |
| Unsupported requests | Explicit Mosaic/Pallas, cuTensorNet/cuEST, multi-device/site-MPI, and distributed requests fail at their owners with actionable reasons. |
| Hostile libraries | Actual competing DLL candidates in a profile-local hostile directory, plus already-loaded conflicts, CUDA_PATH/PATH contamination, and supported import orders. Either only the admitted closure loads or the process refuses. An empty hostile directory is not a challenged pass. |
| Resource/lifecycle | Real allocator counters or explicit unavailable fields; repeated preparation/release; failed synchronization/destruction evidence; clean process exit; no changes to shared-host configuration. |

The smoke is a real composed Phydrax workflow in addition to tests: construct a
small native linear/sparse problem and an MPM or compensated scientific
calculation, JIT and differentiate it, inspect placement/status/results, and
exercise prepare-refresh-release for the mandatory providers. Reuse existing
examples/qualification helpers where they own the scientific scenario. Remove
throwaway smoke scaffolding after retaining its evidence.

### 6.4 Linux no-regression gate

Preserve Linux x86-64 and aarch64 metadata/resolution. Rebuild changed native
sources and exercise mandatory provider numerical/lifecycle checks on each
supported host architecture. Preserve existing release SASS/PTX inspection.

Linux compiler-capability admission must still reach its supported optimized
routes. Existing exact compensated comparisons and MPM numerical bounds must
pass. Do not weaken private JAX identity checks, site-MPI admission, hostile ELF
resolution, or full directed-peer requirements.

No Windows change implies that an earlier failed A40/RTX PRO peer route is now
qualified. Historical A100/Spark/RTX 5090 records retain their original identity.

### 6.5 Benchmarks

Run correctness before timing, then benchmark without competing builds/tests or
agent runtime work. Reuse `benchmarks/_runtime.py` and the existing scientific
owners. Include multiple controlling capacities, not one convenient small case.

Retain preparation, lowering, compilation, first execution, warm execution,
derivative execution, explicit transfer cost, compiler temporary/output/code
bytes, native retained bytes, and host/device memory evidence separately. Record
WDDM allocation policy and unavailable counters. Measure native prepared-versus-
cold solves and the canonical GPU lowerings selected without Mosaic.

Compare CPU/GPU only at the same precision/scientific definition and under
comparable conditions. A Linux comparison requires matching material compiler
and runtime inputs or an explicit caveat. No blanket speedup is a release
criterion; failed agreement is a failure, not a benchmark row to discard.

### 6.6 Commands and integration gates

The existing commands remain the primary interfaces. Proposed additions are
`--python`, explicit work-root inputs, and a separate installation/refusal mode
for the CUDA qualifier; these options do not all exist today.

After Gate 0, execute in the owning worktree/environment:

- `python tools/check_typing.py check` with zero diagnostics for the required
  first-party roots, including the new runtime packaging sources.
- Configured Ruff formatting/lint for changed files, applied before
  source/identity-sensitive verification.
- `python tools/audit_selectors.py` and
  `python tools/audit_contract_candidates.py --signatures`, plus applicable
  StrictModule declaration and import-boundary checks.
- `python -m tools.check_cuda_extras` against a complete Linux/Windows wheelhouse,
  including missing/mismatched-companion refusal cases.
- `python -m tools.qualify_cuda_wheels` in the proposed installation/refusal mode
  on a Windows no-GPU runner, then in hardware mode with
  `--required-gpus 1 --libraries cutensor` on the RTX 5090.
- `python -m tools.qualify_cuda_execution` for the actual 1x1 path. Its one-device
  peer matrices must be labeled not applicable.
- `python -m benchmarks.windows_cuda` for the new bounded installed-runtime
  campaign, after correctness and lifecycle gates pass.
- Required Linux affected-surface qualifications and direct shared-tool consumer
  tests, scheduled separately from Windows hardware work.
- The owning public/capability generators, then
  `tools/check_public_api_manifest.py`, `tools/check_capability_consistency.py`,
  `tools/check_import_boundaries.py`,
  `tests/integration/test_public_api_manifest.py`, and
  `tests/integration/test_capability_inventory.py`.
- `python -m tools.check_installed_typing` for the changed public annotations,
  with its Windows interpreter path corrected and only supported interpreter
  contracts accepted.

Do not copy a command with implicit interpreter/tool discovery onto the shared
machine. Record the exact executable paths, environment, worktree state, backend,
wheel hashes, result/skip/refusal, failing node IDs, and duration in one task-level
verification record. One successful invocation is reusable only while its
relevant inputs remain unchanged.

## 7. Identity, migration, and release

The following are intentional representation changes requiring documentation
and exact caller migration, not identity-preserving cleanups:

- `CudaEnvironmentFacts.plugins` and the new installation record replace the
  major-only public representation; no old-field alias remains.
- Shared dependency evidence moves to its native-format owner and records the
  object format explicitly. Migrate record construction/imports and any
  identity-bound diagnostic artifacts; preserve Linux resolution behavior.
- Compiler-bound target identities include the Windows runtime/compiler build
  and actual lowering support. Reprepare affected execution artifacts rather
  than relabeling old checkpoints.
- Native source/patch/build changes create new companion build hashes, including
  Linux artifacts built from the changed source tree.
- Benchmark environment identity records the actual Windows compiler/runtime and
  effective allocation policy. Do not rewrite historical evidence to match it.

Scientific model/axis identities and portable numerical archives do not acquire
Windows labels merely because execution changed. Keep static metadata out of
device runtime leaves and do not use a new backend string to route around
existing scientific owners.

Publish only after complete qualification:

1. Verify that the unchanged major-13 marker is already available.
2. Publish the matching Windows PJRT runtime and numerical companion, together
   with any affected release companion artifacts required by the existing Linux
   contract. Do not overwrite published files.
3. Verify real package-index resolution/installation in a fresh environment.
   A local `--find-links` result does not prove the public command works.
4. Publish core Phydrax with the platform-marked extra only after its required
   artifacts are available.
5. Bind authored documentation and generated capability data to the actual
   qualified scope and retained evidence. No installed-wheel or performance
   claim is made from a build alone.

## 8. Open prerequisites and stop conditions

These do not make the plan a partial implementation; they are explicit gates
that must be cleared before implementation can be declared complete.

| Prerequisite or risk | Required resolution; prohibited shortcut |
| --- | --- |
| Absolute user-directory file-access restriction | Obtain an explicit execution permission change or a separately authorized runner. No implicit system reads, no copied system dependencies from forbidden paths. |
| JAX 0.12 Windows native port | Produce and qualify the matched PJRT and kernel artifacts. Do not install winjax 0.11.2 with ignored dependencies, change version labels, or relax source guards. |
| Bazel/LLVM/MSVC/SDK and vendor wheel layouts | Lock exact available inputs, inspect RECORDs/import libraries, and prove the build. Do not guess missing package versions, hashes, DLL names, or compiler compatibility. |
| QUDA and AmgX Windows portability | Complete the real ports and mandatory numerical/lifecycle checks. QUDA is the largest unproven compilation surface. Do not remove providers or silently replace them with native-JAX implementations. |
| Windows CRT closure | Verify the actual dependency/ownership contract and document prerequisites. No automatic machine-wide CRT installation and no assumption that a C ABI makes all CRT choices interchangeable. |
| WDDM memory/watchdog limits | Keep bounded workloads, report failures, and preserve allocator evidence. Do not change TDR settings, preempt unrelated processes, or accept a timeout as success. |
| Missing optional vendor binaries | Explicitly refuse unsupported installation/provider requests. No Linux-DLL translation, remote execution, WSL launcher, or numerical substitute hidden under the same provider name. |
| Package index availability/size/license obligations | Resolve actual publication prerequisites before advertising the extra. Local builds are not public availability. |
| Linux/aarch64 regression hardware | Obtain the required runner/evidence for changed artifacts. Existing historical results do not qualify newly built binaries. |

## 9. Source references

- [winjax source and packaging](https://github.com/eterevsky/winjax)
- [winjax package metadata](https://pypi.org/pypi/winjax/json)
- [winjax RTX 5090 audit](https://github.com/eterevsky/winjax/blob/main/docs/TEST_AUDIT_0.11.2.md)
- [winjax known limitations](https://github.com/eterevsky/winjax/blob/main/KNOWN_ISSUES.md)
- [Alternative Windows build scripts](https://github.com/pctablet505/jax-windows-cuda-build)
- [JAX 0.12 pinned module graph](https://github.com/jax-ml/jax/blob/8dcb7aeced770bd22531b42d0b925c7342787c27/MODULE.bazel)
- [JAX 0.12 pinned Bazel version](https://github.com/jax-ml/jax/blob/8dcb7aeced770bd22531b42d0b925c7342787c27/.bazelversion)
- [JAX plugin discovery and backend initialization](https://github.com/jax-ml/jax/blob/8dcb7aeced770bd22531b42d0b925c7342787c27/jax/_src/xla_bridge.py)
- [JAX installation/platform support](https://docs.jax.dev/en/latest/installation.html)
- [Windows cuDSS wheels](https://pypi.org/simple/nvidia-cudss-cu13/)
- [Windows cuTENSOR wheels](https://pypi.org/simple/cutensor-cu13/)
- [cuTensorNet wheel platforms](https://pypi.org/simple/cutensornet-cu13/)
- [CUDA Windows compiler requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Pinned AmgX source](https://github.com/NVIDIA/AMGX/tree/91a8413)
- [Pinned QUDA source](https://github.com/lattice/quda/tree/ba501e4)

Repository evidence is retained in `docs/plans/single-step-cuda-install.md`,
`docs/installation.md`, the existing CUDA qualification records, and the owning
source/tests listed above. None of those historical records is a native Windows
Phydrax qualification result.
