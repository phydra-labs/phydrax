# Single-step CUDA installation: implementation plan

## 0. Authority, evidence, and scope

This is a file-by-file implementation plan. Phase 1 (§3) and Phase 2 (§4: 2a, 2b, 2c together) are implemented; §8 and §9 record the implementations and their evidence, including the release steps that remain (§9). Goal: `pip install "phydrax[cuda12]"` or `pip install "phydrax[cuda13]"` (equivalently `uv add` / `uv pip install`) is the only user step. Every CUDA-dependent component is then automatically consistent with that CUDA major. No system CUDA toolkit, no `LD_LIBRARY_PATH`, no environment variables, and no source builds on the user's machine.

- Implemented on branch `feature/cuda-provider-hardening`, based on `dev` at `e7b82d23c`.
- Line references in §0-§7 are against the planning tree and must be re-read before later edits.

### Evidence established in this session (reuse; do not re-derive)

| Fact | Evidence |
|---|---|
| The current main venv mixes CUDA majors in one process | CUDA 13 JAX runs alongside system CUDA 12.8 cuBLAS/cudart/cuSOLVER/nvJitLink, which QUDA, AmgX, CuPy and the mpi4jax bridge load. The CUDA 12.8 AmgX build silently used CUDA 13's `libcusparse.so.12`, because that soname is the same in both majors. Measured from `/proc/self/maps`. |
| An all-CUDA-13 stack works | A scratch venv with CUDA 13 JAX plus CUDA 13 builds of QUDA, AmgX, pyamgx, PyQUDA, and mpi4jax: provider tests 60 passed / 2 skipped (deliberate dtype parametrizations); CUDA-direct MPI 2 passed. The in-process audit maps only CUDA 13 wheels, driver libraries, the CUDA 13 QUDA/AmgX builds, and the driver-API-only Open MPI. |
| An all-CUDA-12 JAX stack works | A scratch venv with CUDA 12 JAX: GPU tests 409 passed / 14 skipped; providers 43 passed / 2 skipped. Every mapped CUDA library is CUDA 12. |
| QUDA and AmgX compile with only pip NVIDIA wheels as the toolkit | `nvcc` 13.4 comes from `nvidia-cuda-nvcc`. Also required: header wheels `nvidia-curand`, `nvidia-nvml-dev`, `nvidia-cuda-profiler-api`, `nvidia-nvtx`; unversioned dev symlinks in the build tree; an explicit RPATH. |
| QUDA CUDA 13 build | QUDA develop `ba501e4` with `QUDA_DOWNLOAD_CCCL=OFF` (toolkit CCCL 3.4.3) and a reduced Wilson/clover config. `libquda.so` is 31 MB for `sm_120` only. Configure downloaded Eigen 3.4.0 even with `QUDA_USE_EIGEN=OFF`. |
| AmgX CUDA 13 build | Requires AmgX main `91a8413` (merged #374 CCCL fix) plus open PR #386 (classical-AMG invalid device function). The default RUNPATH is empty. `libamgxsh.so` is 150 MB for `sm_120` only. Licensing: `LICENSES/BSD-3-Clause.txt` and `NOTICES`. |
| Provider dependency closures | pyamgx's extension needs only `libamgxsh.so`. `libamgxsh.so` needs `libcublas.so.12`, `libcusparse.so.12`, `libcusolver.so.11`, with cudart linked statically. PyQUDA's `quda*.so` needs `libquda.so`, which needs `libcublas.so.12` and no cudart. mpi4jax's `mpi_xla_bridge_cuda` needs `libcudart.so.12`, and `import mpi4jax` loads it unconditionally (`mpi4jax/_src/xla_bridge/__init__.py:92-93`). |
| AmgX C API | No matrix or solver clone exists (`include/amgx_c.h`). CSR and vector uploads use blocking `cudaMemcpy` (`src/amgx_c.cu:894-898, 1113, 1201`), which accepts device pointers. |
| Spineax's native state model | `spineax/cudss/solver.cpp:104-132`. An id names one immutable numeric state. Every numeric phase rekeys its entry to a fresh id. Every call carries the full CSR data, and a call whose id is not resident rebuilds that state from its own operands. The registry is an LRU whose capacity comes from `SPINEAX_FACTOR_CACHE` (default 8), with a `rebuilds` counter. The Python layer uses a private primitive with MLIR lowering and `has_side_effect=True` (`solver.py:206-275`). |
| Import behavior on the merged tree | `import jax` maps `libcudart.so.13` without initializing a backend. `import phydrax` is device-inert (upstream contract, `docs/api/configuration.md:27-32`). `import phydrax.backends` and `import phydrax.linalg` initialize the JAX backends through `phydrax/_doc.py:9` (`jr.key(0)` at module level). |
| Upstream configuration owners | `phydrax/_jax_config.py` is the sole JAX configuration owner (`pyproject.toml` TID251 banned APIs; `tools/check_import_boundaries.py`). Native library overrides: `ProviderConfig.library_paths` keyed by `LibrarySetting` (`phydrax/_config_types.py:30`), declared variables in `LIBRARY_ENVIRONMENT` (`phydrax/_config_environment.py:39-40`), read through `library_input` (`phydrax/_config_state.py:295-299`), as `phydrax/_meshcore.py:38-39, 264-277` does. |
| Launch coordination | Startup consumes explicit `PHYDRAX_*` coordinator settings, or detects Slurm/Open MPI/PMI/PMIx cluster sizes greater than 1 and autodetects (`phydrax/_config_environment.py:361-369`, `docs/api/configuration.md:31-38`). Collectives use NCCL through `JaxCollectiveProvider` (`docs/api/execution.md:183-186`). |
| Native companion pattern | `native/meshcore`: scikit-build-core, `wheel.py-api = "py3"`, ctypes-loaded C ABI, lockstep version check in `phydrax/_meshcore.py`. Not published on PyPI. |
| Extras exclusivity via metadata | Throwaway packages tested in uv 0.13.0 and pip. A shared marker pinned `==12.*` vs `==13.*` makes `[cuda12,cuda13]` and `[cuda12,cudss]` fail resolution. A self-referencing extra resolves. |
| Public JAX facts | `jax.devices()[0].client.platform_version == "cuda 13040"`. `jax.ffi.ffi_call` exposes `has_side_effect` (an unordered effect that prevents dead-code elimination) and `vmap_method` (`jax/_src/ffi.py:418, 639, 725-753`). |
| No CI or release tooling | No `.github/`; `uv.lock` is gitignored. PyPI `phydrax` 0.2.2 does not match the current `pyproject.toml`. |

### External facts (primary sources gathered this session)

| Topic | Fact |
|---|---|
| uv | `[tool.uv]` settings apply only at the project/workspace root, never to consumers of a published wheel; pip ignores them ([config](https://docs.astral.sh/uv/concepts/configuration-files/), [sources](https://docs.astral.sh/uv/concepts/projects/dependencies/)). The built-wheel cache keys registry sdists by name/version/build settings, not by the environment's CUDA. |
| Wheel variants | PEP 817 is Draft; PEP 825 is Provisional; uv only parses variant labels in preview. Nothing selects a CUDA major from the driver for arbitrary packages today. |
| conda-forge / pixi / mise | conda-forge JAX/jaxlib latest is 0.10.2 (we require 0.11). It has no QUDA, AmgX, pyamgx, Spineax, qlat, or Grid, and its `mpi4jax` has no CUDA build. pixi does nothing for pip users. mise cannot assemble consistent CUDA libraries. Not adopted. |
| PyPI MPI wheels | `openmpi`/`mpich` wheels are built `--without-cuda`. mpi4py's ABI loader prefers an in-process library, then `<venv>/lib`, then the system. |
| mpi4jax | sdist only. Open issue #309 (CUDA 13 segfault). Maintainers recommend sharding over GPU-direct MPI. |
| PyPI size limits | 100 MB per file and 10 GB per project by default. Increases are requested via pypi/support after a release under the limit. |
| Licenses | QUDA: MIT-style. AmgX: BSD-3-Clause plus bundled-component notices. Spineax: MIT (credited in `NOTICE:290-294`). Eigen: MPL-2.0. CUDA libraries and cuDSS are depended on via NVIDIA wheels, not redistributed. |

## 1. Target user contract

```text
pip install "phydrax[cuda13]"     # NVIDIA driver R580+ (reports CUDA >= 13.0)
pip install "phydrax[cuda12]"     # NVIDIA driver R525+ (reports CUDA >= 12.0)
pip install phydrax               # unchanged CPU default
```

| Installed by the extra | Phase |
|---|---|
| `jax[cudaNN]` (JAX plugin and NVIDIA wheels including NCCL) | 1 |
| `phydrax-cuda-major==NN.*` exclusivity marker | 1 |
| cuDSS: Spineax on CUDA 13 in Phase 1; Phydrax-owned binding on both majors in Phase 2a | 1 / 2a |
| AmgX (Phydrax-owned binding, no pyamgx) | 2b |
| QUDA (Phydrax-owned binding, no PyQUDA/CuPy) | 2c |
| Multi-GPU and multi-node communication: JAX/NCCL, coordinated automatically under `srun`/`mpirun` (already implemented) | 1 (documentation) |

What remains outside the single step, with reasons that reach the user through the CUDA environment report:

- **NVIDIA driver.** Pip cannot install it. Phydrax reports the driver's supported CUDA version and the extra that fits.
- **GPU-direct MPI (mpi4jax `gpu_direct=True`).** Requires the site's CUDA-aware MPI and mpi4jax compiled against it. Shipping our own MPI would shadow the site MPI on clusters (mpi4py prefers `<venv>/lib`). `phydrax[mpi]` stays an opt-in interop extra.
- **Multi-process QUDA.** The packaged QUDA is single-process (`QUDA_MPI=OFF`); multi-process lattice QUDA requires a site build.
- **CPU-only native providers** (Grid/GPT, qlat). Out of scope.

## 2. Binding decisions

1. **No new environment manager.** uv and pip are the installers. Every guarantee is expressed in published wheel metadata plus runtime checks. No pixi, no mise, no `[tool.uv]`-only consumer behavior.
2. **One marker, two companions.**
   - Marker distribution `phydrax-cuda-major` is published at versions `12.0` and `13.0` and is never re-released.
   - Both extras require `phydrax-cuda-major==NN.*` unconditionally in every phase, so `[cuda12,cuda13]` is unsatisfiable on every platform in pip and uv (tested).
   - The lockstep companions `phydrax-cuda12` / `phydrax-cuda13` require the same marker.
   - Distinct companion names keep `pip install -U` within one major.
3. **Lockstep and strictness.**
   - Phydrax pins `phydrax-cudaNN==<phydrax version>` with marker `sys_platform == 'linux' and platform_machine == 'x86_64'`.
   - Companions are published as wheels only, never an sdist, so no platform falls back to a CUDA source build.
   - The companion loader requires: an equal version, a CUDA major equal to JAX's, and its own identity symbols (the `phydrax-meshcore` contract).
4. **Phydrax-owned bindings through public JAX FFI only.**
   - Companions are C-ABI shared libraries. Handlers are defined with `XLA_FFI_DEFINE_HANDLER_SYMBOL` (`jaxlib/include/xla/ffi/api/api.h:2909`) and exported as `extern "C"` symbols.
   - Libraries are loaded with ctypes and registered via `jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(fn), platform="CUDA")`.
   - Calls go only through `jax.ffi.ffi_call`, with the per-target settings of §4.2. No MLIR lowering, custom primitives, or jaxpr-effect internals.
   - Wheel tag is `py3` (no CPython ABI). The companions replace Spineax, pyamgx, PyQUDA, and CuPy in one clean cutover.
5. **Spineax's cuDSS native core is vendored**, including its state model (MIT; `solver.cpp`, `csr_transpose.cpp`). Adaptations:
   - registry capacity from a policy attribute instead of `SPINEAX_FACTOR_CACHE`;
   - rebuild counts returned as call outputs;
   - Phydrax target names;
   - symbol handlers instead of nanobind capsules.

   Spineax's Python primitive/lowering layer is not vendored. AmgX and QUDA handlers are new Phydrax code over their public C APIs.
6. **Per-component libraries in one wheel.**
   - `libphydrax_cuda_cudss.so`, `libphydrax_cuda_amgx.so`, and `libphydrax_cuda_quda.so` are each loaded lazily on first use, so a cuDSS solve never maps the AmgX binary.
   - QUDA and AmgX are linked statically with hidden visibility (a version script exporting only `phx_cuda_*` and handler symbols), so a user's own QUDA/AmgX cannot collide.
   - The CUDA runtime is always shared (`CMAKE_CUDA_RUNTIME_LIBRARY=Shared`).
7. **No system CUDA anywhere, including under a hostile `LD_LIBRARY_PATH`.**
   - Builds use only pip NVIDIA wheels as the toolkit.
   - At runtime `RUNPATH` loses to `LD_LIBRARY_PATH`, and several sonames are identical across majors (`libcurand.so.10`, `libcudss.so.0`, `libcusparse.so.12`, `libnccl.so.2`, `libcudnn*.so.9`, `libnvshmem_host.so.3`).
   - The loader therefore computes each component's transitive NVIDIA dependency closure and preloads it by absolute path, dependency-first, before loading the component. NVIDIA package directories are located through their Python package `__path__`, the mechanism JAX uses (`jax_plugins/xla_cuda13/__init__.py:100-144`). The built `RUNPATH` remains a secondary path.
8. **JAX/NCCL is the multi-GPU path.** Startup coordination (explicit `PHYDRAX_*` settings or launcher autodetection) and `JaxCollectiveProvider` already exist. Phase 1 documents them as the GPU route. MPI stays a launcher or interop concern.
9. **CUDA-major checks happen before loading.**
   - A provider's availability is decided before importing it, from static facts: installed distributions, plus the transitive ELF dependency closure of its native libraries, resolved like the dynamic loader (new `phydrax/_elf_dependencies.py`).
   - Probe facts exclude mapped libraries, and closures are resolved as a fresh process would load them, ignoring this process's mappings. Availability therefore never depends on probe order or earlier imports.
   - The `/proc/self/maps` audit is diagnostic evidence only (`cuda_environment()` and release qualification).
10. **Import contract.**
    - `import phydrax` is device-inert; upstream owns and tests that.
    - Importing subpackages initializes the JAX backend through `phydrax/_doc.py:9`. That is pre-existing behavior, outside this plan.
    - This plan guarantees: importing `phydrax`, `phydrax.backends`, or `phydrax.linalg` loads no provider or companion native library (`libcudss`, `libamgxsh`, `libquda`, `libphydrax_cuda_*`) and registers no Phydrax FFI target.
    - The static assessment never initializes a JAX backend.
    - Token/FFI layers live in lazily imported modules; modules imported eagerly hold only static types and validation.
11. **Phased releases, each complete.**
    - Phase 1: extras, exclusivity, the report, the pre-load guards, and documentation.
    - Phase 2a: the companion with cuDSS. It is the smallest wheel, which satisfies PyPI's under-limit-first rule before the size request.
    - Phase 2b adds AmgX; Phase 2c adds QUDA.
    - An experiment that fails its gate is removed, never shipped as a placeholder.
12. **Native state is a bounded memo, not an owned resource.** This is Spineax's model, made explicit:
    - Every FFI call carries all operands needed to rebuild its native state.
    - Each device's registry is partitioned by the owning plan's identity, passed as a handler attribute. Each partition keeps at most its policy-declared number of entries: `SparseLDLT.factor_cache_capacity` (default 8, Spineax's current default) and `AmgXPolicy.hierarchy_cache_capacity` (default 2). Plans never evict each other's entries, and there is no environment variable.
    - A numeric phase or refresh rekeys its entry to a fresh id. A call whose id is not resident rebuilds the identical state from its own operands, and the rebuild count is returned in the call's result evidence.
    - Release frees early: eagerly through ctypes outside traces, through a side-effecting FFI call inside traces.
    - Correctness therefore does not depend on call order, dead-code elimination, common-subexpression elimination, rematerialization, or release; batching follows the explicit rules of §4.2. Each partition stays within its capacity, so traced refreshes cannot leak.
    - A rebuild recomputes the requested function exactly from its inputs. It is not a fallback for a failed path, and it is counted.
    - QUDA keeps no registry: each call loads its gauge links (§4.1).
13. **Exit ordering is owned.**
    - One Phydrax `atexit` handler is registered at first native load. JAX registers its handlers at import (`jax/_src/api.py:2853`, `dispatch.py:176`), so the Phydrax handler runs first.
    - It calls `jax.effects_barrier()`, synchronizes devices, releases all registries, then calls `endQuda`.
    - Native registries are heap-allocated and never destroyed during C++ static destruction.
14. **Configuration goes through the upstream owners.**
    - JAX settings are read through `phydrax._jax_config`.
    - Companion overrides are new `LibrarySetting` values in `ProviderConfig.library_paths` with declared `PHYDRAX_CUDA_<COMPONENT>_LIBRARY` variables, read through `library_input`.
    - ABI checks and loaded-provider identity stay with the provider owner (`docs/guides_configuration.md:150-152`).
    - `LD_LIBRARY_PATH` is read directly as dynamic-loader diagnostics, not as configuration.

## 3. Phase 1 — extras, exclusivity, CUDA environment report, pre-load guards

### 3.1 Packaging

| File | Change |
|---|---|
| **New** `packaging/cuda-major/12/pyproject.toml`, `packaging/cuda-major/13/pyproject.toml` | Metadata-only hatchling wheels (`bypass-selection = true`), name `phydrax-cuda-major`, versions `12.0` / `13.0`, a description stating the exclusivity purpose, license `LICENSE`. No Python module. |
| **New** `packaging/cuda-major/README.md` | Purpose, the never-re-release rule, and publishing order (marker before any Phydrax release that references it). |
| `pyproject.toml:93-95` | Remove the `cudss` extra (clean cutover; the cuDSS bridge moves into `cuda13`). |
| `pyproject.toml` optional-dependencies | `cuda12 = ["jax[cuda12]>=0.11.2,<0.12; sys_platform == 'linux'", "phydrax-cuda-major==12.*"]` and `cuda13 = ["jax[cuda13]>=0.11.2,<0.12; sys_platform == 'linux'", "phydrax-cuda-major==13.*", "spineax>=0.0.5,<0.1; sys_platform == 'linux' and platform_machine == 'x86_64'"]`. |
| `pyproject.toml:248-255` `[tool.uv] conflicts` | Add `[{extra = "cuda12"}, {extra = "cuda13"}]` (developer lock forks only). |
| `pyproject.toml:256` `[tool.uv.sources]` | `phydrax-cuda-major = [{ path = "packaging/cuda-major/12", extra = "cuda12" }, { path = "packaging/cuda-major/13", extra = "cuda13" }]` for development before publication. Consumers never see sources. The release gate uses `--find-links` until the marker is on PyPI. |

### 3.2 Runtime owner: new `phydrax/_cuda_environment.py`

The single owner of CUDA environment facts and judgments. Host-only fact collection is separated from a pure assessment.

**`CudaEnvironmentFacts`** (final `StrictModule`, static fields):

| Field | Meaning and source |
|---|---|
| `requested_major: int \| None` | Major of the installed `phydrax-cuda-major` distribution (`importlib.metadata`). |
| `plugin_majors: tuple[int, ...]` | Installed `jax-cuda12-plugin` / `jax-cuda13-plugin` distributions, read from metadata without importing them. |
| `foreign_nvidia_distributions: tuple[str, ...]` | Installed NVIDIA distributions of the other major. Classified by an explicit per-family table and each distribution's RECORD install location, never by version number. CUDA 13 math libraries are unsuffixed distributions with 12.x versions (`nvidia-cusparse 12.8`, `nvidia-cusolver 12.3`, `nvidia-cufft 12.4`) installing under `nvidia/cu13/`. Leftovers matter because JAX's CUDA 13 plugin loads `libcudart.so.12` whenever `nvidia.cuda_runtime` is importable (`xla_cuda13/__init__.py:125`). |
| `selected_platforms: tuple[str, ...] \| None` | `phydrax._jax_config.platforms()`. |
| `backends_initialized: bool` | `phydrax._jax_config.backends_initialized()`. |
| `runtime_version: tuple[int, int] \| None` | Parsed from `device.client.platform_version` (`"cuda 13040"` → `(13, 4)`). Collected only with `initialize_backend=True`. |
| `device_probed: bool` | Whether JAX devices were queried (`initialize_backend=True`). Without a probe, the runtime is unknown rather than absent. |
| `driver_version: tuple[int, int] \| None` | `cuDriverGetVersion` through `ctypes.CDLL("libcuda.so.1")`. This maps the driver library but calls no `cuInit` and creates no context. |
| `library_path_cuda_dirs: tuple[str, ...]` | `LD_LIBRARY_PATH` entries containing any CUDA-family library (cudart, cublas, cusparse, cusolver, cufft, curand, nvJitLink, nvrtc, cudnn, nccl, cudss). |
| `mapped_libraries: tuple[MappedCudaLibrary, ...]` | From `/proc/self/maps`: soname, real path, attributed major. Collected only by `cuda_environment()`; diagnostic only. |

**Major attribution** is shared with §3.3:

| Kind | Rule |
|---|---|
| Major-bearing sonames | `libcudart.so.N`, `libcublas*.so.N`, `libnvrtc.so.N`, `libnvJitLink.so.N`, `libcupti.so.N` give major N. `libcusolver.so.11` / `libcufft.so.11` mean CUDA 12; `.so.12` means CUDA 13. |
| Ambiguous sonames | `libcusparse.so.12`, `libcurand.so.10`, `libcudss.so.0`, `libnccl.so.2`, `libcudnn*.so.9`, `libnvshmem_host.so.3`. Attributed only by owning distribution (RECORD), by a `/usr/local/cuda-X.Y/` path, or by system package directory. Otherwise `unknown`: reported, not counted as a second major. |

**Issue codes and types:**
- `CudaIssueCode: TypeAlias = Literal["no-cuda-plugin", "duplicate-cuda-plugins", "plugin-major-mismatch", "foreign-nvidia-distributions", "driver-too-old", "cpu-platform-selected", "backend-unavailable", "library-path-shadowing", "mixed-cuda-majors", "provider-major-mismatch", "provider-dependency-unresolved"]` and `CudaIssueSeverity: TypeAlias = Literal["error", "warning"]`. Both are declared once, runtime-importable, and parsed with `phydrax.typing.parse`.
- `CudaEnvironmentIssue` (final `StrictModule`): code, severity, message, related paths.
- `CudaEnvironmentReport` (final `StrictModule`): facts, issues, `successful` (no error-severity issue).

**`assess_cuda_environment(facts) -> CudaEnvironmentReport`** is pure. Its rules:

| Condition | Issue |
|---|---|
| Marker installed, no plugin | `no-cuda-plugin`: error on Linux x86_64; warning elsewhere, naming the platform |
| Both plugins installed | `duplicate-cuda-plugins` (error) |
| Marker major ≠ plugin major | `plugin-major-mismatch` (error) |
| Other-major NVIDIA distributions installed | `foreign-nvidia-distributions` (error, with the exact uninstall list) |
| Driver major below plugin major | `driver-too-old` (error, naming the extra that fits the driver) |
| Plugin installed, selected platforms exclude CUDA | `cpu-platform-selected` (warning; CPU test workers do this deliberately) |
| Plugin installed, CUDA not excluded, device probed, no CUDA device | `backend-unavailable` (error; includes `library_path_cuda_dirs` when present, the observed cause of CUDA 12 plugin failure) |
| Shadowing directories while the backend works | `library-path-shadowing` (warning) |
| Mapped libraries attributed to more than one major | `mixed-cuda-majors` (error, diagnostic only) |

**Functions:**
- `assess_provider_dependencies(closure, facts) -> tuple[CudaEnvironmentIssue, ...]`. Pure. A closure containing a library whose attributed major differs from the plugin major gives `provider-major-mismatch`, naming each library and requesting object. An unresolved CUDA-family soname gives `provider-dependency-unresolved`. A closure with no CUDA-family library (fully static) produces no issue.
- `collect_cuda_environment(*, initialize_backend: bool) -> CudaEnvironmentFacts`. Host-only, never called at import. With `initialize_backend=False` it touches only metadata, `_jax_config` queries, the environment, and the driver version; it sets `device_probed=False` and collects no mapped libraries.
- `cuda_environment() -> CudaEnvironmentReport`: collects with backend initialization and mapped libraries, then assesses.

Exports through the explicit `phydrax.backends` table: `cuda_environment`, `assess_cuda_environment`, `CudaEnvironmentReport`, `CudaEnvironmentIssue`, `CudaEnvironmentFacts`, `MappedCudaLibrary`, `CudaIssueCode`, `CudaIssueSeverity`. Collection and provider-closure functions stay private.

### 3.3 New `phydrax/_elf_dependencies.py`

Host-only dynamic-loader model, standard library only (`struct`, `os`, `pathlib`; no `ldd`/`ldconfig` subprocesses).

- **`read_dynamic_section(path) -> ElfDynamic`.**
  - Parses ELF64 little-endian (x86_64 and aarch64): file header → `PT_DYNAMIC` → `DT_NEEDED`, `DT_RPATH`, `DT_RUNPATH`, `DT_STRTAB`/`DT_STRSZ` (mapped through `PT_LOAD` segments).
  - Any other format raises `ValueError`, which becomes a refusal reason.
- **`resolve_dependency_closure(path, *, library_path) -> DependencyClosure`.** Transitive and breadth-first in the loader's order, as a fresh process would resolve it. This process's mappings are deliberately ignored (decision 9).
  1. `DT_RPATH` of the requesting object and its loaders, when the requesting object has no `DT_RUNPATH`;
  2. `LD_LIBRARY_PATH`;
  3. `DT_RUNPATH` of the requesting object;
  4. `/etc/ld.so.cache`, the `glibc-ld.so.cache1.1` format;
  5. default directories (`/lib/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu`, `/lib64`, `/usr/lib64`, `/lib`, `/usr/lib`, and aarch64 equivalents).

  `$ORIGIN` is expanded. `$LIB`/`$PLATFORM` tokens and `glibc-hwcaps` subdirectories are not modeled: a dependency that would need them is reported unresolved.
- **`DependencyClosure`**: resolved entries (soname, path, requesting object) in dependency-first order, plus unresolved sonames. Classification uses §3.2's attribution table. Phase 2's companion preloading reuses the same closure.

### 3.4 Availability integration

| File | Change |
|---|---|
| `phydrax/backends/_types.py:175-224` `BackendAvailability` | Add `environment_issues: tuple[CudaEnvironmentIssue, ...]` (default empty). Typed issues reach consumers, not only DEBUG logs (`_availability.py:56-68`). This changes the `BackendAvailability` field set (documented). |
| `phydrax/backends/_availability.py:71-172` `probe_backend` | New optional `cuda_libraries: Callable[[], tuple[str, ...]] \| None`, which locates the provider's native libraries without importing it, from `importlib.util.find_spec(name)`: `origin` for single-file extension modules, package-directory globs otherwise. Whenever `cuda_libraries` is given and a JAX CUDA plugin is installed with CUDA not excluded, independent of `device_platform`: first assess static facts (`initialize_backend=False`) and refuse on any error issue; then, before `import_backend_module` (`:136`), refuse on any `assess_provider_dependencies` issue. Existing category order is preserved. |
| `phydrax/backends/spineax.py:28-39` | Requirement text becomes "installed by `phydrax[cuda13]`". Passes `cuda_libraries` for Spineax's extension, whose closure (`libcudart.so.13`, `libcudss.so.0`) is CUDA 13, so a CUDA 12 process refuses it before import. |
| `phydrax/linalg/_sparse_providers.py:194-222` | Replace its private CUDA device check with the same static assessment and closure rule, so `sparse_provider_availability("spineax-cudss")` and `spineax_availability` agree. |
| `phydrax/backends/amg.py:320-341` | Passes `cuda_libraries` for pyamgx, a single-file extension located through `find_spec("pyamgx").origin`. The transitive closure reaches `libamgxsh.so` and its cuBLAS/cuSOLVER sonames. |
| `phydrax/backends/lattice.py:476-486` (`pyquda_availability`, `:479-485`), `phydrax/backends/_lattice_pyquda.py:64-80` | `pyquda_availability` passes `cuda_libraries` for PyQUDA's extensions (closure reaching `libquda.so`), so the check runs before PyQUDA is imported. CuPy's distribution name (`cupy-cuda12x` / `cupy-cuda13x`) is checked against the plugin major before `pyquda.init` or any CuPy import. |
| `phydrax/backends/distributed.py:203-213` `mpi4jax_availability`, `:297-341` `Mpi4JaxCollectiveProvider.__init__` | `mpi4jax_availability` passes `cuda_libraries` for `mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda*.so` (located through `find_spec("mpi4jax").submodule_search_locations`), so the check runs inside `probe_backend` before mpi4jax is imported. This covers both host-staged and GPU-direct routes, because importing mpi4jax loads the CUDA bridge unconditionally. `Mpi4JaxCollectiveProvider.__init__` obtains the module through `mpi4jax_availability()` (`:311`), so no separate check is added. The existing GPU-direct messages asserted by `tests/integration/test_cuda_direct_mpi.py:45-49` are unchanged; the new refusal joins the worker's skip (exit 77) set. |

### 3.5 Distributed and MPI (documentation and an upstream fix only)

- No code change to `JaxCollectiveProvider`, `phydrax/_execution_bootstrap.py`, or the startup coordination in `phydrax/_config_environment.py:361-369`. They already provide the GPU route, coordinated automatically under `srun`/`mpirun`.
- The `phydrax[mpi]` extra stays opt-in.
- External change, not repository code: an upstream mpi4jax PR so its build detects the CUDA 13 wheel layout (`nvidia/cu13/{include,lib}`) and links `libcudart.so.13` with an RPATH, mirroring its CUDA 12 path (`setup.py:117-147, 239-305`).
- Until that is merged, docs give the verified recipe:
  - point `CUDA_ROOT` at a toolkit view;
  - set `LDFLAGS=-Wl,-rpath,<venv>/nvidia/cu13/lib`;
  - run `uv pip install --no-cache --no-build-isolation --reinstall mpi4jax`.

### 3.6 Release gate tool

**New** `tools/check_cuda_extras.py`, run online by the release owner. It must report zero failures before any release.

1. Builds the Phydrax wheel (`uv build --wheel`) and reads its `METADATA`, so the check targets the published metadata, not `pyproject.toml` text. Until the marker is on PyPI, it compiles with `--find-links` to locally built marker wheels.
2. Runs `uv pip compile` of that wheel with `[cuda12]` and with `[cuda13]` (both must succeed) and with `[cuda12,cuda13]` (must fail). Each runs for `--python-platform` `x86_64-manylinux_2_28`, `aarch64-manylinux_2_28`, `aarch64-apple-darwin`, and `x86_64-pc-windows-msvc`. `x86_64-apple-darwin` is excluded: JAX 0.11 publishes no Intel macOS jaxlib, so base Phydrax does not resolve there.
3. Checks single-major consistency of each Linux x86_64 resolution against an explicit table of NVIDIA families (cuda-runtime, cublas, cusparse, cusolver, cufft, curand, nvjitlink, nvrtc, cupti, cudnn, nccl, nvshmem, cudss). Families are classified by distribution name and declared major, never by version number.
   - `[cuda13]` must contain `nvidia-cuda-runtime==13.*` and no `-cu12` distribution.
   - `[cuda12]` must contain `nvidia-cuda-runtime-cu12` and no `-cu13` or unsuffixed CUDA 13 family.
4. Phase 2: on Linux x86_64 the resolved `phydrax-cudaNN` version equals the Phydrax version; elsewhere no companion resolves.
5. Writes JSON evidence with tool versions and the index snapshot time.

### 3.7 Tests (Phase 1)

| File | Contract |
|---|---|
| **New** `tests/_support/elf.py` | Test-owned minimal ELF64 shared-object writer (`write_shared_object(path, needed, rpath=(), runpath=())`). It encodes the external ELF format, so loader-resolution and provider-refusal tests need no compiler and do not mirror the package reader. |
| **New** `tests/unit/test_cuda_environment.py` | Pure assessment through the public `assess_cuda_environment`, one test per rule in §3.2: CPU install (successful, no issues); each issue code; unsuffixed 12.x-versioned CUDA 13 families not misclassified; driver 12.8 with plugin 13 names `phydrax[cuda12]`; driver 13.2 with plugin 13 passes; `cpu-platform-selected` is a warning only; static facts (`device_probed=False`) never yield `backend-unavailable`. Soname attribution: `libcusolver.so.11` vs `.so.12`; ambiguous names by owner vs path vs `unknown`. `platform_version` parsing: `"cuda 13040"`, `"cuda 12090"`, malformed → `ValueError`. Real hardware (skipped without CUDA): the runtime major equals the plugin major, and `cuda_environment()` succeeds in the qualified venvs. |
| **New** `tests/unit/test_elf_dependencies.py` | Real fixture: numpy's `_multiarray_umath` `DT_RPATH` `$ORIGIN/../../numpy.libs` resolves its OpenBLAS inside `numpy.libs`, and an `LD_LIBRARY_PATH` decoy does not win over that RPATH. Constructed fixtures (`tests/_support/elf.py`): `LD_LIBRARY_PATH` beats `DT_RUNPATH`; `$ORIGIN` expansion; transitive closure order; the closure is identical before and after the process loads a same-named decoy; an unresolved soname is reported; `$LIB` gives unresolved; a non-ELF file → `ValueError`. CUDA-marked: JAX's NVIDIA wheel libraries classify to the plugin major. |
| **New** `tests/integration/test_cuda_import_contract.py` | Subprocess: after importing `phydrax`, `phydrax.backends`, and `phydrax.linalg`, `/proc/self/maps` contains no `libcudss`, `libamgxsh`, `libquda`, or `libphydrax_cuda_*`. After `import phydrax` plus `collect_cuda_environment(initialize_backend=False)`, `phydrax.config.snapshot().backends_initialized` is false. |
| `tests/unit/backends/test_backend_contracts.py` | A provider whose constructed extension needs `libcublas.so.12` is refused under the CUDA 13 plugin before import, even when called without `device_platform`: its module is not in `sys.modules`, and `environment_issues[0].code == "provider-major-mismatch"`. Order independence: importing or probing AmgX/PyQUDA first does not change the cuDSS result. |
| `tests/integration/test_cuda_direct_mpi.py` | The new refusal joins the exit-77 skip set. A dedicated case asserts the refusal when a mismatched bridge is installed (skipped otherwise). |
| **New** `tests/typing/cases/cuda_environment.py` | `assert_type(cuda_environment(), CudaEnvironmentReport)` and `assert_type(assess_cuda_environment(facts), CudaEnvironmentReport)`. A deliberate misuse with a narrow ignore is paired with a runtime refusal. |
| `tests/unit/linalg/test_cudss_provider.py:35-145` | Unchanged numerics; skip reasons come from the static assessment. |

Expected behavior change: in the current mixed main venv, AmgX, PyQUDA, and mpi4jax are refused before import, so their real tests skip there. They are qualified in a scratch all-CUDA-13 venv until Phases 2b/2c replace them.

### 3.8 Documentation (Phase 1)

| File | Change |
|---|---|
| `README.md:626-646`, `docs/index.md:134-146` | GPU install is one command per major, with driver requirements. Remove "Install your preferred JAX distribution" for GPU users. Link the new installation page. |
| **New** `docs/installation.md`, `mkdocs.yml:93` (after Getting started) | Canonical page: CPU vs `cuda12`/`cuda13`; driver table; `cuda_environment()` usage and every issue code with its remedy; switching between majors (uninstall the JAX plugin and other-major NVIDIA wheels, or use a fresh environment); multi-GPU via JAX/NCCL with automatic `srun`/`mpirun` coordination; why GPU-direct MPI needs a site MPI; `LD_LIBRARY_PATH` shadowing. |
| `docs/api/backends.md:60-80` | MPI: interop extra, site MPI for GPU-direct, verified CUDA 13 mpi4jax recipe, pre-import major refusal. |
| `docs/api/backends.md:253-280` | Spineax installed via `phydrax[cuda13]` (replacing the `cudss` extra at `:255-256`); CUDA 12 limitation until Phase 2a; pre-import refusal. |
| `docs/api/backends.md` (provider sections) | `BackendAvailability.environment_issues`, `cuda_environment`, `CudaEnvironmentReport`, `CudaEnvironmentIssue`, issue codes. |
| `docs/api/execution.md:8-21, 183-186` | Multi-GPU and multi-node via startup coordination and `JaxCollectiveProvider`; MPI only as launcher or interop. |
| `CHANGELOG.md` | Added: `cuda12`/`cuda13` extras, `cuda_environment()`, `assess_cuda_environment()` and their types, `BackendAvailability.environment_issues`, pre-import CUDA-major refusal. Removed: the `cudss` extra (breaking). |
| `docs/data/public_api.json` | Regenerate after the last public change. |

## 4. Phase 2 — lockstep companion wheels

### 4.1 Layout

| Path | Content |
|---|---|
| **New** `native/cuda/CMakeLists.txt` | One source tree for both majors. Required cache variable `PHX_CUDA_MAJOR` (12 or 13, validated). `PHX_CUDA_ARCHITECTURES` (release list, or `native` for development). `PHX_CUDA_COMPONENTS` (`cudss;amgx;quda`; a release requires every component delivered in its phase). Version read from the companion `pyproject.toml` (meshcore pattern). SHA256 build hash over sources and patches. |
| **New** `native/cuda/cmake/PhxPipCudaToolkit.cmake` | Locates the build environment's NVIDIA wheels via `importlib.metadata` and assembles a toolkit view inside the build tree: `bin/nvcc`; `include` from header-bearing wheels; `lib` with versioned plus unversioned dev links; `lib/stubs/libcuda.so` pointing at the driver stub. Sets `CUDAToolkit_ROOT` / `CMAKE_CUDA_COMPILER` and asserts the `nvcc` major equals `PHX_CUDA_MAJOR`. Never writes into site-packages. RUNPATH entries are computed relative to `$ORIGIN` from the installed wheel layout (RECORD), not hard-coded. |
| **New** `native/cuda/cmake/PhxThirdParty.cmake` | `FetchContent` of pinned archives by commit with SHA256: QUDA develop `ba501e4` (or the newest qualified commit), AmgX main `91a8413`. Applies `native/cuda/patches/*.patch`. Static libraries with hidden visibility, `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`. **2b/2c gate:** the exact packaged configuration (static, hidden visibility, `QUDA_MPI=OFF`, release architectures) must build and qualify before release; this session's evidence used shared libraries and `QUDA_MPI=ON`. **Eigen gate (2c):** determine whether QUDA compiles Eigen code into `libquda` with `QUDA_USE_EIGEN=OFF`. If it does, ship the MPL-2.0 notice and source location; otherwise prevent the download. |
| **New** `native/cuda/patches/amgx-classical-amg-cuda13.patch` | PR #386 content. The header records the upstream URL, PR state, and removal condition (merged and pin advanced). |
| **New** `native/cuda/src/common/` | Identity (`phx_cuda_version`, `phx_cuda_build_hash`, `phx_cuda_major`, `phx_cuda_component`); status enumeration; thread-local last-error text; stream helpers; the per-device memo registry template (decision 12: mutex; partitions keyed by plan identity, each an LRU with its declared capacity; fresh-id minting that skips live ids; rekey; rebuild counter; live/capacity/rebuild totals per partition; heap-allocated and never destroyed at static destruction); symbol version script. |
| **New** `native/cuda/src/cudss/` | Vendored Spineax `solver.cpp` / `csr_transpose.cpp` (MIT header retained) on the common registry. Capacity and partition key are handler attributes; rebuild counts are outputs; `XLA_FFI_DEFINE_HANDLER_SYMBOL` handlers exported `extern "C"`; target names `phydrax_cudss_*`; per-call `device_id` validation kept (`solver.cpp:54-66`). Phases: analyze, factorize, refactorize (rekey), solve (multi-RHS, iterative refinement), inertia, release. Batched values form one block-diagonal entry under the §4.2 batching rules. dtypes f32 and f64. |
| **New** `native/cuda/src/amgx/` | Handlers over the AmgX C API on the common registry. **Setup:** CSR plus a JSON config attribute; uploads pass device pointers directly (`cudaMemcpy` accepts them). AmgX copies with blocking `cudaMemcpy` on the legacy default stream, so each handler synchronizes the XLA stream before and after AmgX work; documented as a synchronization point. **Refresh:** with an unchanged pattern and a resident previous id, `AMGX_matrix_replace_coefficients` plus `AMGX_solver_resetup` on that entry, rekeyed to the new id. Otherwise a full setup under the new id. A superseded preparation stays valid and rebuilds a full setup from its own operands on its next solve (counted). **Solve:** multi-RHS device vectors, per-RHS status, iterations, final residual, rebuild count. **Release:** idempotent. Mode dDDI only, with float32 promoted as today (`docs/api/backends.md:551`). `AMGX_initialize`/`finalize` reference counting moves here from `amg.py:536-576`. |
| **New** `native/cuda/src/quda/` | Handlers over QUDA's C API, built with `QUDA_MPI=OFF`, Wilson + clover, double/single precision, reconstruct 18. `initQuda(device)` is lazy, under a process mutex. Each call loads its gauge links from device memory (QUDA keeps one resident gauge field; no registry, no stale state). Operations: plaquette/gauge-loop trace action, gauge force, and Wilson dslash (single or multi-source, with dagger). One device per process; another device is refused with a reason. `endQuda` runs from the owned exit handler. |
| **New** `native/cuda/include/phydrax_cuda.h` | C ABI identity functions, status codes, and registry introspection (`live`, `capacity`, `rebuilds`). |
| **New** `native/cuda12/pyproject.toml`, `native/cuda13/pyproject.toml` | scikit-build-core; names `phydrax-cuda12` / `phydrax-cuda13`; version equal to Phydrax's; `cmake.source-dir = "../cuda"`; `cmake.define.PHX_CUDA_MAJOR`; `wheel.py-api = "py3"`; `wheel.packages = ["python/phydrax_cudaNN"]`; **no sdist**. License files: Phydrax `LICENSE`/`NOTICE`, `LICENSES/SPINEAX-MIT.txt` (vendored cuDSS core), `LICENSES/QUDA-MIT.txt` (QUDA's full `LICENSE`, including its bundled NVIDIA BSD text), `LICENSES/AMGX-BSD-3-CLAUSE.txt`, and `LICENSES/AMGX-NOTICES.txt`. **Build requires:** scikit-build-core, cmake, ninja, `jaxlib` in the supported range (headers via `jax.ffi.include_dir()`), and the major's toolkit wheels. CUDA 13: `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-cublas`, `nvidia-cusparse`, `nvidia-cusolver`, `nvidia-curand`, `nvidia-nvtx`, `nvidia-nvml-dev`, `nvidia-cuda-profiler-api`, `nvidia-cudss-cu13`. CUDA 12: the `-cu12` counterparts, all verified on PyPI. **Runtime requires:** `phydrax-cuda-major==NN.*` plus the cudss, cublas, cusparse, cusolver, curand, cuda-runtime, and nvjitlink families, each bounded below by the build and above at the next major. cuDSS is pinned to the built minor (e.g. `>=0.8,<0.9`), because `libcudss.so.0` keeps its soname across 0.x. |
| **New** `native/cuda12/python/phydrax_cuda12/__init__.py`, `native/cuda13/python/phydrax_cuda13/__init__.py` | `component_library(name)`, mirroring `phydrax_meshcore.library_path`. |

### 4.2 FFI call contract

| Target | Operands (rebuild data) | `has_side_effect` | Batching |
|---|---|---|---|
| `phydrax_cudss_analyze_{f32,f64}` | pattern, values | `True` (mints an id) | cuDSS rule (below) |
| `phydrax_cudss_factorize_*`, `phydrax_cudss_refactorize_*` | id, pattern, values | `True` (mints and rekeys) | cuDSS rule (below) |
| `phydrax_cudss_solve_*`, `phydrax_cudss_inertia_*` | id, pattern, values (and rhs) | `False` | cuDSS rule (below) |
| `phydrax_cudss_release` | ids | `True` | `vmap_method="broadcast_all"`; the handler deduplicates ids |
| `phydrax_amgx_setup`, `phydrax_amgx_refresh` | (previous id), pattern, values; config attribute | `True` | `vmap_method="sequential"` |
| `phydrax_amgx_solve` | id, pattern, values, rhs | `False` | `vmap_method="sequential"` |
| `phydrax_amgx_release` | ids | `True` | `vmap_method="sequential"` |
| `phydrax_quda_{gauge_action,gauge_force,wilson_dslash}_{c64,c128}` | links (and sources) | `False` | `vmap_method="sequential"` |

- Id minting and release carry an effect, so neither is merged or removed.
- **cuDSS batching rule.** cuDSS calls are wrapped in public `jax.custom_batching.custom_vmap` rules ported from Spineax (`solver.py:324-391`), not `vmap_method`:
  - batched values with an unbatched pattern form one block-diagonal entry, and its id is broadcast to the batch; equal id copies name that single entry;
  - batched right-hand sides with an unbatched preparation are absorbed as a multi-RHS solve;
  - an unbatched preparation with batched values is refused at trace time with a reason;
  - nested `vmap` composes by flattening into the same block-diagonal batch.
- Reads are pure functions of their operands. Because of decision 12 and the batching rules above, every row is correct under any order, elimination, duplication, batching, or rematerialization.
- Rebuild counts report native work actually performed. If the compiler merges two identical reads, only one rebuild is counted.
- `phydrax.linalg.release` and AmgX release work eagerly (ctypes) and inside traces (the release targets).

### 4.3 Python loader: new `phydrax/_cuda_native.py`

- **Library resolution:** `library_input("cuda_cudss" | "cuda_amgx" | "cuda_quda")` override, then `phydrax_cudaNN.component_library(name)`, where NN is JAX's plugin major from static facts. The JAX runtime major is confirmed at first call. Configuration changes:
  - `LibrarySetting` (`phydrax/_config_types.py:30`) gains `"cuda_cudss"`, `"cuda_amgx"`, `"cuda_quda"`;
  - `LIBRARY_ENVIRONMENT` (`phydrax/_config_environment.py:39-40`) gains `PHYDRAX_CUDA_CUDSS_LIBRARY`, `PHYDRAX_CUDA_AMGX_LIBRARY`, `PHYDRAX_CUDA_QUDA_LIBRARY`.
- **Dependency preloading** (decision 7): the component's closure from `phydrax/_elf_dependencies.py`, restricted to NVIDIA libraries and resolved within the matching NVIDIA package directories (Python package `__path__`), is loaded dependency-first by absolute path. A missing distribution gives `native-dependency-missing`, naming it. Resolved paths are recorded in evidence.
- **Per-component lazy binding.** The library must export the identity symbols and every declared handler. Its version must equal `importlib.metadata.version("phydrax")`, and its major must equal JAX's plugin major. Otherwise the loader raises `CudaNativeUnavailableError(ImportError)` with the reason, before any registration. Availability probes convert this into `BackendAvailability.environment_issues`. Phase 2 adds `"companion-missing"`, `"companion-version-mismatch"`, `"companion-major-mismatch"`, and `"native-dependency-missing"` to the closed `CudaIssueCode`.
- **Registration** is once per process per component, idempotent, through `jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="CUDA")`.
- **Module split** (decision 10): `phydrax/linalg/backends/_cudss.py` holds only static state types and validation and stays importable eagerly. The FFI layer `phydrax/linalg/backends/_cudss_ffi.py` is imported on the first prepare. AmgX and QUDA follow the same split.
- **QUDA tune cache:** when `QUDA_RESOURCE_PATH` is unset, the QUDA component sets it natively immediately before the first `initQuda`, to `$XDG_CACHE_HOME/phydrax/quda/<build hash>` (default `~/.cache`), creating the directory. A user value is never overridden, and the path used is reported in evidence.
- **Exit handler** (decision 13) is registered at the first component load.
- **New public** `phydrax.backends.cuda_native_resources() -> CudaNativeResources`: per component, device, and plan partition, the live entries, capacity, and total rebuilds. It is a host call, used for memory monitoring and leak qualification.

### 4.4 Provider cutovers

| Area | Files | Change |
|---|---|---|
| cuDSS | **New** `phydrax/linalg/backends/_cudss.py` and `_cudss_ffi.py` replace `phydrax/linalg/backends/_spineax.py` | Public `jax.ffi.ffi_call` per §4.2. `Spineax*` state/output/function names become `Cudss*`. Keep the storage validation (`_spineax.py:53-91`) and status semantics. The backend output gains `native_rebuilds`, surfaced in solve provenance (`phydrax/linalg/_results.py`). A failed native status maps to explicit failure evidence. |
| cuDSS | `phydrax/linalg/backends/_provider.py:40-46,59-62,303-363`, `phydrax/linalg/_sparse_providers.py:87-101,194-222`, `_policies.py:159-180`, `_plans.py:108,460-486,993-1003,1813,1847`, `_inertia.py:147-171`, `_runtime.py:100,832,1006,1959`, `optim/_interior_point.py:363-365`, `optim/_structured_ipm.py:933-1000`, `backends/__init__.py:149-153,225,244` | Provider literal `"spineax-cudss"` → `"cudss"` everywhere, with no alias. `SpineaxBackend`/`SPINEAX_CAPABILITIES`/`spineax_availability` → `CudssBackend`/`CUDSS_CAPABILITIES`/`cudss_availability`. `SparseLDLT` gains `factor_cache_capacity` (default 8, positive, part of the policy identity). Resource admission (`_plans.py:1813, 1847`) accounts for capacity × per-factor bytes in the plan's own partition. Plan and policy identities change (documented). |
| AmgX | `phydrax/backends/amg.py:53-70,159-168,241-341,454-972` | Replace the pyamgx path with FFI calls. `AmgXPolicy` gains `hierarchy_cache_capacity` (default 2). `refresh_amgx` returns an independent preparation with a new id (in-place reuse for an unchanged pattern per §4.1); the old preparation stays valid, rebuilding on use. `solve_amgx` becomes device-resident and jittable (`transformations=("jit",)`); there is no autodiff claim. Per-RHS status, iterations, and rebuild count reach the result. The Phydrax-side residual certification (`:856-943`) is kept. |
| AmgX | `benchmarks/advanced_solvers/adapters/amgx.py:53,70,343` | Dependency and version provenance switch from `pyamgx` to the companion. |
| QUDA | **New** `phydrax/backends/_lattice_quda.py` replaces `phydrax/backends/_lattice_pyquda.py` | Device-resident FFI calls. Gauge layout conversion in JAX (explicit reshape/transpose to QUDA order). Clover applied through the native JAX clover owner (CuPy before). Wilson r=1 and even-extent constraints kept. Jittable without a host callback; no autodiff claim. Multi-process (`jax.process_count() > 1`) is refused with a reason. |
| QUDA | `phydrax/backends/lattice.py:400-546` | `_QUDA_BACKEND` distributions → companion. `LatticeProviderName` `"pyquda"` → `"quda"`; `pyquda_availability` → `quda_availability`; `create_lattice_provider` dispatch updated. No aliases. |
| Configuration | `phydrax/_config_types.py:30`, `phydrax/_config_environment.py:39-40` | New `LibrarySetting` values and declared variables (§4.3). |
| Packaging | `pyproject.toml` | Both extras keep their unconditional marker (decision 2). `cuda12` adds `phydrax-cuda12==<version>; sys_platform == 'linux' and platform_machine == 'x86_64'`; `cuda13` replaces `spineax` with `phydrax-cuda13==<version>` under the same marker. The marker change lands in the same release as the first published companions. `[tool.uv.sources]` gains the local companion paths per extra, plus `extra-build-variables` setting `SKBUILD_CMAKE_DEFINE=PHX_CUDA_ARCHITECTURES=native` for developer builds. `[tool.ty.src] include` (`:356`) gains `native/cuda12/python` and `native/cuda13/python`. Remove `cupy.**`, `pyquda.**`, `pyquda_utils.**` from `replace-imports-with-any` (`:376, 393-394`). Markers (`:422`) gain `cuda_native`. |
| Removed | `phydrax/linalg/backends/_spineax.py`, `phydrax/backends/spineax.py`, `phydrax/backends/_lattice_pyquda.py` | Deleted in the same change with their imports, exports, tests, and docs. A case-insensitive substring search (`rg -i 'spineax\|pyquda\|pyamgx'`) must return nothing outside `CHANGELOG.md`, `docs/plans/`, `NOTICE`, `LICENSES/`, and vendored source headers; regenerated `docs/data/public_api.json` included. |

### 4.5 Wheel build, repair, size, architectures

- **New** `tools/build_cuda_wheels.py`:
  - Builds `phydrax-cuda12` / `phydrax-cuda13` in a `manylinux_2_28` container (podman or docker), or on the host for development.
  - Runs `auditwheel repair` with explicit excludes for NVIDIA wheel libraries and driver libraries (`libcuda.so.1`, `libnvidia-ml.so.1`).
  - Verifies with `readelf` that every RUNPATH is relative to `$ORIGIN`, that no NEEDED entry resolves outside the wheel or the NVIDIA wheel set, and that no static cudart was linked.
  - Lists SASS/PTX per component (`cuobjdump --list-elf --list-ptx`).
  - Records compressed sizes and writes a JSON manifest: tool versions, upstream pins, patch hashes, architectures, sizes, build hashes.
- **Architectures:**
  - CUDA 13: SASS for 8.0, 8.6, 8.9, 9.0, 10.0, 12.0, plus PTX for the newest target.
  - CUDA 12: additionally 7.0 and 7.5.
  - AmgX may use a reduced SASS set plus PTX if the size gate requires it. First-use PTX JIT time and driver JIT-cache fit are measured before choosing.
- **Size gate:**
  - The cuDSS-only release (2a) must fit under 100 MB.
  - Immediately after 2a, request both the per-file and the per-project PyPI limits, sized from the measured 2b/2c manifest times the retained releases.
  - If the grant is insufficient, AmgX moves to `phydrax-cudaNN-amgx`, still pulled by the same extra. This is the only permitted split.
- **aarch64** (GH200, GB10) is built only once a native aarch64 GPU runner exists. Until then the companion marker excludes it; aarch64 users get JAX, the exclusivity marker, and an explicit companion-unavailable reason.

### 4.6 Release qualification

**New** `tools/qualify_cuda_wheels.py`, run per major in a fresh venv created from the wheelhouse plus PyPI:

1. Installs `phydrax[cudaNN]` with no other configuration.
2. Runs the provider qualification tests: `pytest -m cuda_native` plus the cuDSS, AmgX, lattice, and backend-contract files.
3. Runs the in-process mapped-library audit: every CUDA library attributed to major NN; nothing from `/usr/local/cuda*`, system CUDA packages, or the other major's wheels. Records the preloaded absolute paths.
4. Repeats the audit and a cuDSS/AmgX/QUDA smoke under a hostile `LD_LIBRARY_PATH` pointing at a system CUDA 12 toolkit.
   - **CUDA 13:** success with no CUDA 12 library mapped.
   - **CUDA 12:** if JAX's plugin fails, `cuda_environment()` reports `backend-unavailable` with the directory, and companion loading is refused with that reason. If JAX succeeds, the companion audit must be single-major.
5. Runs `tools/check_cuda_extras.py`.
6. Writes JSON evidence: GPU model and compute capability, driver, JAX version, wheel hashes.

**Publishing order** (documented in **new** `native/cuda/README.md`): `phydrax-cuda-major` (once) → `phydrax-cuda12` and `phydrax-cuda13` → `phydrax`. Never publish Phydrax before its exact companions exist.

**Optional, decision required:** **new** `.github/workflows/cuda-wheels.yml` (CPU build job plus a self-hosted GPU qualification job). The repository has no CI; the tools above run without it.

### 4.7 Tests (Phase 2)

| File | Contract |
|---|---|
| **New** `tests/unit/test_cuda_native_loader.py` | Each refusal names its fix: missing companion (names `phydrax[cudaNN]`); `ProviderConfig` override to a missing file; stale version after a Phydrax upgrade (names the reinstall); major mismatch (raised before registration); missing handler symbol; missing NVIDIA dependency distribution. Registration is idempotent. One component failing does not block another. A CPU-only process with the companion installed reports unavailable with a no-device reason. The real library is marked `cuda_native`. |
| **New** `tests/integration/test_cuda_native_lifecycle.py` | Subprocess on a real GPU: exit with live cuDSS/AmgX/QUDA state and an in-flight solve returns 0 with no CUDA error output. CUDA 13 preloaded paths survive a hostile `LD_LIBRARY_PATH`. |
| `tests/unit/linalg/test_cudss_provider.py` | Provider `"cudss"`. All existing numeric, implicit JVP/VJP, batch, inertia, and release cases kept (`:41-145, :235-266`), including refresh inside `filter_jit(jvp)` (`:53-69`). Added: repeated traced refreshes keep the plan partition's live count within `factor_cache_capacity`. With capacity 1, alternating two preparations gives exact results with positive `native_rebuilds`. Two plans with capacities 1 and 8 never evict each other's entries. A superseded factor still solves its own system. Release inside a jitted function lowers the live count. Release is idempotent. Batching: batched values with an unbatched pattern; batched right-hand sides with an unbatched preparation (multi-RHS); an unbatched preparation with batched values is refused at trace time; nested `vmap`. Singular/indefinite inertia evidence; float32/float64; arrays on a non-default GPU when two are present (skipped otherwise). Rebuilt solves match resident solves bitwise; otherwise the difference is reported and documented. Runs on both majors in qualification. |
| `tests/unit/linalg/test_assembly_contracts.py:521-522`, `tests/unit/optim/test_structured_optimization_spine.py:292-306`, `tests/unit/test_advanced_solver_benchmark_harness.py:246` | Import and provider-name updates. |
| `tests/unit/backends/test_amg_backends.py` | Remove the fake-pyamgx wiring tests (`:242-315`). Real FFI: parity versus a dense solve; multi-RHS; jit versus eager parity. A refreshed preparation equals a fresh setup, and the old preparation still solves with its old coefficients and reports one rebuild. Nonconvergence (tiny `max_iters`) gives a failed per-RHS status, never a silent success. float32 inputs promoted with the promotion recorded. Classical AMG on CUDA 13 (PR #386 regression). The live count stays within `hierarchy_cache_capacity`. Arrays on a non-default GPU (skipped with one GPU). Release idempotence. |
| `tests/unit/backends/test_lattice_providers.py`, `test_backend_contracts.py:14-53` | Provider `"quda"`: gauge action, force, and Wilson/clover dslash parity versus the native JAX provider (complex64/128, single and multi-source, dagger). jit vs eager parity; alternating two gauge fields gives exact results; multi-process refusal; second-device refusal; `QUDA_RESOURCE_PATH` default created and a user value respected; odd extent and r≠1 refusals kept. |
| `tests/unit/test_configuration_environment.py:328-352`, `tests/unit/test_configuration_records.py:20-29` | Library-setting expectations include the new settings and their variables. |
| `tests/unit/test_cuda_environment.py` | Phase 2 issue codes. |

### 4.8 Documentation, licenses, generated data (Phase 2)

- `docs/installation.md`: the extras include the companions; component table per major; architecture coverage; first-use JIT note if AmgX ships PTX for some architectures.
- `docs/api/backends.md`:
  - The Spineax section (`:253-280`) becomes "cuDSS (native)", covering cache capacity, rebuild evidence, and `cuda_native_resources()`.
  - The AmgX section (`:542-600`): no pyamgx; jit; independent refresh preparations; capacity; status evidence.
  - The lattice section (`:113-155`): QUDA provider without PyQUDA/CuPy, jittable, single-process.
- `docs/api/configuration.md` (`ProviderConfig`, `:63`): the new library settings and variables.
- `docs/api/linalg.md:676,1561-1564,2575,2774-2776`, `docs/api/optim.md:1069-1071`, `docs/all-of-phydrax.md:1437`: `SparseLDLT(provider="cudss", factor_cache_capacity=...)` and cuDSS naming.
- `NOTICE`:
  - update the Spineax entry (`:290-294`) for the vendored, adapted sources;
  - add QUDA and AmgX entries for the redistributed static builds, including AmgX's bundled components (from its `NOTICES`), QUDA's bundled NVIDIA BSD components, and Eigen if the 2c gate finds it compiled in.
- `LICENSES/`: **new** `QUDA-MIT.txt` (QUDA's full `LICENSE`, including its bundled NVIDIA BSD text), `AMGX-BSD-3-CLAUSE.txt`, `AMGX-NOTICES.txt`, and `EIGEN-MPL-2.0.txt` if the Eigen gate requires it, following the repository's `NAME-LICENSE.txt` convention. The companion wheels include `LICENSE`, `NOTICE`, the existing `SPINEAX-MIT.txt`, and these files.
- `CHANGELOG.md`:
  - **Breaking:** provider names `spineax-cudss` → `cudss` and `pyquda` → `quda`; the renamed backend classes and availability functions; the new `SparseLDLT`/`AmgXPolicy` fields change plan and policy identities.
  - **Added:** CUDA 12 cuDSS; jittable AmgX and QUDA; AmgX in-place refresh for unchanged patterns; declared native cache capacities; rebuild evidence; `cuda_native_resources()`.
- Regenerate `docs/data/public_api.json` and the capability inventory (`tools/generate_capability_inventory.py`, `tools/check_capability_consistency.py`). The runtime-contract TSVs are dated audit baselines and are not hand-edited.

## 5. Explicit non-goals and deferred items

- **GPU-direct MPI automation.** Requires site MPI ABI and fabric. It stays opt-in through `phydrax[mpi]`, guarded by the pre-load CUDA-major refusal. Revisit when mpi4jax gains standard-ABI wheels or CUDA 13 wheel-layout detection.
- **Multi-process QUDA.** Site builds only; the packaged provider refuses it with a reason.
- **Automatic CUDA-major selection without an extra.** Blocked on wheel variants (PEP 817/825) reaching installers.
- **Device-inert subpackage imports.** `phydrax/_doc.py:9` initializes the JAX backends when subpackages are imported. That is pre-existing, upstream-owned behavior; this plan neither depends on nor changes it.
- **CPU-only providers.** Grid/GPT and qlat packaging.
- **conda-forge distribution of Phydrax.** Could be revisited if conda-forge ships JAX 0.11+. Not needed for the pip/uv contract.

## 6. Verification and acceptance

**Per slice (focused):**
- ruff and `ty` on changed files;
- `tools/check_import_boundaries.py`, `tools/audit_selectors.py`, `tools/audit_contract_candidates.py --signatures`;
- the exact owning tests above;
- a direct smoke run of the changed path.

**Phase 1 acceptance:**
- `tools/check_cuda_extras.py` passes against the built wheel on every listed `--python-platform`.
- `cuda_environment()` reports success in both scratch venvs (CUDA 12 with `LD_LIBRARY_PATH` unset; CUDA 13 as is).
- In the CUDA 12 venv with `LD_LIBRARY_PATH=/usr/local/cuda/lib64`, the report gives `backend-unavailable` with that directory.
- In the current mixed main venv:
  - AmgX, PyQUDA, and mpi4jax are refused before import with `provider-major-mismatch`, naming the CUDA 12 libraries.
  - Spineax/cuDSS stays available regardless of probe order.
- In the all-CUDA-13 venv, every provider is available and the CUDA-direct MPI test passes.
- The import-contract subprocess test passes.

**Phase 2 acceptance** (per major, from published-equivalent wheels in clean venvs):
- `tools/qualify_cuda_wheels.py` passes, including the hostile-`LD_LIBRARY_PATH` audit.
- Provider tests pass on CUDA 12 and CUDA 13, and cuDSS is available on CUDA 12.
- Registry live counts never exceed declared capacities, including after repeated traced refreshes.
- The `auditwheel` manifest shows no bundled NVIDIA or driver libraries and no static cudart.
- Sizes are within the granted limit.

**Integration wave:**
- the typing gate (`tools/check_typing.py check`) with zero first-party diagnostics in changed files;
- typing fixtures (`-m typecheck`);
- `tests/unit/test_contract_declarations.py`, `test_role_declarations.py`, `typing/test_strict_integration.py`;
- the configuration and startup suites;
- the public API manifest and capability consistency checks.

**Resource rules:** CPU tests with at most 5 workers and GPU tests with at most 2 workers under the 54 GB watchdog. Never run plain `uv sync` in this environment.

## 7. Risks and gates

| Risk | Gate or mitigation |
|---|---|
| AmgX CUDA 13 support depends on an unmerged PR | Patch pinned with a removal condition. The classical-AMG regression test on CUDA 13 blocks release. |
| QUDA develop churn | Pin by commit. Qualification parity versus the native JAX provider. Advance only with green qualification. |
| Wheel size and PyPI limits | cuDSS-first release, then a measured limit request. AmgX split is the only fallback. |
| Multi-architecture build time | Release builds only; developers use `native` architectures through `extra-build-variables`. |
| Cache thrash when a plan's live preparations exceed its partition capacity | Exact rebuilds keep results correct. Rebuild counts in result evidence and `cuda_native_resources()` make the cost visible. Capacity is a documented policy field. |
| Rebuilt factors differ from resident ones | Bitwise comparison in qualification. Any difference is reported and documented, never hidden. |
| ELF resolver diverges from `ld.so` | Unmodeled features (`$LIB`, `$PLATFORM`, `glibc-hwcaps`) are reported unresolved and refused, never guessed. Real-library fixtures plus constructed fixtures cover the modeled order. |
| QUDA and AmgX global state under concurrent FFI calls | Process mutex per component; single device per process for QUDA, with explicit refusal otherwise; stream synchronization at handler boundaries, documented. |
| JAX FFI drift | Only the public `jax.ffi` API and the stable XLA FFI C ABI are used. Phydrax and companions pin `jax`/`jaxlib` `<0.12`. Raising the bound requires rebuilding and requalifying the companions. |
| `LD_LIBRARY_PATH` or system CUDA shadows shared sonames | Transitive absolute-path preloading (decision 7); hostile-path qualification. |
| Switching extras leaves other-major wheels behind | `foreign-nvidia-distributions` error with the exact uninstall list; documented switching procedure. |
| Exit-time crashes from native teardown order | Owned exit handler (decision 13); lifecycle subprocess test. |
| Bring-your-own providers installed alongside the companion | Different FFI target names and hidden symbols. Pre-import closure checks refuse other-major providers. |
| Driver too old for the chosen extra | `driver-too-old` names the matching extra. Pip cannot fix it; documented. |

## 8. Phase 1 implementation record

### Delivered files

| Area | Files |
|---|---|
| Packaging | `pyproject.toml` (`cuda12`/`cuda13` extras, `cudss` removed, uv conflict, per-extra marker sources); `packaging/cuda-major/{12,13}/` (`pyproject.toml`, `LICENSE`, `NOTICE`); `packaging/cuda-major/README.md` |
| Runtime owners | `phydrax/_elf_dependencies.py`; `phydrax/_cuda_environment.py`; exports in `phydrax/backends/__init__.py` |
| Availability | `phydrax/backends/_types.py` (`environment_issues`); `phydrax/backends/_availability.py` (`probe_backend(cuda_libraries=...)`, `module_native_libraries`); `spineax.py`, `amg.py`, `lattice.py`, `_lattice_pyquda.py`, `distributed.py`; `phydrax/linalg/_sparse_providers.py`; issue forwarding in `_lattice_gpt.py`, `_lattice_qlat.py`, `iree.py`, `slepc.py` |
| Release gate | `tools/check_cuda_extras.py` |
| Tests | `tests/_support/elf.py`; `tests/unit/test_elf_dependencies.py`; `tests/unit/test_cuda_environment.py`; `tests/typing/cases/cuda_environment.py`; `tests/integration/test_cuda_import_contract.py`; additions in `tests/unit/backends/test_backend_contracts.py` and `tests/integration/test_cuda_direct_mpi.py` |
| Documentation | `docs/installation.md` (new, in `mkdocs.yml`), `README.md`, `docs/index.md`, `docs/api/backends.md`, `docs/api/execution.md`, `CHANGELOG.md`, `docs/data/public_api.json` |

### Refinements of §3 made during implementation

1. **JAX preload seeding.** A provider probed with `device_platform` passes the device check, which initializes JAX; the CUDA plugin then preloads its NVIDIA wheels by absolute path, and later loads bind those sonames. Its closure is therefore resolved with `preloaded` set to the plugin's preload set (`jax_plugin_preloads`, mirroring `_load_nvidia_libraries` of both JAX 0.11 plugins, which are identical). Without seeding, Spineax would be refused under `LD_LIBRARY_PATH=/usr/local/cuda/lib64` because the shared soname `libcusparse.so.12` resolves to CUDA 12.8 there, although the process binds JAX's CUDA 13 copy. Providers without a device check (mpi4jax) are resolved without seeding.
2. **Explicit loader inputs.** `resolve_dependency_closure(path, *, library_path, executable, preloaded=(), ld_so_cache=...)`: `executable` ends the DT_RPATH chain with the main program's RPATH, as glibc does.
3. **Facts additions.** `host_platform`, `host_machine`, `wheel_library_majors` (keeps owner attribution pure), and `backend_error` (text of a JAX backend initialization failure).
4. **Import-free module lookup.** `importlib.util.find_spec` on a dotted name imports the parent package (for example `spineax/__init__.py`), so CUDA-guarded probes locate modules through `PathFinder` without importing parents.
5. **PyQUDA ordering.** The CuPy distribution major is checked before any CuPy/PyQUDA import, but only when PyQUDA is installed, so an absent PyQUDA still reports `not installed`.
6. **Gate platforms.** `x86_64-apple-darwin` is excluded (§3.6).

### Evidence (2026-10-15, RTX 5090, driver 595.91.07 reporting CUDA 13.2)

| Scenario | Result |
|---|---|
| `tools/check_cuda_extras.py` against the built wheel | 12/12 cases pass (4 platforms × `[cuda12]`, `[cuda13]`, `[cuda12,cuda13]`) |
| Main mixed venv, `LD_LIBRARY_PATH=/usr/local/cuda/lib64` | `cuda_environment()` successful with one `library-path-shadowing` warning; mapped majors {13}. Spineax and `sparse_provider_availability("spineax-cudss")` available. AmgX (CUDA 12 cuBLAS/cuBLASLt/cuSOLVER/nvJitLink), PyQUDA (`cupy-cuda12x`), and mpi4jax (`libcudart.so.12`) refused before import with `provider-major-mismatch`. Identical results with the reversed probe order and with `LD_LIBRARY_PATH` unset (then without the warning). |
| CUDA 12 venv, `LD_LIBRARY_PATH` unset | Successful, no issues, mapped majors {12}; CUDA 12 AmgX, PyQUDA, and mpi4jax available; Spineax not installed |
| CUDA 12 venv, `LD_LIBRARY_PATH=/usr/local/cuda/lib64` | `backend-unavailable` naming `/usr/local/cuda/lib64` |
| CUDA 13 venv, `LD_LIBRARY_PATH` unset / hostile | Successful (hostile: `library-path-shadowing` warning only); Spineax, AmgX, PyQUDA, mpi4jax available; mapped majors {13} in both |
| CUDA 13 venv, provider tests | AmgX 6 passed; cuDSS 17 passed; backend contracts 7 passed; environment 50 passed; import contract 2 passed; lattice 10 passed, 2 dtype skips; CUDA-direct and host-staged MPI 2 passed (site Open MPI launcher, root override) |
| Marker metadata | Installing marker 13 into the CUDA 13 venv: requested 13, successful; marker 12: `plugin-major-mismatch` |
| Main venv affected tests | CPU 207 passed, 24 skipped; GPU 109 passed, 12 skipped |

## 9. Phase 2 implementation record

2a, 2b, and 2c were implemented together; Spineax, pyamgx, PyQUDA, and CuPy are removed in one cutover.

### Delivered files

| Area | Files |
|---|---|
| Native tree | `native/cuda/CMakeLists.txt`; `cmake/PhxPipCudaToolkit.cmake`, `PhxCudaComponent.cmake`, `PhxAmgX.cmake`, `PhxQuda.cmake`; `include/phydrax_cuda.h`; `src/common/` (C ABI, identity, partitioned memo registry, export map); `src/cudss/solver.cpp` (adapted Spineax core); `src/amgx/amgx_ffi.cpp`; `src/quda/quda.cu`; `patches/amgx-classical-amg-cuda13.patch`, `amgx-dddi-only.patch`, `amgx-dddi-v-cycle.patch`, `quda-device-gauge-force.patch`, `quda-cmake-subproject.patch`, `quda-portable-warp-election.patch`; `README.md` |
| Companions | `native/cuda12/`, `native/cuda13/` (`pyproject.toml`, sdist-refusing `_build_backend.py`, `python/phydrax_cudaNN/` with `component_library`) |
| Loader | `phydrax/_cuda_native.py`; `LibrarySetting` values and `PHYDRAX_CUDA_*_LIBRARY` variables (`_config_types.py`, `_config_environment.py`); Phase 2 `CudaIssueCode` members; `probe_cuda_component` (`backends/_availability.py`); `cuda_native_resources` exports |
| cuDSS | `phydrax/backends/cudss.py`, `phydrax/linalg/backends/_cudss.py`, `_cudss_ffi.py` (replace `backends/spineax.py`, `linalg/backends/_spineax.py`); provider literal `cudss` across linalg/optim; `SparseLDLT.factor_cache_capacity`; `native_rebuilds` provenance |
| AmgX | `phydrax/backends/amg.py` (FFI path, `AmgXPolicy.hierarchy_cache_capacity`, per-RHS native diagnostics, jittable solve); benchmark adapter |
| QUDA | `phydrax/backends/_lattice_quda.py` (replaces `_lattice_pyquda.py`); `phydrax/backends/lattice.py` (`quda`, `quda_availability`) |
| Release tools | `tools/build_cuda_wheels.py`; `tools/qualify_cuda_wheels.py`; companion checks in `tools/check_cuda_extras.py` (`--companion-wheelhouse`, `--linux-tag`) |
| Tests | `tests/unit/test_cuda_native_loader.py`; `tests/integration/test_cuda_native_lifecycle.py`; rewritten provider tests (`tests/unit/linalg/test_cudss_provider.py`, `tests/unit/backends/test_amg_backends.py`, `test_lattice_providers.py`, `test_backend_contracts.py`); configuration and environment tests |
| Packaging, licenses, docs | `pyproject.toml` (extras pin `phydrax-cudaNN==0.2.2`, uv sources and developer `extra-build-variables`, `cuda_native` marker, typing scope, `scikit-build-core` in `qa`); `LICENSES/QUDA-MIT.txt`, `AMGX-BSD-3-CLAUSE.txt`, `AMGX-NOTICES.txt`, `EIGEN-MPL-2.0.txt`; `NOTICE`; installation, backends, configuration, linalg, optim docs; `CHANGELOG.md`; regenerated `docs/data/public_api.json` |

### Refinements of §4 made during implementation

1. **cuDSS targets.** `phydrax_cudss_diagonal_{f32,f64}` (pivot diagonal in input order) replaces Spineax's 14-field query; inertia is computed from it. The CSR transpose handlers are not vendored (symmetric-only contract).
2. **Release is optional.** `CUDSS_CAPABILITIES` and `AMGX_CAPABILITIES` declare `requires_explicit_release=False`: native state is a capacity-bounded memo, and release only retires entries early; a released preparation rebuilds (counted).
3. **AmgX size.** `amgx-dddi-only.patch` and `amgx-dddi-v-cycle.patch` restrict compiled public modes to dDDI (internal hDDI and integer helpers kept; no solver algorithm removed). CUDA 13 sm_120: 57.4 MB versus 157.3 MB for the unrestricted shared build.
4. **QUDA.** Three patches: device-location gauge force, embeddable CMake roots, and portable warp election (pinned develop calls `cuda::ptx::elect_sync`, absent from CUDA 12 CCCL). QUDA compiles Eigen even with `QUDA_USE_EIGEN=OFF`, so pinned Eigen `e67c494c` is shipped under MPL-2.0. The QDP interface stays enabled because the QDP gauge order is the device pointer-array layout the handlers use.
5. **QUDA tune cache.** The default is `$XDG_CACHE_HOME/phydrax/quda/<build hash>/<QUDA variant>`, where the variant is QUDA's own tune-cache key (CPU architecture, GPU architectures, CUDA compiler version). Qualification found that CUDA 12 and CUDA 13 builds share the source build hash, and QUDA aborts on a tune cache written by another variant.
6. **CUDA 12 compiler.** PyPI's `nvidia-cuda-nvcc-cu12` ships no `nvcc`; the toolkit module uses the SHA256-pinned NVIDIA `cuda_nvcc` 12.9.86 redistributable. Driver stubs and `cuobjdump` come from pinned NVIDIA redistributables and are never shipped.
7. **Refusal under a shadowing path.** When no CUDA device exists, the loader raises the probed environment issue, so JAX's CUDA 12 plugin failing under `LD_LIBRARY_PATH=/usr/local/cuda/lib64` refuses with `backend-unavailable` naming the directory (§4.6 step 4). The lifecycle test and the qualification tool assert exactly this for CUDA 12, and single-major success for CUDA 13.

### Evidence (2026-10-15, RTX 5090 sm_120, driver 595.91.07 reporting CUDA 13.2)

| Scenario | Result |
|---|---|
| Development wheels (`--architectures native`) | `phydrax_cuda13` 20.4 MB, `phydrax_cuda12` 21.9 MB compressed; exports only `phx_cuda_*`/`phydrax_*`; shared `libcudart.so.1x`; `$ORIGIN` RUNPATHs; no bundled NVIDIA or driver libraries |
| Release-architecture `phydrax_cuda13` (SASS 8.0/8.6/8.9/9.0/10.0/12.0 + PTX 12.0) | 81.3 MB compressed (AmgX 211.5 MB → 39.3 MB, QUDA 58.0 MB → 41.8 MB, cuDSS 0.38 MB); built in 45 min at `-j10`; all requested SASS/PTX present |
| Release-architecture `phydrax_cuda12` (adds SASS 7.0/7.5) | **121.8 MB compressed**, above PyPI's default 100 MB per-file limit (AmgX 253.1 MB → 47.1 MB, QUDA 111.6 MB → 74.5 MB, cuDSS 0.38 MB); built in 52 min at `-j9`; all requested SASS/PTX present |
| `tools/check_cuda_extras.py` with companion wheelhouse | 12/12 cases pass; Linux x86_64 resolves `phydrax-cudaNN==0.2.2`; aarch64, macOS, Windows resolve no companion |
| `tools/qualify_cuda_wheels.py --major 13` (fresh venv, development wheel) | 103 passed, 11 skipped (two-GPU cases; Grid/qlat not installed); clean and hostile audits map 14 CUDA 13 wheel libraries only |
| `tools/qualify_cuda_wheels.py --major 13` (release-architecture wheel) | 103 passed, 11 skipped; clean and hostile audits CUDA 13 only |
| `tools/qualify_cuda_wheels.py --major 12` (fresh venv, development and release-architecture wheels) | 102 passed, 12 skipped each (adds the CUDA 13 classical-AMG regression skip); clean audit CUDA 12 only; hostile path refused with `backend-unavailable` naming `/usr/local/cuda/lib64` |
| Per-component qualification during development | cuDSS: rebuilt versus resident solves bitwise identical (f32, f64); capacity-1 partitions rebuild exactly; AmgX 16 passed (CUDA 13) / 15 passed (CUDA 12); QUDA 22 passed on each major |
| Main venv affected tests (development libraries via overrides) | CPU 1218 passed, 103 skipped; GPU 115 passed, 6 skipped |

### Remaining release steps (not executable in this environment)

- **manylinux_2_28 container builds.** `tools/build_cuda_wheels.py --container podman|docker` is implemented but unexercised: this host has neither runtime, so host builds carry `manylinux_2_39`.
- **Publication.** Publish `phydrax-cuda-major`, then both companions, then Phydrax (`native/cuda/README.md`); rerun `tools/check_cuda_extras.py` against the index, with `--linux-tag manylinux_2_28`.
- **Size limits (§4.5 gate).** The release CUDA 13 wheel (81.3 MB) fits PyPI's default 100 MB per-file limit; the release CUDA 12 wheel (121.8 MB) does not. Per §4.5, request the per-file increase (for example 150 MB) and a project increase sized from these manifests (about 203 MB per release pair, so the default 10 GB covers about 50 releases). If the grant is refused, the permitted split moves AmgX into `phydrax-cuda12-amgx`, leaving about 75 MB in `phydrax-cuda12`.
- **Multi-GPU.** Non-default-device and second-device cases are implemented and skip on this single-GPU host.
- **aarch64.** No companion until a native aarch64 GPU runner exists.
