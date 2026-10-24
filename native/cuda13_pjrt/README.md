# Windows CUDA 13 PJRT runtime

This distribution packages a native Windows CUDA compiler/runtime for **stock
`jax==0.12.0` and `jaxlib==0.12.0`**, standard CPython 3.13, AMD64. It does not
install winjax, modify installed JAX Python sources, replace the CPU JAXlib
wheel, or compile anything during end-user installation.

## Locked sources and patch policy

`sources.lock.json` binds downloaded source archives, local patches, licenses,
Bazel 9.2.0, the upstream module declarations, and NVIDIA's Windows CUDA 13.3
redistributables by SHA-256. Source archives were acquired to the worktree's
`.tmp/cuda13-pjrt-sources` directory while preparing this recipe. CUDA archive
hashes are NVIDIA's published redistributable metadata; downloading and checking
the actual CUDA archives happens in the build command. Bazel's executable hash
is from its release checksum. The native Windows PJRT DLL and all nine CPython
3.13 kernel extensions have been built with the locked supported toolchain.
Installed-wheel GPU qualification remains a separate integration gate.

The compiler sources are JAX `8dcb7aeced770bd22531b42d0b925c7342787c27` and XLA
`d867969800d4a8fd045505eabc65e1e66f30794f`. Windows port lineage is winjax
`43824c1930b69f2f77f066296eb5f29024ab46fa`, with native changes from its XLA fork
`9a28ad95fb9bab712917188e622fc1595ca9ead7`. It is build-time source lineage,
**never an installed dependency**.

The native patch retains Windows import-library linkage, CUDA VMM handle-type
selection, real PJRT dynamic loading, CUDA host/device math portability, and
COFF-aware cuDNN/CUPTI profiler symbol resolution. The allocator patch is rebased
onto the current `VmmGranularityProbe` implementation, not its old signature.
JAX's kernel-wheel patch omits only the unsupported Mosaic GPU extension and
runtime; its MAGMA lookup uses real Windows library APIs, not success-shaped
stubs. The stock optional MAGMA behavior remains unchanged when no MAGMA library
is installed; this package does not claim MAGMA acceleration.
The Windows subprocess patch separates successful waiting from the child's exit
code: `Wait()` reports whether it waited, while `Communicate()` preserves the
actual compiler status instead of collapsing every nonzero exit to `-1`.
CUDA compiler subprocesses capture both stdout and stderr, including the PTX ISA
capability probe. Windows ptxas writes its unsupported-version diagnostic to
stdout; discarding it makes the probe fail and lets LLVM emit an ISA newer than
the installed assembler supports. The real probe now selects the supported ISA;
no fixed PTX version, compiler fallback, or code-generation disablement is used.

These old port fixes are **already present** in the pinned XLA and are therefore
not reapplied:

- host-transfer dependency construction before moving its async value;
- host-offloading program-shape lookup before moving its executable;
- exact-boolean FFI attribute overload;
- allocator retry progress, now using allocation granularity rather than the
  older minimum-allocation-size correction.

The old Triton AMD `GCNAsmFormat.h` patch is dropped because that file no longer
exists in the pinned Triton. The current Triton patch retains clang-cl's
standard template parsing and unambiguous unsigned partition lookup.
**Ordinary XLA Triton code generation is not disabled.** Unsupported JAX
Mosaic/Pallas GPU APIs are a separate capability boundary. NCCL is disabled
with the actual upstream `no_nccl_support` setting; the production targets do
not reference the NVSHMEM stub/library (the upstream lit-test target does).
Neither NCCL nor NVSHMEM is a runtime dependency; this is a single-device build.

The `sparse_qr` build capability is retained: pinned JAX
`jaxlib/gpu/solver.cc:45-49` registers `cusolver_csrlsvqr_ffi` for CUDA, and
`//jaxlib/cuda:_solver` is included in the actual kernel wheel. The lock records
the source, prefix definition, target, module, and registration name. Runtime
admission must confirm that the real `_solver.registrations()` map contains
that entry before accepting this capability. This source-backed declaration is
not numerical GPU qualification or a substitute sparse-solver implementation.

JAX's existing protobuf, Abseil, rules_python, gRPC and XLA patches remain in
place. The build explicitly reapplies JAX's two XLA archive patches before
using `--override_module=xla`, since an override would otherwise bypass them.
The additional dependency patches are appended after existing upstream patch
series through repository definitions, not applied to a shared Bazel cache.
Triton's quilt series is preserved before the Windows-specific patch.
The rules_python Windows runtime-venv patch follows JAX's existing three patches.
It materializes file aliases with same-volume hardlinks and directory aliases
with real copies, so build tools do not require Windows symlink privileges.
POSIX behavior is unchanged; no administrator/Developer Mode setting is changed.
The compiler dispatch wrapper decodes response-file quoting with Windows'
`CommandLineToArgvW`, preserving escaped reproducibility macros and paths.
The private merged CUDA include tree is declared as a real header input; Bazel's
undeclared-inclusion checks remain enabled for CRT, cuRAND, and NVTX headers.
Kernel-wheel subprocesses propagate normalized extended-length Windows paths for
all Python import roots, including the hermetic interpreter's native standard
library extensions; this preserves real setuptools builds under deep Bazel paths.
Tensor IR's Windows patch follows its complete upstream quilt series and makes
the loop-count `std::max` type explicitly `int64_t`; it preserves 64-bit counts
on Windows' LLP64 ABI without changing the tiling calculation.

## Build inputs and exact command

Run from the worktree with a standard CPython 3.13 x64 interpreter. Supply an
explicit existing Git executable, profile-local LLVM 22.1.8, the supported MSVC
14.51.36231 (v145 / compiler 19.51) headers/libraries/tools, Windows SDK
10.0.26100.0, and MSYS2 stable 20260927. CUDA 13.3 supports the MSVC 195x / Visual
Studio 18 family; these STL headers require CUDA >=13.2 and Clang >=20. No
unsupported-compiler or STL-version override is permitted. An explicitly
authorized installed Git executable may be read and executed outside
`C:/Users/Agent`; no portable Git installation is required. All mutable state
still stays inside the profile. Do not use a machine installer, global tool
discovery, `vcvarsall`, or a mutable toolkit outside `C:/Users/Agent`.
The build driver downloads the exact locked Bazel itself. LLVM, MSVC, SDK and
MSYS2 inputs are explicit and their content-tree hashes are recorded in the
resulting manifest. Their redistribution/license eligibility remains the
provider's responsibility; those build tools are not included in the wheel.

Create a JSON file with exactly these fields:

| Field | Required value |
| --- | --- |
| `msvc` | Absolute extracted toolset root containing `bin/Hostx64/x64/cl.exe`, `include`, and `lib/x64` |
| `windows_sdk` | Absolute SDK root containing `Include`, `Lib`, and `bin` |
| `windows_sdk_version` | Exact SDK subdirectory version |
| `llvm` | Absolute LLVM root containing `bin/clang-cl.exe`, `clang++.exe`, and `llvm-lib.exe` |
| `llvm_version` | Exact LLVM release version; its major identifies `lib/clang/<major>` |
| `msys` | Absolute MSYS2 root containing `usr/bin/bash.exe` |
| `cuda` | New, nonexistent profile-local destination for the build-owned locked CUDA toolkit |
| `cudnn` | New, nonexistent profile-local destination for the locked cuDNN development wheel |

For example, with an authorized installed Git executable at the following
explicit location and the provisioned input file, the full command is:

```powershell
& C:/Users/Agent/.phydrax-build/build-envs/pjrt/Scripts/python.exe `
  tools/build_cuda_pjrt_wheels.py `
  --python C:/Users/Agent/.phydrax-build/build-envs/pjrt/Scripts/python.exe `
  --git "C:/Program Files/Git/cmd/git.exe" `
  --toolchain-inputs C:/Users/Agent/.phydrax-build/pj/inputs.json `
  --work-root C:/Users/Agent/.phydrax-build/pj `
  --output-dir C:/Users/Agent/.phydrax-build/wheelhouse/pjrt-release `
  --jobs 4 --memory-mb 24576 --timeout 86400
```

The authorized installed Microsoft inputs were copied into profile-local
staging, and LLVM/MSYS2 archives were authenticated before extraction. A real
LLVM 22.1.8 C++20 compile/link/run with these MSVC/SDK inputs succeeded. This is
toolchain evidence, not a successful PJRT/kernel build or GPU qualification.
The integration owner must resolve real native compiler diagnostics;
source analysis is not a successful Bazel build. A work root may be reused for
its authenticated download cache; supply new CUDA/cuDNN destinations for each
full build. `--prepare-only` downloads/authenticates
sources, applies patches, provisions toolkit repositories and renders explicit
paths, but deliberately does not run Bazel or emit a wheel. Its `prepared.json`
records the exact Bazel prefix and targets for the integration owner.

Source acquisition and JAX/XLA patch application can run independently of native
toolchain provisioning. From the worktree, the source-only command is:

```powershell
& C:/Users/Agent/phydra-labs/.worktrees/windows-cuda13/.venv/Scripts/python.exe `
  tools/build_cuda_pjrt_wheels.py `
  --python C:/Users/Agent/phydra-labs/.worktrees/windows-cuda13/.venv/Scripts/python.exe `
  --git "C:/Program Files/Git/cmd/git.exe" `
  --work-root C:/Users/Agent/.phydrax-build/cuda13-pjrt-sources `
  --sources-only
```

This phase does not read a toolchain-input file, download CUDA, invoke a compiler,
or require an output-wheel directory. Each attempt writes `build-*/sources.json`
containing status, the source-lock hash, resolved source paths, and
`source_extractions`: authenticated archive hashes, the link-materialization
policy, and each link's original text, text SHA-256, normalized target, archive
member, output path, and representation. Failure also records the
acquisition/patching phase and original exception type/message; completed patch
commands retain bounded evidence in `build-*/commands.json`.
The command prints its exact evidence path. External dependency patches are wired after
their preserved upstream series; Bazel applies those when materializing the
dependencies in the full build. `--sources-only` and `--prepare-only` are
mutually exclusive; the latter retains full toolchain preparation.

All subprocesses use `tools/_accelerator_process.py` with a finite deadline and
retained bounded command evidence. Bazel is run in batch mode, with explicit CPU
and memory limits and private output/repository caches. All downloads,
interpreter caches, generated paths, stdout/stderr and temporary files remain
under the supplied profile-local work root. Each invocation creates a fresh
`build-*` attempt directory and never adopts a previous partial source tree.
The memory budget reserves one third for the explicit JVM heap cap, one twelfth
for native/JVM overhead, and the remainder for Bazel's action-memory estimate.
For 24576 MiB this is an 8192 MiB heap, 14336 MiB action estimate, and 2048 MiB
reserve; action estimates are scheduling limits, not an OS process-memory quota.
Bazel batch mode serializes its client environment alongside upstream RC settings.
The driver retains only required Windows and proxy/certificate variables and
constructs PATH from explicit tool inputs. It generates the reachable Windows RC
configuration closure, preserving unscoped settings and original line order,
without editing the upstream RC or admitting global/user overrides. The selected
configurations and input/output hashes are retained in `bazelrc.json`. This avoids
serializing inactive Linux/macOS/RBE settings past Windows' command-line limit.
Bazel uses a short versioned installation path; output and user roots remain
private to each build attempt.
Extraction publishes a complete private temporary tree atomically; existing
destinations are refused. Compressed tar inputs are inflated once into a private
seekable temporary tar before indexed materialization; this avoids repeatedly
decompressing archive prefixes during link resolution. Uncompressed tar inputs
are read directly, and the temporary tar is removed on success or failure.
Tar symlinks resolve relative to their archive parent,
hardlinks relative to the archive root, and resolvable file/directory aliases
materialize real archive contents. A contained dangling symlink is represented
as an ordinary file containing its exact original link text, matching Git
`core.symlinks=false`; no empty target or fabricated target content is created.
This includes the upstream rapids-logger 0.2.3 template links to absent
`tests/VERSION`. Dangling hardlinks, escapes, and cycles are refused. Resolution
consults only archive members and never follows host filesystem links. Windows
symlink privileges are not needed. The complete `source_extractions` evidence is
also included in `prepared.json` and the wheel manifest, where it participates
in the canonical `build_id` hash.

After maintainer-reviewed recipe edits or formatting, refresh only the owned
recipe-file hashes before acquisition/build:

```powershell
& C:/Users/Agent/phydra-labs/.worktrees/windows-cuda13/.venv/Scripts/python.exe `
  tools/build_cuda_pjrt_wheels.py --refresh-recipe-lock
```

This explicit maintenance operation updates `sources.lock.json.recipe_inputs`
for the build driver and toolchain renderer. It never updates third-party
archive, source, patch, or license hashes and is never run automatically after
an integrity failure. Runtime Python files and the canonical PE parser are
hashed from their current bytes during verified staging, not by this command.

## Toolchain and staging contract

`toolchains/configure.py` calls only the pinned lineage's rendering helpers,
never its discovery/main entry point. It imports the current rules_cc C++ API
for Bazel 9 and creates private Windows CUDA library repositories. CUDA 13.3's
`cuda.lib` and `cudart.lib` are NVIDIA's static dispatch-loader thunks, not PE
import libraries or a full statically linked CUDA runtime; their vendor ABI is
preserved. The native compiler documents the hybrid runtime as display-driver
provided, so an actual compatible driver/runtime is still required and is not
established by a successful build. cuDNN import libraries are generated from the
canonical PE parser's real exports using `llvm-lib`.
LLVM compiles host code with the MSVC ABI and compiles CUDA code
through the port's CUDA dispatch wrapper. SDK/toolkit inputs are not patched in
place. CCCL is copied to a private directory before its two pinned port fixes.
The Bazel 9 Windows resource-compiler repository is explicitly rendered against
the staged SDK's real `rc.exe`, rather than invoking `vcvarsall` discovery.

The production Bazel targets are:

```text
@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so
//jaxlib/tools:jax_cuda13_plugin_wheel
```

The first output is a PE DLL despite the upstream `.so` target name. Staging
renames it to `phydrax_cuda13_pjrt/xla_cuda_plugin.dll`. The second must be the
real `jax_cuda13_plugin-0.12.0-cp313-cp313-win_amd64.whl`. Its nine compiled
extension modules and `version.py` are retained, with only the package
initializer replaced by the owned early-bootstrap initializer. No source-wheel
fallback or alternate kernel implementation is permitted.

The parser is copied byte-identically from `phydrax/_pe_dependencies.py`.
Bundled compiler tools live under a shared runtime root:

```text
phydrax_cuda13_pjrt/cuda/bin/ptxas.exe
phydrax_cuda13_pjrt/cuda/bin/nvlink.exe
phydrax_cuda13_pjrt/cuda/nvvm/libdevice/libdevice.10.bc
```
Both compiler executables come from the same authenticated toolkit as libdevice;
parallel code generation must not discover a linker in a build-machine CUDA tree.

No end-user nvcc dependency is introduced. Assets are authenticated by SHA-256,
PE machine (`0x8664`), `GetPjrtApi`, and each kernel's real `PyInit_*` export.
The stage consists exactly of manifest-declared assets and locked licenses.
The 46 locked notices include JAX's native aggregate and the materialized
transitive dependency notices. Stripped zstd/NVTX archive-root licenses are read
from authenticated repository-cache archives. ducc's complete BSD-3-Clause
header notice is retained under the build's BSD license choice, not mislabeled
as its alternative GPL license. Manifest license-source records preserve these
paths, archive identities, and the exact selected header range.
The manifest separately records native-build command/source-receipt hashes and
the native source-lock identity, so license-only restaging does not pretend that
native code was recompiled with a newer packaging recipe.
`manifest.json` includes source/patch/toolchain identity, resolved Bazel module
lock identity, actual compiler-input hashes, kernel-wheel hash, capabilities,
and the canonical JSON SHA-256 build identity. The backend receives both
`staging` and `kernel-wheel` config settings and independently authenticates
upstream wheel members, runtime sources, parser, PE exports and licenses before
writing the `cp313-cp313-win_amd64` wheel.
The manifest's `runtime_requirements` must exactly match wheel `Requires-Dist`.
CUPTI is pinned to `nvidia-cuda-cupti==13.3.35`, whose Windows wheel supplies
the compiled `cupti64_2026.2.0.dll` dependency; 13.4.92 supplies a different DLL
basename and is not substituted or renamed. Both `python3.dll` and
`python313.dll` are admitted from the actual CPython installation and explicitly
preloaded by absolute path before kernel extensions.

## Bootstrap and release boundaries

There is one JAX plugin entry point, `phydrax_cuda13_pjrt.plugin`. Kernel import
prepares DLL dependencies before `.pyd` import. PJRT discovery installs a
fail-loud CUDA registration guard before loading native code. Driver/device
observation is separate from metadata-only admission. Installed NVIDIA wheel
RECORDs own DLL dependencies; resident incompatible families must be refused,
not replaced. Compiler tool directories are process-local. WDDM defaults to
on-demand allocation without overriding explicit allocator choices.

The NVIDIA Windows display driver and compatible Microsoft VC runtime remain
host prerequisites; they are not bundled or installed by this build script.
An AMD64 DLL with correct exports is not evidence of successful GPU execution.
The integration owner must perform installed-wheel registration, native FFI,
profiler and scientific numerical qualification on the actual RTX 5090 Laptop
GPU with 24 GiB (not desktop RTX 5090 qualification), and
must verify Linux no-regression separately. Distributed, Mosaic/Pallas GPU and
MAGMA acceleration are not claimed here.

The pinned winjax tree has **no repository-wide license file**, but
`packaging/winjax/pyproject.toml` explicitly declares `Apache-2.0`, and inherited
Bazel templates and upstream XLA/JAX files retain Apache notices. Absence of a
root license is not evidence that all its code is unlicensed. The declaration's
scope for original root configuration/wrapper code still needs rights review.
That code is acquired only into a private build directory, never the runtime
wheel. Resolving that scope is a publication prerequisite. The locked
license inventory covers the acquired compiler/port dependency sources and
CUDA compiler assets; the integration owner must reconcile the final resolved
linked third-party dependency closure and distribution notices before release.

The runtime and numerical companion versions must equal the actual core release.
Do not overwrite an existing published version. Measure the final wheel size
and secure the package-index quota if needed; do not omit kernels, licenses,
compiler assets, or required runtime libraries to meet a quota.
