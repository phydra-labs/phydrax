# Phydrax CUDA companion

`phydrax-cuda13` is the lockstep, wheel-only CUDA 13 companion.
It contains separate lazily loaded cuDSS, AmgX, QUDA, cuTENSOR, cuTensorNet,
and cuEST wrapper libraries using the public JAX FFI C ABI. Optional vendor
runtimes are not new mandatory core dependencies. NVIDIA runtime libraries come
from their owning wheels; the cuEST C runtime has a separate optional
`phydrax-cuest` wheel. The CUDA runtime is shared.

## Build

Use a Python environment containing the companion's build requirements
from `../cuda13/pyproject.toml`. Never build against a
system CUDA toolkit. CMake assembles a symlink-only toolkit from installed NVIDIA
wheel RECORDs inside its build tree without modifying site-packages. Matching
toolkit symlinks are retained across reconfiguration. Downloads default to
`${CMAKE_BINARY_DIR}/downloads`; developers can pass
`-DPHX_CUDA_DOWNLOAD_DIR=/path/to/shared-cache` to reuse archives across builds.
The wheel sets omit driver stubs, so SHA256-pinned NVIDIA
`cuda_cudart` and `cuda_nvml_dev` redistributables supply only `libcuda.so` and
`libnvidia-ml.so` link-time stubs. No runtime from those archives is used or
shipped. CPU-only builders need neither a GPU driver nor a system toolkit.
A pinned `cuda_cuobjdump` redistributable supplies the matching major's binary
inspection tool, which is also not shipped in the companion.
The build host must be Linux x86-64 or aarch64. Support archives use NVIDIA's
`linux-x86_64` or `linux-sbsa` platform respectively, never Jetson artifacts.
Wheel audits verify native little-endian ELF64 machine identity for the
companion and its NVIDIA dependencies.

From the repository root, a development component build is:

```sh
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH -u CUDA_ROOT \
  /path/to/build-env/bin/cmake -S native/cuda -B /path/to/build-tree -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/path/to/build-env/bin/ninja \
  -DPHX_CUDA_ARCHITECTURES=native \
  -DPHX_CUDA_COMPONENTS=cudss \
  -DPython_EXECUTABLE=/path/to/build-env/bin/python
/path/to/build-env/bin/cmake --build /path/to/build-tree -j2
```

Set `PHX_CUDA_COMPONENTS` to a semicolon-separated subset of
`cudss;amgx;quda;cutensor;cutensornet;cuest`; an empty selection configures only
the common substrate. `collective`, `cudss_mgmn`, and `cudecomp` are explicit
site components and are excluded from portable release defaults. A component's
`src/<name>/CMakeLists.txt` calls
`phx_cuda_add_component(<name> SOURCES <files> LINK_LIBRARIES <targets>
HANDLERS <symbols>)`. It defines `component_registry()`, `component_shutdown()`,
and `PHX_CUDA_HANDLER_TABLE({"target_name", handler_symbol}, ...)` once. Declare
handler symbols with `extern "C" PHX_CUDA_EXPORT` before
`XLA_FFI_DEFINE_HANDLER_SYMBOL`, because XLA's macro does not set ELF visibility.
Common helpers and the mutex-protected partitioned registry are in
`src/common/phx_cuda_common.hpp`. Allocate registries on the heap, intentionally
retaining the registry itself; Phydrax explicitly releases their entries before
JAX/CUDA exit teardown. Registry IDs remain unique within a component library,
and callers retain `shared_ptr` ownership across eviction.

Development overrides `PHYDRAX_CUDA_CUDSS_LIBRARY`,
`PHYDRAX_CUDA_AMGX_LIBRARY`, `PHYDRAX_CUDA_QUDA_LIBRARY`,
`PHYDRAX_CUDA_CUTENSOR_LIBRARY`, `PHYDRAX_CUDA_CUTENSORNET_LIBRARY`, and
`PHYDRAX_CUDA_CUEST_LIBRARY` may name built component libraries. Installed
portable RUNPATHs are computed from owning wheel RECORDs and are relative to
`$ORIGIN`; the Python loader additionally preloads the matching NVIDIA
dependency closure by absolute path, protecting against other-major libraries
on `LD_LIBRARY_PATH`. cuTENSOR/cuTensorNet distribution names need not start
with `nvidia-`; their CUDA-major suffix and RECORD ownership remain authoritative.

Build and inspect a wheel without build isolation:

```sh
python tools/build_cuda_wheels.py --architectures native \
  --components cudss --python /path/to/build-env/bin/python \
  --build-dir /path/to/build/cuda13 --download-dir /path/to/shared-cache \
  --wheelhouse /path/to/wheelhouse --container none --jobs 2
```
`--build-dir` is required and should be reused for incremental builds.
`--download-dir` optionally shares verified archives across build trees;
without it, each build tree keeps its own `downloads` directory. `--jobs`
controls parallel compilation (default 2). Container mounts use these explicit
paths together with the wheelhouse, source tree, and supplied Python environment.
When changing the pinned AmgX patch chain in an existing build tree, remove its
generated `_deps/phx_amgx_source-src` and `_deps/phx_amgx_source-subbuild`
directories before reconfiguration. FetchContent otherwise reapplies the old
patches to already-patched sources. Retain the download cache and the rest of
the build tree; neither a clean toolkit installation nor a clean QUDA build is
needed.


Omit `--architectures` for release SASS coverage: CUDA 13 targets 80, 86, 89,
90, 100, and 120, with PTX 120. `native` is for local development and does not
qualify release architecture coverage. H100 artifacts must contain inspected
sm90 device code; use `--architectures 90-real` for an H100-only development
build. Runtime deployment requires an NVIDIA R580+ driver supporting CUDA 13.
The build tool repairs with auditwheel while excluding NVIDIA and driver
libraries, checks ELF dependencies/shared cudart/relative RUNPATHs, lists device
code with `cuobjdump`, and writes a JSON manifest with sizes and build identity.
Use `--container podman` or `--container docker` for the matching
`quay.io/pypa/manylinux_2_28_x86_64` or `manylinux_2_28_aarch64` build environment;
the container platform follows the native host, not an emulated cross-build.
Host development wheels receive the host-compatible manylinux tag and do not
claim manylinux_2_28 compatibility unless their actual symbol requirements
satisfy it. Source distributions are deliberately refused; publish wheels only.

DGX Spark has an aarch64 host and GB10 compute capability 12.1. Build locally
with `--architectures 121-real --components cudss,amgx,quda`; use
`--required-gpus 1` for installed-wheel qualification. The recorded Spark
artifact is `manylinux_2_39_aarch64` with sm121 SASS, not qualification of all
release targets or the manylinux_2_28 container. Keep both the host ELF
architecture and GPU device-code coverage in its build manifest.

B200 has compute capability 10.0, not the 12.0 of RTX PRO Blackwell. Use
`--architectures 100-real` for a B200-local development wheel, or a release
wheel containing sm100 SASS. An sm90-only H100/H200 wheel or sm120-only RTX
wheel does not qualify B200. The single-B200 CUDA 13 qualification exercised
all three components, their numerical contracts, and normal/hostile-path
teardown with `--required-gpus 1`; the inspected local wheel is sm100-only
`manylinux_2_39`, not a full release-architecture artifact. See
`benchmarks/b200_qualification.json` for exact artifact and runtime evidence.

B300 has compute capability 10.3. Its locally qualified x86-64 companion uses
`--architectures 103-real`, inspected sm103 AmgX/QUDA SASS, and
`--required-gpus 1` installed-wheel qualification on Python 3.13/JAX 0.12.
The `manylinux_2_39_x86_64` artifact does not claim full release architecture
or manylinux_2_28 coverage. Generic sm100 cubin compatibility alone is not
scientific qualification; retain the actual build and runtime evidence in
`benchmarks/b300_qualification.json`.

## Optional numerical runtimes and site components

Build headers for cuTENSOR and cuTensorNet are pinned in
`native/cuda13/pyproject.toml`; their runtime wheels are opt-in through the main
package extras. Their wrappers contain host code, not independent device kernels:
absence of wrapper SASS is not inspection or qualification of the vendor binary.
Preserve explicit computation precision, descriptor/workspace bounds, the exact
imported path, and version/build evidence.

`native/cuest_runtime` builds the optional platform wheel `phydrax-cuest`.
It redistributes the checksum-verified cuEST C library and the complete NVIDIA
license, including third-party notices; it does not ship the vendor SDK headers.
The small core companion wrapper links that C ABI but does not duplicate the
runtime into every core CUDA installation. Build the optional runtime wheel with
`uv build --wheel native/cuest_runtime` and include it in the consumer wheelhouse.
The runtime recipe is compiler-free and does not discover a system CUDA toolkit.
It accepts only Linux x86-64/SBSA native builds, verifies the complete archive
before each fresh extraction, and installs only the runtime SONAME plus complete
license material.

Site MPI integration is deliberately separate:

| Component | Required explicit build inputs |
| --- | --- |
| `collective` | `PHX_SITE_MPI_ROOT`, exact MPI library/header identities and runtime CUDA-buffer support |
| `cudss_mgmn` | The shared `collective` target and `PHX_CUDSS_COMM_LAYER`, `PHX_CUDSS_COMM_SHA256`, `PHX_CUDSS_COMM_BUILD_IDENTITY` |
| `cudecomp` | The shared `collective` target and `PHX_CUDECOMP_ROOT` |

Build all selected site components together so they share one live registry and
source identity. Do not compile an independent copy of the communicator registry
into each provider. Runtime overrides are `PHYDRAX_CUDA_COLLECTIVE_LIBRARY`,
`PHYDRAX_CUDA_CUDSS_MGMN_LIBRARY`, and `PHYDRAX_CUDA_CUDECOMP_LIBRARY`.
The MPI transport profile is same-host, one GPU per rank, with the actual
CUDA-aware Open MPI instance already initialized by the owning bootstrap. Native
lease creation verifies that all participants belong to one
`MPI_COMM_TYPE_SHARED` domain; process ancestry alone is insufficient.

The standalone recipe in `src/cudss_mgmn/commlayer` compiles the checksum-pinned,
unmodified NVIDIA Open MPI communication source against the chosen site MPI and
wheel-owned CUDA toolkit. Its JSON build record and the actual resulting shared
object hash supply the parent component's three communication inputs. This shim
uses MPI communicators for both host and device operations; an arbitrary NCCL
shim with a matching filename is not interchangeable.

The native collective bridge owns communicator creation, ordered call tickets,
supervision, and coordinated retirement. It has no fake FFI handlers or local
memo registry. Site-provider resources are retired through their collective
lease, not process-local LRU eviction or a Python atexit callback. Compiled
programs must be staged and admitted before `lease.invoke`; arbitrary callbacks
and unattached compiled executables are not accepted.
Collective staging and invocation require the exact source-verified JAX/JAXlib
0.12.0 dispatch implementation used by the CUDA extra. A downstream-modified
build with the same version string is not admitted.

Device scratch/cache limits, vendor memory estimates, opaque resident bytes,
compiler estimates, and observed memory samples are distinct. A finite vendor
estimate cannot certify an allocation bound. In particular, cuDSS's hybrid
device limit is not a full analysis/host-memory bound.


## A100 hardware qualification

For an A100-only artifact, select `--architectures 80-real` explicitly, not
`native` from a build tree previously used on an RTX 5090:

```sh
python tools/build_cuda_wheels.py --architectures 80-real \
  --components cudss,amgx,quda --python /path/to/cuda13-env/bin/python \
  --build-dir /path/to/build/cuda13-sm80 --download-dir /path/to/shared-cache \
  --wheelhouse /path/to/wheelhouse13 --jobs 12
```

Provision the companion's `build-system.requires` plus `build`, `auditwheel`,
and `patchelf` in that environment before invoking the tool. `readelf`, a host
C/C++ compiler, and `patch` must also be available. Set `--jobs` to the allocated
CPU budget, not the host's total CPU count.

The component build hash identifies native sources and patches, not a target
GPU or compiler installation. Retain the wheel SHA256 and JSON build manifest
alongside it; the manifest records toolkit wheels, compiler version, requested
architectures, and inspected SASS/PTX. An sm80-only wheel is not qualification
of the full release architecture set. The cuDSS adapter is host-only: its GPU
kernels come from the separately installed NVIDIA cuDSS wheel.

Run installed-wheel qualification with both allocated devices visible:

```sh
CUDA_VISIBLE_DEVICES=0,1 python tools/qualify_cuda_wheels.py \
  --required-gpus 2 --venv /path/to/qualification-env \
  --phydrax-wheel /path/to/phydrax.whl --wheelhouse /path/to/wheelhouse13 \
  --evidence /path/to/cuda13-a100-qualification.json
```

The wheelhouse must also contain the `phydrax-cuda-major` marker.
`--required-gpus N` selects the first N visible devices and requires passing
per-case JUnit device evidence, not just visibility. For four H100s, use
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--required-gpus 4`. Missing, skipped, or
partial provider or lifecycle coverage fails qualification. Provider tests run
serially; the concurrent cuDSS scenario owns its synchronized worker threads.
Before interpreting any multi-GPU provider result, run
`tools/qualify_cuda_execution` with all selected physical UUIDs and every
required scenario. Qualification requires direct-access capability for all
directed pairs, changed-content CUDA/JAX copies, and the tiny-buffer destination
consumer/readback matrix; a passing host-staged control does not repair a peer
failure.

cuDSS and AmgX checks cover independent solves with identical plan identities,
distinct numerical coefficients, device-partitioned caches, release/rebuild,
and process-exit cleanup across all selected devices. They do **not** certify
a distributed linear solve. QUDA scientific parity and teardown run in one
independent process per selected GPU; packaged QUDA still refuses a second
device in one process and a multi-process JAX runtime. A100 executes the
pre-SM90 active-mask warp-election branch, while H100 executes the SM90 branch;
hardware evidence remains architecture-specific. Do not run performance
measurements concurrently with builds or correctness checks.

The pinned AmgX patches own cuBLAS/cuSPARSE handles and the complete memory
manager (device pools, allocation bookkeeping, and thread streams) per CUDA
device. No device reuses the first GPU's pool addresses. Provider calls remain
serialized; resources survive device switches and are destroyed with their
owning device selected. Empty manager metadata remains retained so static
destruction never invokes CUDA cleanup.

Single-process cuDSS MG is a synchronous host-orchestrated route. Each phase
drains its configured per-device streams and all provider-owned work on every
reserved device before returning output buffers or admitting teardown.

Releasing a live preparation frees its named memo only. Reading an evicted or
released immutable preparation reconstructs a bounded memo under a fresh native
ID; the original preparation's ID does not change. Releasing that stale ID does
not release the anonymous reconstructed memo. Such entries remain bounded by
their device/plan cache capacity and are reclaimed by eviction or the explicit
component-wide release-all/shutdown sequence before process exit.

cuTENSOR and cuTensorNet propagate every cleanup status and preserve charges for
resources not proven freed. cuEST selects the owning device and drains its stream
before invoking any device-bound destructor. If either prerequisite fails, it
abandons parameters, plans, bases, workspaces, cuBLAS state, and the stream,
retains outstanding workspace charges, and poisons the partition.

Failed cuDSS numeric refresh and AmgX coefficient resetup retire the affected
resident memo without discarding partition rebuild counters. The original
immutable preparation rebuilds from its original operands on its next solve;
provider errors remain visible. cuDSS retains its per-entry operation lock
through retirement, while AmgX retains its serialized provider-call boundary.
Failure-path synchronization completes queued native work before returning
buffer ownership. Severe CUDA failures do not imply a recoverable device.


## Publishing order

1. Publish `phydrax-cuda-major` version `13.0` once. Never re-release this
   immutable installed-intent marker, built from `packaging/cuda-major/13`.
2. Publish `phydrax-cuda13` wheels for the exact Phydrax version, after GPU
   qualification of each delivered component on CUDA 13.
3. Only then publish the matching `phydrax` release.

**Never publish Phydrax before its exact companion wheels exist.** A missing
companion must fail resolution rather than silently fall back to a source build.
The cuDSS-only initial release must fit below PyPI's 100 MB file limit; request
larger per-file and per-project limits from measured AmgX/QUDA wheel manifests
before publishing those components.
