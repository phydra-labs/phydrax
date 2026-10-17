# Phydrax CUDA companions

`phydrax-cuda12` and `phydrax-cuda13` are lockstep, wheel-only companions.
Each contains separate lazily loaded cuDSS, AmgX, and QUDA libraries using the
public JAX FFI C ABI. NVIDIA libraries are runtime dependencies supplied by
NVIDIA wheels, not copied into the companion. The CUDA runtime is shared.

## Build

Use a Python environment containing the selected companion's build requirements
from `../cuda12/pyproject.toml` or `../cuda13/pyproject.toml`. Never build against a
system CUDA toolkit. CMake assembles a symlink-only toolkit from installed NVIDIA
wheel RECORDs inside its build tree without modifying site-packages. Matching
toolkit symlinks are retained across reconfiguration; CUDA 12 compiler archive
files consistently take precedence over overlapping wheel records. CUDA 12's
compiler is the SHA256-verified NVIDIA `cuda_nvcc` 12.9.86 redistributable; NVIDIA's
CUDA 12 compiler wheel lacks `nvcc`. Downloads default to
`${CMAKE_BINARY_DIR}/downloads`; developers can pass
`-DPHX_CUDA_DOWNLOAD_DIR=/path/to/shared-cache` to reuse archives across builds.
The wheel sets omit driver stubs, so SHA256-pinned NVIDIA
`cuda_cudart` and `cuda_nvml_dev` redistributables supply only `libcuda.so` and
`libnvidia-ml.so` link-time stubs. No runtime from those archives is used or
shipped. CPU-only builders need neither a GPU driver nor a system toolkit.
A pinned `cuda_cuobjdump` redistributable supplies the matching major's binary
inspection tool, which is also not shipped in the companion.

From the repository root, a development component build is:

```sh
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH -u CUDA_ROOT \
  /path/to/build-env/bin/cmake -S native/cuda -B /path/to/build-tree -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/path/to/build-env/bin/ninja \
  -DPHX_CUDA_MAJOR=13 -DPHX_CUDA_ARCHITECTURES=native \
  -DPHX_CUDA_COMPONENTS=cudss \
  -DPython_EXECUTABLE=/path/to/build-env/bin/python
/path/to/build-env/bin/cmake --build /path/to/build-tree -j2
```

Set `PHX_CUDA_COMPONENTS` to a semicolon-separated subset of `cudss;amgx;quda`;
an empty selection configures only the common substrate. A component's
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
`PHYDRAX_CUDA_AMGX_LIBRARY`, and `PHYDRAX_CUDA_QUDA_LIBRARY` may name built
component libraries. Installed RUNPATHs are computed from NVIDIA RECORDs and
are relative to `$ORIGIN`; the Python loader additionally preloads the matching
NVIDIA dependency closure by absolute path, protecting against other-major
libraries on `LD_LIBRARY_PATH`.

Build and inspect a wheel without build isolation:

```sh
python tools/build_cuda_wheels.py --major 13 --architectures native \
  --components cudss --python /path/to/build-env/bin/python \
  --build-dir /path/to/build/cuda13 --download-dir /path/to/shared-cache \
  --wheelhouse /path/to/wheelhouse --container none --jobs 2
```
`--build-dir` is required and should be reused for incremental builds of one
CUDA major. `--download-dir` optionally shares verified archives across majors;
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
90, 100, and 120, with PTX 120; CUDA 12 additionally targets 70 and 75. `native`
is for local development and does not qualify release architecture coverage.
The build tool repairs with auditwheel while excluding NVIDIA and driver
libraries, checks ELF dependencies/shared cudart/relative RUNPATHs, lists device
code with `cuobjdump`, and writes a JSON manifest with sizes and build identity.
Use `--container podman` or `--container docker` for the
`quay.io/pypa/manylinux_2_28_x86_64` build environment. Host development wheels
receive the host-compatible manylinux tag and do not claim manylinux_2_28
compatibility unless their actual symbol requirements satisfy it. Source
distributions are deliberately refused; publish wheels only.

## A100 hardware qualification

For an A100-only artifact, select `--architectures 80-real` explicitly, not
`native` from a build tree previously used on an RTX 5090. Use separate build
trees and Python environments for CUDA 12 and 13:

```sh
python tools/build_cuda_wheels.py --major 13 --architectures 80-real \
  --components cudss,amgx,quda --python /path/to/cuda13-env/bin/python \
  --build-dir /path/to/build/cuda13-sm80 --download-dir /path/to/shared-cache \
  --wheelhouse /path/to/wheelhouse13 --jobs 12
```

Provision the companion's `build-system.requires` plus `build`, `auditwheel`,
and `patchelf` in that environment before invoking the tool. `readelf`, a host
C/C++ compiler, and `patch` must also be available. Set `--jobs` to the allocated
CPU budget, not the host's total CPU count. Repeat with `--major 12` and a
separate CUDA 12 environment/build tree when qualifying that major.

The component build hash identifies native sources and patches, not a target
GPU or compiler installation. Retain the wheel SHA256 and JSON build manifest
alongside it; the manifest records toolkit wheels, compiler version, requested
architectures, and inspected SASS/PTX. An sm80-only wheel is not qualification
of the full release architecture set. The cuDSS adapter is host-only: its GPU
kernels come from the separately installed NVIDIA cuDSS wheel.

Run installed-wheel qualification with both allocated devices visible:

```sh
CUDA_VISIBLE_DEVICES=0,1 python tools/qualify_cuda_wheels.py \
  --major 13 --required-gpus 2 --venv /path/to/qualification-env \
  --phydrax-wheel /path/to/phydrax.whl --wheelhouse /path/to/wheelhouse13 \
  --evidence /path/to/cuda13-a100-qualification.json
```

The wheelhouse must also contain the matching `phydrax-cuda-major` marker.
The qualifier checks the required visible GPU count before running provider
tests, records each device's kind and compute capability, and rejects skipped
two-device coverage when `--required-gpus 2` is requested. Provider tests run
serially: one process owns the visible GPUs rather than independent xdist
workers racing to initialize providers on the same hardware.

Two-device cuDSS and AmgX checks cover independent solves with identical plan
identities, device-partitioned native caches, release/rebuild, and process-exit
cleanup. They do **not** certify a distributed linear solve. QUDA remains
one-device-per-process and must visibly refuse a second device. A100 executes
the existing pre-SM90 active-mask warp-election branch; sm80 numerical provider
tests are required in addition to artifact inspection. Do not run performance
measurements concurrently with builds or correctness checks.

The pinned AmgX patch owns cuBLAS and cuSPARSE handles per CUDA device, instead
of reusing the first device's process-global handles. Provider calls remain
serialized, handles survive device switches, and resource teardown destroys
each handle with its owning device selected.

Releasing a live preparation frees its named memo only. Reading an evicted or
released immutable preparation reconstructs a bounded memo under a fresh native
ID; the original preparation's ID does not change. Releasing that stale ID does
not release the anonymous reconstructed memo. Such entries remain bounded by
their device/plan cache capacity and are reclaimed by eviction or the explicit
component-wide release-all/shutdown sequence before process exit.


## Publishing order

1. Publish `phydrax-cuda-major` versions `12.0` and `13.0` once. Never re-release
   these immutable exclusivity markers.
2. Publish both `phydrax-cuda12` and `phydrax-cuda13` wheels for the exact Phydrax
   version, after GPU qualification of each delivered component on each major.
3. Only then publish the matching `phydrax` release.

**Never publish Phydrax before its exact companion wheels exist.** A missing
companion must fail resolution rather than silently fall back to a source build.
The cuDSS-only initial release must fit below PyPI's 100 MB file limit; request
larger per-file and per-project limits from measured AmgX/QUDA wheel manifests
before publishing those components.
