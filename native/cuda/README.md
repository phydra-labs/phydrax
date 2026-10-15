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
