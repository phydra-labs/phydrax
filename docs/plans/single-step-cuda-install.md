# Single-step CUDA installation: implementation record

## Current contract

`pip install "phydrax[cuda13]"` (or the equivalent `uv` command) is the CUDA
installation path. CUDA 13 is the only supported CUDA major. The NVIDIA driver
must support CUDA 13 (R580+); pip cannot install or upgrade it. The base
`pip install phydrax` remains a valid CPU-only installation.

The extra installs JAX's CUDA plugin, NVIDIA runtime/math wheels, and
`phydrax-cuda-major==13.*`. The metadata-only marker, built from
`packaging/cuda-major/13`, records explicit installed CUDA intent for diagnostics.
On Linux x86-64 and aarch64 the extra also pins the wheel-only
`phydrax-cuda13` companion to the Phydrax release. A matching native companion
wheel is required on both architectures. No system CUDA toolkit or
`LD_LIBRARY_PATH` is required for the wheel-based runtime.

The former multi-major implementation plan and its obsolete packaging,
publication, and switching recipes are superseded by this contract. Current
instructions live in [Installation](../installation.md),
[Native build and qualification](../../native/cuda/README.md), and
[Marker publication](../../packaging/cuda-major/README.md).

## Native ownership and release gates

- The companion contains separate lazily loaded cuDSS, AmgX, and QUDA components
  using the public JAX FFI C ABI. NVIDIA and driver libraries are not bundled;
  CUDA runtime linkage is shared.
- Build requirements come from `native/cuda13/pyproject.toml`. CMake assembles a
  toolkit view from NVIDIA wheel RECORDs without changing site-packages.
  SHA256-pinned redistributables supply link-time driver stubs and `cuobjdump`,
  not shipped runtimes.
- `tools/build_cuda_wheels.py` and `tools/qualify_cuda_wheels.py` target CUDA 13
  without a major selector. Native ABI identity and evidence still report major
  13. Component identity, release version, handler, dependency, and lifecycle
  validation remain required.
- Release SASS targets are 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0, plus PTX 12.0.
  Architecture-local development wheels do not qualify other GPUs. H100
  qualification requires inspected sm90 device code; use `--architectures
  90-real` for an H100-only development artifact. Retain wheel SHA256 and build
  manifests alongside numerical qualification evidence.
- cuDSS and AmgX native memos are capacity-bounded and device-partitioned.
  Release retires entries early; immutable preparations can reconstruct them
  from their operands, with rebuild evidence. QUDA remains single-process and
  one-device-per-process, with explicit refusals outside that contract.
- Foreign-major plugins, wheels, mapped libraries, and provider dependencies
  must still be diagnosed or refused. Ambiguous sonames such as
  `libcusparse.so.12` do not imply a foreign CUDA major: ownership and resolved
  library paths determine attribution. Hostile library-path qualification and
  process-exit checks remain release gates.
- GPU-direct mpi4jax requires a matching site MPI and user-provided bridge;
  ordinary MPI wheels do not qualify CUDA-aware transport or cluster fabrics.
- Publish marker 13 once, then the exact `phydrax-cuda13` companion wheels after
  qualification, then Phydrax. No source-build fallback or compatibility alias
  substitutes for a missing companion.

## Historical CUDA 13 evidence

These are retained implementation observations, not results from the current
cutover or an H100 qualification. Historical test counts describe the suites at
that time; they are not assertions about the current qualification inventory.

### 2026-10-15: RTX 5090 sm_120, driver 595.91.07 reporting CUDA 13.2

| Scenario | Recorded result |
| --- | --- |
| Phase 1 CUDA 13 environment, clean and hostile library paths | Successful; hostile path adds only `library-path-shadowing`; mapped majors {13} |
| Phase 1 CUDA 13 provider tests | AmgX 6 passed; cuDSS 17 passed; backend contracts 7 passed; environment 50 passed; import contract 2 passed; lattice 10 passed, 2 dtype skips; CUDA-direct and host-staged MPI 2 passed |
| Phase 2 development companion (`--architectures native`) | `phydrax_cuda13` 20.4 MB compressed; shared CUDA runtime, relative `$ORIGIN` RUNPATHs, no bundled NVIDIA or driver libraries |
| Phase 2 release-architecture companion | 81.3 MB compressed (AmgX 211.5 MB → 39.3 MB, QUDA 58.0 MB → 41.8 MB, cuDSS 0.38 MB); built in 45 min at `-j10`; all requested SASS/PTX present |
| Development-wheel qualification in a fresh environment | 103 passed, 11 skipped (two-GPU cases; Grid/qlat not installed); clean and hostile audits map 14 CUDA 13 wheel libraries only |
| Release-architecture wheel qualification | 103 passed, 11 skipped; clean and hostile audits CUDA 13 only |
| Component qualification | cuDSS rebuilt versus resident solves bitwise identical (f32, f64), capacity-1 partitions rebuild exactly; AmgX 16 passed; QUDA 22 passed |

The AmgX dDDI-only and dDDI V-cycle patches reduced the CUDA 13 sm_120 build
from 157.3 MB to 57.4 MB without removing a solver algorithm. QUDA compiles Eigen
even with `QUDA_USE_EIGEN=OFF`, so pinned Eigen `e67c494c` carries MPL-2.0 notices.
Its default tune cache includes both the source build hash and QUDA's own variant
(CPU architecture, compiled GPU architectures, CUDA compiler version), preventing
incompatible development/release builds from sharing a cache.

The recorded builds used host manylinux_2_39 tags. The manylinux_2_28 container
path was implemented but unexercised on that host, which had no container runtime.
The measured CUDA 13 release wheel fits PyPI's default 100 MB per-file limit;
new releases must be measured rather than assuming the historical size.

### Two-A100 MPI qualification

The CUDA 13 stack passed all eight integration cases on two A100-SXM4-80GB GPUs
with driver 580.159.04. The retained values, transport caveats, and artifact links
are in [Two-A100 results](../installation.md#two-a100-results). Those results do
not qualify H100 execution, network GPUDirect RDMA, or multi-host scaling.
