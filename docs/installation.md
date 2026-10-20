# Installation

Requires Python 3.12. The base install uses CPU JAX:

```bash
pip install phydrax
```

## NVIDIA GPUs

On Linux, install the CUDA 13 extra. `uv pip install` or `uv add` can be
used instead of `pip install`. CUDA 13 is the only supported CUDA major.

| Install | Minimum driver-supported CUDA version | Driver family |
| --- | --- | --- |
| `pip install "phydrax[cuda13]"` | 13.0 | R580+ |

The driver must report at least the listed CUDA version. Pip cannot install or
upgrade the NVIDIA driver. The extra installs JAX's CUDA plugin, NVIDIA runtime
and math-library wheels (including NCCL), and the metadata-only
`phydrax-cuda-major==13.*` marker recording installed CUDA intent.
A system CUDA toolkit and `LD_LIBRARY_PATH` are not required for
this JAX setup. The extra does not install a CUDA backend on non-Linux platforms.

On Linux x86-64, the extra also installs the lockstep wheel-only companion
`phydrax-cuda13==0.2.2`:

| Component | CUDA 13 |
| --- | --- |
| cuDSS sparse LDLᵀ | Included |
| AmgX algebraic multigrid | Included |
| QUDA lattice kernels | Included |

Release companions contain CUDA 13 SASS for sm 8.0, 8.6, 8.9, 9.0, 10.0,
and 12.0, plus PTX 12.0.
Linux aarch64 receives the JAX extra but has no companion yet. GPU-direct
mpi4jax still requires a matching user-provided build. See
[External solver backends](api/backends.md) for component contracts.

For development, set `PHYDRAX_CUDA_CUDSS_LIBRARY`,
`PHYDRAX_CUDA_AMGX_LIBRARY`, or `PHYDRAX_CUDA_QUDA_LIBRARY` to the absolute path
of a built `libphydrax_cuda_<component>.so` before first use. These overrides
retain component, release-version, CUDA-major, and handler validation. NVIDIA
dependencies are preloaded by absolute path from installed NVIDIA wheels.

An architecture-local developer wheel is not portable merely because its CUDA
major matches. In particular, an RTX 5090 `--architectures native` build does not
qualify an A100 or H100. Rebuild with `--architectures 80-real` for an sm80-only
development artifact or `--architectures 90-real` for an H100 sm90 artifact,
or use a release companion containing the required SASS. H100 qualification
requires inspected sm90 device code; a CUDA 13 identity alone is insufficient.
Retain the build manifest and wheel SHA256 and qualify in a clean environment.
See `native/cuda/README.md` in the source checkout for build instructions.

`phydrax.backends.cuda_native_resources()` inspects already loaded components
without loading others. Its `components` record identities; `partitions` report
`component`, `device`, `partition`, `live`, `capacity`, and `rebuilds`. cuDSS and
AmgX keep bounded per-plan native memos; eviction rebuilds from call operands.

## Inspect the environment

```python
from phydrax.backends import cuda_environment

report = cuda_environment()
print(report.successful)
print(report.facts)
for issue in report.issues:
    print(issue.severity, issue.code, issue.message, issue.paths)
```

`cuda_environment()` probes the JAX backend and inspects mapped CUDA libraries
on Linux. A report is successful when it has no error-severity issues; warnings
do not make it unsuccessful. A CPU-only install without a CUDA marker or plugin
is valid. `assess_cuda_environment(facts)` is the pure alternative for assessing
explicit `CudaEnvironmentFacts`, without probing devices or collecting host
state. See the [CUDA environment API](api/backends.md#cuda-environment).

Provider availability exposes related typed issues through
`BackendAvailability.environment_issues`. Provider dependency issues are attached
to availability; the general environment report does not probe every provider.
Availability checks use installed metadata and fresh-process native dependency
resolution, not the diagnostic list of libraries already mapped in this process.

| Issue code | Meaning and remedy |
| --- | --- |
| `unsupported-cuda-major` | The active runtime or single-major mapped toolkit evidence identifies a major other than 13. Remove foreign-major packages and paths, install `phydrax[cuda13]`, and restart; use the CPU-only install if CUDA 13 is unavailable. |
| `no-cuda-plugin` | A CUDA marker is installed without a JAX CUDA plugin. Reinstall the selected extra on supported Linux; use the CPU install on unsupported platforms. This is an error on Linux x86-64 and a warning elsewhere. |
| `duplicate-cuda-plugins` | Multiple CUDA plugins are installed. Remove the other-major plugin and its wheels, or create a fresh environment. |
| `plugin-major-mismatch` | The installed CUDA intent marker or unique plugin does not match required major 13, including a foreign-only plugin without a marker. Remove foreign-major packages and reinstall `phydrax[cuda13]`. |
| `foreign-nvidia-distributions` | Other-major NVIDIA wheels remain installed. Run the exact `pip uninstall` list in the message, then reinstall `phydrax[cuda13]`. |
| `driver-too-old` | The driver does not support CUDA 13. Upgrade to R580+ or use the CPU-only install. |
| `cpu-platform-selected` | Configuration deliberately excludes CUDA despite an installed plugin. Keep this warning for CPU workers, or select CUDA before startup; see [Configuration](api/configuration.md#phases-and-startup). |
| `backend-unavailable` | A device probe found no usable CUDA backend. Inspect the backend error in the message (`facts.backend_error`), GPU visibility, driver, and any listed library-path directories. Remove shadowing CUDA paths and restart. |
| `library-path-shadowing` | `LD_LIBRARY_PATH` contains CUDA libraries that can override wheel libraries. Remove those entries and restart; a currently working backend does not make this safe. |
| `mixed-cuda-majors` | Mapped libraries have more than one known CUDA major. Remove incompatible providers or library paths, rebuild against CUDA 13, and restart in a clean process. |
| `provider-major-mismatch` | A user-provided provider's native dependencies target another major. Rebuild or reinstall it for JAX's CUDA major. |
| `provider-dependency-unresolved` | A provider's CUDA dependency cannot be resolved. Install its matching dependencies and correct its RPATH or deployment paths, then retry in a new process. |
| `companion-missing` | Install the matching CUDA extra on Linux x86-64, or supply a development library override. |
| `companion-invalid` | The library is unreadable or lacks valid identity symbols or handlers. Reinstall the matching companion or rebuild the override. |
| `companion-version-mismatch` | Companion and Phydrax releases differ. Install matching versions. |
| `companion-major-mismatch` | The component targets a different CUDA major from JAX. Install the matching extra or rebuild the override. |
| `native-dependency-missing` | A required native dependency could not be preloaded from NVIDIA wheels. Reinstall the selected extra and its dependencies. |

Unknown library majors are reported as `None`, not guessed from ambiguous
sonames such as `libcusparse.so.12`. NVIDIA distribution version numbers are not
CUDA-major identifiers either.

### Foreign-major contamination

Use a fresh virtual environment for CUDA 13 when an existing environment
contains another CUDA major. Installing `phydrax[cuda13]` does not remove
foreign plugins, companions, or NVIDIA wheels. `foreign-nvidia-distributions`
lists the NVIDIA distributions to uninstall. Rebuild user-provided native
providers for CUDA 13 and restart Python; installed-package changes cannot
unload libraries already in memory.

### Library-path shadowing

Do not add a system toolkit's library directory to `LD_LIBRARY_PATH` for the
wheel-based JAX setup. Such paths can shadow NVIDIA wheel libraries, including
libraries whose sonames are shared across CUDA majors. This produces
`library-path-shadowing` when no backend-unavailable issue is present. It can
also prevent the CUDA plugin from initializing, producing
`backend-unavailable` instead, with the backend error and affected directories.
Remove the CUDA entries (or unset `LD_LIBRARY_PATH` if it has no other deployment
purpose) before starting a new Python process.

## Single-B200 deployment profile

The single NVIDIA B200 host uses compute capability 10.0, driver 595.91.07,
JAX/JAXlib 0.11.2, and CUDA runtime 13.4.92. Qualification selects one visible
GPU before startup, retains CPU reference/callback support, and applies no
NCCL or UCX transport workaround:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m tools.distributed_substrate_benchmark \
  --size 17 --iterations 1
```

The observed runtime has one process and one GPU; the weighted reduction and
PCG solve pass their numerical checks. `--devices-per-process` is not a
single-process visibility limit: use `CUDA_VISIBLE_DEVICES` before JAX
initializes. No MPI installation or distributed coordinator is required.

The development companion was built with `--architectures 100-real`. Inspected
AmgX/QUDA device code is sm100-only; cuDSS uses its separately installed NVIDIA
provider kernels through a host-only wrapper. All three native providers pass
single-device numerical and lifecycle qualification, including clean and hostile
library-path audits. The latter reports the expected `library-path-shadowing`
warning while mapping CUDA 13 wheel libraries. This `manylinux_2_39` artifact
does not qualify the full release architecture set or `manylinux_2_28`.

Keep the audited local wheel when installing from this checkout:

```bash
uv pip install --python .venv/bin/python --no-sources \
  --find-links dist/cuda13-sm100 -e ".[cuda13]"
```

Native-JAX MACE and the explicitly selected Pallas MACE route are distinct.
B200 is admitted for compiled MACE with float32/float64 forward and derivative
evidence; B300 (10.3) remains refused. The exercised native-JAX workflows also
cover ordered sparse adjoints, one-device spectral transforms, PIC evolution
and restart, dense linear algebra, and compiled FMM forward/derivative parity.
Ordered FMM strength adjoints also replay bitwise within the same compiled
program after removing duplicate-index GPU accumulation from the ownership
transposes. Fresh-process exact MACE continuation still requires ordered
components and the declared fixed compiler profile; this is not a
cross-H200/B200 bitwise-continuation guarantee.

Exact commands, numerical outcomes, artifact identities, and measurement
scopes are retained in
[`b200_qualification.json`](https://github.com/phydra-labs/phydrax/blob/dev/benchmarks/b200_qualification.json).
Single-device evidence does not qualify peer transfers, multi-GPU collectives,
multi-host execution, or network RDMA.

## Multi-GPU and multi-node execution

Use JAX global arrays and NCCL for NVIDIA multi-GPU communication, with
`JaxCollectiveProvider` for named-axis operations. Startup automatically
coordinates multi-process `srun` and `mpirun` launches through JAX's launcher
detection. Rank settings must exist before `import phydrax`. See
[Bootstrap before device initialization](api/execution.md#bootstrap-before-device-initialization)
and [Configuration startup](api/configuration.md#phases-and-startup) for explicit
coordinator settings and supported launcher environments. MPI is a launcher or
an optional interoperability route, not a requirement for JAX/NCCL collectives.

Local worker processes must own disjoint GPUs. On four GPUs,
`LocalProcessLaunchPlan(..., devices_per_process=1)` supports four workers;
two workers can instead own two GPUs each. The equivalent smoke-driver options
are `--local-processes 4 --devices-per-process 1` and
`--local-processes 2 --devices-per-process 2`. One process can own all four.
Ranks sharing one GPU do not qualify physical multi-GPU execution.
Resource-inventory discovery is collective; every rank must invoke it in the
same sequence. Multi-process Linux inventories collectively record kernel boot
identities for physical-host ownership; unavailable identity remains unknown
and resource admission refuses it. This intentionally changes multi-process
execution identities, not single-process identities or process-local memory samples.
cuDSS and AmgX preparations remain single-device solves with device-isolated
caches, not distributed sparse solvers. Packaged QUDA requires one JAX process
and refuses a second device within that process.

### Check PCIe peer-transfer integrity

Device visibility and `cuda_environment().successful` do not verify inter-GPU
data movement. Before qualifying a new multi-GPU host, verify peer-copy contents,
not just CUDA return codes or `nvidia-smi topo -p2p` capability reports:

```python
import jax
import numpy as np

devices = jax.devices("gpu")
expected = np.arange(1, 17, dtype=np.float32)
for source in devices:
    value = jax.device_put(expected, source)
    for target in devices:
        if target != source:
            observed = np.asarray(jax.device_put(value, target))
            np.testing.assert_array_equal(
                observed, expected, err_msg=f"{source} -> {target}"
            )
```

On the four RTX PRO 6000 Blackwell Server Edition host (sm120, driver
595.91.07), all four devices initialize with JAX/JAXlib 0.11.2 and CUDA 13.4.
However, all twelve directed peer-copy pairs corrupt data in an independent
CUDA-runtime probe despite successful API returns. Raw JAX device copies also
corrupt data, and the default four-GPU NCCL collective hangs. The kernel command
line includes `intel_iommu=on`, and all four GPUs have translated `DMA-FQ`
IOMMU domains. This matches the configuration implicated by NVIDIA's documented
[Linux bare-metal IOMMU/PCIe P2P failure mode](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html.md#troubleshooting-iommu-p2p);
the host operator must inspect and repair that configuration.

Do not qualify general multi-GPU execution on this host until the host operator
repairs the IOMMU/ACS/driver configuration and peer-copy integrity passes.
Do not change virtualization or isolation settings blindly from a container.
`NCCL_P2P_DISABLE=1` changes NCCL transport only; it does **not** repair JAX
device-to-device copies. A host-origin, four-device raw-JAX reduction and the
four-process/one-GPU-per-process substrate smoke pass with that explicit setting.
The two-process/two-GPUs-per-process launch still aborts. These bounded results
do not qualify arbitrary single-process multi-GPU workflows, GPU-direct MPI,
or peer transfers.

```bash
env -u LD_LIBRARY_PATH NCCL_P2P_DISABLE=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m tools.distributed_substrate_benchmark \
  --local-processes 4 --devices-per-process 1 --size 1024 --iterations 1
```

Benchmark environment fingerprints include `NCCL_P2P_DISABLE`, and the substrate
smoke reports it explicitly. Peer-enabled and peer-disabled measurements are not
interchangeable. This PCIe host does not inherit the H100 NVSwitch deployment
profile below.

The local companion was built with `--architectures 120-real` and all three
components. Its inspected AmgX/QUDA SASS is sm120-only; the wheel targets
manylinux_2_39 and is not a full release-architecture artifact. cuDSS, AmgX,
and QUDA numerical checks pass on each of the four GPUs in independent
single-GPU processes. Installed-wheel clean and hostile-library-path audits
map only CUDA 13 wheel libraries; the hostile audit reports the expected
`library-path-shadowing` warning. Compiled MACE and FMM correctness checks also
pass on one GPU. The full `--required-gpus 4` native qualification timed out and
is **not** claimed as passed. `benchmarks/rtx_pro_6000_qualification.json` retains
the artifact identities, exact verification scopes, and failed peer-transfer
evidence.

From this checkout, keep the audited development wheel rather than rebuilding
the companion implicitly:

```bash
uv pip install --python .venv/bin/python --no-sources \
  --find-links dist/cuda13-sm120 -e ".[cuda13]"
```

### Four-H100 deployment profile

The four-H100 SXM deployment uses driver 580.126.09, JAX/JAXlib 0.11.2, CUDA
runtime 13.4, and NCCL 2.32.3. All six GPU pairs report NV18 connectivity.
Default NVSwitch multicast/NVLS initialization fails with CUDA error 401;
an independent raw-JAX probe with NCCL 2.31.2 also fails during multicast binding.
Ordinary NCCL collectives work with the explicitly selected
`NCCL_NVLS_ENABLE=0` profile. This is a deployment setting, not an automatic
Phydrax fallback or qualification of NVLS. Host driver/Fabric Manager/NVSwitch
multicast configuration remains an external prerequisite for enabling NVLS.

```bash
env -u LD_LIBRARY_PATH NCCL_NVLS_ENABLE=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m tools.distributed_substrate_benchmark \
  --local-processes 4 --devices-per-process 1
```

Keep this setting explicit in launches and retained evidence. Benchmark
environment fingerprints include `NCCL_NVLS_ENABLE`; results from different
transport settings are not interchangeable. A same-host four-GPU run does not
qualify multi-host operation or network GPUDirect RDMA.

### Five-H200 deployment profile

The five-H200 SXM host uses driver 580.95.05, JAX/JAXlib 0.11.2, CUDA 13.4,
and NCCL 2.32.3. All twenty directed CUDA peer-copy content checks and all
twenty directed JAX device-copy checks passed. Raw JAX collectives passed on
one through five selected devices without `NCCL_NVLS_ENABLE` or
`NCCL_P2P_DISABLE` overrides. This does not establish which NCCL algorithm
executed or independently qualify NVLS; do not copy another host's transport
workaround into this deployment.

H200 uses the same sm90 target as H100. The local companion contains inspected
sm90 AmgX/QUDA device code and a host-only cuDSS wrapper; its
`manylinux_2_39` development wheel does not claim full release architecture or
`manylinux_2_28` coverage.

Select device subsets before JAX initializes. In particular, the single-process
substrate smoke uses all visible devices, regardless of `--devices-per-process`;
use `CUDA_VISIBLE_DEVICES` to preserve a smaller allocation. Five devices support
uniform `1×5` and `5×1` process/device layouts. A four-device `2×2` job plus an
independent fifth-device job requires disjoint allocations, not an unequal split
through `ExecutionRuntime.child_groups`.

Five-way spectral slabs require both partitioned canonical and padded dimensions
to be divisible by five. They are not a five-device nondegenerate pencil mesh;
retain the four-device `2×2` pencil route separately. Device-count changes also
change execution ownership: topology-changing PIC restart requires its explicit
tolerance admission rather than a bitwise-continuation claim.

The canonical full-mesh pencil schedule passes the composed-transform and
production `2×2` PIC checks with the default NCCL settings. The earlier subgroup
schedule failed across executable histories. `NCCL_LAUNCH_ORDER_IMPLICIT=1`
helped its minimal reproducer but did not qualify the production path; it is
not a required workaround for the new schedule.

CUDA-aware MPI has a separate explicit profile on this host:
`UCX_MEMTYPE_CACHE=n`. With UCX 1.20.1 and Open MPI 5.0.10, the default cache
misclassified CUDA buffers and the direct tests faulted. Qualification forwards
the caller's setting and records it in the environment identity; it never
silently selects that profile or falls back to host staging. CUDA awareness does
not establish zero host copies or network RDMA.

Exact accelerator continuation requires ordered numerical components. Across
fresh processes it additionally requires the same device/runtime and fixed
compiler choices: the tested replay profile sets
`XLA_FLAGS=--xla_gpu_autotune_level=0` before JAX initializes. Timing-based
autotuning can select different kernels and produce different future rounding
even when a checkpoint restores every bit correctly. This profile is explicit
for exact-replay scenarios, not a process-wide Phydrax default or a condition
silently applied to ordinary performance measurements.

Retained commands, source identities, numerical gates, failed diagnostic
profiles, and phase-separated measurements are in
[`h200_qualification.json`](https://github.com/phydra-labs/phydrax/blob/dev/benchmarks/h200_qualification.json).

The exclusive campaign completed 67 invocations: 24 spectral, 16 canonical
native-JAX MACE, 24 ordered MACE, and three PIC route sweeps. Every invocation
returned successfully with unchanged recorded sources and no foreign compute
process present at admission. Phase-separated preparation, lowering, compilation,
compiler memory, retained bytes, and warmed samples are retained in the record.

Representative warmed medians below are seconds, with five samples per row.
The original package is commit `41a1a1609`, measured on the same H200 allocation
with a matched harness. It is a performance reference, not a bitwise-qualified
MACE implementation. PIC has no completed matched original baseline.

| Workload | Original, 4 GPUs | Current, 4 GPUs | Current, 5 GPUs |
|---|---:|---:|---:|
| Complex64 slab FFT, `480³` | 0.003565 | 0.003658 | 0.003648 |
| Native MACE E/F/strain, 2,048 atoms, fast streaming | 0.164268 | 3.125020 | 2.500433 |
| Native MACE E/F/strain, 2,048 atoms, ordered streaming | 1.393871 | 27.941694 | 22.561710 |
| Reduced PIC, fixed total problem | — | 0.147840 | 0.142669 |
| Global-FFT PIC, fixed total problem | — | 3.831720 | 3.105693 |
| Local-guarded PIC, fixed total problem | — | 4.175138 | 3.332588 |

The determinism repairs have a substantial native-MACE cost; this change is not
a general performance improvement. Five devices help the measured MACE and
spectral-PIC workloads relative to four, but add little for this slab FFT or
reduced-PIC case. Treat cold preparation and compilation separately from these
warmed values. The three-/five-part spectral PIC sweep uses its recorded common
grid, capacities, and guarded tiling, not the smaller historical `1/2/4` problem.

## GPU-direct MPI interoperability

`phydrax[mpi]` is opt-in interoperability support. GPU-direct mpi4jax requires a
site-provided CUDA-aware MPI and a mpi4jax build matching both that MPI and
JAX's CUDA major. Ordinary PyPI MPI wheels are built without CUDA support;
shipping another MPI in the environment can also shadow the cluster's MPI and
fabric integration. Use the site's `mpicc`, its MPI libraries, and its matching
`mpirun` for mpi4py and mpi4jax.

For CUDA 13, point `CUDA_ROOT` at a CUDA 13 toolkit view containing `bin`,
`include`, and `lib` (or `lib64`). This is a build prerequisite, not a runtime
requirement for JAX. The verified mpi4jax rebuild uses an explicit wheel-library
RPATH because automatic CUDA 13 wheel-layout detection is insufficient:

```bash
export MPICC=/path/to/site/mpi/bin/mpicc
export MPI4JAX_BUILD_MPICC="$MPICC"
export CUDA_ROOT=/path/to/cuda13-toolkit-view
export PATH="/path/to/site/mpi/bin:$CUDA_ROOT/bin:$PATH"
# Replace <venv> with the environment's site-packages directory containing nvidia.
export LDFLAGS="-Wl,-rpath,<venv>/nvidia/cu13/lib"
uv pip install --no-cache --no-build-isolation --reinstall mpi4jax
```

Install mpi4py against the same MPI and install mpi4jax's build dependencies,
including nanobind, before using `--no-build-isolation`.
Set `MPI4JAX_USE_CUDA_MPI=1` before importing
mpi4jax for GPU-direct transport. A CUDA-enabled bridge is checked before import
even for host-staged use, because mpi4jax loads that bridge unconditionally.
See [Distributed and vendor execution capabilities](api/backends.md#distributed-and-vendor-execution-capabilities)
for communicator ownership, direct-transport qualification, and deployment limits.

### Single-host CUDA-aware MPI qualification

`tools.qualify_cuda_mpi` accepts an already-built, matching MPI/UCX/Python
installation. It does not install a replacement MPI into an application
environment. Use a clean CUDA 13 prefix and Python environment.
The A100 reference stack uses Open MPI 5.0.10, UCX 1.20.1, mpi4py 4.1.2,
mpi4jax 0.9.1.post2, and JAX/JAXlib 0.11.2. UCX includes CUDA IPC and CUDA-copy
transports; this bounded same-host build does not include RDMA or GDRCopy.

Run from a checkout with its canonical `uv.lock`. Each target Python environment
must contain Phydrax (an editable install is sufficient), its locked dependencies,
and the test dependencies; a `PYTHONPATH` entry alone does not install the package
metadata used by benchmark provenance.

```bash
JAX_PLATFORMS=cpu python -m tools.qualify_cuda_mpi \
  --python /path/to/cuda13-env/bin/python \
  --mpi-prefix /path/to/cuda13-mpi \
  --ucx-prefix /path/to/cuda13-ucx \
  --visible-devices 0,1 \
  --nsys /path/to/compatible/nsys \
  --output cuda13-mpi-qualification.json
```

The driver derives the rank count from the explicit visible-device list; use
`--visible-devices 0,1,2,3` for four ranks. It requires all eight
dtype/placement/transport integration cases for that exact rank count to pass;
optional-suite skips or a two-rank subset fail four-rank qualification.
It then measures sum and ring sendrecv over an explicit bounded message-size
sweep, in fresh processes for CUDA-aware and mpi4jax-host-staged routes.
Every rank must own a distinct physical GPU in one MPI shared-memory domain;
hostnames alone do not prove co-location. The separately labeled shared-GPU
integration cases test compatibility, not physical multi-GPU transport.
Control collectives use a separate communicator and execute
outside timed device calls. Corresponding rank samples are aggregated by
their maximum, rather than taking the maximum of independently computed
medians. Preparation, lowering, compilation, first dispatch, warm samples,
compiler memory, and process-local allocator evidence remain separate.
Allocator counters describe JAX allocations, not all MPI/UCX native storage.

Run without other tests, builds, or GPU jobs when retaining performance
evidence. The default per-buffer bound is 64 MiB and only two changed inputs
are retained. Effective payload throughput is message bytes divided by
synchronized JAX invocation time, not a claim about physical link bandwidth.

`--nsys` adds separate three-repeat capture runs after the timing campaign.
The capture brackets only warmed communication, excluding input generation
and host-side numerical validation. Trace reports are retained beside the
qualification JSON; profiled timings cannot enter the performance aggregate.
Use a profiler compatible with the selected CUDA runtime. Without a capture,
UCX configuration/protocol logs are transport evidence, not proof of absent
host copies. Neither a CUDA-aware MPI query nor successful collectives
qualify network GPUDirect RDMA or multi-host scaling.

#### Two-A100 results

The CUDA 13 stack passed all eight integration cases on two
A100-SXM4-80GB GPUs with driver 580.159.04. The matrix covers distinct-GPU and
deliberately shared-GPU ranks, direct and explicitly host-staged transport,
float32/float64, repeated changed inputs, supported derivatives and explicit
unsupported-transform refusals, and communicator release/recreation. Real
foreign-major mpi4jax bridges were also refused before package import.

The uncontended sweep covers 8 B, 1 KiB, 64 KiB, 1 MiB, 16 MiB, and 64 MiB
per rank, with 31 warmed samples per point. Representative float64 results
below are medians of corresponding-rank maximum synchronized durations,
including JAX dispatch but excluding control barriers:

| CUDA | Operation | Payload per rank | CUDA-aware MPI | Explicit mpi4jax host staging |
| --- | --- | --- | --- | --- |
| 13 | sendrecv | 64 MiB | 0.863 ms | 125.065 ms |
| 13 | sum | 64 MiB | 170.175 ms | 160.533 ms |

Full per-rank samples, phase timings, compiler evidence, memory observations,
and runtime identities are retained in
`benchmarks/cuda_mpi_cuda13_cuda_direct.json`, and
`benchmarks/cuda_mpi_cuda13_host_staged.json`.
Opaque MPI calls report unknown XLA cost estimates; those raw sentinel values
are retained rather than converted into invented FLOP counts.

Separate Nsight Systems 2026.5.1 captures contain exactly three warmed
64 MiB float64 calls per rank. The CUDA 13 captures show these copy totals:

| Route and operation | Host-to-device per rank | Device-to-host per rank | Peer copies per rank | Local device copies per rank |
| --- | --- | --- | --- | --- |
| CUDA-aware sendrecv | 0 | 0 | 192 MiB | 384 MiB |
| CUDA-aware sum | 192 MiB | 384 MiB | 0 | 0 |
| Explicit host staging, either operation | 192 MiB | 192 MiB | 0 | 0 |

UCX protocol records select CUDA IPC for the large device-buffer exchanges;
the peer traces show GPU 0→1 and GPU 1→0. Thus the captured large sendrecv
calls avoid host copies, but still perform device copies. This finding does
not extend to every message size. SUM still stages internally despite CUDA
awareness, so this MPI stack is not a GPU-resident allreduce qualification or
a blanket performance replacement for native GPU collectives.

#### Four-H100 results

The H100 80GB HBM3 NV18 host passed the eight-case MPI matrix separately at
two and four ranks with driver 580.126.09, JAX/JAXlib 0.11.2, CUDA 13.4.92,
and the same Open MPI/UCX/mpi4py/mpi4jax versions above. This deployment
explicitly sets `NCCL_NVLS_ENABLE=0`; it does not qualify the host's failing
NVLS multicast initialization. Native cuDSS, AmgX, and independent-process
QUDA were also exercised on every GPU, including normal and hostile teardown.
The locally built companion contains sm90 device code, not the complete
release architecture set, and targets manylinux_2_39 rather than manylinux_2_28.

The uncontended 64 MiB float64 MPI results use 31 warmed samples per rank and
the corresponding-rank maximum aggregation described above:

| Ranks | Operation | CUDA-aware MPI | Explicit mpi4jax host staging |
| --- | --- | ---: | ---: |
| 2 | sendrecv | 0.819 ms | 78.301 ms |
| 4 | sendrecv | 0.866 ms | 85.975 ms |
| 2 | sum | 117.407 ms | 111.954 ms |
| 4 | sum | 133.999 ms | 130.797 ms |

Separate four-rank captures show the peer ring 0→1→2→3→0. Across three
warmed calls, each direct-sendrecv rank recorded 192 MiB of peer transfers
and 192 MiB of local device copies, with no recorded host copies. Direct SUM
recorded 384 MiB device-to-host and 192 MiB host-to-device per rank; explicit
host staging recorded 192 MiB in each direction. All sixteen captures also
reported that some CUDA/NVTX events might not have been collected. Positive
peer transfers and SUM staging are observed; absence of host-copy records is
not an unconditional zero-host-copy guarantee.

`benchmarks/h100_qualification.json` retains per-rank measurements, numerical
gates, compiler/memory evidence, profiler summaries and warnings, runtime
identities, and the exact native-artifact qualification scope. These are
single-host results, not RDMA, multi-host, distributed native-solver, or
distributed accelerated-MACE qualification.
