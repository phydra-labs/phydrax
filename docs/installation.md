# Installation

Requires Python 3.12. The base install uses CPU JAX:

```bash
pip install phydrax
```

## NVIDIA GPUs

On Linux, choose exactly one CUDA extra. `uv pip install` or `uv add` can be
used instead of `pip install`.

| Install | Minimum driver-supported CUDA version | Driver family |
| --- | --- | --- |
| `pip install "phydrax[cuda12]"` | 12.0 | R525+ |
| `pip install "phydrax[cuda13]"` | 13.0 | R580+ |

The driver must report at least the listed CUDA version. Pip cannot install or
upgrade the NVIDIA driver. Each extra installs JAX's CUDA plugin, NVIDIA runtime
and math-library wheels (including NCCL), and the metadata-only
`phydrax-cuda-major` exclusivity marker. Installing both extras is unsatisfiable
in pip and uv. A system CUDA toolkit and `LD_LIBRARY_PATH` are not required for
this JAX setup. The extras do not install a CUDA backend on non-Linux platforms.

On Linux x86-64, the extras also install the matching wheel-only companion
(`phydrax-cuda12==0.2.2` or `phydrax-cuda13==0.2.2`):

| Component | CUDA 12 | CUDA 13 |
| --- | --- | --- |
| cuDSS sparse LDLᵀ | Included | Included |
| AmgX algebraic multigrid | Included | Included |
| QUDA lattice kernels | Included | Included |

Release companions contain CUDA 13 SASS for sm 8.0, 8.6, 8.9, 9.0, 10.0,
and 12.0, plus PTX 12.0. CUDA 12 additionally contains SASS for sm 7.0 and 7.5.
Linux aarch64 receives the JAX extra but has no companion yet. GPU-direct
mpi4jax still requires a matching user-provided build. See
[External solver backends](api/backends.md) for component contracts.

For development, set `PHYDRAX_CUDA_CUDSS_LIBRARY`,
`PHYDRAX_CUDA_AMGX_LIBRARY`, or `PHYDRAX_CUDA_QUDA_LIBRARY` to the absolute path
of a built `libphydrax_cuda_<component>.so` before first use. These overrides
retain component, release-version, CUDA-major, and handler validation. NVIDIA
dependencies are preloaded by absolute path from installed NVIDIA wheels.

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
| `no-cuda-plugin` | A CUDA marker is installed without a JAX CUDA plugin. Reinstall the selected extra on supported Linux; use the CPU install on unsupported platforms. This is an error on Linux x86-64 and a warning elsewhere. |
| `duplicate-cuda-plugins` | Multiple CUDA plugins are installed. Remove the other-major plugin and its wheels, or create a fresh environment. |
| `plugin-major-mismatch` | The exclusivity marker and unique plugin disagree. Reinstall one chosen extra after removing the old major. |
| `foreign-nvidia-distributions` | Other-major NVIDIA wheels remain installed. Run the exact `pip uninstall` list in the message, then reinstall the chosen extra. |
| `driver-too-old` | The driver's supported CUDA major is below the plugin's. Upgrade the driver or select the compatible extra named in the message. |
| `cpu-platform-selected` | Configuration deliberately excludes CUDA despite an installed plugin. Keep this warning for CPU workers, or select CUDA before startup; see [Configuration](api/configuration.md#phases-and-startup). |
| `backend-unavailable` | A device probe found no usable CUDA backend. Inspect the backend error in the message (`facts.backend_error`), GPU visibility, driver, and any listed library-path directories. Remove shadowing CUDA paths and restart. |
| `library-path-shadowing` | `LD_LIBRARY_PATH` contains CUDA libraries that can override wheel libraries. Remove those entries and restart; a currently working backend does not make this safe. |
| `mixed-cuda-majors` | Mapped libraries have more than one known CUDA major. Remove incompatible providers or library paths, rebuild against the chosen major, and restart in a clean process. |
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

### Switching CUDA majors

A fresh virtual environment is the simplest way to switch. When reusing an
environment, uninstall the old `jax-cuda12-plugin` or `jax-cuda13-plugin` and its
matching `jax-cuda12-pjrt` or `jax-cuda13-pjrt` distribution and old
`phydrax-cuda12` or `phydrax-cuda13` companion, then remove
other-major NVIDIA wheels before installing the new extra. Installing the new
extra alone does not remove old dependencies. `foreign-nvidia-distributions`
lists the NVIDIA distributions to uninstall, including unsuffixed CUDA 13
packages. Rebuild user-provided native providers for the new major and restart
Python; installed-package changes cannot unload libraries already in memory.

### Library-path shadowing

Do not add a system toolkit's library directory to `LD_LIBRARY_PATH` for the
wheel-based JAX setup. Such paths can shadow NVIDIA wheel libraries, including
libraries whose sonames are shared across CUDA majors. This produces
`library-path-shadowing` when no backend-unavailable issue is present. It can
also prevent the CUDA 12 plugin from initializing, producing
`backend-unavailable` instead, with the backend error and affected directories.
Remove the CUDA entries (or unset `LD_LIBRARY_PATH` if it has no other deployment
purpose) before starting a new Python process.

## Multi-GPU and multi-node execution

Use JAX global arrays and NCCL for NVIDIA multi-GPU communication, with
`JaxCollectiveProvider` for named-axis operations. Startup automatically
coordinates multi-process `srun` and `mpirun` launches through JAX's launcher
detection. Rank settings must exist before `import phydrax`. See
[Bootstrap before device initialization](api/execution.md#bootstrap-before-device-initialization)
and [Configuration startup](api/configuration.md#phases-and-startup) for explicit
coordinator settings and supported launcher environments. MPI is a launcher or
an optional interoperability route, not a requirement for JAX/NCCL collectives.

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
export CUDA_ROOT=/path/to/cuda13-toolkit-view
# Replace <venv> with the environment's site-packages directory containing nvidia.
export LDFLAGS="-Wl,-rpath,<venv>/nvidia/cu13/lib"
uv pip install --no-cache --no-build-isolation --reinstall mpi4jax
```

Install mpi4py against the same MPI and provide mpi4jax's build dependencies
before using `--no-build-isolation`. Set `MPI4JAX_USE_CUDA_MPI=1` before importing
mpi4jax for GPU-direct transport. A CUDA-enabled bridge is checked before import
even for host-staged use, because mpi4jax loads that bridge unconditionally.
See [Distributed and vendor execution capabilities](api/backends.md#distributed-and-vendor-execution-capabilities)
for communicator ownership, direct-transport qualification, and deployment limits.
