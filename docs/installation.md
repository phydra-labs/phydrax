# Installation

Requires Python 3.13 (the standard GIL-enabled CPython build) and JAX 0.12. Select
CPU dependencies explicitly with:

```bash
pip install "phydrax[cpu]"
```

The `cpu` extra requests `jax[cpu]>=0.12.0,<0.13`. Plain `pip install phydrax`
continues to install the base JAX dependency, which also supports CPU execution.
Neither command removes existing accelerator plugins or forces CPU execution;
set `JAX_PLATFORMS=cpu` before numerical imports to select CPU at runtime.

The default checkout stack selects matching JAX and JAXlib 0.12.0 releases.
The explicitly selected oneAPI extra instead pins the experimental
JAX 0.12.0 / JAXlib 0.11.2 combination described below. On x86-64, upstream
wheels require AVX2 and FMA (Intel Haswell or newer). The NumPy and SciPy
requirements already exceed JAX 0.12's minimums.

Phydrax installs a source-guarded correction for the premature narrowing of wide
indexing addresses observed in JAX 0.12.0, without an additional version pin.
Compatibility is checked against the audited upstream implementation, not its
release number. Import Phydrax
before initializing a JAX backend or tracing numerical functions, and restart
the process after changing either package. The correction preserves public JAX
callables and delegates indexing layout to the original private frontend;
unknown upstream source or a conflicting patch is refused. Its identity is
recorded in installed-wheel and benchmark evidence. Reprepare compiler-bound
artifacts rather than reusing evidence from an unpatched frontend.

GPU and MPI qualification is specific to the recorded software and hardware
stack. The JAX/JAXlib 0.11.2 results below remain historical evidence, not
qualification of 0.12. Rebuild native companions and the site-specific mpi4jax
bridge against the target environment before qualifying a deployment.

## Python 3.13

Phydrax supports CPython 3.13 only; the free-threaded `3.13t` build is not
supported. Create a new environment rather than upgrading a Python 3.12 one in
place, because compiled dependencies are interpreter-specific.

Artifacts written under Python 3.12 restore through their usual validation:
array archives, portable ML models, atomistic model artifacts, and checkpoints
whose identities are declared or content-derived. Identities derived from the
code of a plain Python function hash interpreter bytecode, so they generally
differ under 3.13. This affects, for example, a `ResidualRelaxationMap` built
without explicit operator IDs and plain functions passed as operator-training
`static_callables`. Checkpoints bound to such identities are refused with their
usual identity-mismatch error; reprepare or restart them under 3.13.

## Apple Silicon GPUs

The Apple extra selects the exact wheel-only `phydrax-metal==0.2.2` companion:

```bash
pip install "phydrax[metal]"
```

This release command requires the matching companion to have been published.
For local qualification, build the core and companion wheels, supply their
wheel directory with `--find-links`, and use the installed-artifact tools
documented in `native/metal/README.md`. No package publication is implied by a
local build.

| Boundary | Companion contract |
| --- | --- |
| Host | Apple Silicon, native Darwin arm64; not Rosetta or Intel macOS |
| Python | Standard GIL-enabled CPython 3.13 only |
| JAX / JAXlib | Exactly 0.12.0 / 0.12.0 |
| Deployment target | macOS 14.0 minimum; actual OS/device qualification is recorded separately |
| Runtime | `mps`, provider `phydrax-metal` |
| Numerical profile | Explicit float32/complex64, exact int64/uint64, x64 enabled |
| Topology | One local Apple GPU; no multi-device or multi-host claim |

The extra has no platform marker that silently removes its companion on an
unsupported host. A missing compatible wheel is an installation failure, not a
successful CPU-only Metal install. Published wheels bundle their native library,
Metal library, and licenses; ordinary installation does not require LLVM, CMake,
Ninja, Homebrew, or an Xcode source build.

Use `JAX_PLATFORMS=mps JAX_ENABLE_X64=1` before launching the program, or configure
`RuntimeConfig(platforms=("mps",), enable_x64=True)` before numerical imports.
Keep x64 enabled for exact indices; explicitly choose float32 model parameters,
geometry, state, and coefficients. Float64/complex128 arrays and computations
are refused, not silently narrowed. CPU execution remains available in a
separate process selected with `JAX_PLATFORMS=cpu`.

`phydrax.backends.metal_environment(initialize_backend=False)` inspects package,
native asset, source/build, and JAX identity without initializing a device.
The default `metal_environment()` additionally probes MPS. A successful unprobed
report is not hardware qualification. Missing assets, mismatched versions,
unsupported hosts, competing plugin entry points, and runtime identity mismatch
retain their own diagnostic issues. Use a fresh environment without other Apple
JAX plugins rather than relying on plugin discovery order; restart after fixing
an installation.

The catalog's Metal profiles remain candidates. Follow the
[Apple GPU execution guide](guides_metal_execution.md) for the scientific scope,
installed-wheel evidence, performance methodology, and unsupported operations.
CUDA providers and Pallas CUDA kernels are not included in the Metal companion.

## Intel GPUs and Cloud TPUs

```bash
pip install "phydrax[oneapi]"  # Linux x86-64; Intel GPU driver required
pip install "phydrax[tpu]"     # Linux x86-64 TPU VM; glibc >= 2.31
```

These extras use upstream binaries, not Phydrax-owned Intel or TPU companions.
The provider requirement is unconditional: unsupported hosts must fail
resolution instead of silently receiving an empty accelerator extra.

| Extra | Exact selected stack | Boundary |
| --- | --- | --- |
| `oneapi` | JAX 0.12.0, JAXlib 0.11.2, oneAPI plugin/PJRT 0.11.2 | JAX's minimum-JAXlib combination; not Intel's documented all-0.11.2 release |
| `tpu` | JAX/JAXlib 0.12.0, libtpu 0.0.50 | Upstream TPU runtime; requires actual TPU hardware |

The oneAPI plugin installs its pinned user-space SDK through `with-oneapi`.
The Intel GPU compute driver remains a system prerequisite. Its JAXlib pin
conflicts with CUDA13, Metal, and TPU: use a separate environment rather than
relaxing metadata or bypassing version checks. Native Windows, WSL, macOS, and
Linux ARM are not supported Intel targets for this release.

Select `JAX_PLATFORMS=oneapi` or `JAX_PLATFORMS=tpu` before numerical work.
Host-only diagnostics are available through
`phydrax.backends.oneapi_environment()` and `tpu_environment()`. Their default
does not initialize devices. `initialize_backend=True` explicitly observes the
selected runtime; a successful report is still not scientific qualification.

Keep x64 enabled for exact indices and choose each numerical role's dtype
explicitly. Neither the x64 flag nor a vendor name proves FP64 support. TPU's
`highest` matmul policy uses multiple BF16 passes, not native IEEE float32
matrix multiplication. For multi-host TPU execution, configure the existing
`RuntimeBootstrap(autodetect=True)` on every process before device discovery.
Installation does not provision cloud resources.

Intel and TPU hardware qualification remains pending. Read the
[Intel guide](guides_oneapi_execution.md) and [TPU guide](guides_tpu_execution.md)
for precise support boundaries and the installed-artifact workflow.
The [retained software evidence](data/accelerator_installation.json) distinguishes
dependency resolution, CPU compatibility, refusal behavior, and the actual
Metal shared-source regression from unperformed Intel/TPU execution.

## AMD GPUs

On native Linux x86-64 with standard CPython 3.13:

```bash
pip install "phydrax[rocm10]"
```

This extra selects JAX/JAXlib and the matching AMD plugin/PJRT 0.12.0 stack
plus the packaged ROCm 10.1.0 userspace runtime. Its `gfx942`, `gfx950`, and
`gfx1250` payloads target the MI300/MI350 families and the published MI400/CDNA 5
architecture, rather than a two-model allowlist. Installation does not detect the
GPU or require a separate architecture-specific command.
A compatible AMD kernel driver, supported OS, and device permissions remain
system prerequisites. The unconditional runtime requirements refuse unsupported
wheel platforms rather than silently producing a CPU-only accelerator install.

The runtime comprises `phydrax-rocm-runtime`, `phydrax-rocm-core`,
`phydrax-rocm-libraries`, `phydrax-rocm-device-gfx1250`,
`phydrax-rocm-device-gfx942`, and `phydrax-rocm-device-gfx950`.
The component versions identify the upstream SDK release. Payload provenance and
Phydrax packaging identity remain distinct.
The loader binds the packaged native dependency closure and compiler resources
automatically; users do not set `LD_LIBRARY_PATH`, `ROCM_PATH`, or `LLVM_PATH`.
Foreign already-loaded runtimes and overlapping SDK distributions are refused,
not silently mixed with the selected bundle.

[AMD's GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
map MI300A, MI300X, and MI325X to `gfx942`, and MI350P, MI350X, and MI355X to
`gfx950`. The release-pinned
[AMD target definitions](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/cmake/therock_amdgpu_targets.cmake)
map MI450, MI450X, and MI455X to `gfx1250`. These are documented examples,
not an exhaustive model-name gate.

MI400 support is provisional: AMD's
[readiness table](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
marks CDNA 5 build-passing and sanity-tested, not release-ready. Published
`gfx1250` assets are included, but unpublished targets and individual future
SKUs are not inferred from a marketing family name. AMD lists MI430X availability
in 2027; no separate compiler-target mapping or Phydrax hardware qualification
is asserted for that model. Physical testing here covers MI300X only.

These release commands require the runtime wheels to be published. Local builds
do not imply PyPI publication. Build the core and runtime wheelhouse locally:

```bash
python tools/build_rocm_wheels.py --download \
  --sources .tmp/rocm-upstream --wheelhouse .tmp/rocm-wheelhouse
uv build --wheel --out-dir .tmp/rocm-wheelhouse
uv pip install --find-links .tmp/rocm-wheelhouse "phydrax[rocm10]"
```

The runtime builder uses pinned upstream URLs and SHA256 identities. Normal
installation uses ordinary wheel dependencies, not AMD indexes, source builds,
or downloads triggered at import time. Large payload wheels require appropriate
publication quotas. Source notices and redistribution obligations must be
reviewed before publication; numerical qualification is not legal authorization.

The runtime platform is `rocm`, not the extra's name. For host callbacks,
including compiled error reporting, use `JAX_PLATFORMS=rocm,cpu`. CPU registration
supports those callbacks; qualification requires numerical results on the
explicitly selected AMD device and refuses CPU substitution. Keep x64 enabled
and choose scientific dtypes at their owners; there is no automatic downcast.

`phydrax.backends.rocm_environment()` inspects installation evidence without
initializing devices. Set `initialize_backend=True` for explicit runtime
observation. The raw HIP runtime version is retained separately from SDK release
identity: ROCm 10.1.0 can report `rocm 71626385`.

Native JAX numerical workflows and `JaxCollectiveProvider` share the existing
execution substrate. For one host, start with one process managing the selected
devices. Two processes with two devices and four processes with one device are
separate process layouts, not scientific mesh shapes. Use
`accelerator_vendor="amd"` for resource admission: inventory platforms preserve
JAX's raw `"gpu"` representation and existing identity. Every distributed rank
must enter collective inventory discovery in the same sequence.

```bash
python -m tools.distributed_substrate_benchmark \
  --local-processes 2 --devices-per-process 2 \
  --expected-vendor amd --expected-global-devices 4 --expected-local-devices 2
```

The CUDA companions, compiled CUDA MACE/FMM kernels, and CUDA-aware mpi4jax
route are not AMD capabilities. Use native-JAX scientific routes explicitly.
Candidate AMD profiles do not authorize release, imply multi-host qualification,
or promise speedup.

The retained external-SDK MI300X records in
`benchmarks/mi300x_rocm10_installed.json` and `benchmarks/mi300x_workflows.json`
document their original installation and runtime identities. They are not
relabeled as proof of bundled installation.

`benchmarks/mi300x_rocm10_bundled.json` records a fresh installed-wheel run with
all three architecture payloads and no manual SDK paths: all 20 scientific cases
and the advertised example pass separately on each of four MI300X devices.
`packaging/rocm/architecture-coverage.json` records the pinned vendor mapping and
payload evidence for the wider families. Non-MI300X payload coverage is not
hardware qualification. Runtime wheels remain unpublished; source-license review
and package-index large-file quotas are separate publication prerequisites.

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

The CUDA extra pins JAX, JAXlib, and the CUDA plugin to 0.12.0. Site collective
programs use an exact source-verified private dispatch boundary from that release;
a later 0.12.x installation is not admitted merely because its public version
range would otherwise be compatible. The base and CPU extras retain their
documented JAX ranges.

On Linux x86-64 and aarch64, the extra also requires the lockstep wheel-only
companion `phydrax-cuda13==0.2.2`:

| Component | CUDA 13 |
| --- | --- |
| cuDSS sparse LDLᵀ | Included |
| AmgX algebraic multigrid | Included |
| QUDA lattice kernels | Included |

Release companions contain CUDA 13 SASS for sm 8.0, 8.6, 8.9, 9.0, 10.0,
and 12.0, plus PTX 12.0.
Installation requires a matching host-architecture companion wheel; a missing
wheel is an installation failure, not a successful JAX-only CUDA-extra install.
Local builds do not imply publication. GPU-direct
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

### Optional numerical CUDA libraries

The CUDA extra keeps the core cuDSS, AmgX, and QUDA dependency closure. Additional
implementations are selected explicitly; installing a library never changes a
native execution default:

| Installation | Native owner and scope |
| --- | --- |
| `phydrax[cuda13,cutensor]` | Device-local prepared tensor contractions |
| `phydrax[cuda13,cutensornet]` | Fixed-path prepared contractions and separately requested shape-only path search |
| `phydrax[cuda13,cuest]` | Nonperiodic, all-electron, real-FP64 density-fitted RHF and analytic first nuclear forces |

The small tensor/cuEST wrapper libraries belong to the lockstep CUDA companion.
cuTENSOR and cuTensorNet runtimes come from their separately pinned CUDA13 wheels.
The cuEST C runtime is a separate optional `phydrax-cuest==0.2.2` wheel built from
the checksum-pinned NVIDIA redistributable, with its license and provenance.
The vendor Python bindings alone do not provide this C-runtime contract.
An unpublished local runtime wheel must be supplied through the same wheelhouse;
building it does not publish it.

```sh
uv sync --extra cuda13 --extra cutensor --extra cutensornet --extra cuest
```

See [tensor execution](guides_tensor_platform.md) and
[molecular electronic structure](guides_computational_chemistry.md) for explicit
preparation and selection. Ordinary `phydrax.ein.contract`, matrix multiplication,
native tensor symmetry routes, and native spectral pencils remain unchanged.
Missing libraries, unsupported transformations, and insufficient resource bounds
are refusals, not implicit CPU or native-provider fallbacks.

The `collective`, `cudss_mgmn`, and `cudecomp` components are separate **site-bound**
builds, not universal MPI wheels. They require one checksum-identified,
CUDA-aware Open MPI stack, the matching communication shim, and supervised
same-host processes. MPI availability alone does not establish device-buffer
integrity, multi-host support, RDMA, or zero host copies.
The native lease also splits the participating communicator by
`MPI_COMM_TYPE_SHARED` and requires every participant to belong to that one
shared-memory domain before publishing a live lease. Local process supervision
does not substitute for this MPI-owned same-host proof.

For local development, `PHYDRAX_CUDA_CUTENSOR_LIBRARY`,
`PHYDRAX_CUDA_CUTENSORNET_LIBRARY`, and `PHYDRAX_CUDA_CUEST_LIBRARY` select
wrapper artifacts through the same strict loader. Site builds use
`PHYDRAX_CUDA_COLLECTIVE_LIBRARY`, `PHYDRAX_CUDA_CUDSS_MGMN_LIBRARY`, and
`PHYDRAX_CUDA_CUDECOMP_LIBRARY`. None bypasses dependency, release, component,
handler, or site-identity validation.

Installed-wheel qualification accepts `--libraries cutensor cutensornet cuest`.
It runs each selected local mathematical profile on every requested physical GPU,
requires all declared precision/scenario coverage, and tests actual hostile
library paths. Tensor slice placement is a separate, explicitly host-orchestrated
profile. Required skips, missing cases, duplicate evidence, failed numerical
results, and timeouts prevent a successful qualification.

Physical execution qualification separately requires every directed GPU pair to
report direct peer capability and pass changed-content CUDA/JAX copies. A second,
vendor-free JAX gate exercises 1, 3, 4, 10, and 256 `int32` words through
ordinary placement, forced copy, and an explicit host control, checking both
destination arithmetic and host readback. Host-control success diagnoses a
failure; it never qualifies an unavailable or corrupt peer route.

### Eight-A40 host admission

A40 is compute capability 8.6. Use an inspected `86-real` developer artifact or
a release artifact containing compatible inspected device code. The installed
driver and the library's own architecture constraints remain separate checks.

The initial eight-A40 observation failed: outgoing transfers from physical GPUs
1 and 2 corrupted contents in both CUDA and JAX despite correct source-local
readback. GPU1 subsequently reported a reset requirement and disappeared from
CUDA enumeration. Those failed observations are retained.

A later read-only discovery again found all eight devices. All 1,008 directed
CUDA/JAX copy observations then passed, followed by the complete
1×1/1×2/1×4/1×8/2×4/4×2/8×1 physical execution matrix. Same-host cuDSS MG/MGMN
and cuDecomp also passed on two, four, and eight GPUs. The agent made no driver,
PCIe, IOMMU, ACS, reboot, or GPU-reset changes; the cause of the host's recovery
was not established. `benchmarks/a40_qualification.json` separates this hardware
history, installed-provider qualification, and bounded performance observations.

Those passing snapshots do **not** establish current general peer safety.
Subsequent cuTENSOR host-slicing checks lost GPU1's complete contribution.
A Phydrax/vendor-free JAX program then reproduced incorrect peer and forced-copy
GPU0-to-GPU1 transfers at five sizes, while explicit host transport passed.
The underlying driver/JAX/hardware cause remains unproved. At the user's request,
debugging stopped with the peer-dependent route **unqualified**; no host-staging
cutover, silent fallback, or host-setting change was added.

Do not turn a successful one-device workflow or independent host-staged work into
an eight-device collective claim. Run the explicit physical campaign in
`tools.qualify_cuda_execution` before using a host for coupled GPU execution, and
preserve failed evidence rather than narrowing the requested device set silently.
Historical H100/H200 evidence used a different JAX stack; single-B300 evidence
does not qualify this A40 topology. Compiled Pallas MACE still refuses sm86;
native-JAX execution is a separate route, not an accelerated fallback.


### Eight-A100 NVLink deployment profile

The current same-host profile uses eight NVIDIA A100-SXM4-80GB devices
(compute capability 8.0), driver 595.91.07, CPython 3.13.16, JAX/JAXlib and the
CUDA plugin 0.12.0, and wheel-provided CUDA runtime 13.4.92. Every GPU pair is
connected by the reported bonded `NV12` topology. The audited
`manylinux_2_39_x86_64` companion contains only sm80 AmgX/QUDA SASS and has
build hash
`82f862503cfbfb5485ef0b7ffe623c8499e9c8bf08d2dceb8a2d53b0f9f01e3d`;
cuDSS and the tensor/cuEST wrappers call separately installed vendor runtimes.

All 1,008 changed-content CUDA/JAX directed-copy observations pass with direct
peer capability reported for every ordered pair. All 2,520 tiny-buffer
peer-placement, forced-copy, and host-control consumer/readback observations
also pass. The native JAX weighted-reduction/PCG matrix passes at
1×1, 1×2, 1×4, 1×8, 2×4, 4×2, and 8×1 process/device layouts without a
transport override.

The fresh installed-wheel campaign passes all required device-local cuDSS,
AmgX, and QUDA numerical/lifecycle cases on every device; optional cuTENSOR,
cuTensorNet, and cuEST profiles, including multi-GPU host slices, pass their
declared precision and scientific matrices. Clean and hostile-library-path
audits load only owned CUDA 13 libraries. Missing optional Grid/qlat providers
remain skips, not QUDA failures. The separate publication resolver gate remains
incomplete because this x86-64 campaign did not build the required aarch64
companion; it is not relabeled as release-architecture qualification.

Single-process cuDSS MG passes eighteen float32/float64 positive-definite and
indefinite lifecycle, true-multiple-RHS, transpose, derivative, failure, and
retirement cases across 2, 4, and 8 GPUs. A source-pinned CUDA-aware Open MPI
5.0.11 site build, cuDSS Open MPI communication shim, and cuDecomp 0.7.0 commit
`73121159fec42eb9144dfed9159e53fde6603fca` pass 21 MGMN cases across 2, 4,
and 8 one-GPU ranks plus 16 cuDecomp cases. Those cases cover both precisions,
composed histories, derivatives, resource refusal, closed-lease behavior, and
coupled failure without orphan ranks. These are same-host site-artifact
results, not portable-wheel or multi-host claims.

The separate mpi4jax 0.9.1.post2 CUDA-direct campaign is **not qualified** on
JAX 0.12.0: upstream declares JAX 0.11.2 as its latest supported release and all
four direct integration cases terminate with a native segmentation fault. The
four explicitly host-staged cases pass, but the campaign stops before timing and
cannot be promoted. This does not invalidate the independently owned native
collective/MGMN/cuDecomp routes above.

Compiled sm80 MACE passes all four precision/accumulation value, derivative,
and resource cases; the distributed MACE matrix passes 29 scenarios. Nineteen
current FMM, pencil-spectral, and eight-owner LES cases pass, as do the
local-guarded PSATD/restart pair and the genuinely local 2×2×2 PIC
science/scaling pair with every owner populated.

Bounded, uncontended measurements retain preparation, lowering, compilation,
first-call, warm, derivative, transfer, and memory scopes separately. At the
16- and 32-axis tensor points all five path/executor combinations pass; exact
warm samples remain in the qualification evidence, and no blanket tensor
speedup is claimed.
The 2×2×2 PIC campaign passes local-width 6 and 8 cases against its one-device
reference, but XLA reports involuntary full rematerialization for several
sharding transitions; that remains a compiler-cost limitation, not a hidden
success claim. The cuEST campaign completes six s/p/d capacity points spanning
4–80 orbital and 10–1,000 auxiliary functions against independently matched
density-fitting references. These bounded records do not extrapolate to
unmeasured chemistry, graph sizes, or general application speedup.

These observations resolve the prior A40 host-slice blocker only for this
A100/source/artifact/runtime tuple. They do not prove the A40 host recovered,
multi-host execution, network RDMA, arbitrary workloads, or release-wide
architecture coverage.
Exact artifact hashes, support scopes, successful and failed gates, scientific
nodes, and phase-separated records are retained in
[`a100_qualification.json`](https://github.com/phydra-labs/phydrax/blob/dev/benchmarks/a100_qualification.json).


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
| `no-cuda-plugin` | A CUDA marker is installed without a JAX CUDA plugin. Reinstall the selected extra on supported Linux; use the CPU install on unsupported platforms. This is an error on Linux x86-64 and aarch64, and a warning elsewhere. |
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
| `companion-missing` | Install the matching CUDA extra on Linux x86-64 or aarch64, or supply a development library override. |
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

## DGX Spark deployment profile

The qualified single-device profile is Linux aarch64, NVIDIA GB10 (compute
capability 12.1), driver 580.159.03, CPython 3.13.14, JAX/JAXlib and CUDA plugin
0.12.0, and wheel-provided CUDA runtime 13.4.92. The installed driver reports
CUDA 13.0; the recorded workload checks passed without a driver upgrade.
The system toolkit is not used by the native build or selected as a runtime
library search path.

Build the companion natively on aarch64 with `--architectures 121-real`, using
the build environment and audit procedure in `native/cuda/README.md`. The local
wheel is `manylinux_2_39_aarch64`, with inspected sm121 AmgX/QUDA device code;
cuDSS kernels remain owned by its separately installed NVIDIA provider.
This does not qualify the full release architecture set or a manylinux_2_28
container build.

Keep the audited local wheel when installing from the checkout:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python --no-sources \
  --find-links dist/cuda13-sm121 -e ".[cuda13]"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu \
  JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m tools.distributed_substrate_benchmark \
  --size 17 --iterations 1
```

The weighted mean and PCG smoke agree with their references. Installed-wheel
cuDSS, AmgX, and QUDA qualification covers numerical contracts, bounded
release/rebuild behavior, and normal/hostile-path process teardown on device 0.
The clean audit reports no CUDA issues; the hostile audit retains the expected
`library-path-shadowing` warning while using wheel-provided CUDA libraries.
Missing optional Grid/qlat providers and tests requiring a second GPU are not
single-device native-provider failures.

GB10 is separately admitted for MACE's Mosaic GPU coupling in float32/float64
with deterministic/compensated accumulation, coefficient and field derivatives
through order two, and resource/third-order refusal checks. FMM's optional
Triton route is distinct: its forward kernel runs on CUDA, while its custom
derivative rule executes the JAX reference. The pinned JAX release warns that
the Pallas Triton backend is deprecated; this qualification is not a migration
to another lowering.

Keep x64 and scientific precision choices explicit; do not apply the Metal
float32-only policy to CUDA or enable TF32 implicitly. Spark's shared physical
memory is not independent host RAM plus GPU RAM. Allocator limits, live device
arrays, host RSS, and compiler estimates retain their separate scopes.
No multi-GPU, multi-host, RDMA, cross-device bitwise-replay, or general speedup
claim follows from these checks. Exact evidence and benchmark scopes are in
[`spark_qualification.json`](https://github.com/phydra-labs/phydrax/blob/dev/benchmarks/spark_qualification.json).
The checkout benchmark identity requires the local, git-ignored `uv.lock`;
generate it with `uv lock` before running the recorded benchmark commands.

For checkout verification on shared-memory Spark hosts, hide CUDA from CPU-only
test workers with `CUDA_VISIBLE_DEVICES=""`, keep
`XLA_PYTHON_CLIENT_PREALLOCATE=false` in their inherited environment, and bound
worker count. A subprocess that clears `JAX_PLATFORMS` can otherwise select
CUDA and reserve a large fraction of shared physical memory. Run GPU
qualification separately with one visible device; do not overlap it with
parallel test workers or the repository typing gate. Transient host cgroup
memory limits provide an additional containment boundary, not a GPU-memory
accounting guarantee.

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
evidence specific to that recorded stack. The exercised native-JAX workflows also
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

## Single-B300 deployment profile

The single NVIDIA B300 SXM6 AC profile uses Linux x86-64, compute capability
10.3, driver 595.91.07, CPython 3.13.15, JAX/JAXlib and CUDA plugin 0.12.0,
and wheel-provided CUDA runtime 13.4.92. A fresh environment avoids the host's
Python 3.12/CUDA 12 packages. Select one GPU and retain CPU reference support:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu \
  JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m tools.distributed_substrate_benchmark \
  --size 17 --iterations 1
```

No NCCL or UCX workaround is selected. The substrate smoke passes weighted
reduction, resource admission, and the PCG reference solve on one physical GPU.
This does not exercise peer transfers or multi-GPU communication.

The local companion uses `--architectures 103-real` and contains inspected sm103
AmgX/QUDA device code. cuDSS kernels belong to the separately installed NVIDIA
provider. The artifact is `manylinux_2_39_x86_64`, not a full release-target or
manylinux_2_28 qualification. Keep the audited artifact when installing locally:

```bash
uv pip install --python .venv/bin/python --no-sources \
  --find-links dist/cuda13-sm103 -e ".[cuda13]"
```

Installed-wheel qualification passes all required provider and teardown coverage
on GPU 0, including hostile-library-path diagnostics. Its 116 passing tests have
11 explicit skips: three require a second GPU and eight require unrelated
optional GPT/Qlat providers. Failed cuDSS refactorization and AmgX resetup now
retire partial native state; original preparations rebuild their original
coefficients rather than reusing corrupted factors or hierarchies.

Compiled MACE admits B300 with float32/float64 deterministic/compensated forward,
operand/coefficient derivatives through order two, and resource/order refusals.
Compiled FMM, ordered sparse adjoints, native-JAX MACE periodic derivatives and
fresh-process restart, PIC workflow, and selected dense linear algebra also pass.
FMM's Pallas near-field primal retains its native-JAX derivative rule and the
upstream Triton deprecation warning. Exact fresh-process MACE replay uses its
explicit fixed compiler profile, not an automatic deployment default.

The uncontended campaign passes both-precision spectral gates, single-device
reduced PIC physics checks, FMM phase checks at 32/64 particles, and 80 MACE
route/transform agreement checks. Representative float64 warmed medians use three
samples; preparation, lowering, compilation, and memory are recorded separately:

| Workload | Reference route | Reference median (ms) | Accelerated median (ms) |
| --- | --- | ---: | ---: |
| Prepared order-three FMM, 64 particles | Native JAX | 8.406 | 6.891 |
| MACE coupling energy/forces, 16 nodes | Dense native JAX | 0.154 | 4.107 |
| One-interaction MACE energy/forces, 8 atoms | Exact trainable model | 92.884 | 91.757 |

The model comparison uses the trainable accelerated route, not frozen inference.
These bounded measurements expose the tiny dense-coupling advantage and do not
establish a general speedup or large-model scaling result.

[`b300_qualification.json`](https://github.com/phydra-labs/phydrax/blob/dev/benchmarks/b300_qualification.json)
retains source/artifact identities, failure reproductions, numerical gates, and
phase-separated bounded benchmarks. Single-device evidence does not establish
general acceleration, multi-GPU/MPI execution, network RDMA, multi-host scaling,
or cross-device bitwise continuation. No wheel publication is implied.

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
