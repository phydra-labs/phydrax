# Experimental Intel GPU execution with oneAPI

`phydrax[oneapi]` uses the upstream JAX oneAPI plugin and PJRT runtime. It does
not build or ship a Phydrax-owned Intel runtime. Scientific programs continue to
use canonical JAX transformations and Phydrax's existing execution groups and
sharding; there is no separate Intel array API or kernel backend.

**Status:** software integration is experimental; Intel GPU hardware and
scientific qualification are pending. A successful dependency resolution, CPU
smoke test, host assessment, or backend initialization is not GPU qualification.
No CPU execution is counted as Intel GPU evidence.

## Installation and release boundary

Use a fresh **native Linux x86_64 standard-GIL CPython 3.13** environment:

```bash
python3.13 -m venv /tmp/phydrax-oneapi
source /tmp/phydrax-oneapi/bin/activate
python -m pip install --upgrade pip
python -m pip install 'phydrax[oneapi]'
```

The selected stack is intentionally exact:

| Distribution | Version |
| --- | --- |
| `jax` | `0.12.0` |
| `jaxlib` | `0.11.2` |
| `jax-oneapi-plugin[with-oneapi]` | `0.11.2` |
| `jax-oneapi-pjrt` | `0.11.2` |

This **mixed frontend stack is not Intel's documented matched release**.
[Intel's 0.11.2 release notes](https://github.com/Intel-tensorflow/jax/releases/tag/oneapi-jax-v0.11.2-alpha)
match JAX, jaxlib, and the oneAPI plugin/PJRT wheels at 0.11.2. Phydrax retains
its JAX 0.12.0 frontend and therefore keeps an explicit qualification warning,
including when host checks pass. Do not replace the pinned jaxlib with 0.12.0 or
silently bypass compatibility validation.

The upstream wheels target Linux x86_64, glibc 2.27 or newer, and CPython
3.12–3.15, but **Phydrax requires standard-GIL CPython 3.13 only**. The vendor's
broader wheel range does not expand the Phydrax interpreter contract.
Diagnostics record `python_free_threaded` from the canonical
`sysconfig.get_config_var("Py_GIL_DISABLED")` build fact and refuse free-threaded
builds. Missing Python implementation, version, GIL-build, or libc evidence
also refuses admission rather than treating an unknown host as compatible.
There are no macOS, Windows, or ARM oneAPI wheel lanes. **WSL is unsupported**
for this experimental JAX integration. Linux wheel platform tags alone cannot
distinguish WSL from native Linux. Diagnostics additionally record the kernel
release and check WSL environment evidence; this is host evidence, not a claim
that wheel metadata detects WSL or that every virtualization configuration can
be identified.

Do not combine this environment with Phydrax's CUDA13, Metal, or TPU extras.
They require different jaxlib/runtime sets. Keep other oneAPI-based frameworks,
including PyTorch, in separate environments when their dependencies conflict.
The archived `intel-extension-for-openxla` stack is not used.

### Runtime libraries and the external GPU driver

`with-oneapi` installs the SDK runtime dependencies; a full locally installed
SDK is not required for the wheel route. The selected upstream release uses
**oneAPI 2025.1: DPC++ 2025.1.1 and oneMKL 2025.1.0**. Its exact dependencies are:

- `dpcpp-cpp-rt`, `intel-sycl-rt`, `intel-cmplr-lib-rt`,
  `intel-cmplr-lib-ur`, `intel-cmplr-lic-rt`, `intel-opencl-rt`, and
  `intel-openmp`: **2025.1.1**;
- `mkl`, `onemkl-sycl-blas`, `onemkl-sycl-dft`, `onemkl-sycl-lapack`,
  `onemkl-sycl-rng`, and `onemkl-sycl-sparse`: **2025.1.0**;
- `tbb`: **2022.1.0**, `impi-rt`: **2021.15.0**, `intel-pti`: **0.12.0**,
  `umf`: **0.10.0**, and `tcmlib`: **1.3.0**.

The **Intel Graphics Compute Runtime driver is external**: pip does not install
it. Intel's release validation boundary is **26.31.39395.13 (OMIX 0.4.0)**, not
a generic promise that any installed Intel driver works. Follow Intel's
[compute runtime installation instructions](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)
or the release's driver-container instructions. The upstream validated devices
are Intel Arc Pro B50/B60/B65/B70 and Intel Data Center GPU Max Series; this is
upstream validation, not Phydrax hardware qualification.

Host metadata cannot establish that a compatible driver is installed, that a
GPU is visible, or which shared libraries the loader ultimately uses. Missing
plugin distributions are installation evidence; missing drivers and absent GPUs
are only reported when an explicit runtime probe actually encounters them.
Diagnostics preserve SDK and loader/device-selection overrides, including
`ONEAPI_ROOT`, `CMPLR_ROOT`, `MKLROOT`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
`ONEAPI_DEVICE_SELECTOR`, and `SYCL_DEVICE_FILTER`. Their presence produces a
warning, not an invented missing-driver diagnosis. Sourcing an unrelated SDK or
preloading another oneMKL/SYCL build can change the runtime despite correct
wheel metadata.

## Host diagnostics and explicit runtime selection

The public diagnostics live in `phydrax.backends`:

```python
from phydrax.backends import (
    assess_oneapi_environment,
    collect_oneapi_environment,
    oneapi_environment,
)

facts = collect_oneapi_environment()
report = assess_oneapi_environment(facts)
# Equivalent convenient host inspection; initialize_backend=False is the default.
report = oneapi_environment()
for issue in report.issues:
    print(issue.severity, issue.code, issue.message)
```

The collector reads host facts, installed distribution versions and provider
library paths, JAX platform selection, and relevant environment variables.
The assessor is pure: constructing synthetic facts and assessing them performs
no host inspection. Neither host API calls `jax.devices`, initializes a provider
client, imports the oneAPI plugin, or loads its libraries. Ordinary JAX/package
imports may retain their existing startup effects; device-inert diagnostics do
not promise that importing JAX itself has no side effects.

`report.successful` means **the assessed checks succeeded**, not that an Intel
GPU was exercised or scientifically qualified. Unprobed facts have
`device_probed=False`, an empty `devices` tuple, and no runtime observation.
`versions` is a canonical tuple of distribution/version pairs;
`provider_libraries` records distribution/absolute library-path pairs, not a
fabricated Phydrax source/build ID. These installed paths do not prove that
those exact files were dynamically loaded.

Choose oneAPI **before any numerical work** using a fresh process:

```bash
JAX_PLATFORMS=oneapi python examples/accelerator_scientific_execution.py --accelerator oneapi
```

Or configure the existing runtime before backend initialization:

```python
from phydrax import config
from phydrax.backends import oneapi_environment

config.configure(config.RuntimeConfig(platforms=("oneapi",)))
report = oneapi_environment(initialize_backend=True)
if not report.successful:
    raise RuntimeError(report.issues)
for device in report.facts.devices:
    print(device.platform, device.client_platform, device.runtime_version)
```

Only the explicit probe requests `jax.devices("oneapi")`. It records each
actual device's platform, client platform/runtime version, kind, ID, and process
index. Empty results, backend exceptions, CPU results, or an unrelated GPU
runtime refuse rather than falling back to CPU. The registered JAX platform is
`oneapi`; the upstream plugin requests XLA's `SYCL` platform. A generic `gpu`
device is identified as oneAPI only with explicit oneAPI/SYCL client identity or
a leading oneAPI/SYCL runtime token, never because its display name says Intel.
`xpu` and `sycl` are not alternate public platform selectors.

Explicit CPU use remains supported:

```bash
JAX_PLATFORMS=cpu python your_program.py
```

A oneAPI diagnostic probe does not replace that selection. It returns host
assessment with `platform-not-selected` and leaves the backend unprobed. An
unsuccessful host assessment also prevents provider initialization. Diagnostics
do not mutate the global platform selection.

## Numerical and upstream limitations

The experimental Phydrax Intel profile admits single-device float32 scientific
candidates, not a blanket capability promise. Intel documents GEMM support
including float64/complex128 on applicable hardware, but Phydrax's **FP64
hardware qualification is unknown/pending**. `jax_enable_x64=True` is not FP64
hardware evidence; it is also used for wide integer addressing. No global dtype
casts, precision downgrades, or CPU fallback are installed.

Upstream release capabilities and limitations must not be confused:

- JIT, autodiff, vectorization, canonical JAX arrays, FFT and dense linear algebra
  are the intended scientific routes. **General `eig` uses hybrid host/GPU
  execution**; it must not be described as wholly device-native.
- Upstream supports single-host multi-GPU collectives/sharding, but Phydrax's
  current Intel qualification candidates are single-device. **Multi-host /
  multi-process scale-out is unsupported**. The installed `impi-rt` dependency
  is not evidence of a supported MPI execution lane.
- **Pallas, Mosaic-GPU, CUDA-only features, sparse kernels, and RNN kernels are
  unsupported** in the selected release. There is no CUDA/Pallas kernel port.
- Executable serialization and PJRT topology deserialization are broken in this
  release. Do not promise portable serialized oneAPI executables.
- `jax.profiler.trace` and pinned-host DLPack round trips have known failures.
  Export of oneAPI solver/linalg custom calls and symbolic-shape SVD are limited.
- Reverse-mode gradients through `odeint` and a rotation conversion can crash
  compilation; long-vector SPIR-V conversion is also a known failure. These
  routes are not silently rewritten or claimed as supported.
- Known numerical failures include some single-element bfloat16 operations,
  bfloat16 sort, signed-int4 operations, unsigned-preferred convolution or
  correlation, and NVFP4 scaled dot products. TF32 dot algorithm presets are
  unsupported. A passing float32 smoke case cannot qualify these routes.

See the [complete upstream release limitations](https://github.com/Intel-tensorflow/jax/releases/tag/oneapi-jax-v0.11.2-alpha)
for exact failures. Higher-order derivative and FFT support must be established
for the exact Phydrax qualification cases on actual selected hardware, not
inferred from package installation or from a CPU reference computation.

## Installed-wheel evidence

The resolver check is software evidence only:

```bash
python -m tools.check_accelerator_extras \
  --phydrax-wheel dist/<core>.whl \
  --work-dir /tmp/extras-fresh \
  --output /tmp/extras.json
```

Actual hardware qualification uses a new work directory and an installed wheel
isolated from the source checkout:

```bash
python -m tools.qualify_accelerator_wheels \
  --accelerator oneapi \
  --phydrax-wheel dist/<core>.whl \
  --python /path/to/python3.13 \
  --work-dir /tmp/qualification-fresh \
  --output /tmp/qualification.json
```

Run on native Linux x86_64 with the external driver and an actual Intel GPU;
work directories must not already exist. Qualification binds installed
distribution/file/runtime identities, requires the real selected oneAPI/SYCL
runtime, and exercises all five exact scientific cases. Skipped cases cannot
count as passes. Provisioning hardware or cloud resources is not part of this
feature. Until such a hardware report exists, qualification remains explicitly
pending even if every software-only check passes.
