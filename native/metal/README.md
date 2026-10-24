# Phydrax Metal companion

`phydrax-metal` is the wheel-only Apple Silicon PJRT companion selected by
`phydrax[metal]`. Its JAX platform is `mps`. The package reuses the pinned JAX-MPS
and MLX implementation with the explicit patches in `sources.lock.json`; it is
not a CUDA compatibility layer or a separate Phydrax scientific array API.

The current Python contract is standard GIL-enabled CPython 3.13. The companion
pins JAX and JAXlib 0.12.0. Its Darwin-arm64 wheel tag records the minimum macOS
read from the built binary. A deployment target is not hardware qualification:
retain the exact build and hardware qualification reports for each artifact.

## Numerical and runtime boundary

- Keep JAX x64 enabled for exact int64/uint64 addressing; construct numerical
  arrays explicitly as float32 or complex64 for the scientific candidate profile.
- Runtime float64/complex128 operands, results, and arithmetic are refused,
  not silently converted. After allocation admission and inlining, explicit
  casts into float32/float16/bfloat16 can consume real literals or bounded,
  fully constant scalar setup graphs. Evaluation visits at most 256 nodes with
  depth 64, uses declared precision and nearest-ties-to-even rounding, and never
  executes callbacks or simplifies arithmetic involving runtime values.
- Registration does not replace JAX numerical functions or disable Shardy.
  Single-device Shardy annotations are handled by the native runtime.
- Execution is synchronous. Host callbacks are explicit effects with owned
  lifetimes; they are not numerical CPU fallback routes.
- There is no CPU numerical route. Program admission refuses every
  `stablehlo.custom_call` target without a Metal GPU handler anywhere in the
  module, including unselected and nested branches, before an executable
  exists. The inherited CPU-LAPACK `mps.eigh`, `mps.qr`, and `mps.svd` handlers
  are removed rather than retained as dead routes; jaxlib `lapack_*_ffi` targets
  are refused the same way. Host callbacks, single-device sharding annotations,
  and admitted GPU custom calls remain. This is not a GPU factorization claim.
  Shape-derived immutable FFT planning tables are host preparation, not execution
  of the caller's numerical arrays on CPU.
- Bitwise XOR reductions are exact at the declared bool or integer width through
  the strict custom reduction shader; they are not a parallel-reduction or
  performance claim.
- Compile-time per-value allocation admission uses actual native limits and
  checked size arithmetic. It is not a bound on total program working-set memory.
- Raw pointer and zero-copy export APIs must report unsupported contracts rather
  than return a successful null pointer.
- PJRT `OptimizedProgram`, used by `Compiled.as_text()` and `hlo_modules()`,
  returns MLIR text (`mlir`) of the finalized module the executable runs. It is
  printed once, cached for the executable's lifetime, and never compiles,
  executes, or runs callbacks. The executed module is never rewritten. For XLA's
  HLO conversion, single-device Shardy constraints are reported as the
  replicated `@Sharding` call and admitted `mhlo.*` custom calls as `mps.mhlo.*`,
  which the companion admits and executes identically. Unknown targets in either
  namespace are refused. Undersized buffers return `INVALID_ARGUMENT`.
  Executable serialization remains unimplemented, and reports containing host
  callbacks cannot be recompiled in another process.
- Native source identity, binary asset hashes, and the matched JAX pair are
  verified separately. Installation or successful initialization does not grant
  scientific or performance qualification.

## Build inputs

Use an explicitly provisioned CPython 3.13 interpreter, CMake, Ninja, and a
selected Xcode installation containing the Metal compiler. The build tool records
the selected Xcode, compiler, SDK, source closure, patches, and audit results. It
uses the recorded Xcode selection consistently rather than an inherited
`DEVELOPER_DIR` override.

The canonical source lock includes the JAX 0.12 XLA/PJRT, LLVM/MLIR, StableHLO,
Shardy, MLX, Metal C++ headers, and required header-library inputs. Every archive
and applied patch is hash-verified. The build uses locked sources instead of
unbounded dependency fetches from CMake. Shared downloads are immutable;
intermediate build directories must have one integration owner.

The full-width gather-fill lowering has one canonical source,
`phydrax/_jax_indexing.py`. The build includes that source in `source_id` and
packages a byte-identical private `phydrax_metal/_jax_indexing.py`, whose hash
is included in the mandatory assets and `build_id`. The standalone plugin does
not import or depend on the Phydrax core package at runtime. Build from the
complete repository checkout; a copied `native/metal` directory alone is not
the source closure. Companions without the bound helper asset are refused.

```sh
python tools/build_metal_wheels.py \
  --python /absolute/path/to/python3.13 \
  --cmake /absolute/path/to/cmake \
  --ninja /absolute/path/to/ninja \
  --build-dir .tmp/metal-native/build \
  --wheelhouse .tmp/metal-native/wheelhouse \
  --output .tmp/metal-native/evidence/build.json \
  --jobs 6
```

`--dependencies-only` prepares dependencies but does not produce or qualify a
wheel. Dependency builds are reused only when their source, patch, toolchain,
and configuration input identities agree. The builder never installs its own
Python dependencies. A build failure retains evidence and returns nonzero.

The native allocation guards and numerical corrections must be compiled into
an artifact before its real-device tests are run. In particular, oversized
allocation qualification only compiles the request and requires the native
admission refusal; it never dispatches a giant allocation or kernel.

## Wheel packaging

The package backend only packages an audited native staging tree. Metadata-only
PEP 517 preparation reads the project and source-license declarations without
loading a GPU, fetching dependencies, or requiring a native build. This permits
the repository's optional dependency to be resolved before compilation.

Wheel construction requires `config_settings={"staging": ...}` supplied by the
build tool. Source distributions and editable installs are refused. The wheel
contains the package-local PJRT library, its Metal library, source/build manifest,
Python registration boundary, and all required third-party license texts.

The Mach-O audit checks architecture, deployment target, exported entry points,
system-only dependencies, signatures, and embedded build-prefix leakage. Native
inputs are checked again after compilation and packaging; a changing source tree
cannot produce a successful build report.

## Installed-artifact qualification

Build the core Phydrax wheel from the same implementation, then use the resolver
and hardware tools on the exact core and companion artifacts:

```sh
python -m tools.check_metal_extras \
  --phydrax-wheel /absolute/path/to/phydrax.whl \
  --companion-wheel /absolute/path/to/phydrax_metal.whl \
  --work-dir /absolute/path/to/fresh/resolver-work \
  --output /absolute/path/to/resolution.json

python -m tools.qualify_metal_wheels \
  --python /absolute/path/to/python3.13 \
  --phydrax-wheel /absolute/path/to/phydrax.whl \
  --companion-wheel /absolute/path/to/phydrax_metal.whl \
  --work-dir /absolute/path/outside/checkout/fresh-qualification \
  --output /absolute/path/to/qualification.json
```

The hardware qualifier requires a fresh environment outside the checkout,
installed-file provenance, the current approved source identity, real MPS
placement, all required case outcomes, and the standalone scientific example.
Missing, failed, skipped, expected-failing, duplicated, or stale cases fail the
campaign. The exact inventory covers primitive, scientific, import, lifetime,
custom-call admission (including CPU-LAPACK targets in unselected and nested
branches), XOR and lineage, and optimized-program inspection cases. Each device
file runs in its own process with `JAX_PLATFORMS=mps` and `JAX_ENABLE_X64=1`;
host-only import cases run separately. Do not switch backends after numerical
imports have initialized JAX.

Run performance campaigns only after numerical and lifecycle qualification,
without concurrent builds or tests. Preserve preparation, lowering, compilation,
first execution, warm execution, synchronization, and memory observations as
separate quantities. Never turn unavailable compiler memory estimates into zero
or add overlapping unified-memory host/device observations into a physical limit.

## Updating the source closure

Change the pinned source or a reviewable patch, record its actual hash, and
rebuild. Runtime loader code and build-driver inputs participate in the native
source identity as well as the C++ source/patch closure. Do not substitute a
same-version older wheel, reuse a historical source ID, or waive required gates.
The first Python 3.12 build is retained only as historical development evidence;
it does not qualify the merged Python 3.13 stack.

Publish qualified companion artifacts before advertising the matching core
extra in a release. Building and qualifying locally does not authorize package
publication. CUDA kernels, CUDA native providers, and multi-device or multi-host
execution remain outside this companion's support claim.
