# Apple GPU execution

The optional `phydrax[metal]` extra selects the lockstep `phydrax-metal`
companion. It implements JAX's `mps` platform using a pinned and patched JAX-MPS,
MLX, and OpenXLA/StableHLO source closure. It does not introduce a second Phydrax
array API or translate CUDA-specific providers into Metal kernels.

The capability catalog records **candidate profiles**, not released scientific
qualification. A successful install, device probe, or individual numerical
example does not qualify every Phydrax operation. Local artifacts and their
exact qualification reports must be distinguished from published releases.

## Installation and explicit selection

See [Installation](installation.md#apple-silicon-gpus) for the wheel contract and
[the native build guide](https://github.com/phydra-labs/phydrax/tree/dev/native/metal)
for developer builds. Users of published, qualified wheels do not need a source
compiler toolchain.

Select the platform before a numerical import or array initializes JAX:

```python
import phydrax as phx

phx.config.configure(
    phx.config.RuntimeConfig(platforms=("mps",), enable_x64=True)
)

from phydrax.backends import metal_environment

report = metal_environment()
print(report.successful, report.facts.device_kind, report.facts.runtime_version)
for issue in report.issues:
    print(issue.code, issue.message)
```

The equivalent process selection is `JAX_PLATFORMS=mps JAX_ENABLE_X64=1`.
Select `JAX_PLATFORMS=cpu` in a separate process to run CPU references with the
companion installed. Do not change platform selection after device initialization.
An unavailable or conflicting MPS runtime is an error, not permission to fall
back to CPU. Pure inspection uses `metal_environment(initialize_backend=False)`;
it does not discover devices. The default inspection explicitly probes MPS.

## Numerical precision is distinct from index width

Apple GPU arithmetic in this profile is float32 or complex64. Exact int64 and
uint64 addressing remains necessary for Morton keys, stable IDs, and sparse
ordering. **Keep x64 enabled**, and explicitly construct each numerical role in
the supported precision. Enabling x64 does not require floating-point arrays to
use float64. Disabling it globally would corrupt full-width addressing rather
than establish a safe lower-precision scientific profile.

`Linear` and `MLP` accept `dtype=jnp.float32`; `HyperRectangle` accepts an explicit
float32 geometry dtype. Their omitted-dtype defaults are unchanged. A float32
model does not convert its inputs, geometry, coefficients, optimizer state, or
other modules automatically. Phydrax does not silently downcast an existing
float64 model or artifact. Host arrays should be rounded once at their declared
boundary, then compared against references constructed from those same inputs.

Float64 and complex128 device transfers, runtime values, and arithmetic are
refused by the companion. Explicit compile-time casts of real float64 literals
or bounded, fully constant scalar setup graphs are evaluated before device-type
admission. Arithmetic depending on runtime values is not erased merely because
its final result is cast to float32. This permits ordinary JAX weak scalar
literals and fixed time-direction sentinels without emulated float64 kernels or
broad floating-point simplification. Float32 storage alone
is not an accuracy guarantee: conditioning, accumulation order, cancellation,
underflow, derivative order, and the problem's error budget still determine
whether a calculation is scientifically admissible. Preserve native status,
residual, support, and derivative-validity evidence instead of accepting finite
values as proof of success.

The native build disables MLX TF32 math and does not relax Phydrax's highest
matmul precision. Unsupported precision algorithms fail. Ordinary reductions
retain their declared reducer; compensated sparse execution remains an explicit
Phydrax policy rather than an invisible replacement of all sums.

## Scientific profile boundaries

| Candidate workload | Required boundary |
| --- | --- |
| MLP and Poisson residual | Explicit float32 parameters, geometry, input derivatives, Laplacian, parameter gradient, and optimizer update |
| Prepared Morton neighbors | Float32 geometry, exact full-width keys and stable IDs, prepared topology, capacity and stale-source refusal |
| Streamed sparse relations | Fixed replay, deterministic or explicitly compensated accumulation, masks, zero-event derivatives, and failure evidence |
| Native-JAX MACE | Float32 for every numerical role, native graph preparation, ordered energy/force/strain and parameter derivatives |
| Fourier resampling | Float32/complex64, declared mode ordering and normalization, Nyquist transfer and derivative/adjoint contracts |
| Native iterative linalg | Supported operator actions, true residual and termination status, preserved failure and derivative validity |
| Explicit temporal recurrence | Float32 state and clock, explicit fixed-step controller, bounded recurrence and sensitivity checks |

These boundaries are not blanket claims for every model architecture, hierarchy,
solver, FFT operation, or precision policy. In particular, other hierarchy and
voxel geometry can still require float64. Active dropout has not been admitted
by this profile. CUDA Pallas MACE, cuDSS, AmgX, QUDA, MPI, multiple devices,
GPU LAPACK, and arbitrary direct factorizations are not Metal capabilities.
Native MACE is a JAX route, not a new Pallas `metal` target.

The temporal qualification and benchmark use a fixed interval, save schedule, and
step size, with dynamic state and rate parameters. Dynamically supplied Diffrax
time bounds can introduce wide sentinel arithmetic and are not qualified here.

Prepared-neighbor measurements use
`tools/spatial_query_benchmarks.py --scenario nearest --route morton`. The generic
graph-neighbor reference deliberately uses float64; the Morton-only campaign
instead validates against an independent, bounded host dense reference on the
same rounded input coordinates. It does not run that reference as a GPU route.

## Bounded native optimizations

The companion selects optimizations from the actual lowered operation and its
declared shapes/dtypes; callers retain the same Phydrax APIs.

- **Ordinary reductions:** row-contiguous float32 sums, integer sums/products,
  and Boolean AND/OR can use MLX's parallel GPU reductions. Admission bounds
  conservative reduction scratch to 64 MiB without materializing input layouts.
  StableHLO permits implementation-defined ordinary reduction trees, so float32
  rounding can differ from the earlier serial implementation. This does not
  permit reassociation of explicitly deterministic or compensated event loops.
  Specialized extrema, paired-index, and window contracts retain their owning
  implementations.
- **Single indexed windows:** a provably injective scatter window bypasses
  destination sorting and searching. Complete-window dropping, full-width
  indices, reducer arithmetic, and NaN/signed-zero behavior remain intact.
  Multiple or overlapping updates retain the stable general route.
- **Pure bounded regions:** selected static-trip indexed maps and small-state
  recurrences execute inside one GPU kernel, preserving iteration order and
  selected-branch-only execution. Admission limits are 4,096 trips, 2,048 mutable
  elements, 32 KiB of declared temporaries, four million estimated element-work
  units, 512 operations, and a bounded argument list. All custom calls, including
  pure callbacks, are excluded. Nested, guarded, dynamic-bound, and unsupported
  regions retain the existing GPU execution with host orchestration. This is
  not whole-solver GPU residency.
- **Weighted O(3) contractions:** a matched three-stage float32 `uvu` contraction
  can use one kernel, including intermediate outputs required by JAX-generated
  derivatives. Coefficients and weights remain dynamic; component/path ordering
  is retained. Admission bounds total fused outputs, including saved derivative
  activations, to 64 MiB. Unsupported shapes or consumer dependencies retain
  ordinary native execution.

Metal's fill-gather lowering checks signed/unsigned indices before any address
narrowing. This also prevents out-of-range scatter cotangents from wrapping into
valid entries through JAX 0.12's generic shape-selected index conversion. The
correction is a platform-specific lowering registration, not a replacement of
JAX numerical functions or a change to CPU execution.

## Execution, memory, and failure

The companion executes synchronously. Completion evidence must still be
observed at timing and host-read boundaries. Host control-flow predicates and
explicit callbacks are not hidden numerical CPU fallback, but they can make
small or highly iterative workloads CPU-favored. Host topology preparation and
shape-derived FFT tables are immutable preparation; execution of the caller's
numerical FFT arrays remains on the GPU.

Compile-time admission checks each value's dimensions and byte count against
native limits before allocating or folding it. This is **not** an aggregate
program working-set bound. Unknown compiler memory estimates remain unknown.
Apple unified memory means host RSS, allocator counters, device arrays, and
logical retained bytes can overlap; do not add them and call the sum physical
memory usage. Input donation is an ownership contract, not a promise that every
operation physically reuses storage in place.

Unsupported operations and zero-copy/pointer exports fail explicitly. Callback,
readback, allocation, and numerical errors must reach the consumer. No failed
solve may be replaced with a plausible zero or a CPU result under an MPS label.

## Reproduce scientific and performance evidence

`examples/metal_scientific_execution.py` runs a float32 Poisson PINN with native
coordinate derivatives and Adam updates. It checks actual MPS placement and
reports numerical error, loss, gradient/update norms, runtime identity, and
status. Run it with the exact installed core and companion wheels; it refuses a
CPU-only setup.

The installed-wheel qualifier in `tools/qualify_metal_wheels.py` creates a fresh
environment outside the checkout, verifies installed-file hashes and the current
native source closure, and runs the mandatory primitive, scientific, import,
and lifetime cases in isolated processes. Missing, skipped, duplicated, stale,
or failed cases cannot produce a successful campaign. Its source/build and
hardware identities are necessary to interpret any results.

Performance measurements follow numerical and lifetime qualification. Compare
CPU float32 against Metal float32 using the same rounded inputs and algorithm;
use independent host float64 only for accuracy references. Run one campaign at
a time, without competing builds or tests. Keep preparation, transfer, lowering,
compilation, first execution, warm samples, explicit synchronization, compiler
memory, and logical retained bytes separate. Sweep the controlling capacity and
retain slow cases as well as fast ones. A favorable GEMM or successful device
probe is not evidence of end-to-end scientific acceleration.

No speedup is implied by the extra's name. Report the actual workload, capacity,
precision, numerical checks, runtime/build, and measurements before recommending
Metal over CPU for that workload.

## Local M1 Max results

The initial baseline is retained in `benchmarks/m1_max_qualification.json`.
`benchmarks/m1_max_optimization.json` records the optimized artifacts, all raw
measurements, 81 scientific and 12 primitive comparisons, accuracy checks,
failed development attempts, and nonclaims. The host is an Apple M1 Max with
32 GPU cores and 64 GiB unified memory, macOS 15.7.2, standard CPython 3.13.11,
and JAX/JAXlib 0.12.0.

The optimized companion passed all **293 mandatory installed-wheel cases**
without skips, plus the standalone Poisson example. Its source identity is
`f346b2ade03dd5b73fe0542791411f2d2c154489cbcfc0b5da90795086d7f4b8`;
the frozen wheels are under `.tmp/metal-native/frozen/f346b2ade03dd5b7/`.
These are local artifacts, not a publication or qualification of every Apple GPU.

Representative synchronized warm medians, in milliseconds:

| Workload | Capacity / route | CPU | Initial Metal | Optimized Metal |
| --- | --- | ---: | ---: | ---: |
| Ordinary sum | 1,000,001 float32 values | 0.067 | 114.975 | 0.470 |
| Singleton scatter | 512 rows, 8-value window | 0.009 | 0.338 | 0.233 |
| Ordered indexed map | 32 rows, 8 channels | 0.008 | 1.485 | 0.675 |
| Weighted `uvu` contraction | 256 lanes, 8 channels | 0.059 | 0.367 | 0.205 |
| Streamed sparse forward | 512 receivers, deterministic | 0.468 | 1,384.647 | 710.216 |
| MACE energy and forces | 16 atoms, streamed | 2.777 | 4,013.035 | 986.006 |
| Poisson PINN update | Width 256, depth 4 | 4.193 | 2.559 | 1.788 |
| Fixed Heun with reverse derivative | 512 steps, 128 states | 0.484 | 6,193.137 | 6,674.414 |
| Prepared Morton query | 512 sources, 128 queries, 8 neighbors | 11.817 | 47.995 | 52.098 |

The initial serial sum fails the new mixed-magnitude accuracy gate; the optimized
sum passes. Its timing ratio is not a comparison between two equally accurate
reductions. The smaller scatter changes are modest, not universal acceleration.
Heun, Morton, and some spectral/iterative cases regress; every measured capacity
and route is retained rather than selecting only favorable results.

The Poisson PINN uses identical rounded inputs and initial parameters: float32
MLP, 198,145 parameters at width 256, 64 training points, 500 Adam updates, and
learning rate `0.001 * 32 / width`. Width 256 passes the unchanged
`rtol=1e-4, atol=1e-5` CPU/Metal comparisons for all losses, initial diagnostics,
sampled first/second derivatives, and final errors. Its median update is
**2.34× faster than CPU** across 499 post-first updates. The 500-update training
phase is 2.147 seconds on CPU and 1.082 seconds on optimized Metal; preparation,
lowering, and compilation are separate. Final relative L2 error is about 0.00337.

Width 32 also passes agreement checks but remains CPU-favored. Width 1024 remains
faster on Metal but fails trajectory/derivative/error agreement, so it is not an
accuracy-qualified speedup. No tolerance was relaxed. The reporting-only
gradient norm uses host float64 over actual float32 gradient leaves; training
precision remains float32.

Complete MACE coupling/model outputs at every measured capacity match the CPU
reference within the predeclared `1e-4` relative/absolute bound, with matching
input/model hashes. The speedups over initial Metal do not make these MACE cases
competitive with CPU. CPU remains the recommendation for the other measured
scientific workloads.

Metal System Trace captured the optimized workload, but its default shader
timeline does not expose the kernel labels needed for per-kernel dispatch counts.
Explicit GPU capture was unavailable under the qualified runtime policy; that
policy was not bypassed, and no launch-count speedup is claimed. Instrumented
trace timings are excluded from the table. Each benchmark process used a fresh
JAX cache; the MLX cache was not reset. There is no cold Metal compiler-cache
claim, and unsupported compiler estimates remain unknown.
