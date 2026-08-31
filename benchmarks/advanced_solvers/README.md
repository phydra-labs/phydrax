# Advanced solver benchmarks

This package runs deterministic, schema-validated comparisons of Phydrax's advanced
solver paths and mathematically comparable third-party implementations. It is a
reproducibility harness, not a benchmark result or a claim that one method wins every
problem.

## Run

From the repository root:

```bash
python -m benchmarks.advanced_solvers capabilities
python -m benchmarks.advanced_solvers run \
  --preset ci \
  --output benchmarks/advanced-solver-ci.json
python -m benchmarks.advanced_solvers run \
  --preset convex \
  --output benchmarks/advanced-solver-convex.json
python -m benchmarks.advanced_solvers control \
  --horizon 8 --horizon 32 --horizon 128 \
  --output benchmarks/control-horizon-warm.json
python -m benchmarks.advanced_solvers compare \
  reference.json candidate.json \
  --output comparison.json
```

Repeat `--adapter` or `--case` to select an exact subset. `--size`, `--seed`,
`--warmup`, `--repeats`, tolerances, and maximum steps override the preset. JSON is
written with `allow_nan=False`; incomplete measurements use `null` plus explicit
evidence rather than fabricated numeric sentinels.

`compare --relative-performance-tolerance R` and
`--absolute-performance-tolerance-ms A` enable a solve-time regression decision.
`--performance-confidence`, `--performance-bootstrap-resamples`, and
`--performance-minimum-samples` control its uncertainty contract. Omitting both
practical tolerances keeps comparison descriptive.

## Problem families

The default campaign contains one deterministic representative of each declared
contract:

| Case | Mathematical contract | Independent certificate |
| --- | --- | --- |
| `linear-scalar` | sparse SPD system, one RHS | original-system residual and backward error |
| `linear-block` | sparse SPD system, multiple RHS | per-RHS original-system residual and backward error |
| `nonlinear-root` | nonlinear algebraic root | physical residual |
| `nonlinear-vi` | bound variational inequality | projected natural-map residual and feasibility |
| `general-eigen` | nonsymmetric partial eigenproblem | original eigenpair residuals |
| `continuation-fold` | pseudo-arclength traversal of a quadratic fold | branch residual, state/tangent sign changes, and fold bracket |
| `optimization-unconstrained` | nonquadratic Rosenbrock minimization | objective, reference gap, gradient norm, and distance |
| `optimization-constrained` | Maratos-type equality/inequality problem | objective, feasibility, and independently estimated KKT stationarity |
| `optimization-proximal` | smooth plus L1 composite objective | proximal-gradient stationarity and reference gap |
| `optimization-bounded-least-squares` | unit-box residual minimization with active bounds | projected stationarity, exact feasibility, objective/reference gap |
| `optimization-linear-program` | bounded separable LP | projected KKT stationarity, feasibility, objective/reference gap |
| `optimization-quadratic-program` | bounded diagonal positive-definite QP | projected KKT stationarity, feasibility, objective/reference gap |
| `optimization-conic-program` | active Lorentz-cone QP | cone feasibility, estimated KKT stationarity, objective/reference gap |

Generators are seed-deterministic. Their fingerprints cover numerical values,
shapes, dtypes, and semantic configuration. A refresh case changes coefficients while
preserving declared structure so symbolic reuse can be checked independently.

Three opt-in root cases isolate solver architecture rather than adapter defaults:

- `nonlinear-root-dense` runs the same separable root through Newton with an explicit
  dense LU linear solve in both Phydrax and Optimistix;
- `nonlinear-root-matrix-free` runs that root through Newton with restart-16 GMRES in
  both adapters;
- `nonlinear-root-sparse-pde` uses a semilinear finite-difference boundary-value
  problem. Phydrax compiles its declared symmetric tridiagonal Jacobian, applies
  Jacobi-preconditioned native PCG, and reports symbolic reuse plus numeric refresh.
  Optimistix supplies a matrix-free Newton-GMRES reference on the identical problem.

All adapters receive the same initial values, targets, tolerances, outer nonlinear
step limit, and independent residual certificate. The sparse case gives both inner
solvers a dimension-scaled work ceiling; PCG with Jacobi and GMRES with identity
preconditioning remain different algorithms, so that case is a lifecycle and scaling
reference rather than an algorithm-matched speed contest.

The best-nonlinear root campaign additionally includes `phydrax-lagged` on
declared frozen-factor problems. `diagonal-polynomial` freezes one polynomial
factor; `quasilinear-diffusion` freezes the positive state-dependent
diffusivity inside one periodic implicit stage. Cases without an explicit
lagged operator retain an `unsupported-mathematics` row. Every lagged result is
certified with the same original physical residual as Newton and peer methods;
convergence of the inner linear solve is not a root certificate.

The opt-in `convex` preset selects Phydrax, MPAX, and Clarabel across the LP/QP/SOCP
cases. Unsupported backends remain explicit skipped rows. Preparation, numeric refresh,
solve, certificates, and memory/transfer evidence use the same phase schema as every
other advanced-solver case.

## Adapters

The canonical registry contains:

- `phydrax`: public native linear, nonlinear, general-eigen, continuation, and
  optimization APIs;
- `mpax`: optional device raPDHG LP/QP execution through the Phydrax audit boundary;
- `clarabel`: optional host quadratic-conic interior-point execution through the
  Phydrax audit boundary;
- `jax`: direct JAX baselines where the same mathematics is available;
- `lineax`: linear scalar/block paths;
- `optimistix`: nonlinear-root and unconstrained optimization paths;
- `scipy`: sparse linear, nonlinear, VI, general eigen, and optimization paths;
- `pyamg`: host algebraic-multigrid linear paths;
- `amgcl`: PyAMGCL linear paths;
- `amgx`: Phydrax's explicit NVIDIA AmgX backend lifecycle;
- `petsc`: petsc4py KSP and SNES paths;
- `slepc`: slepc4py EPS path.

Adapters import optional dependencies lazily. Every selected case × adapter row is
emitted in stable order. Unsupported mathematics and unavailable packages become
precise `skipped` rows; they are not removed from the report and are not replaced by
a different algorithm.

## Timing protocol

Each row separates and labels:

1. setup,
2. compilation,
3. preparation,
4. warmup (excluded from statistics),
5. repeated solve samples,
6. implicit-differentiation compilation and repeated execution when supported,
7. independent verification of the unmodified canonical problem,
8. numeric refresh,
9. refreshed solve,
10. independent verification of the refreshed problem.

Compilation may be deferred until after numerical preparation when an adapter needs a
prepared state to lower its executable. The repository-wide private benchmark runtime
provides one recursive JAX synchronization boundary and one raw duration-distribution
implementation. Timing summaries are recomputed from their raw samples during schema
validation. A device adapter must block at every phase boundary before a sample is
accepted. Setup, compilation, preparation, differentiation, refresh, and verification
are never silently folded into steady-state solve timing.

Campaigns enable JAX float64 before capturing their runtime fingerprint. The fingerprint
covers Python, Phydrax, NumPy, JAX, jaxlib, backend/device identity, default precision,
performance-affecting environment variables, and the normalized installed-package set.
Source revision remains separate so two commits can be compared in one identical runtime.
The nonlinear-root Phydrax and Optimistix adapters both differentiate the converged
solution with respect to the target through each library's implicit-root contract; an
analytic diagonal sensitivity check guards this comparison.

## Transfers, memory, and operations

Transfer fields distinguish host-to-device and device-to-host bytes and name the
measured timing phase or phases containing each transfer. `null` means not measured;
zero means measured and absent. AmgX reports exact CSR/RHS uploads, solution downloads,
and synchronizations from its backend result. Host-only adapters report zero device
transfer.

Memory evidence distinguishes input matrix bytes, setup bytes, and peak estimate.
An unavailable provider metric remains `null` with a reason. Operation evidence uses
provider counters when exposed and otherwise remains `null`; an iteration count is not
relabelled as a matvec count unless the algorithm makes that equality exact.

## Convergence and comparability

The adapter's own status is preserved under `outcome`. The main certificate is computed
again from the original, unmodified problem relation in `certificates.py`; consequently,
all adapters for one case retain the same canonical problem fingerprint. A finite
returned array is not success. When refresh applies, its perturbed problem fingerprint,
relative residual, backward error, convergence result, and independent-certification
flag are recorded separately under `refresh`. Continuation success additionally requires
demonstrated fold traversal; an endpoint residual alone is insufficient. Optimization
rows retain stationarity/KKT/proximal evidence and reference objective gaps.

`compare` requires matching schema, selected case × adapter cross-product, case
fingerprints, implementations, timing protocol, transfer contract, and—unless explicitly
overridden—runtime fingerprint. Optional practical thresholds add deterministic
bootstrap intervals over the retained solve samples. A cross-runtime comparison may be
descriptive, but it is never regression-eligible.

## Layout

- `benchmarks/_runtime.py`: shared synchronization, timing distributions, compiler
  evidence, logical array bytes, and runtime fingerprints;
- `benchmarks/_comparison.py`: deterministic paired or independent performance
  comparisons with practical thresholds;
- `benchmarks/_io.py`: finite atomic JSON and generic atomic file replacement;
- `problems.py`: deterministic generators and identity fingerprints;
- `certificates.py`: independent mathematical verification;
- `adapters/`: lazy, source-specific setup/compile/prepare/solve/differentiate/refresh
  bridges;
- `harness.py`: solver lifecycle execution and row construction;
- `schema.py`: strict row/report validation;
- `compare.py`: comparability checks and report deltas;
- `campaign.py`: presets and exact case construction;
- `cli.py`: `run`, `capabilities`, and `compare` commands.
