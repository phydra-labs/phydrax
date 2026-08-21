# Changelog

## Unreleased

### Added

- `phydrax.weighting` exact and quadratically reconciled relative-entropy moment
  calibration for dense, sparse, and matrix-free feature actions, with affine-rank
  reduction, audited convergence/regularity evidence, warm starts, and implicit
  target/prior derivatives. The mathematical formulation follows Barratt, Angeris,
  and Boyd's *Optimal Representative Sample Weighting*; the public `cvxgrp/rsw`
  and Apache-2.0 `andytimm/rswjax` packages are acknowledged as design
  inspiration, while this implementation is independent and Phydrax-native.
- Finite-measure calibration in `phydrax.integration`, shared calibration/coreset
  lowering, and ordered transformation diagnostics that preserve physical mass,
  masks, named axes, ancestry, support validity, execution keys, and provenance
  while invalidating inapplicable inherited integration-error bounds.
- Dense/sparse exact/soft calibration benchmarks with separate setup, first
  compilation, steady-state, cold-nearby, and warm-nearby timing and numerical
  evidence.
- Newton--Krylov now passes its adaptive forcing tolerance into each inner linear
  solve, so forcing policy changes actual Krylov work under eager and compiled
  execution.

- Learned function frames with masked, quadrature- and channel-metric-aware
  projection, explicit rank and residual evidence, reusable source encodings,
  arbitrary-query reconstruction, frozen inference, portable artifacts, and a
  research-tier operator benchmark composition.
- Scalar Tikhonov damping for dense SVD least-squares solves and a direct
  weighted least-squares path that reuses one factorization for coefficients,
  rank diagnostics, residuals, and differentiation.
- `phydrax.nonlinear` contracts for algebraic systems, Newton line-search and
  trust-region roots, nonlinear GMRES and preconditioning, fixed-point acceleration,
  full-approximation multigrid cycles, variational inequalities, semismooth
  complementarity solves, and implicit root derivatives with explicit failure status.
- Reusable nonlinear Newton preparation/refresh/solve lifecycles, adaptive
  Eisenstat--Walker forcing, explicit Jacobian refresh policies, hard aggregate
  inner-linear work budgets, and physical-root certification for transformed
  nonlinear preconditioners.
- Dynamic per-invocation controls for prepared native Krylov solves, plus
  capability-checked mixed-precision dense LU with pre-factorization condition
  screening, high-precision residual certification, iterative refinement, and
  requested/effective precision evidence.
- Matrix-free low-rank ADI lifecycles for factored continuous Lyapunov equations,
  including fixed-capacity factors, rank/truncation evidence, per-shift convergence,
  exact low-rank residual certification, numeric refresh, and factor-versus-dense
  storage accounting.
- Typed nonlinear optimization with matrix-free Newton--Krylov and trust regions,
  nonlinear conjugate gradients and strong-Wolfe search, Gauss--Newton,
  Levenberg--Marquardt, deterministic finite-difference least squares, proximal
  gradient/Newton methods and built-in functionals, filter/SOC SQP, primal--dual
  predictor--corrector KKT solves, state/design adjoints, stochastic risks and
  decomposition, explicit Optimistix interoperation, and `FunctionalSolver`
  integration.
- `phydrax.continuation` contracts for generic parameterized residual curves,
  arbitrary physical-parameter PyTrees, natural and pseudo-arclength
  predictor/corrector methods, reusable bordered solves, adaptive rejection,
  event localization, dense and Krylov stability evidence, explicit branch switching,
  fold/Hopf/pitchfork extended systems and certificates, normal forms, homotopies, and
  metric-aware root deflation.
- Standard and generalized nonsymmetric eigenproblem lifecycles with dense Schur/QZ
  and native restarted-Arnoldi/Krylov--Schur methods, standard, shift-invert, and
  Cayley transforms, homogeneous finite/infinite classification, paired left/right
  eigenvectors, residual and conditioning evidence, resource accounting, and
  isolated-eigenvalue derivatives.
- Lazy optional PETSc KSP/SNES, SLEPc EPS, PyAMGCL, and NVIDIA AmgX backends with
  dependency-free package import, explicit capability probes and lifecycle plans,
  sparse or matrix-free execution contracts, independently verified residual/status
  evidence, numeric refresh, transfer accounting, and explicit GPU/collective release.
- Canonical sparse triangular analysis and numeric solve, provider-neutral sparse
  factorization plans, incomplete Cholesky/ILU/ILUT preconditioner builders, and
  explicit sparse-provider availability/capability inspection.
- Array-backed finite axes and lazy Cartesian products, exact streaming exhaustive reduction, deterministic finite MAP screening, exact finite control-catalog search, portable MAP candidate archives, and a dense-oracle benchmark harness; independently implemented with [Brutax](https://github.com/michael-0brien/brutax) acknowledged as design inspiration.
- Typed preconditioner builders, prepared actions, planning costs, refresh provenance, and materialization-aware resource rejection.
- Additive and multiplicative subspace correction, Chebyshev smoothing, block factorization, and immutable multigrid hierarchy preparation.
- Native fixed-capacity BlockGMRES and BlockCG with explicit right-hand-side layouts, shared-subspace diagnostics, and block-aware differentiation.
- Immutable GCRO-DR recycling state with explicit reuse and rebuild policy.
- Standard and generalized Hermitian eigensolve plans with LOBPCG and thick-restarted Lanczos, residual/status diagnostics, refresh, and isolated-eigenvalue differentiation.
- Exact Galerkin and smoothed-aggregation hierarchy builders, deterministic diagnostics, transfer reuse, and optional PyAMG conversion.
- Exact diagonal and uniform local-block assembly, local block operators and factorizations, block-Jacobi preparation, and native Kronecker-sum direct solves.
- Policy-bounded canonical sparse assembly plans with reusable prepare/refresh recipes for sparse algebraic operator graphs.
- Deduplicated resident operator-state and per-right-hand-side action-workspace estimates, propagated into solve candidate costs.
- Symbolic `LinearSolveTemplate` planning with separate numeric binding, scoped kernel and spectral-interval certificates, and quotient-space `ProjectedPCG`.
- Bounded dense standard/generalized Hermitian eigensolves and pairing-aware singular-value decompositions with plan/prepare/refresh lifecycles and restricted scalar differentiation.
- Arbitrary-base `BasePlusLowRankLinearOperator` Woodbury solves with reusable base state, correction conditioning, resources, status, and provenance.
- Two-sided equilibration, iterative refinement, and resilient solve lifecycles that verify residuals in the original coordinates.
- Explicit structure compilation for diagonal, permutation, tridiagonal, triangular, banded, DCT-diagonal, and FFT-diagonal operators, including refresh-safe transform-diagonal operators.
- Numerically bound reusable Arnoldi/Lanczos projections, shared-basis shifted solve families, and partial-fraction rational matrix-function actions.
- Adaptive fixed-capacity stochastic trace and log-determinant estimation with separate statistical and projection-error evidence.
- General dense real/complex Schur eigensolves, nonnormal spectral observables, Riesz spectral subspaces, and first-order projector derivatives.
- Generalized, Sylvester, and continuous/discrete Lyapunov matrix-equation lifecycles built on the shared linear runtime.
- An advanced JSON benchmark harness covering Krylov reuse, shifted/rational actions, matrix equations, spectral projectors, low-rank updates, resilience, and adaptive spectral estimation.
- A schema-validated advanced-solver benchmark package with deterministic problem
  generators, independent original and refreshed certificates, explicit setup/
  compilation/preparation/solve/differentiation/refresh/verification and transfer
  accounting, reproducible JSON comparison, and lazy Phydrax, JAX, Lineax, Optimistix,
  SciPy, PyAMG, AMGCL, PETSc, and SLEPc adapters.
- Reusable full self-adjoint spectra with exact cluster-safe projector,
  density-kernel, and Loewner spectral-function derivatives for standard and
  generalized real or complex problems.
- Native batched dense Hermitian eigensolves and batched self-adjoint spectral
  calculus with per-member diagnostics, status, provenance, and batch-scaled
  resource accounting.

### Changed

- DeepONet now accepts generalized coordinate-evaluated basis trunks, exposes
  explicit output-bias control, and preserves existing pointwise and POD
  behavior while supporting projection branches and frozen nested models.
- Nonlinear algebraic systems now have one public owner, `phydrax.nonlinear`, and
  generic continuation/bifurcation workflows have one public owner,
  `phydrax.continuation`; obsolete optimization and dynamics-analysis continuation
  exports were removed rather than retained as aliases.

- Linear solve planning now treats preconditioning as an explicit prepared subsystem with compatibility, memory, workspace, and setup-action budgets.
- Nonlinear optimizer runtimes now keep callable refresh structure static, preserve
  float32 and accepted-state carries under JIT, distinguish all-nonfinite globalization
  failures, reject unenforceable adapter evaluation budgets, and aggregate nested
  refresh and derivative diagnostics without negative sentinel arithmetic.
- Operator property evidence is propagated conservatively through preconditioners, block solvers, eigensolvers, and multigrid construction.
- RHS-width, GCRO-DR extraction, and multigrid setup/reuse resources are now rejected and reported before numerical work, including transformed prepared solves and dependency-invalidated hierarchy refreshes.
- Galerkin and smoothed-aggregation hierarchies now retain sparse assembly recipes and refresh coarse coefficients without rebuilding symbolic routes or silently densifying downstream levels.
- The linear benchmark harnesses now cover block-Jacobi preparation, sparse assembly planning/refresh/action, structured Kronecker-sum solves, and the advanced reusable and higher-operator lifecycles against dense, exact, finite-difference, or invariant references.
- The advanced-solver benchmark now provides algorithmically matched explicit-dense
  Newton-LU and matrix-free Newton-GMRES root cases for Phydrax and Optimistix, plus a
  semilinear sparse-PDE case covering Phydrax sparse-Jacobian preparation, symbolic
  reuse, numeric refresh, and native Jacobi-preconditioned PCG. It separately measures
  compiled root solves, implicit-root derivative compilation/execution, numeric
  refresh, refreshed solves, and refreshed verification; campaigns preserve float64
  inputs and canonical problem identities across adapters.
- Tensor contractions across the package, tests, benchmarks, and tools now consistently use `opt_einsum.contract` instead of direct `jax.numpy.einsum` calls.
- Closure-converted matrix-free SVD, eigensolve, spectral-projector, density-kernel, and spectral-function derivatives now support filtered JVP, reverse mode, and JIT.
- Named truncated-normal initializers now produce their conventional target variance, while rectangular orthogonal initialization avoids max-dimension square samples.
- `phydrax.nn.layers.inference_mode` now switches every inference-aware Equinox or Phydrax leaf in mixed model trees.
