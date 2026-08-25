# Solver

Phydrax separates functional minimization over physics/data terms, persistent-chain
variational Monte Carlo, and direct finite-dimensional integration. Direct solvers
cover explicit ODE/SDE, differential-algebraic, differentiable controlled,
probabilistic ODE, Lyapunov, finite-activity jump, hybrid jump-differential, and
semidiscrete spatial systems.

For a conceptual overview (loss evaluation, exact enforcement, training loop behavior), see
[Guides → Solvers and training](../../guides_solver.md).

- [Time integrators](time_integrators.md) gives the equation/method compatibility
  matrix, method properties, backend ownership, and differentiation semantics.
- [Differential equation integration](differential.md) defines reproducible ODE/SDE,
  differentiable CDE and neural-CDE training, probabilistic ODE filtering,
  finite-time Lyapunov spectra, finite-activity jump and hybrid trajectories,
  finite-rank semidiscrete SPDEs, and process ensembles.
- [Differential-algebraic equation integration](differential_algebraic.md) defines
  consistent initialization, prepared fixed/adaptive BDF1--BDF5, endpoint theta,
  segmented continuation, frozen-grid replay derivatives, local regularity evidence,
  and implicit semidiscrete PDE residuals.
- [Delay and functional differential equations](delay.md) defines causal method-of-steps,
  stochastic/geometric/rough/jump histories, functional/distributed/state-dependent/
  neutral delays, bounded and infinite memory, convolution, Caputo integration, and
  global collocation for future arguments.
- [Functional solver](functional_solver.md) assembles training terms, evaluation
  terms, exact enforcement, and model-attached losses for optimization.
- [Variational Monte Carlo](variational_monte_carlo.md) combines persistent Markov
  chains, connected local observables, centered score geometry, and the existing
  linear runtime for discrete amplitude optimization.
- [Variational TDVP](variational_tdvp.md) reuses the same chains, connected
  observables, and score geometry for fixed-step real- or imaginary-time parameter
  evolution.

!!! note
    Key notes:

    - Use `FunctionalSolver(functions=..., terms=..., evaluation_terms=...,
      enforcement=...)` to optimize scalar terms and report separate evaluation terms.
    - Compile exact conditions into an `EnforcementProgram` and pass that program
      as `enforcement`.
    - Use `log_terms` to control per-term reporting and `train_term_sample_size`
      to take an unbiased fixed-size subset of training terms per optimizer step.
    - `phydrax.optim.kfac(...)` accepts quadratic `ResidualPenalty` terms and freezes
      each active term realization across its gradient, curvature update, and line search.
    - Use `VariationalMonteCarloProblem` and `solve_variational_monte_carlo` for a
      discrete connected operator and user-defined log amplitude. The VMC path is not
      a `FunctionalSolver` optimizer because its target-dependent chains and covariance
      gradient have a separate state transition.
    - Use `VariationalTDVPPolicy` and `solve_variational_tdvp` when the same amplitude
      manifold must evolve under projected Schrödinger or imaginary-time dynamics.
    - Use `DifferentialProblem` plus `solve_diffrax` for numerical ODE/SDE trajectories.
    - Use `DifferentialAlgebraicProblem`, a `TimeGrid`, and `solve_dae` for regular
      fixed-grid or adaptive index-one residuals `F(t, y, ydot, args) = 0`.
    - Use `AbstractDifferentiableDrivingPath` plus `solve_diffrax_cde` for smooth
      first-level controls; rough controls and their second level belong to
      `solve_rough_differential`.
    - Use `NeuralCDETrainingData` and `train_neural_cde` for masked irregular
      physical-time training with exact optimizer/batch resume.
    - Use `solve_probabilistic_ode` for a deterministic Euclidean ODE posterior
      with separately attributed numerical, process, observation, initial, and
      parameter covariance.
    - Use `phydrax.dynamics.analysis.finite_time_lyapunov_spectrum` with a
      differentiable flow/map evolution for periodic-QR spectra and resumable
      tangent checkpoints.
    - Use `solve_diffrax_ensemble` with a global `WienerRealization` for coupled process draws.
    - Use `DelayDifferentialProblem` plus `solve_diffrax_delay` for causal delay IVPs.
    - Use `solve_rough_delay` for geometric rough paths with delayed state.
    - Use `solve_jump_delay` for prescribed finite-activity events in a delay equation.
    - Use `solve_convolution_volterra` or `solve_caputo_fractional` for causal integral
      and power-law memory equations.
    - Use `solve_functional_differential` for advanced or mixed future/past equations.
    - Use `solve_next_reaction` or `solve_direct_ssa` with a
      `PoissonClockRealization` for pure jump paths.
    - Use `solve_jump_differential` with explicit Poisson and optional Wiener
      realizations for coupled hybrid dynamics.
    - Use `phydrax.discretization.TensorSpectralDiscretization`,
      `phydrax.discretization.PreparedFiniteDifferenceDiscretization`, or
      `phydrax.discretization.EigenbasisDiscretization`, a
      `phydrax.stochastic.SpatialNoiseBasis`, and `semidiscretize_spde` for
      finite-rank spatial dynamics.

`spatial_measure` reuses a spatial discretization's physical quadrature as a
deterministic external integration target. Tensor-grid axis weights remain
separable until reduction.

::: phydrax.integration.spatial_measure
