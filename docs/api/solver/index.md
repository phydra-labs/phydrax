# Solver

Phydrax has two separate solver paths: functional minimization for physics/data
objectives and direct finite-dimensional ODE/SDE or semidiscrete SPDE integration
through Diffrax.

For a conceptual overview (loss evaluation, enforced pipelines, training loop behavior), see
[Guides → Solvers and training](../../guides_solver.md).

- [Differential equation integration](differential.md) defines reproducible ODE/SDE
  trajectories, finite-rank semidiscrete SPDEs, and process ensembles.
- [Functional solver](functional_solver.md) assembles constraints, objectives, and
  model losses for optimization.

!!! note
    Key notes:

    - Use `FunctionalSolver` to sum constraint losses and attached model losses.
    - Use enforced constraint pipelines to enforce conditions by construction (no penalty term).
    - Use `DifferentialProblem` plus `solve_diffrax` for numerical ODE/SDE trajectories.
    - Use `solve_diffrax_ensemble` with an explicit `WienerDriver` for process draws.
    - Use `TensorGridDiscretization` or `SpectralSpatialDiscretization`, a
      `SpatialNoiseBasis`, and `semidiscretize_spde` for finite-rank spatial dynamics.
