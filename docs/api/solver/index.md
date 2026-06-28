# Solver

Solvers assemble fields, constraints, and optional enforced-constraint pipelines into a
scalar loss suitable for optimization.

For a conceptual overview (loss evaluation, enforced pipelines, training loop behavior), see
[Guides → Solvers and training](../../guides_solver.md).

!!! note
    Key notes:

    - Use `FunctionalSolver` to sum constraint losses and attached model losses.
    - Use enforced constraint pipelines to enforce conditions by construction (no penalty term).
