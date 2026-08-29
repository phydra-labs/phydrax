# Variational Monte Carlo

Phydrax variational Monte Carlo optimizes a user-supplied discrete amplitude model
against a matrix-free connected operator. It composes the shared fixed-kernel Markov
sampler, weighted empirical-measure reduction, parameter subspaces, empirical Gram
operators, and the existing linear solve runtime.

The sampler targets the real density `2 log |ψ|`. Samples are treated as correlated;
training never differentiates through accept/reject decisions and never reports an IID
standard error.

## Problem and policy

::: phydrax.solver.VariationalMonteCarloProblem
    options:
        members:
            - __init__
            - model_from_coordinates
            - initial_state

::: phydrax.solver.VariationalMonteCarloPolicy

The parameter mode is explicit:

- `"real"`: real parameters with real or complex amplitude output;
- `"holomorphic"`: complex parameters under a holomorphic contract;
- `"nonholomorphic"`: complex parameters represented by separate real coordinates.

No mode is inferred from output dtype. The policy uses a fixed proposal, persistent
walkers, centered score geometry, explicit damping, and the existing
`LinearSolvePolicy`/`NullspacePolicy` contracts.

## State and results

::: phydrax.solver.VariationalMonteCarloState

::: phydrax.solver.VariationalMonteCarloEstimate

::: phydrax.solver.VariationalMonteCarloResult

::: phydrax.solver.evaluate_variational_monte_carlo

::: phydrax.solver.solve_variational_monte_carlo

The result retains every training energy, variance, acceptance rate, update norm,
status, and linear solve result, plus a separate frozen-model final evaluation.
`failure_mode="record"` stops without applying a failed update; `"raise"` raises at
the first invalid estimator or metric solve.

When `final_chain_diagnostics=True`, the separate frozen-model evaluation reports
rank-normalized R-hat, bulk ESS, and tail ESS for configurations and the real/imaginary
local-energy components. These diagnostics are not computed across training
iterations because each parameter update changes the target.

## Multi-state invariant subspaces

`VariationalMonteCarloSubspaceProblem` owns two or more log-amplitude models,
one connected Hermitian operator, and one persistent Markov ensemble. The
sampler targets the mixture proportional to the sum of state densities. On that
same sample set, relative amplitudes produce an overlap matrix and node-safe
Hamiltonian actions produce a projected Hamiltonian matrix; their unknown
common normalization cancels in the generalized Ritz solve.

The evaluator reports raw and Hermitian-projected matrices, Hermitian defects,
Gram minimum eigenvalue/rank/condition, Ritz energies and coefficient modes,
per-mode residual variances, acceptance, active samples, and chain diagnostics.
It never regularizes a collapsed state span into success. Optimization uses a
score-corrected block objective and one responsibility-weighted native SR solve
per trainable state. Accept/reject decisions remain outside differentiation.

::: phydrax.solver.VariationalMonteCarloSubspaceProblem

---

::: phydrax.solver.VariationalMonteCarloSubspaceState

---

::: phydrax.solver.VariationalMonteCarloSubspaceEstimate

---

::: phydrax.solver.VariationalMonteCarloSubspaceResult

---

::: phydrax.solver.evaluate_variational_monte_carlo_subspace

---

::: phydrax.solver.solve_variational_monte_carlo_subspace

---

::: phydrax.solver.vmc_subspace_status_name


## Checkpoint and resume

::: phydrax.solver.write_variational_monte_carlo_checkpoint

::: phydrax.solver.read_variational_monte_carlo_checkpoint

The checkpoint is a checksum-validated, pickle-free array archive. It retains model
arrays, selected parameter coordinates, walker positions and target values, transition
index, iteration, and root key. Resume requires the same problem, sampler/proposal,
parameter structure, complex mode, and step policy. `num_iterations` and final
evaluation length are intentionally excluded from compatibility so a caller can select
additional work after restoring.

`solve_variational_monte_carlo(..., state=restored)` uses the checkpoint root key when
`key` is omitted. Supplying a different key is rejected. Frozen final evaluation does
not advance the continuation state, so splitting a run across checkpoints reproduces
the same training trajectory as an uninterrupted run.

## Status

::: phydrax.solver.vmc_status_name

::: phydrax.solver.VMC_SUCCESS

::: phydrax.solver.VMC_INVALID_SAMPLES

::: phydrax.solver.VMC_NONFINITE

::: phydrax.solver.VMC_IMAGINARY_ENERGY

::: phydrax.solver.VMC_LINEAR_FAILURE

A Hermitian operator should have a real expectation, but local energies may be complex.
The solver records the absolute imaginary mean and refuses to apply an update when it
exceeds `energy_imag_tolerance`.
