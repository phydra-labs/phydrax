# Variational TDVP

Variational TDVP evolves the same canonical amplitude models and persistent Markov
state used by variational Monte Carlo. Each fixed time step samples the current
target, evaluates a generalized local quantum action, builds centered score
geometry, solves one damped matrix-free system, and advances the selected parameter
coordinates.

## Policy

::: phydrax.solver.VariationalTDVPPolicy

Two evolution modes are explicit:

- `"imaginary-time"` follows the negative projected energy force;
- `"real-time"` follows the projected Schrödinger force.

For holomorphic complex coordinates these are respectively `−S⁻¹F` and
`−i S⁻¹F`. Real and non-holomorphic coordinates use the corresponding real and
imaginary covariance projections. The metric, damping, nullspace handling, solve
policy, and failure evidence are the same Phydrax linalg contracts used by SR.

The established VMC-coupled API remains fixed-step Euler.
`AdaptiveTDVPPlan` adds a separate fixed-attempt Euler/Heun or symmetric
midpoint epoch. It reports temporal defect and sampling uncertainty separately;
noise-dominated attempts do not repeatedly shrink a step or commit state.
Generic midpoint is symmetric, not an exact energy/unitarity claim.

::: phydrax.solver.AdaptiveTDVPPlan

::: phydrax.solver.solve_adaptive_tdvp

`FiniteVariationalSubspaceTDVPProblem` is the only route permitted to claim
norm/energy preservation. Its Cayley update requires a positive-definite
finite overlap and a Hermitian time-independent finite Hamiltonian, and retains
linear residual and reversibility evidence.

::: phydrax.solver.FiniteVariationalSubspaceTDVPProblem

::: phydrax.solver.solve_finite_subspace_tdvp

## Result and solve

::: phydrax.solver.VariationalTDVPResult

::: phydrax.solver.solve_variational_tdvp

The result retains the parameter trajectory, local time grid, energy, variance,
acceptance rate, velocity norm, status, and every linear solve result. Persistent
walkers are refreshed after each parameter update. The final frozen-model evaluation
may compute the same chain R-hat/ESS diagnostics as VMC.

A `VariationalMonteCarloState` can seed or continue TDVP. Its root key is part of the
state; supplying a different key on resume is rejected. The portable VMC state archive
can store a TDVP continuation state, but checkpoint compatibility is evaluated against
the supplied VMC step policy rather than a TDVP policy, so long TDVP workflows should
archive only at explicitly coordinated application boundaries for now.
