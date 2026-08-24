# Production many-body and non-Markovian quantum dynamics

The production layer deepens the representation-specific foundations without
combining their approximation semantics.

## Event-driven trajectories

`solve_event_driven_quantum_jump` evolves the unnormalized no-jump state,
brackets survival-threshold crossings, and refines event times by bisection.

::: phydrax.solver.solve_event_driven_quantum_jump

## Matrix-product evolution

`canonicalize_mps` provides mixed-canonical MPS form. `tebd_step` applies
nearest-neighbor first- or second-order Trotter layers and returns discarded
bond weight.

::: phydrax.tensor_network.canonicalize_mps

::: phydrax.tensor_network.tebd_step

MPS trajectories and locally purified evolution retain separate stochastic and
deterministic approximation evidence.

## Production HEOM and cross-representation campaigns

HEOM continuation refines hierarchy depth while preserving root-state identity.
`lorentzian_qubit_comparison` evaluates pseudomode, HEOM, and direct-memory
representations on one physical design.

::: phydrax.solver.solve_heom_continuation

::: phydrax.solver.lorentzian_qubit_comparison

## Additional representations

- adaptive Fock continuation;
- fermionic Gaussian covariance dynamics;
- process-comb causality diagnostics;
- neural jump-state projection.

Every result retains its own cutoff, bond, hierarchy, sampling, or projection
error. None is folded into a universal convergence scalar.
