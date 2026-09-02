# Production many-body and non-Markovian quantum dynamics

The production layer deepens the representation-specific foundations without
combining their approximation semantics.

## Event-driven trajectories

`solve_event_driven_quantum_jump` evolves the unnormalized no-jump state,
brackets survival-threshold crossings, and refines event times by bisection.

::: phydrax.solver.solve_event_driven_quantum_jump

## Matrix-product evolution

`canonicalize_mps` provides mixed-canonical MPS form. Fixed-capacity compression
uses one factorization-precision SVD path for MPS, MPO, and purified-state updates
and reports the discarded squared singular-value weight before factors return to
storage precision. `tebd_step` applies nearest-neighbor first- or second-order
Trotter layers without hiding that approximation.

MPO algebra remains representation-specific: product construction, adjoint,
addition, composition, MPS action, and compression operate locally on the open
chain. Bra–MPO–ket and MPO–MPO environments evaluate expectations, Frobenius
norms, and Hermiticity residuals without global dense materialization. Dense MPS,
MPO, and LPDO conversion is explicitly capacity bounded.

::: phydrax.tensor_network.canonicalize_mps

::: phydrax.tensor_network.apply_mpo

::: phydrax.tensor_network.mps_mpo_expectation

::: phydrax.tensor_network.mpo_hermiticity_residual

::: phydrax.tensor_network.tebd_step

MPS trajectories and locally purified evolution retain separate stochastic,
channel, bond, purification, and canonical approximation evidence.

Two-site DMRG prepares a fixed open-chain problem, audits MPO Hermiticity through
network contractions, solves local effective Hamiltonians through
`phydrax.linalg`, and converges on a truncation-aware Galerkin residual rather
than energy change alone. It retains the best-energy state and reports every
local eigensolver residual and discarded bond weight.

::: phydrax.solver.DMRGPolicy

::: phydrax.solver.solve_dmrg

One-site matrix-product TDVP uses the symmetric projector-splitting sweep:
forward site evolution, backward bond evolution, then the mirrored reverse
sweep. Real- and imaginary-time modes are distinct from sampled variational
TDVP. Normalization is always an explicit policy choice and defaults off, so
norm loss remains observable.

::: phydrax.solver.MatrixProductTDVPPolicy

::: phydrax.solver.solve_matrix_product_tdvp

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
