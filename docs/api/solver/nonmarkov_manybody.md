# Non-Markovian and many-body open quantum systems

Phydrax uses representation-specific open-system solvers. A finite truncation is
never presented as an exact unbounded-Hilbert result.

## Exact Gaussian bosonic sector

`BosonicGaussianState` represents means and covariances under the declared
canonical commutation convention. `GaussianLindbladProblem` evolves quadratic
Hamiltonian and linear-noise systems without a Fock cutoff.

::: phydrax.metrix.BosonicGaussianState

::: phydrax.metrix.BosonicGaussianChannel

::: phydrax.solver.GaussianLindbladProblem

## Quantum trajectories and Fock spaces

`QuantumJumpProblem` unravels Lindblad dynamics into pure-state trajectories.
`BosonicFockSpace` supplies explicit local cutoffs, matrix-free ladder actions,
and top-occupation evidence.

::: phydrax.solver.QuantumJumpProblem

::: phydrax.solver.solve_quantum_jump_ensemble

::: phydrax.operators.quantum.BosonicFockSpace

## Non-Markovian embeddings and HEOM

Pseudomodes enlarge the physical system into a Markovian model. HEOM represents
bath memory with a fixed auxiliary hierarchy; only the root auxiliary is a
physical density matrix.

::: phydrax.operators.quantum.BathCorrelationExpansion

::: phydrax.solver.PseudomodeEmbeddingProblem

::: phydrax.solver.HEOMProblem

## Memory kernels and TCL

`MemoryKernelMasterEquation` and `TimeLocalOpenSystemProblem` remain separate
from `LindbladProblem`. Negative time-local rates are not clipped. Bounded
small-system dynamical maps may be checked through Choi positivity.

::: phydrax.solver.MemoryKernelMasterEquation

::: phydrax.solver.TimeLocalOpenSystemProblem

::: phydrax.solver.DynamicalMapPhysicality

## Tensor and process networks

The `phydrax.tensor_network` package provides fixed-capacity MPS, MPO, locally
purified density, two-site gate truncation, and process-tensor MPO contracts.
Every bond truncation reports discarded weight.
