# Non-Markovian and many-body open quantum systems

Phydrax uses representation-specific open-system solvers. A finite truncation is
never presented as an exact unbounded-Hilbert result.

## Exact Gaussian bosonic sector

`BosonicGaussianState` represents means and covariances under the declared
canonical commutation convention. `GaussianLindbladProblem` evolves quadratic
Hamiltonian and linear-noise systems without a Fock cutoff.

Gaussian state and channel diagnostics compose `GeometryPrecisionPolicy` with
`HermitianPrecisionPolicy`. Gaussian Lindblad time stepping uses
`TemporalPrecisionPolicy`; its stationary mean and Lyapunov systems use
`phydrax.linalg` rather than incidental backend solves.

::: phydrax.metrix.BosonicGaussianState

::: phydrax.metrix.BosonicGaussianChannel

::: phydrax.solver.GaussianLindbladProblem

## Quantum trajectories and Fock spaces

`QuantumJumpProblem` unravels Lindblad dynamics into pure-state trajectories.
`BosonicFockSpace` supplies explicit local cutoffs, matrix-free ladder actions,
and top-occupation evidence.

Quantum-jump evolution uses temporal precision for state and stage arithmetic and
geometry precision for rates, normalization, observables, ensemble moments, and
statistical error. Trajectory and Fock-cutoff approximation records archive the
effective precision envelope and policy IDs beside their truncation axes.

::: phydrax.solver.QuantumJumpProblem

::: phydrax.solver.solve_quantum_jump_ensemble

::: phydrax.operators.quantum.BosonicFockSpace

## Non-Markovian embeddings and HEOM

Pseudomodes enlarge the physical system into a Markovian model. HEOM represents
bath memory with a fixed auxiliary hierarchy; only the root auxiliary is a
physical density matrix.

HEOM applies temporal precision to the entire auxiliary hierarchy. Geometry and
Hermitian policies certify only the physical root density; auxiliary operators are
not incorrectly projected onto density-matrix constraints. Hierarchy depth, bath
rank, time step, local error, and all contributing policy IDs remain in
`OpenSystemApproximationEvidence`.

::: phydrax.operators.quantum.BathCorrelationExpansion

::: phydrax.solver.PseudomodeEmbeddingProblem

::: phydrax.solver.HEOMProblem

## Memory kernels and TCL

`MemoryKernelMasterEquation` and `TimeLocalOpenSystemProblem` remain separate
from `LindbladProblem`. Negative time-local rates are not clipped. Bounded
small-system dynamical maps may be checked through Choi positivity.

Memory-kernel quadrature uses `IntegrationPrecisionPolicy` before temporal state
updates. Time-local and memory-kernel solutions retain nested temporal, integration,
geometry, and Hermitian evidence together with physicality and approximation
metadata.

::: phydrax.solver.MemoryKernelMasterEquation

::: phydrax.solver.TimeLocalOpenSystemProblem

::: phydrax.solver.DynamicalMapPhysicality

## Tensor and process networks

The `phydrax.tensor_network` package provides fixed-capacity MPS, MPO, locally
purified density, two-site gate truncation, and process-tensor MPO contracts.
Every bond truncation reports discarded weight.

`TensorNetworkPrecisionPolicy` separates tensor storage, contraction,
factorization/SVD, accumulation, certification, and output roles. MPS/MPO and
locally purified contractions, two-site truncation, interventions, process-tensor
contraction, tomography, and local Choi checks retain effective precision evidence.
Discarded weight is accumulated and certified before truncated factors are cast
back to storage precision.

::: phydrax.tensor_network.TensorNetworkPrecisionPolicy
