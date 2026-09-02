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

The `phydrax.tensor_network` package provides fixed-capacity open-boundary MPS,
MPO, locally purified density, two-site gate truncation, MPO algebra and
process-tensor MPO contracts. MPO application and composition form exact local
product bonds and then compress only at the caller-declared capacity. Every bond
truncation reports retained and available rank plus discarded squared weight.
Accumulated local discarded weight is evidence from the sequence of local
approximations; it is not presented as an unproved global norm-error bound.

`TensorNetworkPrecisionPolicy` separates tensor storage, contraction,
factorization/SVD, accumulation, certification, and output roles. MPS/MPO and
locally purified contractions, canonical QR sweeps, two-site truncation,
interventions, process-tensor contraction, tomography, and local Choi checks
retain effective precision evidence. Discarded weight is accumulated and
certified before truncated factors are cast back to storage precision.

MPS, MPO, and LPDO dense materialization has an explicit element capacity.
Expectations, MPO Frobenius norms, and MPO Hermiticity residuals instead use
open-chain environments and do not allocate the global operator.

::: phydrax.tensor_network.TensorNetworkPrecisionPolicy

## Prepared labelled contractions

`ContractionStructure` separates static operand/leg topology from numerical
arrays. V1 ordinary contractions require every free label exactly once and in
the explicit output order, and every contracted label exactly twice; hyperedges
and log-semiring inference are rejected rather than assigned implicit semantics.

Planning is host-only and records the exact `opt_einsum` equation, path, FLOP
estimate, largest intermediate, workspace, optimizer, structure fingerprint,
precision policy, and resource policy. Preparation binds arrays; refresh accepts
new values only at the identical shape, dtype, operand order, and topology.
Execution is pure fixed-path JAX computation. Prepared MPS-inner and MPO-inner
adapters are the first concrete consumers; normal chain environments remain
available for sweep algorithms.

::: phydrax.tensor_network.ContractionResourcePolicy

::: phydrax.tensor_network.plan_contraction

::: phydrax.tensor_network.execute_contraction

## Static Abelian sectors

`AbelianGroup` represents an ordered direct product of U(1) and finite cyclic
charge components. Oriented `AbelianLeg` values declare charge catalogues and
fixed degeneracy capacities; `AbelianTensorLayout` enumerates the exact
charge-conserving blocks. Numerical storage is an immutable ordered tuple of
fixed-shape JAX blocks. It is not a dynamic sparse dictionary or a universal
array backend.

Abelian MPS sites use `(left:+, physical:+, right:-)` and preserve
`q_left + q_physical = q_right`. Abelian MPO sites use
`(left:+, output:-, input:+, right:-)`. Dense conversion is an explicit bounded
interoperability operation. Inner products, local expectations, canonical
sweeps, symmetry-preserving two-site gates, and TEBD retain the declared block
structure.

Two-site truncation factorizes every virtual-charge sector, then selects one
global largest-mode budget across all sectors with deterministic tie order.
Per-sector retained ranks, the selected-mode mask, and global discarded squared
weight remain explicit. Charge-breaking gates are rejected rather than
projected. Fermionic grading and non-Abelian fusion are not claimed.

::: phydrax.tensor_network.AbelianGroup

::: phydrax.tensor_network.AbelianMatrixProductState

::: phydrax.tensor_network.apply_abelian_two_site_gate
