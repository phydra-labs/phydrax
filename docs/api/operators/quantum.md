# Quantum

Complex matrix algebra plus closed- and open-system quantum dynamics on labeled
`DomainFunction`s. See [Quantum operators and dynamics](../../guides_quantum.md) for
conventions, examples, and the distinction between matrix, vector-field, and Poisson
brackets.

## Matrix algebra

::: phydrax.operators.commutator

::: phydrax.operators.anticommutator

::: phydrax.operators.quantum_bracket

## States and observables

::: phydrax.operators.state_norm_residual

::: phydrax.operators.state_expectation

::: phydrax.operators.density_expectation

::: phydrax.operators.observable_variance

::: phydrax.operators.density_from_factor

## Quantum information

::: phydrax.operators.purity

::: phydrax.operators.von_neumann_entropy

::: phydrax.operators.state_fidelity

::: phydrax.operators.density_fidelity

::: phydrax.operators.trace_distance

## Variational amplitudes and local operators

::: phydrax.operators.LogAmplitude

::: phydrax.operators.AmplitudeRatio

::: phydrax.operators.sampling_log_weight

::: phydrax.operators.amplitude_ratio

::: phydrax.operators.AbstractLocalQuantumOperator

::: phydrax.operators.LocalOperatorEstimate

::: phydrax.operators.LocalOperatorStatus

::: phydrax.operators.evaluate_local_operator

::: phydrax.operators.ConnectedConfigurations

::: phydrax.operators.AbstractDiscreteQuantumOperator

::: phydrax.operators.CallableDiscreteQuantumOperator

### Continuum molecular electrons

Electronic VMC is admitted by `ElectronicVMCResourcePlan`; electron,
determinant, pair-storage, kinetic-trace, and caller resource capacities remain
finite and explicit. This removes the former global four-electron ceiling
without making an unrestricted-scaling claim.

::: phydrax.operators.ElectronicKineticPolicy

::: phydrax.operators.quantum.ElectronicVMCResourcePlan

::: phydrax.operators.quantum.StochasticElectronicKineticPolicy

::: phydrax.operators.quantum.ElectronicIntegralHamiltonian

::: phydrax.operators.quantum.periodic_coulomb_energy


::: phydrax.operators.ElectronicCoulombHamiltonian

::: phydrax.operators.electronic_initial_walkers

::: phydrax.operators.harmonic_mean_electron_proposal

### Finite symmetry sectors

::: phydrax.operators.FiniteSignedPermutationSymmetry

::: phydrax.operators.SymmetryProjectedAmplitude

## Composite systems

::: phydrax.operators.tensor_product

::: phydrax.operators.partial_trace

::: phydrax.operators.embed_operator

## Finite mode reduction

::: phydrax.operators.quantum.NamedModeOperator

::: phydrax.operators.quantum.ModeReductionProblem

::: phydrax.operators.quantum.ModeReductionPolicy

::: phydrax.operators.quantum.ModeReductionPlan

::: phydrax.operators.quantum.PreparedModeReduction

::: phydrax.operators.quantum.plan_mode_reduction

::: phydrax.operators.quantum.prepare_mode_reduction

::: phydrax.operators.quantum.refresh_mode_reduction

::: phydrax.operators.quantum.compare_mode_resolutions

## Circuit-QED modes

::: phydrax.operators.quantum.ChargeBasis

::: phydrax.operators.quantum.OscillatorBasis

::: phydrax.operators.quantum.TransmonParameters

::: phydrax.operators.quantum.FluxoniumParameters

::: phydrax.operators.quantum.HarmonicModeParameters

::: phydrax.operators.quantum.transmon_mode_problem

::: phydrax.operators.quantum.fluxonium_mode_problem

::: phydrax.operators.quantum.harmonic_mode_problem

## Logical subspaces and gate quality

::: phydrax.operators.quantum.BasisStateSubspace

::: phydrax.operators.quantum.DenseQuantumSubspace

::: phydrax.operators.quantum.basis_state_subspace

::: phydrax.operators.quantum.project_quantum_operator

::: phydrax.operators.quantum.unitary_gate_quality

::: phydrax.operators.quantum.finite_channel_gate_quality

::: phydrax.operators.quantum.coherent_pauli_expansion

## Explicit local quantum programs

The local-operation namespace uses an explicit ordered Hilbert factorization and
does not materialize global embedded operators. See
[Dense local quantum programs](../../guides_quantum_programs.md).

::: phydrax.operators.quantum.HilbertRegisterLayout

::: phydrax.operators.quantum.LocalUnitaryOperation

::: phydrax.operators.quantum.LocalKrausChannelOperation

::: phydrax.operators.quantum.QuantumProgram

### Parameterized programs and local observables

`QuantumProgramTemplate` lowers angle-bound Pauli rotations and fixed local
operations to the canonical numeric `QuantumProgram`; it is not independently
executable.

::: phydrax.operators.quantum.LocalObservable

::: phydrax.operators.quantum.local_state_expectation

::: phydrax.operators.quantum.local_density_expectation

::: phydrax.operators.quantum.PauliRotationInstruction

::: phydrax.operators.quantum.QuantumProgramTemplate

::: phydrax.operators.quantum.materialize_quantum_program

::: phydrax.operators.quantum.apply_local_operator_to_state

::: phydrax.operators.quantum.apply_local_unitary_to_state

::: phydrax.operators.quantum.conjugate_local_density

::: phydrax.operators.quantum.apply_local_kraus_to_density

::: phydrax.operators.quantum.kraus_trace_preservation_residual

### Canonical finite channels

`FiniteCPTPMap` is the representation-independent finite Choi action. Kraus,
Choi, superoperator, unitary, canonical-program, and specialized local-channel
adapters retain CP/TP/reconstruction evidence. Invalid input is not projected
or normalized.

::: phydrax.operators.quantum.FiniteCPTPMap

::: phydrax.operators.quantum.finite_cptp_from_kraus

::: phydrax.operators.quantum.finite_cptp_from_choi

::: phydrax.operators.quantum.finite_cptp_from_superoperator

::: phydrax.operators.quantum.compose_finite_cptp

::: phydrax.operators.quantum.tensor_finite_cptp

::: phydrax.operators.quantum.factor_finite_cptp

These certificates apply only to the represented finite dimensions. Gaussian,
process-tensor, and tensor-local channel types remain specialized public
representations and use explicit adapters rather than aliases.

### Bounded named amplitudes

The public catalog under `phydrax.nn.quantum` contains
`JastrowSpinAmplitude`, `RestrictedBoltzmannAmplitude`,
`AutoregressiveSpinAmplitude`, `SlaterJastrowAmplitude`,
`CircuitAmplitude`, `TensorNetworkAmplitude`, `PeriodicFermiNet`, and the
existing `FermiNet`. There is no model registry. Local Jastrow/RBM cache
providers use the root incremental Markov target contract.

## Structural residuals

::: phydrax.operators.hermiticity_residual

::: phydrax.operators.unit_trace_residual

## Closed-system dynamics

::: phydrax.operators.schrodinger_residual

::: phydrax.operators.heisenberg_residual

::: phydrax.operators.von_neumann_residual

## Open-system dynamics

::: phydrax.operators.lindblad_dissipator

::: phydrax.operators.lindblad_residual

## Related complex linear algebra

::: phydrax.operators.conjugate

::: phydrax.operators.adjoint

::: phydrax.operators.real_part

::: phydrax.operators.imag_part

## Geometric vector-field bracket

::: phydrax.operators.lie_bracket
