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

The exact small-system contract is
`phydrax.operators.ELECTRONIC_MAX_ELECTRONS == 4`; all electronic construction
and sampling entry points reject larger systems.

::: phydrax.operators.ElectronicKineticPolicy

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

## Explicit local quantum programs

The local-operation namespace uses an explicit ordered Hilbert factorization and
does not materialize global embedded operators. See
[Dense local quantum programs](../../guides_quantum_programs.md).

::: phydrax.operators.quantum.HilbertRegisterLayout

::: phydrax.operators.quantum.LocalUnitaryOperation

::: phydrax.operators.quantum.LocalKrausChannelOperation

::: phydrax.operators.quantum.QuantumProgram

::: phydrax.operators.quantum.apply_local_unitary_to_state

::: phydrax.operators.quantum.conjugate_local_density

::: phydrax.operators.quantum.apply_local_kraus_to_density

::: phydrax.operators.quantum.kraus_trace_preservation_residual

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
