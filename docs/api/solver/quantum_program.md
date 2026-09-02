# Local quantum programs

See [Local quantum programs](../../guides_quantum_programs.md) for target-order,
state, physicality, resource, routing, and transformation contracts.

## Policy, cost, and lifecycle

::: phydrax.solver.DenseQuantumProgramPolicy

::: phydrax.solver.DenseQuantumProgramCostEstimate

::: phydrax.solver.DenseQuantumProgramPlan

::: phydrax.solver.PreparedDenseQuantumProgram

::: phydrax.solver.plan_dense_quantum_program

::: phydrax.solver.prepare_dense_quantum_program

::: phydrax.solver.refresh_dense_quantum_program

::: phydrax.solver.execute_dense_quantum_program

## Local observables and parameterized gradients

::: phydrax.solver.DenseQuantumObservablePolicy

::: phydrax.solver.DenseQuantumObservableCostEstimate

::: phydrax.solver.DenseQuantumObservablePlan

::: phydrax.solver.DenseQuantumExpectationDiagnostics

::: phydrax.solver.DenseQuantumExpectationResult

::: phydrax.solver.plan_dense_quantum_observables

::: phydrax.solver.evaluate_dense_quantum_observables

::: phydrax.solver.PreparedDenseQuantumTemplate

::: phydrax.solver.prepare_dense_quantum_template

::: phydrax.solver.execute_dense_quantum_template

::: phydrax.solver.ParameterShiftPlan

::: phydrax.solver.ParameterShiftJacobianResult

::: phydrax.solver.plan_parameter_shift

::: phydrax.solver.evaluate_parameter_shift_jacobian

## Open-chain MPS lifecycle

::: phydrax.solver.MPSQuantumProgramPolicy

::: phydrax.solver.MPSQuantumProgramPlan

::: phydrax.solver.PreparedMPSQuantumProgram

::: phydrax.solver.plan_mps_quantum_program

::: phydrax.solver.prepare_mps_quantum_program

::: phydrax.solver.refresh_mps_quantum_program

::: phydrax.solver.execute_mps_quantum_program

## Open-chain LPDO lifecycle

::: phydrax.solver.LPDOQuantumProgramPolicy

::: phydrax.solver.LPDOQuantumProgramPlan

::: phydrax.solver.PreparedLPDOQuantumProgram

::: phydrax.solver.plan_lpdo_quantum_program

::: phydrax.solver.prepare_lpdo_quantum_program

::: phydrax.solver.refresh_lpdo_quantum_program

::: phydrax.solver.execute_lpdo_quantum_program

## Results and evidence

::: phydrax.solver.DenseQuantumProgramStatus

::: phydrax.solver.DenseQuantumOperationEvidence

::: phydrax.solver.DenseQuantumProgramDiagnostics

::: phydrax.solver.DenseQuantumProgramResult

::: phydrax.solver.MPSQuantumProgramDiagnostics

::: phydrax.solver.MPSQuantumProgramResult

::: phydrax.solver.LPDOQuantumProgramDiagnostics

::: phydrax.solver.LPDOQuantumProgramResult

## Instruments and experiments

::: phydrax.solver.QuantumPOVM

::: phydrax.solver.QuantumInstrument

::: phydrax.solver.QuantumExperimentProgram

::: phydrax.solver.prepare_quantum_experiment

::: phydrax.solver.execute_quantum_experiment_exact

::: phydrax.solver.sample_quantum_experiment

::: phydrax.solver.estimate_quantum_experiment_gradient

## Compilation and controls

::: phydrax.solver.HardwareTopology

::: phydrax.solver.QuantumCompilationPolicy

::: phydrax.solver.compile_quantum_program

::: phydrax.solver.FixedGridQuantumControl

::: phydrax.solver.discretize_fixed_grid_control

## Tensor open systems and process learning

::: phydrax.solver.MPOHamiltonian

::: phydrax.solver.MPOLindbladian

::: phydrax.solver.LPDOChannelEvolutionPlan

::: phydrax.solver.evolve_lpdo_local_channels

::: phydrax.solver.StinespringProcessModel

::: phydrax.solver.fit_stinespring_process_model

::: phydrax.solver.QuantumDigitalTwinState

## Service boundary

::: phydrax.solver.QuantumServicePolicy

::: phydrax.solver.QuantumServiceRequest

::: phydrax.solver.admit_quantum_service_request
