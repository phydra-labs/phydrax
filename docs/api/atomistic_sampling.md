# Atomistic sampling and free energy

## Collective variables
::: phydrax.atomistic.sampling.AbstractCollectiveVariableProgram


::: phydrax.atomistic.sampling.CollectiveVariableMetric

::: phydrax.atomistic.sampling.CollectiveVariablePlan

::: phydrax.atomistic.sampling.PreparedCollectiveVariable

::: phydrax.atomistic.sampling.CollectiveVariableProgram
::: phydrax.atomistic.sampling.ModelCollectiveVariableProgram


::: phydrax.atomistic.sampling.CollectiveVariableEvaluation

## Bias methods

::: phydrax.atomistic.sampling.AtomisticBiasPlan

::: phydrax.atomistic.sampling.PreparedAtomisticBias

::: phydrax.atomistic.sampling.AtomisticBiasState

::: phydrax.atomistic.sampling.PreparedBiasedDynamics

::: phydrax.atomistic.sampling.BiasedDynamicsCheckpointPlan

::: phydrax.atomistic.sampling.write_biased_dynamics_checkpoint

::: phydrax.atomistic.sampling.read_biased_dynamics_checkpoint
::: phydrax.atomistic.sampling.LearnedFreeEnergyBiasPlan

::: phydrax.atomistic.sampling.PreparedLearnedFreeEnergyBias

::: phydrax.atomistic.sampling.RestrainedMeanForcePlan

::: phydrax.atomistic.sampling.estimate_restrained_free_energy_gradient

::: phydrax.atomistic.sampling.fit_free_energy_model


## Thermodynamic states and multistate execution

::: phydrax.atomistic.AtomisticPhaseSpaceMeasurePlan

::: phydrax.atomistic.AtomisticThermodynamicStatePlan

::: phydrax.atomistic.PreparedThermodynamicStateTable

::: phydrax.atomistic.sampling.AtomisticCanonicalSamplingQualification


::: phydrax.atomistic.sampling.AtomisticReplicaExchangePlan

::: phydrax.atomistic.sampling.AtomisticSAMSPlan

::: phydrax.atomistic.sampling.AtomisticSAMSState

::: phydrax.atomistic.sampling.AtomisticMultistatePlan

::: phydrax.atomistic.sampling.PreparedAtomisticMultistate

::: phydrax.atomistic.sampling.AtomisticMultistateState

::: phydrax.atomistic.sampling.AtomisticMultistateIteration

::: phydrax.atomistic.sampling.AtomisticMultistateSegmentPlan

::: phydrax.atomistic.sampling.AtomisticMultistateSegmentResult

::: phydrax.atomistic.sampling.AtomisticMultistateCheckpointPlan

::: phydrax.atomistic.sampling.write_atomistic_multistate_checkpoint

::: phydrax.atomistic.sampling.read_atomistic_multistate_checkpoint

## Free-energy observations and estimators

::: phydrax.uq.ReducedPotentialDataset

::: phydrax.uq.ReducedWorkDataset

::: phydrax.uq.ThermodynamicDerivativeDataset

::: phydrax.uq.FreeEnergySelectionPlan

::: phydrax.uq.FreeEnergySelectionEvidence

::: phydrax.uq.FreeEnergyResult

::: phydrax.uq.reduced_potential_dataset_from_multistate

::: phydrax.uq.reduced_potential_dataset_from_alchemical_evaluation

::: phydrax.uq.reduced_work_dataset_from_alchemical_switching

::: phydrax.uq.free_energy_perturbation

::: phydrax.uq.thermodynamic_integration

::: phydrax.uq.bennett_acceptance_ratio

::: phydrax.uq.multistate_bennett_acceptance_ratio

## Free-energy networks

::: phydrax.uq.FreeEnergyEdgeObservation

::: phydrax.uq.FreeEnergyNetworkPlan

::: phydrax.uq.FreeEnergyNetworkResult

::: phydrax.uq.analyze_free_energy_network

::: phydrax.uq.SparseReducedPotentialDataset

::: phydrax.uq.SparsePairwiseFreeEnergyNetworkResult

::: phydrax.uq.sparse_pairwise_free_energy_network


## Targeted maps


::: phydrax.uq.TargetedMapPlan

::: phydrax.uq.TargetedFreeEnergyProblem

::: phydrax.uq.evaluate_targeted_work

::: phydrax.uq.fit_targeted_free_energy_map
