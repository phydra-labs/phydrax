# Multi-fidelity workflows

`phydrax.fidelity` owns model-agnostic fidelity identities, directed relations,
sparse observations, physical-case splits, and portable corpus archives. Numerical
estimators and learned models remain in their existing integration, UQ, neural-operator,
and ROM namespaces.

## Hierarchies

::: phydrax.fidelity.FidelityLevelSpec

---

::: phydrax.fidelity.FidelityRelation

---

::: phydrax.fidelity.FidelityHierarchy

---

::: phydrax.fidelity.FidelityPath

## Cases and observations

::: phydrax.fidelity.FidelityCaseSpec

---

::: phydrax.fidelity.FidelityEvaluation

---

::: phydrax.fidelity.FidelityDataset

---

::: phydrax.fidelity.FidelityDatasetSplit

---

::: phydrax.fidelity.split_fidelity_dataset

## Persistence

::: phydrax.fidelity.write_fidelity_dataset

---

::: phydrax.fidelity.read_fidelity_dataset

## Coupled estimation

::: phydrax.integration.FidelityBatchEvaluation

---

::: phydrax.integration.FidelityMultilevelSampler

---

::: phydrax.integration.fidelity_multilevel_target

## Gaussian processes and acquisition

::: phydrax.uq.AutoregressiveFidelityKernel

---

::: phydrax.uq.FidelityGaussianProcess

---

::: phydrax.uq.MultiOutputGaussianProcessKernelFitPolicy

---

::: phydrax.uq.fit_multioutput_gaussian_process_kernel

---

::: phydrax.uq.TargetVarianceAcquisitionPolicy

---

::: phydrax.uq.select_fidelity_acquisition

## Physics-informed neural fields

::: phydrax.fidelity.FidelitySplitRequirements

---

::: phydrax.terms.PreparedFidelityObservation

---

::: phydrax.terms.prepare_fidelity_observation_penalty

---

::: phydrax.solver.FidelityFieldTransfer

---

::: phydrax.solver.FidelityPINNStage

---

::: phydrax.solver.FidelityPINNResult

---

::: phydrax.solver.prepare_fidelity_pinn_stage

---

::: phydrax.solver.bind_fidelity_pinn_level

---

::: phydrax.solver.condition_fidelity_correction

---

::: phydrax.solver.FidelityPINNEvaluation

---

::: phydrax.solver.evaluate_fidelity_pinn

## Neural operators and ROM

::: phydrax.nn.operator.architectures.FidelityCorrectionOperator

---

::: phydrax.nn.operator.training.prepare_fidelity_operator_dataset

---

::: phydrax.rom.ROMFidelityEvaluator
