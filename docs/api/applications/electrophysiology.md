# Neural electrophysiology

`phydrax.applications.electrophysiology` owns physical compartmental and point-cell dynamics, bounded spike transport, local plasticity, neural execution, and host-only morphology/circuit interchange. Artificial surrogate-gradient recurrence remains in `phydrax.nn.layers`; regional neural masses and BOLD remain in `phydrax.applications.neuroscience`.

## Cells and cable execution

::: phydrax.applications.electrophysiology.CellMorphologyPlan

---

::: phydrax.applications.electrophysiology.MembraneProgram

---

::: phydrax.applications.electrophysiology.CableSolverPlan

---

::: phydrax.applications.electrophysiology.step_cable

---

::: phydrax.applications.electrophysiology.LeakyIntegrateAndFire

---

::: phydrax.applications.electrophysiology.AdaptiveExponentialIntegrateAndFire

## Synapses and learning

::: phydrax.applications.electrophysiology.SynapseNetworkPlan

---

::: phydrax.applications.electrophysiology.SynapseConnection

---

::: phydrax.applications.electrophysiology.PairSTDPPlan

---

::: phydrax.applications.electrophysiology.EligibilitySTDPPlan

## Coupled execution

::: phydrax.applications.electrophysiology.NeuralCellPlan

---

::: phydrax.applications.electrophysiology.NeuralNetworkPlan

---

::: phydrax.applications.electrophysiology.initialize_neural_network

---

::: phydrax.applications.electrophysiology.step_neural_network

---

::: phydrax.applications.electrophysiology.run_neural_network

---

::: phydrax.applications.electrophysiology.apply_neural_relation_event

## Interchange

::: phydrax.applications.electrophysiology.parse_swc_file

---

::: phydrax.applications.electrophysiology.import_sonata

---

::: phydrax.applications.electrophysiology.prepare_sonata_network

---

::: phydrax.applications.electrophysiology.export_sonata
