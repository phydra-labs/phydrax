# Cardiovascular applications

`phydrax.applications.cardiovascular` is the canonical public application surface for
composing cardiac anatomy, electrophysiology, mechanics, circulation, hemodynamics,
observations, personalization, and auditable execution. The root facade owns only
cross-domain quantity, case, execution, and release contracts; numerical domain symbols
remain owned by their named subpackages.

The implementation is bounded, fixed-capacity, fail-closed research software. A
successful numerical evidence record applies only to the declared model, discretization,
input support, tolerances, precision, and execution route. It is not clinical validation,
diagnostic performance, treatment guidance, regulatory clearance, or a commercial
licence.

## Public layout

```python
from phydrax.applications import cardiovascular as cardio

quantity = cardio.cardiovascular_quantity("transmembrane_potential")
coordinate_plan = cardio.anatomy.HarmonicCoordinatePlan
reaction_model = cardio.electrophysiology.TenTusscherPanfilov2006Model
passive_material = cardio.mechanics.HolzapfelOgden2009TensionOnlyEnergy
closed_loop = cardio.circulation.biventricular_closed_loop
flow_plan = cardio.hemodynamics.FixedWallLBMPlan
pv_plan = cardio.observations.PressureVolumeLoopPlan
inverse_problem = cardio.personalization.ElectrophysiologyInverseProblem
```

Shared substrates retain their generic public owners: use
`phydrax.equations.TensorDiffusionAction`, `phydrax.ArrayArchiveLimits`, and
`phydrax.lifecycle.SupportBundleAuthorization` rather than cardiovascular copies.

## Cross-domain contracts

::: phydrax.applications.cardiovascular.CardiovascularQuantitySpec

---

::: phydrax.applications.cardiovascular.CardiovascularCaseManifest

---

::: phydrax.applications.cardiovascular.CardiovascularExecutionManifest

---

::: phydrax.applications.cardiovascular.CardiovascularMultiratePlan

---

::: phydrax.applications.cardiovascular.CardiovascularLifecycleCheckpointCodec

---

::: phydrax.applications.cardiovascular.CardiovascularReleaseAssessment

## Anatomy

The anatomy facade owns image identities, semantic boundary roles, harmonic coordinate
solves, oriented chamber surfaces, Purkinje attachments, field transfers, and ventricular
microstructure. Coordinate and microstructure candidates must be committed only after
their evidence is successful.

::: phydrax.applications.cardiovascular.anatomy.HarmonicCoordinatePlan

---

::: phydrax.applications.cardiovascular.anatomy.VentricularMicrostructurePlan

## Electrophysiology

The electrophysiology facade separates phenomenological monodomain studies from physical
reaction--diffusion integration, bidomain routes, eikonal and Purkinje conduction, pacing,
regional assignment, and named atrial, nodal, Purkinje, and ventricular cell models.
`MonodomainStatus` belongs to the phenomenological route;
`PhysicalMonodomainStatus` belongs to the physical integration route.

::: phydrax.applications.cardiovascular.electrophysiology.PhenomenologicalMonodomainPlan

---

::: phydrax.applications.cardiovascular.electrophysiology.PhysicalMonodomainPlan

---

::: phydrax.applications.cardiovascular.electrophysiology.ActivationObservationPlan

## Mechanics

The mechanics facade exposes passive constitutive energies, exact and finite-bulk material
routes, chamber loads and supports, active stress and strain, reaction-driven contraction,
electromechanical coupling, growth epochs, sarcomere kinetics, and unloaded-reference
recovery. Evidence labels such as `active-mechanics-only` do not imply whole-organ or
clinical validity.

::: phydrax.applications.cardiovascular.mechanics.FiniteBulkCardiacMaterial

---

::: phydrax.applications.cardiovascular.mechanics.ActivationDrivenContractionPlan

---

::: phydrax.applications.cardiovascular.mechanics.GrowthPlan

## Circulation and hemodynamics

Circulation owns 0D closed loops, valves, vascular 1D models, coronary flow, devices,
oxygen transport, periodic closure, and conservation ledgers. Hemodynamics owns bounded
fixed-wall LBM/MAC comparisons, rheology, terminal measurements, immersed FSI, ALE
transitions, and leaflet contact workflows. Coupling does not erase either side's validity
limits.

::: phydrax.applications.cardiovascular.circulation.CirculationNetwork

---

::: phydrax.applications.cardiovascular.circulation.pressure_volume_work

---

::: phydrax.applications.cardiovascular.hemodynamics.FixedWallLBMPlan

---

::: phydrax.applications.cardiovascular.hemodynamics.LeafletContactWorkflowPlan

## Observations and personalization

Observation plans preserve time bases, spatial frames, gauges, references, validity masks,
and data-rights identities. Personalization exposes bounded parameter, likelihood,
experimental-design, random-field, cohort, surrogate-refusal, and full-native reanalysis
contracts. A fitted parameter or calibrated surrogate is not patient-specific clinical
evidence.

::: phydrax.applications.cardiovascular.observations.ObservationRecord

---

::: phydrax.applications.cardiovascular.observations.PressureVolumeLoopPlan

---

::: phydrax.applications.cardiovascular.personalization.CardiacParameterSchema

---

::: phydrax.applications.cardiovascular.personalization.MultimodalLikelihoodPlan

## End-to-end example

Run `python examples/cardiovascular_platform.py`. The example constructs harmonic
coordinates and ventricular microstructure, advances phenomenological electrophysiology,
extracts activation, verifies checkpoint replay, compares circulation and observation
pressure--volume work, and proves that incomplete release evidence cannot authorize a
commercial release. Every evidence failure raises instead of being silently accepted.

The detailed domain guides in the **Cardiovascular platform** navigation section specify
supported routes, qualification boundaries, benchmark commands, and refusal behavior.
