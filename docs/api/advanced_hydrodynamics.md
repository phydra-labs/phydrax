# Advanced hydrodynamics

::: phydrax.applications.hydrodynamics.GraphCapillarityPlan

---

::: phydrax.equations.IncidentWavePlan

---

::: phydrax.equations.WaveComponent

---

::: phydrax.applications.hydrodynamics.WaveForcingPlan

---

::: phydrax.applications.hydrodynamics.FreeSurfaceRezonePlan

---

::: phydrax.applications.hydrodynamics.GraphShorelineEventPlan

---

::: phydrax.applications.hydrodynamics.MappedRigidHydroelasticBodyPlan

---

::: phydrax.applications.hydrodynamics.RigidHydroelasticALEMethod

---

## Two-phase VOF

The structured VOF product is documented in
[Two-phase hydrodynamics](../guides_two_phase_hydrodynamics.md). Its
interface-geometry and capillarity owners are on the finite-volume page:
`StructuredPLICPlan`, `plane_volume_fraction`, `HeightFunctionCurvaturePlan`,
`CurvatureStatus`, and `MACBalancedCapillaryOperator`
(see [Structured finite volume](discretization/finite_volume.md#mac-interface-geometry-and-balanced-capillarity)).
The implicit viscous stage is `phydrax.solver.MACVariationalViscosityPlan`.

::: phydrax.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan

---
::: phydrax.applications.two_phase_flow.alpha_bound_tolerance

---


::: phydrax.applications.two_phase_flow.PreparedIncompressibleTwoPhaseVOF

---

::: phydrax.applications.two_phase_flow.TwoPhaseVOFState

---

::: phydrax.applications.two_phase_flow.TwoPhaseVOFView

---

::: phydrax.applications.two_phase_flow.TwoPhaseInterfaceGeometry

---

::: phydrax.applications.two_phase_flow.PLICGeometry

---

::: phydrax.applications.two_phase_flow.TwoPhaseStepEvidence

---

::: phydrax.applications.two_phase_flow.TwoPhaseVOFLedger

---

::: phydrax.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod

---

::: phydrax.applications.two_phase_flow.TwoPhaseMaterialPlan

---

::: phydrax.applications.two_phase_flow.TwoPhaseMovingBodyPlan

---

::: phydrax.applications.two_phase_flow.two_phase_diagnostic_view

---

::: phydrax.applications.two_phase_flow.two_phase_inspection_frame

---

::: phydrax.applications.two_phase_flow.two_phase_inspection_frames

---

::: phydrax.applications.free_boundary.hysing_bubble_benchmark

---

## FLIP inspection

::: phydrax.equations.flip_inspection_frames
