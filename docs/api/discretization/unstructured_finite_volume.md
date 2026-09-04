# Unstructured finite volume

## Cell topology and geometry

::: phydrax.discretization.PolygonalConnectivity

---

::: phydrax.discretization.TetrahedralConnectivity
---

::: phydrax.discretization.PolyhedralConnectivity

---

::: phydrax.discretization.PreparedPolyhedralFiniteVolumeGeometry

---

::: phydrax.discretization.prepare_polyhedral_finite_volume_geometry


---

::: phydrax.discretization.UnstructuredFiniteVolumePlan

---

::: phydrax.discretization.UnstructuredFiniteVolumeDiscretization

---

::: phydrax.discretization.UnstructuredFiniteVolumeQualityReport

---

::: phydrax.discretization.FiniteVolumeFaceBlock

---

::: phydrax.discretization.evaluate_unstructured_fv_geometry

## Dynamics and boundaries

::: phydrax.discretization.UnstructuredFiniteVolumeMethodPlan

---

::: phydrax.discretization.UnstructuredFiniteVolumeBoundarySet

---

::: phydrax.discretization.PreparedUnstructuredFiniteVolumeDynamics

---

::: phydrax.discretization.UnstructuredFiniteVolumeDiagnostics

## Polynomial and WENO reconstruction

::: phydrax.discretization.CellPolynomialBasis

---

::: phydrax.discretization.CellPolynomialReconstructionPlan

---

::: phydrax.discretization.PreparedCellPolynomialReconstruction

---

::: phydrax.discretization.UnstructuredWENOZReconstructionPlan

---

::: phydrax.discretization.PreparedUnstructuredWENOZReconstruction

## Implicit and pressure-correction solvers

::: phydrax.solver.FiniteVolumeBackwardEulerPlan

---

::: phydrax.solver.PreparedFiniteVolumeBackwardEulerStep

---

::: phydrax.discretization.PreparedUnstructuredCollocatedOperators

---

::: phydrax.solver.UnstructuredPressureProjectionPlan

---

::: phydrax.solver.UnstructuredPressureCorrectionPlan

## Persistence and output

::: phydrax.discretization.write_unstructured_fv_archive

---

::: phydrax.discretization.read_unstructured_fv_archive

---

::: phydrax.solver.FiniteVolumeCaseSpec

---

::: phydrax.solver.FiniteVolumeCheckpointPlan

---

::: phydrax.solver.FiniteVolumeOutputPlan

## Motion, remap, embedded boundary, and VOF

::: phydrax.discretization.FixedConnectivityMotionPlan

---

::: phydrax.discretization.UnstructuredConservativeRemapPlan
---

::: phydrax.discretization.FiniteVolumeStageEpochTransition

---

::: phydrax.discretization.FiniteVolumeStageEpochTransfer
---

::: phydrax.solver.PreparedUnstructuredSSPRK3Runtime

---

::: phydrax.solver.UnstructuredSSPRK3EpochResult



---

::: phydrax.discretization.EmbeddedBoundaryPlan

---

::: phydrax.discretization.EmbeddedBoundaryMetrics

---

::: phydrax.discretization.UnstructuredVOFPlan

---

::: phydrax.discretization.PLICReconstruction
---

::: phydrax.discretization.VariableSurfaceTensionPolicy

---

### Conservative low-Mach LES

`UnstructuredLowMachLESPlan` is a single-device fixed-conforming 3-D tetrahedral
constitutive transport action with Favre transport and optional static KSGS.
`UnstructuredLowMachLESFixedStepMethod` adds a gauged pressure projection,
forward-Euler predictor/correction, complete pressure/flux restart state, explicit
stability/positivity bounds, and atomic rollback. It refuses 2-D/polyhedral,
periodic/open, moving/coupled, and dynamic/low-Re KSGS routes. See the normative
[LES equations](../equations/les.md#backend-support-and-refusals) and
[LES guide](../../guides_large_eddy_simulation.md#unstructured-low-mach-favre-les).

::: phydrax.equations.UnstructuredLowMachLESPlan

---

::: phydrax.equations.PreparedUnstructuredLowMachLES

---

::: phydrax.equations.UnstructuredLowMachLESState

---

::: phydrax.equations.UnstructuredLowMachLESRateResult

---

::: phydrax.equations.UnstructuredLowMachLESConservationEvidence

---

::: phydrax.solver.UnstructuredLowMachLESFixedStepMethod

---

::: phydrax.solver.UnstructuredLowMachLESRestartState

---

::: phydrax.solver.UnstructuredLowMachLESStepEvidence

---

---

::: phydrax.equations.StefanPhaseChangePlan

## AMR and interface coupling

::: phydrax.discretization.UnstructuredAMRHierarchyPlan

---

::: phydrax.discretization.UnstructuredAMRFluxRegister

---

::: phydrax.discretization.UnstructuredOversetPlan

---

::: phydrax.discretization.PeriodicSlidingInterfacePlan

## Triangle specialization

::: phydrax.discretization.TriangleFiniteVolumePlan

---

::: phydrax.discretization.PreparedTriangleWLSQ

---

::: phydrax.discretization.TriangleKExactReconstructionPlan

---

::: phydrax.discretization.TriangleViscousFluxPlan
