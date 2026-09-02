# Finite elements

## Mesh, selections, and reference elements

::: phydrax.discretization.CellBlock

---

::: phydrax.discretization.CellMesh

---

::: phydrax.discretization.EntitySelection

---

::: phydrax.discretization.FiniteElementSpec

---

::: phydrax.discretization.lagrange_element

---

::: phydrax.discretization.discontinuous_element

---

::: phydrax.discretization.raviart_thomas_element

---

::: phydrax.discretization.nedelec_element

## Fields, geometry, and preparation

::: phydrax.discretization.FiniteElementCoordinateSpec

---

::: phydrax.discretization.FiniteElementFieldSpec

---

::: phydrax.discretization.FiniteElementDofMap

---

::: phydrax.discretization.FiniteElementRuntimeData

---

::: phydrax.discretization.FiniteElementPlan

---

::: phydrax.discretization.FiniteElementDiscretization

---

::: phydrax.discretization.IntegrationDomain

---

::: phydrax.discretization.FiniteElementPrecisionPolicy

## Fixed-topology mesh motion

`FiniteElementMeshMotionPlan` consumes a fixed-route boundary coordinate provider,
solves a graph-harmonic interior extension, and returns `FiniteElementMeshRealization`.
Its runtime preserves the prepared topology and coordinate layout. Signed-Jacobian,
displacement, boundary-provider, and linear-solve evidence determine acceptance;
rejected proposals expose the base runtime and remain explicitly rejected.

::: phydrax.discretization.FiniteElementMeshMotionPlan

---

::: phydrax.discretization.FiniteElementMeshMotionPolicy

---

::: phydrax.discretization.FiniteElementMeshRealization

---

::: phydrax.discretization.FiniteElementGeometryEvidence

## Constraints

::: phydrax.linalg.ConstraintMap

---

::: phydrax.discretization.FiniteElementDirichletConstraint

---

::: phydrax.discretization.dirichlet_constraint

---

::: phydrax.discretization.affine_dof_constraint

## Weak forms and execution

::: phydrax.equations.coefficient

---

::: phydrax.equations.DiffusionAction

---

::: phydrax.equations.MassAction

---

::: phydrax.equations.SourceAction

---

::: phydrax.equations.BoundaryLoadAction

---

::: phydrax.equations.CellResidualAction

---

::: phydrax.equations.LocalFunctionalAction

---
::: phydrax.equations.CellEnergyAction

---


::: phydrax.equations.CellBilinearAction

---

::: phydrax.equations.InteriorFacetAction

---

::: phydrax.equations.FiniteElementForm

---
::: phydrax.equations.FiniteElementFunctional

---


::: phydrax.equations.compile_finite_element_functional

---

::: phydrax.equations.finite_element_form_from_functional

---

::: phydrax.equations.FiniteElementExecutionContext

---

::: phydrax.equations.FiniteElementExecutionPolicy

---

::: phydrax.equations.CompiledFiniteElementProblem


::: phydrax.equations.compile_finite_element_problem

---

::: phydrax.equations.fem.SIPGPenaltyPolicy

---

::: phydrax.equations.fem.sipg_poisson_form

---

::: phydrax.equations.fem.solve_hdg_poisson

## Local-action IR and high order

::: phydrax.equations.fem.LocalActionIR

---

::: phydrax.equations.fem.FiniteElementActionIR

---

::: phydrax.equations.fem.WorksetProgram

---

::: phydrax.discretization.fem.ReferenceNodalFamily

---

::: phydrax.discretization.fem.TensorProductTabulation

---

::: phydrax.discretization.fem.SumFactorizationPlan

---

::: phydrax.discretization.fem.QuadratureChunkPolicy

---

::: phydrax.sparse.ElementTensorOperator

---

::: phydrax.equations.fem.PartialAssemblyOperator

---

::: phydrax.equations.fem.TensorProductPartialAssemblyOperator

---

::: phydrax.integration.GaussLobattoLegendreRule

---

::: phydrax.discretization.fem.PreparedFiniteElementReference

---

::: phydrax.equations.VariationalCoefficient

---

::: phydrax.equations.FiniteElementMassPolicy

## Tensor SBP and DGSEM

::: phydrax.discretization.fem.TensorGLLSBPPlan

---

::: phydrax.discretization.fem.ElementLocalSBPReport

---

::: phydrax.discretization.fem.MappedTensorMetricPlan

---

::: phydrax.equations.fem.DGSEMConservationMethodPlan

---

::: phydrax.equations.fem.DGSEMSampledFluxCompatibilityEvidence

---

::: phydrax.equations.fem.sample_dgsem_flux_compatibility

## High-order hierarchy, mortars, and hp

::: phydrax.discretization.fem.FiniteElementPTransfer

---

::: phydrax.discretization.fem.FiniteElementPMultigridPlan

---

::: phydrax.discretization.fem.TensorFastDiagonalizationBuilder

---

::: phydrax.discretization.fem.FiniteElementPatchPreconditionerBuilder

---

::: phydrax.discretization.fem.FiniteElementMortarPlan

---

::: phydrax.discretization.fem.FiniteElementHPTransaction

---

::: phydrax.discretization.fem.FiniteElementHPEpoch

---

::: phydrax.discretization.fem.FiniteElementHPInterfacePlan

---

::: phydrax.discretization.fem.FiniteElementHPDecision

---

::: phydrax.discretization.fem.FiniteElementHPCondensationPlan

---

::: phydrax.discretization.fem.FiniteElementHPMultigridPlan

---

::: phydrax.discretization.fem.FiniteElementHPPartitionPlan

---

::: phydrax.equations.fem.DGSEMMortarCompatibilityCertificate

---

::: phydrax.equations.fem.certify_dgsem_mortar_compatibility

---

::: phydrax.discretization.fem.FiniteElementPartitionWorksetPlan

---

::: phydrax.discretization.fem.DistributedFiniteElementMortarPlan

## High-order conservation

::: phydrax.discretization.fem.FiniteElementBoundarySet

---

::: phydrax.discretization.fem.FiniteElementPeriodicTransform

---

::: phydrax.equations.fem.DGSEMConservationMethodPlan

---

::: phydrax.equations.fem.NodalDGConservationMethodPlan

---

::: phydrax.equations.fem.EntropyStableDGPlan

---

::: phydrax.equations.fem.EntropyFilterPlan

---

::: phydrax.equations.fem.ViscousDGPlan

---

::: phydrax.equations.fem.ConservativeSubcellPlan

---

::: phydrax.equations.fem.ConservationCorrectionLadderPlan

---

::: phydrax.discretization.fem.FiniteElementGeometryQualityEvidence

---

::: phydrax.equations.fem.FiniteElementGeometrySnapshot

---

::: phydrax.equations.fem.ConservativeRemapPlan

---

::: phydrax.discretization.fem.FiniteElementMeshImport

---

::: phydrax.discretization.fem.CostAwareFiniteElementPartition

---

::: phydrax.discretization.fem.FiniteElementDistributedPhasePlan
## Complete spectral hp

::: phydrax.discretization.fem.AnisotropicHPattern

---

::: phydrax.discretization.fem.TensorDeRhamComplex

---


::: phydrax.discretization.fem.HybridReferenceFamily

---


::: phydrax.discretization.fem.PersistentSemanticCache

---




::: phydrax.solver.HPNewtonKrylovBuilder

---

::: phydrax.solver.FrozenHPAdjointSchedule

## Materials and local algebra

::: phydrax.equations.ConstitutiveModel

---

::: phydrax.equations.ConstitutiveResponse

---

::: phydrax.equations.MaterialState

---

::: phydrax.equations.MaterialTransaction

---

::: phydrax.equations.fem.FiniteElementAuxiliaryEvaluation

---

::: phydrax.equations.fem.CoordinateObservation

---

::: phydrax.equations.fem.FiniteElementLeastSquaresObjective

---

::: phydrax.linalg.LocalEliminationPlan

---

::: phydrax.discretization.HDGTraceSpace

---

::: phydrax.discretization.HDGCondensationPlan

## Hierarchy and embedding




::: phydrax.discretization.FiniteElementAdaptationMap

---

::: phydrax.discretization.FiniteElementTransferBundle

---

::: phydrax.solver.FiniteElementAcceptedStepSchedule

---

::: phydrax.solver.FiniteElementTopologyTransaction

---

::: phydrax.solver.FiniteElementRestartManifest

---

::: phydrax.solver.FiniteElementResult

---

::: phydrax.solver.FiniteElementRunConfiguration

---

::: phydrax.solver.FiniteElementSolveDiagnostics

---

::: phydrax.discretization.FiniteElementErrorEstimate

---

::: phydrax.discretization.EmbeddedQuadrature

---

::: phydrax.discretization.FiniteElementEnrichment

---

::: phydrax.discretization.MultiscaleFiniteElementBasis

---

::: phydrax.discretization.PartitionedFiniteElementDofMap

---

::: phydrax.discretization.FiniteElementHaloPlan

---

::: phydrax.discretization.write_finite_element_field
