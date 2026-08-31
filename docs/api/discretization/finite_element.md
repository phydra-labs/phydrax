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

::: phydrax.equations.fem.DGSEMFluxCompatibilityCertificate

---

::: phydrax.equations.fem.certify_dgsem_flux_compatibility

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

## Materials and local algebra

::: phydrax.equations.ConstitutiveModel

---

::: phydrax.equations.ConstitutiveResponse

---

::: phydrax.equations.FiniteElementMaterialState

---

::: phydrax.equations.FiniteElementMaterialTransaction

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
