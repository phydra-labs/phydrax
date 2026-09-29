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

::: phydrax.discretization.CellGeometrySpec

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

## Masked capacity layouts

`MaskedFiniteElementPlan` is the static P1 Lagrange plan of one `MaskedSimplexMesh`
capacity bucket; its `plan_id` depends on the family, degree, layout signature, and
precision policy only. `assemble_masked_finite_element` is a module-level compiled
entry: every layout of one signature, whatever its active counts or topology, reuses
one executable. Mass and stiffness act on `(vertex_capacity,)` vectors over one edge
relation of cell routes valid on active cells plus identity routes valid on inactive
vertex slots, so padding DOFs are pinned: operators stay square and nonsingular on
padding, and solves and transpose actions keep the capacity shape. Masked layouts
carry vertex DOFs only, so degrees other than 1 are rejected.
`constrain_masked_dofs` pins a traced DOF mask (for example `boundary_dofs`) through
identity rows and columns with a static route capacity, giving the homogeneous
Dirichlet operator.

::: phydrax.discretization.MaskedFiniteElementPlan

---

::: phydrax.discretization.MaskedFiniteElementSystem

---

::: phydrax.discretization.assemble_masked_finite_element

---

::: phydrax.discretization.constrain_masked_dofs

## Field views and point evaluation

Discrete field views are shared by every discretization family; the finite-element
factory prepares one from native tabulation, oriented DOF routes, and an inverse
cell map.

::: phydrax.discretization.fem.prepare_finite_element_field_reconstruction

---

::: phydrax.discretization.fem.FiniteElementFieldReconstructionKernel

---

::: phydrax.discretization.fem.prepare_finite_element_point_interpolation

---

::: phydrax.discretization.fem.PreparedFiniteElementPointInterpolation

---

::: phydrax.discretization.fem.prepare_finite_element_side_trace

---

::: phydrax.discretization.FiniteElementDiscretization.prepare_side_trace

---

::: phydrax.discretization.fem.reference_facet_embedding

---

::: phydrax.discretization.PreparedFieldReconstruction

---

::: phydrax.discretization.DiscreteFieldFunctionView

---

::: phydrax.discretization.DiscreteFieldEvaluator

---

::: phydrax.discretization.AbstractFieldReconstructionKernel

---

::: phydrax.discretization.FieldTracePolicy

---

::: phydrax.discretization.FieldSideBinding

---

::: phydrax.discretization.FieldQueryEvidence

---

::: phydrax.discretization.FieldQueryResult

---

::: phydrax.discretization.FieldQueryStatus

---

::: phydrax.discretization.InterpolationTransposeEvidence

## Fixed-topology mesh motion

`FiniteElementMeshMotionPlan` consumes a fixed-route boundary coordinate provider,
extends the boundary displacement to the interior along the policy's
`FiniteElementMeshMotionRoute`, and returns `FiniteElementMeshRealization`. Routes are
`HARMONIC` (graph-Laplacian extension), `LINEAR_ELASTICITY` (finite-element elasticity
with Jacobian stiffening `E = (J_max / J)**chi`), `WINSLOW` (inverse harmonic map with
an inversion barrier), `MMPDE` (steady state of the Huang–Russell moving-mesh PDE for a
monitor passed to `realize(..., monitor=...)`, relaxation time `tau`), and `PRESCRIBED`
(the provider realizes every vertex). Linear routes solve matrix-free element tensors
through one prepared Krylov solve; nonlinear routes solve their stationarity equations
through `phydrax.nonlinear` with implicit root derivatives, so realizations are
differentiable in the boundary design and the monitor parameters.
`FiniteElementMotionExtension` is the reusable route owner consumed by fixed-connectivity
finite-volume ALE and variable-patch ALE.

`realize(design, numeric_version=...)` requires a nonempty caller-owned identifier for
the proposed numeric coordinate state. That version participates in runtime identity
without changing prepared topology or coordinate layout. Sampled corner-Jacobian
evidence from the shared `MotionValidityPlan`, displacement, boundary-provider, and
route-solver evidence determine acceptance inside the compiled step; rejected proposals
expose the base runtime and remain explicitly rejected. `certify(coordinates)` is the
host-epoch Bernstein certificate of a moved state.

::: phydrax.discretization.FiniteElementMeshMotionPlan

---

::: phydrax.discretization.FiniteElementMeshMotionPolicy

---

::: phydrax.discretization.FiniteElementMeshMotionRoute

---

::: phydrax.discretization.FiniteElementMotionExtension

---

::: phydrax.discretization.FiniteElementMotionExtensionResult

---

::: phydrax.discretization.FiniteElementMeshRealization

---

::: phydrax.discretization.MotionValidityPlan

---

::: phydrax.discretization.MotionValidityPolicy

---

::: phydrax.discretization.MotionValidityEvidence

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

---

::: phydrax.equations.CompiledFiniteElementProblem.prepare_conormal_flux

---

::: phydrax.equations.CompiledFiniteElementProblem.prepare_pointwise_flux

---

::: phydrax.equations.CompiledFiniteElementProblem.certify_flux_stability

---

::: phydrax.equations.CompiledFiniteElementProblem.prepare_mass

---

::: phydrax.equations.PreparedFiniteElementMass

---

::: phydrax.equations.CompiledFiniteElementProblem.boundary_impositions

---

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

Integration-site laws implement the `MODEL`-authority slot
`AbstractConstitutiveModel`; local constitutive roots implement
`AbstractLocalImplicitMaterial`. `ConstitutiveModel` and `LocalImplicitMaterial`
are fixed analytic implementations (FIXED wherever they are held).
`LearnedConstitutiveModel` and `LearnedLocalImplicitMaterial` hold a learned model
as a trainable child bound to the slot; construction admits only models with first
input and parameter derivatives, classical `C^1` value regularity, deterministic
randomness, and a declared precision contract. A learned law's per-site
`AdmissibilityHeader` marks out-of-support or nonfinite sites (and an unresolved
learned local root) invalid; such sites keep their primal response while their
derivatives, including the consistent tangent, are NaN. `MaterialIntegrationPlan`
is neutral, so a learned law trains through it and implicit mechanics
differentiates it by the implicit function theorem.

::: phydrax.equations.AbstractConstitutiveModel

---

::: phydrax.equations.ConstitutiveModel

---

::: phydrax.equations.LearnedConstitutiveModel

---

::: phydrax.equations.ConstitutiveResponse

---

::: phydrax.equations.MaterialState

---

::: phydrax.equations.MaterialTransaction

---

::: phydrax.equations.fem.AbstractLocalImplicitMaterial

---

::: phydrax.equations.fem.LocalImplicitMaterial

---

::: phydrax.equations.fem.LearnedLocalImplicitMaterial

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




::: phydrax.discretization.FiniteElementTopologyTransfer

---

::: phydrax.discretization.FiniteElementL2Projection

---

::: phydrax.discretization.PreparedL2ProjectionTarget

---

::: phydrax.discretization.prepare_l2_projection_target

---

::: phydrax.discretization.refresh_l2_projection_target

---

::: phydrax.discretization.prepare_l2_projection_transfer

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

## Tetrahedral H(curl)

`TetrahedralNedelecSpace` is the oriented lowest-order tetrahedral Nédélec space used
by conductive electromagnetic plans. It exposes edge orientation, covariant Piola
mapping, mass and curl–curl actions, and discrete gradient/curl/divergence incidence.

::: phydrax.discretization.TetrahedralNedelecSpace
