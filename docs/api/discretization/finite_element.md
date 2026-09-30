# Finite elements

## Canonical form elements

`form_element` declares a `FormValueSpec`, not independently chosen mapping and
conformity strings. Scalar, circulation, flux, density and component proxies
retain their declared degree/twist. In 2-D degree1, circulation and flux are
different explicit representations. Physical flux and density boundaries declare
twist; density uses signed determinant for untwisted and absolute determinant for
twisted maps. Ordinary scalar `discontinuous_element` remains a zero-form.
Full-family complexes admit top polynomial order≥0: the minimal order0 sequence
is P_nΛ0 → … → P_0Λn. Trimmed/tensor-trimmed complexes require order≥1.
An individual full `form_element` likewise admits its own polynomial order≥0.


`FiniteElementDeRhamComplex` supplies exact sparse d and metric-only sparse Gram
Hodges. Material/constitutive operators are separate weighted coordinate forms.
Use `hilbert_complex(boundary=...)` for degree vector spaces and linear operators.
`constitutive_operator(k, coefficient, boundary=...)` assembles a separate
weighted coordinate form: admitted real SPD material forms may be certified,
while complex lossy forms remain uncertified. `coefficient_dtype` complexifies
field coordinates/operators without changing real geometry or metric Hodges;
`FiniteElementPlan` propagates the same explicit dtype choice.
`reconstruction(k)` returns native `PreparedFieldReconstruction`, including
homogeneous multiblock support. `trace_complex_map(boundary_mask=...)` binds an
explicit topological facet mask and its outward-oriented entity closure. Trace
map, boundary-complex, degree-space and differential identities include the
parent realization, selected entity/DOF indices in canonical order and induced
orientation. Repeating a selection preserves those identities; equal-dimensional
patches with different selected facets are incompatible, including for prepared
operators and harmonic artifacts. Restricted relative spaces likewise retain
their parent-complex and ordered active-coordinate identity.

Declared `DiscreteFieldSpace.form_type` checks H1 degree0, H(curl) degree1 and
H(div) degree n−1. Generic intermediate-degree forms use `"HLambda"`;
L2/discontinuous/unrestricted/HLambda admit any declared degree. Conformity
therefore does not silently reinterpret every discontinuous field as density.


`transfer(target, /, *, parent_cells=None)` returns a `ComplexMap` for same-mesh
p transfer or nested h transfer with an explicit target-cell→source-cell parent
relation. Nested admission checks designated-parent containment, exact moments
and commutation. A measure sum alone does not certify global nonoverlap/coverage;
the topology transaction owns separate coverage evidence. Nonnested remeshing
is refused, not inferred from equal dimensions.


Mesh admission includes `CellMesh.from_simplices(..., dimension=...)` for
full-dimensional affine `simplex:N`, including intervals. Generic tensor meshes
are dimension1–3 even though standalone `tensor:N` bases admit n-D.
Reconstruction uses exact membership/exterior-distance queries, not an invented
box or an integration-measure guarantee. Warped hexahedral geometry needs an
explicit curved witness; planar-face hex and quadrilateral regions are native.


::: phydrax.discretization.fem.form_element

`FormBasis` retains exact polynomial coefficients, canonical entity moments and
permutation maps. Tensor-trimmed moments are product functionals: scalar interval
endpoints plus interior P_(r−2) moments, and degree1 P_(r−1) interval moments.
Higher-order moments are not silently substituted by segmented nodal integrals.
Tensor basis admission reuses one-dimensional moment-dual factors prepared with
the native rank-checked QR owner. Scalar interior factors retain their endpoint
bubble, and compiled values/derivatives use separable shifted-Legendre products,
not a global monomial Vandermonde. The complete monomial coefficient artifact
remains available for polynomial interchange; it is not the tensor evaluation
representation. Equal-order exterior derivatives use the exact interval Stokes
map in canonical component/entity order. Cube pullbacks apply the same
functional densities to the stable product tabulation.
`functional_weights_at(entity_vertices, entity_points, quadrature_weights)`
evaluates the actual entity polynomial densities on caller-declared cubature.
Lowest-order scalar vertices and circulation incidence remain canonical.

::: phydrax.discretization.fem.FormBasis


::: phydrax.discretization.fem.FormFieldReconstructionKernel


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

::: phydrax.discretization.fem.FiniteElementDeRhamComplex

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

`FiniteElementDeRhamComplex` provides tetrahedral degree-one circulation fields,
covariant mapping and native sparse metric/coordinate operators. The exact
sequence and relative PEC restriction use the same exterior protocol as other
form elements; material weights are separate from the metric pairing.
