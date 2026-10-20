"""Consumer view of an installed Phydrax wheel.

Checked by `tools/check_installed_typing.py` from outside the repository, so every
`phydrax` import resolves to the installed distribution. Deliberately invalid
lines carry `# ty: ignore[<rule>]`; unused suppressions are errors.
"""

from typing import assert_type, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import phydrax as phx
import phydrax.typing as pt
from phydrax import (
    equations as equations,
    geometry as geometry,
    linalg as linalg,
    rheology as rheology,
    solver as solver,
)
from phydrax.applications import contact as contact, solid_mechanics as solid
from phydrax.applications.incompressible_flow import (
    VOLUME_FRACTION_CFD_DEM_SUPPORT_TUPLE,
)
from phydrax.discretization import (
    DistributedSpectralExecutionPlan,
    mpm as mpm,
    particle as particle,
    spatial as spatial,
    SpectralExecutionResult,
    SpectralGlobalDiagnostics,
    SpectralMeshTopology,
    SpectralPrecisionPolicy,
)
from phydrax.discretization.flip import FLIPViscousEvidence
from phydrax.equations import ChemicalComponentCatalog
from phydrax.linalg.svd import (
    prepare_svd,
    PreparedSVDSolve,
    projector_action,
    RandomizedSVD,
    SingularSubspaceResponse,
    svd,
    SVDProblem,
    SVDRankEvidence,
    SVDSolvePolicy,
    SVDSolveResult,
)
from phydrax.precision import precision_dtype_name, ScalarPrecisionDType
from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    PreparedProjectorMonteCarlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloAnalysis,
    ProjectorMonteCarloResult,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStepResult,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
)
from phydrax.uq import CorrelatedRatioResult


class ComponentDim(pt.Dim, minimum=1):
    """Number of chemical components."""


def tensor_forms(
    masses: pt.Float64[ComponentDim],
    table: pt.HostFloat64[pt.AnyDim, ComponentDim],
    count: pt.Size[ComponentDim],
    names: pt.Identifiers[ComponentDim],
    key: pt.PRNGKey,
) -> None:
    assert_type(masses, jax.Array)
    assert_type(table, npt.NDArray[np.float64])
    assert_type(count, int)
    assert_type(names, tuple[str, ...])
    assert_type(key, jax.Array)


def boundaries() -> None:
    basis = pt.parse("dense", Literal["dense", "sparse"], "basis")
    assert_type(basis, Literal["dense", "sparse"])
    values = pt.as_array([1.0, 2.0], pt.Float64[ComponentDim], "values")
    assert_type(values, jax.Array)
    host = pt.as_host_array(values, pt.HostFloat64[ComponentDim], "host")
    assert_type(host, npt.NDArray[np.float64])
    assert_type(precision_dtype_name("float32"), ScalarPrecisionDType)


def dtype_specifications(indices: jax.Array) -> None:
    from phydrax.graph import degree

    assert_type(degree(indices, dtype=jnp.float32), jax.Array)
    assert_type(degree(indices, dtype=np.dtype("float64")), jax.Array)


def constructors() -> None:
    catalog = ChemicalComponentCatalog(
        ("H2",), np.asarray((2.016,)), ("H",), np.asarray(((2,),))
    )
    assert_type(catalog, ChemicalComponentCatalog)
    assert_type(catalog.molar_masses, jax.Array)
    ChemicalComponentCatalog(("H2",))  # ty: ignore[missing-argument]
    ChemicalComponentCatalog(
        ("H2",),
        jnp.ones((1,)),
        ("H",),
        np.ones((1, 1), dtype=np.int32),
        weights=1,  # ty: ignore[unknown-argument]
    )
    phx.typing.validate(catalog)


@pt.checked
def scaled_masses(catalog: ChemicalComponentCatalog, scale: float, /) -> jax.Array:
    return scale * catalog.molar_masses


def checked_signatures(catalog: ChemicalComponentCatalog) -> None:
    assert_type(scaled_masses(catalog, 2.0), jax.Array)
    scaled_masses(catalog)  # ty: ignore[missing-argument]
    scaled_masses("catalog", 2.0)  # ty: ignore[invalid-argument-type]


def singular_subspaces(problem: SVDProblem, key: pt.PRNGKey, probe: jax.Array) -> None:
    policy = SVDSolvePolicy(RandomizedSVD(), count=2, differentiation="projector")
    prepared = prepare_svd(problem, policy, key=key)
    assert_type(prepared, PreparedSVDSolve)
    result = svd(prepared)
    assert_type(result, SVDSolveResult)
    assert_type(result.rank_evidence, SVDRankEvidence)
    assert_type(result.right_response, SingularSubspaceResponse)
    assert_type(result.derivative_valid, jax.Array)
    assert_type(
        projector_action(result.right_coordinates, result.right_response, probe),
        jax.Array,
    )


def projector_consumer(
    prepared: PreparedProjectorMonteCarlo,
    key: pt.PRNGKey,
    result: ProjectorMonteCarloResult,
) -> None:
    state = initialize_projector_monte_carlo(prepared, key)
    assert_type(state, ProjectorMonteCarloState)
    assert_type(state.support_keys, jax.Array)
    assert_type(state.root_key, jax.Array)
    assert_type(
        step_projector_monte_carlo(prepared, state), ProjectorMonteCarloStepResult
    )
    assert_type(
        solve_projector_monte_carlo(prepared, state, steps=1), ProjectorMonteCarloResult
    )
    analysis = analyze_projector_monte_carlo(
        prepared, result, policy=ProjectorEstimatorPolicy()
    )
    assert_type(analysis, ProjectorMonteCarloAnalysis)
    assert_type(analysis.projected, CorrelatedRatioResult)
    assert_type(analysis.projected.mean_covariance, jax.Array)
    initialize_projector_monte_carlo(prepared, 1)  # ty: ignore[invalid-argument-type]


def atomistic_execution(
    potential: phx.nn.atomistic.MACEPotential,
    batch: phx.atomistic.AtomisticBatch,
    execution: phx.atomistic.AtomisticGraphExecutionPlan,
) -> None:
    assert_type(potential.configuration, phx.nn.atomistic.MACEArchitecture)
    topology = phx.atomistic.prepare_atomistic_graph_topology(
        batch, execution, cutoff=potential.configuration.cutoff
    )
    assert_type(topology, phx.atomistic.AtomisticGraphTopology)
    derivatives = phx.atomistic.atomistic_energy_derivatives(
        potential, batch, execution, batch.positions, topology=topology
    )
    assert_type(derivatives, phx.atomistic.AtomisticEnergyDerivatives)


def ordered_sparse_routes(
    relation: phx.sparse.EdgeRelation | phx.sparse.RowRelation,
    coefficients: jax.Array,
    values: jax.Array,
) -> None:
    phx.sparse.gather_routes(relation, values, accumulation="deterministic")
    phx.sparse.route_reduce(
        relation, values, reduction="mean", accumulation="compensated"
    )
    phx.sparse.linear_apply(relation, coefficients, values, accumulation="deterministic")
    phx.sparse.linear_apply(
        relation,
        coefficients,
        values,
        accumulation="unordered",  # ty: ignore[invalid-argument-type]
    )


def distributed_spectral_consumer(
    topology: SpectralMeshTopology,
    state: jax.Array,
    vector_state: jax.Array,
) -> None:
    precision = phx.discretization.SpectralPrecisionPolicy(
        jnp.float32,
        coefficient_dtype=jnp.complex64,
        transform_dtype=jnp.complex64,
        reduction_dtype=jnp.float32,
    )
    assert_type(precision, SpectralPrecisionPolicy)
    plan = phx.discretization.DistributedSpectralExecutionPlan(
        topology,
        (8, 8, 8),
        owner_id="installed-distributed-spectral",
        precision=precision,
        admitted_payload_shapes=((), (3,)),
        state_shape=(),
        padded_shape=(12, 12, 12),
        maximum_bytes=64 * 1024**2,
    ).prepare()
    assert_type(plan, DistributedSpectralExecutionPlan)
    assert_type(plan.owner_id, str)
    assert_type(plan.numerical_id, str)
    assert_type(plan.execution_id, str)
    assert_type(plan.plan_id, str)
    transpose = phx.discretization.SpectralTranspose(
        plan.physical_layout, plan.modal_layout, topology
    )
    assert_type(transpose, phx.discretization.SpectralTranspose)
    phx.discretization.SpectralTranspose(
        plan.physical_layout,
        plan.modal_layout,
        "mesh",  # ty: ignore[invalid-argument-type]
    )
    assert_type(plan.to_modal(state), jax.Array)
    assert_type(plan.to_physical(plan.to_modal(state)), jax.Array)
    assert_type(plan.to_modal_batched(vector_state), jax.Array)
    result = plan.execute_transform(state, direction="physical_to_modal")
    assert_type(result, SpectralExecutionResult)
    assert_type(result.value, jax.Array)
    diagnostics = plan.diagnostics(result.value)
    assert_type(diagnostics, SpectralGlobalDiagnostics)
    assert_type(diagnostics.total, jax.Array)


def spectral_principal_map(
    eigenvalues: jax.Array, coefficient: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return (coefficient * eigenvalues)[None, :], jnp.sum(eigenvalues)


def geomechanics_spectral(matrix: jax.Array, coefficient: jax.Array) -> None:
    result = linalg.hermitian_spectral_map(
        matrix, spectral_principal_map, coefficient, function_id="installed-principal-map"
    )
    assert_type(result, linalg.HermitianSpectralMapResult[jax.Array])
    assert_type(result.derivative_valid, jax.Array)
    assert_type(linalg.hermitian_log(matrix).derivative_valid, jax.Array)


def geomechanics_materials(
    mc: solid.MohrCoulombMPMConstitutivePlan,
    mc_parameters: solid.MohrCoulombParameters,
    mcc: solid.ModifiedCamClayMPMConstitutivePlan,
    mu_parameters: rheology.MuIParameters,
    stress: jax.Array,
    deformation: jax.Array,
) -> None:
    mc_history = mc.initialize_state((1,), jnp.float64)
    assert_type(mc_history, jax.Array)  # Packed metric plus hardening: width 7.
    assert_type(
        mc.initialize_state_from_stress(stress, mc_parameters),
        solid.PrestressedMaterialState,
    )
    assert_type(
        mc.evaluate_linearized(deformation, mc_history, 2000.0, mc_parameters, 0.0, 0.01),
        equations.MPMLinearizedConstitutiveResponse,
    )
    parameters = solid.ModifiedCamClayParameters(1000.0, 0.0, 1.2, 0.2, 0.05)
    assert_type(parameters, solid.ModifiedCamClayParameters)
    assert_type(
        mcc.initialize_state_from_stress(
            stress, parameters, preconsolidation_pressure=200.0, initial_void_ratio=0.8
        ),
        solid.PrestressedMaterialState,
    )
    mu = solid.MuIMPMConstitutivePlan(1000.0, 2000.0)
    history = mu.initialize_state((1,), jnp.float64)
    assert_type(history, jax.Array)
    assert_type(
        mu.evaluate(deformation, history, 2000.0, mu_parameters, 0.0, 0.01),
        equations.MPMConstitutiveResponse,
    )
    assert_type(
        mu.evaluate_linearized(deformation, history, 2000.0, mu_parameters, 0.0, 0.01),
        equations.MPMLinearizedConstitutiveResponse,
    )
    assert_type(
        mu.initialize_state_from_stress(stress, mu_parameters),
        solid.PrestressedMaterialState,
    )


def granular_rheology(
    law: rheology.MuILaw,
    parameters: rheology.MuIParameters,
    generalized: rheology.GeneralizedNewtonianLaw,
    packing: rheology.MuIPackingFractionLaw,
) -> None:
    assert_type(law.evaluate(1.0, 100.0, parameters), rheology.MuILawEvaluation)
    assert_type(generalized.viscosity(1.0), rheology.GeneralizedNewtonianEvaluation)
    assert_type(packing.evaluate(0.1), jax.Array)


def geomechanics_public_surfaces() -> None:
    # Canonical facade resolution also runs against the installed wheel.
    surfaces = (
        mpm.MPMVolumetricStabilizationKind,
        mpm.MPMVolumetricStabilizationPlan,
        mpm.MPMGeometryKind,
        mpm.MPMGeometryPlan,
        mpm.gather_apic,
        mpm.build_apic_route_payload,
        mpm.PreparedMPMBoundaryLoad,
        mpm.MPMBoundaryState,
        mpm.MPMBoundaryLoadResult,
        mpm.LysmerKuhlemeyerDashpot,
        mpm.MPMDashpotResult,
        mpm.MPMPredictorResult,
        solver.PreparedImplicitMPMResidual,
        solver.ImplicitMPMResidualEvaluation,
        solver.DoublePointMPMPlan,
        solver.advance_double_point_mpm_step,
        solver.project_double_point_velocity,
        solid.FiniteStrainConstitutiveModel,
        solid.NorSandParameters,
        solid.NorSandMPMConstitutivePlan,
        solid.MixedMPMPressureOperator,
        solid.effective_pore_pressure_first_piola,
        particle.CompliantDEMMethodPlan,
        particle.PreparedCompliantDEMDynamics,
        particle.CompliantDEMBodyProperties,
        particle.CompliantContactShapeKind,
        particle.AbstractRigidContactShapePlan,
        particle.AbstractPreparedRigidContactShape,
        particle.RigidContactGeometryBatch,
        particle.ConvexShapeSetPlan,
        particle.ContactMeasureKind,
        particle.LevelSetShapeTemplatePlan,
        particle.LevelSetRigidShapeSetPlan,
        particle.PreparedLevelSetRigidShapeSet,
        particle.LevelSetMassPropertyEvidence,
        particle.LevelSetPointQueryResult,
        particle.evaluate_level_set_point_query,
        particle.LevelSetPenaltyNormalPlan,
        particle.LevelSetTangentialContactPlan,
        particle.PosedSignedFieldQueryResult,
        particle.RandomSequentialPackingPlan,
        particle.RigidDEMPackingResult,
        spatial.SparseVoxelDifferentialQueryResult,
        geometry.level_set_interpolation_certificate,
        contact.MPMLevelSetGrainContactParticipant,
        contact.MPMLevelSetGrainContactEvaluation,
        contact.MonolithicContactPlan,
        contact.MonolithicContactState,
        contact.MonolithicContactResult,
        contact.PreparedMonolithicContactStep,
        contact.PreparedMonolithicContactEquilibrium,
        contact.initialize_monolithic_contact_state,
        contact.prepare_monolithic_contact_step,
        contact.prepare_monolithic_contact_equilibrium,
        contact.solve_monolithic_contact_step,
        contact.solve_monolithic_contact_equilibrium,
        contact.monolithic_contact_solution_jvp,
        contact.monolithic_contact_solution_vjp,
        contact.AbstractImplicitContactParticipant,
        contact.FunctionalImplicitContactParticipant,
        contact.make_implicit_contact_participant,
        contact.AffineBodyPlan,
        contact.prepare_affine_body_participant,
        contact.prepare_finite_element_contact_participant,
        contact.prepare_iga_contact_participant,
        contact.prepare_implicit_mpm_contact_participant,
        contact.prepare_reduced_rod_contact_participant,
        contact.AffineBodyHistory,
        contact.AffineBodyParticipant,
        equations.AbstractInterphaseDragLaw,
        equations.InterphaseDragLawPlan,
        equations.InterphaseDragCorrelation,
        equations.InterphaseDragEvaluation,
        equations.FLIPViscousLaw,
        FLIPViscousEvidence,
        rheology.MuIRegularizationKind,
        equations.MACSDFCutFractionPlan,
        equations.MACVolumeFractionIBCFDEMCouplingPlan,
        solver.advance_mac_volume_fraction_ib_cfd_dem,
        VOLUME_FRACTION_CFD_DEM_SUPPORT_TUPLE,
    )
    del surfaces


def implicit_mpm_residual(
    dynamics: solver.PreparedImplicitMPMDynamics,
    state: mpm.MPMRuntimeState,
    velocity: jax.Array,
    arguments: equations.MaterialPointArguments,
) -> None:
    prepared = dynamics.prepare_residual(state, 0.01, arguments)
    assert_type(prepared, solver.PreparedImplicitMPMResidual)
    assert_type(
        prepared.evaluate_detailed(velocity), solver.ImplicitMPMResidualEvaluation
    )


def incremental_flip_projection(
    plan: solver.MACFreeSurfaceProjectionPlan,
    velocity: phx.discretization.finite_volume.FaceVelocity,
    liquid: jax.Array,
    pressure: jax.Array,
) -> None:
    assert_type(
        plan.project(velocity, liquid, 0.01, pressure=pressure, incremental=True),
        solver.MACFreeSurfaceProjectionResult,
    )
