# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native robotics contracts: models, plant/control state, contact and rank evidence."""

from typing import assert_type

from jax import Array

from phydrax.applications.contact import (
    ArticulatedActuationMode,
    ArticulatedContactAdmissibility,
    ArticulatedContactEnergyLedger,
    ArticulatedContactSource,
    ArticulatedContactStepBinding,
    ArticulatedContactStepJVP,
    ArticulatedContactStepPolicy,
    ArticulatedContactStepResult,
    ArticulatedContactStepVJP,
    certify_contact_cone_impulse,
    classify_contact_cone_branches,
    contact_cone_fixed_branch_admitted,
    contact_cone_fixed_branch_residual,
    contact_cone_solution_jvp,
    contact_cone_solution_vjp,
    ContactBlockLayout,
    ContactConeBranch,
    ContactConeCertificate,
    ContactConeFixedBranch,
    ContactConeJVP,
    ContactConeProgram,
    ContactConeResult,
    ContactConeTangent,
    ContactConeVJP,
    ContactManifoldObservation,
    ContactManifoldPolicy,
    ContactManifoldState,
    ContactManifoldTransition,
    fix_contact_cone_branch,
    prepare_contact_cone_sensitivity,
    PreparedArticulatedContactStep,
    PreparedContactConeSensitivity,
    update_contact_manifold,
)
from phydrax.applications.robotics import (
    adapt_mujoco_model,
    ArticulatedPlantCommand,
    ArticulatedPlantControlView,
    ArticulatedPlantObservation,
    ArticulatedPlantRefreshResult,
    MujocoJointStopTransformation,
    MujocoModelEvidence,
    MujocoNativeContactTransformation,
    NativeArticulatedPlantPlan,
    NativeRobotModel,
    parse_urdf_text,
    prepare_articulated_plant,
    PreparedArticulatedPlant,
    PreparedNativeRobotModel,
    RobotAdaptation,
    RobotContactMaterialTable,
    URDFFormatEvidence,
)
from phydrax.control import (
    AbstractControlParameterization,
    PlantControlDynamics,
    PlantControlHistory,
    SamplingMPCRealizations,
)
from phydrax.discretization.contact import (
    AdmissiblePathEvidence,
    AdmissiblePathPlan,
    ArticulatedCollisionGeometry,
    ArticulatedContactTrajectory,
    ArticulatedLiftedPath,
    certify_admissible_path,
    ContactCandidateEpoch,
    ContactFeatureCatalogue,
    ContactParticipantScene,
    ContactPathEnclosure,
    ContactPathMidpointBound,
    ContactRouteKeySchema,
    ContactStencilKind,
    ContactStencilLayout,
)
from phydrax.discretization.particle import (
    ArticulatedInverseMassOperator,
    ArticulatedMassActionResult,
    ArticulatedMassAdjointEvidence,
    ArticulatedMassPolicy,
    ArticulatedMassProductResult,
    LiftedJointCoordinateState,
    LiftedJointCoordinateUpdate,
    prepare_articulated_mass,
    PreparedArticulatedMass,
    PreparedReducedArticulation,
    RigidConstraintDiagnostics,
    RigidConstraintRankAudit,
    RigidConstraintRankEvidence,
)
from phydrax.dynamics import (
    ControlVectorCodec,
    PlantParameters,
    PlantRuntimeState,
    PlantStateVectorCodec,
    PlantStepContext,
    PlantStepResult,
    TimeGrid,
)
from phydrax.linalg import (
    BlockForestRankCertificate,
    BlockForestTopology,
    certify_block_forest_rank,
    FunctionLinearOperator,
    prepare_block_forest,
    PreparedBlockForest,
    RectangularRankCertificate,
    refresh_block_forest,
)
from phydrax.linalg.svd import SVDSolvePolicy


def articulated_mass(
    articulation: PreparedReducedArticulation,
    configuration: Array,
    effort: Array,
    efforts: Array,
) -> None:
    mass = prepare_articulated_mass(
        articulation, configuration, policy=ArticulatedMassPolicy(maximum_rhs=8)
    )
    assert_type(mass, PreparedArticulatedMass)
    assert_type(mass.inverse_mass_operator, ArticulatedInverseMassOperator)
    assert_type(mass.is_current(articulation, configuration), Array)
    assert_type(
        mass.apply_inverse(articulation, configuration, efforts, rhs_axis=1),
        ArticulatedMassActionResult,
    )
    assert_type(
        mass.apply_mass(articulation, configuration, effort),
        ArticulatedMassProductResult,
    )
    assert_type(mass.adjoint_evidence(effort, effort), ArticulatedMassAdjointEvidence)
    mass.apply_inverse(articulation, configuration, efforts, rhs_axis="columns")  # ty: ignore[invalid-argument-type]
    ArticulatedMassPolicy(maximum_rhs=8.0)  # ty: ignore[invalid-argument-type]


def native_model(
    model: NativeRobotModel,
    configuration: Array,
    velocity: Array,
    step: Array,
    winding: LiftedJointCoordinateState,
) -> None:
    prepared = model.prepare(maximum_turns=64)
    assert_type(prepared, PreparedNativeRobotModel)
    assert_type(prepared.collision, ArticulatedCollisionGeometry | None)
    assert_type(
        prepared.coordinates.advance(configuration, winding, velocity, step),
        LiftedJointCoordinateUpdate,
    )
    stops = prepared.stops
    if stops is not None:
        assert_type(stops.jacobian_operator(), FunctionLinearOperator)


def source_adaptation(
    urdf: str, compiled: object, materials: RobotContactMaterialTable
) -> NativeRobotModel:
    urdf_adaptation = parse_urdf_text(
        urdf, root_policy="fixed_world", joint_stops="native-unilateral-stops"
    )
    assert_type(urdf_adaptation, RobotAdaptation[URDFFormatEvidence])
    assert_type(urdf_adaptation.evidence, URDFFormatEvidence)
    parse_urdf_text(urdf, root_policy="fixed_world", joint_stops="hard-stops")  # ty: ignore[invalid-argument-type]
    urdf_adaptation.articulation  # ty: ignore[unresolved-attribute]
    mujoco_adaptation = adapt_mujoco_model(
        compiled,
        contact=MujocoNativeContactTransformation(
            materials, "declared native hard contact", default_material_id=0
        ),
        joint_stops=MujocoJointStopTransformation("declared native hard stops"),
    )
    assert_type(mujoco_adaptation, RobotAdaptation[MujocoModelEvidence])
    assert_type(mujoco_adaptation.evidence, MujocoModelEvidence)
    MujocoJointStopTransformation()  # ty: ignore[missing-argument]
    return mujoco_adaptation.model


def native_plant(
    model: PreparedNativeRobotModel,
    policy: ArticulatedContactStepPolicy,
    gravity: Array,
    command: Array,
    context: PlantStepContext,
    state: PlantRuntimeState,
) -> PreparedArticulatedPlant:
    plan = NativeArticulatedPlantPlan(
        policy,
        gravity=gravity,
        actuation_mode=ArticulatedActuationMode.MAPPED_MOTORS,
        case_ndim=1,
    )
    NativeArticulatedPlantPlan(policy, gravity=gravity, actuation_mode="mapped-motors")  # ty: ignore[invalid-argument-type]
    NativeArticulatedPlantPlan(
        policy,
        gravity=gravity,
        actuation_mode=ArticulatedActuationMode.AUTONOMOUS,
        search_backend="bvh",  # ty: ignore[invalid-argument-type]
    )
    plant = prepare_articulated_plant(model, plan)
    assert_type(plant, PreparedArticulatedPlant)
    assert_type(plant.actuation_mode, ArticulatedActuationMode)
    parameters = plant.bind_parameters()
    assert_type(parameters, PlantParameters)
    assert_type(
        plant.step(context, state, ArticulatedPlantCommand(command), parameters),
        PlantStepResult,
    )
    assert_type(plant.observe(state), ArticulatedPlantObservation)
    assert_type(plant.refresh(state), ArticulatedPlantRefreshResult)
    assert_type(plant.control_view(state), ArticulatedPlantControlView)
    return plant


def complete_state_control(
    plant: PreparedArticulatedPlant,
    states: PlantStateVectorCodec,
    commands: ControlVectorCodec,
    grid: TimeGrid,
    initial: PlantRuntimeState,
    parameterization: AbstractControlParameterization,
    coefficients: Array,
) -> PlantRuntimeState:
    bound = plant.bind_parameters()
    dynamics = PlantControlDynamics(
        plant, states, commands, parameters=bound, retention="full"
    )
    PlantControlDynamics(plant, states, commands)  # ty: ignore[missing-argument]
    PlantControlDynamics(plant, states, commands, parameters=bound, retention="all")  # ty: ignore[invalid-argument-type]
    trajectory = dynamics.rollout(
        grid, initial, parameterization, coefficients, problem_id="native-rollout"
    )
    history = trajectory.plant_history
    assert_type(history, PlantControlHistory | None)
    assert history is not None
    assert_type(history.final_state, PlantRuntimeState)
    assert_type(history.realization_count, int | None)
    if history.realization_count is not None:
        selected = history.select_realization(0)
        assert_type(selected, PlantControlHistory)
        return selected.state_at(1)
    history.select_realization("heavy")  # ty: ignore[invalid-argument-type]
    return history.state_at(1)


def articulated_step(
    step: PreparedArticulatedContactStep,
    configuration: Array,
    winding: Array,
    velocity: Array,
    actuation: Array,
    gravity: Array,
    duration: Array,
) -> ArticulatedContactSource:
    source = step.initial_source(configuration, winding, velocity)
    assert_type(source, ArticulatedContactSource)
    assert_type(step.admissibility(source), ArticulatedContactAdmissibility)
    result = step.step(source, duration, actuation, gravity)
    assert_type(result, ArticulatedContactStepResult)
    assert_type(result.ledger, ArticulatedContactEnergyLedger)
    assert_type(result.fixed_branch_candidate, Array)
    assert_type(result.binding, ArticulatedContactStepBinding)
    result.derivative_admitted  # ty: ignore[unresolved-attribute]
    tangent = step.step_jvp(
        result, actuation, gravity, (None, configuration, velocity, actuation, gravity)
    )
    assert_type(tangent, ArticulatedContactStepJVP)
    assert_type(tangent.coupled_regular, Array)
    assert_type(tangent.coupled_regularity, RectangularRankCertificate | None)
    tangent.velocity_regular  # ty: ignore[unresolved-attribute]
    cotangent = step.step_vjp(
        result,
        actuation,
        gravity,
        (configuration, velocity, winding),
        friction_cotangent=False,
    )
    assert_type(cotangent, ArticulatedContactStepVJP)
    assert_type(cotangent.step_cotangent, PreparedArticulatedContactStep)
    assert_type(cotangent.friction_cotangent, bool)
    return result.accepted


def operator_cone(
    program: ContactConeProgram,
    result: ContactConeResult,
    impulse: Array,
    direction: Array,
    admissible: Array,
    regularity: SVDSolvePolicy,
) -> None:
    assert_type(ContactBlockLayout.mixed(2, (1, 2), family_count=3), ContactBlockLayout)
    ContactBlockLayout.mixed(2, (1, 2))  # ty: ignore[missing-argument]
    certificate = certify_contact_cone_impulse(program, impulse)
    assert_type(certificate, ContactConeCertificate)
    assert_type(
        classify_contact_cone_branches(program, impulse, certificate),
        tuple[Array, Array],
    )
    sensitivity = prepare_contact_cone_sensitivity(program, result, regularity=regularity)
    assert_type(sensitivity, PreparedContactConeSensitivity)
    assert_type(sensitivity.regularity, RectangularRankCertificate | None)
    assert_type(sensitivity.jacobian_regular, Array)
    assert_type(sensitivity.fixed_branch, ContactConeFixedBranch)
    prepare_contact_cone_sensitivity(program, result, margin_factor=2.0)  # ty: ignore[unknown-argument]
    sensitivity.qualified  # ty: ignore[unresolved-attribute]
    sensitivity.branch  # ty: ignore[unresolved-attribute]
    fixed = fix_contact_cone_branch(program, result)
    assert_type(fixed, ContactConeFixedBranch)
    assert_type(
        contact_cone_fixed_branch_residual(fixed, impulse, direction, admissible), Array
    )
    assert_type(
        contact_cone_fixed_branch_admitted(
            program, fixed, margin_factor=1.0, friction_derivative=False
        ),
        Array,
    )
    contact_cone_fixed_branch_admitted(program, fixed, margin_factor=1.0)  # ty: ignore[missing-argument]
    tangent = ContactConeTangent(program, free_velocity=direction)
    assert_type(
        contact_cone_solution_jvp(
            program,
            result,
            tangent,
            sensitivity=sensitivity,
            geometry_admissible=admissible,
            margin_factor=2.0,
        ),
        ContactConeJVP,
    )
    vjp = contact_cone_solution_vjp(
        program, result, impulse, sensitivity=sensitivity, friction_cotangent=False
    )
    assert_type(vjp, ContactConeVJP)
    assert_type(vjp.cotangent.static_friction, Array | None)
    frictionless: ContactConeBranch = ContactConeBranch.FRICTIONLESS_BOUNDARY
    assert_type(int(frictionless), int)
    contact_cone_solution_jvp(program, result, direction)  # ty: ignore[invalid-argument-type]
    program.effective_mass  # ty: ignore[unresolved-attribute]


def model_realizations(parameters: Array) -> SamplingMPCRealizations:
    realizations = SamplingMPCRealizations(
        parameters,
        ("nominal", "heavy"),
        posterior_id="payload-posterior",
        campaign_id="native-arm",
        policy="resample",
    )
    assert_type(realizations.realization_ids, tuple[str, ...])
    SamplingMPCRealizations(
        parameters,
        ("nominal",),
        posterior_id="p",
        campaign_id="c",
        policy="random",  # ty: ignore[invalid-argument-type]
    )
    return realizations


def mechanical_history(observation: ContactManifoldObservation) -> ContactManifoldState:
    state = ContactManifoldState.empty(8, catalogue_id=observation.catalogue_id)
    transition = update_contact_manifold(
        state, observation, policy=ContactManifoldPolicy(retention_steps=3)
    )
    assert_type(transition, ContactManifoldTransition)
    update_contact_manifold(state, observation)  # ty: ignore[missing-argument]
    assert_type(transition.rollback(), ContactManifoldState)
    return transition.commit()


def routes_and_paths(
    participants: Array,
    kinds: Array,
    features: Array,
    scene: ContactParticipantScene,
    epoch: ContactCandidateEpoch,
    geometry: ArticulatedCollisionGeometry,
    articulation: PreparedReducedArticulation,
    configuration: Array,
    velocity: Array,
    step: Array,
) -> AdmissiblePathEvidence:
    schema = ContactRouteKeySchema(ContactFeatureCatalogue(participants, kinds, features))
    keys = schema.encode(kinds, features, features, kinds)
    assert_type(keys, Array)
    assert_type(schema.decode(keys), tuple[Array, Array, Array, Array])
    ContactStencilLayout(schema, ContactStencilKind.EDGE_PLANE, 4, witness_slot_count=2)
    trajectory = ArticulatedContactTrajectory(
        geometry, ArticulatedLiftedPath(articulation, configuration, velocity, step)
    )
    assert_type(
        trajectory.interval_bounds(0.0, 0.5),
        tuple[ContactPathEnclosure, ContactPathMidpointBound],
    )
    plan = AdmissiblePathPlan(1.0e-9, maximum_intervals=1024)
    AdmissiblePathPlan(gap_floor=1.0e-9)  # ty: ignore[positional-only-parameter-as-kwarg]
    evidence = certify_admissible_path(plan, scene, epoch, trajectory)
    assert_type(evidence.successful, Array)
    return evidence


def rank_evidence(
    diagnostics: RigidConstraintDiagnostics,
    parents: Array,
    rows: Array,
    child_blocks: Array,
    parent_blocks: Array,
) -> None:
    RigidConstraintRankAudit("certified-forest", workspace_bytes=1 << 20)
    RigidConstraintRankAudit("forest")  # ty: ignore[invalid-argument-type]
    assert_type(diagnostics.rank, RigidConstraintRankEvidence)
    assert_type(diagnostics.rank.certificate, BlockForestRankCertificate | None)
    diagnostics.constraint_rank  # ty: ignore[unresolved-attribute]
    topology = BlockForestTopology(parents, rows, column_width=6)
    prepared = prepare_block_forest(topology, child_blocks, parent_blocks)
    assert_type(prepared, PreparedBlockForest)
    assert_type(
        refresh_block_forest(prepared, child_blocks, parent_blocks), PreparedBlockForest
    )
    assert_type(
        certify_block_forest_rank(prepared, relative_cutoff=1.0e-10),
        BlockForestRankCertificate,
    )
    certify_block_forest_rank(prepared)  # ty: ignore[missing-argument]
