"""Public bounded cut-fraction, coupling, and transactional solver annotations."""

from typing import assert_type, Literal

from jax import Array

from phydrax.applications.incompressible_flow import (
    ImmersedBodyRegimePlan,
    ImmersedRegime,
    VOLUME_FRACTION_CFD_DEM_SUPPORT_TUPLE,
)
from phydrax.discretization import (
    AbstractPreparedRigidContactShape,
    DEMRuntimeState,
    PreparedCompliantDEMDynamics,
    PreparedMACOperators,
    RigidBodyKinematics,
)
from phydrax.discretization.finite_volume._incompressible import FaceVelocity
from phydrax.equations import (
    CompiledMACVariableDensityDynamics,
    evaluate_mac_volume_fraction_ib_cfd_dem,
    MACSDFCutFractionEvaluation,
    MACSDFCutFractionPlan,
    MACVolumeFractionIBCFDEMCouplingPlan,
    MACVolumeFractionIBEvaluation,
    MACVolumeFractionIBStatus,
)
from phydrax.qualification import SupportTuple
from phydrax.solver import (
    advance_mac_volume_fraction_ib_cfd_dem,
    MACVariableDensityProjectionPlan,
    MACVariableDensityProjectionResult,
    MACVolumeFractionIBState,
    MACVolumeFractionIBStepResult,
)


def volume_fraction_contract(
    operators: PreparedMACOperators,
    shape: AbstractPreparedRigidContactShape,
    kinematics: RigidBodyKinematics,
    scale: Array,
    active: Array,
    fluid: CompiledMACVariableDensityDynamics,
    dynamics: PreparedCompliantDEMDynamics,
    projection: MACVariableDensityProjectionPlan,
    dem_state: DEMRuntimeState,
    fluid_state: Array,
    pressure: Array,
    predictor: FaceVelocity,
    stress: FaceVelocity,
    time: Array,
    step: Array,
) -> None:
    cut = MACSDFCutFractionPlan(
        operators,
        shape,
        subdivisions=4,
        maximum_unresolved_fraction=0.02,
        maximum_queries=4096,
    )
    assert_type(cut, MACSDFCutFractionPlan)
    geometry = cut.evaluate(kinematics, time, shape_scale=scale, active_mask=active)
    assert_type(geometry, MACSDFCutFractionEvaluation)
    assert_type(geometry.solid_fraction_by_body, Array)
    assert_type(geometry.solid_fraction, Array)
    assert_type(geometry.fluid_fraction, Array)
    assert_type(geometry.fraction_lower, Array)
    assert_type(geometry.fraction_upper, Array)
    assert_type(geometry.unresolved_fraction, Array)
    assert_type(geometry.geometry_margin, Array)
    assert_type(geometry.query_overflow, Array)
    assert_type(geometry.query_reliable, Array)
    assert_type(geometry.domain_supported, Array)
    assert_type(geometry.query_count, Array)
    assert_type(geometry.query_capacity, int)
    assert_type(geometry.successful, Array)
    assert_type(geometry.plan_id, str)
    coupling = MACVolumeFractionIBCFDEMCouplingPlan(
        fluid,
        dynamics,
        cut,
        projection,
        fluid_density=1.0,
        solid_density=2.5,
    )
    assert_type(coupling, MACVolumeFractionIBCFDEMCouplingPlan)
    evaluation = evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling,
        kinematics,
        geometry,
        predictor,
        stress,
        step,
    )
    assert_type(evaluation, MACVolumeFractionIBEvaluation)
    assert_type(evaluation.mixture_density, Array)
    assert_type(evaluation.solid_mass_fraction, Array)
    assert_type(evaluation.solid_velocity, Array)
    assert_type(evaluation.fluid_ib_force_density, FaceVelocity)
    assert_type(evaluation.face_mixture_density, FaceVelocity)
    assert_type(evaluation.body_force, Array)
    assert_type(evaluation.body_torque, Array)
    assert_type(evaluation.pressure_viscous_load, Array)
    assert_type(evaluation.pressure_viscous_torque, Array)
    assert_type(evaluation.fluid_external_impulse, Array)
    assert_type(evaluation.body_direct_impulse, Array)
    assert_type(evaluation.impulse_residual, Array)
    assert_type(evaluation.torque_residual, Array)
    assert_type(evaluation.work_adjoint_residual, Array)
    assert_type(evaluation.status, Array)
    assert_type(evaluation.successful, Array)
    assert_type(evaluation.plan_id, str)
    assert_type(
        MACVolumeFractionIBStatus.SUCCESS, Literal[MACVolumeFractionIBStatus.SUCCESS]
    )
    state = MACVolumeFractionIBState.initialize(
        coupling,
        dem_state,
        fluid_state,
        pressure=pressure,
    )
    assert_type(state, MACVolumeFractionIBState)
    assert_type(state.cumulative_body_impulse, Array)
    assert_type(state.cumulative_body_angular_impulse, Array)
    assert_type(state.pressure, Array)
    assert_type(state.accepted_steps, Array)
    result = advance_mac_volume_fraction_ib_cfd_dem(coupling, state, time, step)
    assert_type(result, MACVolumeFractionIBStepResult)
    assert_type(result.candidate_state, MACVolumeFractionIBState)
    assert_type(result.accepted_state, MACVolumeFractionIBState)
    assert_type(result.evaluation, MACVolumeFractionIBEvaluation)
    assert_type(result.pressure_projection, MACVariableDensityProjectionResult)
    assert_type(result.candidate_cut_fraction, MACSDFCutFractionEvaluation)
    assert_type(result.transport_successful, Array)
    assert_type(result.status, Array)
    assert_type(result.successful, Array)
    assert_type(result.plan_id, str)
    regime = ImmersedBodyRegimePlan(
        coupling,
        marker_set_id="no-markers-volume-fraction",
        geometry_id=shape.shape_id,
        route_id=cut.plan_id,
        topology_epoch_id="fixed-capacity",
        geometry_epoch=0,
        moving=True,
        motion_epoch_id="accepted-pose",
    )
    assert_type(regime.regime, ImmersedRegime)
    assert_type(regime.support_tuple, SupportTuple)
    assert_type(VOLUME_FRACTION_CFD_DEM_SUPPORT_TUPLE, SupportTuple)
