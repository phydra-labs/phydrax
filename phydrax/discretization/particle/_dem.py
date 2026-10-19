#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntFlag
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._differentiation import BranchDifferentiationPolicy
from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...equations._dem_material import DEMMaterialTable
from ...linalg import HermitianSpectrum
from ...metrix._state_geometry import AbstractStateGeometry
from ...typing import checked
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._periodic_cell import PeriodicCell
from ._dem_boundary import (
    DEMBoundaryResponse,
    evaluate_dem_barrier,
    evaluate_dem_triangle_wall,
    ImplicitDEMBarrier,
)
from ._dem_cohesion import BagheriCapillaryBridgePlan, CompositeDEMCohesionPlan
from ._dem_contact import (
    DEMContactBatch,
    DEMContactHistory,
    DEMContactModelPlan,
    DEMContactResponse,
    HertzNormalContactPlan,
    LevelSetPenaltyNormalPlan,
    LinearSpringDashpotNormalPlan,
    PreparedDEMContactModel,
)
from ._dem_contact_state import (
    DEMCohesionHistory,
    DEMContactEvaluationContext,
    remap_dem_contact_history,
)
from ._dem_kernels import reduce_rigid_contact
from ._dem_liquid import (
    conserved_bagheri_component,
    ConservedLiquidBridgeProcessPlan,
    DEMBarrierLiquidAllocation,
    DEMLiquidAllocation,
    DEMLiquidEvaluation,
    DEMLiquidState,
)
from ._dem_multicontact import (
    AbstractDEMContactGraphCorrectionPlan,
    DEMMulticontactCorrection,
)
from ._dem_periodic import (
    dem_bulk_stress,
    DEMBulkStress,
    DEMPeriodicCellControlPlan,
    DEMPeriodicCellState,
)
from ._neighborhood import (
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pair_state import match_particle_pair_keys, ParticlePairKeys, ParticlePairKeySpace
from ._pairwise import particle_pair_geometry
from ._population import ParticlePopulationPlan, ParticlePopulationState
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._rigid_body import (
    PreparedRigidBodySet,
    rigid_body_angular_acceleration,
    rigid_body_drift,
    rigid_body_world_inertia,
    RigidBodyKinematics,
    RigidBodyLoad,
    RigidBodyStateGeometry,
)
from ._rigid_contact import RigidContactGeometry
from ._rigid_shape import AbstractPreparedRigidContactShape, RigidContactGeometryBatch
from ._rigid_sphere import PreparedRigidSphereSet, sphere_lever_torque
from ._superquadric_wall import SuperquadricTriangleContactPlan
from ._triangle_wall import PreparedTriangleWall
from ._verlet import ParticleVerletState, PreparedVerletParticleNeighborhood


ExternalDEMLoad = Callable[[Array, Array, Array, Array, Any], "DEMExternalLoad"]


class DEMRejectionReason(IntFlag):
    NONE = 0
    CELL_CAPACITY = 1 << 0
    PAIR_CAPACITY = 1 << 1
    DOMAIN = 1 << 2
    PAIR_KEY = 1 << 3
    GEOMETRY = 1 << 4
    BODY = 1 << 10
    CONTACT = 1 << 5
    NONFINITE = 1 << 6
    OVERLAP = 1 << 7
    ENERGY = 1 << 8
    FRAME = 1 << 9
    CELL_CONTROL = 1 << 11
    LIQUID = 1 << 12
    SHAPE_QUERY = 1 << 13
    CONTACT_CAPACITY = 1 << 14
    CURVATURE = 1 << 15


@final
class CompliantDEMMethodPlan(StrictModule, NonTrainableState):
    """Fixed-capacity penalty DEM with branchwise contact topology."""

    contact: DEMContactModelPlan
    multicontact: AbstractDEMContactGraphCorrectionPlan | None
    periodic_cell_control: DEMPeriodicCellControlPlan | None
    liquid_process: ConservedLiquidBridgeProcessPlan | None
    maximum_contact_routes: int | None = eqx.field(static=True)
    wall_geometry: SuperquadricTriangleContactPlan
    maximum_overlap_fraction: float = eqx.field(static=True)
    distance_tolerance: float = eqx.field(static=True)
    frame_tolerance: float = eqx.field(static=True)
    differentiability: BranchDifferentiationPolicy = eqx.field(static=True)
    key: DiscretizationKey
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        contact: DEMContactModelPlan,
        /,
        *,
        multicontact: AbstractDEMContactGraphCorrectionPlan | None = None,
        periodic_cell_control: DEMPeriodicCellControlPlan | None = None,
        liquid_process: ConservedLiquidBridgeProcessPlan | None = None,
        maximum_contact_routes: int | None = None,
        wall_geometry: SuperquadricTriangleContactPlan | None = None,
        maximum_overlap_fraction: float = 0.1,
        distance_tolerance: float = 1.0e-12,
        frame_tolerance: float = 1.0e-10,
        name: str = "compliant-dem",
        method_id: str | None = None,
    ) -> None:
        if maximum_contact_routes is not None and maximum_contact_routes <= 0:
            raise ValueError(
                f"CompliantDEMMethodPlan.maximum_contact_routes must be positive; got {maximum_contact_routes!r}"
            )
        wall_geometry_ = (
            SuperquadricTriangleContactPlan() if wall_geometry is None else wall_geometry
        )
        if liquid_process is not None:
            conserved_bagheri_component(contact.cohesion)
        overlap = float(maximum_overlap_fraction)
        distance = float(distance_tolerance)
        frame = float(frame_tolerance)
        if (
            not np.isfinite(overlap)
            or overlap <= 0.0
            or not np.isfinite(distance)
            or distance <= 0.0
            or not np.isfinite(frame)
            or frame <= 0.0
        ):
            raise ValueError(
                f"CompliantDEMMethodPlan.tolerances must be finite and positive; got {(overlap, distance, frame)!r}"
            )
        key = DiscretizationKey(
            name,
            DiscretizationRole.RESIDUAL,
            domain_labels=("material_point", "discrete_element"),
        )
        generated = canonical_fingerprint(
            {
                "kind": "compliant-dem-method-plan",
                "contact": contact.contact_model_id,
                "multicontact": (None if multicontact is None else multicontact.plan_id),
                "periodic_cell_control": (
                    None
                    if periodic_cell_control is None
                    else periodic_cell_control.plan_id
                ),
                "liquid_process": (
                    None if liquid_process is None else liquid_process.plan_id
                ),
                "maximum_contact_routes": maximum_contact_routes,
                "wall_geometry": wall_geometry_.plan_id,
                "maximum_overlap_fraction": overlap,
                "distance_tolerance": distance,
                "frame_tolerance": frame,
                "differentiability": BranchDifferentiationPolicy.BRANCHWISE.value,
                "key": key.key_id,
            }
        )
        identifier = generated if method_id is None else str(method_id)
        if not identifier:
            raise ValueError(
                f"CompliantDEMMethodPlan.contact must be nonempty; got {type(contact).__name__}"
            )
        self.contact = contact
        self.multicontact = multicontact
        self.periodic_cell_control = periodic_cell_control
        self.liquid_process = liquid_process
        self.maximum_contact_routes = maximum_contact_routes
        self.wall_geometry = wall_geometry_
        self.maximum_overlap_fraction = overlap
        self.distance_tolerance = distance
        self.frame_tolerance = frame
        self.differentiability = BranchDifferentiationPolicy.BRANCHWISE
        self.key = key
        self.method_id = identifier


class DEMExternalLoad(StrictModule):
    force: Array
    torque: Array


class DEMResolvedLoad(StrictModule):
    """Source-resolved endpoint loads used by the discrete work ledger."""

    particle_contact: RigidBodyLoad
    boundaries: tuple[RigidBodyLoad, ...]
    gravity: RigidBodyLoad
    external: RigidBodyLoad
    total: RigidBodyLoad


class DEMEnergyLedgerState(StrictModule):
    """Accepted cumulative mechanical-energy and source-work ledger."""

    initial_kinetic_energy: Array
    initial_contact_energy: Array
    initial_gravity_potential: Array
    kinetic_energy: Array
    contact_energy: Array
    gravity_potential: Array
    current_boundary_wall_power: Array
    cumulative_particle_contact_work: Array
    cumulative_boundary_contact_work: Array
    cumulative_prescribed_wall_work: Array
    cumulative_gravity_work: Array
    cumulative_external_work: Array
    cumulative_cell_work: Array
    cumulative_contact_balance_loss: Array
    cumulative_energy_residual: Array
    last_relative_energy_residual: Array
    contact_births: Array
    contact_deaths: Array
    stick_to_slip_events: Array
    slip_to_stick_events: Array
    accepted_steps: Array


class DEMStepEnergyLedger(StrictModule):
    """Candidate energy identity for one fixed step."""

    kinetic_before: Array
    kinetic_after: Array
    contact_energy_before: Array
    contact_energy_after: Array
    gravity_potential_before: Array
    gravity_potential_after: Array
    particle_contact_work: Array
    boundary_contact_work: Array
    prescribed_wall_work: Array
    boundary_wall_power_after: Array
    gravity_work: Array
    external_work: Array
    cell_work: Array
    contact_balance_loss: Array
    energy_residual: Array
    relative_energy_residual: Array
    contact_births: Array
    contact_deaths: Array
    stick_to_slip_events: Array
    slip_to_stick_events: Array
    accepted: Array


@final
class CompliantDEMBodyProperties(StrictModule):
    """Dynamic COM mass, uniform geometric scale, and body-frame inertia."""

    population: ParticlePopulationState
    inverse_masses: Array
    shape_scale: Array
    inertia_com: Array
    inverse_inertia_com: Array
    shape_id: str = eqx.field(static=True)

    @property
    def masses(self) -> Array:
        return self.population.mass

    @property
    def active(self) -> Array:
        return self.population.active


class DEMRuntimeState(StrictModule):
    kinematics: RigidBodyKinematics
    body_properties: CompliantDEMBodyProperties
    particle_history: DEMContactHistory
    boundary_histories: tuple[DEMContactHistory, ...]
    neighborhood_cache: ParticleVerletState | None
    loads: DEMResolvedLoad
    energy: DEMEnergyLedgerState
    periodic_cell: DEMPeriodicCellState | None = None
    liquid: DEMLiquidState | None = None


class DEMDiagnostics(StrictModule):
    active_contacts: Array
    sticking_contacts: Array
    sliding_contacts: Array
    cohesion_births: Array
    cohesion_ruptures: Array
    rolling_yield_contacts: Array
    torsional_yield_contacts: Array
    maximum_overlap_fraction: Array
    minimum_gap_margin: Array
    minimum_no_tension_margin: Array
    minimum_frame_transport_margin: Array
    minimum_cohesion_birth_margin: Array
    minimum_cohesion_rupture_margin: Array
    minimum_cohesion_model_validity_margin: Array
    minimum_cohesion_fit_extrapolation_margin: Array
    liquid_balance_residual: Array
    evaporated_liquid_volume: Array
    minimum_rolling_yield_margin: Array
    minimum_torsional_yield_margin: Array
    acceptance_margin: Array
    minimum_friction_switch_margin: Array
    neighborhood_rebuilt: Array
    neighborhood_rebuild_count: Array
    neighborhood_certificate_margin: Array
    total_linear_momentum: Array
    total_angular_momentum: Array
    kinetic_energy: Array
    elastic_energy: Array
    gravity_potential_energy: Array
    energy: DEMEnergyLedgerState
    net_internal_force: Array
    net_internal_torque: Array
    maximum_friction_cone_defect: Array
    wall_action_reaction_defect: Array
    contact_history_continuity_defect: Array
    multicontact_residual: Array
    minimum_multicontact_regularity_margin: Array
    maximum_bridge_volume_residual: Array
    successful: Array
    rejection_reasons: Array
    minimum_shape_query_margin: Array
    minimum_curvature_margin: Array
    contact_capacity_margin: Array


class DEMStepRestriction(StrictModule):
    contact_period: Array
    rayleigh: Array
    selected: Array


class DEMEvaluation(StrictModule):
    neighborhood: ParticleNeighborhoodState
    neighborhood_cache: ParticleVerletState | None
    geometry: RigidContactGeometryBatch
    particle_contact: DEMContactResponse
    boundaries: tuple[DEMBoundaryResponse, ...]
    loads: DEMResolvedLoad
    diagnostics: DEMDiagnostics
    bulk_stress: DEMBulkStress | None
    liquid: DEMLiquidEvaluation | None
    contact_energy: Array
    contact_births: Array
    contact_deaths: Array
    cohesion_births: Array
    cohesion_ruptures: Array
    multicontact: DEMMulticontactCorrection | None
    rolling_yield_events: Array
    torsional_yield_events: Array
    boundary_wall_power: Array
    stick_to_slip_events: Array
    slip_to_stick_events: Array
    work: Array
    successful: Array

    rejection_reasons: Array


class DEMStepEvaluation(StrictModule):
    candidate_state: DEMRuntimeState
    accepted_state: DEMRuntimeState
    evaluation: DEMEvaluation
    energy: DEMStepEnergyLedger
    successful: Array
    residual: Array
    work: Array

    rejection_reasons: Array


class DEMBodyPropertyUpdateResult(StrictModule):
    candidate_state: DEMRuntimeState
    accepted_state: DEMRuntimeState
    evaluation: DEMEvaluation
    successful: Array


class DEMStateGeometry(AbstractStateGeometry):
    """Rigid-pose charts and additive continuous history with frozen routes."""

    rigid: RigidBodyStateGeometry
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    @checked
    def __init__(self, dynamics_id: str, bodies: PreparedRigidBodySet, /) -> None:
        if not dynamics_id:
            raise ValueError(
                f"DEMStateGeometry.dynamics_id must be nonempty; got {dynamics_id!r}"
            )
        self.rigid = RigidBodyStateGeometry(bodies)
        self.geometry_id = f"state-geometry:dem:{dynamics_id}"
        self.retraction_method = "rigid-pose-and-continuous-history"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    @checked
    def contains(self, state: DEMRuntimeState, /) -> Array:
        return tree_allfinite(state) & self.rigid.contains(state.kinematics)

    def _continuous_role(
        self, state: DEMRuntimeState, vector: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        # Angular-vector roles differ from scalar-first quaternion points.
        record = eqx.tree_at(lambda value: value.kinematics, vector, state.kinematics)
        continuous = _continuous_tangent(state, record)
        return eqx.tree_at(lambda value: value.kinematics, continuous, vector.kinematics)

    @checked
    def project_tangent(
        self, state: DEMRuntimeState, vector: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        result = self._continuous_role(state, vector)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.project_tangent(state.kinematics, vector.kinematics),
        )

    @checked
    def retract(
        self, state: DEMRuntimeState, local_tangent: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        local = self._continuous_role(state, local_tangent)
        result = jax.tree.map(
            lambda base, tangent: base + tangent if eqx.is_inexact_array(base) else base,
            state,
            local,
            is_leaf=lambda value: isinstance(value, RigidBodyKinematics),
        )
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.retract(state.kinematics, local.kinematics),
        )

    @checked
    def inverse_retract(
        self, state: DEMRuntimeState, point: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        target = _require_frozen_compatible(state, point)
        result = jax.tree.map(
            lambda base, value: (
                base
                if isinstance(base, RigidBodyKinematics)
                else value - base
                if eqx.is_inexact_array(base)
                else jnp.zeros_like(base)
            ),
            state,
            target,
            is_leaf=lambda value: isinstance(value, RigidBodyKinematics),
        )
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.inverse_retract(state.kinematics, point.kinematics),
        )

    @checked
    def retraction_jvp(
        self,
        state: DEMRuntimeState,
        local_tangent: DEMRuntimeState,
        local_velocity: DEMRuntimeState,
        /,
    ) -> DEMRuntimeState:
        result = self._continuous_role(state, local_velocity)
        self._continuous_role(state, local_tangent)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.retraction_jvp(
                state.kinematics,
                local_tangent.kinematics,
                local_velocity.kinematics,
            ),
        )

    @checked
    def retraction_inverse_jvp(
        self, state: DEMRuntimeState, point: DEMRuntimeState, tangent: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        _require_frozen_compatible(state, point)
        result = self._continuous_role(point, tangent)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.retraction_inverse_jvp(
                state.kinematics, point.kinematics, tangent.kinematics
            ),
        )

    @checked
    def retraction_vjp(
        self,
        state: DEMRuntimeState,
        local_tangent: DEMRuntimeState,
        cotangent: DEMRuntimeState,
        /,
    ) -> DEMRuntimeState:
        result = self._continuous_role(state, cotangent)
        self._continuous_role(state, local_tangent)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.retraction_vjp(
                state.kinematics, local_tangent.kinematics, cotangent.kinematics
            ),
        )

    @checked
    def transport_tangent(
        self, state: DEMRuntimeState, point: DEMRuntimeState, tangent: DEMRuntimeState, /
    ) -> DEMRuntimeState:
        _require_frozen_compatible(state, point)
        result = self._continuous_role(state, tangent)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.transport_tangent(
                state.kinematics, point.kinematics, tangent.kinematics
            ),
        )

    @checked
    def transport_cotangent_pullback(
        self,
        state: DEMRuntimeState,
        point: DEMRuntimeState,
        cotangent: DEMRuntimeState,
        /,
    ) -> DEMRuntimeState:
        _require_frozen_compatible(state, point)
        result = self._continuous_role(state, cotangent)
        return eqx.tree_at(
            lambda value: value.kinematics,
            result,
            self.rigid.transport_cotangent_pullback(
                state.kinematics, point.kinematics, cotangent.kinematics
            ),
        )

    @checked
    def cut_locus_margin(
        self, state: DEMRuntimeState, point: DEMRuntimeState, /
    ) -> Array:
        _require_frozen_compatible(state, point)
        return self.rigid.cut_locus_margin(state.kinematics, point.kinematics)


class _DEMBodyPhase(StrictModule):
    properties: CompliantDEMBodyProperties
    radii: Array
    body_valid: Array
    kinematics: RigidBodyKinematics


class _DEMMulticontactPhase(StrictModule):
    batch: DEMContactBatch
    particle_contact: DEMContactResponse
    multicontact: DEMMulticontactCorrection | None


class _DEMRoutePhase(StrictModule):
    neighborhood_cache: ParticleVerletState | None
    neighborhood: ParticleNeighborhoodState
    geometry_batch: RigidContactGeometryBatch
    remapped_history: DEMContactHistory
    continued: Array
    alignment_successful: Array
    left: Array
    right: Array
    batch: DEMContactBatch


class _DEMPairLiquidPhase(StrictModule):
    liquid_allocation: DEMLiquidAllocation | None
    minimum_bridge_volume: Array
    remapped_history: DEMContactHistory


class _DEMBoundaryLiquidPhase(StrictModule):
    boundary_responses: tuple[DEMBoundaryResponse, ...]
    barrier_allocation: DEMBarrierLiquidAllocation | None
    barrier_particles: Array | None
    barrier_indices: Array | None
    barrier_minimum_volume: Array | None


class _DEMLiquidAdvancePhase(StrictModule):
    particle_contact: DEMContactResponse
    boundary_responses: tuple[DEMBoundaryResponse, ...]
    liquid_evaluation: DEMLiquidEvaluation | None


class _DEMLoadPhase(StrictModule):
    pair_force: Array
    pair_torque: Array
    resolved_loads: DEMResolvedLoad


class _DEMEventPhase(StrictModule):
    births: Array
    deaths: Array
    stick_to_slip: Array
    slip_to_stick: Array
    cohesion_births: Array
    cohesion_ruptures: Array
    rolling_yields: Array
    torsional_yields: Array
    contact_energy: Array
    boundary_wall_power: Array


class _DEMContactDiagnostics(StrictModule):
    friction_defect: Array
    multicontact_residual: Array
    multicontact_margin: Array
    wall_defect: Array
    neighborhood_rebuilt: Array
    neighborhood_rebuild_count: Array
    neighborhood_certificate_margin: Array
    active_count: Array
    sticking_count: Array
    sliding_count: Array
    cohesion_birth_count: Array
    cohesion_rupture_count: Array
    rolling_yield_count: Array
    torsional_yield_count: Array
    bridge_residual: Array
    liquid_balance_residual: Array
    evaporated_liquid_volume: Array
    maximum_overlap: Array
    elastic_energy: Array
    friction_margin: Array
    activation_margin: Array
    no_tension_margin: Array
    frame_margin: Array
    cohesion_birth_margin: Array
    cohesion_rupture_margin: Array
    cohesion_model_margin: Array
    cohesion_extrapolation_margin: Array
    rolling_yield_margin: Array
    torsional_yield_margin: Array


class _DEMStatusPhase(StrictModule):
    reasons: Array
    successful: Array


class _DEMBindingPhase(StrictModule):
    bodies: PreparedRigidBodySet
    walls_: tuple[PreparedTriangleWall, ...]
    barriers_: tuple[ImplicitDEMBarrier, ...]
    barrier_ids: tuple[str, ...] = eqx.field(static=True)


class _DEMProcessPhase(StrictModule):
    maximum_interaction_radius: float = eqx.field(static=True)
    periodic_cell: PeriodicCell | None
    liquid_component_index: int = eqx.field(static=True)


class _DEMBoundaryAdvancePhase(StrictModule):
    boundary_responses: tuple[DEMBoundaryResponse, ...]
    liquid_evaluation: DEMLiquidEvaluation


@final
class PreparedCompliantDEMDynamics(StrictModule, NonTrainableState):
    """Prepared compliant contact dynamics for shape-prepared rigid bodies."""

    shapes: AbstractPreparedRigidContactShape
    bodies: PreparedRigidBodySet
    contact_capacity: int = eqx.field(static=True)
    neighborhood: AbstractPreparedParticleNeighborhood
    pair_key_space: ParticlePairKeySpace
    contact_model: PreparedDEMContactModel
    method: CompliantDEMMethodPlan
    materials: DEMMaterialTable
    barriers: tuple[ImplicitDEMBarrier, ...]
    walls: tuple[PreparedTriangleWall, ...]
    gravity: Array
    external_load: ExternalDEMLoad | None
    external_load_id: str | None = eqx.field(static=True)
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    periodic_cell: PeriodicCell | None
    maximum_interaction_radius: float = eqx.field(static=True)
    liquid_component_index: int = eqx.field(static=True)
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        shapes: AbstractPreparedRigidContactShape,
        neighborhood: AbstractPreparedParticleNeighborhood,
        pair_key_space: ParticlePairKeySpace,
        contact_model: PreparedDEMContactModel,
        method: CompliantDEMMethodPlan,
        materials: DEMMaterialTable,
        /,
        *,
        barriers: Sequence[ImplicitDEMBarrier] = (),
        walls: Sequence[PreparedTriangleWall] = (),
        gravity: ArrayLike | None = None,
        external_load: ExternalDEMLoad | None = None,
        external_load_id: str | None = None,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
    ) -> None:
        """Bind one identity-sensitive compliant DEM execution plan.

        Shape capabilities, contact laws, neighborhoods, walls, process
        physics, and execution backends are admitted together because their
        ordered combination defines capacities, PyTree structure, and
        ``prepared_id``. Keeping this fail-closed dispatch central avoids
        duplicated shape-law compatibility rules.
        """
        phase = self._admit_contact_bindings(
            shapes,
            neighborhood,
            pair_key_space,
            contact_model,
            method,
            materials,
            walls,
            barriers,
        )
        bodies = phase.bodies
        walls_ = phase.walls_
        barriers_ = phase.barriers_
        barrier_ids = phase.barrier_ids
        gravity_host = (
            np.zeros((bodies.ambient_dimension,), dtype=shapes.bounding_radii.dtype)
            if gravity is None
            else np.asarray(jax.device_get(gravity), dtype=shapes.bounding_radii.dtype)
        )
        if gravity_host.shape != (bodies.ambient_dimension,):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.gravity must match the rigid-sphere ambient dimension; got {gravity_host.shape!r}"
            )
        if not np.all(np.isfinite(gravity_host)):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.gravity must be finite; got {type(gravity).__name__}"
            )
        gravity_ = jnp.asarray(gravity_host)
        if external_load is not None and not callable(external_load):
            raise TypeError(
                f"PreparedCompliantDEMDynamics.external_load must be callable or None; got {type(external_load).__name__}"
            )
        if external_load is None and external_load_id is not None:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.external_load must provide external_load; got {external_load!r}"
            )
        if external_load is not None and not external_load_id:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.external_load must provide a stable nonempty ID; got {type(external_load).__name__}"
            )
        execution_ = ParticleExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if execution_.realization != neighborhood.backend:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.neighborhood must match neighborhood backend; got {type(neighborhood).__name__}"
            )
        if (
            execution_.kernel_backend == "dense_fused"
            and neighborhood.backend != "dense_pairs"
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.neighborhood must provide a dense-pair neighborhood; got {type(neighborhood).__name__}"
            )
        if (
            execution_.kernel_backend == "cell_fused"
            and neighborhood.backend != "cell_edge_list"
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.neighborhood must provide a cell-list neighborhood; got {type(neighborhood).__name__}"
            )
        if execution_.kernel_backend == "verlet_fused" and not isinstance(
            neighborhood, PreparedVerletParticleNeighborhood
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.neighborhood must provide a prepared Verlet neighborhood; got {type(neighborhood).__name__}"
            )
        phase = self._prepare_process_bindings(
            bodies,
            shapes,
            method,
            materials,
            barriers_,
            barrier_ids,
            neighborhood,
            gravity_,
        )
        maximum_interaction_radius = phase.maximum_interaction_radius
        periodic_cell = phase.periodic_cell
        liquid_component_index = phase.liquid_component_index
        requested_capacity = (
            neighborhood.pair_capacity * shapes.routes_per_pair
            if method.maximum_contact_routes is None
            else method.maximum_contact_routes
        )
        contact_capacity = shapes.contact_capacity(
            neighborhood.pair_capacity, requested_capacity
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY,
            ),
            diagnostics=(
                "shape-generic compliant contact",
                "stable pair-key history remapping",
                (
                    "conserved particle-film and bridge-volume transaction"
                    if liquid_component_index >= 0
                    else "prescribed contact liquid sources"
                ),
                "kick-drift-contact-kick integration",
                "branchwise differentiation through fixed realized routes",
                (
                    "deforming fully periodic cell with mixed tensor control"
                    if periodic_cell is not None
                    else "fixed particle domain"
                ),
            ),
            resource_counts={
                "particle_capacity": bodies.capacity,
                "pair_capacity": neighborhood.pair_capacity,
                "contact_capacity": contact_capacity,
                "wall_count": len(walls_),
                "barrier_count": len(barriers_),
                "ambient_dimension": bodies.ambient_dimension,
            },
        )
        self.bodies = bodies
        self.shapes = shapes
        self.contact_capacity = contact_capacity
        self.neighborhood = neighborhood
        self.pair_key_space = pair_key_space
        self.contact_model = contact_model
        self.method = method
        self.materials = materials
        self.barriers = barriers_
        self.walls = walls_
        self.gravity = gravity_
        self.external_load = external_load
        self.external_load_id = (
            None if external_load_id is None else str(external_load_id)
        )
        self.execution = execution_
        self.liquid_component_index = liquid_component_index
        self.precision = precision_
        self.periodic_cell = periodic_cell
        self.maximum_interaction_radius = maximum_interaction_radius
        self.key = method.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-compliant-dem-dynamics",
                "bodies": bodies.prepared_id,
                "shapes": shapes.shape_id,
                "contact_capacity": contact_capacity,
                "walls": [wall.prepared_id for wall in walls_],
                "neighborhood": neighborhood.prepared_id,
                "pair_key_space": pair_key_space.key_space_id,
                "contact_model": contact_model.prepared_id,
                "method": method.method_id,
                "materials": materials.material_id,
                "barriers": list(barrier_ids),
                "gravity": np.asarray(gravity_).tolist(),
                "external_load": external_load_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def state_geometry(self) -> DEMStateGeometry:
        return DEMStateGeometry(self.prepared_id, self.bodies)

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def initial_body_properties(self) -> CompliantDEMBodyProperties:
        population = ParticlePopulationPlan(self.bodies.particles).initialize(
            active_mask=self.bodies.particles.active_mask,
            masses=self.bodies.particles.safe_masses,
        )
        return CompliantDEMBodyProperties(
            population,
            self.bodies.inverse_masses,
            jnp.ones_like(self.shapes.bounding_radii),
            self.bodies.inertia_com,
            self.bodies.inverse_inertia_com,
            self.shapes.shape_id,
        )

    def empty_particle_history(self) -> DEMContactHistory:
        return self.contact_model.empty_history(
            self.contact_capacity,
            self.shapes.bounding_radii.dtype,
        )

    def empty_boundary_histories(self) -> tuple[DEMContactHistory, ...]:
        capacities = (self.bodies.capacity,) * len(self.barriers) + tuple(
            self.bodies.capacity * wall.face_count for wall in self.walls
        )
        return tuple(
            self.contact_model.empty_history(capacity, self.shapes.bounding_radii.dtype)
            for capacity in capacities
        )

    def _zero_load(self) -> RigidBodyLoad:
        return RigidBodyLoad(
            jnp.zeros(
                (self.bodies.capacity, self.bodies.ambient_dimension),
                dtype=self.shapes.bounding_radii.dtype,
            ),
            jnp.zeros(
                (self.bodies.capacity, self.bodies.angular_dimension),
                dtype=self.shapes.bounding_radii.dtype,
            ),
        )

    def _angular_momentum(
        self,
        kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        /,
    ) -> Array:
        if self.bodies.ambient_dimension == 2:
            return properties.inertia_com[:, None] * kinematics.angular_velocity
        if self.shapes.shape_kind == "sphere":
            return properties.inertia_com[:, 0, 0, None] * kinematics.angular_velocity
        inertia, _ = rigid_body_world_inertia(
            self.bodies,
            kinematics.orientation,
            inertia_com=properties.inertia_com,
            inverse_inertia_com=properties.inverse_inertia_com,
        )
        return (inertia @ kinematics.angular_velocity[..., None])[..., 0]

    def _angular_acceleration(
        self,
        kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        torque: Array,
        /,
    ) -> Array:
        if self.bodies.ambient_dimension == 2:
            return properties.inverse_inertia_com[:, None] * torque
        if self.shapes.shape_kind == "sphere":
            return properties.inverse_inertia_com[:, 0, 0, None] * torque
        return rigid_body_angular_acceleration(
            self.bodies,
            kinematics,
            torque,
            inertia_com=properties.inertia_com,
            inverse_inertia_com=properties.inverse_inertia_com,
        )

    def _kinetic_energy(
        self,
        kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        /,
    ) -> Array:
        return 0.5 * jnp.sum(
            properties.masses[:, None] * kinematics.velocity**2
        ) + 0.5 * jnp.sum(
            self._angular_momentum(kinematics, properties) * kinematics.angular_velocity
        )

    def _gravity_potential(
        self,
        kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        /,
    ) -> Array:
        potential = -properties.masses * jnp.sum(
            kinematics.position * self.gravity[None, :], axis=-1
        )
        return jnp.sum(jnp.where(properties.active, potential, 0.0))

    def _ledger_view(
        self,
        ledger: DEMEnergyLedgerState,
        kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        contact_energy: Array,
        boundary_wall_power: Array,
        /,
    ) -> DEMEnergyLedgerState:
        return DEMEnergyLedgerState(
            ledger.initial_kinetic_energy,
            ledger.initial_contact_energy,
            ledger.initial_gravity_potential,
            self._kinetic_energy(kinematics, properties),
            contact_energy,
            self._gravity_potential(kinematics, properties),
            boundary_wall_power,
            ledger.cumulative_particle_contact_work,
            ledger.cumulative_boundary_contact_work,
            ledger.cumulative_prescribed_wall_work,
            ledger.cumulative_gravity_work,
            ledger.cumulative_external_work,
            ledger.cumulative_cell_work,
            ledger.cumulative_contact_balance_loss,
            ledger.cumulative_energy_residual,
            ledger.last_relative_energy_residual,
            ledger.contact_births,
            ledger.contact_deaths,
            ledger.stick_to_slip_events,
            ledger.slip_to_stick_events,
            ledger.accepted_steps,
        )

    def _source_work(
        self,
        previous: RigidBodyLoad,
        current: RigidBodyLoad,
        previous_kinematics: RigidBodyKinematics,
        current_kinematics: RigidBodyKinematics,
        properties: CompliantDEMBodyProperties,
        step_size: Array,
        /,
    ) -> Array:
        impulse = 0.5 * step_size * (previous.force + current.force)
        angular_impulse = 0.5 * step_size * (previous.torque + current.torque)
        average_velocity = 0.5 * (
            previous_kinematics.velocity + current_kinematics.velocity
        )
        average_angular_velocity = 0.5 * (
            previous_kinematics.angular_velocity + current_kinematics.angular_velocity
        )
        mobile = properties.active & ~self.bodies.fixed_mask
        translational = jnp.sum(impulse * average_velocity, axis=-1)
        rotational = jnp.sum(angular_impulse * average_angular_velocity, axis=-1)
        return jnp.sum(jnp.where(mobile, translational + rotational, 0.0))

    def _step_energy(
        self,
        state: DEMRuntimeState,
        next_kinematics: RigidBodyKinematics,
        evaluation: DEMEvaluation,
        step_size: Array,
        cell_work: Array,
        /,
    ) -> DEMStepEnergyLedger:
        kinetic_before = self._kinetic_energy(state.kinematics, state.body_properties)
        kinetic_after = self._kinetic_energy(next_kinematics, state.body_properties)
        contact_before = state.energy.contact_energy
        contact_after = evaluation.contact_energy
        gravity_before = self._gravity_potential(state.kinematics, state.body_properties)
        gravity_after = self._gravity_potential(next_kinematics, state.body_properties)
        particle_work = self._source_work(
            state.loads.particle_contact,
            evaluation.loads.particle_contact,
            state.kinematics,
            next_kinematics,
            state.body_properties,
            step_size,
        )
        boundary_work = (
            jnp.stack(
                tuple(
                    self._source_work(
                        previous,
                        current,
                        state.kinematics,
                        next_kinematics,
                        state.body_properties,
                        step_size,
                    )
                    for previous, current in zip(
                        state.loads.boundaries,
                        evaluation.loads.boundaries,
                        strict=True,
                    )
                )
            )
            if self.barriers
            else jnp.zeros((0,), dtype=kinetic_after.dtype)
        )
        prescribed_wall_work = (
            0.5
            * step_size
            * (state.energy.current_boundary_wall_power + evaluation.boundary_wall_power)
        )
        gravity_work = self._source_work(
            state.loads.gravity,
            evaluation.loads.gravity,
            state.kinematics,
            next_kinematics,
            state.body_properties,
            step_size,
        )
        external_work = self._source_work(
            state.loads.external,
            evaluation.loads.external,
            state.kinematics,
            next_kinematics,
            state.body_properties,
            step_size,
        )
        contact_work = particle_work + jnp.sum(boundary_work)
        contact_delta = contact_after - contact_before
        balance_loss = jnp.sum(prescribed_wall_work) - contact_work - contact_delta
        residual = (
            kinetic_after
            - kinetic_before
            + contact_delta
            + balance_loss
            - gravity_work
            - external_work
            - cell_work
            - jnp.sum(prescribed_wall_work)
        )
        scale = jnp.maximum(
            jnp.max(
                jnp.stack(
                    (
                        jnp.abs(kinetic_before),
                        jnp.abs(kinetic_after),
                        jnp.abs(contact_before),
                        jnp.abs(contact_after),
                        jnp.abs(gravity_work),
                        jnp.abs(external_work),
                        jnp.abs(cell_work),
                        jnp.max(
                            jnp.abs(prescribed_wall_work),
                            initial=jnp.asarray(0.0, dtype=kinetic_after.dtype),
                        ),
                    )
                )
            ),
            jnp.asarray(1.0e-30, dtype=kinetic_after.dtype),
        )
        return DEMStepEnergyLedger(
            kinetic_before,
            kinetic_after,
            contact_before,
            contact_after,
            gravity_before,
            gravity_after,
            particle_work,
            boundary_work,
            prescribed_wall_work,
            evaluation.boundary_wall_power,
            gravity_work,
            external_work,
            cell_work,
            balance_loss,
            residual,
            jnp.abs(residual) / scale,
            evaluation.contact_births,
            evaluation.contact_deaths,
            evaluation.stick_to_slip_events,
            evaluation.slip_to_stick_events,
            jnp.asarray(False),
        )

    def _accumulated_energy(
        self,
        previous: DEMEnergyLedgerState,
        step: DEMStepEnergyLedger,
        /,
    ) -> DEMEnergyLedgerState:
        return DEMEnergyLedgerState(
            previous.initial_kinetic_energy,
            previous.initial_contact_energy,
            previous.initial_gravity_potential,
            step.kinetic_after,
            step.contact_energy_after,
            step.gravity_potential_after,
            step.boundary_wall_power_after,
            previous.cumulative_particle_contact_work + step.particle_contact_work,
            previous.cumulative_boundary_contact_work + step.boundary_contact_work,
            previous.cumulative_prescribed_wall_work + step.prescribed_wall_work,
            previous.cumulative_gravity_work + step.gravity_work,
            previous.cumulative_external_work + step.external_work,
            previous.cumulative_cell_work + step.cell_work,
            previous.cumulative_contact_balance_loss + step.contact_balance_loss,
            previous.cumulative_energy_residual + step.energy_residual,
            step.relative_energy_residual,
            previous.contact_births + step.contact_births,
            previous.contact_deaths + step.contact_deaths,
            previous.stick_to_slip_events + step.stick_to_slip_events,
            previous.slip_to_stick_events + step.slip_to_stick_events,
            previous.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )

    def initialize_state(
        self,
        time: ArrayLike,
        position: ArrayLike,
        velocity: ArrayLike,
        angular_velocity: ArrayLike | None = None,
        /,
        *,
        orientation: ArrayLike | None = None,
        args: Any = None,
    ) -> DEMRuntimeState:
        raw_position = jnp.asarray(position)
        if self.periodic_cell is None:
            periodic_state = None
        else:
            periodic_control = self.method.periodic_cell_control
            if periodic_control is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.initialize_state.time must provide the prepared transaction state; got {periodic_control!r}"
                )
            periodic_state = periodic_control.initialize(
                self.periodic_cell, raw_position.dtype
            )
        initial_position = raw_position
        if periodic_state is not None:
            periodic_cell = self.periodic_cell
            if periodic_cell is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.initialize_state.time must provide the prepared transaction state; got {periodic_cell!r}"
                )
            initial_position, _ = periodic_cell.wrap_with_vectors(
                raw_position, periodic_state.vectors
            )
        kinematics = _initial_kinematics(
            self, raw_position, initial_position, velocity, angular_velocity, orientation
        )
        body_properties = self.initial_body_properties()
        liquid_state = (
            None
            if self.method.liquid_process is None
            else self.method.liquid_process.initialize(
                self.bodies.capacity,
                kinematics.position.dtype,
                body_properties.active,
            )
        )
        zero = self._zero_load()
        resolved_zero = DEMResolvedLoad(
            zero,
            tuple(self._zero_load() for _ in self.barriers),
            self._zero_load(),
            self._zero_load(),
            self._zero_load(),
        )
        kinetic = self._kinetic_energy(kinematics, body_properties)
        gravity_potential = self._gravity_potential(kinematics, body_properties)
        scalar_zero = jnp.zeros((), dtype=kinematics.position.dtype)
        zero_boundary = jnp.zeros((len(self.barriers),), dtype=kinematics.position.dtype)
        energy_seed = DEMEnergyLedgerState(
            kinetic,
            scalar_zero,
            gravity_potential,
            kinetic,
            scalar_zero,
            gravity_potential,
            zero_boundary,
            scalar_zero,
            zero_boundary,
            zero_boundary,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        neighborhood_cache = (
            self.neighborhood.initialize(
                kinematics.position,
                active_mask=body_properties.active,
                orientation=None
                if self.shapes.shape_kind == "sphere"
                else kinematics.orientation,
                radii=None
                if self.shapes.shape_kind == "sphere"
                else self.shapes.bounding_radii * body_properties.shape_scale,
            )
            if isinstance(self.neighborhood, PreparedVerletParticleNeighborhood)
            else None
        )
        seed = DEMRuntimeState(
            kinematics,
            body_properties,
            self.empty_particle_history(),
            self.empty_boundary_histories(),
            neighborhood_cache,
            resolved_zero,
            energy_seed,
            periodic_state,
            liquid_state,
        )
        evaluation = self.evaluate(
            jnp.asarray(time, dtype=kinematics.position.dtype),
            seed,
            jnp.zeros((), dtype=kinematics.position.dtype),
            args,
        )
        initialized_energy = DEMEnergyLedgerState(
            kinetic,
            evaluation.contact_energy,
            gravity_potential,
            kinetic,
            evaluation.contact_energy,
            gravity_potential,
            evaluation.boundary_wall_power,
            scalar_zero,
            zero_boundary,
            zero_boundary,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            scalar_zero,
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        initialized_liquid = (
            liquid_state if evaluation.liquid is None else evaluation.liquid.next_state
        )
        initialized = DEMRuntimeState(
            kinematics,
            body_properties,
            evaluation.particle_contact.next_history,
            tuple(value.contact.next_history for value in evaluation.boundaries),
            evaluation.neighborhood_cache,
            evaluation.loads,
            initialized_energy,
            periodic_state,
            initialized_liquid,
        )
        checked_position = eqx.error_if(
            initialized.kinematics.position,
            ~evaluation.successful,
            "Initial DEM contact state is not admissible.",
        )
        return DEMRuntimeState(
            RigidBodyKinematics(
                checked_position,
                initialized.kinematics.velocity,
                initialized.kinematics.orientation,
                initialized.kinematics.angular_velocity,
            ),
            initialized.body_properties,
            initialized.particle_history,
            initialized.boundary_histories,
            initialized.neighborhood_cache,
            initialized.loads,
            initialized.energy,
            initialized.periodic_cell,
            initialized.liquid,
        )

    @checked
    def apply_body_properties(
        self,
        time: Array,
        state: DEMRuntimeState,
        properties: CompliantDEMBodyProperties,
        rebuild_neighborhood: Array,
        /,
        *,
        args: Any = None,
    ) -> DEMBodyPropertyUpdateResult:
        mobile = (properties.active & ~self.bodies.fixed_mask)[:, None]
        kinematics = RigidBodyKinematics(
            state.kinematics.position,
            jnp.where(mobile, state.kinematics.velocity, 0.0),
            state.kinematics.orientation,
            jnp.where(mobile, state.kinematics.angular_velocity, 0.0),
        )
        cache = state.neighborhood_cache
        match self.neighborhood:
            case PreparedVerletParticleNeighborhood():
                if cache is None:
                    raise ValueError(
                        f"PreparedCompliantDEMDynamics.apply_body_properties.state must provide a neighborhood cache; got {cache!r}"
                    )
                initialized = self.neighborhood.initialize(
                    kinematics.position,
                    active_mask=properties.active,
                    orientation=None
                    if self.shapes.shape_kind == "sphere"
                    else kinematics.orientation,
                    radii=None
                    if self.shapes.shape_kind == "sphere"
                    else self.shapes.bounding_radii * properties.shape_scale,
                )
                cache = tree_where(jnp.asarray(rebuild_neighborhood), initialized, cache)
            case _:
                pass
        staged = DEMRuntimeState(
            kinematics,
            properties,
            state.particle_history,
            state.boundary_histories,
            cache,
            state.loads,
            state.energy,
            state.periodic_cell,
            state.liquid,
        )
        evaluation = self.evaluate(
            jnp.asarray(time),
            staged,
            jnp.zeros((), dtype=kinematics.position.dtype),
            args,
        )
        energy = self._ledger_view(
            state.energy,
            kinematics,
            properties,
            evaluation.contact_energy,
            evaluation.boundary_wall_power,
        )
        candidate = DEMRuntimeState(
            kinematics,
            properties,
            evaluation.particle_contact.next_history,
            tuple(value.contact.next_history for value in evaluation.boundaries),
            evaluation.neighborhood_cache,
            evaluation.loads,
            energy,
            state.periodic_cell,
            state.liquid if evaluation.liquid is None else evaluation.liquid.next_state,
        )
        successful = evaluation.successful & tree_allfinite(candidate)
        accepted = tree_where(successful, candidate, state)
        return DEMBodyPropertyUpdateResult(candidate, accepted, evaluation, successful)

    @checked
    def evaluate(
        self,
        time: Array,
        state: DEMRuntimeState,
        step_size: Array,
        args: Any,
        /,
    ) -> DEMEvaluation:
        """Evaluate one ordered contact/history/liquid transaction.

        Liquid allocation precedes history updates and deterministic load
        reductions. Each phase preserves failure evidence for rollback.
        """
        phase = self._prepare_body_state(state)
        properties = phase.properties
        radii = phase.radii
        body_valid = phase.body_valid
        kinematics = phase.kinematics
        phase = self._prepare_contact_routes(state, properties, radii, kinematics)
        neighborhood_cache = phase.neighborhood_cache
        neighborhood = phase.neighborhood
        geometry_batch = phase.geometry_batch
        geometry = geometry_batch.geometry
        keys = geometry_batch.keys
        remapped_history = phase.remapped_history
        continued = phase.continued
        alignment_successful = phase.alignment_successful
        left = phase.left
        right = phase.right
        batch = phase.batch
        phase = self._allocate_pair_liquid(state, batch, left, right, remapped_history)
        liquid_allocation = phase.liquid_allocation
        minimum_bridge_volume = phase.minimum_bridge_volume
        remapped_history = phase.remapped_history
        contact_context = DEMContactEvaluationContext(
            keys.keys,
            keys.valid,
            continued,
            state.body_properties.inverse_masses[left],
            state.body_properties.inverse_masses[right],
            radii[left],
            radii[right],
            geometry_batch.left_material_ids,
            geometry_batch.right_material_ids,
            step_size,
            -jnp.ones((), dtype=jnp.int32),
        )
        particle_contact = self.contact_model.evaluate(
            batch,
            remapped_history,
            contact_context,
            frame_tolerance=self.method.frame_tolerance,
        )
        phase = self._correct_multicontact(
            batch,
            particle_contact,
            kinematics,
            left,
            right,
            radii,
            remapped_history,
            contact_context,
        )
        batch = phase.batch
        particle_contact = phase.particle_contact
        multicontact = phase.multicontact

        phase = self._allocate_boundary_liquid(
            state, kinematics, time, step_size, args, liquid_allocation, batch
        )
        boundary_responses = phase.boundary_responses
        barrier_allocation = phase.barrier_allocation
        barrier_particles = phase.barrier_particles
        barrier_indices = phase.barrier_indices
        barrier_minimum_volume = phase.barrier_minimum_volume

        phase = self._advance_liquid(
            state,
            liquid_allocation,
            batch,
            particle_contact,
            left,
            right,
            minimum_bridge_volume,
            step_size,
            boundary_responses,
            barrier_allocation,
            barrier_particles,
            barrier_indices,
            barrier_minimum_volume,
        )
        particle_contact = phase.particle_contact
        boundary_responses = phase.boundary_responses
        liquid_evaluation = phase.liquid_evaluation
        phase = self._resolve_loads(
            geometry,
            particle_contact,
            boundary_responses,
            properties,
            time,
            kinematics,
            args,
        )
        pair_force = phase.pair_force
        pair_torque = phase.pair_torque
        resolved_loads = phase.resolved_loads
        phase = self._contact_events(
            state, remapped_history, particle_contact, boundary_responses
        )
        births = phase.births
        deaths = phase.deaths
        stick_to_slip = phase.stick_to_slip
        slip_to_stick = phase.slip_to_stick
        cohesion_births = phase.cohesion_births
        cohesion_ruptures = phase.cohesion_ruptures
        rolling_yields = phase.rolling_yields
        torsional_yields = phase.torsional_yields
        contact_energy = phase.contact_energy
        boundary_wall_power = phase.boundary_wall_power
        phase = self._evaluation_status(
            body_valid,
            neighborhood,
            keys,
            alignment_successful,
            geometry_batch,
            batch,
            geometry,
            particle_contact,
            boundary_responses,
            liquid_evaluation,
            resolved_loads,
            state,
            contact_energy,
        )
        reasons = phase.reasons
        successful = phase.successful
        diagnostic_energy = self._ledger_view(
            state.energy,
            kinematics,
            state.body_properties,
            contact_energy,
            boundary_wall_power,
        )
        diagnostics = self._diagnostics(
            kinematics,
            neighborhood_cache,
            particle_contact,
            geometry_batch,
            multicontact,
            boundary_responses,
            liquid_evaluation,
            state.body_properties,
            pair_force,
            pair_torque,
            diagnostic_energy,
            successful,
            reasons,
        )
        work = (
            geometry_batch.candidate_count
            + jnp.sum(properties.active, dtype=jnp.int32) * len(self.barriers)
            + sum(
                jnp.sum(response.geometry.valid, dtype=jnp.int32)
                for response in boundary_responses[len(self.barriers) :]
            )
        )
        result = DEMEvaluation(
            neighborhood=neighborhood,
            neighborhood_cache=neighborhood_cache,
            geometry=geometry_batch,
            particle_contact=particle_contact,
            boundaries=boundary_responses,
            loads=resolved_loads,
            diagnostics=diagnostics,
            bulk_stress=None,
            liquid=liquid_evaluation,
            contact_energy=contact_energy,
            contact_births=births,
            contact_deaths=deaths,
            cohesion_births=cohesion_births,
            cohesion_ruptures=cohesion_ruptures,
            multicontact=multicontact,
            rolling_yield_events=rolling_yields,
            torsional_yield_events=torsional_yields,
            boundary_wall_power=boundary_wall_power,
            stick_to_slip_events=stick_to_slip,
            slip_to_stick_events=slip_to_stick,
            work=work.astype(jnp.int32),
            successful=successful,
            rejection_reasons=reasons,
        )
        if state.periodic_cell is not None:
            result = eqx.tree_at(
                lambda value: value.bulk_stress,
                result,
                dem_bulk_stress(self, state, result),
                is_leaf=lambda value: value is None,
            )
        return result

    @checked
    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: DEMRuntimeState,
        step_size: Array,
        args: Any,
        /,
    ) -> DEMStepEvaluation:
        del step_index
        half = 0.5 * step_size
        mobile = (state.body_properties.active & ~self.bodies.fixed_mask)[:, None]
        half_velocity = state.kinematics.velocity + half * (
            state.body_properties.inverse_masses[:, None] * state.loads.total.force
        )
        half_angular = state.kinematics.angular_velocity + half * (
            self._angular_acceleration(
                state.kinematics, state.body_properties, state.loads.total.torque
            )
        )
        half_velocity = jnp.where(mobile, half_velocity, 0.0)
        half_angular = jnp.where(mobile, half_angular, 0.0)
        drifted = rigid_body_drift(
            self.bodies,
            RigidBodyKinematics(
                state.kinematics.position,
                half_velocity,
                state.kinematics.orientation,
                half_angular,
            ),
            step_size,
        )
        next_position = drifted.position
        next_orientation = drifted.orientation
        if state.periodic_cell is not None:
            periodic_cell = self.periodic_cell
            if periodic_cell is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.step_detailed.state must provide the prepared transaction state; got {periodic_cell!r}"
                )
            next_position, _ = periodic_cell.wrap_with_vectors(
                next_position, state.periodic_cell.vectors
            )
        staged = DEMRuntimeState(
            RigidBodyKinematics(
                next_position, half_velocity, next_orientation, half_angular
            ),
            state.body_properties,
            state.particle_history,
            state.boundary_histories,
            state.neighborhood_cache,
            state.loads,
            state.energy,
            state.periodic_cell,
            state.liquid,
        )
        evaluation = self.evaluate(time + step_size, staged, step_size, args)
        cell_work = jnp.zeros((), dtype=next_position.dtype)
        cell_successful = jnp.asarray(True)
        periodic_state = state.periodic_cell
        if self.method.periodic_cell_control is not None:
            if evaluation.bulk_stress is None or periodic_state is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.step_detailed.state must provide bulk stress; got {evaluation.bulk_stress!r}"
                )
            periodic_cell = self.periodic_cell
            if periodic_cell is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.step_detailed.state must provide the prepared transaction state; got {periodic_cell!r}"
                )
            cell_update = self.method.periodic_cell_control.update(
                periodic_cell,
                periodic_state,
                next_position,
                half_velocity,
                evaluation.bulk_stress.total_stress,
                step_size,
                self.maximum_interaction_radius,
            )
            next_position = cell_update.position
            half_velocity = jnp.where(mobile, cell_update.velocity, 0.0)
            periodic_state = cell_update.state
            cell_work = cell_update.work
            cell_successful = cell_update.successful
            staged = DEMRuntimeState(
                RigidBodyKinematics(
                    next_position, half_velocity, next_orientation, half_angular
                ),
                state.body_properties,
                state.particle_history,
                state.boundary_histories,
                state.neighborhood_cache,
                state.loads,
                state.energy,
                periodic_state,
                state.liquid,
            )
            evaluation = self.evaluate(time + step_size, staged, step_size, args)
        next_velocity = half_velocity + half * (
            state.body_properties.inverse_masses[:, None] * evaluation.loads.total.force
        )
        next_angular = half_angular + half * (
            self._angular_acceleration(
                staged.kinematics, state.body_properties, evaluation.loads.total.torque
            )
        )
        next_velocity = jnp.where(mobile, next_velocity, 0.0)
        next_angular = jnp.where(mobile, next_angular, 0.0)
        next_kinematics = RigidBodyKinematics(
            next_position, next_velocity, next_orientation, next_angular
        )
        energy = self._step_energy(
            state, next_kinematics, evaluation, step_size, cell_work
        )
        candidate_energy = self._accumulated_energy(state.energy, energy)
        candidate = DEMRuntimeState(
            next_kinematics,
            state.body_properties,
            evaluation.particle_contact.next_history,
            tuple(value.contact.next_history for value in evaluation.boundaries),
            evaluation.neighborhood_cache,
            evaluation.loads,
            candidate_energy,
            periodic_state,
            state.liquid if evaluation.liquid is None else evaluation.liquid.next_state,
        )
        overlap_residual = evaluation.diagnostics.maximum_overlap_fraction
        residual = jnp.maximum(overlap_residual, energy.relative_energy_residual)
        reasons = evaluation.rejection_reasons
        reasons = reasons | jnp.where(
            overlap_residual > self.method.maximum_overlap_fraction,
            int(DEMRejectionReason.OVERLAP),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~cell_successful,
            int(DEMRejectionReason.CELL_CONTROL),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~(tree_allfinite(candidate) & tree_allfinite(energy)),
            int(DEMRejectionReason.ENERGY),
            0,
        ).astype(jnp.int32)
        successful = reasons == 0
        energy = eqx.tree_at(lambda value: value.accepted, energy, successful)
        accepted = tree_where(successful, candidate, state)
        return DEMStepEvaluation(
            candidate,
            accepted,
            evaluation,
            energy,
            successful,
            residual,
            evaluation.work,
            reasons,
        )

    def step_restriction(
        self, properties: CompliantDEMBodyProperties | None = None, /
    ) -> DEMStepRestriction:
        selected_properties = (
            self.initial_body_properties() if properties is None else properties
        )
        active = selected_properties.active & ~self.bodies.fixed_mask
        masses = jnp.where(active, selected_properties.masses, jnp.inf)
        minimum_mass = jnp.min(masses)
        normal = self.method.contact.normal
        match normal:
            case LinearSpringDashpotNormalPlan():
                maximum_stiffness = jnp.max(jnp.asarray(normal.stiffness))
                contact_period = jnp.pi * jnp.sqrt(0.5 * minimum_mass / maximum_stiffness)
            case _:
                contact_period = jnp.asarray(
                    jnp.inf, dtype=self.shapes.bounding_radii.dtype
                )
        if self.shapes.shape_kind != "sphere":
            # The Rayleigh estimate is a sphere formula, not a shape-bound radius law.
            rayleigh = jnp.asarray(jnp.inf, dtype=contact_period.dtype)
            return DEMStepRestriction(contact_period, rayleigh, 0.1 * contact_period)
        material = self.shapes.material_ids
        young = self.materials.young_modulus[material]
        poisson = self.materials.poisson_ratio[material]
        shear = young / (2.0 * (1.0 + poisson))
        radius = self.shapes.bounding_radii * selected_properties.shape_scale
        if self.bodies.ambient_dimension == 2:
            measure = jnp.pi * radius**2
        else:
            measure = (4.0 / 3.0) * jnp.pi * radius**3
        density = selected_properties.masses / jnp.maximum(measure, 1.0e-30)
        rayleigh_values = (
            jnp.pi * radius * jnp.sqrt(density / shear) / (0.1631 * poisson + 0.8766)
        )
        rayleigh = jnp.min(jnp.where(active, rayleigh_values, jnp.inf))
        selected = jnp.minimum(0.1 * contact_period, 0.2 * rayleigh)
        return DEMStepRestriction(contact_period, rayleigh, selected)

    def _external_load(
        self, time: Array, kinematics: RigidBodyKinematics, args: Any, /
    ) -> DEMExternalLoad:
        if self.external_load is None:
            return DEMExternalLoad(
                jnp.zeros_like(kinematics.position),
                jnp.zeros_like(kinematics.angular_velocity),
            )
        result = self.external_load(
            time,
            kinematics.position,
            kinematics.velocity,
            kinematics.angular_velocity,
            args,
        )
        if not isinstance(result, DEMExternalLoad):
            raise TypeError(
                f"PreparedCompliantDEMDynamics.external_load.time must return DEMExternalLoad; got {type(time).__name__}"
            )
        validated_load = self.bodies.load(result.force, result.torque)
        return DEMExternalLoad(validated_load.force, validated_load.torque)

    def _diagnostics(
        self,
        kinematics: RigidBodyKinematics,
        neighborhood_cache: ParticleVerletState | None,
        particle_contact: DEMContactResponse,
        geometry_batch: RigidContactGeometryBatch,
        multicontact: DEMMulticontactCorrection | None,
        boundaries: tuple[DEMBoundaryResponse, ...],
        liquid_evaluation: DEMLiquidEvaluation | None,
        body_properties: CompliantDEMBodyProperties,
        pair_force: Array,
        pair_torque: Array,
        energy: DEMEnergyLedgerState,
        successful: Array,
        rejection_reasons: Array,
        /,
    ) -> DEMDiagnostics:
        phase = self._contact_diagnostics(
            particle_contact,
            multicontact,
            neighborhood_cache,
            energy,
            liquid_evaluation,
            boundaries,
        )
        friction_defect = phase.friction_defect
        multicontact_residual = phase.multicontact_residual
        multicontact_margin = phase.multicontact_margin
        wall_defect = phase.wall_defect
        neighborhood_rebuilt = phase.neighborhood_rebuilt
        neighborhood_rebuild_count = phase.neighborhood_rebuild_count
        neighborhood_certificate_margin = phase.neighborhood_certificate_margin
        active_count = phase.active_count
        sticking_count = phase.sticking_count
        sliding_count = phase.sliding_count
        cohesion_birth_count = phase.cohesion_birth_count
        cohesion_rupture_count = phase.cohesion_rupture_count
        rolling_yield_count = phase.rolling_yield_count
        torsional_yield_count = phase.torsional_yield_count
        bridge_residual = phase.bridge_residual
        liquid_balance_residual = phase.liquid_balance_residual
        evaporated_liquid_volume = phase.evaporated_liquid_volume
        maximum_overlap = phase.maximum_overlap
        elastic_energy = phase.elastic_energy
        friction_margin = phase.friction_margin
        activation_margin = phase.activation_margin
        no_tension_margin = phase.no_tension_margin
        frame_margin = phase.frame_margin
        cohesion_birth_margin = phase.cohesion_birth_margin
        cohesion_rupture_margin = phase.cohesion_rupture_margin
        cohesion_model_margin = phase.cohesion_model_margin
        cohesion_extrapolation_margin = phase.cohesion_extrapolation_margin
        rolling_yield_margin = phase.rolling_yield_margin
        torsional_yield_margin = phase.torsional_yield_margin
        masses = body_properties.masses
        linear_momentum = jnp.sum(masses[:, None] * kinematics.velocity, axis=0)
        angular_momentum = jnp.sum(
            sphere_lever_torque(
                kinematics.position,
                masses[:, None] * kinematics.velocity,
                self.bodies.ambient_dimension,
            )
            + self._angular_momentum(kinematics, body_properties),
            axis=0,
        )
        net_force = jnp.sum(pair_force, axis=0)
        net_torque = jnp.sum(
            sphere_lever_torque(
                kinematics.position, pair_force, self.bodies.ambient_dimension
            )
            + pair_torque,
            axis=0,
        )
        return DEMDiagnostics(
            active_contacts=active_count,
            sticking_contacts=sticking_count,
            sliding_contacts=sliding_count,
            cohesion_births=cohesion_birth_count,
            cohesion_ruptures=cohesion_rupture_count,
            rolling_yield_contacts=rolling_yield_count,
            torsional_yield_contacts=torsional_yield_count,
            maximum_overlap_fraction=maximum_overlap,
            minimum_gap_margin=activation_margin,
            minimum_no_tension_margin=no_tension_margin,
            minimum_frame_transport_margin=frame_margin,
            minimum_cohesion_birth_margin=cohesion_birth_margin,
            minimum_cohesion_rupture_margin=cohesion_rupture_margin,
            minimum_rolling_yield_margin=rolling_yield_margin,
            minimum_torsional_yield_margin=torsional_yield_margin,
            acceptance_margin=(self.method.maximum_overlap_fraction - maximum_overlap),
            minimum_friction_switch_margin=friction_margin,
            neighborhood_rebuilt=neighborhood_rebuilt,
            neighborhood_rebuild_count=neighborhood_rebuild_count,
            neighborhood_certificate_margin=neighborhood_certificate_margin,
            total_linear_momentum=linear_momentum,
            total_angular_momentum=angular_momentum,
            kinetic_energy=energy.kinetic_energy,
            elastic_energy=elastic_energy,
            gravity_potential_energy=energy.gravity_potential,
            energy=energy,
            net_internal_force=net_force,
            net_internal_torque=net_torque,
            maximum_friction_cone_defect=friction_defect,
            wall_action_reaction_defect=wall_defect,
            contact_history_continuity_defect=jnp.where(
                (rejection_reasons & int(DEMRejectionReason.PAIR_KEY)) != 0,
                1.0,
                0.0,
            ),
            minimum_cohesion_model_validity_margin=cohesion_model_margin,
            minimum_cohesion_fit_extrapolation_margin=(cohesion_extrapolation_margin),
            multicontact_residual=multicontact_residual,
            minimum_multicontact_regularity_margin=multicontact_margin,
            maximum_bridge_volume_residual=bridge_residual,
            liquid_balance_residual=liquid_balance_residual,
            evaporated_liquid_volume=evaporated_liquid_volume,
            successful=successful,
            rejection_reasons=rejection_reasons,
            minimum_shape_query_margin=jnp.min(
                jnp.where(
                    geometry_batch.geometry.valid,
                    geometry_batch.geometry.certificate_margin,
                    jnp.inf,
                ),
                initial=jnp.inf,
            ),
            minimum_curvature_margin=(
                jnp.min(
                    jnp.where(
                        geometry_batch.geometry.valid,
                        jnp.where(
                            geometry_batch.geometry.curvature_valid,
                            geometry_batch.geometry.effective_radius,
                            0.0,
                        ),
                        jnp.inf,
                    ),
                    initial=jnp.inf,
                )
                if isinstance(self.method.contact.normal, HertzNormalContactPlan)
                else jnp.asarray(jnp.inf, dtype=energy.kinetic_energy.dtype)
            ),
            contact_capacity_margin=(
                jnp.asarray(jnp.inf, dtype=energy.kinetic_energy.dtype)
                if self.contact_capacity
                >= self.neighborhood.pair_capacity * self.shapes.routes_per_pair
                else geometry_batch.capacity - geometry_batch.candidate_count
            ),
        )

    def _prepare_body_state(
        self,
        state: DEMRuntimeState,
    ) -> _DEMBodyPhase:
        properties = state.body_properties
        if properties.shape_id != self.shapes.shape_id:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.prepare_body_state.state must belong to the bound prepared shape; got {type(state).__name__}"
            )
        radii = self.shapes.bounding_radii * properties.shape_scale
        inertia_shape = self.bodies.inertia_com.shape
        expected = (self.bodies.capacity,)
        body_shapes_valid = (
            properties.masses.shape == expected
            and properties.inverse_masses.shape == expected
            and properties.shape_scale.shape == expected
            and properties.inertia_com.shape == inertia_shape
            and properties.inverse_inertia_com.shape == inertia_shape
            and properties.active.shape == expected
        )
        if not body_shapes_valid:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.prepare_body_state.state must match the prepared capacity and inertia shapes; got {type(state).__name__}"
            )
        body_valid = (
            jnp.all(jnp.isfinite(properties.masses) & (properties.masses >= 0.0))
            & jnp.all(jnp.isfinite(properties.inverse_masses))
            & jnp.all(jnp.isfinite(radii) & (radii >= 0.0))
            & jnp.all(jnp.isfinite(properties.inertia_com))
            & jnp.all(jnp.isfinite(properties.inverse_inertia_com))
            & jnp.all(~properties.active | ((properties.masses > 0.0) & (radii > 0.0)))
        )
        if self.bodies.ambient_dimension == 2:
            inertia_valid = jnp.all(properties.inertia_com > 0.0)
        elif self.shapes.shape_kind == "sphere":
            diagonal = properties.inertia_com[:, 0, 0]
            inertia_valid = jnp.all(diagonal > 0.0) & jnp.all(
                properties.inertia_com
                == diagonal[:, None, None] * jnp.eye(3, dtype=diagonal.dtype)[None, :, :]
            )
        else:
            inertia_spectrum = HermitianSpectrum(
                jax.lax.stop_gradient(properties.inertia_com), tolerance=0.0
            )
            inertia_valid = jnp.all(
                inertia_spectrum.valid & (inertia_spectrum.minimum_eigenvalue > 0.0)
            )
        body_valid = body_valid & inertia_valid & jnp.all(properties.shape_scale >= 0.0)
        orientation_valid = jnp.all(jnp.isfinite(state.kinematics.orientation))
        if self.bodies.ambient_dimension == 3:
            orientation_valid = orientation_valid & jnp.all(
                jnp.abs(jnp.sum(state.kinematics.orientation**2, axis=-1) - 1.0)
                <= self.method.frame_tolerance
            )
        body_valid = body_valid & orientation_valid
        kinematics = self.bodies.kinematics(
            state.kinematics.position,
            state.kinematics.velocity,
            state.kinematics.orientation,
            state.kinematics.angular_velocity,
        )
        return _DEMBodyPhase(properties, radii, body_valid, kinematics)

    def _correct_multicontact(
        self,
        batch: DEMContactBatch,
        particle_contact: DEMContactResponse,
        kinematics: RigidBodyKinematics,
        left: Array,
        right: Array,
        radii: Array,
        remapped_history: DEMContactHistory,
        contact_context: DEMContactEvaluationContext,
    ) -> _DEMMulticontactPhase:
        multicontact = None
        if self.method.multicontact is not None and batch.gap.shape[0] > 0:
            base_batch = batch
            previous_correction = jnp.zeros_like(batch.gap)
            correction_residual = jnp.asarray(jnp.inf, dtype=batch.gap.dtype)
            for _ in range(self.method.multicontact.iterations):
                contact_point = kinematics.position[left] + base_batch.left_arm
                compressive_force = jnp.maximum(
                    jnp.sum(
                        particle_contact.normal_force * base_batch.normal,
                        axis=-1,
                    ),
                    0.0,
                )
                multicontact = self.method.multicontact.evaluate(
                    left,
                    right,
                    contact_point,
                    base_batch.normal,
                    compressive_force,
                    self.shapes.material_ids,
                    self.materials,
                    base_batch.valid & particle_contact.active,
                )
                scale = jnp.maximum(
                    jnp.minimum(
                        radii[left],
                        radii[right],
                    ),
                    1.0e-30,
                )
                correction_residual = jnp.max(
                    jnp.abs(multicontact.gap_correction - previous_correction) / scale
                )
                previous_correction = multicontact.gap_correction
                corrected_gap = base_batch.gap - multicontact.gap_correction
                batch = DEMContactBatch(
                    base_batch.normal,
                    corrected_gap,
                    jnp.maximum(-corrected_gap, 0.0),
                    base_batch.effective_radius,
                    base_batch.left_arm,
                    base_batch.right_arm,
                    base_batch.normal_velocity,
                    base_batch.tangential_velocity,
                    base_batch.left_angular_velocity,
                    base_batch.right_angular_velocity,
                    base_batch.valid,
                    normal_gradient=base_batch.normal_gradient,
                    surface_area_weight=base_batch.surface_area_weight,
                    normal_moment=base_batch.normal_moment,
                )
                particle_contact = self.contact_model.evaluate(
                    batch,
                    remapped_history,
                    contact_context,
                    frame_tolerance=self.method.frame_tolerance,
                )
            if multicontact is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.correct_multicontact.batch must provide the prepared transaction state; got {multicontact!r}"
                )
            correction_successful = multicontact.successful & (
                correction_residual <= self.method.multicontact.convergence_tolerance
            )
            multicontact = eqx.tree_at(
                lambda value: value.residual,
                multicontact,
                correction_residual,
            )
            multicontact = eqx.tree_at(
                lambda value: value.successful,
                multicontact,
                correction_successful,
            )
            particle_contact = eqx.tree_at(
                lambda value: value.successful,
                particle_contact,
                particle_contact.successful & correction_successful,
            )
        return _DEMMulticontactPhase(batch, particle_contact, multicontact)

    def _prepare_contact_routes(
        self,
        state: DEMRuntimeState,
        properties: CompliantDEMBodyProperties,
        radii: Array,
        kinematics: RigidBodyKinematics,
    ) -> _DEMRoutePhase:
        match self.neighborhood:
            case PreparedVerletParticleNeighborhood():
                if state.neighborhood_cache is None:
                    raise ValueError(
                        f"PreparedCompliantDEMDynamics.prepare_contact_routes.state must provide a neighborhood cache; got {state.neighborhood_cache!r}"
                    )
                neighborhood_cache = self.neighborhood.update(
                    kinematics.position,
                    state.neighborhood_cache,
                    active_mask=properties.active,
                    orientation=None
                    if self.shapes.shape_kind == "sphere"
                    else kinematics.orientation,
                    radii=None if self.shapes.shape_kind == "sphere" else radii,
                )
                neighborhood = neighborhood_cache.neighborhood
                rebuilt = neighborhood_cache.rebuilt
            case _:
                neighborhood_cache = None
                neighborhood = self.neighborhood.build(
                    kinematics.position, active_mask=properties.active
                )
                rebuilt = jnp.asarray(True)
        pairs = neighborhood.pair_relation
        center_geometry = particle_pair_geometry(
            kinematics.position,
            pairs,
            box=neighborhood.box,
            cell_vectors=None
            if state.periodic_cell is None
            else state.periodic_cell.vectors,
        )
        geometry_batch = self.shapes.pair_geometry(
            kinematics,
            properties.shape_scale,
            pairs,
            self.pair_key_space,
            maximum_contact_routes=self.contact_capacity,
            center_geometry=center_geometry,
            distance_tolerance=self.method.distance_tolerance,
        )
        geometry = geometry_batch.geometry
        keys = geometry_batch.keys

        def align_rebuilt(_: None) -> tuple[DEMContactHistory, Array, Array]:
            remap = match_particle_pair_keys(
                state.particle_history.pair_keys,
                state.particle_history.valid,
                keys.keys,
                keys.valid,
            )
            history = remap_dem_contact_history(
                state.particle_history,
                remap,
                keys.keys,
                keys.valid,
            )
            return history, remap.continued, remap.successful

        def align_reused(_: None) -> tuple[DEMContactHistory, Array, Array]:
            same_identity = jnp.all(
                state.particle_history.pair_keys == keys.keys, axis=-1
            )
            same_keys = jnp.all(
                ~state.particle_history.valid | (keys.valid & same_identity)
            )
            history = state.particle_history.with_routes(keys.keys, keys.valid)
            return history, keys.valid & history.valid, same_keys

        remapped_history, continued, alignment_successful = jax.lax.cond(
            rebuilt | jnp.asarray(self.shapes.shape_kind == "level_set"),
            align_rebuilt,
            align_reused,
            operand=None,
        )
        left = geometry.left_owner_index
        right = geometry.right_owner_index
        batch = DEMContactBatch(
            geometry.normal,
            geometry.gap,
            geometry.overlap,
            geometry.effective_radius,
            geometry.left_contact_arm,
            geometry.right_contact_arm,
            geometry.normal_velocity,
            geometry.tangential_velocity,
            kinematics.angular_velocity[left],
            kinematics.angular_velocity[right],
            geometry.valid & properties.active[left] & properties.active[right],
            normal_gradient=geometry.normal_gradient,
            surface_area_weight=geometry.surface_area_weight,
            measure_kind="surface_area"
            if self.shapes.shape_kind == "level_set"
            else "point_force",
            normal_moment=self.shapes.shape_kind != "sphere",
        )
        return _DEMRoutePhase(
            neighborhood_cache,
            neighborhood,
            geometry_batch,
            remapped_history,
            continued,
            alignment_successful,
            left,
            right,
            batch,
        )

    def _allocate_pair_liquid(
        self,
        state: DEMRuntimeState,
        batch: DEMContactBatch,
        left: Array,
        right: Array,
        remapped_history: DEMContactHistory,
    ) -> _DEMPairLiquidPhase:
        liquid_allocation = None
        minimum_bridge_volume = jnp.zeros_like(batch.gap)
        if self.method.liquid_process is not None:
            if state.liquid is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.allocate_pair_liquid.state must provide liquid inventory; got {state.liquid!r}"
                )
            bridge_plan, component_index = conserved_bagheri_component(
                self.method.contact.cohesion
            )
            component = remapped_history.cohesion.components[component_index]
            requested_volume = bridge_plan.pair_bridge_volume(
                self.shapes.material_ids[left],
                self.shapes.material_ids[right],
                self.materials.material_count,
            ).astype(batch.gap.dtype)
            characteristic_radius = 2.0 * batch.effective_radius
            minimum_bridge_volume = 1.0e-6 * characteristic_radius**3
            birth_candidates = (
                batch.valid
                & (batch.gap <= 0.0)
                & ~component.active
                & (requested_volume > 0.0)
            )
            liquid_allocation = self.method.liquid_process.allocate(
                state.liquid,
                left,
                right,
                requested_volume,
                minimum_bridge_volume,
                birth_candidates,
                self.bodies.capacity,
            )
            seeded_component = eqx.tree_at(
                lambda value: value.bridge_volume,
                component,
                jnp.where(
                    birth_candidates,
                    liquid_allocation.bridge_volume,
                    component.bridge_volume,
                ),
            )
            components = list(remapped_history.cohesion.components)
            components[component_index] = seeded_component
            remapped_history = eqx.tree_at(
                lambda value: value.cohesion,
                remapped_history,
                DEMCohesionHistory(tuple(components)),
            )
        return _DEMPairLiquidPhase(
            liquid_allocation, minimum_bridge_volume, remapped_history
        )

    def _allocate_boundary_liquid(
        self,
        state: DEMRuntimeState,
        kinematics: RigidBodyKinematics,
        time: Array,
        step_size: Array,
        args: Any,
        liquid_allocation: DEMLiquidAllocation | None,
        batch: DEMContactBatch,
    ) -> _DEMBoundaryLiquidPhase:
        def evaluate_boundaries(
            histories: tuple[DEMContactHistory, ...],
        ) -> tuple[DEMBoundaryResponse, ...]:
            capillary_plans = (
                (None,) * len(self.barriers)
                if self.method.liquid_process is None
                else self.method.liquid_process.barrier_capillaries
            )
            if len(histories) != len(self.barriers) + len(self.walls):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.evaluate_boundaries.histories must match prepared boundaries; got {type(histories).__name__}"
                )
            shape = self.shapes
            barrier_responses: tuple[DEMBoundaryResponse, ...] = ()
            if self.barriers:
                if not isinstance(shape, PreparedRigidSphereSet):
                    raise RuntimeError(
                        f"PreparedCompliantDEMDynamics.evaluate_boundaries.histories must provide sphere shapes; got {type(histories).__name__}"
                    )
                barrier_responses = tuple(
                    evaluate_dem_barrier(
                        barrier,
                        shape,
                        kinematics,
                        self.contact_model,
                        history,
                        step_size,
                        time=time,
                        args=args,
                        body_properties=state.body_properties,
                        normal_tolerance=self.method.distance_tolerance,
                        capillary_plan=capillary_plan,
                        frame_tolerance=self.method.frame_tolerance,
                    )
                    for barrier, history, capillary_plan in zip(
                        self.barriers,
                        histories[: len(self.barriers)],
                        capillary_plans,
                        strict=True,
                    )
                )
            wall_responses = tuple(
                evaluate_dem_triangle_wall(
                    shape,
                    self.bodies,
                    kinematics,
                    wall,
                    self.contact_model,
                    history,
                    state.body_properties,
                    step_size,
                    self.method.wall_geometry,
                    frame_tolerance=self.method.frame_tolerance,
                )
                for wall, history in zip(
                    self.walls, histories[len(self.barriers) :], strict=True
                )
            )
            return barrier_responses + wall_responses

        boundary_responses = evaluate_boundaries(state.boundary_histories)
        barrier_allocation = None
        barrier_particles = None
        barrier_indices = None
        barrier_minimum_volume = None
        if self.method.liquid_process is not None and self.barriers:
            if state.liquid is None or liquid_allocation is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.allocate_boundary_liquid.liquid_allocation must be initialized before advancing; got {state.liquid!r}"
                )
            bridge_plan, component_index = conserved_bagheri_component(
                self.method.contact.cohesion
            )
            barrier_particles = jnp.tile(
                jnp.arange(self.bodies.capacity, dtype=jnp.int32),
                len(self.barriers),
            )
            barrier_indices = jnp.repeat(
                jnp.arange(len(self.barriers), dtype=jnp.int32),
                self.bodies.capacity,
            )
            requested_barrier_volume = []
            barrier_birth_candidates = []
            for barrier, history, response in zip(
                self.barriers,
                state.boundary_histories,
                boundary_responses,
                strict=True,
            ):
                previous_component = history.cohesion.components[component_index]
                evaluated_component = response.contact.next_history.cohesion.components[
                    component_index
                ]
                requested_barrier_volume.append(
                    bridge_plan.pair_bridge_volume(
                        self.shapes.material_ids,
                        jnp.full(
                            (self.bodies.capacity,),
                            barrier.material_id,
                            dtype=jnp.int32,
                        ),
                        self.materials.material_count,
                    ).astype(batch.gap.dtype)
                )
                barrier_birth_candidates.append(
                    state.body_properties.active
                    & response.contact.active
                    & (evaluated_component.previous_gap <= 0.0)
                    & ~previous_component.active
                )
            requested_barrier_volume = jnp.concatenate(tuple(requested_barrier_volume))
            barrier_birth_candidates = jnp.concatenate(tuple(barrier_birth_candidates))
            barrier_minimum_volume = jnp.zeros_like(requested_barrier_volume)
            allocation_state = DEMLiquidState(
                liquid_allocation.film_volume,
                state.liquid.barrier_reservoir_volume,
                state.liquid.cumulative_evaporated_volume,
                state.liquid.initial_total_volume,
                state.liquid.balance_residual,
                state.liquid.successful & liquid_allocation.successful,
            )
            barrier_allocation = self.method.liquid_process.allocate_barriers(
                allocation_state,
                barrier_particles,
                barrier_indices,
                requested_barrier_volume,
                barrier_minimum_volume,
                barrier_birth_candidates,
                self.bodies.capacity,
            )
            seeded_histories = []
            for barrier_index, history in enumerate(state.boundary_histories):
                start = barrier_index * self.bodies.capacity
                stop = start + self.bodies.capacity
                component = history.cohesion.components[component_index]
                seeded_component = eqx.tree_at(
                    lambda value: value.bridge_volume,
                    component,
                    jnp.where(
                        barrier_birth_candidates[start:stop],
                        barrier_allocation.bridge_volume[start:stop],
                        component.bridge_volume,
                    ),
                )
                components = list(history.cohesion.components)
                components[component_index] = seeded_component
                seeded_histories.append(
                    eqx.tree_at(
                        lambda value: value.cohesion,
                        history,
                        DEMCohesionHistory(tuple(components)),
                    )
                )
            boundary_responses = evaluate_boundaries(tuple(seeded_histories))
        return _DEMBoundaryLiquidPhase(
            boundary_responses,
            barrier_allocation,
            barrier_particles,
            barrier_indices,
            barrier_minimum_volume,
        )

    def _advance_liquid(
        self,
        state: DEMRuntimeState,
        liquid_allocation: DEMLiquidAllocation | None,
        batch: DEMContactBatch,
        particle_contact: DEMContactResponse,
        left: Array,
        right: Array,
        minimum_bridge_volume: Array,
        step_size: Array,
        boundary_responses: tuple[DEMBoundaryResponse, ...],
        barrier_allocation: DEMBarrierLiquidAllocation | None,
        barrier_particles: Array | None,
        barrier_indices: Array | None,
        barrier_minimum_volume: Array | None,
    ) -> _DEMLiquidAdvancePhase:
        liquid_evaluation = None
        if self.method.liquid_process is not None:
            if state.liquid is None or liquid_allocation is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.advance_liquid.liquid_allocation must be initialized before advancing; got {state.liquid!r}"
                )
            component = particle_contact.next_history.cohesion.components[
                self.liquid_component_index
            ]
            boundary_bridge_pending = jnp.zeros((), dtype=batch.gap.dtype)
            pair_liquid_state = state.liquid
            if barrier_allocation is not None:
                liquid_allocation = eqx.tree_at(
                    lambda value: value.film_volume,
                    liquid_allocation,
                    barrier_allocation.film_volume,
                )
                pair_liquid_state = DEMLiquidState(
                    state.liquid.film_volume,
                    barrier_allocation.barrier_reservoir_volume,
                    state.liquid.cumulative_evaporated_volume,
                    state.liquid.initial_total_volume,
                    state.liquid.balance_residual,
                    state.liquid.successful & barrier_allocation.successful,
                )
                for response in boundary_responses:
                    boundary_component = (
                        response.contact.next_history.cohesion.components[
                            self.liquid_component_index
                        ]
                    )
                    boundary_bridge_pending = (
                        boundary_bridge_pending
                        + jnp.sum(boundary_component.bridge_volume)
                        + jnp.sum(response.contact.bridge_volume_release)
                    )
            next_component, liquid_evaluation = self.method.liquid_process.advance(
                pair_liquid_state,
                liquid_allocation,
                component,
                left,
                right,
                particle_contact.bridge_volume_release,
                particle_contact.bridge_surface_area,
                minimum_bridge_volume,
                step_size,
                self.bodies.capacity,
                additional_bridge_volume=boundary_bridge_pending,
            )
            components = list(particle_contact.next_history.cohesion.components)
            components[self.liquid_component_index] = next_component
            next_cohesion = DEMCohesionHistory(tuple(components))
            next_active = particle_contact.next_history.active & (
                ~liquid_evaluation.evaporated_ruptures | (batch.gap <= 0.0)
            )
            next_history = eqx.tree_at(
                lambda value: (value.active, value.cohesion),
                particle_contact.next_history,
                (next_active, next_cohesion),
            )
            particle_contact = eqx.tree_at(
                lambda value: (
                    value.next_history,
                    value.cohesion_ruptures,
                    value.bridge_evaporation_loss,
                    value.successful,
                ),
                particle_contact,
                (
                    next_history,
                    particle_contact.cohesion_ruptures
                    | liquid_evaluation.evaporated_ruptures,
                    liquid_evaluation.evaporated_bridge_volume,
                    particle_contact.successful & liquid_evaluation.successful,
                ),
            )
            phase = self._advance_boundary_liquid(
                state,
                liquid_evaluation,
                next_component.bridge_volume,
                boundary_responses,
                barrier_allocation,
                barrier_particles,
                barrier_indices,
                barrier_minimum_volume,
                step_size,
            )
            boundary_responses = phase.boundary_responses
            liquid_evaluation = phase.liquid_evaluation
        return _DEMLiquidAdvancePhase(
            particle_contact, boundary_responses, liquid_evaluation
        )

    def _resolve_loads(
        self,
        geometry: RigidContactGeometry,
        particle_contact: DEMContactResponse,
        boundary_responses: tuple[DEMBoundaryResponse, ...],
        properties: CompliantDEMBodyProperties,
        time: Array,
        kinematics: RigidBodyKinematics,
        args: Any,
    ) -> _DEMLoadPhase:
        pair_load = reduce_rigid_contact(
            geometry,
            particle_contact,
            particle_capacity=self.bodies.capacity,
            ambient_dimension=self.bodies.ambient_dimension,
            angular_dimension=self.bodies.angular_dimension,
            execution=self.execution,
            precision=self.precision,
        )
        pair_force = pair_load.force
        pair_torque = pair_load.torque
        particle_load = RigidBodyLoad(
            self.precision.output(pair_force), self.precision.output(pair_torque)
        )
        boundary_loads = tuple(
            RigidBodyLoad(response.particle_force, response.particle_torque)
            for response in boundary_responses
        )
        boundary_force = jnp.zeros_like(pair_force)
        boundary_torque = jnp.zeros_like(pair_torque)
        for load in boundary_loads:
            boundary_force = boundary_force + load.force
            boundary_torque = boundary_torque + load.torque
        gravity_force = properties.masses[:, None] * self.gravity[None, :]
        gravity_force = jnp.where(properties.active[:, None], gravity_force, 0.0)
        gravity_load = RigidBodyLoad(gravity_force, jnp.zeros_like(pair_torque))
        external_value = self._external_load(time, kinematics, args)
        external_load = RigidBodyLoad(
            jnp.where(properties.active[:, None], external_value.force, 0.0),
            jnp.where(properties.active[:, None], external_value.torque, 0.0),
        )
        total_load = RigidBodyLoad(
            self.precision.output(
                pair_force + boundary_force + gravity_load.force + external_load.force
            ),
            self.precision.output(
                pair_torque + boundary_torque + gravity_load.torque + external_load.torque
            ),
        )
        resolved_loads = DEMResolvedLoad(
            particle_load,
            boundary_loads,
            gravity_load,
            external_load,
            total_load,
        )
        return _DEMLoadPhase(pair_force, pair_torque, resolved_loads)

    def _contact_events(
        self,
        state: DEMRuntimeState,
        remapped_history: DEMContactHistory,
        particle_contact: DEMContactResponse,
        boundary_responses: tuple[DEMBoundaryResponse, ...],
    ) -> _DEMEventPhase:
        continued_active = remapped_history.active & particle_contact.active
        births = jnp.sum(
            particle_contact.active & ~remapped_history.active, dtype=jnp.int32
        )
        deaths = jnp.sum(state.particle_history.active, dtype=jnp.int32) - jnp.sum(
            continued_active, dtype=jnp.int32
        )
        stick_to_slip = jnp.sum(
            continued_active
            & ~remapped_history.tangential.sliding
            & particle_contact.sliding,
            dtype=jnp.int32,
        )
        slip_to_stick = jnp.sum(
            continued_active
            & remapped_history.tangential.sliding
            & ~particle_contact.sliding,
            dtype=jnp.int32,
        )
        cohesion_births = jnp.sum(particle_contact.cohesion_births, dtype=jnp.int32)
        cohesion_ruptures = jnp.sum(particle_contact.cohesion_ruptures, dtype=jnp.int32)
        rolling_yields = jnp.sum(particle_contact.rolling_yielded, dtype=jnp.int32)
        torsional_yields = jnp.sum(particle_contact.torsional_yielded, dtype=jnp.int32)
        contact_energy = jnp.sum(particle_contact.elastic_energy)
        boundary_wall_power = (
            jnp.stack(tuple(response.wall_power for response in boundary_responses))
            if boundary_responses
            else jnp.zeros((0,), dtype=contact_energy.dtype)
        )
        for old_history, response in zip(
            state.boundary_histories, boundary_responses, strict=True
        ):
            new_history = response.contact.next_history
            continued_boundary = old_history.active & new_history.active
            births = births + jnp.sum(
                new_history.active & ~old_history.active, dtype=jnp.int32
            )
            deaths = deaths + jnp.sum(
                old_history.active & ~new_history.active, dtype=jnp.int32
            )
            stick_to_slip = stick_to_slip + jnp.sum(
                continued_boundary
                & ~old_history.tangential.sliding
                & new_history.tangential.sliding,
                dtype=jnp.int32,
            )
            slip_to_stick = slip_to_stick + jnp.sum(
                continued_boundary
                & old_history.tangential.sliding
                & ~new_history.tangential.sliding,
                dtype=jnp.int32,
            )
            cohesion_births = cohesion_births + jnp.sum(
                response.contact.cohesion_births, dtype=jnp.int32
            )
            cohesion_ruptures = cohesion_ruptures + jnp.sum(
                response.contact.cohesion_ruptures, dtype=jnp.int32
            )
            rolling_yields = rolling_yields + jnp.sum(
                response.contact.rolling_yielded, dtype=jnp.int32
            )
            torsional_yields = torsional_yields + jnp.sum(
                response.contact.torsional_yielded, dtype=jnp.int32
            )
            contact_energy = contact_energy + jnp.sum(response.contact.elastic_energy)
        return _DEMEventPhase(
            births,
            deaths,
            stick_to_slip,
            slip_to_stick,
            cohesion_births,
            cohesion_ruptures,
            rolling_yields,
            torsional_yields,
            contact_energy,
            boundary_wall_power,
        )

    def _contact_diagnostics(
        self,
        particle_contact: DEMContactResponse,
        multicontact: DEMMulticontactCorrection | None,
        neighborhood_cache: ParticleVerletState | None,
        energy: DEMEnergyLedgerState,
        liquid_evaluation: DEMLiquidEvaluation | None,
        boundaries: tuple[DEMBoundaryResponse, ...],
    ) -> _DEMContactDiagnostics:
        friction_defect = jnp.max(
            jnp.concatenate(
                (
                    particle_contact.friction_defect,
                    jnp.zeros((1,), dtype=energy.kinetic_energy.dtype),
                )
            )
        )
        multicontact_residual = (
            jnp.zeros((), dtype=energy.kinetic_energy.dtype)
            if multicontact is None
            else multicontact.residual
        )
        multicontact_margin = (
            jnp.asarray(jnp.inf, dtype=energy.kinetic_energy.dtype)
            if multicontact is None
            else multicontact.regularity_margin
        )
        wall_defect = jnp.zeros((), dtype=energy.kinetic_energy.dtype)
        neighborhood_rebuilt = (
            neighborhood_cache.rebuilt
            if neighborhood_cache is not None
            else jnp.asarray(True)
        )
        neighborhood_rebuild_count = (
            neighborhood_cache.rebuild_count
            if neighborhood_cache is not None
            else jnp.zeros((), dtype=jnp.int32)
        )
        neighborhood_certificate_margin = (
            neighborhood_cache.certificate_margin
            if neighborhood_cache is not None
            else jnp.asarray(jnp.inf, dtype=energy.kinetic_energy.dtype)
        )
        active_count = jnp.sum(particle_contact.active, dtype=jnp.int32)
        sticking_count = jnp.sum(particle_contact.sticking, dtype=jnp.int32)
        sliding_count = jnp.sum(particle_contact.sliding, dtype=jnp.int32)
        cohesion_birth_count = jnp.sum(particle_contact.cohesion_births, dtype=jnp.int32)
        cohesion_rupture_count = jnp.sum(
            particle_contact.cohesion_ruptures, dtype=jnp.int32
        )
        rolling_yield_count = jnp.sum(particle_contact.rolling_yielded, dtype=jnp.int32)
        torsional_yield_count = jnp.sum(
            particle_contact.torsional_yielded, dtype=jnp.int32
        )
        bridge_residual = jnp.max(
            jnp.concatenate(
                (
                    jnp.abs(particle_contact.bridge_volume_residual),
                    jnp.zeros((1,), dtype=energy.kinetic_energy.dtype),
                )
            )
        )
        liquid_balance_residual = (
            jnp.zeros((), dtype=energy.kinetic_energy.dtype)
            if liquid_evaluation is None
            else liquid_evaluation.next_state.balance_residual
        )
        evaporated_liquid_volume = (
            jnp.zeros((), dtype=energy.kinetic_energy.dtype)
            if liquid_evaluation is None
            else liquid_evaluation.next_state.cumulative_evaporated_volume
        )
        maximum_overlap = particle_contact.maximum_overlap_fraction
        elastic_energy = jnp.sum(particle_contact.elastic_energy)
        friction_margin = jnp.min(
            jnp.concatenate(
                (
                    particle_contact.switch_margin,
                    jnp.full((1,), jnp.inf, dtype=energy.kinetic_energy.dtype),
                )
            )
        )
        activation_margin = particle_contact.activation_margin
        no_tension_margin = particle_contact.no_tension_margin
        frame_margin = particle_contact.frame_transport_margin
        cohesion_birth_margin = particle_contact.cohesion_birth_margin
        cohesion_rupture_margin = particle_contact.cohesion_rupture_margin
        cohesion_model_margin = particle_contact.cohesion_model_validity_margin
        cohesion_extrapolation_margin = particle_contact.cohesion_fit_extrapolation_margin
        rolling_yield_margin = particle_contact.rolling_yield_margin
        torsional_yield_margin = particle_contact.torsional_yield_margin
        for response in boundaries:
            active_count = active_count + jnp.sum(
                response.contact.active, dtype=jnp.int32
            )
            sticking_count = sticking_count + jnp.sum(
                response.contact.sticking, dtype=jnp.int32
            )
            sliding_count = sliding_count + jnp.sum(
                response.contact.sliding, dtype=jnp.int32
            )
            cohesion_birth_count = cohesion_birth_count + jnp.sum(
                response.contact.cohesion_births, dtype=jnp.int32
            )
            cohesion_rupture_count = cohesion_rupture_count + jnp.sum(
                response.contact.cohesion_ruptures, dtype=jnp.int32
            )
            rolling_yield_count = rolling_yield_count + jnp.sum(
                response.contact.rolling_yielded, dtype=jnp.int32
            )
            torsional_yield_count = torsional_yield_count + jnp.sum(
                response.contact.torsional_yielded, dtype=jnp.int32
            )
            bridge_residual = jnp.maximum(
                bridge_residual,
                jnp.max(jnp.abs(response.contact.bridge_volume_residual)),
            )
            maximum_overlap = jnp.maximum(
                maximum_overlap, response.contact.maximum_overlap_fraction
            )
            elastic_energy = elastic_energy + jnp.sum(response.contact.elastic_energy)
            friction_defect = jnp.maximum(
                friction_defect, jnp.max(response.contact.friction_defect)
            )
            wall_defect = jnp.maximum(
                wall_defect,
                jnp.linalg.norm(
                    response.reaction_force + jnp.sum(response.particle_force, axis=0)
                ),
            )
            friction_margin = jnp.minimum(
                friction_margin, jnp.min(response.contact.switch_margin)
            )
            activation_margin = jnp.minimum(
                activation_margin, response.contact.activation_margin
            )
            no_tension_margin = jnp.minimum(
                no_tension_margin, response.contact.no_tension_margin
            )
            frame_margin = jnp.minimum(
                frame_margin, response.contact.frame_transport_margin
            )
            cohesion_birth_margin = jnp.minimum(
                cohesion_birth_margin, response.contact.cohesion_birth_margin
            )
            cohesion_rupture_margin = jnp.minimum(
                cohesion_rupture_margin, response.contact.cohesion_rupture_margin
            )
            rolling_yield_margin = jnp.minimum(
                rolling_yield_margin, response.contact.rolling_yield_margin
            )
            torsional_yield_margin = jnp.minimum(
                torsional_yield_margin, response.contact.torsional_yield_margin
            )
            cohesion_model_margin = jnp.minimum(
                cohesion_model_margin,
                response.contact.cohesion_model_validity_margin,
            )
            cohesion_extrapolation_margin = jnp.minimum(
                cohesion_extrapolation_margin,
                response.contact.cohesion_fit_extrapolation_margin,
            )
        return _DEMContactDiagnostics(
            friction_defect,
            multicontact_residual,
            multicontact_margin,
            wall_defect,
            neighborhood_rebuilt,
            neighborhood_rebuild_count,
            neighborhood_certificate_margin,
            active_count,
            sticking_count,
            sliding_count,
            cohesion_birth_count,
            cohesion_rupture_count,
            rolling_yield_count,
            torsional_yield_count,
            bridge_residual,
            liquid_balance_residual,
            evaporated_liquid_volume,
            maximum_overlap,
            elastic_energy,
            friction_margin,
            activation_margin,
            no_tension_margin,
            frame_margin,
            cohesion_birth_margin,
            cohesion_rupture_margin,
            cohesion_model_margin,
            cohesion_extrapolation_margin,
            rolling_yield_margin,
            torsional_yield_margin,
        )

    def _evaluation_status(
        self,
        body_valid: Array,
        neighborhood: ParticleNeighborhoodState,
        keys: ParticlePairKeys,
        alignment_successful: Array,
        geometry_batch: RigidContactGeometryBatch,
        batch: DEMContactBatch,
        geometry: RigidContactGeometry,
        particle_contact: DEMContactResponse,
        boundary_responses: tuple[DEMBoundaryResponse, ...],
        liquid_evaluation: DEMLiquidEvaluation | None,
        resolved_loads: DEMResolvedLoad,
        state: DEMRuntimeState,
        contact_energy: Array,
    ) -> _DEMStatusPhase:
        boundary_successful = (
            jnp.all(jnp.stack(tuple(value.successful for value in boundary_responses)))
            if boundary_responses
            else jnp.asarray(True)
        )
        reasons = jnp.zeros((), dtype=jnp.int32)
        reasons = reasons | jnp.where(
            ~body_valid,
            int(DEMRejectionReason.BODY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.cell_overflow,
            int(DEMRejectionReason.CELL_CAPACITY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.pair_overflow,
            int(DEMRejectionReason.PAIR_CAPACITY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            neighborhood.domain_violation,
            int(DEMRejectionReason.DOMAIN),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~(keys.successful & alignment_successful),
            int(DEMRejectionReason.PAIR_KEY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~geometry_batch.successful,
            int(DEMRejectionReason.GEOMETRY | DEMRejectionReason.SHAPE_QUERY),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            geometry_batch.overflow, int(DEMRejectionReason.CONTACT_CAPACITY), 0
        ).astype(jnp.int32)
        if isinstance(self.method.contact.normal, HertzNormalContactPlan):
            reasons = reasons | jnp.where(
                jnp.any(batch.valid & ~geometry.curvature_valid),
                int(DEMRejectionReason.CURVATURE),
                0,
            ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~(particle_contact.successful & boundary_successful),
            int(DEMRejectionReason.CONTACT),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            (
                jnp.asarray(False)
                if liquid_evaluation is None
                else ~liquid_evaluation.successful
            ),
            int(DEMRejectionReason.LIQUID),
            0,
        ).astype(jnp.int32)
        reasons = reasons | jnp.where(
            ~(
                tree_allfinite(resolved_loads)
                & tree_allfinite(state.kinematics)
                & jnp.isfinite(contact_energy)
            ),
            int(DEMRejectionReason.NONFINITE),
            0,
        ).astype(jnp.int32)
        successful = reasons == 0
        return _DEMStatusPhase(reasons, successful)

    def _admit_contact_bindings(
        self,
        shapes: AbstractPreparedRigidContactShape,
        neighborhood: AbstractPreparedParticleNeighborhood,
        pair_key_space: ParticlePairKeySpace,
        contact_model: PreparedDEMContactModel,
        method: CompliantDEMMethodPlan,
        materials: DEMMaterialTable,
        walls: Sequence[PreparedTriangleWall],
        barriers: Sequence[ImplicitDEMBarrier],
    ) -> _DEMBindingPhase:
        bodies = shapes.bodies
        walls_ = tuple(walls)
        if any(not isinstance(wall, PreparedTriangleWall) for wall in walls_):
            raise TypeError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.walls must contain PreparedTriangleWall values; got {type(walls).__name__}"
            )
        if (
            walls_
            and shapes.shape_kind != "sphere"
            and shapes.shape_kind != "superquadric"
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must use sphere or superquadric shapes for triangle walls; got {type(shapes).__name__}"
            )
        if walls_ and method.liquid_process is not None:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.walls must use implicit barriers instead of triangle walls for conserved liquid; got {type(walls).__name__}"
            )
        if shapes.shape_kind != "sphere" and (
            method.multicontact is not None
            or method.periodic_cell_control is not None
            or method.liquid_process is not None
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must provide sphere shapes; got {type(shapes).__name__}"
            )
        if (
            shapes.shape_kind != "sphere"
            and neighborhood.box is not None
            and any(neighborhood.box.periodic_axes)
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must provide sphere shapes; got {type(shapes).__name__}"
            )
        if (shapes.shape_kind == "level_set") != isinstance(
            method.contact.normal, LevelSetPenaltyNormalPlan
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must use per-area penalty laws for level-set shapes and point-force laws for other shapes; got {type(shapes).__name__}"
            )
        if shapes.shape_kind == "convex" and isinstance(
            method.contact.normal, HertzNormalContactPlan
        ):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must provide Hertz contact curvature when using a Hertz law; got {type(shapes).__name__}"
            )
        if shapes.shape_kind != "sphere" and barriers:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must use sphere shapes for implicit barriers; got {type(shapes).__name__}"
            )
        if neighborhood.particle_discretization_id != bodies.particles.prepared_id:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.neighborhood must be prepared for the bound particle support; got {type(neighborhood).__name__}"
            )
        if pair_key_space.particle_discretization_id != bodies.particles.prepared_id:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.pair_key_space must be prepared for the bound support; got {type(pair_key_space).__name__}"
            )
        if contact_model.plan.contact_model_id != method.contact.contact_model_id:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.method must match the DEM method plan; got {type(method).__name__}"
            )
        barriers_ = tuple(barriers)
        if any(not isinstance(value, ImplicitDEMBarrier) for value in barriers_):
            raise TypeError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.barriers must contain ImplicitDEMBarrier values; got {type(barriers).__name__}"
            )
        barrier_ids = tuple(value.barrier_id for value in barriers_)
        if len(set(barrier_ids)) != len(barrier_ids):
            raise ValueError(
                f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must be unique; got {type(shapes).__name__}"
            )
        for barrier in barriers_:
            if barrier.geometry.ambient_dimension != bodies.ambient_dimension:
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.admit_contact_bindings.shapes must match rigid spheres; got {type(shapes).__name__}"
                )
            if isinstance(method.contact.normal, HertzNormalContactPlan) and (
                "contact_curvature"
                not in {value.value for value in barrier.geometry.capabilities}
            ):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.admit_contact_bindings.method must provide contact-curvature capability; got {type(method).__name__}"
                )
            if barrier.material_id >= materials.material_count:
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.admit_contact_bindings.materials must be within the supported range; got {type(materials).__name__}"
                )
        return _DEMBindingPhase(bodies, walls_, barriers_, barrier_ids)

    def _prepare_process_bindings(
        self,
        bodies: PreparedRigidBodySet,
        shapes: AbstractPreparedRigidContactShape,
        method: CompliantDEMMethodPlan,
        materials: DEMMaterialTable,
        barriers_: tuple[ImplicitDEMBarrier, ...],
        barrier_ids: tuple[str, ...],
        neighborhood: AbstractPreparedParticleNeighborhood,
        gravity_: Array,
    ) -> _DEMProcessPhase:
        active_mask = np.asarray(bodies.particles.active_mask, dtype=np.bool_)
        interaction_extents = method.contact.interaction_extents_for_radii(
            np.asarray(shapes.bounding_radii)[active_mask],
            np.asarray(shapes.material_ids)[active_mask],
            materials.material_count,
        )
        if shapes.shape_kind == "clump":
            maximum_bound = float(np.max(np.asarray(shapes.bounding_radii)[active_mask]))
            contact_materials = np.asarray(shapes.contact_material_ids)
            material_extents = method.contact.interaction_extents_for_radii(
                np.full(contact_materials.shape, maximum_bound, dtype=np.float64),
                contact_materials,
                materials.material_count,
            )
            maximum_extension = float(np.max(material_extents) - maximum_bound)
            interaction_extents = (
                np.asarray(shapes.bounding_radii)[active_mask] + maximum_extension
            )
        maximum_interaction_radius = 2.0 * float(np.max(interaction_extents))
        cohesion = method.contact.cohesion
        bagheri_present = isinstance(cohesion, BagheriCapillaryBridgePlan) or (
            isinstance(cohesion, CompositeDEMCohesionPlan)
            and any(
                isinstance(component, BagheriCapillaryBridgePlan)
                for component in cohesion.components
            )
        )
        # Wall bridges need an explicit sphere/surface capillary binding, which
        # only the conserved liquid process supplies (validated below).
        if bagheri_present and barriers_ and method.liquid_process is None:
            raise ValueError(
                f"PreparedCompliantDEMDynamics.method.contact.cohesion must use sphere pairs or bind implicit barriers through a conserved liquid process; got {method.liquid_process!r}"
            )
        periodic_cell = self._prepare_periodic_binding(
            method,
            neighborhood,
            bodies,
            barriers_,
            active_mask,
            gravity_,
            maximum_interaction_radius,
        )
        liquid_component_index = -1
        if method.liquid_process is not None:
            if barriers_:
                bindings = method.liquid_process.barrier_capillaries
                if tuple(value.barrier_id for value in bindings) != barrier_ids:
                    raise ValueError(
                        f"PreparedCompliantDEMDynamics.prepare_process_bindings.barrier_ids must bind every DEM barrier exactly once and in prepared barrier order; got {type(barrier_ids).__name__}"
                    )
                for barrier, binding in zip(barriers_, bindings, strict=True):
                    if binding.law != "bagheri":
                        raise ValueError(
                            f"PreparedCompliantDEMDynamics.prepare_process_bindings.bodies must provide the Bagheri sphere-surface law; got {type(bodies).__name__}"
                        )
                    if (
                        binding.geometry_policy == "isotropic_curvature"
                        and "contact_curvature"
                        not in {value.value for value in barrier.geometry.capabilities}
                    ):
                        raise ValueError(
                            f"PreparedCompliantDEMDynamics.prepare_process_bindings.bodies must provide certified contact curvature; got {type(bodies).__name__}"
                        )
            _, liquid_component_index = conserved_bagheri_component(
                method.contact.cohesion
            )
        return _DEMProcessPhase(
            maximum_interaction_radius, periodic_cell, liquid_component_index
        )

    def _advance_boundary_liquid(
        self,
        state: DEMRuntimeState,
        liquid_evaluation: DEMLiquidEvaluation,
        pair_bridge_volume: Array,
        boundary_responses: tuple[DEMBoundaryResponse, ...],
        barrier_allocation: DEMBarrierLiquidAllocation | None,
        barrier_particles: Array | None,
        barrier_indices: Array | None,
        barrier_minimum_volume: Array | None,
        step_size: Array,
    ) -> _DEMBoundaryAdvancePhase:
        if barrier_allocation is not None:
            liquid_process = self.method.liquid_process
            if liquid_process is None:
                raise RuntimeError(
                    "PreparedCompliantDEMDynamics.advance_boundary_liquid.liquid_process must provide the prepared transaction process; got None"
                )
            if barrier_particles is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.advance_boundary_liquid.state must provide the prepared transaction state; got {barrier_particles!r}"
                )
            if barrier_indices is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.advance_boundary_liquid.state must provide the prepared transaction state; got {barrier_indices!r}"
                )
            if barrier_minimum_volume is None:
                raise RuntimeError(
                    f"PreparedCompliantDEMDynamics.advance_boundary_liquid.state must provide the prepared transaction state; got {barrier_minimum_volume!r}"
                )
            barrier_bridge_volume = jnp.concatenate(
                tuple(
                    response.contact.next_history.cohesion.components[
                        self.liquid_component_index
                    ].bridge_volume
                    for response in boundary_responses
                )
            )
            barrier_release = jnp.concatenate(
                tuple(
                    response.contact.bridge_volume_release
                    for response in boundary_responses
                )
            )
            barrier_surface = jnp.concatenate(
                tuple(
                    response.contact.bridge_surface_area
                    for response in boundary_responses
                )
            )
            pair_bridge_total = jnp.sum(pair_bridge_volume)
            barrier_allocation_for_advance = DEMBarrierLiquidAllocation(
                barrier_allocation.bridge_volume,
                barrier_allocation.particle_withdrawal,
                barrier_allocation.barrier_withdrawal,
                liquid_evaluation.next_state.film_volume,
                liquid_evaluation.next_state.barrier_reservoir_volume,
                barrier_allocation.successful & liquid_evaluation.successful,
            )
            barrier_evaluation = liquid_process.advance_barriers(
                liquid_evaluation.next_state,
                barrier_allocation_for_advance,
                barrier_particles,
                barrier_indices,
                barrier_bridge_volume,
                barrier_release,
                barrier_surface,
                barrier_minimum_volume,
                step_size,
                self.bodies.capacity,
                other_bridge_volume=pair_bridge_total,
            )
            updated_boundaries = []
            for barrier_index, response in enumerate(boundary_responses):
                start = barrier_index * self.bodies.capacity
                stop = start + self.bodies.capacity
                component = response.contact.next_history.cohesion.components[
                    self.liquid_component_index
                ]
                next_boundary_component = eqx.tree_at(
                    lambda value: (value.active, value.bridge_volume),
                    component,
                    (
                        component.active
                        & ~barrier_evaluation.evaporated_ruptures[start:stop],
                        barrier_evaluation.bridge_volume[start:stop],
                    ),
                )
                boundary_components = list(
                    response.contact.next_history.cohesion.components
                )
                boundary_components[self.liquid_component_index] = next_boundary_component
                boundary_history = eqx.tree_at(
                    lambda value: value.cohesion,
                    response.contact.next_history,
                    DEMCohesionHistory(tuple(boundary_components)),
                )
                boundary_contact = eqx.tree_at(
                    lambda value: (
                        value.next_history,
                        value.cohesion_ruptures,
                        value.bridge_evaporation_loss,
                        value.successful,
                    ),
                    response.contact,
                    (
                        boundary_history,
                        response.contact.cohesion_ruptures
                        | barrier_evaluation.evaporated_ruptures[start:stop],
                        barrier_evaluation.evaporated_bridge_volume[start:stop],
                        response.contact.successful & barrier_evaluation.successful,
                    ),
                )
                updated_boundaries.append(
                    eqx.tree_at(
                        lambda value: (
                            value.contact,
                            value.successful,
                        ),
                        response,
                        (
                            boundary_contact,
                            response.successful & barrier_evaluation.successful,
                        ),
                    )
                )
            boundary_responses = tuple(updated_boundaries)
            liquid_evaluation = eqx.tree_at(
                lambda value: (value.next_state, value.successful),
                liquid_evaluation,
                (
                    barrier_evaluation.next_state,
                    liquid_evaluation.successful & barrier_evaluation.successful,
                ),
            )
        return _DEMBoundaryAdvancePhase(boundary_responses, liquid_evaluation)

    def _prepare_periodic_binding(
        self,
        method: CompliantDEMMethodPlan,
        neighborhood: AbstractPreparedParticleNeighborhood,
        bodies: PreparedRigidBodySet,
        barriers_: tuple[ImplicitDEMBarrier, ...],
        active_mask: np.ndarray,
        gravity_: Array,
        maximum_interaction_radius: float,
    ) -> PeriodicCell | None:
        periodic_cell = None
        if method.periodic_cell_control is not None:
            if not isinstance(neighborhood.box, PeriodicCell):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.neighborhood must provide a PeriodicCell neighborhood; got {type(neighborhood).__name__}"
                )
            if neighborhood.backend != "dense_pairs":
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.neighborhood must provide dense pair support; got {type(neighborhood).__name__}"
                )
            if barriers_:
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.barriers_ must have no implicit barriers for periodic cell control; got {type(barriers_).__name__}"
                )
            if not neighborhood.box.fully_periodic:
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.neighborhood must be fully periodic; got {type(neighborhood).__name__}"
                )
            if np.any(np.asarray(bodies.fixed_mask, dtype=np.bool_) & active_mask):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.bodies must have no active fixed particles for periodic cell control; got {type(bodies).__name__}"
                )
            if np.any(np.asarray(gravity_) != 0.0):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.gravity_ must provide zero body gravity; got {type(gravity_).__name__}"
                )
            if method.periodic_cell_control.ambient_dimension != bodies.ambient_dimension:
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.bodies must match bodies; got {type(bodies).__name__}"
                )
            if (
                method.periodic_cell_control.maximum_condition_number
                > neighborhood.box.certified_condition_number
            ):
                raise ValueError(
                    f"PreparedCompliantDEMDynamics.prepare_process_bindings.method must cover the cell-control maximum condition number; got {type(method).__name__}"
                )
            neighborhood.box.require_unique_image(maximum_interaction_radius)
            periodic_cell = neighborhood.box
        return periodic_cell


def _initial_kinematics(
    dynamics: PreparedCompliantDEMDynamics,
    raw_position: Array,
    initial_position: Array,
    velocity: ArrayLike,
    angular_velocity: ArrayLike | None,
    orientation: ArrayLike | None,
) -> RigidBodyKinematics:
    if orientation is None:
        if dynamics.shapes.shape_kind != "sphere":
            raise ValueError(
                f"PreparedCompliantDEMDynamics.initialize_state.orientation must provide explicit orientation; got {type(orientation).__name__}"
            )
        orientation_ = jnp.zeros(
            (dynamics.bodies.capacity, dynamics.bodies.orientation_dimension),
            dtype=raw_position.dtype,
        )
        if dynamics.bodies.ambient_dimension == 3:
            orientation_ = orientation_.at[:, 0].set(1.0)
    else:
        orientation_ = jnp.asarray(orientation, dtype=raw_position.dtype)
    angular_ = (
        jnp.zeros(
            (dynamics.bodies.capacity, dynamics.bodies.angular_dimension),
            dtype=raw_position.dtype,
        )
        if angular_velocity is None
        else jnp.asarray(angular_velocity, dtype=raw_position.dtype)
    )
    orientation_valid = jnp.all(jnp.isfinite(orientation_))
    if dynamics.bodies.ambient_dimension == 3:
        orientation_valid = orientation_valid & jnp.all(
            jnp.abs(jnp.sum(orientation_**2, axis=-1) - 1.0)
            <= dynamics.method.frame_tolerance
        )
    orientation_ = eqx.error_if(
        orientation_,
        ~orientation_valid,
        "DEM orientation must be finite and normalized.",
    )
    return dynamics.bodies.kinematics(initial_position, velocity, orientation_, angular_)


def _matching_tree_leaves(
    reference: PyTree[ArrayLike], value: PyTree[ArrayLike], name: str, /
) -> tuple[list[ArrayLike], list[ArrayLike]]:
    reference_leaves, reference_structure = jax.tree.flatten(reference)
    value_leaves, value_structure = jax.tree.flatten(value)
    if reference_structure != value_structure:
        raise ValueError(
            f"dem.matching_tree_leaves.reference must preserve the DEM state structure; got {type(reference).__name__}; context {name}"
        )
    for reference_leaf, value_leaf in zip(reference_leaves, value_leaves, strict=True):
        if jnp.shape(reference_leaf) != jnp.shape(value_leaf):
            raise ValueError(
                f"dem.matching_tree_leaves.reference must match the DEM state; got {jnp.shape!r}; context {name}"
            )
    return reference_leaves, value_leaves


def _require_frozen_compatible(
    state: PyTree[ArrayLike],
    point: PyTree[ArrayLike],
    value: PyTree[ArrayLike] | None = None,
    /,
) -> PyTree[Array]:
    state_leaves, point_leaves = _matching_tree_leaves(state, point, "Point")
    compatible = jnp.asarray(True)
    for base, target in zip(state_leaves, point_leaves, strict=True):
        if not eqx.is_inexact_array(base):
            compatible = compatible & jnp.all(jnp.asarray(base) == jnp.asarray(target))
    checked = point if value is None else value
    return jax.tree.map(
        lambda leaf: eqx.error_if(
            leaf,
            ~compatible,
            "DEM points have incompatible frozen route or topology leaves.",
        ),
        checked,
    )


def _continuous_tangent(
    state: PyTree[ArrayLike], vector: PyTree[ArrayLike], /
) -> PyTree[Array]:
    _matching_tree_leaves(state, vector, "Tangent")
    return jax.tree.map(
        lambda base, tangent: (
            tangent if eqx.is_inexact_array(base) else jnp.zeros_like(base)
        ),
        state,
        vector,
    )


__all__ = [
    "DEMDiagnostics",
    "DEMEnergyLedgerState",
    "DEMEvaluation",
    "DEMExternalLoad",
    "DEMRejectionReason",
    "DEMResolvedLoad",
    "DEMRuntimeState",
    "DEMStateGeometry",
    "DEMStepEnergyLedger",
    "DEMStepEvaluation",
    "DEMStepRestriction",
    "CompliantDEMBodyProperties",
    "PreparedCompliantDEMDynamics",
    "CompliantDEMMethodPlan",
]
