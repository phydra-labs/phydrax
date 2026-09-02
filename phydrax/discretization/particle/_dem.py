#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntFlag
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...metrix._state_geometry import AbstractStateGeometry
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
    ImplicitDEMBarrier,
)
from ._dem_cohesion import BagheriCapillaryBridgePlan, CompositeDEMCohesionPlan
from ._dem_contact import (
    DEMContactBatch,
    DEMContactHistory,
    DEMContactModelPlan,
    DEMContactResponse,
    HertzNormalContactPlan,
    LinearSpringDashpotNormalPlan,
    PreparedDEMContactModel,
)
from ._dem_contact_state import (
    DEMCohesionHistory,
    DEMContactEvaluationContext,
    remap_dem_contact_history,
)
from ._dem_kernels import reduce_dem_contact
from ._dem_liquid import (
    conserved_bagheri_component,
    ConservedLiquidBridgeProcessPlan,
    DEMBarrierLiquidAllocation,
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
from ._pair_state import match_particle_pair_keys, ParticlePairKeySpace
from ._pairwise import particle_pair_geometry
from ._particle_morphology import ParticleDynamicBodyProperties
from ._population import ParticlePopulationPlan
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._rigid_sphere import (
    PreparedRigidSphereSet,
    RigidSphereKinematics,
    RigidSphereLoad,
    sphere_lever_torque,
    sphere_pair_contact_geometry,
)
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


class SoftSphereDEMMethodPlan(StrictModule, NonTrainableState):
    """Fixed-capacity penalty DEM with branchwise contact topology."""

    contact: DEMContactModelPlan
    multicontact: AbstractDEMContactGraphCorrectionPlan | None
    periodic_cell_control: DEMPeriodicCellControlPlan | None
    liquid_process: ConservedLiquidBridgeProcessPlan | None
    maximum_overlap_fraction: float = eqx.field(static=True)
    distance_tolerance: float = eqx.field(static=True)
    frame_tolerance: float = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)
    key: DiscretizationKey
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        contact: DEMContactModelPlan,
        /,
        *,
        multicontact: AbstractDEMContactGraphCorrectionPlan | None = None,
        periodic_cell_control: DEMPeriodicCellControlPlan | None = None,
        liquid_process: ConservedLiquidBridgeProcessPlan | None = None,
        maximum_overlap_fraction: float = 0.1,
        distance_tolerance: float = 1.0e-12,
        frame_tolerance: float = 1.0e-10,
        name: str = "soft-sphere-dem",
        method_id: str | None = None,
    ):
        if not isinstance(contact, DEMContactModelPlan):
            raise TypeError("contact must be a DEMContactModelPlan.")
        if multicontact is not None and not isinstance(
            multicontact, AbstractDEMContactGraphCorrectionPlan
        ):
            raise TypeError(
                "multicontact must be an AbstractDEMContactGraphCorrectionPlan or None."
            )
        if periodic_cell_control is not None and not isinstance(
            periodic_cell_control, DEMPeriodicCellControlPlan
        ):
            raise TypeError(
                "periodic_cell_control must be a DEMPeriodicCellControlPlan or None."
            )
        if liquid_process is not None:
            if not isinstance(liquid_process, ConservedLiquidBridgeProcessPlan):
                raise TypeError(
                    "liquid_process must be a ConservedLiquidBridgeProcessPlan or None."
                )
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
                "DEM overlap and geometry tolerances must be finite and positive."
            )
        key = DiscretizationKey(
            name,
            DiscretizationRole.RESIDUAL,
            domain_labels=("material_point", "discrete_element"),
        )
        generated = canonical_fingerprint(
            {
                "kind": "soft-sphere-dem-method-plan",
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
                "maximum_overlap_fraction": overlap,
                "distance_tolerance": distance,
                "frame_tolerance": frame,
                "differentiability": "branchwise",
                "key": key.key_id,
            }
        )
        identifier = generated if method_id is None else str(method_id)
        if not identifier:
            raise ValueError("method_id must be nonempty.")
        self.contact = contact
        self.multicontact = multicontact
        self.periodic_cell_control = periodic_cell_control
        self.liquid_process = liquid_process
        self.maximum_overlap_fraction = overlap
        self.distance_tolerance = distance
        self.frame_tolerance = frame
        self.differentiability = "branchwise"
        self.key = key
        self.method_id = identifier


class DEMExternalLoad(StrictModule):
    force: Array
    torque: Array


class DEMResolvedLoad(StrictModule):
    """Source-resolved endpoint loads used by the discrete work ledger."""

    particle_contact: RigidSphereLoad
    boundaries: tuple[RigidSphereLoad, ...]
    gravity: RigidSphereLoad
    external: RigidSphereLoad
    total: RigidSphereLoad


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


class DEMRuntimeState(StrictModule):
    kinematics: RigidSphereKinematics
    body_properties: ParticleDynamicBodyProperties
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


class DEMStepRestriction(StrictModule):
    contact_period: Array
    rayleigh: Array
    selected: Array


class DEMEvaluation(StrictModule):
    neighborhood: ParticleNeighborhoodState
    neighborhood_cache: ParticleVerletState | None
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
    """Additive geometry on continuous DEM leaves with frozen discrete routes."""

    geometry_id: str = eqx.field(static=True)
    retraction_method: str = "continuous-leaf-addition"
    trivial: bool = True
    supports_exact_pullback: bool = True
    supports_commutator_free: bool = True

    def __init__(self, dynamics_id: str, /):
        identifier = str(dynamics_id)
        if not identifier:
            raise ValueError("dynamics_id must be nonempty.")
        self.geometry_id = f"state-geometry:dem:{identifier}"

    def contains(self, state, /):
        return tree_allfinite(state)

    def project_tangent(self, state, vector, /):
        return _continuous_tangent(state, vector)

    def to_local(self, state, tangent, /):
        return _continuous_tangent(state, tangent)

    def from_local(self, state, local_tangent, /):
        return _continuous_tangent(state, local_tangent)

    def retract(self, state, local_tangent, /):
        return jax.tree.map(
            lambda base, tangent: base + tangent if eqx.is_inexact_array(base) else base,
            state,
            local_tangent,
        )

    def inverse_retract(self, state, point, /):
        return jax.tree.map(
            lambda base, target: (
                target - base if eqx.is_inexact_array(base) else jnp.zeros_like(base)
            ),
            state,
            point,
        )

    def pullback(self, state, local_tangent, tangent, /):
        del local_tangent
        return _continuous_tangent(state, tangent)


class PreparedSoftSphereDEMDynamics(StrictModule, NonTrainableState):
    """Prepared spherical DEM contact evaluation and explicit kinetic update."""

    bodies: PreparedRigidSphereSet
    neighborhood: AbstractPreparedParticleNeighborhood
    pair_key_space: ParticlePairKeySpace
    contact_model: PreparedDEMContactModel
    method: SoftSphereDEMMethodPlan
    materials: Any
    barriers: tuple[ImplicitDEMBarrier, ...]
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

    def __init__(
        self,
        bodies: PreparedRigidSphereSet,
        neighborhood: AbstractPreparedParticleNeighborhood,
        pair_key_space: ParticlePairKeySpace,
        contact_model: PreparedDEMContactModel,
        method: SoftSphereDEMMethodPlan,
        materials: Any,
        /,
        *,
        barriers: Sequence[ImplicitDEMBarrier] = (),
        gravity: ArrayLike | None = None,
        external_load: ExternalDEMLoad | None = None,
        external_load_id: str | None = None,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
    ):
        if not isinstance(bodies, PreparedRigidSphereSet):
            raise TypeError("bodies must be a PreparedRigidSphereSet.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError(
                "neighborhood must be an AbstractPreparedParticleNeighborhood."
            )
        if not isinstance(pair_key_space, ParticlePairKeySpace):
            raise TypeError("pair_key_space must be a ParticlePairKeySpace.")
        if not isinstance(contact_model, PreparedDEMContactModel):
            raise TypeError("contact_model must be a PreparedDEMContactModel.")
        if not isinstance(method, SoftSphereDEMMethodPlan):
            raise TypeError("method must be a SoftSphereDEMMethodPlan.")
        if neighborhood.particle_discretization_id != bodies.particles.prepared_id:
            raise ValueError(
                "DEM neighborhood was prepared for another particle support."
            )
        if pair_key_space.particle_discretization_id != bodies.particles.prepared_id:
            raise ValueError("DEM pair key space was prepared for another support.")
        if contact_model.plan.contact_model_id != method.contact.contact_model_id:
            raise ValueError("Prepared contact model does not match the DEM method plan.")
        barriers_ = tuple(barriers)
        if any(not isinstance(value, ImplicitDEMBarrier) for value in barriers_):
            raise TypeError("barriers must contain ImplicitDEMBarrier values.")
        barrier_ids = tuple(value.barrier_id for value in barriers_)
        if len(set(barrier_ids)) != len(barrier_ids):
            raise ValueError("DEM barrier IDs must be unique.")
        for barrier in barriers_:
            if barrier.geometry.ambient_dimension != bodies.ambient_dimension:
                raise ValueError("DEM barrier dimension does not match rigid spheres.")
            if isinstance(method.contact.normal, HertzNormalContactPlan) and (
                "contact_curvature"
                not in {value.value for value in barrier.geometry.capabilities}
            ):
                raise ValueError(
                    "Hertz barrier contact requires contact-curvature capability."
                )
            if barrier.material_id >= materials.material_count:
                raise ValueError("DEM barrier material ID is out of range.")
        gravity_ = (
            jnp.zeros((bodies.ambient_dimension,), dtype=bodies.radii.dtype)
            if gravity is None
            else jnp.asarray(gravity, dtype=bodies.radii.dtype)
        )
        if gravity_.shape != (bodies.ambient_dimension,):
            raise ValueError("gravity must match the rigid-sphere ambient dimension.")
        if not bool(np.all(np.isfinite(np.asarray(gravity_)))):
            raise ValueError("gravity must be finite.")
        if external_load is not None and not callable(external_load):
            raise TypeError("external_load must be callable or None.")
        if external_load is None and external_load_id is not None:
            raise ValueError("external_load_id requires external_load.")
        if external_load is not None and not external_load_id:
            raise ValueError("External DEM load requires a stable nonempty ID.")
        execution_ = ParticleExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if execution_.realization != neighborhood.backend:
            raise ValueError(
                "DEM execution realization does not match neighborhood backend."
            )
        if (
            execution_.kernel_backend == "dense_fused"
            and neighborhood.backend != "dense_pairs"
        ):
            raise ValueError("dense_fused requires a dense-pair neighborhood.")
        if (
            execution_.kernel_backend == "cell_fused"
            and neighborhood.backend != "cell_edge_list"
        ):
            raise ValueError("cell_fused requires a cell-list neighborhood.")
        if execution_.kernel_backend == "verlet_fused" and not isinstance(
            neighborhood, PreparedVerletParticleNeighborhood
        ):
            raise ValueError("verlet_fused requires a prepared Verlet neighborhood.")
        active_mask = np.asarray(bodies.particles.active_mask, dtype=bool)
        interaction_extents = method.contact.interaction_extents_for_radii(
            np.asarray(bodies.radii)[active_mask],
            np.asarray(bodies.material_ids)[active_mask],
            materials.material_count,
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
        periodic_cell = None
        if method.periodic_cell_control is not None:
            if not isinstance(neighborhood.box, PeriodicCell):
                raise ValueError(
                    "Periodic DEM cell control requires a PeriodicCell neighborhood."
                )
            if neighborhood.backend != "dense_pairs":
                raise ValueError(
                    "Periodic DEM cell control currently requires dense pair authority."
                )
            if barriers_:
                raise ValueError(
                    "Periodic DEM cell control does not support implicit barriers."
                )
            if not neighborhood.box.fully_periodic:
                raise ValueError("Controlled DEM cells must be fully periodic.")
            if np.any(np.asarray(bodies.fixed_mask, dtype=bool) & active_mask):
                raise ValueError(
                    "Controlled periodic DEM cells do not support fixed particles."
                )
            if np.any(np.asarray(gravity_) != 0.0):
                raise ValueError(
                    "Controlled periodic DEM cells require zero body gravity."
                )
            if method.periodic_cell_control.ambient_dimension != bodies.ambient_dimension:
                raise ValueError("DEM cell-control dimension does not match bodies.")
            if (
                method.periodic_cell_control.maximum_condition_number
                > neighborhood.box.certified_condition_number
            ):
                raise ValueError(
                    "PeriodicCell condition certificate does not cover cell control."
                )
            neighborhood.box.require_unique_image(maximum_interaction_radius)
            periodic_cell = neighborhood.box
        liquid_component_index = -1
        if method.liquid_process is not None:
            if barriers_:
                bindings = method.liquid_process.barrier_capillaries
                if tuple(value.barrier_id for value in bindings) != barrier_ids:
                    raise ValueError(
                        "Conserved barrier capillaries must bind every DEM barrier "
                        "exactly once and in prepared barrier order."
                    )
                for barrier, binding in zip(barriers_, bindings, strict=True):
                    if binding.law != "bagheri":
                        raise ValueError(
                            "Conserved DEM barrier liquid currently requires "
                            "the Bagheri sphere-surface law."
                        )
                    if (
                        binding.geometry_policy == "isotropic_curvature"
                        and "contact_curvature"
                        not in {value.value for value in barrier.geometry.capabilities}
                    ):
                        raise ValueError(
                            "Isotropic barrier capillarity requires certified "
                            "contact curvature."
                        )
            _, liquid_component_index = conserved_bagheri_component(
                method.contact.cohesion
            )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY,
            ),
            diagnostics=(
                "soft-sphere penalty contact",
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
                "barrier_count": len(barriers_),
                "ambient_dimension": bodies.ambient_dimension,
            },
        )
        self.bodies = bodies
        self.neighborhood = neighborhood
        self.pair_key_space = pair_key_space
        self.contact_model = contact_model
        self.method = method
        self.materials = materials
        self.barriers = barriers_
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
                "kind": "prepared-soft-sphere-dem-dynamics",
                "bodies": bodies.prepared_id,
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
        return DEMStateGeometry(self.prepared_id)

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def initial_body_properties(self) -> ParticleDynamicBodyProperties:
        population = ParticlePopulationPlan(self.bodies.particles).initialize(
            active_mask=self.bodies.particles.active_mask,
            masses=self.bodies.particles.safe_masses,
        )
        return ParticleDynamicBodyProperties(
            population,
            self.bodies.inverse_masses,
            self.bodies.radii,
            self.bodies.inertias,
            self.bodies.inverse_inertias,
        )

    def empty_particle_history(self) -> DEMContactHistory:
        return self.contact_model.empty_history(
            self.neighborhood.pair_capacity,
            self.bodies.radii.dtype,
        )

    def empty_boundary_histories(self) -> tuple[DEMContactHistory, ...]:
        return tuple(
            self.contact_model.empty_history(
                self.bodies.capacity,
                self.bodies.radii.dtype,
            )
            for _ in self.barriers
        )

    def _zero_load(self) -> RigidSphereLoad:
        return RigidSphereLoad(
            jnp.zeros(
                (self.bodies.capacity, self.bodies.ambient_dimension),
                dtype=self.bodies.radii.dtype,
            ),
            jnp.zeros(
                (self.bodies.capacity, self.bodies.angular_dimension),
                dtype=self.bodies.radii.dtype,
            ),
        )

    def _kinetic_energy(
        self,
        kinematics: RigidSphereKinematics,
        properties: ParticleDynamicBodyProperties,
        /,
    ) -> Array:
        return 0.5 * jnp.sum(
            properties.masses[:, None] * kinematics.velocity**2
        ) + 0.5 * jnp.sum(properties.inertias[:, None] * kinematics.angular_velocity**2)

    def _gravity_potential(
        self,
        kinematics: RigidSphereKinematics,
        properties: ParticleDynamicBodyProperties,
        /,
    ) -> Array:
        potential = -properties.masses * jnp.sum(
            kinematics.position * self.gravity, axis=-1
        )
        return jnp.sum(jnp.where(properties.active, potential, 0.0))

    def _ledger_view(
        self,
        ledger: DEMEnergyLedgerState,
        kinematics: RigidSphereKinematics,
        properties: ParticleDynamicBodyProperties,
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
        previous: RigidSphereLoad,
        current: RigidSphereLoad,
        previous_kinematics: RigidSphereKinematics,
        current_kinematics: RigidSphereKinematics,
        properties: ParticleDynamicBodyProperties,
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
        next_kinematics: RigidSphereKinematics,
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
        args: Any = None,
    ) -> DEMRuntimeState:
        raw_position = jnp.asarray(position)
        periodic_state = (
            None
            if self.periodic_cell is None
            else self.method.periodic_cell_control.initialize(
                self.periodic_cell, raw_position.dtype
            )
        )
        initial_position = raw_position
        if periodic_state is not None:
            initial_position, _ = self.periodic_cell.wrap_with_vectors(
                raw_position, periodic_state.vectors
            )
        kinematics = self.bodies.kinematics(initial_position, velocity, angular_velocity)
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
                kinematics.position, active_mask=body_properties.active
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
            RigidSphereKinematics(
                checked_position,
                initialized.kinematics.velocity,
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

    def apply_body_properties(
        self,
        time: Array,
        state: DEMRuntimeState,
        properties: ParticleDynamicBodyProperties,
        rebuild_neighborhood: Array,
        /,
        *,
        args: Any = None,
    ) -> DEMBodyPropertyUpdateResult:
        if not isinstance(state, DEMRuntimeState):
            raise TypeError("state must be a DEMRuntimeState.")
        if not isinstance(properties, ParticleDynamicBodyProperties):
            raise TypeError("properties must be ParticleDynamicBodyProperties.")
        mobile = (properties.active & ~self.bodies.fixed_mask)[:, None]
        kinematics = RigidSphereKinematics(
            state.kinematics.position,
            jnp.where(mobile, state.kinematics.velocity, 0.0),
            jnp.where(mobile, state.kinematics.angular_velocity, 0.0),
        )
        cache = state.neighborhood_cache
        if isinstance(self.neighborhood, PreparedVerletParticleNeighborhood):
            if cache is None:
                raise ValueError(
                    "Verlet morphology update requires a neighborhood cache."
                )
            initialized = self.neighborhood.initialize(
                kinematics.position, active_mask=properties.active
            )
            cache = tree_where(jnp.asarray(rebuild_neighborhood), initialized, cache)
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

    def evaluate(
        self,
        time: Array,
        state: DEMRuntimeState,
        step_size: Array,
        args: Any,
        /,
    ) -> DEMEvaluation:
        if not isinstance(state, DEMRuntimeState):
            raise TypeError("state must be a DEMRuntimeState.")
        properties = state.body_properties
        expected = (self.bodies.capacity,)
        body_shapes_valid = (
            properties.masses.shape == expected
            and properties.inverse_masses.shape == expected
            and properties.radii.shape == expected
            and properties.inertias.shape == expected
            and properties.inverse_inertias.shape == expected
            and properties.active.shape == expected
        )
        if not body_shapes_valid:
            raise ValueError("Dynamic DEM body properties have invalid shapes.")
        body_valid = (
            jnp.all(jnp.isfinite(properties.masses) & (properties.masses >= 0.0))
            & jnp.all(jnp.isfinite(properties.inverse_masses))
            & jnp.all(jnp.isfinite(properties.radii) & (properties.radii >= 0.0))
            & jnp.all(jnp.isfinite(properties.inertias) & (properties.inertias > 0.0))
            & jnp.all(jnp.isfinite(properties.inverse_inertias))
            & jnp.all(
                ~properties.active
                | ((properties.masses > 0.0) & (properties.radii > 0.0))
            )
        )
        kinematics = self.bodies.kinematics(
            state.kinematics.position,
            state.kinematics.velocity,
            state.kinematics.angular_velocity,
        )
        if isinstance(self.neighborhood, PreparedVerletParticleNeighborhood):
            if state.neighborhood_cache is None:
                raise ValueError(
                    "Prepared Verlet DEM state requires a neighborhood cache."
                )
            neighborhood_cache = self.neighborhood.update(
                kinematics.position,
                state.neighborhood_cache,
                active_mask=properties.active,
            )
            neighborhood = neighborhood_cache.neighborhood
            rebuilt = neighborhood_cache.rebuilt
        else:
            neighborhood_cache = None
            neighborhood = self.neighborhood.build(
                kinematics.position, active_mask=properties.active
            )
            rebuilt = jnp.asarray(True)
        pairs = neighborhood.pair_relation
        keys = self.pair_key_space.keys(pairs)

        def align_rebuilt(_):
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

        def align_reused(_):
            same_identity = jnp.all(
                state.particle_history.pair_keys == keys.keys, axis=-1
            )
            same_keys = jnp.all(
                ~state.particle_history.valid | (keys.valid & same_identity)
            )
            history = state.particle_history.with_routes(keys.keys, keys.valid)
            return history, keys.valid & history.valid, same_keys

        remapped_history, continued, alignment_successful = jax.lax.cond(
            rebuilt, align_rebuilt, align_reused, operand=None
        )
        center_geometry = particle_pair_geometry(
            kinematics.position,
            pairs,
            box=neighborhood.box,
            cell_vectors=(
                None if state.periodic_cell is None else state.periodic_cell.vectors
            ),
        )
        sphere_geometry = sphere_pair_contact_geometry(
            self.bodies,
            kinematics,
            pairs,
            center_geometry,
            distance_tolerance=self.method.distance_tolerance,
            radii=state.body_properties.radii,
        )
        batch = DEMContactBatch(
            sphere_geometry.normal,
            sphere_geometry.gap,
            sphere_geometry.overlap,
            (
                properties.radii[pairs.left_indices]
                * properties.radii[pairs.right_indices]
                / (
                    properties.radii[pairs.left_indices]
                    + properties.radii[pairs.right_indices]
                )
            ),
            sphere_geometry.left_arm,
            sphere_geometry.right_arm,
            sphere_geometry.normal_velocity,
            sphere_geometry.tangential_velocity,
            kinematics.angular_velocity[pairs.left_indices],
            kinematics.angular_velocity[pairs.right_indices],
            sphere_geometry.valid
            & properties.active[pairs.left_indices]
            & properties.active[pairs.right_indices],
        )
        left = pairs.left_indices
        right = pairs.right_indices
        liquid_allocation = None
        minimum_bridge_volume = jnp.zeros_like(batch.gap)
        if self.method.liquid_process is not None:
            if state.liquid is None:
                raise RuntimeError("Prepared liquid DEM state lacks liquid inventory.")
            bridge_plan, component_index = conserved_bagheri_component(
                self.method.contact.cohesion
            )
            component = remapped_history.cohesion.components[component_index]
            requested_volume = bridge_plan.pair_bridge_volume(
                self.bodies.material_ids[left],
                self.bodies.material_ids[right],
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
        contact_context = DEMContactEvaluationContext(
            keys.keys,
            keys.valid,
            continued,
            state.body_properties.inverse_masses[left],
            state.body_properties.inverse_masses[right],
            state.body_properties.radii[left],
            state.body_properties.radii[right],
            self.bodies.material_ids[left],
            self.bodies.material_ids[right],
            step_size,
            -jnp.ones((), dtype=jnp.int32),
        )
        particle_contact = self.contact_model.evaluate(
            batch,
            remapped_history,
            contact_context,
            frame_tolerance=self.method.frame_tolerance,
        )
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
                    self.bodies.material_ids,
                    self.materials,
                    base_batch.valid & particle_contact.active,
                )
                scale = jnp.maximum(
                    jnp.minimum(
                        state.body_properties.radii[left],
                        state.body_properties.radii[right],
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
                )
                particle_contact = self.contact_model.evaluate(
                    batch,
                    remapped_history,
                    contact_context,
                    frame_tolerance=self.method.frame_tolerance,
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

        def evaluate_boundaries(histories):
            capillary_plans = (
                (None,) * len(self.barriers)
                if self.method.liquid_process is None
                else self.method.liquid_process.barrier_capillaries
            )
            return tuple(
                evaluate_dem_barrier(
                    barrier,
                    self.bodies,
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
                    histories,
                    capillary_plans,
                    strict=True,
                )
            )

        boundary_responses = evaluate_boundaries(state.boundary_histories)
        barrier_allocation = None
        barrier_particles = None
        barrier_indices = None
        barrier_minimum_volume = None
        if self.method.liquid_process is not None and self.barriers:
            if state.liquid is None or liquid_allocation is None:
                raise RuntimeError("Liquid bridge transaction was not initialized.")
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
                        self.bodies.material_ids,
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

        liquid_evaluation = None
        if self.method.liquid_process is not None:
            if state.liquid is None or liquid_allocation is None:
                raise RuntimeError("Liquid bridge transaction was not initialized.")
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
            if barrier_allocation is not None:
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
                pair_bridge_total = jnp.sum(next_component.bridge_volume)
                barrier_allocation_for_advance = DEMBarrierLiquidAllocation(
                    barrier_allocation.bridge_volume,
                    barrier_allocation.particle_withdrawal,
                    barrier_allocation.barrier_withdrawal,
                    liquid_evaluation.next_state.film_volume,
                    liquid_evaluation.next_state.barrier_reservoir_volume,
                    barrier_allocation.successful & liquid_evaluation.successful,
                )
                barrier_evaluation = self.method.liquid_process.advance_barriers(
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
                    boundary_components[self.liquid_component_index] = (
                        next_boundary_component
                    )
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
        pair_load = reduce_dem_contact(
            pairs,
            particle_contact,
            particle_capacity=self.bodies.capacity,
            ambient_dimension=self.bodies.ambient_dimension,
            angular_dimension=self.bodies.angular_dimension,
            execution=self.execution,
            precision=self.precision,
        )
        pair_force = pair_load.force
        pair_torque = pair_load.torque
        particle_load = self.bodies.load(
            self.precision.output(pair_force), self.precision.output(pair_torque)
        )
        boundary_loads = tuple(
            self.bodies.load(response.particle_force, response.particle_torque)
            for response in boundary_responses
        )
        boundary_force = jnp.zeros_like(pair_force)
        boundary_torque = jnp.zeros_like(pair_torque)
        for load in boundary_loads:
            boundary_force = boundary_force + load.force
            boundary_torque = boundary_torque + load.torque
        gravity_force = properties.masses[:, None] * self.gravity
        gravity_force = jnp.where(properties.active[:, None], gravity_force, 0.0)
        gravity_load = self.bodies.load(gravity_force, jnp.zeros_like(pair_torque))
        external_value = self._external_load(time, kinematics, args)
        external_load = self.bodies.load(
            jnp.where(properties.active[:, None], external_value.force, 0.0),
            jnp.where(properties.active[:, None], external_value.torque, 0.0),
        )
        total_load = self.bodies.load(
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
            ~sphere_geometry.successful,
            int(DEMRejectionReason.GEOMETRY),
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
            ~(tree_allfinite(resolved_loads) & jnp.isfinite(contact_energy)),
            int(DEMRejectionReason.NONFINITE),
            0,
        ).astype(jnp.int32)
        successful = reasons == 0
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
            neighborhood.candidate_pair_count
            + self.bodies.particles.active_count * len(self.barriers)
        )
        result = DEMEvaluation(
            neighborhood=neighborhood,
            neighborhood_cache=neighborhood_cache,
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
            state.body_properties.inverse_inertias[:, None] * state.loads.total.torque
        )
        half_velocity = jnp.where(mobile, half_velocity, 0.0)
        half_angular = jnp.where(mobile, half_angular, 0.0)
        next_position = state.kinematics.position + step_size * half_velocity
        next_position = jnp.where(mobile, next_position, state.kinematics.position)
        if state.periodic_cell is not None:
            next_position, _ = self.periodic_cell.wrap_with_vectors(
                next_position, state.periodic_cell.vectors
            )
        staged = DEMRuntimeState(
            RigidSphereKinematics(next_position, half_velocity, half_angular),
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
                raise RuntimeError("Prepared periodic DEM state lacks bulk stress.")
            cell_update = self.method.periodic_cell_control.update(
                self.periodic_cell,
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
                RigidSphereKinematics(next_position, half_velocity, half_angular),
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
            state.body_properties.inverse_inertias[:, None]
            * evaluation.loads.total.torque
        )
        next_velocity = jnp.where(mobile, next_velocity, 0.0)
        next_angular = jnp.where(mobile, next_angular, 0.0)
        next_kinematics = RigidSphereKinematics(
            next_position, next_velocity, next_angular
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
        self, properties: ParticleDynamicBodyProperties | None = None, /
    ) -> DEMStepRestriction:
        selected_properties = (
            self.initial_body_properties() if properties is None else properties
        )
        active = selected_properties.active & ~self.bodies.fixed_mask
        masses = jnp.where(active, selected_properties.masses, jnp.inf)
        minimum_mass = jnp.min(masses)
        normal = self.method.contact.normal
        if isinstance(normal, LinearSpringDashpotNormalPlan):
            maximum_stiffness = jnp.max(jnp.asarray(normal.stiffness))
            contact_period = jnp.pi * jnp.sqrt(0.5 * minimum_mass / maximum_stiffness)
        else:
            contact_period = jnp.asarray(jnp.inf, dtype=self.bodies.radii.dtype)
        material = self.bodies.material_ids
        young = self.materials.young_modulus[material]
        poisson = self.materials.poisson_ratio[material]
        shear = young / (2.0 * (1.0 + poisson))
        radius = selected_properties.radii
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
        self, time: Array, kinematics: RigidSphereKinematics, args: Any, /
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
            raise TypeError("external_load must return DEMExternalLoad.")
        checked = self.bodies.load(result.force, result.torque)
        return DEMExternalLoad(checked.force, checked.torque)

    def _diagnostics(
        self,
        kinematics,
        neighborhood_cache,
        particle_contact,
        multicontact,
        boundaries,
        liquid_evaluation,
        body_properties,
        pair_force,
        pair_torque,
        energy,
        successful,
        rejection_reasons,
        /,
    ) -> DEMDiagnostics:
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
        masses = body_properties.masses
        linear_momentum = jnp.sum(masses[:, None] * kinematics.velocity, axis=0)
        angular_momentum = jnp.sum(
            sphere_lever_torque(
                kinematics.position,
                masses[:, None] * kinematics.velocity,
                self.bodies.ambient_dimension,
            )
            + body_properties.inertias[:, None] * kinematics.angular_velocity,
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
        )


def _continuous_tangent(state, vector, /):
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
    "PreparedSoftSphereDEMDynamics",
    "SoftSphereDEMMethodPlan",
]
