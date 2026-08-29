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
from ._dem_boundary import (
    DEMBoundaryResponse,
    evaluate_dem_barrier,
    ImplicitDEMBarrier,
)
from ._dem_contact import (
    DEMContactBatch,
    DEMContactHistory,
    DEMContactModelPlan,
    DEMContactResponse,
    HertzNormalContactPlan,
    LinearSpringDashpotNormalPlan,
    PreparedDEMContactModel,
)
from ._dem_kernels import reduce_dem_contact
from ._neighborhood import (
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pair_state import (
    match_particle_pair_keys,
    ParticlePairKeySpace,
    remap_particle_pair_values,
)
from ._pairwise import particle_pair_geometry
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
    CONTACT = 1 << 5
    NONFINITE = 1 << 6
    OVERLAP = 1 << 7
    ENERGY = 1 << 8
    FRAME = 1 << 9


class SoftSphereDEMMethodPlan(StrictModule, NonTrainableState):
    """Fixed-capacity penalty DEM with branchwise contact topology."""

    contact: DEMContactModelPlan
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
        maximum_overlap_fraction: float = 0.1,
        distance_tolerance: float = 1.0e-12,
        frame_tolerance: float = 1.0e-10,
        name: str = "soft-sphere-dem",
        method_id: str | None = None,
    ):
        if not isinstance(contact, DEMContactModelPlan):
            raise TypeError("contact must be a DEMContactModelPlan.")
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
    particle_history: DEMContactHistory
    boundary_histories: tuple[DEMContactHistory, ...]
    neighborhood_cache: ParticleVerletState | None
    loads: DEMResolvedLoad
    energy: DEMEnergyLedgerState


class DEMDiagnostics(StrictModule):
    active_contacts: Array
    sticking_contacts: Array
    sliding_contacts: Array
    maximum_overlap_fraction: Array
    minimum_gap_margin: Array
    minimum_no_tension_margin: Array
    minimum_frame_transport_margin: Array
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
    contact_energy: Array
    contact_births: Array
    contact_deaths: Array
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
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY,
            ),
            diagnostics=(
                "soft-sphere penalty contact",
                "stable pair-key history remapping",
                "kick-drift-contact-kick integration",
                "branchwise differentiation through fixed realized routes",
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
        self.precision = precision_
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

    def empty_particle_history(self) -> DEMContactHistory:
        return DEMContactHistory.empty(
            self.neighborhood.pair_capacity,
            self.bodies.ambient_dimension,
            self.bodies.radii.dtype,
        )

    def empty_boundary_histories(self) -> tuple[DEMContactHistory, ...]:
        return tuple(
            DEMContactHistory.empty(
                self.bodies.capacity,
                self.bodies.ambient_dimension,
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

    def _kinetic_energy(self, kinematics: RigidSphereKinematics, /) -> Array:
        masses = self.bodies.particles.safe_masses
        return 0.5 * jnp.sum(masses[:, None] * kinematics.velocity**2) + 0.5 * jnp.sum(
            self.bodies.inertias[:, None] * kinematics.angular_velocity**2
        )

    def _gravity_potential(self, kinematics: RigidSphereKinematics, /) -> Array:
        masses = self.bodies.particles.safe_masses
        potential = -masses * jnp.sum(kinematics.position * self.gravity, axis=-1)
        return jnp.sum(jnp.where(self.bodies.particles.active_mask, potential, 0.0))

    def _ledger_view(
        self,
        ledger: DEMEnergyLedgerState,
        kinematics: RigidSphereKinematics,
        contact_energy: Array,
        boundary_wall_power: Array,
        /,
    ) -> DEMEnergyLedgerState:
        return DEMEnergyLedgerState(
            ledger.initial_kinetic_energy,
            ledger.initial_contact_energy,
            ledger.initial_gravity_potential,
            self._kinetic_energy(kinematics),
            contact_energy,
            self._gravity_potential(kinematics),
            boundary_wall_power,
            ledger.cumulative_particle_contact_work,
            ledger.cumulative_boundary_contact_work,
            ledger.cumulative_prescribed_wall_work,
            ledger.cumulative_gravity_work,
            ledger.cumulative_external_work,
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
        mobile = self.bodies.particles.active_mask & ~self.bodies.fixed_mask
        translational = jnp.sum(impulse * average_velocity, axis=-1)
        rotational = jnp.sum(angular_impulse * average_angular_velocity, axis=-1)
        return jnp.sum(jnp.where(mobile, translational + rotational, 0.0))

    def _step_energy(
        self,
        state: DEMRuntimeState,
        next_kinematics: RigidSphereKinematics,
        evaluation: DEMEvaluation,
        step_size: Array,
        /,
    ) -> DEMStepEnergyLedger:
        kinetic_before = self._kinetic_energy(state.kinematics)
        kinetic_after = self._kinetic_energy(next_kinematics)
        contact_before = state.energy.contact_energy
        contact_after = evaluation.contact_energy
        gravity_before = self._gravity_potential(state.kinematics)
        gravity_after = self._gravity_potential(next_kinematics)
        particle_work = self._source_work(
            state.loads.particle_contact,
            evaluation.loads.particle_contact,
            state.kinematics,
            next_kinematics,
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
            step_size,
        )
        external_work = self._source_work(
            state.loads.external,
            evaluation.loads.external,
            state.kinematics,
            next_kinematics,
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
        kinematics = self.bodies.kinematics(position, velocity, angular_velocity)
        zero = self._zero_load()
        resolved_zero = DEMResolvedLoad(
            zero,
            tuple(self._zero_load() for _ in self.barriers),
            self._zero_load(),
            self._zero_load(),
            self._zero_load(),
        )
        kinetic = self._kinetic_energy(kinematics)
        gravity_potential = self._gravity_potential(kinematics)
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
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        neighborhood_cache = (
            self.neighborhood.initialize(kinematics.position)
            if isinstance(self.neighborhood, PreparedVerletParticleNeighborhood)
            else None
        )
        seed = DEMRuntimeState(
            kinematics,
            self.empty_particle_history(),
            self.empty_boundary_histories(),
            neighborhood_cache,
            resolved_zero,
            energy_seed,
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
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        initialized = DEMRuntimeState(
            kinematics,
            evaluation.particle_contact.next_history,
            tuple(value.contact.next_history for value in evaluation.boundaries),
            evaluation.neighborhood_cache,
            evaluation.loads,
            initialized_energy,
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
            initialized.particle_history,
            initialized.boundary_histories,
            initialized.neighborhood_cache,
            initialized.loads,
            initialized.energy,
        )

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
                kinematics.position, state.neighborhood_cache
            )
            neighborhood = neighborhood_cache.neighborhood
            rebuilt = neighborhood_cache.rebuilt
        else:
            neighborhood_cache = None
            neighborhood = self.neighborhood.build(kinematics.position)
            rebuilt = jnp.asarray(True)
        pairs = neighborhood.pair_relation
        keys = self.pair_key_space.keys(pairs)
        maximum_key = max(self.pair_key_space.pair_count - 1, 0)

        def align_rebuilt(_):
            remap = match_particle_pair_keys(
                state.particle_history.pair_keys,
                state.particle_history.valid,
                keys.keys,
                keys.valid,
                maximum_key=maximum_key,
            )
            values = remap_particle_pair_values(
                remap,
                {
                    "active": state.particle_history.active,
                    "sliding": state.particle_history.sliding,
                    "normal_maximum_overlap": (
                        state.particle_history.normal_maximum_overlap
                    ),
                    "normal_plastic_overlap": (
                        state.particle_history.normal_plastic_overlap
                    ),
                    "normal_previous_overlap": (
                        state.particle_history.normal_previous_overlap
                    ),
                    "previous_normal": state.particle_history.previous_normal,
                    "tangential_displacement": (
                        state.particle_history.tangential_displacement
                    ),
                },
            )
            history = DEMContactHistory(
                keys.keys,
                keys.valid,
                values["active"].astype(bool),
                values["sliding"].astype(bool),
                values["normal_maximum_overlap"],
                values["normal_plastic_overlap"],
                values["normal_previous_overlap"],
                values["previous_normal"],
                values["tangential_displacement"],
            )
            return history, remap.continued, remap.successful

        def align_reused(_):
            same_keys = jnp.all(
                ~state.particle_history.valid
                | (keys.valid & (state.particle_history.pair_keys == keys.keys))
            )
            history = DEMContactHistory(
                keys.keys,
                keys.valid,
                state.particle_history.active,
                state.particle_history.sliding,
                state.particle_history.normal_maximum_overlap,
                state.particle_history.normal_plastic_overlap,
                state.particle_history.normal_previous_overlap,
                state.particle_history.previous_normal,
                state.particle_history.tangential_displacement,
            )
            return history, keys.valid & history.valid, same_keys

        remapped_history, continued, alignment_successful = jax.lax.cond(
            rebuilt, align_rebuilt, align_reused, operand=None
        )
        center_geometry = particle_pair_geometry(
            kinematics.position, pairs, box=neighborhood.box
        )
        sphere_geometry = sphere_pair_contact_geometry(
            self.bodies,
            kinematics,
            pairs,
            center_geometry,
            distance_tolerance=self.method.distance_tolerance,
        )
        batch = DEMContactBatch(
            sphere_geometry.normal,
            sphere_geometry.gap,
            sphere_geometry.overlap,
            (
                self.bodies.radii[pairs.left_indices]
                * self.bodies.radii[pairs.right_indices]
                / (
                    self.bodies.radii[pairs.left_indices]
                    + self.bodies.radii[pairs.right_indices]
                )
            ),
            sphere_geometry.left_arm,
            sphere_geometry.right_arm,
            sphere_geometry.normal_velocity,
            sphere_geometry.tangential_velocity,
            kinematics.angular_velocity[pairs.left_indices],
            kinematics.angular_velocity[pairs.right_indices],
            sphere_geometry.valid,
        )
        left = pairs.left_indices
        right = pairs.right_indices
        particle_contact = self.contact_model.evaluate(
            batch,
            remapped_history,
            keys.keys,
            keys.valid,
            continued,
            self.bodies.inverse_masses[left],
            self.bodies.inverse_masses[right],
            self.bodies.radii[left],
            self.bodies.radii[right],
            self.bodies.material_ids[left],
            self.bodies.material_ids[right],
            step_size,
            frame_tolerance=self.method.frame_tolerance,
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
        boundary_responses = tuple(
            evaluate_dem_barrier(
                barrier,
                self.bodies,
                kinematics,
                self.contact_model,
                history,
                step_size,
                time=time,
                args=args,
                normal_tolerance=self.method.distance_tolerance,
                frame_tolerance=self.method.frame_tolerance,
            )
            for barrier, history in zip(
                self.barriers, state.boundary_histories, strict=True
            )
        )
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
        gravity_force = self.bodies.particles.safe_masses[:, None] * self.gravity
        gravity_force = jnp.where(
            self.bodies.particles.active_mask[:, None], gravity_force, 0.0
        )
        gravity_load = self.bodies.load(gravity_force, jnp.zeros_like(pair_torque))
        external_value = self._external_load(time, kinematics, args)
        external_load = self.bodies.load(external_value.force, external_value.torque)
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
            continued_active & ~remapped_history.sliding & particle_contact.sliding,
            dtype=jnp.int32,
        )
        slip_to_stick = jnp.sum(
            continued_active & remapped_history.sliding & ~particle_contact.sliding,
            dtype=jnp.int32,
        )
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
                continued_boundary & ~old_history.sliding & new_history.sliding,
                dtype=jnp.int32,
            )
            slip_to_stick = slip_to_stick + jnp.sum(
                continued_boundary & old_history.sliding & ~new_history.sliding,
                dtype=jnp.int32,
            )
            contact_energy = contact_energy + jnp.sum(response.contact.elastic_energy)
        boundary_successful = (
            jnp.all(jnp.stack(tuple(value.successful for value in boundary_responses)))
            if boundary_responses
            else jnp.asarray(True)
        )
        reasons = jnp.zeros((), dtype=jnp.int32)
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
            ~(tree_allfinite(resolved_loads) & jnp.isfinite(contact_energy)),
            int(DEMRejectionReason.NONFINITE),
            0,
        ).astype(jnp.int32)
        successful = reasons == 0
        diagnostic_energy = self._ledger_view(
            state.energy,
            kinematics,
            contact_energy,
            boundary_wall_power,
        )
        diagnostics = self._diagnostics(
            kinematics,
            neighborhood_cache,
            particle_contact,
            boundary_responses,
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
        return DEMEvaluation(
            neighborhood,
            neighborhood_cache,
            particle_contact,
            boundary_responses,
            resolved_loads,
            diagnostics,
            contact_energy,
            births,
            deaths,
            boundary_wall_power,
            stick_to_slip,
            slip_to_stick,
            work.astype(jnp.int32),
            successful,
            reasons,
        )

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
        mobile = (self.bodies.particles.active_mask & ~self.bodies.fixed_mask)[:, None]
        half_velocity = state.kinematics.velocity + half * (
            self.bodies.inverse_masses[:, None] * state.loads.total.force
        )
        half_angular = state.kinematics.angular_velocity + half * (
            self.bodies.inverse_inertias[:, None] * state.loads.total.torque
        )
        half_velocity = jnp.where(mobile, half_velocity, 0.0)
        half_angular = jnp.where(mobile, half_angular, 0.0)
        next_position = state.kinematics.position + step_size * half_velocity
        next_position = jnp.where(mobile, next_position, state.kinematics.position)
        staged = DEMRuntimeState(
            RigidSphereKinematics(next_position, half_velocity, half_angular),
            state.particle_history,
            state.boundary_histories,
            state.neighborhood_cache,
            state.loads,
            state.energy,
        )
        evaluation = self.evaluate(time + step_size, staged, step_size, args)
        next_velocity = half_velocity + half * (
            self.bodies.inverse_masses[:, None] * evaluation.loads.total.force
        )
        next_angular = half_angular + half * (
            self.bodies.inverse_inertias[:, None] * evaluation.loads.total.torque
        )
        next_velocity = jnp.where(mobile, next_velocity, 0.0)
        next_angular = jnp.where(mobile, next_angular, 0.0)
        next_kinematics = RigidSphereKinematics(
            next_position, next_velocity, next_angular
        )
        energy = self._step_energy(state, next_kinematics, evaluation, step_size)
        candidate_energy = self._accumulated_energy(state.energy, energy)
        candidate = DEMRuntimeState(
            next_kinematics,
            evaluation.particle_contact.next_history,
            tuple(value.contact.next_history for value in evaluation.boundaries),
            evaluation.neighborhood_cache,
            evaluation.loads,
            candidate_energy,
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

    def step_restriction(self) -> DEMStepRestriction:
        active = self.bodies.particles.active_mask & ~self.bodies.fixed_mask
        masses = jnp.where(active, self.bodies.particles.safe_masses, jnp.inf)
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
        radius = self.bodies.radii
        if self.bodies.ambient_dimension == 2:
            measure = jnp.pi * radius**2
        else:
            measure = (4.0 / 3.0) * jnp.pi * radius**3
        density = self.bodies.particles.safe_masses / jnp.maximum(measure, 1.0e-30)
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
        boundaries,
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
        masses = self.bodies.particles.safe_masses
        linear_momentum = jnp.sum(masses[:, None] * kinematics.velocity, axis=0)
        angular_momentum = jnp.sum(
            sphere_lever_torque(
                kinematics.position,
                masses[:, None] * kinematics.velocity,
                self.bodies.ambient_dimension,
            )
            + self.bodies.inertias[:, None] * kinematics.angular_velocity,
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
            active_count,
            sticking_count,
            sliding_count,
            maximum_overlap,
            activation_margin,
            no_tension_margin,
            frame_margin,
            self.method.maximum_overlap_fraction - maximum_overlap,
            friction_margin,
            neighborhood_rebuilt,
            neighborhood_rebuild_count,
            neighborhood_certificate_margin,
            linear_momentum,
            angular_momentum,
            energy.kinetic_energy,
            elastic_energy,
            energy.gravity_potential,
            energy,
            net_force,
            net_torque,
            friction_defect,
            wall_defect,
            jnp.where(
                (rejection_reasons & int(DEMRejectionReason.PAIR_KEY)) != 0,
                1.0,
                0.0,
            ),
            successful,
            rejection_reasons,
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
    "DEMResolvedLoad",
    "DEMRejectionReason",
    "DEMRuntimeState",
    "DEMStateGeometry",
    "DEMStepEvaluation",
    "DEMStepEnergyLedger",
    "DEMStepRestriction",
    "PreparedSoftSphereDEMDynamics",
    "SoftSphereDEMMethodPlan",
]
