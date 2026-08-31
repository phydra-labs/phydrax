#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ..particle import ParticleDiscretization
from ..splatting import PreparedParticleGridSplat
from ._boundary import PrescribedGridVelocityPlan, PrescribedGridVelocityResult
from ._contact import MPMGridConstraintResult, RigidMPMContactPlan
from ._domain import MPMParticleDomainPlan
from ._fields import MPMNodalFieldPlan
from ._method import ExplicitMPMMethodPlan, MPMResourcePolicy
from ._multifield_dynamics import multifield_step_detailed
from ._phases import advance_grid_velocity, normalize_grid_momentum, update_deformation
from ._schedule import MUSLMPMSchedule, USFMPMSchedule
from ._storage import MPMActiveBlockPlan
from ._transfer import (
    apic_particle_angular_momentum,
    apic_particle_kinetic_energy,
    build_apic_route_payload,
    gather_apic,
    grid_angular_momentum,
)
from ._types import (
    MPMDiagnostics,
    MPMEnergyLedger,
    MPMGridState,
    MPMLimitingProcess,
    MPMParticleState,
    MPMPreparationEvidence,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
    MPMScheduleEvidence,
    MPMStepRestriction,
    MPMStepResult,
    MPMTransferEvidence,
)


ExternalMPMAcceleration = Callable[[Array, Array, Array, Any], ArrayLike]


def _relative_defect(left: Array, right: Array, /) -> Array:
    scale = jnp.maximum(1.0, jnp.maximum(jnp.linalg.norm(left), jnp.linalg.norm(right)))
    return jnp.linalg.norm(left - right) / scale


def _grid_kinetic(mass: Array, velocity: Array, active: Array, /) -> Array:
    terms = 0.5 * mass * jnp.sum(velocity * velocity, axis=-1)
    return compensated_sum(jnp.where(active, terms, 0.0).reshape((-1,)))


def _route_digest(state) -> Array:
    slots = jnp.arange(state.stencil.indices.shape[1], dtype=jnp.int64)[None, :]
    values = jnp.where(
        state.stencil.valid,
        state.stencil.indices.astype(jnp.int64) + 1,
        0,
    )
    return jnp.sum(values * (slots + 17))


class PreparedMPMDynamics(StrictModule, NonTrainableState):
    """Prepared fixed-capacity explicit USL/APIC material-point dynamics."""

    particles: ParticleDiscretization
    splat: PreparedParticleGridSplat
    method: ExplicitMPMMethodPlan
    material: Any
    particle_domain: MPMParticleDomainPlan
    boundary: PrescribedGridVelocityPlan | None
    contact: RigidMPMContactPlan | None
    nodal_fields: MPMNodalFieldPlan
    active_blocks: MPMActiveBlockPlan | None
    external_acceleration: ExternalMPMAcceleration | None
    external_acceleration_id: str | None = eqx.field(static=True)
    resource_policy: MPMResourcePolicy
    key: DiscretizationKey
    grid_coordinates: Array
    minimum_spacing: float = eqx.field(static=True)
    preparation: PreparationReport
    resource_evidence: MPMPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        splat: PreparedParticleGridSplat,
        method: ExplicitMPMMethodPlan,
        material: Any,
        particle_domain: MPMParticleDomainPlan,
        /,
        *,
        boundary: PrescribedGridVelocityPlan | None = None,
        contact: RigidMPMContactPlan | None = None,
        nodal_fields: MPMNodalFieldPlan | None = None,
        active_blocks: MPMActiveBlockPlan | None = None,
        external_acceleration: ExternalMPMAcceleration | None = None,
        external_acceleration_id: str | None = None,
        resource_policy: MPMResourcePolicy | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if not isinstance(splat, PreparedParticleGridSplat):
            raise TypeError("splat must be PreparedParticleGridSplat.")
        if not isinstance(method, ExplicitMPMMethodPlan):
            raise TypeError("method must be ExplicitMPMMethodPlan.")
        if not isinstance(particle_domain, MPMParticleDomainPlan):
            raise TypeError("particle_domain must be MPMParticleDomainPlan.")
        if boundary is not None and not isinstance(boundary, PrescribedGridVelocityPlan):
            raise TypeError("boundary must be PrescribedGridVelocityPlan or None.")
        if contact is not None and not isinstance(contact, RigidMPMContactPlan):
            raise TypeError("contact must be RigidMPMContactPlan or None.")
        fields = (
            MPMNodalFieldPlan(
                ("material",),
                np.zeros((particles.capacity,), dtype=np.int32),
            )
            if nodal_fields is None
            else nodal_fields
        )
        if not isinstance(fields, MPMNodalFieldPlan):
            raise TypeError("nodal_fields must be MPMNodalFieldPlan or None.")
        if fields.initial_particle_field_slots.shape != (particles.capacity,):
            raise ValueError("Nodal field slots must match particle capacity.")
        if fields.field_count > 1 and method.schedule.common_name != "usl-minus":
            raise ValueError("Initial multifield MPM supports USL-minus only.")
        if active_blocks is not None and not isinstance(
            active_blocks, MPMActiveBlockPlan
        ):
            raise TypeError("active_blocks must be MPMActiveBlockPlan or None.")
        if active_blocks is not None and active_blocks.grid_shape != splat.target_shape:
            raise ValueError("Active-block logical grid must match the splat target.")
        if splat.particles.prepared_id != particles.prepared_id:
            raise ValueError("MPM splat was prepared for a different particle support.")
        dimension = particles.ambient_dimension
        if dimension not in (2, 3) or particle_domain.dimension != dimension:
            raise ValueError("MPM dimensions must agree and be two or three.")
        if tuple(splat.layout.axis_entities) != ("point",) * dimension:
            raise ValueError("Explicit MPM requires a nodal tensor-grid target.")
        assignment_capabilities = splat.plan.assignment.capabilities
        if (
            not assignment_capabilities.partition_of_unity
            or assignment_capabilities.maximum_explicit_derivative_order < 1
            or assignment_capabilities.polynomial_reproduction_order < 1
            or not assignment_capabilities.apic_compatible
        ):
            raise ValueError(
                "Explicit APIC MPM requires conservative first-order assignment "
                "weights, physical gradients, and particle moments."
            )
        if splat.plan.boundary != "reject":
            raise ValueError("Closed-domain MPM requires splat boundary='reject'.")
        if material.dimension != dimension:
            raise ValueError("MPM material dimension does not match particle support.")
        expected_kinematics = (
            ("plane_strain", "plane_stress") if dimension == 2 else ("three_dimensional",)
        )
        if material.kinematics not in expected_kinematics:
            raise ValueError(
                "MPM material kinematics do not match the spatial dimension."
            )
        if external_acceleration is not None and not callable(external_acceleration):
            raise TypeError("external_acceleration must be callable or None.")
        if external_acceleration is None and external_acceleration_id is not None:
            raise ValueError("external_acceleration_id requires external acceleration.")
        if external_acceleration is not None and not external_acceleration_id:
            raise ValueError("External acceleration requires a stable non-empty ID.")

        axes = splat.plan.target.structured_axes
        coordinates = splat.layout.coordinates_by_axis
        spacings = []
        for axis_index, (axis, coordinate) in enumerate(
            zip(axes, coordinates, strict=True)
        ):
            values = np.asarray(coordinate, dtype=float)
            spacing = float(np.diff(values)[0])
            spacings.append(spacing)
            if bool(axis.periodic) != particle_domain.periodic[axis_index]:
                raise ValueError("MPM domain and grid periodic axes must match.")
            required_margin = (
                assignment_capabilities.maximum_support_radius_cells * spacing
            )
            if not axis.periodic:
                grid_bounds = np.asarray(axis.bounds, dtype=float)
                material_bounds = np.asarray(particle_domain.bounds)[:, axis_index]
                lower_gap = float(material_bounds[0] - grid_bounds[0])
                upper_gap = float(grid_bounds[1] - material_bounds[1])
                declared = particle_domain.support_margin[axis_index]
                if declared < required_margin or min(lower_gap, upper_gap) < declared:
                    raise ValueError(
                        "Nonperiodic MPM needs the assignment's complete declared "
                        "support halo."
                    )
        mesh = jnp.meshgrid(*coordinates, indexing="ij")
        grid_coordinates = jnp.stack(mesh, axis=-1).reshape((-1, dimension))
        if boundary is not None and boundary.mask.shape != splat.target_shape + (
            dimension,
        ):
            raise ValueError("Prescribed boundary layout must match the MPM nodal grid.")

        if boundary is not None and contact is not None:
            overlap = np.asarray(contact.prospective_mask(grid_coordinates)).reshape(
                splat.target_shape
            )
            if np.any(overlap[..., None] & np.asarray(boundary.mask)):
                raise ValueError(
                    "Prescribed velocity and rigid contact regions must be disjoint."
                )
        resource = MPMResourcePolicy() if resource_policy is None else resource_policy
        if not isinstance(resource, MPMResourcePolicy):
            raise TypeError("resource_policy must be MPMResourcePolicy or None.")
        itemsize = np.dtype(splat.plan.precision.evaluation_dtype).itemsize
        particle_count = particles.capacity
        grid_count = splat.target_size
        route_payload_width = 3 * dimension
        step_values = (
            splat.route_count * route_payload_width
            + grid_count * (1 + 6 * dimension)
            + particle_count * (3 * dimension * dimension + 5 * dimension + 8)
        )
        state_values = particle_count * (
            3 * dimension * dimension
            + 2 * dimension
            + 3
            + int(np.prod(material.state_shape))
        )
        workspace_bytes = step_values * itemsize
        state_bytes = state_values * itemsize
        resource.admit(
            step_workspace_bytes=workspace_bytes,
            state_bytes=state_bytes,
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.FIELD_TRANSFER,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            ),
            diagnostics=(
                "explicit updated-Lagrangian USL",
                "quadratic nodal B-spline assignment",
                "matched APIC momentum and affine reconstruction",
                "first-Piola reference-volume internal force",
                "fixed material population and temporal topology",
            ),
            resource_counts={
                "particle_capacity": particle_count,
                "grid_node_count": grid_count,
                "route_count": splat.route_count,
                "route_payload_width": route_payload_width,
                "step_workspace_bytes": workspace_bytes,
                "state_bytes": state_bytes,
            },
        )
        resource_evidence = MPMPreparationEvidence(
            particle_count,
            grid_count,
            splat.route_count,
            route_payload_width,
            workspace_bytes,
            state_bytes,
            preparation.report_id,
        )
        key = DiscretizationKey(
            "mpm",
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "background_grid"),
        )
        self.particles = particles
        self.splat = splat
        self.method = method
        self.material = material
        self.particle_domain = particle_domain
        self.boundary = boundary
        self.contact = contact
        self.nodal_fields = fields
        self.active_blocks = active_blocks
        self.external_acceleration = external_acceleration
        self.external_acceleration_id = external_acceleration_id
        self.resource_policy = resource
        self.key = key
        self.grid_coordinates = grid_coordinates
        self.minimum_spacing = min(spacings)
        self.preparation = preparation
        self.resource_evidence = resource_evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mpm-dynamics",
                "key": key.key_id,
                "particles": particles.prepared_id,
                "splat": splat.prepared_id,
                "method": method.method_id,
                "material": material.plan_id,
                "particle_domain": particle_domain.plan_id,
                "boundary": None if boundary is None else boundary.plan_id,
                "contact": None if contact is None else contact.plan_id,
                "nodal_fields": fields.plan_id,
                "active_blocks": (
                    None if active_blocks is None else active_blocks.plan_id
                ),
                "external_acceleration": external_acceleration_id,
                "resource": resource.policy_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def dimension(self) -> int:
        return self.particles.ambient_dimension

    @property
    def precision_evidence(self):
        return self.splat.precision_evidence

    @property
    def resource_evidence_id(self) -> str:
        return self.resource_evidence.evidence_id

    def _vector(self, name: str, value: ArrayLike) -> Array:
        array = self.splat.plan.precision.evaluation(value)
        expected = (self.particles.capacity, self.dimension)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        return array

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_volume: ArrayLike,
        arguments: Any,
        /,
        *,
        deformation_gradient: ArrayLike | None = None,
        affine_velocity: ArrayLike | None = None,
        material_state: ArrayLike | None = None,
        assignment_input: object = None,
        material_slots: ArrayLike | None = None,
        body_ids: ArrayLike | None = None,
        velocity_field_slots: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> MPMRuntimeState:
        position_ = self._vector("position", position)
        velocity_ = self._vector("velocity", velocity)
        count = self.particles.capacity
        dimension = self.dimension
        dtype = position_.dtype
        volume = jnp.asarray(reference_volume, dtype=dtype)
        if volume.shape != (count,):
            raise ValueError("reference_volume must have particle-capacity shape.")
        identity = jnp.broadcast_to(
            jnp.eye(dimension, dtype=dtype), (count, dimension, dimension)
        )
        deformation = (
            identity
            if deformation_gradient is None
            else jnp.asarray(deformation_gradient, dtype=dtype)
        )
        affine = (
            jnp.zeros_like(identity)
            if affine_velocity is None
            else jnp.asarray(affine_velocity, dtype=dtype)
        )
        if deformation.shape != identity.shape or affine.shape != identity.shape:
            raise ValueError("Initial F and C must have particle tensor shape.")
        history_shape = (count,) + tuple(self.material.state_shape)
        if material_state is None:
            history = self.material.initialize_state((count,), dtype)
        else:
            history = jnp.asarray(material_state, dtype=dtype)
        if history.shape != history_shape:
            raise ValueError(f"material_state must have shape {history_shape}.")
        active = self.particles.active_mask
        density = self.particles.safe_masses.astype(dtype) / jnp.where(
            active, volume, 1.0
        )
        response = self.material.evaluate(
            deformation,
            history,
            density,
            arguments.material_parameters,
            jnp.asarray(time, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        runtime_assignment = self.splat.plan.assignment.update_input(
            position_, deformation, assignment_input
        )
        routes = self.splat.build(position_, assignment_input=runtime_assignment)
        storage_state = (
            None if self.active_blocks is None else self.active_blocks.build(routes)
        )
        valid = (
            jnp.all((~active) | self.particle_domain.contains(position_))
            & routes.successful
            & (jnp.asarray(True) if storage_state is None else storage_state.successful)
            & jnp.all((~active) | (jnp.isfinite(volume) & (volume > 0.0)))
            & jnp.all((~active) | response.successful)
            & jnp.all((~active) | response.admissible)
        )
        particles = MPMParticleState(
            position_,
            velocity_,
            deformation,
            affine,
            volume,
            response.first_piola,
            response.reference_energy_density,
            response.maximum_wave_speed,
            response.trial_state,
        )
        checked = eqx.error_if(
            particles.position,
            ~valid | ~tree_allfinite(particles),
            "Initial MPM state is inadmissible.",
        )
        resolved_field_slots = (
            self.nodal_fields.initial_particle_field_slots
            if velocity_field_slots is None
            else velocity_field_slots
        )
        particles = eqx.tree_at(lambda value: value.position, particles, checked)
        return MPMRuntimeState(
            particles,
            jnp.asarray(time, dtype=dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(int(MPMRunStatus.SUCCESS), dtype=jnp.int32),
            0,
            runtime_assignment,
            material_slots,
            body_ids,
            resolved_field_slots,
            storage_state,
        )

    def _external(
        self, time: Array, state: MPMParticleState, arguments: Any
    ) -> tuple[Array, Array]:
        if self.external_acceleration is None:
            return jnp.zeros_like(state.position), jnp.asarray(True)
        value = jnp.asarray(
            self.external_acceleration(
                time,
                state.position,
                state.velocity,
                arguments.external_arguments,
            ),
            dtype=state.position.dtype,
        )
        if value.shape != state.position.shape:
            raise ValueError("External MPM acceleration must match particle positions.")
        active = self.particles.active_mask[:, None]
        finite = jnp.all(jnp.where(active, jnp.isfinite(value), True))
        return jnp.where(active, value, 0.0), finite

    def _apply_contact(self, velocity, mass, time, step_size, arguments):
        if self.contact is None:
            return MPMGridConstraintResult(
                velocity,
                jnp.zeros((self.dimension,), dtype=velocity.dtype),
                jnp.zeros((), dtype=velocity.dtype),
                jnp.zeros((), dtype=velocity.dtype),
                jnp.asarray(jnp.inf, dtype=velocity.dtype),
                jnp.zeros(mass.shape, dtype=bool),
                jnp.zeros(mass.shape, dtype=jnp.int32),
                jnp.asarray(True),
            )
        coordinates = self.grid_coordinates.reshape(
            self.splat.target_shape + (self.dimension,)
        )
        return self.contact.apply(
            coordinates,
            velocity,
            mass,
            time,
            step_size,
            arguments.external_arguments,
        )

    def _empty_grid(self, dtype) -> MPMGridState:
        scalar = jnp.zeros(
            (self.nodal_fields.field_count,) + self.splat.target_shape,
            dtype=dtype,
        )
        vector = jnp.zeros(
            (self.nodal_fields.field_count,)
            + self.splat.target_shape
            + (self.dimension,),
            dtype=dtype,
        )
        return MPMGridState(
            scalar, vector, vector, vector, vector, vector, scalar.astype(bool)
        )

    def _empty_diagnostics(self, state: MPMRuntimeState, route_state) -> MPMDiagnostics:
        dtype = state.particles.position.dtype
        vector_shape = (self.dimension,) if self.dimension == 3 else ()
        zero_vector = jnp.zeros((self.dimension,), dtype=dtype)
        zero_angular = jnp.zeros(vector_shape, dtype=dtype)
        transfer = MPMTransferEvidence(
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_vector,
            zero_vector,
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_angular,
            zero_angular,
            jnp.asarray(False),
            jnp.asarray(jnp.inf, dtype=dtype),
            zero_vector,
            jnp.max(jnp.abs(route_state.partition_sums - 1.0), initial=0.0),
            jnp.max(jnp.abs(route_state.gradient_sums), initial=0.0),
            jnp.max(jnp.abs(route_state.first_moments), initial=0.0),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=dtype),
            jnp.asarray(True),
            route_state.valid_route_count,
            _route_digest(route_state),
            jnp.asarray(False),
        )
        schedule = MPMScheduleEvidence(
            jnp.asarray(self.method.schedule.schedule_code, dtype=jnp.int32),
            jnp.asarray(isinstance(self.method.schedule, USFMPMSchedule)),
            jnp.asarray(isinstance(self.method.schedule, MUSLMPMSchedule)),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=jnp.int64),
            jnp.asarray(False),
        )
        energy = MPMEnergyLedger(*(jnp.zeros((), dtype=dtype) for _ in range(13)))
        return MPMDiagnostics(
            transfer,
            schedule,
            energy,
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(False),
        )

    def _rejected(
        self,
        state: MPMRuntimeState,
        route_state,
        step_size: Array,
        reason: Any,
        status: Any,
        /,
        *,
        grid: MPMGridState | None = None,
        restriction: MPMStepRestriction | None = None,
    ) -> MPMStepResult:
        dtype = state.particles.position.dtype
        rejected = MPMRuntimeState(
            state.particles,
            state.time,
            state.accepted_step,
            jnp.asarray(status, dtype=jnp.int32),
            state.topology_generation,
            state.assignment_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            state.storage_state,
        )
        restriction_ = (
            MPMStepRestriction(
                *(jnp.asarray(jnp.nan, dtype=dtype) for _ in range(8)),
                jnp.asarray(int(MPMLimitingProcess.NONE), dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=dtype),
            )
            if restriction is None
            else restriction
        )
        return MPMStepResult(
            rejected,
            rejected,
            self._empty_grid(dtype) if grid is None else grid,
            restriction_,
            self._empty_diagnostics(state, route_state),
            jnp.asarray(False),
            jnp.asarray(reason, dtype=jnp.int32),
            step_size,
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(False),
            restriction_.suggested_step,
        )

    def step_detailed(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: Any,
        /,
    ) -> MPMStepResult:
        if not isinstance(state, MPMRuntimeState):
            raise TypeError("state must be MPMRuntimeState.")
        dt = jnp.asarray(step_size, dtype=state.particles.position.dtype).reshape(())
        routes = self.splat.build(
            state.particles.position,
            assignment_input=state.assignment_input,
        )
        storage_state = (
            None
            if self.active_blocks is None
            else self.active_blocks.build(routes, state.storage_state)
        )
        active = self.particles.active_mask
        domain_ok = jnp.all(
            (~active) | self.particle_domain.contains(state.particles.position)
        )
        finite = tree_allfinite(state.particles) & jnp.isfinite(dt) & (dt > 0.0)
        route_ok = (
            routes.successful
            & ~jnp.any(routes.truncated_support_mask)
            & (jnp.asarray(True) if storage_state is None else storage_state.successful)
        )

        def invalid(_):
            reason = jnp.where(
                ~finite,
                int(MPMRejectionReason.NONFINITE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRejectionReason.DOMAIN),
                    int(MPMRejectionReason.ROUTE),
                ),
            )
            status = jnp.where(
                ~finite,
                int(MPMRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~domain_ok,
                    int(MPMRunStatus.DOMAIN_REJECTED),
                    int(MPMRunStatus.ROUTE_REJECTED),
                ),
            )
            return self._rejected(
                state,
                routes,
                dt,
                reason,
                status,
            )

        def execute(_):
            if self.nodal_fields.field_count > 1:
                return multifield_step_detailed(self, state, dt, arguments, routes)
            particle = state.particles
            mass = self.particles.safe_masses.astype(particle.position.dtype)
            acceleration_external, external_ok = self._external(
                state.time, particle, arguments
            )
            mass_result = self.splat.deposit_content(routes, mass)
            route_payload = build_apic_route_payload(
                routes,
                mass,
                particle.velocity,
                particle.affine_velocity,
                particle.reference_volume,
                particle.first_piola,
                particle.deformation_gradient,
                acceleration_external,
                active,
            )
            scattered = self.splat.scatter_route_payload(routes, route_payload)
            dimension = self.dimension
            grid_mass = mass_result.content
            grid_momentum = scattered.values[..., :dimension]
            internal_force = scattered.values[..., dimension : 2 * dimension]
            external_force = scattered.values[..., 2 * dimension :]
            normalized_grid = normalize_grid_momentum(
                grid_mass,
                grid_momentum,
                mass_tolerance_factor=self.method.mass_tolerance_factor,
            )
            grid_active = normalized_grid.active
            velocity_before = normalized_grid.velocity
            if storage_state is not None:
                grid_active = grid_active & storage_state.active_node_mask
                velocity_before = jnp.where(grid_active[..., None], velocity_before, 0.0)
                normalized_grid = eqx.tree_at(
                    lambda value: (value.active, value.velocity),
                    normalized_grid,
                    (grid_active, velocity_before),
                )
            density = mass / jnp.where(active, particle.reference_volume, 1.0)
            identity = jnp.broadcast_to(
                jnp.eye(dimension, dtype=particle.position.dtype),
                particle.deformation_gradient.shape,
            )
            schedule_p2g_success = jnp.asarray(True)
            if isinstance(self.method.schedule, USFMPMSchedule):
                first_gather = gather_apic(
                    routes,
                    velocity_before.reshape((self.splat.target_size, dimension)),
                    active,
                    self.method.transfer.maximum_condition,
                )
                scheduled_deformation = update_deformation(
                    particle.deformation_gradient,
                    first_gather.velocity_gradient,
                    dt,
                )
                scheduled_material = self.material.evaluate(
                    scheduled_deformation,
                    particle.material_state,
                    density,
                    arguments.material_parameters,
                    state.time + dt,
                    dt,
                )
                scheduled_payload = build_apic_route_payload(
                    routes,
                    mass,
                    particle.velocity,
                    particle.affine_velocity,
                    particle.reference_volume,
                    scheduled_material.first_piola,
                    scheduled_deformation,
                    acceleration_external,
                    active,
                )
                scheduled_scatter = self.splat.scatter_route_payload(
                    routes, scheduled_payload
                )
                internal_force = scheduled_scatter.values[..., dimension : 2 * dimension]
                external_force = scheduled_scatter.values[..., 2 * dimension :]
                schedule_p2g_success = (
                    first_gather.successful & scheduled_scatter.successful
                )
            grid_update = advance_grid_velocity(
                normalized_grid,
                internal_force,
                external_force,
                dt,
            )
            grid_acceleration = grid_update.acceleration
            wave_speed = (
                scheduled_material.maximum_wave_speed
                if isinstance(self.method.schedule, USFMPMSchedule)
                else particle.maximum_wave_speed
            )
            maximum_wave = jnp.max(jnp.where(active, wave_speed, 0.0), initial=0.0)
            maximum_velocity = jnp.max(
                jnp.where(
                    active,
                    jnp.sqrt(jnp.sum(particle.velocity * particle.velocity, axis=-1)),
                    0.0,
                ),
                initial=0.0,
            )
            maximum_acceleration = grid_update.maximum_acceleration
            tiny = jnp.finfo(grid_mass.dtype).tiny
            acoustic = (
                self.method.acoustic_cfl
                * self.minimum_spacing
                / jnp.maximum(maximum_wave, tiny)
            )
            advective = jnp.where(
                maximum_velocity > 0.0,
                self.method.advective_cfl
                * self.minimum_spacing
                / jnp.maximum(maximum_velocity, tiny),
                jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
            )
            force_limit = jnp.where(
                maximum_acceleration > 0.0,
                self.method.force_cfl
                * jnp.sqrt(
                    self.minimum_spacing / jnp.maximum(maximum_acceleration, tiny)
                ),
                jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
            )
            infinite = jnp.asarray(jnp.inf, dtype=grid_mass.dtype)
            material_limit = (
                jnp.min(
                    jnp.where(
                        active,
                        scheduled_material.suggested_step,
                        jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
                    )
                )
                if isinstance(self.method.schedule, USFMPMSchedule)
                else infinite
            )
            limits = jnp.stack(
                (
                    acoustic,
                    advective,
                    force_limit,
                    infinite,
                    material_limit,
                    infinite,
                    infinite,
                )
            )
            selected = jnp.min(limits)
            limiting_process = jnp.argmin(limits).astype(jnp.int32) + 1
            restriction = MPMStepRestriction(
                acoustic,
                advective,
                force_limit,
                infinite,
                material_limit,
                infinite,
                infinite,
                selected,
                limiting_process,
                selected,
            )
            stable = external_ok & jnp.isfinite(selected) & (dt <= selected)
            grid_before = MPMGridState(
                grid_mass[None, ...],
                grid_momentum[None, ...],
                velocity_before[None, ...],
                internal_force[None, ...],
                external_force[None, ...],
                velocity_before[None, ...],
                grid_active[None, ...],
            )

            def unstable(_):
                reason = jnp.where(
                    external_ok,
                    int(MPMRejectionReason.STABILITY),
                    int(MPMRejectionReason.NONFINITE),
                )
                status = jnp.where(
                    external_ok,
                    int(MPMRunStatus.STABILITY_LIMIT_EXCEEDED),
                    int(MPMRunStatus.NONFINITE_STATE),
                )
                return self._rejected(
                    state,
                    routes,
                    dt,
                    reason,
                    status,
                    grid=grid_before,
                    restriction=restriction,
                )

            def advance(_):
                velocity_trial = grid_update.velocity
                contact_result = self._apply_contact(
                    velocity_trial, grid_mass, state.time, dt, arguments
                )
                if self.boundary is None:
                    boundary_result = PrescribedGridVelocityResult(
                        contact_result.velocity,
                        jnp.zeros((dimension,), dtype=velocity_trial.dtype),
                        jnp.zeros((), dtype=velocity_trial.dtype),
                        jnp.asarray(True),
                    )
                else:
                    boundary_result = self.boundary.apply(
                        contact_result.velocity, grid_mass, dt
                    )
                grid_after = boundary_result.velocity
                gathered = gather_apic(
                    routes,
                    grid_after.reshape((self.splat.target_size, dimension)),
                    active,
                    self.method.transfer.maximum_condition,
                )
                candidate_position = particle.position + dt * gathered.velocity
                second_mass_defect = jnp.asarray(0.0, dtype=grid_mass.dtype)
                second_momentum_defect = jnp.asarray(0.0, dtype=grid_mass.dtype)
                second_constraint_work = jnp.asarray(0.0, dtype=grid_mass.dtype)
                second_contact_work = jnp.asarray(0.0, dtype=grid_mass.dtype)
                second_contact_dissipation = jnp.asarray(0.0, dtype=grid_mass.dtype)
                second_contact_limit = jnp.asarray(jnp.inf, dtype=grid_mass.dtype)
                second_successful = jnp.asarray(True)
                kinematic_gradient = gathered.velocity_gradient
                if isinstance(self.method.schedule, MUSLMPMSchedule):
                    second_mass = self.splat.deposit_content(routes, mass)
                    second_particle_momentum = mass[:, None] * gathered.velocity
                    second_momentum = self.splat.deposit_content(
                        routes, second_particle_momentum
                    )
                    second_grid = normalize_grid_momentum(
                        second_mass.content,
                        second_momentum.content,
                        mass_tolerance_factor=self.method.mass_tolerance_factor,
                    )
                    second_contact = self._apply_contact(
                        second_grid.velocity,
                        second_mass.content,
                        state.time,
                        dt,
                        arguments,
                    )
                    if self.boundary is None:
                        second_boundary = PrescribedGridVelocityResult(
                            second_contact.velocity,
                            jnp.zeros((dimension,), dtype=grid_mass.dtype),
                            jnp.zeros((), dtype=grid_mass.dtype),
                            jnp.asarray(True),
                        )
                    else:
                        second_boundary = self.boundary.apply(
                            second_contact.velocity, second_mass.content, dt
                        )
                    second_contact_work = second_contact.work
                    second_contact_dissipation = second_contact.dissipation
                    second_contact_limit = second_contact.contact_step_limit
                    second_gather = gather_apic(
                        routes,
                        second_boundary.velocity.reshape(
                            (self.splat.target_size, dimension)
                        ),
                        active,
                        self.method.transfer.maximum_condition,
                    )
                    kinematic_gradient = second_gather.velocity_gradient
                    second_mass_defect = (
                        second_mass.balance.maximum_absolute_balance_defect
                        / jnp.maximum(
                            1.0,
                            jnp.abs(second_mass.balance.active_source_total),
                        )
                    )
                    second_source_momentum = compensated_sum(
                        jnp.where(active[:, None], second_particle_momentum, 0.0),
                        axis=0,
                    )
                    second_target_momentum = compensated_sum(
                        second_momentum.content.reshape((-1, dimension)), axis=0
                    )
                    second_momentum_defect = _relative_defect(
                        second_source_momentum, second_target_momentum
                    )
                    second_constraint_work = second_boundary.work
                    second_successful = (
                        second_mass.successful
                        & second_momentum.successful
                        & second_contact.successful
                        & second_boundary.successful
                        & second_gather.successful
                    )
                if isinstance(self.method.schedule, USFMPMSchedule):
                    candidate_deformation = scheduled_deformation
                    material = scheduled_material
                else:
                    candidate_deformation = update_deformation(
                        particle.deformation_gradient,
                        kinematic_gradient,
                        dt,
                    )
                    material = self.material.evaluate(
                        candidate_deformation,
                        particle.material_state,
                        density,
                        arguments.material_parameters,
                        state.time + dt,
                        dt,
                    )
                determinant = solve_small_linear(
                    SmallLinearSolvePlan(dimension),
                    candidate_deformation,
                    identity,
                ).determinant
                jacobian_ok = jnp.all(
                    (~active) | (jnp.isfinite(determinant) & (determinant > 0.0))
                )
                material_ok = jnp.all(
                    (~active) | (material.successful & material.admissible)
                )
                candidate_material_limit = jnp.min(
                    jnp.where(
                        active,
                        material.suggested_step,
                        jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
                    )
                )
                candidate_material_limit = jnp.where(
                    jnp.isfinite(candidate_material_limit)
                    & (candidate_material_limit > 0.0),
                    candidate_material_limit,
                    jnp.asarray(jnp.inf, dtype=grid_mass.dtype),
                )
                candidate_contact_limit = jnp.minimum(
                    contact_result.contact_step_limit, second_contact_limit
                )
                final_selected = jnp.minimum(
                    restriction.selected,
                    jnp.minimum(candidate_material_limit, candidate_contact_limit),
                )
                final_limiting = jnp.where(
                    candidate_contact_limit
                    < jnp.minimum(restriction.selected, candidate_material_limit),
                    int(MPMLimitingProcess.CONTACT),
                    jnp.where(
                        candidate_material_limit < restriction.selected,
                        int(MPMLimitingProcess.MATERIAL),
                        restriction.limiting_process,
                    ),
                ).astype(jnp.int32)
                final_restriction = MPMStepRestriction(
                    restriction.acoustic,
                    restriction.advective,
                    restriction.force,
                    candidate_contact_limit,
                    candidate_material_limit,
                    restriction.source_domain_motion,
                    restriction.nonlinear,
                    final_selected,
                    final_limiting,
                    final_selected,
                )
                material_step_ok = dt <= candidate_material_limit
                contact_step_ok = dt <= candidate_contact_limit
                candidate_particle = MPMParticleState(
                    candidate_position,
                    gathered.velocity,
                    candidate_deformation,
                    gathered.affine_velocity,
                    particle.reference_volume,
                    material.first_piola,
                    material.reference_energy_density,
                    material.maximum_wave_speed,
                    material.trial_state,
                )
                finite_candidate = tree_allfinite(candidate_particle)
                successful = (
                    gathered.successful
                    & schedule_p2g_success
                    & second_successful
                    & contact_result.successful
                    & boundary_result.successful
                    & material_ok
                    & material_step_ok
                    & contact_step_ok
                    & jacobian_ok
                    & finite_candidate
                    & mass_result.successful
                    & scattered.successful
                )
                reasons = jnp.zeros((), dtype=jnp.int32)
                reasons = reasons | jnp.where(
                    gathered.successful & schedule_p2g_success & second_successful,
                    0,
                    int(MPMRejectionReason.APIC_MOMENT),
                ).astype(jnp.int32)
                reasons = reasons | jnp.where(
                    material_ok & material_step_ok,
                    0,
                    int(MPMRejectionReason.MATERIAL),
                ).astype(jnp.int32)
                reasons = reasons | jnp.where(
                    contact_result.successful & contact_step_ok,
                    0,
                    int(MPMRejectionReason.CONTACT),
                ).astype(jnp.int32)
                reasons = reasons | jnp.where(
                    jacobian_ok,
                    0,
                    int(MPMRejectionReason.JACOBIAN),
                ).astype(jnp.int32)
                reasons = reasons | jnp.where(
                    finite_candidate & boundary_result.successful,
                    0,
                    int(MPMRejectionReason.NONFINITE),
                ).astype(jnp.int32)
                status = jnp.where(
                    successful,
                    int(MPMRunStatus.SUCCESS),
                    jnp.where(
                        ~(gathered.successful & schedule_p2g_success & second_successful),
                        int(MPMRunStatus.APIC_MOMENT_FAILED),
                        jnp.where(
                            ~contact_result.successful | ~contact_step_ok,
                            int(MPMRunStatus.CONTACT_REJECTED),
                            jnp.where(
                                ~material_ok | ~material_step_ok | ~jacobian_ok,
                                int(MPMRunStatus.MATERIAL_REJECTED),
                                int(MPMRunStatus.NONFINITE_STATE),
                            ),
                        ),
                    ),
                ).astype(jnp.int32)
                candidate_assignment = self.splat.plan.assignment.update_input(
                    candidate_position,
                    candidate_deformation,
                    state.assignment_input,
                )
                accepted_assignment = tree_where(
                    successful, candidate_assignment, state.assignment_input
                )
                accepted_storage = tree_where(
                    successful, storage_state, state.storage_state
                )
                accepted_particle = tree_where(successful, candidate_particle, particle)
                accepted_state = MPMRuntimeState(
                    accepted_particle,
                    jnp.where(successful, state.time + dt, state.time),
                    jnp.where(successful, state.accepted_step + 1, state.accepted_step),
                    status,
                    state.topology_generation,
                    accepted_assignment,
                    state.material_slots,
                    state.body_ids,
                    state.velocity_field_slots,
                    accepted_storage,
                )
                candidate_state = MPMRuntimeState(
                    candidate_particle,
                    state.time + dt,
                    state.accepted_step + 1,
                    status,
                    state.topology_generation,
                    candidate_assignment,
                    state.material_slots,
                    state.body_ids,
                    state.velocity_field_slots,
                    storage_state,
                )
                particle_mass = compensated_sum(jnp.where(active, mass, 0.0))
                target_mass = compensated_sum(grid_mass.reshape((-1,)))
                particle_momentum = compensated_sum(
                    jnp.where(active[:, None], mass[:, None] * particle.velocity, 0.0),
                    axis=0,
                )
                target_momentum = compensated_sum(
                    grid_momentum.reshape((-1, dimension)), axis=0
                )
                particle_angular = apic_particle_angular_momentum(
                    particle.position,
                    particle.velocity,
                    particle.affine_velocity,
                    mass,
                    routes,
                    active,
                )
                target_angular = grid_angular_momentum(
                    self.grid_coordinates,
                    grid_momentum.reshape((-1, dimension)),
                    grid_active.reshape((-1,)),
                )
                angular_valid = jnp.asarray(not any(self.particle_domain.periodic))
                angular_defect = jnp.where(
                    angular_valid,
                    _relative_defect(particle_angular, target_angular),
                    0.0,
                )
                net_internal = compensated_sum(
                    internal_force.reshape((-1, dimension)), axis=0
                )
                maximum_condition = jnp.max(
                    jnp.where(active, gathered.condition_estimate, 0.0), initial=0.0
                )
                transfer_success = (
                    mass_result.balance.closed_domain_conservation_valid
                    & gathered.successful
                    & (_relative_defect(particle_momentum, target_momentum) <= 1.0e-10)
                    & ((~angular_valid) | (angular_defect <= 1.0e-10))
                )
                transfer = MPMTransferEvidence(
                    particle_mass,
                    target_mass,
                    jnp.abs(particle_mass - target_mass)
                    / jnp.maximum(1.0, jnp.abs(particle_mass)),
                    particle_momentum,
                    target_momentum,
                    _relative_defect(particle_momentum, target_momentum),
                    particle_angular,
                    target_angular,
                    angular_valid,
                    angular_defect,
                    net_internal,
                    mass_result.balance.maximum_partition_defect,
                    jnp.max(jnp.abs(routes.gradient_sums), initial=0.0),
                    jnp.max(jnp.abs(routes.first_moments), initial=0.0),
                    maximum_condition,
                    jnp.sum(grid_active, dtype=jnp.int32),
                    jnp.zeros((), dtype=grid_mass.dtype),
                    jnp.asarray(True),
                    routes.valid_route_count,
                    _route_digest(routes),
                    transfer_success,
                )
                particle_kinetic_before = apic_particle_kinetic_energy(
                    mass,
                    particle.velocity,
                    particle.affine_velocity,
                    routes.second_moments,
                    active,
                )
                particle_kinetic_after = apic_particle_kinetic_energy(
                    mass,
                    candidate_particle.velocity,
                    candidate_particle.affine_velocity,
                    routes.second_moments,
                    active,
                )
                grid_kinetic_before = _grid_kinetic(
                    grid_mass, velocity_before, grid_active
                )
                grid_kinetic_after = _grid_kinetic(grid_mass, grid_after, grid_active)
                material_before = compensated_sum(
                    jnp.where(
                        active,
                        particle.reference_volume * particle.reference_energy_density,
                        0.0,
                    )
                )
                material_after = compensated_sum(
                    jnp.where(
                        active,
                        particle.reference_volume * material.reference_energy_density,
                        0.0,
                    )
                )
                average_velocity = 0.5 * (particle.velocity + candidate_particle.velocity)
                external_work = dt * compensated_sum(
                    jnp.where(
                        active,
                        mass * jnp.sum(acceleration_external * average_velocity, axis=-1),
                        0.0,
                    )
                )
                total_before = particle_kinetic_before + material_before
                total_after = particle_kinetic_after + material_after
                boundary_work = boundary_result.work + second_constraint_work
                contact_work = contact_result.work + second_contact_work
                contact_dissipation = (
                    contact_result.dissipation + second_contact_dissipation
                )
                plastic_dissipation = compensated_sum(
                    jnp.where(
                        active,
                        particle.reference_volume * material.dissipation_increment,
                        0.0,
                    )
                )
                balance_defect = (
                    total_after
                    + plastic_dissipation
                    - total_before
                    - external_work
                    - boundary_work
                    - contact_work
                )
                energy = MPMEnergyLedger(
                    particle_kinetic_before,
                    grid_kinetic_before,
                    grid_kinetic_after,
                    particle_kinetic_after,
                    material_before,
                    material_after,
                    external_work,
                    boundary_work,
                    contact_work,
                    contact_dissipation,
                    plastic_dissipation,
                    jnp.zeros((), dtype=grid_mass.dtype),
                    balance_defect,
                )
                schedule = MPMScheduleEvidence(
                    jnp.asarray(self.method.schedule.schedule_code, dtype=jnp.int32),
                    jnp.asarray(isinstance(self.method.schedule, USFMPMSchedule)),
                    jnp.asarray(isinstance(self.method.schedule, MUSLMPMSchedule)),
                    second_mass_defect,
                    second_momentum_defect,
                    second_constraint_work,
                    _route_digest(routes)
                    + jnp.asarray(
                        self.method.schedule.schedule_code * 104729, dtype=jnp.int64
                    ),
                    schedule_p2g_success & second_successful,
                )
                diagnostics = MPMDiagnostics(
                    transfer,
                    schedule,
                    energy,
                    jnp.min(jnp.where(active, determinant, jnp.inf)),
                    jnp.max(jnp.where(active, determinant, 0.0)),
                    material_ok,
                    finite_candidate,
                )
                grid = MPMGridState(
                    grid_mass[None, ...],
                    grid_momentum[None, ...],
                    velocity_before[None, ...],
                    internal_force[None, ...],
                    external_force[None, ...],
                    grid_after[None, ...],
                    grid_active[None, ...],
                )
                return MPMStepResult(
                    candidate_state,
                    accepted_state,
                    grid,
                    final_restriction,
                    diagnostics,
                    successful,
                    reasons,
                    dt,
                    final_selected - dt,
                    ~material_ok | ~material_step_ok,
                    jnp.where(
                        material_ok & material_step_ok,
                        final_selected,
                        jnp.minimum(final_selected, 0.5 * dt),
                    ),
                )

            return jax.lax.cond(stable, advance, unstable, operand=None)

        return jax.lax.cond(finite & domain_ok & route_ok, execute, invalid, operand=None)


__all__ = ["ExternalMPMAcceleration", "PreparedMPMDynamics"]
