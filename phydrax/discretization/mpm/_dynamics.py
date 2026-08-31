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
import opt_einsum as oe
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
from ..splatting import PreparedParticleGridSplat, TensorBSplineSplatAssignment
from ._boundary import PrescribedGridVelocityPlan, PrescribedGridVelocityResult
from ._domain import MPMParticleDomainPlan
from ._method import ExplicitMPMMethodPlan, MPMResourcePolicy
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
    MPMParticleState,
    MPMPreparationEvidence,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
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
        if splat.particles.prepared_id != particles.prepared_id:
            raise ValueError("MPM splat was prepared for a different particle support.")
        dimension = particles.ambient_dimension
        if dimension not in (2, 3) or particle_domain.dimension != dimension:
            raise ValueError("MPM dimensions must agree and be two or three.")
        if tuple(splat.layout.axis_entities) != ("point",) * dimension:
            raise ValueError("Explicit MPM requires a nodal tensor-grid target.")
        if not isinstance(splat.plan.assignment, TensorBSplineSplatAssignment):
            raise TypeError("Explicit MPM requires TensorBSplineSplatAssignment.")
        if splat.plan.assignment.degree != 2:
            raise ValueError("The qualified MPM baseline requires quadratic B-splines.")
        if splat.plan.boundary != "reject":
            raise ValueError("Closed-domain MPM requires splat boundary='reject'.")
        if material.dimension != dimension:
            raise ValueError("MPM material dimension does not match particle support.")
        expected_kinematics = "plane_strain" if dimension == 2 else "three_dimensional"
        if material.kinematics != expected_kinematics:
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
            required_margin = 1.5 * spacing
            if not axis.periodic:
                grid_bounds = np.asarray(axis.bounds, dtype=float)
                material_bounds = np.asarray(particle_domain.bounds)[:, axis_index]
                lower_gap = float(material_bounds[0] - grid_bounds[0])
                upper_gap = float(grid_bounds[1] - material_bounds[1])
                declared = particle_domain.support_margin[axis_index]
                if declared < required_margin or min(lower_gap, upper_gap) < declared:
                    raise ValueError(
                        "Nonperiodic quadratic MPM needs a complete declared support halo."
                    )
        mesh = jnp.meshgrid(*coordinates, indexing="ij")
        grid_coordinates = jnp.stack(mesh, axis=-1).reshape((-1, dimension))
        if boundary is not None and boundary.mask.shape != splat.target_shape + (
            dimension,
        ):
            raise ValueError("Prescribed boundary layout must match the MPM nodal grid.")

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
            if tuple(self.material.state_shape) != (0,):
                raise ValueError(
                    "Stateful MPM materials require explicit initial history."
                )
            history = jnp.empty(history_shape, dtype=dtype)
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
        routes = self.splat.build(position_)
        valid = (
            jnp.all((~active) | self.particle_domain.contains(position_))
            & routes.successful
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
        particles = eqx.tree_at(lambda value: value.position, particles, checked)
        return MPMRuntimeState(
            particles,
            jnp.asarray(time, dtype=dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(int(MPMRunStatus.SUCCESS), dtype=jnp.int32),
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

    def _empty_grid(self, dtype) -> MPMGridState:
        scalar = jnp.zeros(self.splat.target_shape, dtype=dtype)
        vector = jnp.zeros(self.splat.target_shape + (self.dimension,), dtype=dtype)
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
            route_state.valid_route_count,
            _route_digest(route_state),
            jnp.asarray(False),
        )
        energy = MPMEnergyLedger(*(jnp.zeros((), dtype=dtype) for _ in range(9)))
        return MPMDiagnostics(
            transfer,
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
        )
        restriction_ = (
            MPMStepRestriction(*(jnp.asarray(jnp.nan, dtype=dtype) for _ in range(4)))
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
        routes = self.splat.build(state.particles.position)
        active = self.particles.active_mask
        domain_ok = jnp.all(
            (~active) | self.particle_domain.contains(state.particles.position)
        )
        finite = tree_allfinite(state.particles) & jnp.isfinite(dt) & (dt > 0.0)
        route_ok = routes.successful & ~jnp.any(routes.truncated_support_mask)

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
            maximum_mass = jnp.max(grid_mass, initial=0.0)
            mass_tolerance = (
                self.method.mass_tolerance_factor
                * jnp.finfo(grid_mass.dtype).eps
                * jnp.maximum(maximum_mass, 1.0)
            )
            grid_active = grid_mass > mass_tolerance
            denominator = jnp.where(grid_active, grid_mass, 1.0)
            velocity_before = jnp.where(
                grid_active[..., None], grid_momentum / denominator[..., None], 0.0
            )
            total_force = internal_force + external_force
            grid_acceleration = jnp.where(
                grid_active[..., None], total_force / denominator[..., None], 0.0
            )
            maximum_wave = jnp.max(
                jnp.where(active, particle.maximum_wave_speed, 0.0), initial=0.0
            )
            maximum_velocity = jnp.max(
                jnp.where(
                    active,
                    jnp.sqrt(jnp.sum(particle.velocity * particle.velocity, axis=-1)),
                    0.0,
                ),
                initial=0.0,
            )
            maximum_acceleration = jnp.max(
                jnp.where(
                    grid_active,
                    jnp.sqrt(jnp.sum(grid_acceleration * grid_acceleration, axis=-1)),
                    0.0,
                ),
                initial=0.0,
            )
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
            selected = jnp.minimum(acoustic, jnp.minimum(advective, force_limit))
            restriction = MPMStepRestriction(acoustic, advective, force_limit, selected)
            stable = external_ok & jnp.isfinite(selected) & (dt <= selected)
            grid_before = MPMGridState(
                grid_mass,
                grid_momentum,
                velocity_before,
                internal_force,
                external_force,
                velocity_before,
                grid_active,
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
                velocity_trial = velocity_before + dt * grid_acceleration
                if self.boundary is None:
                    boundary_result = PrescribedGridVelocityResult(
                        velocity_trial,
                        jnp.zeros((dimension,), dtype=velocity_trial.dtype),
                        jnp.zeros((), dtype=velocity_trial.dtype),
                        jnp.asarray(True),
                    )
                else:
                    boundary_result = self.boundary.apply(velocity_trial, grid_mass, dt)
                grid_after = boundary_result.velocity
                gathered = gather_apic(
                    routes,
                    grid_after.reshape((self.splat.target_size, dimension)),
                    active,
                    self.method.transfer.maximum_condition,
                )
                candidate_position = particle.position + dt * gathered.velocity
                identity = jnp.broadcast_to(
                    jnp.eye(dimension, dtype=particle.position.dtype),
                    particle.deformation_gradient.shape,
                )
                candidate_deformation = oe.contract(
                    "pij,pjk->pik",
                    identity + dt * gathered.velocity_gradient,
                    particle.deformation_gradient,
                )
                density = mass / jnp.where(active, particle.reference_volume, 1.0)
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
                    & boundary_result.successful
                    & material_ok
                    & jacobian_ok
                    & finite_candidate
                    & mass_result.successful
                    & scattered.successful
                )
                reasons = jnp.zeros((), dtype=jnp.int32)
                reasons = reasons | jnp.where(
                    gathered.successful,
                    0,
                    int(MPMRejectionReason.APIC_MOMENT),
                ).astype(jnp.int32)
                reasons = reasons | jnp.where(
                    material_ok,
                    0,
                    int(MPMRejectionReason.MATERIAL),
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
                        ~gathered.successful,
                        int(MPMRunStatus.APIC_MOMENT_FAILED),
                        jnp.where(
                            ~material_ok | ~jacobian_ok,
                            int(MPMRunStatus.MATERIAL_REJECTED),
                            int(MPMRunStatus.NONFINITE_STATE),
                        ),
                    ),
                ).astype(jnp.int32)
                accepted_particle = tree_where(successful, candidate_particle, particle)
                accepted_state = MPMRuntimeState(
                    accepted_particle,
                    jnp.where(successful, state.time + dt, state.time),
                    jnp.where(successful, state.accepted_step + 1, state.accepted_step),
                    status,
                )
                candidate_state = MPMRuntimeState(
                    candidate_particle,
                    state.time + dt,
                    state.accepted_step + 1,
                    status,
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
                balance_defect = (
                    total_after - total_before - external_work - boundary_result.work
                )
                energy = MPMEnergyLedger(
                    particle_kinetic_before,
                    grid_kinetic_before,
                    grid_kinetic_after,
                    particle_kinetic_after,
                    material_before,
                    material_after,
                    external_work,
                    boundary_result.work,
                    balance_defect,
                )
                diagnostics = MPMDiagnostics(
                    transfer,
                    energy,
                    jnp.min(jnp.where(active, determinant, jnp.inf)),
                    jnp.max(jnp.where(active, determinant, 0.0)),
                    material_ok,
                    finite_candidate,
                )
                grid = MPMGridState(
                    grid_mass,
                    grid_momentum,
                    velocity_before,
                    internal_force,
                    external_force,
                    grid_after,
                    grid_active,
                )
                return MPMStepResult(
                    candidate_state,
                    accepted_state,
                    grid,
                    restriction,
                    diagnostics,
                    successful,
                    reasons,
                    dt,
                    selected - dt,
                )

            return jax.lax.cond(stable, advance, unstable, operand=None)

        return jax.lax.cond(finite & domain_ok & route_ok, execute, invalid, operand=None)


__all__ = ["ExternalMPMAcceleration", "PreparedMPMDynamics"]
