#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from ._core import ParticleDiscretization
from ._graph import particle_graph_view
from ._neighborhood import (
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import (
    particle_pair_geometry,
    ParticlePairGeometry,
    scatter_pair_exchange,
    scatter_pair_sum,
)
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._smoothing import AbstractSPHSmoothingKernel


ParticleDifferentiabilityPolicy: TypeAlias = Literal["branchwise"]
ExternalParticlePotential = Callable[[Array, Array, Any], ArrayLike]


class BarotropicSPHMethodPlan(StrictModule, NonTrainableState):
    """Fixed-h conservative SPH with summation density."""

    kernel: AbstractSPHSmoothingKernel
    smoothing_length: float = eqx.field(static=True)
    acoustic_cfl: float = eqx.field(static=True)
    force_cfl: float = eqx.field(static=True)
    differentiability: ParticleDifferentiabilityPolicy = eqx.field(static=True)
    key: DiscretizationKey
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractSPHSmoothingKernel,
        smoothing_length: float,
        /,
        *,
        acoustic_cfl: float = 0.25,
        force_cfl: float = 0.25,
        name: str = "barotropic-sph",
        method_id: str | None = None,
    ):
        if not isinstance(kernel, AbstractSPHSmoothingKernel):
            raise TypeError("kernel must be an AbstractSPHSmoothingKernel.")
        smoothing = float(smoothing_length)
        acoustic = float(acoustic_cfl)
        force = float(force_cfl)
        if (
            not np.isfinite(smoothing)
            or smoothing <= 0.0
            or not np.isfinite(acoustic)
            or acoustic <= 0.0
            or not np.isfinite(force)
            or force <= 0.0
        ):
            raise ValueError(
                "SPH smoothing length and CFL coefficients must be finite and positive."
            )
        key = DiscretizationKey(
            name,
            DiscretizationRole.RESIDUAL,
            domain_labels=("material_point", "barotropic_fluid"),
        )
        self.kernel = kernel
        self.smoothing_length = smoothing
        self.acoustic_cfl = acoustic
        self.force_cfl = force
        self.differentiability = "branchwise"
        self.key = key
        self.method_id = resolved_identifier(
            "method_id",
            method_id,
            {
                "kind": "barotropic-sph-method-plan",
                "kernel": kernel.kernel_id,
                "smoothing_length": smoothing,
                "acoustic_cfl": acoustic,
                "force_cfl": force,
                "differentiability": "branchwise",
                "key": key.key_id,
            },
        )


class BarotropicSPHStepRestriction(StrictModule):
    """Acoustic and force restrictions for one particle state."""

    acoustic: Array
    force: Array
    selected: Array


class BarotropicSPHDiagnostics(StrictModule):
    """Conservation, admissibility, and neighborhood evidence."""

    density_minimum: Array
    density_maximum: Array
    neighbor_count_minimum: Array
    neighbor_count_maximum: Array
    total_mass: Array
    linear_momentum: Array
    angular_momentum: Array
    kinetic_energy: Array
    internal_energy: Array
    external_potential_energy: Array
    total_energy: Array
    net_internal_force: Array
    net_internal_torque: Array
    admissible: Array
    active_pairs: Array


class _BarotropicSPHEvaluation(StrictModule):
    position: Array
    neighborhood: ParticleNeighborhoodState
    geometry: ParticlePairGeometry
    physical_pairs: Array
    density: Array
    pressure: Array


class PreparedBarotropicSPHDynamics(StrictModule, NonTrainableState):
    """Pure fixed-h conservative SPH Hamiltonian ingredients."""

    particles: ParticleDiscretization
    neighborhood: AbstractPreparedParticleNeighborhood
    method: BarotropicSPHMethodPlan
    material: Any
    external_potential: ExternalParticlePotential | None
    external_potential_id: str | None = eqx.field(static=True)
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        neighborhood: AbstractPreparedParticleNeighborhood,
        method: BarotropicSPHMethodPlan,
        material: Any,
        /,
        *,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        external_potential: ExternalParticlePotential | None = None,
        external_potential_id: str | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError(
                "neighborhood must be an AbstractPreparedParticleNeighborhood."
            )
        if not isinstance(method, BarotropicSPHMethodPlan):
            raise TypeError("method must be a BarotropicSPHMethodPlan.")
        if neighborhood.particle_discretization_id != particles.prepared_id:
            raise ValueError(
                "Particle neighborhood was prepared for a different support."
            )
        if method.kernel.dimension != particles.ambient_dimension:
            raise ValueError("SPH kernel dimension does not match particle support.")
        if external_potential is not None and not callable(external_potential):
            raise TypeError("external_potential must be callable or None.")
        if external_potential is None and external_potential_id is not None:
            raise ValueError("external_potential_id requires an external potential.")
        if external_potential is not None and not external_potential_id:
            raise ValueError("An external potential requires a stable non-empty ID.")
        execution_ = ParticleExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if not isinstance(execution_, ParticleExecutionPolicy):
            raise TypeError("execution must be a ParticleExecutionPolicy or None.")
        if not isinstance(precision_, ParticlePrecisionPolicy):
            raise TypeError("precision must be a ParticlePrecisionPolicy or None.")
        if execution_.realization != neighborhood.backend:
            raise ValueError(
                "Particle execution realization does not match the prepared "
                "neighborhood backend."
            )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "fixed smoothing length",
                "summation density",
                "one unordered pressure interaction per pair",
                f"{neighborhood.backend} pair topology with branchwise support mask",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "active_particles": particles.active_count,
                "pair_capacity": neighborhood.pair_capacity,
                "ambient_dimension": particles.ambient_dimension,
            },
        )
        self.particles = particles
        self.neighborhood = neighborhood
        self.method = method
        self.material = material
        self.external_potential = external_potential
        self.external_potential_id = external_potential_id
        self.execution = execution_
        self.precision = precision_
        self.key = method.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-barotropic-sph-dynamics",
                "particles": particles.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "method": method.method_id,
                "material": material.material_id,
                "external_potential": external_potential_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def _configuration(self, position: ArrayLike, /) -> Array:
        value = self.precision.geometry(position)
        expected = (self.particles.capacity, self.particles.ambient_dimension)
        if value.shape != expected:
            raise ValueError(
                f"Particle position shape must be {expected}, got {value.shape}."
            )
        active = self.particles.active_mask[:, None]
        value = eqx.error_if(
            value,
            jnp.any(jnp.where(active, ~jnp.isfinite(value), False)),
            "Active particle positions must be finite.",
        )
        return jnp.where(active, value, 0.0)

    def _momentum(self, momentum: ArrayLike, /) -> Array:
        value = self.precision.evaluation(momentum)
        expected = (self.particles.capacity, self.particles.ambient_dimension)
        if value.shape != expected:
            raise ValueError(
                f"Particle momentum shape must be {expected}, got {value.shape}."
            )
        active = self.particles.active_mask[:, None]
        value = eqx.error_if(
            value,
            jnp.any(jnp.where(active, ~jnp.isfinite(value), False)),
            "Active particle momenta must be finite.",
        )
        return jnp.where(active, value, 0.0)

    def neighborhood_state(self, position: ArrayLike, /) -> ParticleNeighborhoodState:
        return self.neighborhood.build(self._configuration(position))

    def _physical_pair_mask(self, geometry: ParticlePairGeometry, /) -> Array:
        support = self.method.kernel.support_radius(self.method.smoothing_length)
        return geometry.valid & (geometry.distance < support)

    def _density_from_pairs(
        self,
        pairs,
        geometry: ParticlePairGeometry,
        valid: Array,
        /,
    ) -> Array:
        masses = self.precision.evaluation(self.particles.safe_masses)
        pair_kernel = self.precision.evaluation(
            self.method.kernel.value(geometry.distance, self.method.smoothing_length)
        )
        left_mass = masses[pairs.left_indices]
        right_mass = masses[pairs.right_indices]
        neighbor_density = scatter_pair_sum(
            pairs,
            right_mass * pair_kernel,
            left_mass * pair_kernel,
            size=self.particles.capacity,
            accumulation=self.execution.accumulation,
            valid=valid,
        )
        zero = jnp.asarray(0.0, dtype=pair_kernel.dtype)
        self_kernel = self.method.kernel.value(zero, self.method.smoothing_length)
        self_density = masses * self_kernel
        return self.precision.evaluation(
            jnp.where(
                self.particles.active_mask,
                self_density + neighbor_density,
                0.0,
            )
        )

    def _evaluate(self, position: ArrayLike, /) -> _BarotropicSPHEvaluation:
        configuration = self._configuration(position)
        neighborhood = self.neighborhood.build(configuration)
        configuration = neighborhood.require_success(configuration)
        geometry = particle_pair_geometry(
            configuration,
            neighborhood.pair_relation,
            box=self.neighborhood.box,
        )
        physical_pairs = self._physical_pair_mask(geometry)
        density = self._density_from_pairs(
            neighborhood.pair_relation,
            geometry,
            physical_pairs,
        )
        safe_density = jnp.where(
            self.particles.active_mask,
            density,
            jnp.asarray(self.material.reference_density, dtype=density.dtype),
        )
        pressure = self.precision.evaluation(self.material.pressure(safe_density))
        pressure = jnp.where(self.particles.active_mask, pressure, 0.0)
        return _BarotropicSPHEvaluation(
            configuration,
            neighborhood,
            geometry,
            physical_pairs,
            density,
            pressure,
        )

    def pair_geometry(self, position: ArrayLike, /) -> ParticlePairGeometry:
        return self._evaluate(position).geometry

    def graph_view(self, position: ArrayLike, /, *, directed: bool = True):
        evaluation = self._evaluate(position)
        return particle_graph_view(
            self.particles,
            evaluation.neighborhood,
            evaluation.position,
            directed=directed,
            edge_mask=evaluation.physical_pairs,
            geometry=evaluation.geometry,
        )

    def density(self, position: ArrayLike, /) -> Array:
        return self._evaluate(position).density

    def _internal_energy_from_evaluation(
        self, evaluation: _BarotropicSPHEvaluation, /
    ) -> Array:
        safe_density = jnp.where(
            self.particles.active_mask,
            evaluation.density,
            jnp.asarray(
                self.material.reference_density,
                dtype=evaluation.density.dtype,
            ),
        )
        specific_energy = self.material.specific_internal_energy(safe_density)
        contributions = jnp.where(
            self.particles.active_mask,
            self.precision.evaluation(self.particles.safe_masses) * specific_energy,
            0.0,
        )
        return jnp.sum(self.precision.accumulation(contributions))

    def internal_potential_energy(self, position: ArrayLike, /) -> Array:
        return self._internal_energy_from_evaluation(self._evaluate(position))

    def external_potential_energy(
        self,
        time: ArrayLike,
        position: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        value = self._configuration(position)
        if self.external_potential is None:
            return jnp.zeros((), dtype=value.dtype)
        energy = jnp.asarray(self.external_potential(jnp.asarray(time), value, args))
        if energy.shape != ():
            raise ValueError("external_potential must return a scalar.")
        return energy.astype(value.dtype)

    def potential_energy(
        self,
        time: ArrayLike,
        position: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.internal_potential_energy(position) + self.external_potential_energy(
            time, position, args
        )

    def _internal_gradient_from_evaluation(
        self, evaluation: _BarotropicSPHEvaluation, /
    ) -> Array:
        pairs = evaluation.neighborhood.pair_relation
        masses = self.precision.evaluation(self.particles.safe_masses)
        left = pairs.left_indices
        right = pairs.right_indices
        coefficient = (
            evaluation.pressure[left] / evaluation.density[left] ** 2
            + evaluation.pressure[right] / evaluation.density[right] ** 2
        )
        kernel_gradient = self.method.kernel.gradient(
            evaluation.geometry.displacement,
            evaluation.geometry.distance,
            self.method.smoothing_length,
        )
        pair_gradient = (
            masses[left, None]
            * masses[right, None]
            * coefficient[:, None]
            * kernel_gradient
        )
        return self.precision.output(
            scatter_pair_exchange(
                pairs,
                self.precision.accumulation(pair_gradient),
                size=self.particles.capacity,
                accumulation=self.execution.accumulation,
                valid=evaluation.physical_pairs,
            )
        )

    def internal_potential_gradient(self, position: ArrayLike, /) -> Array:
        return self._internal_gradient_from_evaluation(self._evaluate(position))

    def _external_gradient(
        self,
        time: ArrayLike,
        position: Array,
        args: Any,
        /,
    ) -> Array:
        if self.external_potential is None:
            return jnp.zeros_like(position)

        def external(configuration):
            return self.external_potential_energy(time, configuration, args)

        return jax.grad(external)(position)

    def potential_gradient(
        self,
        time: ArrayLike,
        position: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        evaluation = self._evaluate(position)
        return self._internal_gradient_from_evaluation(
            evaluation
        ) + self._external_gradient(time, evaluation.position, args)

    def kinetic_gradient(
        self,
        time: ArrayLike,
        momentum: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del time, args
        value = self._momentum(momentum)
        masses = self.precision.evaluation(self.particles.safe_masses)
        velocity = value / masses[:, None]
        return self.precision.output(
            jnp.where(self.particles.active_mask[:, None], velocity, 0.0)
        )

    def force(
        self,
        time: ArrayLike,
        position: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return -self.potential_gradient(time, position, args)

    def acceleration(
        self,
        time: ArrayLike,
        position: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        force = self.force(time, position, args)
        masses = self.precision.evaluation(self.particles.safe_masses)
        return jnp.where(
            self.particles.active_mask[:, None], force / masses[:, None], 0.0
        )

    def pack_phase_state(self, position: ArrayLike, velocity: ArrayLike, /) -> Array:
        configuration = self._configuration(position)
        velocity_ = self.precision.evaluation(velocity)
        if velocity_.shape != configuration.shape:
            raise ValueError("Particle velocity must match particle position shape.")
        velocity_ = eqx.error_if(
            velocity_,
            jnp.any(
                jnp.where(
                    self.particles.active_mask[:, None],
                    ~jnp.isfinite(velocity_),
                    False,
                )
            ),
            "Active particle velocities must be finite.",
        )
        masses = self.precision.evaluation(self.particles.safe_masses)
        momentum = jnp.where(
            self.particles.active_mask[:, None], masses[:, None] * velocity_, 0.0
        )
        return jnp.concatenate((configuration, momentum), axis=-1)

    def unpack_phase_state(self, state: ArrayLike, /) -> tuple[Array, Array, Array]:
        value = jnp.asarray(state)
        dimension = self.particles.ambient_dimension
        expected = (self.particles.capacity, 2 * dimension)
        if value.shape != expected:
            raise ValueError(f"Particle phase state shape must be {expected}.")
        position = value[..., :dimension]
        momentum = value[..., dimension:]
        velocity = self.kinetic_gradient(jnp.asarray(0.0), momentum, None)
        return position, momentum, velocity

    @staticmethod
    def _angular_sum(position: Array, vector: Array, /) -> Array:
        dimension = int(position.shape[-1])
        if dimension == 1:
            return jnp.zeros((), dtype=position.dtype)
        if dimension == 2:
            torque = position[:, 0] * vector[:, 1] - position[:, 1] * vector[:, 0]
            return compensated_sum(torque)
        if dimension == 3:
            return compensated_sum(jnp.cross(position, vector), axis=0)
        raise ValueError("Angular diagnostics support dimensions 1, 2, and 3.")

    def diagnostics(
        self,
        time: ArrayLike,
        position: ArrayLike,
        momentum: ArrayLike,
        args: Any = None,
        /,
    ) -> BarotropicSPHDiagnostics:
        evaluation = self._evaluate(position)
        configuration = evaluation.position
        momentum_ = self._momentum(momentum)
        density = evaluation.density
        physical_pairs = evaluation.physical_pairs
        pairs = evaluation.neighborhood.pair_relation
        counts = jnp.zeros((self.particles.capacity,), dtype=jnp.int32)
        increments = physical_pairs.astype(jnp.int32)
        counts = counts.at[pairs.left_indices].add(increments)
        counts = counts.at[pairs.right_indices].add(increments)
        active = self.particles.active_mask
        density_minimum = jnp.min(jnp.where(active, density, jnp.inf))
        density_maximum = jnp.max(jnp.where(active, density, -jnp.inf))
        count_minimum = jnp.min(jnp.where(active, counts, jnp.iinfo(jnp.int32).max))
        count_maximum = jnp.max(jnp.where(active, counts, 0))
        masses = self.precision.certification(self.particles.safe_masses)
        momentum_cert = self.precision.certification(momentum_)
        configuration_cert = self.precision.certification(configuration)
        kinetic_terms = jnp.where(
            active,
            0.5 * jnp.sum(momentum_cert * momentum_cert, axis=-1) / masses,
            0.0,
        )
        internal_energy = self.precision.certification(
            self._internal_energy_from_evaluation(evaluation)
        )
        external_energy = self.precision.certification(
            self.external_potential_energy(time, configuration, args)
        )
        internal_force = -self.precision.certification(
            self._internal_gradient_from_evaluation(evaluation)
        )
        total_mass = compensated_sum(jnp.where(active, masses, 0.0))
        linear_momentum = compensated_sum(
            jnp.where(active[:, None], momentum_cert, 0.0), axis=0
        )
        kinetic_energy = compensated_sum(kinetic_terms)
        angular_momentum = self._angular_sum(configuration_cert, momentum_cert)
        net_internal_force = compensated_sum(internal_force, axis=0)
        net_internal_torque = self._angular_sum(configuration_cert, internal_force)
        safe_density = jnp.where(
            active,
            density,
            jnp.asarray(self.material.reference_density, dtype=density.dtype),
        )
        admissible = jnp.all(
            jnp.where(active, self.material.admissible(safe_density), True)
        )
        return BarotropicSPHDiagnostics(
            density_minimum=density_minimum,
            density_maximum=density_maximum,
            neighbor_count_minimum=count_minimum,
            neighbor_count_maximum=count_maximum,
            total_mass=total_mass,
            linear_momentum=linear_momentum,
            angular_momentum=angular_momentum,
            kinetic_energy=kinetic_energy,
            internal_energy=internal_energy,
            external_potential_energy=external_energy,
            total_energy=kinetic_energy + internal_energy + external_energy,
            net_internal_force=net_internal_force,
            net_internal_torque=net_internal_torque,
            admissible=admissible,
            active_pairs=jnp.sum(physical_pairs.astype(jnp.int32)),
        )

    def stable_step(
        self,
        time: ArrayLike,
        position: ArrayLike,
        momentum: ArrayLike,
        args: Any = None,
        /,
    ) -> BarotropicSPHStepRestriction:
        evaluation = self._evaluate(position)
        self._momentum(momentum)
        density = evaluation.density
        safe_density = jnp.where(
            self.particles.active_mask,
            density,
            jnp.asarray(self.material.reference_density, dtype=density.dtype),
        )
        sound_speed = self.material.sound_speed(safe_density)
        maximum_sound_speed = jnp.max(
            jnp.where(self.particles.active_mask, sound_speed, 0.0)
        )
        acoustic = (
            self.method.acoustic_cfl
            * self.method.smoothing_length
            / jnp.maximum(
                maximum_sound_speed,
                jnp.finfo(sound_speed.dtype).tiny,
            )
        )
        gradient = self._internal_gradient_from_evaluation(
            evaluation
        ) + self._external_gradient(time, evaluation.position, args)
        masses = self.precision.evaluation(self.particles.safe_masses)
        acceleration = jnp.where(
            self.particles.active_mask[:, None],
            -gradient / masses[:, None],
            0.0,
        )
        acceleration_norm = jnp.sqrt(jnp.sum(acceleration * acceleration, axis=-1))
        maximum_acceleration = jnp.max(
            jnp.where(self.particles.active_mask, acceleration_norm, 0.0)
        )
        force = jnp.where(
            maximum_acceleration > 0.0,
            self.method.force_cfl
            * jnp.sqrt(self.method.smoothing_length / maximum_acceleration),
            jnp.asarray(jnp.inf, dtype=acoustic.dtype),
        )
        return BarotropicSPHStepRestriction(
            acoustic=acoustic,
            force=force,
            selected=jnp.minimum(acoustic, force),
        )

    def linearize(
        self,
        time: Array,
        position: Array,
        args: Any = None,
        /,
    ):
        value, jvp = jax.linearize(
            lambda configuration: self.potential_gradient(time, configuration, args),
            position,
        )
        _, vjp = jax.vjp(
            lambda configuration: self.potential_gradient(time, configuration, args),
            position,
        )
        return value, jvp, vjp


__all__ = [
    "BarotropicSPHDiagnostics",
    "BarotropicSPHMethodPlan",
    "BarotropicSPHStepRestriction",
    "PreparedBarotropicSPHDynamics",
]
