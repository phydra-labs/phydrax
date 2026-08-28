#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._pairwise import scatter_pair_sum
from ._wcsph import PreparedWeaklyCompressibleSPHDynamics


class TransportVelocityStateLayout(StrictModule, NonTrainableState):
    particle_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, particle_capacity: int, ambient_dimension: int, /):
        self.particle_capacity = int(particle_capacity)
        self.ambient_dimension = int(ambient_dimension)
        self.width = 3 * self.ambient_dimension + 1
        self.layout_id = canonical_fingerprint(
            {
                "kind": "transport-velocity-state-layout",
                "capacity": self.particle_capacity,
                "dimension": self.ambient_dimension,
            }
        )

    def pack(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        transport_velocity: ArrayLike,
        density: ArrayLike,
        /,
    ) -> Array:
        values = tuple(
            jnp.asarray(value) for value in (position, velocity, transport_velocity)
        )
        expected = (self.particle_capacity, self.ambient_dimension)
        if any(value.shape != expected for value in values):
            raise ValueError("Transport-velocity vector state shape mismatch.")
        density_ = jnp.asarray(density)
        if density_.shape != (self.particle_capacity,):
            raise ValueError("Transport-velocity density shape mismatch.")
        return jnp.concatenate(values + (density_[:, None],), axis=-1)

    def unpack(self, state: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        value = jnp.asarray(state)
        if value.shape != (self.particle_capacity, self.width):
            raise ValueError("Transport-velocity packed state shape mismatch.")
        dimension = self.ambient_dimension
        return (
            value[:, :dimension],
            value[:, dimension : 2 * dimension],
            value[:, 2 * dimension : 3 * dimension],
            value[:, -1],
        )


class TransportVelocitySPHMethodPlan(StrictModule, NonTrainableState):
    background_pressure: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, background_pressure: float, /):
        pressure = float(background_pressure)
        if not np.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("Transport background pressure must be finite and positive.")
        self.background_pressure = pressure
        self.plan_id = canonical_fingerprint(
            {"kind": "transport-velocity-sph", "background_pressure": pressure}
        )


class TransportVelocityDiagnostics(StrictModule):
    maximum_velocity_difference: Array
    background_acceleration_norm: Array
    transport_stress_power: Array
    momentum_defect: Array


class PreparedTransportVelocityDynamics(StrictModule, NonTrainableState):
    base: PreparedWeaklyCompressibleSPHDynamics
    plan: TransportVelocitySPHMethodPlan
    layout: TransportVelocityStateLayout
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: PreparedWeaklyCompressibleSPHDynamics,
        plan: TransportVelocitySPHMethodPlan,
        /,
    ):
        if not base.state_layout.density_evolved:
            raise ValueError("Transport velocity requires continuity density.")
        if base.method.free_surface_detection is not None:
            raise ValueError("Transport velocity is incompatible with free surfaces.")
        self.base = base
        self.plan = plan
        self.layout = TransportVelocityStateLayout(
            base.particles.capacity, base.particles.ambient_dimension
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-transport-velocity",
                "base": base.prepared_id,
                "plan": plan.plan_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density: ArrayLike | None = None,
        /,
        *,
        transport_velocity: ArrayLike | None = None,
    ) -> Array:
        base_state = self.base.initialize_state(position, velocity, density)
        position_, velocity_, density_ = self.base.state_layout.unpack(base_state)
        transport = (
            velocity_ if transport_velocity is None else jnp.asarray(transport_velocity)
        )
        return self.layout.pack(position_, velocity_, transport, density_)

    def _terms(self, time: Array, state: Array, args, /):
        position, velocity, transport, density = self.layout.unpack(state)
        base_state = self.base.state_layout.pack(position, velocity, density)
        base_rate = self.base(time, base_state, args)
        _, physical_acceleration, density_rate = self.base.state_layout.unpack_rate(
            base_rate
        )
        evaluation = self.base._evaluate(time, base_state, args)
        pairs = evaluation.neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        gradient = self.base.method.kernel.gradient(
            evaluation.geometry.displacement,
            evaluation.geometry.distance,
            self.base.method.smoothing_length,
        )
        volume = self.base.particles.safe_masses / density
        pair_volume = volume[left] ** 2 + volume[right] ** 2
        left_background = (
            -self.plan.background_pressure
            / self.base.particles.safe_masses[left]
            * pair_volume
        )[:, None] * gradient
        right_background = (
            self.plan.background_pressure
            / self.base.particles.safe_masses[right]
            * pair_volume
        )[:, None] * gradient
        background = scatter_pair_sum(
            pairs,
            left_background,
            right_background,
            size=self.base.particles.capacity,
            accumulation=self.base.execution.accumulation,
            valid=evaluation.physical_pairs,
        )
        relative = transport - velocity
        stress = density[:, None, None] * contract("ni,nj->nij", velocity, relative)
        pair_stress = stress[left] + stress[right]
        stress_vector = contract("eij,ej->ei", pair_stress, gradient)
        left_stress = (
            -self.base.particles.safe_masses[right] / (density[left] * density[right])
        )[:, None] * stress_vector
        right_stress = (
            self.base.particles.safe_masses[left] / (density[left] * density[right])
        )[:, None] * stress_vector
        stress_acceleration = scatter_pair_sum(
            pairs,
            left_stress,
            right_stress,
            size=self.base.particles.capacity,
            accumulation=self.base.execution.accumulation,
            valid=evaluation.physical_pairs,
        )
        physical_rate = physical_acceleration + stress_acceleration
        return (
            transport,
            physical_rate,
            physical_rate + background,
            density_rate,
            background,
            stress_acceleration,
        )

    def __call__(self, time: Array, state: Array, args=None, /) -> Array:
        position_rate, velocity_rate, transport_rate, density_rate, _, _ = self._terms(
            time, state, args
        )
        return self.layout.pack(
            position_rate, velocity_rate, transport_rate, density_rate
        )

    def refresh_transport_velocity(
        self, time: Array, state: Array, step_size: Array, args=None, /
    ) -> Array:
        position, velocity, _, density = self.layout.unpack(state)
        _, _, _, _, background, _ = self._terms(time, state, args)
        transport = velocity + step_size * background
        return self.layout.pack(position, velocity, transport, density)

    def diagnostics(
        self, time: Array, state: Array, args=None, /
    ) -> TransportVelocityDiagnostics:
        _, velocity, transport, _ = self.layout.unpack(state)
        _, _, _, _, background, stress_acceleration = self._terms(time, state, args)
        mass = self.base.particles.safe_masses
        return TransportVelocityDiagnostics(
            jnp.max(jnp.sqrt(jnp.sum((transport - velocity) ** 2, axis=-1))),
            jnp.max(jnp.sqrt(jnp.sum(background**2, axis=-1))),
            compensated_sum(mass * jnp.sum(velocity * stress_acceleration, axis=-1)),
            compensated_sum(mass[:, None] * stress_acceleration, axis=0),
        )


__all__ = [
    "PreparedTransportVelocityDynamics",
    "TransportVelocityDiagnostics",
    "TransportVelocitySPHMethodPlan",
    "TransportVelocityStateLayout",
]
