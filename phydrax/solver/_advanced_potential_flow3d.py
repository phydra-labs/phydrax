#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class NonlinearPotentialFlowPolicy3D(StrictModule, NonTrainableState):
    time_step: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    viscosity: float = eqx.field(static=True)
    maximum_surface_speed: float = eqx.field(static=True)
    minimum_vertical_clearance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time_step: float,
        gravity: float = 9.81,
        viscosity: float = 0.0,
        maximum_surface_speed: float,
        minimum_vertical_clearance: float,
    ):
        values = tuple(
            map(
                float,
                (
                    time_step,
                    gravity,
                    viscosity,
                    maximum_surface_speed,
                    minimum_vertical_clearance,
                ),
            )
        )
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] <= 0
            or values[1] <= 0
            or values[2] < 0
            or values[3] <= 0
            or values[4] <= 0
        ):
            raise ValueError(
                "Potential-flow policy values violate their positive bounded envelope."
            )
        (
            self.time_step,
            self.gravity,
            self.viscosity,
            self.maximum_surface_speed,
            self.minimum_vertical_clearance,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "nonlinear-potential-flow-policy-3d",
                "values": values,
                "model": "mixed-Eulerian-Lagrangian-Bernoulli-viscous-potential",
            }
        )


class NonlinearPotentialFlowState3D(StrictModule):
    free_surface_points: Array
    potential: Array
    accepted_time: Array
    mass: Array
    energy: Array
    valid: Array


class NonlinearPotentialFlowStep3D(StrictModule):
    state: NonlinearPotentialFlowState3D
    mass_defect: Array
    energy_work_defect: Array
    successful: Array


class PreparedNonlinearPotentialFlow3D(StrictModule, NonTrainableState):
    policy: NonlinearPotentialFlowPolicy3D
    reference_area_weights: Array
    point_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self, points: ArrayLike, potential: ArrayLike, /
    ) -> NonlinearPotentialFlowState3D:
        points_ = jnp.asarray(points)
        potential_ = jnp.asarray(potential, dtype=points_.dtype)
        if points_.shape != (self.point_count, 3) or potential_.shape != (
            self.point_count,
        ):
            raise ValueError(
                "Potential-flow state shapes differ from the prepared surface."
            )
        mass = jnp.sum(self.reference_area_weights * points_[:, 2])
        energy = jnp.sum(
            self.reference_area_weights * (0.5 * self.policy.gravity * points_[:, 2] ** 2)
        )
        valid = jnp.all(jnp.isfinite(points_)) & jnp.all(jnp.isfinite(potential_))
        return NonlinearPotentialFlowState3D(
            points_,
            potential_,
            jnp.asarray(0.0, dtype=points_.dtype),
            mass,
            energy,
            valid,
        )

    def step(
        self,
        state: NonlinearPotentialFlowState3D,
        surface_velocity: ArrayLike,
        /,
        *,
        external_pressure: ArrayLike = 0.0,
        surface_laplacian_potential: ArrayLike | None = None,
    ) -> NonlinearPotentialFlowStep3D:
        velocity = jnp.asarray(surface_velocity, dtype=state.free_surface_points.dtype)
        pressure = jnp.asarray(external_pressure, dtype=velocity.dtype)
        if velocity.shape != (self.point_count, 3) or pressure.shape not in (
            (),
            (self.point_count,),
        ):
            raise ValueError("Surface velocity/pressure shapes are incompatible.")
        speed = jnp.linalg.norm(velocity, axis=-1)
        laplacian = (
            jnp.zeros((self.point_count,), dtype=velocity.dtype)
            if surface_laplacian_potential is None
            else jnp.asarray(surface_laplacian_potential, dtype=velocity.dtype)
        )
        if laplacian.shape != (self.point_count,):
            raise ValueError("surface_laplacian_potential must have one value per point.")
        dt = self.policy.time_step
        new_points = state.free_surface_points + dt * velocity
        bernoulli = (
            -self.policy.gravity * state.free_surface_points[:, 2]
            - 0.5 * speed**2
            - pressure
            + 2.0 * self.policy.viscosity * laplacian
        )
        new_potential = state.potential + dt * bernoulli
        mass = jnp.sum(self.reference_area_weights * new_points[:, 2])
        energy = jnp.sum(
            self.reference_area_weights
            * (0.5 * speed**2 + 0.5 * self.policy.gravity * new_points[:, 2] ** 2)
        )
        successful = (
            state.valid
            & jnp.all(jnp.isfinite(new_points))
            & jnp.all(jnp.isfinite(new_potential))
            & jnp.all(speed <= self.policy.maximum_surface_speed)
            & (jnp.min(-new_points[:, 2]) >= self.policy.minimum_vertical_clearance)
        )
        candidate = NonlinearPotentialFlowState3D(
            new_points, new_potential, state.accepted_time + dt, mass, energy, successful
        )
        return NonlinearPotentialFlowStep3D(
            candidate, mass - state.mass, energy - state.energy, successful
        )


class SecondOrderPotentialFlowPlan3D(StrictModule, NonTrainableState):
    frequencies: Array
    maximum_frequency_pairs: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, frequencies: ArrayLike, /, *, maximum_frequency_pairs: int):
        values = np.asarray(frequencies, dtype=float)
        limit = int(maximum_frequency_pairs)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0)
            or values.size**2 > limit
        ):
            raise ValueError("Second-order frequencies violate the finite pair envelope.")
        self.frequencies = jnp.asarray(values)
        self.maximum_frequency_pairs = limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "second-order-potential-flow-plan-3d",
                "frequencies": values.tolist(),
                "limit": limit,
            }
        )

    def quadratic_transfer(
        self, first_order_amplitudes: ArrayLike, /
    ) -> tuple[Array, Array]:
        amplitudes = jnp.asarray(first_order_amplitudes)
        if amplitudes.shape[0] != self.frequencies.shape[0]:
            raise ValueError("First-order amplitudes must begin with the frequency axis.")
        sum_frequency = self.frequencies[:, None] + self.frequencies[None, :]
        difference_frequency = self.frequencies[:, None] - self.frequencies[None, :]
        qtf = amplitudes[:, None, ...] * jnp.conj(amplitudes[None, :, ...])
        return jnp.stack((sum_frequency, difference_frequency)), qtf


def prepare_nonlinear_potential_flow_3d(
    area_weights: ArrayLike, policy: NonlinearPotentialFlowPolicy3D, /
) -> PreparedNonlinearPotentialFlow3D:
    weights = np.asarray(area_weights, dtype=float)
    if (
        weights.ndim != 1
        or weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("area_weights must be finite positive point weights.")
    if not isinstance(policy, NonlinearPotentialFlowPolicy3D):
        raise TypeError("policy must be NonlinearPotentialFlowPolicy3D.")
    return PreparedNonlinearPotentialFlow3D(
        policy=policy,
        reference_area_weights=jnp.asarray(weights),
        point_count=weights.size,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-nonlinear-potential-flow-3d",
                "policy": policy.policy_id,
                "point_count": weights.size,
            }
        ),
    )


__all__ = [
    "NonlinearPotentialFlowPolicy3D",
    "NonlinearPotentialFlowState3D",
    "NonlinearPotentialFlowStep3D",
    "PreparedNonlinearPotentialFlow3D",
    "SecondOrderPotentialFlowPlan3D",
    "prepare_nonlinear_potential_flow_3d",
]
