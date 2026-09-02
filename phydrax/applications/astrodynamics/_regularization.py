#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext
from ._scalable_gravity import EncounterEvaluation
from ._status import AstrodynamicsStatus


KSGaugePolicy: TypeAlias = Literal["positive-first", "largest-component"]


def _stumpff(z: Array, /) -> tuple[Array, Array]:
    root_positive = jnp.sqrt(jnp.maximum(z, 0.0))
    root_negative = jnp.sqrt(jnp.maximum(-z, 0.0))
    small = jnp.abs(z) < 1.0e-6
    c_series = 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0
    s_series = 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0
    c_positive = (1.0 - jnp.cos(root_positive)) / jnp.where(z > 0.0, z, 1.0)
    s_positive = (root_positive - jnp.sin(root_positive)) / jnp.where(
        root_positive > 0.0, root_positive**3, 1.0
    )
    c_negative = (jnp.cosh(root_negative) - 1.0) / jnp.where(-z > 0.0, -z, 1.0)
    s_negative = (jnp.sinh(root_negative) - root_negative) / jnp.where(
        root_negative > 0.0, root_negative**3, 1.0
    )
    c = jnp.where(small, c_series, jnp.where(z > 0.0, c_positive, c_negative))
    s = jnp.where(small, s_series, jnp.where(z > 0.0, s_positive, s_negative))
    return c, s


def _ks_position(coordinate: Array, /) -> Array:
    u0, u1, u2, u3 = coordinate
    return jnp.asarray(
        (
            u0 * u0 - u1 * u1 - u2 * u2 + u3 * u3,
            2.0 * (u0 * u1 - u2 * u3),
            2.0 * (u0 * u2 + u1 * u3),
        )
    )


def _ks_coordinate(position: Array, gauge: KSGaugePolicy, /) -> Array:
    radius = jnp.sqrt(jnp.sum(position * position))
    first = jnp.sqrt(jnp.maximum(0.5 * (radius + position[0]), 0.0))
    safe_first = jnp.where(first > 1.0e-14, first, 1.0)
    positive = jnp.asarray(
        (first, position[1] / (2.0 * safe_first), position[2] / (2.0 * safe_first), 0.0)
    )
    alternate = jnp.asarray(
        (
            position[1]
            / (2.0 * jnp.sqrt(jnp.maximum(0.5 * (radius - position[0]), 1.0e-28))),
            jnp.sqrt(jnp.maximum(0.5 * (radius - position[0]), 0.0)),
            0.0,
            position[2]
            / (2.0 * jnp.sqrt(jnp.maximum(0.5 * (radius - position[0]), 1.0e-28))),
        )
    )
    if gauge == "largest-component":
        return jnp.where(
            radius + position[0] >= radius - position[0], positive, alternate
        )
    return jnp.where(first > 1.0e-14, positive, alternate)


class CloseEncounterRegularizationResult(StrictModule):
    positions: Array
    velocities: Array
    pair: Array
    center_of_mass_position: Array
    center_of_mass_velocity: Array
    ks_coordinate: Array
    ks_momentum: Array
    physical_time: Array
    step_mask: Array
    energy_residual: Array
    angular_momentum_residual: Array
    time_residual: Array
    perturbation_ratio: Array
    derivative_available: Array
    successful: Array
    status: Array
    context: AstrodynamicsContext
    plan_id: str = eqx.field(static=True)


class CloseEncounterRegularizationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity single-pair Sundman/universal-variable regularization."""

    encounter_radius: float = eqx.field(static=True)
    collision_radius: float = eqx.field(static=True)
    maximum_fictitious_steps: int = eqx.field(static=True)
    physical_time_tolerance: float = eqx.field(static=True)
    maximum_perturbation_ratio: float = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    gauge_policy: KSGaugePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        encounter_radius: float,
        collision_radius: float,
        /,
        *,
        maximum_fictitious_steps: int = 32,
        physical_time_tolerance: float = 1.0e-11,
        maximum_perturbation_ratio: float = 1.0e-2,
        gravitational_constant: float = 1.0,
        gauge_policy: KSGaugePolicy = "largest-component",
    ):
        values = tuple(
            float(value)
            for value in (
                encounter_radius,
                collision_radius,
                physical_time_tolerance,
                maximum_perturbation_ratio,
                gravitational_constant,
            )
        )
        capacity = int(maximum_fictitious_steps)
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[1] >= values[0]
            or values[2] <= 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
            or capacity <= 0
            or gauge_policy not in ("positive-first", "largest-component")
        ):
            raise ValueError("Close-encounter regularization policy is invalid.")
        self.encounter_radius = values[0]
        self.collision_radius = values[1]
        self.maximum_fictitious_steps = capacity
        self.physical_time_tolerance = values[2]
        self.maximum_perturbation_ratio = values[3]
        self.gravitational_constant = values[4]
        self.gauge_policy = gauge_policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-pair-close-encounter-regularization",
                "encounter_radius": values[0],
                "collision_radius": values[1],
                "maximum_fictitious_steps": capacity,
                "physical_time_tolerance": values[2],
                "maximum_perturbation_ratio": values[3],
                "gravitational_constant": values[4],
                "gauge_policy": gauge_policy,
            }
        )

    def prepare(
        self,
        masses: ArrayLike,
        positions: ArrayLike,
        velocities: ArrayLike,
        encounter: EncounterEvaluation,
        context: AstrodynamicsContext,
        /,
    ) -> "PreparedCloseEncounterSegment":
        return PreparedCloseEncounterSegment(
            self, masses, positions, velocities, encounter, context
        )


class PreparedCloseEncounterSegment(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: CloseEncounterRegularizationPlan
    masses: Array
    initial_positions: Array
    initial_velocities: Array
    pair: tuple[int, int] = eqx.field(static=True)
    context: AstrodynamicsContext
    preparation_status: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CloseEncounterRegularizationPlan,
        masses: ArrayLike,
        positions: ArrayLike,
        velocities: ArrayLike,
        encounter: EncounterEvaluation,
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(encounter, EncounterEvaluation):
            raise TypeError("encounter must be an EncounterEvaluation.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        mass = np.asarray(masses, dtype=float)
        position = np.asarray(positions, dtype=float)
        velocity = np.asarray(velocities, dtype=float)
        if (
            mass.ndim != 1
            or mass.size < 2
            or position.shape != (mass.size, 3)
            or velocity.shape != position.shape
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(velocity))
        ):
            raise ValueError("Regularized segment state arrays are invalid.")
        pair_values = np.asarray(encounter.pair, dtype=int)
        if pair_values.shape != (2,):
            raise ValueError("Encounter pair must contain two particle indices.")
        pair = tuple(sorted((int(pair_values[0]), int(pair_values[1]))))
        if pair[0] < 0 or pair[1] >= mass.size or pair[0] == pair[1]:
            raise ValueError("Encounter pair is outside particle capacity.")
        displacement = position[:, None, :] - position[None, :, :]
        distance = np.linalg.norm(displacement, axis=-1)
        close = np.triu(distance <= plan.encounter_radius, k=1)
        selected_distance = distance[pair]
        status = int(AstrodynamicsStatus.SUCCESS)
        if selected_distance <= plan.collision_radius or bool(
            np.asarray(encounter.collided)
        ):
            status = int(AstrodynamicsStatus.COLLISION)
        elif np.count_nonzero(close) != 1 or not bool(np.asarray(encounter.encountered)):
            status = int(AstrodynamicsStatus.SINGULAR_GEOMETRY)
        self.plan = plan
        self.masses = jnp.asarray(mass)
        self.initial_positions = jnp.asarray(position)
        self.initial_velocities = jnp.asarray(velocity)
        self.pair = pair
        self.context = context
        self.preparation_status = status
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-close-encounter-segment",
                "plan": plan.plan_id,
                "pair": list(pair),
                "particle_capacity": int(mass.size),
                "state": array_tree_fingerprint((mass, position, velocity)),
                "context": context.context_id,
                "preparation_status": status,
            }
        )

    def propagate(
        self,
        physical_duration: ArrayLike,
        /,
        *,
        external_acceleration: ArrayLike | None = None,
    ) -> CloseEncounterRegularizationResult:
        duration = jnp.asarray(
            physical_duration, dtype=self.initial_positions.dtype
        ).reshape(())
        acceleration = (
            jnp.zeros_like(self.initial_positions)
            if external_acceleration is None
            else jnp.asarray(external_acceleration, dtype=self.initial_positions.dtype)
        )
        if acceleration.shape != self.initial_positions.shape:
            raise ValueError("external_acceleration must match the particle state shape.")
        first, second = self.pair
        first_mass, second_mass = self.masses[first], self.masses[second]
        total_mass = first_mass + second_mass
        reduced_mass = first_mass * second_mass / total_mass
        mu = self.plan.gravitational_constant * total_mass
        relative_position = self.initial_positions[second] - self.initial_positions[first]
        relative_velocity = (
            self.initial_velocities[second] - self.initial_velocities[first]
        )
        relative_acceleration = acceleration[second] - acceleration[first]
        radius0 = jnp.sqrt(jnp.sum(relative_position * relative_position))
        speed2 = jnp.sum(relative_velocity * relative_velocity)
        radial_velocity = jnp.sum(relative_position * relative_velocity) / radius0
        alpha = 2.0 / radius0 - speed2 / mu
        root_mu = jnp.sqrt(mu)
        initial_chi = jnp.where(
            jnp.abs(alpha) > 1.0e-8,
            root_mu * jnp.abs(alpha) * duration,
            root_mu * duration / radius0,
        )

        def newton_step(carry, _):
            chi, converged = carry
            z = alpha * chi * chi
            c, s = _stumpff(z)
            residual = (
                radius0 * radial_velocity / root_mu * chi * chi * c
                + (1.0 - alpha * radius0) * chi**3 * s
                + radius0 * chi
                - root_mu * duration
            )
            derivative = (
                radius0 * radial_velocity / root_mu * chi * (1.0 - z * s)
                + (1.0 - alpha * radius0) * chi * chi * c
                + radius0
            )
            candidate = chi - residual / jnp.where(
                jnp.abs(derivative) > 0.0, derivative, 1.0
            )
            candidate_residual = jnp.abs(residual) / jnp.maximum(root_mu, 1.0)
            now_converged = candidate_residual <= self.plan.physical_time_tolerance
            return (
                jnp.where(converged, chi, candidate),
                converged | now_converged,
            ), now_converged

        (chi, converged), convergence_history = jax.lax.scan(
            newton_step,
            (initial_chi, jnp.asarray(False)),
            xs=None,
            length=self.plan.maximum_fictitious_steps,
        )
        z = alpha * chi * chi
        c, s = _stumpff(z)
        time_equation = (
            radius0 * radial_velocity / root_mu * chi * chi * c
            + (1.0 - alpha * radius0) * chi**3 * s
            + radius0 * chi
        ) / root_mu
        time_residual = jnp.abs(time_equation - duration)
        f = 1.0 - chi * chi / radius0 * c
        g = duration - chi**3 / root_mu * s
        propagated_relative_position = f * relative_position + g * relative_velocity
        propagated_radius = jnp.sqrt(
            jnp.sum(propagated_relative_position * propagated_relative_position)
        )
        fdot = root_mu / (propagated_radius * radius0) * (alpha * chi**3 * s - chi)
        gdot = 1.0 - chi * chi / propagated_radius * c
        propagated_relative_velocity = fdot * relative_position + gdot * relative_velocity
        propagated_relative_position = (
            propagated_relative_position + 0.5 * relative_acceleration * duration**2
        )
        propagated_relative_velocity = (
            propagated_relative_velocity + relative_acceleration * duration
        )
        pair_force = mu / jnp.maximum(radius0 * radius0, jnp.finfo(radius0.dtype).tiny)
        perturbation_ratio = jnp.sqrt(jnp.sum(relative_acceleration**2)) / pair_force
        center_position = (
            first_mass * self.initial_positions[first]
            + second_mass * self.initial_positions[second]
        ) / total_mass
        center_velocity = (
            first_mass * self.initial_velocities[first]
            + second_mass * self.initial_velocities[second]
        ) / total_mass
        center_acceleration = (
            first_mass * acceleration[first] + second_mass * acceleration[second]
        ) / total_mass
        propagated_center = (
            center_position
            + duration * center_velocity
            + 0.5 * center_acceleration * duration**2
        )
        propagated_center_velocity = center_velocity + center_acceleration * duration
        positions = (
            self.initial_positions
            + duration * self.initial_velocities
            + 0.5 * acceleration * duration**2
        )
        velocities = self.initial_velocities + acceleration * duration
        positions = positions.at[first].set(
            propagated_center - second_mass / total_mass * propagated_relative_position
        )
        positions = positions.at[second].set(
            propagated_center + first_mass / total_mass * propagated_relative_position
        )
        velocities = velocities.at[first].set(
            propagated_center_velocity
            - second_mass / total_mass * propagated_relative_velocity
        )
        velocities = velocities.at[second].set(
            propagated_center_velocity
            + first_mass / total_mass * propagated_relative_velocity
        )
        initial_energy = 0.5 * jnp.sum(relative_velocity**2) - mu / radius0
        final_energy = 0.5 * jnp.sum(propagated_relative_velocity**2) - mu / jnp.sqrt(
            jnp.sum(propagated_relative_position**2)
        )
        energy_residual = jnp.abs(final_energy - initial_energy) / jnp.maximum(
            jnp.abs(initial_energy), 1.0
        )
        initial_angular = jnp.cross(relative_position, relative_velocity)
        final_angular = jnp.cross(
            propagated_relative_position, propagated_relative_velocity
        )
        angular_residual = jnp.sqrt(
            jnp.sum((final_angular - initial_angular) ** 2)
        ) / jnp.maximum(jnp.sqrt(jnp.sum(initial_angular**2)), 1.0)
        ks_coordinate = _ks_coordinate(
            propagated_relative_position, self.plan.gauge_policy
        )
        jacobian = jax.jacfwd(_ks_position)(ks_coordinate)
        ks_momentum = contract(
            "ij,i->j", jacobian, reduced_mass * propagated_relative_velocity
        )
        finite = (
            jnp.isfinite(duration)
            & jnp.all(jnp.isfinite(acceleration))
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(velocities))
            & jnp.all(jnp.isfinite(ks_coordinate))
            & jnp.all(jnp.isfinite(ks_momentum))
        )
        prepared = self.preparation_status == int(AstrodynamicsStatus.SUCCESS)
        perturbation_valid = perturbation_ratio <= self.plan.maximum_perturbation_ratio
        time_valid = converged & (time_residual <= self.plan.physical_time_tolerance)
        final_radius = jnp.sqrt(jnp.sum(propagated_relative_position**2))
        collision = final_radius <= self.plan.collision_radius
        successful = prepared & finite & perturbation_valid & time_valid & ~collision
        status = jnp.where(
            self.preparation_status != int(AstrodynamicsStatus.SUCCESS),
            self.preparation_status,
            jnp.where(
                collision,
                int(AstrodynamicsStatus.COLLISION),
                jnp.where(
                    ~finite,
                    int(AstrodynamicsStatus.NONFINITE_INPUT),
                    jnp.where(
                        ~perturbation_valid,
                        int(AstrodynamicsStatus.UNSUPPORTED_REGIME),
                        jnp.where(
                            ~time_valid,
                            int(AstrodynamicsStatus.CAPACITY_EXCEEDED),
                            int(AstrodynamicsStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        safe_positions = jnp.where(successful, positions, self.initial_positions)
        safe_velocities = jnp.where(successful, velocities, self.initial_velocities)
        return CloseEncounterRegularizationResult(
            positions=safe_positions,
            velocities=safe_velocities,
            pair=jnp.asarray(self.pair, dtype=jnp.int32),
            center_of_mass_position=propagated_center,
            center_of_mass_velocity=propagated_center_velocity,
            ks_coordinate=ks_coordinate,
            ks_momentum=ks_momentum,
            physical_time=duration,
            step_mask=jnp.cumsum(convergence_history.astype(jnp.int32)) <= 1,
            energy_residual=energy_residual,
            angular_momentum_residual=angular_residual,
            time_residual=time_residual,
            perturbation_ratio=perturbation_ratio,
            derivative_available=successful,
            successful=successful,
            status=status,
            context=self.context,
            plan_id=self.prepared_id,
        )


__all__ = [
    "CloseEncounterRegularizationPlan",
    "CloseEncounterRegularizationResult",
    "KSGaugePolicy",
    "PreparedCloseEncounterSegment",
]
