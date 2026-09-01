#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus
from ._two_body import propagate_universal_kepler, UniversalKeplerPolicy


def _norm(value: Array, /, *, axis=-1) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=axis))


class NearlyKeplerianState(StrictModule):
    position: Array
    velocity: Array
    context: AstrodynamicsContext

    def __init__(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        if (
            position_.ndim != 2
            or position_.shape[-1] != 3
            or velocity_.shape != position_.shape
        ):
            raise ValueError("Nearly Keplerian states require matching (N,3) arrays.")
        self.position = position_
        self.velocity = velocity_
        self.context = context


class NearlyKeplerianResult(StrictModule):
    times: Array
    positions: Array
    velocities: Array
    valid: Array
    status: Array
    minimum_separation: Array
    perturbation_ratio: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NearlyKeplerianPlan(StrictModule, NonTrainableState):
    central_mass: Array
    planet_masses: Array
    times: Array
    gravitational_constant: Array
    context: AstrodynamicsContext
    kepler_policy: UniversalKeplerPolicy
    close_approach_distance: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        central_mass: ArrayLike,
        planet_masses: ArrayLike,
        times: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        gravitational_constant: ArrayLike = 1.0,
        close_approach_distance: ArrayLike = 0.0,
        kepler_policy: UniversalKeplerPolicy | None = None,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        central = jnp.asarray(central_mass).reshape(())
        masses = jnp.asarray(planet_masses)
        if masses.ndim != 1 or int(masses.size) == 0:
            raise ValueError("planet_masses must be a nonempty vector.")
        times_host = np.asarray(times, dtype=float)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
            or not np.allclose(np.diff(times_host), np.diff(times_host)[0])
        ):
            raise ValueError(
                "Nearly Keplerian times must be finite, increasing, and uniform."
            )
        policy = UniversalKeplerPolicy() if kepler_policy is None else kepler_policy
        if not isinstance(policy, UniversalKeplerPolicy):
            raise TypeError("kepler_policy must be UniversalKeplerPolicy or None.")
        self.central_mass = central
        self.planet_masses = masses
        self.times = jnp.asarray(times_host)
        self.gravitational_constant = jnp.asarray(gravitational_constant).reshape(())
        self.context = context
        self.kepler_policy = policy
        self.close_approach_distance = jnp.asarray(close_approach_distance).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nearly-keplerian-plan",
                "context": context.context_id,
                "planet_count": int(masses.size),
                "num_times": int(times_host.size),
                "kepler_policy": policy.policy_id,
            }
        )

    def _perturbation(self, positions: Array, /) -> tuple[Array, Array, Array]:
        count = int(self.planet_masses.size)
        displacement = positions[None, :, :] - positions[:, None, :]
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        pair = ~jnp.eye(count, dtype=bool)
        safe_distance = jnp.where(pair, distance_squared, 1.0)
        direct = jnp.sum(
            self.planet_masses[None, :, None]
            * displacement
            * jnp.where(pair, safe_distance ** (-1.5), 0.0)[..., None],
            axis=1,
        )
        central_distance = _norm(positions)
        indirect = jnp.sum(
            self.planet_masses[:, None]
            * positions
            / jnp.where(
                central_distance[:, None] > 0.0, central_distance[:, None] ** 3, 1.0
            ),
            axis=0,
        )
        perturbation = self.gravitational_constant * (direct - indirect[None, :])
        pair_distance = jnp.sqrt(jnp.where(pair, distance_squared, jnp.inf))
        minimum = jnp.min(pair_distance)
        valid = (
            jnp.all(jnp.isfinite(perturbation))
            & jnp.all(central_distance > 0.0)
            & (minimum > self.close_approach_distance)
        )
        return perturbation, minimum, valid

    def _kepler_drift(
        self, position: Array, velocity: Array, delta_time: Array, /
    ) -> tuple[Array, Array, Array]:
        def one(r, v, mass):
            state = CartesianOrbitState(r, v, self.context)
            result = propagate_universal_kepler(
                state,
                delta_time,
                self.gravitational_constant * (self.central_mass + mass),
                policy=self.kepler_policy,
            )
            return result.state.position, result.state.velocity, result.valid

        return jax.vmap(one)(position, velocity, self.planet_masses)

    def _corrector_step(
        self,
        position: Array,
        velocity: Array,
        kepler_step: Array,
        kick_step: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        first_position, first_velocity, first_valid = self._kepler_drift(
            position, velocity, -kepler_step
        )
        perturbation, _, perturbation_valid = self._perturbation(first_position)
        kicked_velocity = first_velocity + kick_step * perturbation
        final_position, final_velocity, final_valid = self._kepler_drift(
            first_position, kicked_velocity, kepler_step
        )
        return (
            final_position,
            final_velocity,
            jnp.all(first_valid)
            & perturbation_valid
            & jnp.all(final_valid)
            & jnp.all(jnp.isfinite(final_velocity)),
        )

    def _real_to_map(
        self, position: Array, velocity: Array, dt: Array, /
    ) -> tuple[Array, Array, Array]:
        alpha = jnp.sqrt(7.0 / 40.0)
        beta = 1.0 / (48.0 * alpha)
        position, velocity, valid_second = self._corrector_step(
            position, velocity, alpha * dt, 0.5 * beta * dt
        )
        position, velocity, valid_first = self._corrector_step(
            position, velocity, -alpha * dt, -0.5 * beta * dt
        )
        return position, velocity, valid_first & valid_second

    def _map_to_real(
        self, position: Array, velocity: Array, dt: Array, /
    ) -> tuple[Array, Array, Array]:
        alpha = jnp.sqrt(7.0 / 40.0)
        beta = 1.0 / (48.0 * alpha)
        position, velocity, valid_first = self._corrector_step(
            position, velocity, -alpha * dt, 0.5 * beta * dt
        )
        position, velocity, valid_second = self._corrector_step(
            position, velocity, alpha * dt, -0.5 * beta * dt
        )
        return position, velocity, valid_first & valid_second

    def rollout(self, initial_state: NearlyKeplerianState, /) -> NearlyKeplerianResult:
        if not isinstance(initial_state, NearlyKeplerianState):
            raise TypeError("initial_state must be a NearlyKeplerianState.")
        self.context.require_compatible(initial_state.context)
        expected = (int(self.planet_masses.size), 3)
        if initial_state.position.shape != expected:
            raise ValueError(f"Nearly Keplerian state must have shape {expected}.")
        dt = self.times[1] - self.times[0]
        map_position, map_velocity, corrector_valid = self._real_to_map(
            initial_state.position, initial_state.velocity, dt
        )
        initial_perturbation, initial_minimum, perturbation_valid = self._perturbation(
            map_position
        )
        initial_valid = corrector_valid & perturbation_valid

        def step(carry, _):
            position, velocity, perturbation, active = carry
            half_velocity = velocity + 0.5 * dt * perturbation
            next_position, drift_velocity, drift_valid = self._kepler_drift(
                position, half_velocity, dt
            )
            next_perturbation, minimum, next_perturbation_valid = self._perturbation(
                next_position
            )
            next_velocity = drift_velocity + 0.5 * dt * next_perturbation
            map_valid = (
                active
                & jnp.all(drift_valid)
                & next_perturbation_valid
                & jnp.all(jnp.isfinite(next_velocity))
            )
            accepted_position = jnp.where(map_valid, next_position, position)
            accepted_velocity = jnp.where(map_valid, next_velocity, velocity)
            accepted_perturbation = jnp.where(map_valid, next_perturbation, perturbation)
            physical_position, physical_velocity, output_valid = self._map_to_real(
                accepted_position, accepted_velocity, dt
            )
            valid = map_valid & output_valid
            central_acceleration = (
                self.gravitational_constant
                * (self.central_mass + self.planet_masses)[:, None]
                * physical_position
                / jnp.where(
                    _norm(physical_position)[:, None] > 0.0,
                    _norm(physical_position)[:, None] ** 3,
                    1.0,
                )
            )
            physical_perturbation, physical_minimum, physical_valid = self._perturbation(
                physical_position
            )
            valid = valid & physical_valid
            ratio = jnp.max(
                _norm(physical_perturbation)
                / jnp.maximum(_norm(central_acceleration), 1.0e-30)
            )
            status = jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.COLLISION),
            ).astype(jnp.int32)
            return (
                accepted_position,
                accepted_velocity,
                accepted_perturbation,
                valid,
            ), (
                physical_position,
                physical_velocity,
                valid,
                status,
                jnp.minimum(minimum, physical_minimum),
                ratio,
            )

        (_, _, _, completed), outputs = jax.lax.scan(
            step,
            (
                map_position,
                map_velocity,
                initial_perturbation,
                initial_valid,
            ),
            xs=None,
            length=int(self.times.size) - 1,
        )
        positions = jnp.concatenate((initial_state.position[None], outputs[0]), axis=0)
        velocities = jnp.concatenate((initial_state.velocity[None], outputs[1]), axis=0)
        valid = jnp.concatenate((initial_valid[None], outputs[2]))
        status = jnp.concatenate(
            (
                jnp.where(
                    initial_valid,
                    int(AstrodynamicsStatus.SUCCESS),
                    int(AstrodynamicsStatus.COLLISION),
                )[None].astype(jnp.int32),
                outputs[3],
            )
        )
        minimum = jnp.concatenate((initial_minimum[None], outputs[4]))
        ratio = jnp.concatenate((jnp.asarray(0.0)[None], outputs[5]))
        return NearlyKeplerianResult(
            self.times,
            positions,
            velocities,
            valid,
            status,
            minimum,
            ratio,
            completed,
            self.plan_id,
        )


__all__ = ["NearlyKeplerianPlan", "NearlyKeplerianResult", "NearlyKeplerianState"]
