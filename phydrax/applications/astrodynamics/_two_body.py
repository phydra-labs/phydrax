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
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


def stumpff_c(z: ArrayLike, /) -> Array:
    """Evaluate the universal-variable Stumpff C function stably near zero."""

    value = jnp.asarray(z)
    magnitude = jnp.abs(value)
    series = (
        0.5 - value / 24.0 + value**2 / 720.0 - value**3 / 40320.0 + value**4 / 3628800.0
    )
    positive_root = jnp.sqrt(jnp.maximum(value, 0.0))
    negative_root = jnp.sqrt(jnp.maximum(-value, 0.0))
    positive = (1.0 - jnp.cos(positive_root)) / jnp.where(value > 0.0, value, 1.0)
    negative = (jnp.cosh(negative_root) - 1.0) / jnp.where(value < 0.0, -value, 1.0)
    regular = jnp.where(value >= 0.0, positive, negative)
    return jnp.where(magnitude < 1.0e-4, series, regular)


def stumpff_s(z: ArrayLike, /) -> Array:
    """Evaluate the universal-variable Stumpff S function stably near zero."""

    value = jnp.asarray(z)
    magnitude = jnp.abs(value)
    series = (
        1.0 / 6.0
        - value / 120.0
        + value**2 / 5040.0
        - value**3 / 362880.0
        + value**4 / 39916800.0
    )
    positive_root = jnp.sqrt(jnp.maximum(value, 0.0))
    negative_root = jnp.sqrt(jnp.maximum(-value, 0.0))
    positive = (positive_root - jnp.sin(positive_root)) / jnp.where(
        value > 0.0, positive_root**3, 1.0
    )
    negative = (jnp.sinh(negative_root) - negative_root) / jnp.where(
        value < 0.0, negative_root**3, 1.0
    )
    regular = jnp.where(value >= 0.0, positive, negative)
    return jnp.where(magnitude < 1.0e-4, series, regular)


class UniversalKeplerPolicy(StrictModule, NonTrainableState):
    max_iterations: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, max_iterations: int = 48, relative_tolerance: float = 1.0e-12):
        iterations = int(max_iterations)
        tolerance = float(relative_tolerance)
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive.")
        self.max_iterations = iterations
        self.relative_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "universal-kepler-policy",
                "max_iterations": iterations,
                "relative_tolerance": tolerance,
            }
        )


class UniversalKeplerResult(StrictModule):
    state: CartesianOrbitState
    universal_anomaly: Array
    iterations: Array
    residual: Array
    specific_energy_before: Array
    specific_energy_after: Array
    angular_momentum_defect: Array
    valid: Array
    status: Array
    policy_id: str = eqx.field(static=True)


def _parameters(position: Array, velocity: Array, mu: Array, /):
    radius = _norm(position)
    speed_squared = jnp.sum(velocity * velocity)
    radial_dot = jnp.sum(position * velocity)
    alpha = 2.0 / jnp.where(radius > 0.0, radius, 1.0) - speed_squared / jnp.where(
        mu > 0.0, mu, 1.0
    )
    return radius, radial_dot, alpha


def _kepler_residual(
    anomaly: Array,
    position: Array,
    velocity: Array,
    delta_time: Array,
    mu: Array,
    /,
) -> Array:
    radius, radial_dot, alpha = _parameters(position, velocity, mu)
    root_mu = jnp.sqrt(jnp.where(mu > 0.0, mu, 1.0))
    z = alpha * anomaly * anomaly
    c = stumpff_c(z)
    s = stumpff_s(z)
    return (
        radial_dot / root_mu * anomaly * anomaly * c
        + (1.0 - alpha * radius) * anomaly**3 * s
        + radius * anomaly
        - root_mu * delta_time
    )


def _kepler_residual_derivative(
    anomaly: Array,
    position: Array,
    velocity: Array,
    mu: Array,
    /,
) -> Array:
    radius, radial_dot, alpha = _parameters(position, velocity, mu)
    root_mu = jnp.sqrt(jnp.where(mu > 0.0, mu, 1.0))
    z = alpha * anomaly * anomaly
    c = stumpff_c(z)
    s = stumpff_s(z)
    return (
        radial_dot / root_mu * anomaly * (1.0 - z * s)
        + (1.0 - alpha * radius) * anomaly * anomaly * c
        + radius
    )


def _solve_universal_anomaly(
    position: Array,
    velocity: Array,
    delta_time: Array,
    mu: Array,
    policy: UniversalKeplerPolicy,
    /,
) -> tuple[Array, Array, Array, Array]:
    radius, _, alpha = _parameters(position, velocity, mu)
    root_mu = jnp.sqrt(jnp.where(mu > 0.0, mu, 1.0))
    near_parabolic = jnp.abs(alpha) < 32.0 * jnp.finfo(position.dtype).eps
    initial = jnp.where(
        near_parabolic,
        root_mu * delta_time / jnp.where(radius > 0.0, radius, 1.0),
        root_mu * jnp.abs(alpha) * delta_time,
    )
    initial = jnp.where(delta_time == 0.0, 0.0, initial)
    scale = 1.0 + jnp.abs(root_mu * delta_time)

    def iteration(index, carry):
        anomaly, converged, first_iteration = carry
        residual = _kepler_residual(anomaly, position, velocity, delta_time, mu)
        derivative = _kepler_residual_derivative(anomaly, position, velocity, mu)
        safe_derivative = jnp.where(jnp.abs(derivative) > 0.0, derivative, 1.0)
        candidate = anomaly - residual / safe_derivative
        now_converged = jnp.abs(residual) <= policy.relative_tolerance * scale
        update = ~converged & ~now_converged & jnp.isfinite(candidate)
        next_anomaly = jnp.where(update, candidate, anomaly)
        next_first = jnp.where(
            (first_iteration < 0) & now_converged, index + 1, first_iteration
        )
        return next_anomaly, converged | now_converged, next_first

    anomaly, converged, first_iteration = jax.lax.fori_loop(
        0,
        policy.max_iterations,
        iteration,
        (initial, jnp.asarray(False), jnp.asarray(-1, dtype=jnp.int32)),
    )
    residual = _kepler_residual(anomaly, position, velocity, delta_time, mu)
    converged = converged | (jnp.abs(residual) <= policy.relative_tolerance * scale)
    iterations = jnp.where(first_iteration >= 0, first_iteration, policy.max_iterations)
    return anomaly, residual, iterations.astype(jnp.int32), converged


def _reconstruct_state_primal(
    anomaly: Array,
    position: Array,
    velocity: Array,
    delta_time: Array,
    mu: Array,
) -> Array:
    radius, _, alpha = _parameters(position, velocity, mu)
    z = alpha * anomaly * anomaly
    c = stumpff_c(z)
    s = stumpff_s(z)
    f = 1.0 - anomaly * anomaly / jnp.where(radius > 0.0, radius, 1.0) * c
    g = delta_time - anomaly**3 / jnp.sqrt(jnp.where(mu > 0.0, mu, 1.0)) * s
    next_position = f * position + g * velocity
    next_radius = _norm(next_position)
    fdot = (
        jnp.sqrt(jnp.where(mu > 0.0, mu, 1.0))
        / jnp.where(radius * next_radius > 0.0, radius * next_radius, 1.0)
        * (alpha * anomaly**3 * s - anomaly)
    )
    gdot = 1.0 - anomaly * anomaly / jnp.where(next_radius > 0.0, next_radius, 1.0) * c
    next_velocity = fdot * position + gdot * velocity
    return jnp.concatenate((next_position, next_velocity))


@jax.custom_jvp
def _reconstruct_state_implicit(
    anomaly: Array,
    position: Array,
    velocity: Array,
    delta_time: Array,
    mu: Array,
) -> Array:
    return _reconstruct_state_primal(anomaly, position, velocity, delta_time, mu)


@_reconstruct_state_implicit.defjvp
def _reconstruct_state_implicit_jvp(primals, tangents):
    anomaly, position, velocity, delta_time, mu = primals
    _, position_dot, velocity_dot, delta_time_dot, mu_dot = tangents
    residual_input_dot = jax.jvp(
        lambda r, v, dt, coupling: _kepler_residual(anomaly, r, v, dt, coupling),
        (position, velocity, delta_time, mu),
        (position_dot, velocity_dot, delta_time_dot, mu_dot),
    )[1]
    derivative = _kepler_residual_derivative(anomaly, position, velocity, mu)
    anomaly_dot = -residual_input_dot / jnp.where(
        jnp.abs(derivative) > 0.0, derivative, 1.0
    )
    value, tangent = jax.jvp(
        _reconstruct_state_primal,
        (anomaly, position, velocity, delta_time, mu),
        (anomaly_dot, position_dot, velocity_dot, delta_time_dot, mu_dot),
    )
    return value, tangent


def propagate_universal_kepler(
    state: CartesianOrbitState,
    delta_time: ArrayLike,
    mu: ArrayLike,
    /,
    *,
    policy: UniversalKeplerPolicy | None = None,
) -> UniversalKeplerResult:
    """Propagate one Cartesian state through the universal Kepler equation."""

    if not isinstance(state, CartesianOrbitState):
        raise TypeError("state must be a CartesianOrbitState.")
    resolved = UniversalKeplerPolicy() if policy is None else policy
    if not isinstance(resolved, UniversalKeplerPolicy):
        raise TypeError("policy must be a UniversalKeplerPolicy or None.")
    time = jnp.asarray(delta_time, dtype=state.position.dtype).reshape(())
    coupling = jnp.asarray(mu, dtype=state.position.dtype).reshape(())
    radius = _norm(state.position)
    finite = (
        jnp.isfinite(time)
        & jnp.isfinite(coupling)
        & jnp.all(jnp.isfinite(state.position))
        & jnp.all(jnp.isfinite(state.velocity))
    )
    domain = finite & (coupling > 0.0) & (radius > 0.0)
    safe_mu = jnp.where(domain, coupling, 1.0)
    safe_position = jnp.where(domain, state.position, jnp.asarray((1.0, 0.0, 0.0)))
    safe_velocity = jnp.where(
        domain, state.velocity, jnp.zeros((3,), dtype=state.velocity.dtype)
    )
    anomaly, residual, iterations, converged = _solve_universal_anomaly(
        safe_position, safe_velocity, time, safe_mu, resolved
    )
    packed = _reconstruct_state_implicit(
        jax.lax.stop_gradient(anomaly), safe_position, safe_velocity, time, safe_mu
    )
    next_position, next_velocity = packed[:3], packed[3:]
    output_finite = jnp.all(jnp.isfinite(packed))
    valid = domain & converged & output_finite
    accepted_position = jnp.where(valid, next_position, state.position)
    accepted_velocity = jnp.where(valid, next_velocity, state.velocity)
    energy_before = 0.5 * jnp.sum(state.velocity**2) - coupling / jnp.where(
        radius > 0.0, radius, 1.0
    )
    next_radius = _norm(accepted_position)
    energy_after = 0.5 * jnp.sum(accepted_velocity**2) - coupling / jnp.where(
        next_radius > 0.0, next_radius, 1.0
    )
    angular_before = jnp.cross(state.position, state.velocity)
    angular_after = jnp.cross(accepted_position, accepted_velocity)
    status = jnp.where(
        ~finite,
        int(AstrodynamicsStatus.NONFINITE_INPUT),
        jnp.where(
            radius <= 0.0,
            int(AstrodynamicsStatus.COLLISION),
            jnp.where(
                coupling <= 0.0,
                int(AstrodynamicsStatus.INVALID_DOMAIN),
                jnp.where(
                    ~converged,
                    int(AstrodynamicsStatus.NONCONVERGED),
                    jnp.where(
                        output_finite,
                        int(AstrodynamicsStatus.SUCCESS),
                        int(AstrodynamicsStatus.NONFINITE_INPUT),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return UniversalKeplerResult(
        CartesianOrbitState(accepted_position, accepted_velocity, state.context),
        jax.lax.stop_gradient(anomaly),
        iterations,
        jnp.abs(residual),
        energy_before,
        energy_after,
        _norm(angular_after - angular_before),
        valid,
        status,
        resolved.policy_id,
    )


__all__ = [
    "UniversalKeplerPolicy",
    "UniversalKeplerResult",
    "propagate_universal_kepler",
    "stumpff_c",
    "stumpff_s",
]
