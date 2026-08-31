#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._context import AstrodynamicsContext
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=-1))


def _safe_divide(numerator: Array, denominator: Array, /) -> Array:
    return numerator / jnp.where(denominator != 0.0, denominator, 1.0)


def _wrap_angle(value: Array, /) -> Array:
    return jnp.mod(value, 2.0 * jnp.pi)


class ModifiedEquinoctialElements(StrictModule):
    """Modified equinoctial elements ``(p, f, g, h, k, L)``."""

    values: Array
    context: AstrodynamicsContext
    retrograde_factor: int = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        retrograde_factor: int = 1,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        factor = int(retrograde_factor)
        if factor not in (-1, 1):
            raise ValueError("retrograde_factor must be -1 or +1.")
        values_ = jnp.asarray(values)
        if values_.shape != (6,):
            raise ValueError("Modified equinoctial elements must have shape (6,).")
        self.values = values_
        self.context = context
        self.retrograde_factor = factor

    @property
    def p(self) -> Array:
        return self.values[0]


class ClassicalOrbitalElements(StrictModule):
    """Classical elements ``(p, e, i, raan, argument_of_periapsis, anomaly)``."""

    values: Array
    context: AstrodynamicsContext

    def __init__(
        self,
        values: ArrayLike,
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        values_ = jnp.asarray(values)
        if values_.shape != (6,):
            raise ValueError("Classical orbital elements must have shape (6,).")
        self.values = values_
        self.context = context


class ModifiedEquinoctialConversionResult(StrictModule):
    elements: ModifiedEquinoctialElements
    valid: Array
    status: Array
    singularity_margin: Array


class ClassicalConversionResult(StrictModule):
    elements: ClassicalOrbitalElements
    valid: Array
    status: Array
    circular: Array
    equatorial: Array


def _orbit_geometry(state: CartesianOrbitState, mu: ArrayLike, /):
    coupling = jnp.asarray(mu, dtype=state.position.dtype).reshape(())
    radius = _norm(state.position)
    speed_squared = jnp.sum(state.velocity * state.velocity)
    radial_velocity = jnp.sum(state.position * state.velocity)
    angular_momentum = jnp.cross(state.position, state.velocity)
    angular_momentum_norm = _norm(angular_momentum)
    eccentricity_vector = (
        (speed_squared - coupling / jnp.where(radius > 0.0, radius, 1.0)) * state.position
        - radial_velocity * state.velocity
    ) / jnp.where(coupling > 0.0, coupling, 1.0)
    eccentricity = _norm(eccentricity_vector)
    finite = (
        jnp.isfinite(coupling)
        & (coupling > 0.0)
        & jnp.all(jnp.isfinite(state.position))
        & jnp.all(jnp.isfinite(state.velocity))
    )
    collision = radius <= 0.0
    radial = angular_momentum_norm <= jnp.sqrt(jnp.finfo(state.position.dtype).eps)
    valid = finite & ~collision & ~radial
    return (
        coupling,
        radius,
        angular_momentum,
        angular_momentum_norm,
        eccentricity_vector,
        eccentricity,
        valid,
        collision,
        radial,
    )


def cartesian_to_modified_equinoctial(
    state: CartesianOrbitState,
    mu: ArrayLike,
    /,
    *,
    retrograde_factor: int = 1,
) -> ModifiedEquinoctialConversionResult:
    """Convert one non-radial Cartesian state to a nonsingular element chart."""

    if not isinstance(state, CartesianOrbitState):
        raise TypeError("state must be a CartesianOrbitState.")
    factor = int(retrograde_factor)
    if factor not in (-1, 1):
        raise ValueError("retrograde_factor must be -1 or +1.")
    (
        coupling,
        radius,
        angular_momentum,
        angular_momentum_norm,
        eccentricity_vector,
        _,
        valid,
        collision,
        radial,
    ) = _orbit_geometry(state, mu)
    normal = angular_momentum / jnp.where(
        angular_momentum_norm > 0.0, angular_momentum_norm, 1.0
    )
    denominator = 1.0 + factor * normal[2]
    margin = jnp.abs(denominator)
    chart_valid = margin > 64.0 * jnp.finfo(state.position.dtype).eps
    h = -normal[1] / jnp.where(chart_valid, denominator, 1.0)
    k = normal[0] / jnp.where(chart_valid, denominator, 1.0)
    scale = 1.0 + h * h + k * k
    f_basis = jnp.asarray((1.0 - k * k + h * h, 2.0 * h * k, -2.0 * factor * k)) / scale
    g_basis = (
        jnp.asarray((2.0 * factor * h * k, factor * (1.0 + k * k - h * h), 2.0 * h))
        / scale
    )
    longitude = _wrap_angle(
        jnp.arctan2(jnp.sum(state.position * g_basis), jnp.sum(state.position * f_basis))
    )
    p = (
        angular_momentum_norm
        * angular_momentum_norm
        / jnp.where(coupling > 0.0, coupling, 1.0)
    )
    f = jnp.sum(eccentricity_vector * f_basis)
    g = jnp.sum(eccentricity_vector * g_basis)
    all_valid = valid & chart_valid & (radius > 0.0) & (p > 0.0)
    safe = jnp.where(
        all_valid,
        jnp.stack((p, f, g, h, k, longitude)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=state.position.dtype),
    )
    status = jnp.where(
        collision,
        int(AstrodynamicsStatus.COLLISION),
        jnp.where(
            radial | ~chart_valid,
            int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
            jnp.where(
                all_valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ),
    )
    return ModifiedEquinoctialConversionResult(
        ModifiedEquinoctialElements(safe, state.context, retrograde_factor=factor),
        all_valid,
        status.astype(jnp.int32),
        margin,
    )


def modified_equinoctial_to_cartesian(
    elements: ModifiedEquinoctialElements,
    mu: ArrayLike,
    /,
) -> tuple[CartesianOrbitState, Array, Array]:
    """Convert modified equinoctial elements to Cartesian position and velocity."""

    if not isinstance(elements, ModifiedEquinoctialElements):
        raise TypeError("elements must be ModifiedEquinoctialElements.")
    coupling = jnp.asarray(mu, dtype=elements.values.dtype).reshape(())
    p, f, g, h, k, longitude = elements.values
    factor = elements.retrograde_factor
    scale = 1.0 + h * h + k * k
    f_basis = jnp.asarray((1.0 - k * k + h * h, 2.0 * h * k, -2.0 * factor * k)) / scale
    g_basis = (
        jnp.asarray((2.0 * factor * h * k, factor * (1.0 + k * k - h * h), 2.0 * h))
        / scale
    )
    cosine = jnp.cos(longitude)
    sine = jnp.sin(longitude)
    denominator = 1.0 + f * cosine + g * sine
    valid = (
        jnp.all(jnp.isfinite(elements.values))
        & jnp.isfinite(coupling)
        & (coupling > 0.0)
        & (p > 0.0)
        & (denominator > 0.0)
    )
    safe_p = jnp.where(valid, p, 1.0)
    safe_denominator = jnp.where(valid, denominator, 1.0)
    position = safe_p / safe_denominator * (cosine * f_basis + sine * g_basis)
    velocity = jnp.sqrt(jnp.where(valid, coupling / safe_p, 1.0)) * (
        -(sine + g) * f_basis + (cosine + f) * g_basis
    )
    position = jnp.where(valid, position, jnp.zeros((3,), dtype=position.dtype))
    velocity = jnp.where(valid, velocity, jnp.zeros((3,), dtype=velocity.dtype))
    status = jnp.where(
        valid,
        int(AstrodynamicsStatus.SUCCESS),
        int(AstrodynamicsStatus.INVALID_DOMAIN),
    ).astype(jnp.int32)
    return CartesianOrbitState(position, velocity, elements.context), valid, status


def cartesian_to_classical(
    state: CartesianOrbitState,
    mu: ArrayLike,
    /,
    *,
    singularity_tolerance: float = 1.0e-10,
) -> ClassicalConversionResult:
    """Convert Cartesian state to classical elements on its nonsingular domain."""

    if not isinstance(state, CartesianOrbitState):
        raise TypeError("state must be a CartesianOrbitState.")
    tolerance = float(singularity_tolerance)
    if tolerance <= 0.0:
        raise ValueError("singularity_tolerance must be positive.")
    (
        coupling,
        _,
        angular_momentum,
        angular_momentum_norm,
        eccentricity_vector,
        eccentricity,
        geometric_valid,
        collision,
        radial,
    ) = _orbit_geometry(state, mu)
    node = jnp.asarray((-angular_momentum[1], angular_momentum[0], 0.0))
    node_norm = _norm(node)
    circular = eccentricity <= tolerance
    equatorial = node_norm <= tolerance
    fully_defined = geometric_valid & ~circular & ~equatorial
    p = (
        angular_momentum_norm
        * angular_momentum_norm
        / jnp.where(coupling > 0.0, coupling, 1.0)
    )
    inclination = jnp.arccos(
        jnp.clip(_safe_divide(angular_momentum[2], angular_momentum_norm), -1.0, 1.0)
    )
    raan = _wrap_angle(jnp.arctan2(node[1], node[0]))
    argument = _wrap_angle(
        jnp.arctan2(
            _safe_divide(
                jnp.sum(jnp.cross(node, eccentricity_vector) * angular_momentum),
                node_norm * eccentricity * angular_momentum_norm,
            ),
            _safe_divide(jnp.sum(node * eccentricity_vector), node_norm * eccentricity),
        )
    )
    anomaly = _wrap_angle(
        jnp.arctan2(
            _safe_divide(
                jnp.sum(
                    jnp.cross(eccentricity_vector, state.position) * angular_momentum
                ),
                eccentricity * _norm(state.position) * angular_momentum_norm,
            ),
            _safe_divide(
                jnp.sum(eccentricity_vector * state.position),
                eccentricity * _norm(state.position),
            ),
        )
    )
    safe = jnp.where(
        fully_defined,
        jnp.stack((p, eccentricity, inclination, raan, argument, anomaly)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=state.position.dtype),
    )
    status = jnp.where(
        collision,
        int(AstrodynamicsStatus.COLLISION),
        jnp.where(
            radial | circular | equatorial,
            int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
            jnp.where(
                fully_defined,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ),
    )
    return ClassicalConversionResult(
        ClassicalOrbitalElements(safe, state.context),
        fully_defined,
        status.astype(jnp.int32),
        circular,
        equatorial,
    )


def classical_to_cartesian(
    elements: ClassicalOrbitalElements,
    mu: ArrayLike,
    /,
) -> tuple[CartesianOrbitState, Array, Array]:
    """Convert nonsingular classical elements to Cartesian state."""

    if not isinstance(elements, ClassicalOrbitalElements):
        raise TypeError("elements must be ClassicalOrbitalElements.")
    coupling = jnp.asarray(mu, dtype=elements.values.dtype).reshape(())
    p, eccentricity, inclination, raan, argument, anomaly = elements.values
    valid = (
        jnp.all(jnp.isfinite(elements.values))
        & jnp.isfinite(coupling)
        & (coupling > 0.0)
        & (p > 0.0)
        & (eccentricity >= 0.0)
        & (1.0 + eccentricity * jnp.cos(anomaly) > 0.0)
    )
    safe_p = jnp.where(valid, p, 1.0)
    denominator = jnp.where(valid, 1.0 + eccentricity * jnp.cos(anomaly), 1.0)
    perifocal_position = (
        safe_p / denominator * jnp.asarray((jnp.cos(anomaly), jnp.sin(anomaly), 0.0))
    )
    perifocal_velocity = jnp.sqrt(jnp.where(valid, coupling / safe_p, 1.0)) * jnp.asarray(
        (-jnp.sin(anomaly), eccentricity + jnp.cos(anomaly), 0.0)
    )
    cosine_o, sine_o = jnp.cos(raan), jnp.sin(raan)
    cosine_i, sine_i = jnp.cos(inclination), jnp.sin(inclination)
    cosine_w, sine_w = jnp.cos(argument), jnp.sin(argument)
    rotation = jnp.asarray(
        (
            (
                cosine_o * cosine_w - sine_o * sine_w * cosine_i,
                -cosine_o * sine_w - sine_o * cosine_w * cosine_i,
                sine_o * sine_i,
            ),
            (
                sine_o * cosine_w + cosine_o * sine_w * cosine_i,
                -sine_o * sine_w + cosine_o * cosine_w * cosine_i,
                -cosine_o * sine_i,
            ),
            (sine_w * sine_i, cosine_w * sine_i, cosine_i),
        )
    )
    position = rotation @ perifocal_position
    velocity = rotation @ perifocal_velocity
    position = jnp.where(valid, position, jnp.zeros((3,), dtype=position.dtype))
    velocity = jnp.where(valid, velocity, jnp.zeros((3,), dtype=velocity.dtype))
    status = jnp.where(
        valid,
        int(AstrodynamicsStatus.SUCCESS),
        int(AstrodynamicsStatus.INVALID_DOMAIN),
    ).astype(jnp.int32)
    return CartesianOrbitState(position, velocity, elements.context), valid, status


__all__ = [
    "ClassicalConversionResult",
    "ClassicalOrbitalElements",
    "ModifiedEquinoctialConversionResult",
    "ModifiedEquinoctialElements",
    "cartesian_to_classical",
    "cartesian_to_modified_equinoctial",
    "classical_to_cartesian",
    "modified_equinoctial_to_cartesian",
]
