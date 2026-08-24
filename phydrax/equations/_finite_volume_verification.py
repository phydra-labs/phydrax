#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._hyperbolic_systems import (
    AbstractConservationSystem,
    EulerSystem,
    ScalarConservationSystem,
)


VerificationField = Callable[[Array, Array, Any], Array]


class FiniteVolumeVerificationCase(StrictModule):
    name: str = eqx.field(static=True)
    system: AbstractConservationSystem
    initial_state: VerificationField = eqx.field(static=True)
    exact_state: VerificationField | None = eqx.field(static=True)
    final_time: float = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        system: AbstractConservationSystem,
        initial_state: VerificationField,
        final_time: float,
        /,
        *,
        exact_state: VerificationField | None = None,
    ):
        name_ = str(name)
        final = float(final_time)
        if not name_ or not isinstance(system, AbstractConservationSystem):
            raise ValueError("Verification case requires a name and conservation system.")
        if not callable(initial_state) or (
            exact_state is not None and not callable(exact_state)
        ):
            raise TypeError("Verification initial/exact states must be callable.")
        if not np.isfinite(final) or final <= 0.0:
            raise ValueError("Verification final time must be finite and positive.")
        self.name = name_
        self.system = system
        self.initial_state = initial_state
        self.exact_state = exact_state
        self.final_time = final
        self.case_id = canonical_fingerprint(
            {
                "kind": "finite-volume-verification-case",
                "name": name_,
                "system": system.system_id,
                "final_time": final,
                "initial": repr(initial_state),
                "exact": None if exact_state is None else repr(exact_state),
            }
        )


class FiniteVolumeErrorNorms(StrictModule):
    l1: Array
    l2: Array
    linf: Array


class FiniteVolumeConvergenceResult(StrictModule):
    resolutions: tuple[int, ...] = eqx.field(static=True)
    errors: Array
    observed_orders: Array
    expected_order: float = eqx.field(static=True)
    passed: Array


class FiniteVolumeConservationBudget(StrictModule):
    initial_integral: Array
    final_integral: Array
    boundary_integral: Array
    source_integral: Array
    defect: Array


def finite_volume_error_norms(
    numerical: ArrayLike,
    exact: ArrayLike,
    cell_volumes: ArrayLike,
    /,
) -> FiniteVolumeErrorNorms:
    numerical_ = jnp.asarray(numerical)
    exact_ = jnp.asarray(exact)
    volumes = jnp.asarray(cell_volumes)
    if (
        numerical_.shape != exact_.shape
        or numerical_.shape[: volumes.ndim] != volumes.shape
    ):
        raise ValueError("Verification values and cell volumes must align.")
    error = jnp.abs(numerical_ - exact_)
    weights = volumes / jnp.sum(volumes)
    reshape = weights.shape + (1,) * (error.ndim - weights.ndim)
    return FiniteVolumeErrorNorms(
        l1=jnp.sum(weights.reshape(reshape) * error),
        l2=jnp.sqrt(jnp.sum(weights.reshape(reshape) * error**2)),
        linf=jnp.max(error),
    )


def finite_volume_convergence_result(
    resolutions: Sequence[int],
    errors: ArrayLike,
    expected_order: float,
    /,
    *,
    order_tolerance: float = 0.25,
) -> FiniteVolumeConvergenceResult:
    resolutions_ = tuple(int(value) for value in resolutions)
    errors_ = jnp.asarray(errors)
    expected = float(expected_order)
    tolerance = float(order_tolerance)
    if (
        len(resolutions_) < 2
        or errors_.shape != (len(resolutions_),)
        or any(value <= 0 for value in resolutions_)
        or expected <= 0.0
        or tolerance < 0.0
    ):
        raise ValueError("Finite-volume convergence inputs are invalid.")
    ratios = jnp.asarray(resolutions_[1:], dtype=errors_.dtype) / jnp.asarray(
        resolutions_[:-1], dtype=errors_.dtype
    )
    orders = jnp.log(errors_[:-1] / errors_[1:]) / jnp.log(ratios)
    return FiniteVolumeConvergenceResult(
        resolutions=resolutions_,
        errors=errors_,
        observed_orders=orders,
        expected_order=expected,
        passed=jnp.all(orders[-1:] >= expected - tolerance),
    )


def periodic_advection_verification_case(
    speed: float = 1.0,
    /,
) -> FiniteVolumeVerificationCase:
    speed_ = float(speed)
    if not np.isfinite(speed_):
        raise ValueError("Advection speed must be finite.")
    system = ScalarConservationSystem(
        1,
        lambda state, axis, args: speed_ * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1], abs(speed_), dtype=left.dtype
        ),
        system_id=canonical_fingerprint(
            {"kind": "verification-advection", "speed": speed_}
        ),
    )

    def initial(points: Array, time: Array, args: Any) -> Array:
        del time, args
        return jnp.sin(2.0 * jnp.pi * points[..., :1])

    def exact(points: Array, time: Array, args: Any) -> Array:
        del args
        return jnp.sin(2.0 * jnp.pi * (points[..., :1] - speed_ * time))

    return FiniteVolumeVerificationCase(
        "periodic-advection",
        system,
        initial,
        1.0 / max(abs(speed_), 1.0),
        exact_state=exact,
    )


def sod_verification_case() -> FiniteVolumeVerificationCase:
    system = EulerSystem()

    def initial(points: Array, time: Array, args: Any) -> Array:
        del time, args
        coordinate = points[..., 0]
        primitive = jnp.stack(
            (
                jnp.where(coordinate < 0.5, 1.0, 0.125),
                jnp.zeros_like(coordinate),
                jnp.where(coordinate < 0.5, 1.0, 0.1),
            ),
            axis=-1,
        )
        return system.primitive_to_conserved(primitive)

    return FiniteVolumeVerificationCase("sod", system, initial, 0.2)


def euler_riemann_verification_case(
    name: str,
    left_primitive: ArrayLike,
    right_primitive: ArrayLike,
    final_time: float,
    /,
) -> FiniteVolumeVerificationCase:
    system = EulerSystem()
    left = jnp.asarray(left_primitive)
    right = jnp.asarray(right_primitive)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("Euler Riemann primitive states must have three entries.")

    def initial(points: Array, time: Array, args: Any) -> Array:
        del time, args
        coordinate = points[..., 0]
        primitive = jnp.where((coordinate < 0.5)[..., None], left, right)
        return system.primitive_to_conserved(primitive)

    return FiniteVolumeVerificationCase(str(name), system, initial, float(final_time))


def lax_verification_case() -> FiniteVolumeVerificationCase:
    return euler_riemann_verification_case(
        "lax",
        jnp.asarray([0.445, 0.698, 3.528]),
        jnp.asarray([0.5, 0.0, 0.571]),
        0.14,
    )


def double_rarefaction_verification_case() -> FiniteVolumeVerificationCase:
    return euler_riemann_verification_case(
        "double-rarefaction",
        jnp.asarray([1.0, -2.0, 0.4]),
        jnp.asarray([1.0, 2.0, 0.4]),
        0.1,
    )


def woodward_colella_verification_case() -> FiniteVolumeVerificationCase:
    system = EulerSystem()

    def initial(points: Array, time: Array, args: Any) -> Array:
        del time, args
        coordinate = points[..., 0]
        pressure = jnp.where(
            coordinate < 0.1,
            1000.0,
            jnp.where(coordinate < 0.9, 0.01, 100.0),
        )
        primitive = jnp.stack(
            (
                jnp.ones_like(coordinate),
                jnp.zeros_like(coordinate),
                pressure,
            ),
            axis=-1,
        )
        return system.primitive_to_conserved(primitive)

    return FiniteVolumeVerificationCase("woodward-colella", system, initial, 0.038)


def couette_velocity_profile(
    coordinate: ArrayLike,
    lower_velocity: float,
    upper_velocity: float,
    /,
) -> Array:
    y = jnp.asarray(coordinate)
    return lower_velocity + (upper_velocity - lower_velocity) * y


def poiseuille_velocity_profile(
    coordinate: ArrayLike,
    pressure_gradient: float,
    dynamic_viscosity: float,
    /,
) -> Array:
    y = jnp.asarray(coordinate)
    return -0.5 * pressure_gradient / dynamic_viscosity * y * (1.0 - y)


__all__ = [
    "FiniteVolumeConservationBudget",
    "FiniteVolumeConvergenceResult",
    "FiniteVolumeErrorNorms",
    "FiniteVolumeVerificationCase",
    "couette_velocity_profile",
    "double_rarefaction_verification_case",
    "euler_riemann_verification_case",
    "lax_verification_case",
    "finite_volume_convergence_result",
    "finite_volume_error_norms",
    "periodic_advection_verification_case",
    "poiseuille_velocity_profile",
    "sod_verification_case",
    "woodward_colella_verification_case",
]
